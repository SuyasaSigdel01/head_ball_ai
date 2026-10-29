"""
header_game.py — webcam basketball, scored with your head (Header Hoops)

A hoop sits in the upper court. A ball drops in. Head it through the rim.
Every basket is a point. 60-second clock. Beat your best.

Requires (keep mediapipe pinned — 0.10.31+ dropped the legacy solutions API):
    mediapipe==0.10.21
    opencv-contrib-python==4.11.0.86
    pygame numpy

Run:
    python header_game.py

Controls:
    SPACE / R  start / restart
    ESC / Q    quit
"""

import math
import random
import sys
import time
from collections import deque

import cv2
import numpy as np
import pygame

try:
    import mediapipe as mp
except ImportError:
    sys.exit("mediapipe missing.  ->  pip install mediapipe==0.10.21")


# ------------------------------------------------------------------ tunables
GRAVITY = 1400.0
BALL_R = 24
HEAD_R = 62
BASE_BOUNCE = 680.0        # header power (needs to reach the rim)
POWER_GAIN = 0.5
MAX_POWER_BONUS = 560.0
STEER = 520.0              # sideways aim from off-centre contact
HIT_COOLDOWN = 0.18
SMOOTH_N = 5
VEL_WINDOW = 4
FOREHEAD_LM = 10
MAX_DT = 1.0 / 30.0

GAME_TIME = 60.0           # seconds per round

# --- hoop geometry (fractions of screen, resolved at runtime) ---
HOOP_FX, HOOP_FY = 0.72, 0.24   # rim centre
RIM_W = 130                     # rim opening width
BOARD_H = 95                    # backboard height
RIM_END_R = 7                   # rim-end collider radius (causes rattle-outs)
MOVING_HOOP = False             # flip to True for a sliding hoop (hard mode)
HOOP_SPEED = 90.0               # px/s when moving

WHITE = (255, 255, 255)
BLACK = (10, 10, 10)
GREEN = (60, 220, 120)
RED = (240, 80, 80)
YELLOW = (250, 210, 60)
ORANGE = (255, 140, 30)
BOARD_COL = (230, 230, 230)


# ------------------------------------------------------------------ helpers
def frame_to_surface(rgb):
    return pygame.image.frombuffer(
        np.ascontiguousarray(rgb).tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB"
    )


def draw_text(surf, font, text, pos, color=WHITE, center=False):
    shadow = font.render(text, True, BLACK)
    label = font.render(text, True, color)
    rect = label.get_rect()
    if center:
        rect.center = pos
    else:
        rect.topleft = pos
    surf.blit(shadow, (rect.x + 2, rect.y + 2))
    surf.blit(label, rect)


class Ball:
    def __init__(self, x, y):
        self.reset(x, y)

    def reset(self, x, y):
        self.x, self.y = float(x), float(y)
        self.prev_x, self.prev_y = float(x), float(y)
        self.vx, self.vy = 0.0, 0.0
        self.headed = False          # became True after first header this spawn

    def step(self, dt, w):
        self.prev_x, self.prev_y = self.x, self.y
        self.vy += GRAVITY * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        if self.x < BALL_R:
            self.x, self.vx = float(BALL_R), abs(self.vx) * 0.85
        elif self.x > w - BALL_R:
            self.x, self.vx = float(w - BALL_R), -abs(self.vx) * 0.85
        if self.y < BALL_R:
            self.y, self.vy = float(BALL_R), abs(self.vy) * 0.5

    def draw(self, surf):
        p = (int(self.x), int(self.y))
        pygame.draw.circle(surf, ORANGE, p, BALL_R)
        pygame.draw.circle(surf, BLACK, p, BALL_R, 2)
        # basketball seams
        pygame.draw.line(surf, BLACK, (p[0] - BALL_R, p[1]), (p[0] + BALL_R, p[1]), 2)
        pygame.draw.line(surf, BLACK, (p[0], p[1] - BALL_R), (p[0], p[1] + BALL_R), 2)


class HeadTracker:
    def __init__(self):
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.history = deque(maxlen=SMOOTH_N)
        self.samples = deque(maxlen=VEL_WINDOW)

    def update(self, rgb):
        h, w = rgb.shape[:2]
        rgb.flags.writeable = False
        res = self.mesh.process(rgb)
        rgb.flags.writeable = True
        if not res.multi_face_landmarks:
            self.history.clear()
            self.samples.clear()
            return None
        lm = res.multi_face_landmarks[0].landmark[FOREHEAD_LM]
        self.history.append((lm.x * w, lm.y * h))
        sx = sum(p[0] for p in self.history) / len(self.history)
        sy = sum(p[1] for p in self.history) / len(self.history)
        self.samples.append((time.perf_counter(), sx, sy))
        return sx, sy

    def velocity(self):
        if len(self.samples) < 2:
            return 0.0, 0.0
        t0, x0, y0 = self.samples[0]
        t1, x1, y1 = self.samples[-1]
        dt = t1 - t0
        return (0.0, 0.0) if dt <= 1e-4 else ((x1 - x0) / dt, (y1 - y0) / dt)


class Hoop:
    """Rim geometry + collisions + score detection."""

    def __init__(self, W, H):
        self.base_x = W * HOOP_FX
        self.y = H * HOOP_FY
        self.x = self.base_x
        self.dir = 1
        self.W = W

    def update(self, dt):
        if MOVING_HOOP:
            self.x += self.dir * HOOP_SPEED * dt
            half = RIM_W / 2
            if self.x - half < self.W * 0.45:
                self.x, self.dir = self.W * 0.45 + half, 1
            elif self.x + half > self.W - 10:
                self.x, self.dir = self.W - 10 - half, -1

    @property
    def left(self):
        return self.x - RIM_W / 2

    @property
    def right(self):
        return self.x + RIM_W / 2

    def collide(self, ball):
        """Bounce ball off the two rim ends and the backboard."""
        # Rim ends (small circles) -> rattle-outs
        for ex in (self.left, self.right):
            dx, dy = ball.x - ex, ball.y - self.y
            d = math.hypot(dx, dy)
            if d < BALL_R + RIM_END_R and d > 1e-3:
                nx, ny = dx / d, dy / d
                ball.x = ex + nx * (BALL_R + RIM_END_R)
                ball.y = self.y + ny * (BALL_R + RIM_END_R)
                dot = ball.vx * nx + ball.vy * ny
                ball.vx -= 2 * dot * nx
                ball.vy -= 2 * dot * ny
                ball.vx *= 0.7
                ball.vy *= 0.7

        # Backboard: vertical wall just behind the right rim end
        board_x = self.right + 4
        board_top, board_bot = self.y - BOARD_H, self.y + 8
        if (board_top <= ball.y <= board_bot
                and ball.prev_x + BALL_R <= board_x < ball.x + BALL_R
                and ball.vx > 0):
            ball.x = board_x - BALL_R
            ball.vx = -abs(ball.vx) * 0.7

    def scored(self, ball):
        """True on a clean downward pass through the rim (after a header)."""
        crossed = ball.prev_y < self.y <= ball.y
        inside = self.left + BALL_R * 0.4 < ball.x < self.right - BALL_R * 0.4
        return crossed and inside and ball.vy > 0 and ball.headed

    def draw(self, surf):
        board_x = int(self.right + 4)
        pygame.draw.rect(surf, BOARD_COL,
                         (board_x, int(self.y - BOARD_H), 8, BOARD_H), border_radius=3)
        pygame.draw.line(surf, ORANGE, (int(self.left), int(self.y)),
                         (int(self.right), int(self.y)), 6)
        for ex in (self.left, self.right):
            pygame.draw.circle(surf, ORANGE, (int(ex), int(self.y)), RIM_END_R)
        # little net hint
        for k in range(1, 5):
            fx = self.left + (self.right - self.left) * k / 5
            pygame.draw.line(surf, WHITE, (int(fx), int(self.y)),
                             (int(self.x), int(self.y + 34)), 1)


def try_header(ball, head_pos, head_vel):
    """Bounce the ball off a head if they touch. Returns True on a hit.

    Shared by every player (webcam human now, bot later) so the physics are
    identical for both. Cooldown is NOT checked here — the caller does that,
    because each player keeps its own last_hit time.
    """
    # 1. distance between ball centre and head centre
    dx, dy = ball.x - head_pos[0], ball.y - head_pos[1]
    dist = math.hypot(dx, dy)

    # 2. no contact, or ball is going up (already headed) -> no hit
    if dist >= BALL_R + HEAD_R or ball.vy <= 0:
        return False

    # 3. sideways aim from off-centre contact, power from upward head speed
    offset = max(-1.0, min(1.0, dx / HEAD_R))
    ball.vx = offset * STEER
    bonus = min(max(-head_vel[1], 0.0) * POWER_GAIN, MAX_POWER_BONUS)
    ball.vy = -(BASE_BOUNCE + bonus)

    # 4. push the ball just outside the head so they don't overlap next frame
    d = max(dist, 1e-3)
    ball.x = head_pos[0] + dx / d * (BALL_R + HEAD_R + 1)
    ball.y = head_pos[1] + dy / d * (BALL_R + HEAD_R + 1)

    # 5. mark it so a basket counts
    ball.headed = True
    return True


def spawn_ball(W, H, hoop):
    """Drop the ball on the opposite side from the hoop so it must be headed in."""
    x = W * random.uniform(0.15, 0.35) if hoop.x > W / 2 else W * random.uniform(0.65, 0.85)
    b = Ball(x, H * 0.15)
    b.vx = random.uniform(-40, 40)
    return b


# ------------------------------------------------------------------ main
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        sys.exit("No camera found.")
    ok, probe = cap.read()
    if not ok:
        sys.exit("Camera opened but returned no frame.")
    H, W = probe.shape[:2]

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Header Hoops")
    clock = pygame.time.Clock()
    font_big = pygame.font.SysFont("consolas", 60, bold=True)
    font = pygame.font.SysFont("consolas", 30, bold=True)

    tracker = HeadTracker()
    hoop = Hoop(W, H)
    ball = spawn_ball(W, H, hoop)
    state = "WAIT"            # WAIT -> PLAY -> OVER
    score = 0
    best = 0
    time_left = GAME_TIME
    last_hit = 0.0
    flash_until = 0.0        # green flash after a made basket

    running = True
    while running:
        dt = min(clock.tick(60) / 1000.0, MAX_DT)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif e.key in (pygame.K_SPACE, pygame.K_r) and state in ("WAIT", "OVER"):
                    score, time_left = 0, GAME_TIME
                    ball = spawn_ball(W, H, hoop)
                    state = "PLAY"

        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        head = tracker.update(rgb)

        screen.blit(frame_to_surface(rgb), (0, 0))
        hoop.update(dt)

        if state == "PLAY":
            if head is None:
                # Pause the clock — don't let bad lighting eat their time.
                draw_text(screen, font, "FACE NOT DETECTED", (W // 2, 70),
                          RED, center=True)
            else:
                time_left -= dt
                ball.step(dt, W)
                hoop.collide(ball)

                # header contact — physics live in try_header(), cooldown here
                now = time.perf_counter()
                if (now - last_hit > HIT_COOLDOWN
                        and try_header(ball, head, tracker.velocity())):
                    last_hit = now

                if hoop.scored(ball):
                    score += 1
                    best = max(best, score)
                    flash_until = now + 0.4
                    ball = spawn_ball(W, H, hoop)
                elif ball.y - BALL_R > H:      # missed, hit the floor
                    ball = spawn_ball(W, H, hoop)

                if time_left <= 0:
                    time_left = 0
                    state = "OVER"

        # ---------------- draw ----------------
        hoop.draw(screen)
        if head:
            pygame.draw.circle(screen, GREEN, (int(head[0]), int(head[1])), HEAD_R, 3)
        if state == "PLAY":
            ball.draw(screen)
            if time.perf_counter() < flash_until:
                draw_text(screen, font_big, "SCORE!", (W // 2, H // 2), GREEN, center=True)

        draw_text(screen, font, f"BASKETS {score}", (16, 12), ORANGE)
        draw_text(screen, font, f"BEST {best}", (16, 48), YELLOW)
        if state == "PLAY":
            col = RED if time_left < 10 else WHITE
            draw_text(screen, font, f"{time_left:04.1f}s", (W - 150, 12), col)

        if state == "WAIT":
            draw_text(screen, font_big, "HEADER HOOPS", (W // 2, H // 2 - 40),
                      ORANGE, center=True)
            draw_text(screen, font, "SPACE to start", (W // 2, H // 2 + 20),
                      WHITE, center=True)
        elif state == "OVER":
            draw_text(screen, font_big, f"{score} BASKETS", (W // 2, H // 2 - 40),
                      ORANGE, center=True)
            draw_text(screen, font, "SPACE to play again", (W // 2, H // 2 + 20),
                      WHITE, center=True)

        pygame.display.flip()

    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()