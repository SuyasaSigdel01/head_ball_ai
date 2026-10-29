"""
pass_game.py — co-op heading rally: you (webcam) vs-with a bot

Stand on the LEFT half of the camera. The bot lives on the RIGHT half.
Head the ball across the middle line, the bot heads it back to you.
Every completed pass adds 1 to the rally. Ball hits the floor -> rally over.

The bot here is RULE-BASED (no ML yet). It simulates the ball's future path
with the same physics as the game, runs to where the ball will come down,
and picks how off-centre to hit it so the pass lands on your head.
Later ML bots (imitation learning, RL) are compared against this baseline.

Reuses Ball / HeadTracker / try_header from header_game.py (same folder).

Run:
    python pass_game.py

Controls:
    SPACE / R  start / restart
    ESC / Q    quit
"""

import math
import random
import sys
import time

import cv2
import pygame

from header_game import (
    Ball, HeadTracker, try_header, frame_to_surface, draw_text,
    GRAVITY, BALL_R, HEAD_R, BASE_BOUNCE, POWER_GAIN, MAX_POWER_BONUS, STEER,
    HIT_COOLDOWN, MAX_DT,
    WHITE, BLACK, GREEN, RED, YELLOW, ORANGE,
)


# ------------------------------------------------------------------ bot tunables
BOT_FY = 0.55              # bot head height (fraction of screen height)
BOT_HOME_FX = 0.75         # where the bot waits when the ball is on your side
BOT_SPEED = 750.0          # max sideways speed, px/s
BOT_NOD = 600.0            # fake upward head speed at contact -> header power
BOT_BONUS = min(BOT_NOD * POWER_GAIN, MAX_POWER_BONUS)
SIM_DT = 1.0 / 120.0       # time step for the bot's "imagine the future" sim
SIM_HORIZON = 3.0          # seconds ahead the bot looks

BOT_COL = (70, 130, 255)


class RuleBot:
    """Baseline bot: predict where the ball comes down, go there, aim at the human."""

    def __init__(self, W, H):
        self.W, self.H = W, H
        self.mid = W / 2
        self.x = W * BOT_HOME_FX
        self.y = H * BOT_FY
        self.last_hit = 0.0
        self.target_x = self.x       # kept for drawing/debugging

    # -- 1. look into the future -------------------------------------------
    def predict_landing(self, ball, contact_y):
        """Copy the ball, step it forward with the SAME physics, and return the
        x where it falls through contact_y. None if it doesn't within the horizon."""
        sim = Ball(ball.x, ball.y)
        sim.vx, sim.vy = ball.vx, ball.vy
        for _ in range(int(SIM_HORIZON / SIM_DT)):
            sim.step(SIM_DT, self.W)
            if sim.vy > 0 and sim.y >= contact_y:
                return sim.x
        return None

    # -- 2. decide how to hit it -------------------------------------------
    def aim_offset(self, from_x, human_head):
        """How off-centre (-1..1, same meaning as in try_header) to hit the ball
        so it comes down on the human's head. Negative = send it left."""
        tx, ty = human_head if human_head else (self.W * 0.25, self.y)
        reach = BALL_R + HEAD_R
        y0 = self.y - reach * 0.9            # height where the ball leaves our head
        y1 = ty - reach * 0.9                # height where the human will meet it
        vy0 = -(BASE_BOUNCE + BOT_BONUS)     # launch speed (up is negative)
        # solve y0 + vy0*t + 0.5*g*t^2 = y1, take the later (falling) root
        disc = vy0 * vy0 - 2 * GRAVITY * (y0 - y1)
        t = (-vy0 + math.sqrt(max(disc, 0.0))) / GRAVITY
        vx_needed = (tx - from_x) / t
        return max(-1.0, min(-0.15, vx_needed / STEER))

    # -- 3. move ------------------------------------------------------------
    def update(self, dt, ball, human_head):
        reach = BALL_R + HEAD_R
        target = self.W * BOT_HOME_FX

        land_x = self.predict_landing(ball, self.y - reach * 0.9)
        if land_x is not None and land_x > self.mid:
            offset = self.aim_offset(land_x, human_head)
            # try_header uses dx = ball.x - head.x = offset * HEAD_R
            # so stand at head.x = ball.x - offset * HEAD_R
            # re-predict at the exact contact height for that offset
            contact_dy = math.sqrt(reach ** 2 - (offset * HEAD_R) ** 2)
            land_x2 = self.predict_landing(ball, self.y - contact_dy)
            if land_x2 is not None:
                land_x = land_x2
            target = land_x - offset * HEAD_R

        # stay on our own half
        target = max(self.mid + HEAD_R * 0.5, min(self.W - HEAD_R, target))
        self.target_x = target

        step = BOT_SPEED * dt
        self.x += max(-step, min(step, target - self.x))
        return self.x, self.y

    def velocity(self):
        # the bot "nods" upward at contact so its header has enough power
        return 0.0, -BOT_NOD

    def draw(self, surf, ball):
        cx, cy = int(self.x), int(self.y)
        pygame.draw.circle(surf, BOT_COL, (cx, cy), HEAD_R)
        pygame.draw.circle(surf, WHITE, (cx, cy), HEAD_R, 3)
        # eyes follow the ball
        ang = math.atan2(ball.y - cy, ball.x - cx)
        for ex in (cx - 22, cx + 22):
            ey = cy - 12
            pygame.draw.circle(surf, WHITE, (ex, ey), 12)
            pygame.draw.circle(surf, BLACK,
                               (int(ex + 5 * math.cos(ang)), int(ey + 5 * math.sin(ang))), 5)
        pygame.draw.arc(surf, WHITE, (cx - 22, cy + 5, 44, 26), math.pi * 1.1, math.pi * 1.9, 3)


def spawn_on_human_side(W, H):
    b = Ball(W * random.uniform(0.18, 0.32), H * 0.12)
    b.vx = random.uniform(-30, 30)
    return b


def draw_midline(surf, W, H):
    x = W // 2
    for y in range(0, H, 28):
        pygame.draw.line(surf, WHITE, (x, y), (x, y + 14), 2)


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
    mid = W / 2

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Header Pass — you + bot")
    clock = pygame.time.Clock()
    font_big = pygame.font.SysFont("consolas", 60, bold=True)
    font = pygame.font.SysFont("consolas", 30, bold=True)

    tracker = HeadTracker()
    bot = RuleBot(W, H)
    ball = spawn_on_human_side(W, H)
    state = "WAIT"
    rally, best = 0, 0
    last_touch = None          # "human" / "bot" — who headed it last
    human_last_hit = 0.0
    msg, msg_until = "", 0.0

    running = True
    while running:
        dt = min(clock.tick(60) / 1000.0, MAX_DT)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif e.key in (pygame.K_SPACE, pygame.K_r):
                    rally, last_touch = 0, None
                    ball = spawn_on_human_side(W, H)
                    state = "PLAY"

        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        head = tracker.update(rgb)

        screen.blit(frame_to_surface(rgb), (0, 0))
        draw_midline(screen, W, H)

        if state == "PLAY":
            if head is None:
                draw_text(screen, font, "FACE NOT DETECTED", (W // 2, 70), RED, center=True)
            else:
                ball.step(dt, W)
                now = time.perf_counter()

                # human — only counts while on the left half
                if (head[0] < mid and now - human_last_hit > HIT_COOLDOWN
                        and try_header(ball, head, tracker.velocity())):
                    human_last_hit = now
                    if last_touch == "bot":
                        rally += 1
                    last_touch = "human"

                # bot — same try_header(), same physics as the human
                bot_pos = bot.update(dt, ball, head)
                if (now - bot.last_hit > HIT_COOLDOWN
                        and try_header(ball, bot_pos, bot.velocity())):
                    bot.last_hit = now
                    if last_touch == "human":
                        rally += 1
                    last_touch = "bot"

                best = max(best, rally)

                if ball.y - BALL_R > H:          # dropped
                    who = "YOU" if ball.x < mid else "BOT"
                    msg, msg_until = f"{who} DROPPED IT  -  rally {rally}", now + 1.5
                    rally, last_touch = 0, None
                    ball = spawn_on_human_side(W, H)

        # ---------------- draw ----------------
        if head:
            col = GREEN if head[0] < mid else RED
            pygame.draw.circle(screen, col, (int(head[0]), int(head[1])), HEAD_R, 3)
            if head[0] >= mid and state == "PLAY":
                draw_text(screen, font, "STAY ON YOUR SIDE", (W // 4, H - 50), RED, center=True)
        bot.draw(screen, ball)
        if state == "PLAY":
            ball.draw(screen)

        draw_text(screen, font, f"RALLY {rally}", (16, 12), ORANGE)
        draw_text(screen, font, f"BEST {best}", (16, 48), YELLOW)
        draw_text(screen, font, "YOU", (W // 4, 20), WHITE, center=True)
        draw_text(screen, font, "BOT", (3 * W // 4, 20), BOT_COL, center=True)
        if time.perf_counter() < msg_until:
            draw_text(screen, font, msg, (W // 2, H // 2 + 80), RED, center=True)

        if state == "WAIT":
            draw_text(screen, font_big, "HEADER PASS", (W // 2, H // 2 - 40), ORANGE, center=True)
            draw_text(screen, font, "stand on the LEFT  -  SPACE to start",
                      (W // 2, H // 2 + 20), WHITE, center=True)

        pygame.display.flip()

    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()