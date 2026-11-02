"""
pass_game.py — co-op heading rally: you (webcam) + a bot

Stand on the LEFT half of the camera. The bot lives on the RIGHT half.
Head the ball across the middle line, the bot heads it back to you.
Every completed pass adds 1 to the rally. Ball hits the floor -> rally over.

Two bots (see bots.py), switch with B:
  RULE — hand-written baseline
  ML   — imitation-learning model trained on your own play (models/bc_mlp.pt)
The bottom of the screen shows average rally per bot, so you can compare them.

Every frame you play is recorded to data/raw/ (see data_logger.py) — that's
the training data for the imitation-learning bot (prepare.py -> train_bc.py).

Run:
    python pass_game.py

Controls:
    SPACE / R  start / restart
    B          switch bot (RULE <-> ML)
    L          pause / resume recording
    ESC / Q    quit
"""

import random
import sys
import time

import cv2
import pygame

from header_game import (
    Ball, HeadTracker, try_header, frame_to_surface, draw_text,
    BALL_R, HEAD_R, HIT_COOLDOWN, MAX_DT,
    WHITE, GREEN, RED, YELLOW, ORANGE,
)
from data_logger import GameLogger
from bots import RuleBot, MLBot


def spawn_on_human_side(W, H):
    b = Ball(W * random.uniform(0.18, 0.32), H * 0.12)
    b.vx = random.uniform(-30, 30)
    return b


def draw_midline(surf, W, H):
    x = W // 2
    for y in range(0, H, 28):
        pygame.draw.line(surf, WHITE, (x, y), (x, y + 14), 2)


def make_bot(kind, W, H):
    """Return a bot, or None if the ML bot can't load (no torch / no model yet)."""
    if kind == "ML":
        try:
            return MLBot(W, H)
        except (ImportError, FileNotFoundError, OSError) as err:
            print(f"ML bot unavailable: {err}")
            return None
    return RuleBot(W, H)


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
    font_small = pygame.font.SysFont("consolas", 22, bold=True)

    tracker = HeadTracker()
    bot = RuleBot(W, H)
    logger = GameLogger(W, H)
    ball = spawn_on_human_side(W, H)
    state = "WAIT"
    rally, best = 0, 0
    last_touch = None          # "human" / "bot" — who headed it last
    human_last_hit = 0.0
    msg, msg_until = "", 0.0
    rallies = {"RULE": [], "ML": []}   # finished rally lengths per bot

    running = True
    while running:
        dt = min(clock.tick(60) / 1000.0, MAX_DT)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif e.key == pygame.K_l:
                    logger.enabled = not logger.enabled
                elif e.key == pygame.K_b:
                    new = make_bot("ML" if bot.name == "RULE" else "RULE", W, H)
                    if new is None:
                        msg, msg_until = "NO ML MODEL - run train_bc.py", time.perf_counter() + 2
                    else:
                        bot = new
                        rally, last_touch = 0, None          # fresh rally for a fair count
                        ball = spawn_on_human_side(W, H)
                        msg, msg_until = f"{bot.name} BOT", time.perf_counter() + 1.2
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
                human_hit = bot_hit = False

                # human — only counts while on the left half
                if (head[0] < mid and now - human_last_hit > HIT_COOLDOWN
                        and try_header(ball, head, tracker.velocity())):
                    human_last_hit = now
                    human_hit = True
                    if last_touch == "bot":
                        rally += 1
                    last_touch = "human"

                # bot — same try_header(), same physics as the human
                bot_pos = bot.update(dt, ball, head)
                if (now - bot.last_hit > HIT_COOLDOWN
                        and try_header(ball, bot_pos, bot.velocity())):
                    bot.last_hit = now
                    bot_hit = True
                    if last_touch == "human":
                        rally += 1
                    last_touch = "bot"

                best = max(best, rally)
                dropped = ball.y - BALL_R > H

                # record this frame BEFORE respawning, so the drop is in the data
                logger.log(dt, ball, head, tracker.velocity(), bot_pos,
                           human_hit, bot_hit, dropped, rally)

                if dropped:
                    who = "YOU" if ball.x < mid else "BOT"
                    msg, msg_until = f"{who} DROPPED IT  -  rally {rally}", now + 1.5
                    rallies[bot.name].append(rally)
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
        draw_text(screen, font, f"{bot.name} BOT", (3 * W // 4, 20), bot.color, center=True)
        if logger.enabled:
            pygame.draw.circle(screen, RED, (W - 200, 64), 9)
            draw_text(screen, font, f"REC {logger.rows}", (W - 182, 48), RED)
        else:
            draw_text(screen, font, "REC OFF (L)", (W - 210, 48), WHITE)

        # bot comparison: average rally per bot
        parts = []
        for name, rs in rallies.items():
            if rs:
                parts.append(f"{name}: avg {sum(rs) / len(rs):.1f}  best {max(rs)}  ({len(rs)} rallies)")
        stats = "   |   ".join(parts) if parts else "B: switch bot"
        draw_text(screen, font_small, stats, (W - 16 - font_small.size(stats)[0], H - 34), WHITE)

        if time.perf_counter() < msg_until:
            draw_text(screen, font, msg, (W // 2, H // 2 + 80), RED, center=True)

        if state == "WAIT":
            draw_text(screen, font_big, "HEADER PASS", (W // 2, H // 2 - 40), ORANGE, center=True)
            draw_text(screen, font, "stand on the LEFT  -  SPACE to start  -  B: switch bot",
                      (W // 2, H // 2 + 20), WHITE, center=True)

        pygame.display.flip()

    path, rows = logger.close()
    print(f"saved {rows} frames -> {path}" if rows else "nothing recorded")
    for name, rs in rallies.items():
        if rs:
            print(f"{name} bot: avg rally {sum(rs) / len(rs):.2f}, best {max(rs)}, {len(rs)} rallies")
    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()