"""
sim.py — the pass game WITHOUT camera or window: pure physics, much faster than real time.

Used for:
  - eval_bots.py: fair, repeatable bot comparison (same partner, same balls)
  - next step: training a reinforcement-learning bot, which needs millions of frames
    — far too many to play by hand

It uses the SAME Ball physics and try_header() as the real game, so whatever
works here works in pass_game.py too.
"""

import random

from header_game import Ball, try_header, BALL_R, HEAD_R, HIT_COOLDOWN

W, H = 1280, 720           # sim screen size (features are normalised, so any size works)
SIM_FPS = 60
PREDICT_DT = 1.0 / 60.0    # SimHuman's look-ahead step


class SimHuman:
    """A scripted stand-in for YOU on the left half.

    Chases where the ball will come down and heads it to the right — but with
    human-like limits: slower than the bots, and each approach has a random aim
    error, so it misses sometimes. Fixed seed -> every bot faces the same "you".
    """

    def __init__(self, rng, speed=600.0, aim_error_px=30.0, nod=350.0):
        self.rng = rng
        self.speed = speed
        self.aim_error_px = aim_error_px
        self.nod = nod
        self.x, self.y = 0.3 * W, 0.45 * H
        self.vx = 0.0
        self.err = 0.0
        self.last_hit = -1.0

    def new_approach(self):
        """Called whenever a new ball comes our way: pick this attempt's aim error."""
        self.err = self.rng.gauss(0.0, self.aim_error_px)

    def _predict_landing(self, ball, contact_y):
        sim = Ball(ball.x, ball.y)
        sim.vx, sim.vy = ball.vx, ball.vy
        for _ in range(int(3.0 / PREDICT_DT)):
            sim.step(PREDICT_DT, W)
            if sim.vy > 0 and sim.y >= contact_y:
                return sim.x
        return None

    def update(self, dt, ball, partner):
        mid = W / 2
        target = 0.3 * W
        land = self._predict_landing(ball, self.y - 0.9 * (BALL_R + HEAD_R))
        if land is not None and land < mid:
            # head.x left of the ball -> ball hits the right side of the head -> goes right
            target = land - 0.55 * HEAD_R + self.err
        step = self.speed * dt
        move = max(-step, min(step, target - self.x))
        self.vx = move / dt
        self.x = max(HEAD_R * 0.5, min(mid - HEAD_R * 0.5, self.x + move))
        return self.x, self.y

    def velocity(self):
        return self.vx, -self.nod


def run_rally(bot, human, rng, max_time=90.0, max_passes=30):
    """Play ONE rally (until the ball drops or the cap) and return what happened.

    Counts, from the bot's point of view:
      bot_hits / bot_drops        — balls you sent over: did the bot reach them?
      human_hits / human_drops    — balls the bot sent back: could "you" reach them?
                                    (= how good the bot's passes are)
    """
    ball = Ball(W * rng.uniform(0.18, 0.32), H * 0.12)
    ball.vx = rng.uniform(-30, 30)
    human.new_approach()

    t, dt = 0.0, 1.0 / SIM_FPS
    last_touch = None
    out = {"passes": 0, "bot_hits": 0, "bot_drops": 0,
           "human_hits": 0, "human_drops": 0, "capped": 0}

    while t < max_time:
        t += dt
        ball.step(dt, W)

        hpos = human.update(dt, ball, (bot.x, bot.y))
        if t - human.last_hit > HIT_COOLDOWN and try_header(ball, hpos, human.velocity()):
            human.last_hit = t
            if last_touch == "bot":
                out["passes"] += 1
                out["human_hits"] += 1
            last_touch = "human"

        bpos = bot.update(dt, ball, hpos)
        if t - bot.last_hit > HIT_COOLDOWN and try_header(ball, bpos, bot.velocity()):
            bot.last_hit = t
            if last_touch == "human":
                out["passes"] += 1
                out["bot_hits"] += 1
            last_touch = "bot"
            human.new_approach()

        if ball.y - BALL_R > H:                        # dropped
            if ball.x >= W / 2:
                out["bot_drops"] += last_touch == "human"
            else:
                out["human_drops"] += last_touch == "bot"
            break
        if out["passes"] >= max_passes:
            out["capped"] = 1
            break
    return out