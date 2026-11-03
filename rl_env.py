"""
rl_env.py — the pass game as a Gymnasium environment, for reinforcement learning.

The agent is the bot on the RIGHT half. The partner on the left is SimHuman
(sim.py). Physics, header rule, speed limit and nod power are exactly the same
as the other bots, so the comparison in eval_bots.py stays fair.

Each step = 1/60 s of game time.

  observation (8 numbers, normalised by screen size):
      ball x, y, vx, vy | my head x, y | partner head x, y
  action (2 numbers in [-1, 1]):
      how fast to move my head sideways / up-down (x BOT_SPEED)
  reward:
      +1    I head a ball that my partner sent over         (receive)
      +1    my partner heads back a ball I sent             (good pass)
      -1    the ball drops on my side                       (miss)
      -0.5  my pass drops on the partner's side             (bad pass)
      +0.01 per frame while the partner's ball falls on my side,
            scaled by how close I am under it               (shaping: a hint
            so the bot learns faster; only for INCOMING balls, otherwise
            the bot could juggle to itself and farm it)

Why RL can beat imitation here: behaviour cloning only ever saw YOUR states,
so its own small mistakes lead it into situations it never learned about.
An RL bot learns from its OWN states and its own mistakes.
"""

import random

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from header_game import Ball, try_header, BALL_R, HEAD_R, HIT_COOLDOWN
from bots import BOT_SPEED, BOT_NOD, BOT_HOME_FX, BOT_FY
from sim import SimHuman, W, H, SIM_FPS

Y_MIN, Y_MAX = 0.3, 0.8          # where the bot's head may go (fraction of H)

R_RECEIVE = 1.0
R_PASS_RETURNED = 1.0
R_DROP_MINE = -1.0
R_BAD_PASS = -0.5
R_SHAPE = 0.01


# ---- shared with RLBot in bots.py, so training and playing see the same numbers
def make_obs(ball, bot_x, bot_y, partner_x, partner_y, W, H):
    return np.array([ball.x / W, ball.y / H, ball.vx / W, ball.vy / H,
                     bot_x / W, bot_y / H, partner_x / W, partner_y / H],
                    dtype=np.float32)


def apply_action(x, y, action, dt, W, H):
    """action in [-1, 1]^2 -> head movement, same speed limit as the other bots."""
    ax = float(np.clip(action[0], -1.0, 1.0))
    ay = float(np.clip(action[1], -1.0, 1.0))
    x = min(max(x + ax * BOT_SPEED * dt, W / 2 + HEAD_R * 0.5), W - HEAD_R)
    y = min(max(y + ay * BOT_SPEED * dt, H * Y_MIN), H * Y_MAX)
    return x, y


class HeaderPassEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_time=60.0, max_passes=30):
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.dt = 1.0 / SIM_FPS
        self.max_time = max_time
        self.max_passes = max_passes

    def _obs(self):
        return make_obs(self.ball, self.x, self.y, self.human.x, self.human.y, W, H)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = random.Random(int(self.np_random.integers(2 ** 31)))
        self.human = SimHuman(random.Random(rng.random()))
        self.ball = Ball(W * rng.uniform(0.18, 0.32), H * 0.12)
        self.ball.vx = rng.uniform(-30, 30)
        self.human.new_approach()
        self.x, self.y = W * BOT_HOME_FX, H * BOT_FY
        self.t = 0.0
        self.bot_last_hit = -1.0
        self.last_touch = None
        self.passes = 0
        return self._obs(), {}

    def step(self, action):
        dt, ball, human = self.dt, self.ball, self.human
        self.t += dt
        reward = 0.0
        ball.step(dt, W)

        # partner (same order as sim.run_rally: ball, partner, bot)
        hpos = human.update(dt, ball, (self.x, self.y))
        if self.t - human.last_hit > HIT_COOLDOWN and try_header(ball, hpos, human.velocity()):
            human.last_hit = self.t
            if self.last_touch == "bot":
                reward += R_PASS_RETURNED
                self.passes += 1
            self.last_touch = "human"
            human.new_approach()

        # me
        self.x, self.y = apply_action(self.x, self.y, action, dt, W, H)
        if (self.t - self.bot_last_hit > HIT_COOLDOWN
                and try_header(ball, (self.x, self.y), (0.0, -BOT_NOD))):
            self.bot_last_hit = self.t
            if self.last_touch == "human":
                reward += R_RECEIVE
                self.passes += 1
            self.last_touch = "bot"
            human.new_approach()

        # shaping: partner's ball is coming down on my side -> be under it
        if self.last_touch == "human" and ball.x > W / 2 and ball.vy > 0:
            reward += R_SHAPE * (1.0 - min(abs(ball.x - self.x) / (W / 2), 1.0))

        terminated = truncated = False
        if ball.y - BALL_R > H:
            terminated = True
            if ball.x >= W / 2:
                reward += R_DROP_MINE
            elif self.last_touch == "bot":
                reward += R_BAD_PASS
        elif self.passes >= self.max_passes or self.t >= self.max_time:
            truncated = True

        return self._obs(), float(reward), terminated, truncated, {"passes": self.passes}