"""
bots.py — the bots that play on the RIGHT half of pass_game.py.

Both bots share the same "body": same speed limit, same header power (nod),
same try_header() physics. Only the "brain" differs, so comparing them is fair.

  RuleBot — hand-written rules: simulate the ball, go where it lands, aim at you.
  MLBot   — imitation learning: a model trained on YOUR moves decides where to go.
  RLBot   — reinforcement learning: a PPO policy that learned by playing sim.py.
"""

import math
import os

import pygame

from header_game import (
    Ball, GRAVITY, BALL_R, HEAD_R, BASE_BOUNCE, POWER_GAIN, MAX_POWER_BONUS, STEER,
    WHITE, BLACK,
)

# ------------------------------------------------------------------ shared body
BOT_FY = 0.55              # rule bot head height (fraction of screen height)
BOT_HOME_FX = 0.75         # where the rule bot waits when the ball is on your side
BOT_SPEED = 750.0          # max head speed, px/s (both bots)
BOT_NOD = 600.0            # fake upward head speed at contact -> header power (both bots)
BOT_BONUS = min(BOT_NOD * POWER_GAIN, MAX_POWER_BONUS)
SIM_DT = 1.0 / 120.0       # rule bot's "imagine the future" time step
SIM_HORIZON = 3.0          # seconds ahead the rule bot looks

RULE_COL = (70, 130, 255)
ML_COL = (150, 90, 230)
RL_COL = (20, 170, 170)
ML_MODEL_PATH = "models/bc_mlp.pt"
RL_MODEL_PATH = "models/rl_ppo.zip"


class RuleBot:
    """Baseline bot: predict where the ball comes down, go there, aim at the human."""

    name = "RULE"

    def __init__(self, W, H):
        self.W, self.H = W, H
        self.mid = W / 2
        self.x = W * BOT_HOME_FX
        self.y = H * BOT_FY
        self.last_hit = 0.0
        self.color = RULE_COL

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
            contact_dy = math.sqrt(reach ** 2 - (offset * HEAD_R) ** 2)
            land_x2 = self.predict_landing(ball, self.y - contact_dy)
            if land_x2 is not None:
                land_x = land_x2
            target = land_x - offset * HEAD_R

        target = max(self.mid + HEAD_R * 0.5, min(self.W - HEAD_R, target))
        step = BOT_SPEED * dt
        self.x += max(-step, min(step, target - self.x))
        return self.x, self.y

    def velocity(self):
        # the bot "nods" upward at contact so its header has enough power
        return 0.0, -BOT_NOD

    def draw(self, surf, ball):
        cx, cy = int(self.x), int(self.y)
        pygame.draw.circle(surf, self.color, (cx, cy), HEAD_R)
        pygame.draw.circle(surf, WHITE, (cx, cy), HEAD_R, 3)
        # eyes follow the ball
        ang = math.atan2(ball.y - cy, ball.x - cx)
        for ex in (cx - 22, cx + 22):
            ey = cy - 12
            pygame.draw.circle(surf, WHITE, (ex, ey), 12)
            pygame.draw.circle(surf, BLACK,
                               (int(ex + 5 * math.cos(ang)), int(ey + 5 * math.sin(ang))), 5)
        pygame.draw.arc(surf, WHITE, (cx - 22, cy + 5, 44, 26), math.pi * 1.1, math.pi * 1.9, 3)


class MLBot(RuleBot):
    """Imitation-learning bot. Same body as RuleBot (speed limit, nod power,
    drawing); only update() — the brain — is replaced by the trained model.

    v2: the model predicts a TARGET POSITION ("where would Shreeya's head be in
    0.3 s?") and the bot walks there at the shared speed limit — exactly how the
    rule bot moves. v1 predicted a speed, small errors added up every frame and
    the bot slid into a wall (0% reach in eval_bots.py)."""

    name = "ML"

    def __init__(self, W, H, model_path=ML_MODEL_PATH):
        super().__init__(W, H)
        # imported here so the game still runs with RuleBot if torch/model is missing
        import torch
        from bc_model import load_policy
        from features import make_features

        self.torch = torch
        self.make_features = make_features
        self.model, self.info = load_policy(model_path)
        if "target_x" not in self.info["actions"]:
            raise FileNotFoundError(
                f"{model_path} is an old (speed) model - run prepare.py then train_bc.py")
        self.color = ML_COL
        # start where YOU usually stood, mirrored to the right half
        tx = float(self.model.y_mean[self.info["actions"].index("target_x")])
        ty = float(self.model.y_mean[self.info["actions"].index("target_y")])
        self.x, self.y = (1 - tx) * W, ty * H

    def update(self, dt, ball, human_head):
        W, H = self.W, self.H
        hx, hy = human_head if human_head else (W * 0.25, self.y)

        # 1. MIRROR the world: the model was trained as the LEFT player,
        #    so x -> 1 - x and vx -> -vx. The human becomes the "partner" on the right.
        feats = self.make_features(
            1 - ball.x / W, ball.y / H, -ball.vx / W, ball.vy / H,   # ball
            1 - hx / W, hy / H,                                      # partner (you)
        )

        # 2. ASK the model: "where would Shreeya's head be in 0.3 s?"
        with self.torch.no_grad():
            act = self.model(self.torch.from_numpy(feats).unsqueeze(0))[0].numpy()

        # 3. UN-MIRROR the target back to screen pixels
        target_x = (1 - float(act[0])) * W
        target_y = float(act[1]) * H
        target_x = max(self.mid + HEAD_R * 0.5, min(W - HEAD_R, target_x))
        target_y = max(H * 0.25, min(H * 0.8, target_y))

        # 4. walk there with the same body limit as the rule bot
        step = BOT_SPEED * dt
        self.x += max(-step, min(step, target_x - self.x))
        self.y += max(-step, min(step, target_y - self.y))
        return self.x, self.y


class RLBot(RuleBot):
    """Reinforcement-learning bot: a PPO policy trained by train_rl.py, playing
    hundreds of thousands of rallies in sim.py. Same body as the other bots;
    the brain is the policy network.

    Note the "sim-to-real gap": it learned against SimHuman, not a real person,
    so it may do worse with you than its eval_bots.py score suggests."""

    name = "RL"

    def __init__(self, W, H, model_path=RL_MODEL_PATH):
        super().__init__(W, H)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} not found - run train_rl.py")
        # imported here so the game still runs if stable-baselines3 isn't installed
        from stable_baselines3 import PPO
        from rl_env import make_obs, apply_action

        self.policy = PPO.load(model_path, device="cpu")
        self.make_obs = make_obs
        self.apply_action = apply_action
        self.color = RL_COL

    def update(self, dt, ball, human_head):
        hx, hy = human_head if human_head else (self.W * 0.25, self.H * 0.45)
        # exactly the observation/action code the env used in training (rl_env.py)
        obs = self.make_obs(ball, self.x, self.y, hx, hy, self.W, self.H)
        action, _ = self.policy.predict(obs, deterministic=True)
        self.x, self.y = self.apply_action(self.x, self.y, action, dt, self.W, self.H)
        return self.x, self.y