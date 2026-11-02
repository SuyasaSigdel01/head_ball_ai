"""
features.py — the ONE place that turns game state into model input.

prepare_data.py uses it to build the training set, and the ML bot will use the
same function while playing. Same code on both sides = the model sees exactly
the same numbers in training and in the game (no "train/serve skew").

Canonical view: the player is ALWAYS on the LEFT half, the partner on the right.
Your recorded data is already like that. The bot plays on the RIGHT, so it will
mirror the world before asking the model (x -> 1 - x, vx -> -vx) and mirror the
model's answer back. One model, either side of the court.
"""

import numpy as np

FEATURES = [
    "ball_x", "ball_y", "ball_vx", "ball_vy",   # where the ball is / where it's going
    "head_x", "head_y",                         # where I am
    "rel_x", "rel_y",                           # ball relative to my head
    "partner_x", "partner_y",                   # who I'm passing to
]
ACTIONS = ["act_vx", "act_vy"]                  # how my head moves next (screens/s)


def make_features(ball_x, ball_y, ball_vx, ball_vy,
                  head_x, head_y, partner_x, partner_y):
    """All inputs normalised (x/W, y/H, vx/W, vy/H).
    Works on single numbers (in-game) or whole pandas columns (training)."""
    cols = [ball_x, ball_y, ball_vx, ball_vy,
            head_x, head_y,
            ball_x - head_x, ball_y - head_y,
            partner_x, partner_y]
    return np.stack([np.asarray(c, dtype=np.float32) for c in cols], axis=-1)


def mirror_x(x):
    """Flip a normalised x position left <-> right."""
    return 1.0 - x


def mirror_vx(vx):
    """Flip a sideways velocity left <-> right."""
    return -vx