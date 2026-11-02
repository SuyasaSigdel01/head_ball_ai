"""
features.py — the ONE place that turns game state into model input.

prepare.py uses it to build the training set, and the ML bot uses the same
function while playing. Same code on both sides = the model sees exactly the
same numbers in training and in the game (no "train/serve skew").

Canonical view: the player is ALWAYS on the LEFT half, the partner on the right.
Your recorded data is already like that. The bot plays on the RIGHT, so it
mirrors the world before asking the model (x -> 1 - x, vx -> -vx) and mirrors
the model's answer back. One model, either side of the court.

v2 — what changed and why:
  - ACTION is now a TARGET POSITION ("where my head will be in LOOKAHEAD s"),
    not a speed. A slightly wrong speed is added up every frame and the bot
    slides into a wall (compounding error — what the v1 ML bot did).
    A slightly wrong target just means standing a few pixels off.
  - The player's OWN position is no longer an input. Otherwise the model learns
    "target = where I am now + a bit" and the drift comes straight back.
    It now answers: "given the ball and my partner, where should I be?"
"""

import numpy as np

FEATURES = [
    "ball_x", "ball_y", "ball_vx", "ball_vy",   # where the ball is / where it's going
    "partner_x", "partner_y",                   # who I'm passing to
]
ACTIONS = ["target_x", "target_y"]              # where my head will be LOOKAHEAD s later
LOOKAHEAD = 0.3                                 # seconds


def make_features(ball_x, ball_y, ball_vx, ball_vy, partner_x, partner_y):
    """All inputs normalised (x/W, y/H, vx/W, vy/H).
    Works on single numbers (in-game) or whole pandas columns (training)."""
    cols = [ball_x, ball_y, ball_vx, ball_vy, partner_x, partner_y]
    return np.stack([np.asarray(c, dtype=np.float32) for c in cols], axis=-1)


def mirror_x(x):
    """Flip a normalised x position left <-> right."""
    return 1.0 - x


def mirror_vx(vx):
    """Flip a sideways velocity left <-> right."""
    return -vx