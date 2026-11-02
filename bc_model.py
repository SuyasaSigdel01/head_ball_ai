"""
bc_model.py — the bot's "brain" for behaviour cloning (imitation learning).

Kept separate from train_bc.py so the game can load the model later
without importing any training code.

The feature/action normalisation lives INSIDE the model (as buffers), so:
  - raw features in  -> raw actions out, both in training and in the game
  - the numbers needed to normalise are saved together with the weights
  - a later ONNX export (for the web version) carries them along automatically
"""

import torch
from torch import nn


class BCPolicy(nn.Module):
    def __init__(self, n_in, n_out, hidden, x_mean, x_std, y_mean, y_std):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_out),
        )
        # buffers = saved in state_dict, moved with .to(device), but not trained
        self.register_buffer("x_mean", torch.as_tensor(x_mean, dtype=torch.float32))
        self.register_buffer("x_std", torch.as_tensor(x_std, dtype=torch.float32))
        self.register_buffer("y_mean", torch.as_tensor(y_mean, dtype=torch.float32))
        self.register_buffer("y_std", torch.as_tensor(y_std, dtype=torch.float32))

    def forward(self, x):
        """x: raw features (N, n_in)  ->  raw actions (N, n_out)"""
        z = (x - self.x_mean) / self.x_std
        return self.net(z) * self.y_std + self.y_mean


def load_policy(path, device="cpu"):
    """Load a model saved by train_bc.py. Returns (model, checkpoint_info)."""
    ckpt = torch.load(path, map_location=device)
    sd = ckpt["state_dict"]
    model = BCPolicy(len(ckpt["features"]), len(ckpt["actions"]), ckpt["hidden"],
                     sd["x_mean"], sd["x_std"], sd["y_mean"], sd["y_std"])
    model.load_state_dict(sd)
    model.eval()
    return model, ckpt