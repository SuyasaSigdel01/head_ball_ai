"""
train_rl.py — train a bot with reinforcement learning (PPO) in the simulator.

No recorded data needed: the bot plays hundreds of thousands of rallies against
SimHuman (rl_env.py) and learns from the rewards.

Output: models/rl_ppo.zip           (the trained policy)
        logs/rl/*.monitor.csv       (one row per finished rally)
        reports/rl_training.png     (rally length while learning)

Run:
    python train_rl.py --steps 50000     # quick smoke test (~1-2 min)
    python train_rl.py                   # full run, 500k steps (~10-20 min on CPU)
"""

import argparse
import glob
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from rl_env import HeaderPassEnv

MODEL_PATH = os.path.join("models", "rl_ppo.zip")
LOG_DIR = os.path.join("logs", "rl")
REPORT_PATH = os.path.join("reports", "rl_training.png")
RULE_REFERENCE = 15.3       # RULE bot avg rally in eval_bots.py (same SimHuman)

BLUE = "#2a78d6"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def plot_progress():
    files = glob.glob(os.path.join(LOG_DIR, "*.monitor.csv"))
    df = pd.concat([pd.read_csv(f, skiprows=1) for f in files]).sort_values("t")
    df["steps"] = df["l"].cumsum()
    smooth = df["passes"].rolling(200, min_periods=20).mean()

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(df["steps"], smooth, color=BLUE, linewidth=2)
    ax.axhline(RULE_REFERENCE, color=INK_2, linestyle="--", linewidth=1)
    ax.text(df["steps"].iloc[0], RULE_REFERENCE + 0.4, "RULE bot", color=INK_2, fontsize=9)
    ax.set_title("RL bot — rally length while training (avg of last 200 rallies)",
                 color=INK, fontsize=11, loc="left")
    ax.set_xlabel("training steps (1 step = 1/60 s of game)", color=INK_2)
    ax.set_ylabel("passes per rally", color=INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(REPORT_PATH, dpi=120, facecolor=SURFACE)
    plt.close(fig)

    first = df["passes"].head(200).mean()
    last = df["passes"].tail(200).mean()
    print(f"\nrallies played: {len(df)}")
    print(f"avg passes per rally — first 200: {first:.2f}   last 200: {last:.2f}   "
          f"(RULE bot: {RULE_REFERENCE})")
    print(f"saved chart -> {REPORT_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    for f in glob.glob(os.path.join(LOG_DIR, "*.monitor.csv")):
        os.remove(f)                              # fresh log for this run

    # 8 copies of the game played side by side -> faster, more varied experience
    env = make_vec_env(HeaderPassEnv, n_envs=args.envs, seed=args.seed,
                       monitor_dir=LOG_DIR,
                       monitor_kwargs={"info_keywords": ("passes",)})

    model = PPO(
        "MlpPolicy", env,
        n_steps=1024,               # steps per env before each update
        batch_size=512,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.995,                # care about rewards ~3 s ahead (ball flight ~1 s)
        gae_lambda=0.95,
        ent_coef=0.005,             # a little extra exploration
        policy_kwargs={"net_arch": [64, 64]},
        seed=args.seed,
        device="cpu",
        verbose=1,
    )
    model.learn(total_timesteps=args.steps)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\nsaved model -> {MODEL_PATH}")
    plot_progress()


if __name__ == "__main__":
    main()