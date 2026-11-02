"""
prepare_data.py — raw gameplay CSVs -> clean training set for imitation learning.

  1. load every data/raw/session_*.csv
  2. print a per-session summary + save reports/data_overview.png
  3. build (state -> action) pairs:
        state  = features at frame t            (features.py)
        action = your head velocity t -> t+1    (what you did NEXT)
  4. clean: drop pauses/gaps, tracking-jitter spikes, and the second right
     before each of YOUR drops (we want the bot to copy success, not mistakes)
  5. save data/processed/bc_dataset.npz  (X, y, session)

Run:
    python prepare_data.py
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                 # save PNGs, no window needed
import matplotlib.pyplot as plt

from features import FEATURES, ACTIONS, make_features

RAW_GLOB = os.path.join("data", "raw", "session_*.csv")
OUT_PATH = os.path.join("data", "processed", "bc_dataset.npz")
REPORT_PATH = os.path.join("reports", "data_overview.png")

MAX_GAP = 0.1        # s — a bigger jump between rows = recording paused / face lost
MAX_SPEED = 3.0      # screens/s — head faster than this is tracking jitter, not you
FAIL_WINDOW = 1.0    # s — frames this close before YOUR drop are left out

# chart styling (one hue, quiet axes)
BLUE = "#2a78d6"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


# ------------------------------------------------------------------ 1. load
def load_sessions():
    files = sorted(glob.glob(RAW_GLOB))
    if not files:
        raise SystemExit(f"No files match {RAW_GLOB} — play pass_game.py first.")
    parts = []
    for i, path in enumerate(files):
        df = pd.read_csv(path)
        df["session"] = i
        df["file"] = os.path.basename(path)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------------ 2. look
def summarize(df):
    rows = []
    for s, d in df.groupby("session"):
        drops = d[d.dropped == 1]
        rows.append({
            "file": d.file.iloc[0],
            "frames": len(d),
            "minutes": round(len(d) * d.dt.median() / 60, 1),
            "fps": round(1 / d.dt.median()),
            "your_hits": int(d.human_hit.sum()),
            "bot_hits": int(d.bot_hit.sum()),
            "you_dropped": int((drops.ball_x < 0.5).sum()),
            "bot_dropped": int((drops.ball_x >= 0.5).sum()),
            "best_rally": int(d.rally.max()),
        })
    table = pd.DataFrame(rows)
    print("\n=== sessions ===")
    print(table.to_string(index=False))
    print(f"\nTOTAL: {table.frames.sum()} frames  ~{table.minutes.sum():.1f} min  "
          f"your hits {table.your_hits.sum()}  best rally {table.best_rally.max()}")


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.tick_params(colors=INK_2, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_overview(df, clean):
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor=SURFACE)

    # (a) where you stand — screen coords, so y points down
    ax = axes[0]
    style(ax, "Where your head was (all frames)")
    ax.grid(False)
    hb = ax.hexbin(df.head_x, df.head_y, gridsize=30, cmap="Blues",
                   extent=(0, 1, 0, 1), mincnt=1)
    ax.axvline(0.5, color=INK_2, linestyle="--", linewidth=1)
    ax.text(0.52, 0.05, "midline", color=INK_2, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_xlabel("x (fraction of screen)", color=INK_2)
    ax.set_ylabel("y (fraction of screen)", color=INK_2)
    fig.colorbar(hb, ax=ax, label="frames")

    # (b) the action the bot will learn
    ax = axes[1]
    style(ax, "Your sideways head speed (training frames)")
    ax.hist(clean.act_vx, bins=60, color=BLUE, edgecolor=SURFACE, linewidth=0.5)
    ax.set_xlabel("act_vx (screens per second, + = right)", color=INK_2)
    ax.set_ylabel("frames", color=INK_2)

    # (c) how long rallies lasted
    ax = axes[2]
    style(ax, "Rally length when the ball dropped")
    rallies = df.loc[df.dropped == 1, "rally"].astype(int)
    if len(rallies):
        counts = rallies.value_counts().sort_index()
        ax.bar(counts.index, counts.values, color=BLUE, width=0.8, edgecolor=SURFACE)
    ax.set_xlabel("rally (completed passes)", color=INK_2)
    ax.set_ylabel("drops", color=INK_2)

    fig.tight_layout()
    fig.savefig(REPORT_PATH, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved chart -> {REPORT_PATH}")


# ------------------------------------------------------------------ 3+4. pairs
def build_pairs(df):
    df = df.copy()
    g = df.groupby("session")
    gap = g["t"].shift(-1) - df["t"]                       # never crosses sessions
    df["act_vx"] = (g["head_x"].shift(-1) - df["head_x"]) / gap
    df["act_vy"] = (g["head_y"].shift(-1) - df["head_y"]) / gap

    ok_gap = gap.notna() & (gap > 0) & (gap <= MAX_GAP)
    ok_speed = np.hypot(df.act_vx, df.act_vy) <= MAX_SPEED

    # frames in the FAIL_WINDOW seconds before each drop on YOUR side
    before_fail = np.zeros(len(df), dtype=bool)
    your_drops = df[(df.dropped == 1) & (df.ball_x < 0.5)]
    sess, t = df.session.to_numpy(), df.t.to_numpy()
    for s, t_drop in zip(your_drops.session, your_drops.t):
        before_fail |= (sess == s) & (t > t_drop - FAIL_WINDOW) & (t <= t_drop)

    keep = ok_gap & ok_speed & ~before_fail
    print("\n=== cleaning ===")
    print(f"all frames            {len(df)}")
    print(f"  - gap / pause       {int((~ok_gap).sum())}")
    print(f"  - jitter spikes     {int((ok_gap & ~ok_speed).sum())}")
    print(f"  - before your drop  {int((ok_gap & ok_speed & before_fail).sum())}")
    print(f"kept for training     {int(keep.sum())}")

    clean = df[keep]
    X = make_features(clean.ball_x, clean.ball_y, clean.ball_vx, clean.ball_vy,
                      clean.head_x, clean.head_y, clean.bot_x, clean.bot_y)
    y = clean[ACTIONS].to_numpy(dtype=np.float32)
    return X, y, clean.session.to_numpy(), clean


# ------------------------------------------------------------------ main
def main():
    df = load_sessions()
    summarize(df)
    X, y, session, clean = build_pairs(df)
    plot_overview(df, clean)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez_compressed(OUT_PATH, X=X, y=y, session=session,
                        feature_names=np.array(FEATURES), action_names=np.array(ACTIONS))
    print(f"saved dataset -> {OUT_PATH}   X {X.shape}  y {y.shape}  "
          f"sessions {len(np.unique(session))}")


if __name__ == "__main__":
    main()
    