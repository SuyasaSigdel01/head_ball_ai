"""
prepare.py — raw gameplay CSVs -> clean training set for imitation learning.

  1. load every data/raw/session_*.csv
  2. print a per-session summary + save reports/data_overview.png
  3. build (state -> action) pairs:
        state  = ball + partner at frame t                      (features.py)
        action = where YOUR head was LOOKAHEAD seconds later    (target position)
  4. clean: drop windows with a pause/gap, tracking-jitter jumps, and the second
     right before each of YOUR drops (copy success, not mistakes)
  5. save data/processed/bc_dataset.npz  (X, y, session)

Run:
    python prepare.py
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                 # save PNGs, no window needed
import matplotlib.pyplot as plt

from features import FEATURES, ACTIONS, LOOKAHEAD, make_features

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
    style(ax, f"Where your head went {LOOKAHEAD}s later (training frames)")
    ax.hist(clean.target_x, bins=50, range=(0, 0.6), color=BLUE,
            edgecolor=SURFACE, linewidth=0.5)
    ax.axvline(0.5, color=INK_2, linestyle="--", linewidth=1)
    ax.set_xlabel("target_x (fraction of screen)", color=INK_2)
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
    n = len(df)
    t_all = df.t.to_numpy()
    hx_all, hy_all = df.head_x.to_numpy(), df.head_y.to_numpy()

    target_x = np.full(n, np.nan)
    target_y = np.full(n, np.nan)
    ok_window = np.zeros(n, dtype=bool)
    ok_speed = np.zeros(n, dtype=bool)

    for _, idx in df.groupby("session").indices.items():     # never crosses sessions
        t, hx, hy = t_all[idx], hx_all[idx], hy_all[idx]
        pauses = np.cumsum(np.diff(t, prepend=t[0]) > MAX_GAP)  # pauses seen so far
        j = np.searchsorted(t, t + LOOKAHEAD)                   # row LOOKAHEAD s later
        has_future = j < len(t)
        j = np.minimum(j, len(t) - 1)
        no_pause = pauses[j] == pauses                          # no gap inside the window
        good = has_future & no_pause

        span = np.maximum(t[j] - t, 1e-6)
        speed = np.hypot(hx[j] - hx, hy[j] - hy) / span

        target_x[idx] = np.where(good, hx[j], np.nan)
        target_y[idx] = np.where(good, hy[j], np.nan)
        ok_window[idx] = good
        ok_speed[idx] = speed <= MAX_SPEED

    df["target_x"], df["target_y"] = target_x, target_y

    # frames in the FAIL_WINDOW seconds before each drop on YOUR side
    before_fail = np.zeros(n, dtype=bool)
    your_drops = df[(df.dropped == 1) & (df.ball_x < 0.5)]
    sess = df.session.to_numpy()
    for s, t_drop in zip(your_drops.session, your_drops.t):
        before_fail |= (sess == s) & (t_all > t_drop - FAIL_WINDOW) & (t_all <= t_drop)

    keep = ok_window & ok_speed & ~before_fail
    print("\n=== cleaning ===")
    print(f"all frames            {n}")
    print(f"  - gap / pause / end {int((~ok_window).sum())}")
    print(f"  - jitter jumps      {int((ok_window & ~ok_speed).sum())}")
    print(f"  - before your drop  {int((ok_window & ok_speed & before_fail).sum())}")
    print(f"kept for training     {int(keep.sum())}")

    clean = df[keep]
    X = make_features(clean.ball_x, clean.ball_y, clean.ball_vx, clean.ball_vy,
                      clean.bot_x, clean.bot_y)
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