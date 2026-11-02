"""
train_bc.py — train the first ML bot brain: behaviour cloning (imitation learning).

Input : data/processed/bc_dataset.npz   (made by prepare_data.py)
Output: models/bc_mlp.pt                (weights + normaliser + info)
        reports/bc_training.png         (learning curve)

Model : small MLP, 10 features -> 64 -> 64 -> 2 actions (head vx, head vy)
Split : by SESSION, not by random frame — see split_by_session()
Score : R^2 vs a "dumb" baseline that always predicts the average move

Run:
    python train_bc.py
"""

import copy
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bc_model import BCPolicy

DATA_PATH = os.path.join("data", "processed", "bc_dataset.npz")
MODEL_PATH = os.path.join("models", "bc_mlp.pt")
REPORT_PATH = os.path.join("reports", "bc_training.png")

SEED = 0
HIDDEN = 64
EPOCHS = 200
BATCH = 256
LR = 1e-3
PATIENCE = 15          # stop if validation hasn't improved for this many epochs
VAL_FRACTION = 0.2

# chart styling
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def split_by_session(session):
    """Hold out WHOLE sessions for validation.

    At 30 fps, frame 100 and frame 101 are almost identical. A random split puts
    one in train and the other in validation, so the model just "remembers" the
    neighbour and the validation score looks great but means nothing (data leakage).
    Holding out a whole session asks the real question: does it work on a game
    it has never seen?
    """
    ids = np.unique(session)
    if len(ids) >= 2:
        n_val = max(1, round(len(ids) * VAL_FRACTION))
        val_ids = ids[-n_val:]                       # most recent session(s)
        val = np.isin(session, val_ids)
        how = f"held out session(s) {val_ids.tolist()} of {len(ids)}"
    else:
        cut = int(len(session) * (1 - VAL_FRACTION))  # last 20% in time order
        val = np.zeros(len(session), dtype=bool)
        val[cut:] = True
        how = "only 1 session -> held out its last 20% (record more sessions!)"
    return ~val, val, how


def plot_curve(history, best_epoch):
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    tr = [h[0] for h in history]
    va = [h[1] for h in history]
    ep = np.arange(1, len(history) + 1)

    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(ep, tr, color=BLUE, linewidth=2, label="train")
    ax.plot(ep, va, color=ORANGE, linewidth=2, label="validation (unseen session)")
    ax.axvline(best_epoch + 1, color=INK_2, linestyle="--", linewidth=1)
    ax.text(best_epoch + 1.5, max(max(tr), max(va)) * 0.95, "best model kept",
            color=INK_2, fontsize=9)
    # direct labels at the line ends
    ax.text(ep[-1] + 0.5, tr[-1], " train", color=INK, fontsize=9, va="center")
    ax.text(ep[-1] + 0.5, va[-1], " validation", color=INK, fontsize=9, va="center")
    ax.set_title("Behaviour cloning — loss per epoch (lower is better)",
                 color=INK, fontsize=11, loc="left")
    ax.set_xlabel("epoch", color=INK_2)
    ax.set_ylabel("MSE (normalised)", color=INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper right")
    ax.set_xlim(1, ep[-1] * 1.12 + 1)
    fig.tight_layout()
    fig.savefig(REPORT_PATH, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved chart -> {REPORT_PATH}")


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    d = np.load(DATA_PATH)
    X, y, session = d["X"], d["y"], d["session"]
    features = d["feature_names"].tolist()
    actions = d["action_names"].tolist()

    tr, va, how = split_by_session(session)
    print(f"split: {how}")
    print(f"train {tr.sum()} frames | validation {va.sum()} frames")

    # normalisation stats from TRAIN only (validation must stay "unseen")
    x_mean, x_std = X[tr].mean(0), X[tr].std(0) + 1e-6
    y_mean, y_std = y[tr].mean(0), y[tr].std(0) + 1e-6

    model = BCPolicy(X.shape[1], y.shape[1], HIDDEN, x_mean, x_std, y_mean, y_std)
    Xtr, ytr = torch.tensor(X[tr]), torch.tensor(y[tr])
    Xva, yva = torch.tensor(X[va]), torch.tensor(y[va])
    y_std_t = torch.tensor(y_std)

    def loss_fn(pred, target):
        # MSE measured in "standard deviations", so vx and vy count equally
        return (((pred - target) / y_std_t) ** 2).mean()

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best_val, best_epoch, best_state = float("inf"), 0, None
    history = []

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr))
        total = 0.0
        for i in range(0, len(Xtr), BATCH):
            idx = perm[i:i + BATCH]
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        train_loss = total / len(Xtr)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva), yva).item()
        history.append((train_loss, val_loss))

        if val_loss < best_val - 1e-5:
            best_val, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch % 10 == 0:
            print(f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}")
        if epoch - best_epoch >= PATIENCE:
            print(f"early stop at epoch {epoch} (best was {best_epoch})")
            break

    model.load_state_dict(best_state)

    # ---- how good is it? compare with "always predict the average move"
    with torch.no_grad():
        pred = model(Xva).numpy()
    yv = y[va]
    mse_model = (((pred - yv) / y_std) ** 2).mean(0)
    mse_mean = (((y_mean - yv) / y_std) ** 2).mean(0)
    r2 = 1 - mse_model / mse_mean
    print("\n=== validation (unseen session) ===")
    print(f"{'action':8s} {'baseline MSE':>13s} {'model MSE':>10s} {'R^2':>6s}")
    for name, b, m, r in zip(actions, mse_mean, mse_model, r2):
        print(f"{name:8s} {b:13.3f} {m:10.3f} {r:6.2f}")
    print("R^2: 0 = no better than the average move, 1 = perfect copy of you")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "hidden": HIDDEN,
        "features": features,
        "actions": actions,
        "val_r2": dict(zip(actions, r2.round(3).tolist())),
        "split": how,
        "best_epoch": best_epoch,
    }, MODEL_PATH)
    print(f"\nsaved model -> {MODEL_PATH}")
    plot_curve(history, best_epoch)


if __name__ == "__main__":
    main()