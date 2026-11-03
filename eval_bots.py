"""
eval_bots.py — fair, repeatable bot comparison in the headless simulator (sim.py).

Every bot faces the SAME simulated partner and the SAME random balls (fixed seed),
so any difference comes from the bot's brain only.

  STILL — never moves (the floor: any real bot must beat this)
  RULE  — hand-written baseline
  ML    — imitation-learning model (models/bc_mlp.pt), skipped if not available
  RL    — PPO policy (models/rl_ppo.zip), skipped if not available
          (careful: RL trained in this same simulator, so this is its home ground)

Run:
    python eval_bots.py
    python eval_bots.py --rallies 500

Saves the numbers to reports/bot_eval.json (for the README later).
"""

import argparse
import json
import os
import random
import statistics
import time

from bots import RuleBot, MLBot, RLBot
from sim import SimHuman, run_rally, W, H

LOAD_ERRORS = (ImportError, OSError, ValueError, RuntimeError)

OUT_PATH = os.path.join("reports", "bot_eval.json")


class StillBot(RuleBot):
    """Lower bound: stands at home and never moves."""
    name = "STILL"

    def update(self, dt, ball, human_head):
        return self.x, self.y


def evaluate(bot, n_rallies, seed):
    rng = random.Random(seed)                  # same balls for every bot
    x0, y0 = bot.x, bot.y
    totals = {"bot_hits": 0, "bot_drops": 0, "human_hits": 0, "human_drops": 0, "capped": 0}
    lengths = []
    start = time.perf_counter()
    for i in range(n_rallies):
        bot.x, bot.y, bot.last_hit = x0, y0, -1.0
        human = SimHuman(random.Random(seed * 100_000 + i))   # same "you" for every bot
        r = run_rally(bot, human, rng)
        lengths.append(r["passes"])
        for k in totals:
            totals[k] += r[k]

    def pct(a, b):
        return round(100 * a / b, 1) if b else 0.0

    return {
        "rallies": n_rallies,
        "avg_rally": round(statistics.mean(lengths), 2),
        "median_rally": statistics.median(lengths),
        "hit_cap_pct": pct(totals["capped"], n_rallies),
        "bot_reach_pct": pct(totals["bot_hits"], totals["bot_hits"] + totals["bot_drops"]),
        "pass_quality_pct": pct(totals["human_hits"], totals["human_hits"] + totals["human_drops"]),
        "seconds": round(time.perf_counter() - start, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rallies", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bots = [StillBot(W, H), RuleBot(W, H)]
    for cls in (MLBot, RLBot):
        try:
            bots.append(cls(W, H))
        except LOAD_ERRORS as err:
            print(f"({cls.name} bot skipped: {err})")

    results = {}
    for bot in bots:
        print(f"evaluating {bot.name} ...", flush=True)
        results[bot.name] = evaluate(bot, args.rallies, args.seed)

    print(f"\n=== {args.rallies} rallies each, seed {args.seed} ===")
    print(f"{'bot':6s} {'avg rally':>9s} {'median':>7s} {'hit cap':>8s} "
          f"{'bot reach':>10s} {'pass quality':>13s} {'time':>6s}")
    for name, r in results.items():
        print(f"{name:6s} {r['avg_rally']:9.2f} {r['median_rally']:7.1f} {r['hit_cap_pct']:7.1f}% "
              f"{r['bot_reach_pct']:9.1f}% {r['pass_quality_pct']:12.1f}% {r['seconds']:5.1f}s")
    print("\nbot reach    = balls you sent over that the bot got to")
    print("pass quality = bot's passes that 'you' could head back")
    print("hit cap      = rallies that reached 30 passes (sim stops there)")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({"rallies": args.rallies, "seed": args.seed, "results": results}, f, indent=2)
    print(f"\nsaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()