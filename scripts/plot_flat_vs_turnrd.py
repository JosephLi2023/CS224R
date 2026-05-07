#!/usr/bin/env python3
"""Three-panel comparison plot: flat_grpo vs turnRD (method_b_lean).

Reads per-round `train_log.json` files from two glob patterns and writes
a PNG with:
  • Top: per-episode training mean_reward (5-pt moving average), all
    rounds concatenated in order, one curve per method.
  • Mid: per-round held-out eval avg_return (dots+line).
  • Bot: per-round held-out eval pct_success (dots+line).

Usage:
    python3 scripts/plot_flat_vs_turnrd.py \\
        --flat-glob 'experiments/manifests/_flat_grpo/flat_grpo_compare_seed11_round*_train_log.json' \\
        --turnrd-glob 'experiments/manifests/_baseline_turnrd/method_b_lean_seed11_round*_train_log.json' \\
        --out experiments/manifests/flat_grpo_vs_turnrd.png
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROUND_RE = re.compile(r"round(\d+)")


def _round_idx(p: str) -> int:
    m = ROUND_RE.search(Path(p).name)
    return int(m.group(1)) if m else -1


def _moving_average(xs: list[float], window: int = 5) -> list[float]:
    if window <= 1 or not xs:
        return xs
    out = []
    for i in range(len(xs)):
        lo = max(0, i - window + 1)
        chunk = xs[lo : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def _load(pattern: str) -> dict:
    paths = sorted(glob.glob(pattern), key=_round_idx)
    train_rewards: list[float] = []
    round_boundaries: list[int] = []  # episode index where each new round starts
    eval_rounds: list[int] = []
    eval_R: list[float] = []
    eval_succ: list[float] = []
    for p in paths:
        d = json.load(open(p))
        ridx = _round_idx(p)
        rows = [r for r in d.get("rows", []) if "mean_reward" in r]
        if rows:
            round_boundaries.append(len(train_rewards))
            train_rewards.extend(float(r["mean_reward"]) for r in rows)
        ev = d.get("eval")
        if ev:
            eval_rounds.append(ridx)
            eval_R.append(float(ev.get("avg_return", 0.0)))
            eval_succ.append(float(ev.get("pct_success", 0.0)))
    return {
        "train_rewards": train_rewards,
        "round_boundaries": round_boundaries,
        "eval_rounds": eval_rounds,
        "eval_R": eval_R,
        "eval_succ": eval_succ,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flat-glob", required=True)
    ap.add_argument("--turnrd-glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ma-window", type=int, default=5)
    args = ap.parse_args()

    flat = _load(args.flat_glob)
    tr = _load(args.turnrd_glob)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)

    # --- Panel 1: per-episode training reward (concatenated across rounds)
    ax = axes[0]
    if flat["train_rewards"]:
        xs = list(range(len(flat["train_rewards"])))
        ax.plot(
            xs,
            _moving_average(flat["train_rewards"], args.ma_window),
            color="C0", label=f"flat_grpo (n={len(xs)} eps)", linewidth=1.6,
        )
    if tr["train_rewards"]:
        xs = list(range(len(tr["train_rewards"])))
        ax.plot(
            xs,
            _moving_average(tr["train_rewards"], args.ma_window),
            color="C3", label=f"turnRD (n={len(xs)} eps)", linewidth=1.6,
        )
    # Mark round boundaries (use flat's, since both should align)
    for b in (flat["round_boundaries"] or [])[1:]:
        ax.axvline(b, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
    ax.set_xlabel("episode (rounds concatenated)")
    ax.set_ylabel("train mean_reward (MA-5)")
    ax.set_title("Training reward across protocol rounds (seed 11, K=4)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # --- Panel 2: held-out eval avg_return per round
    ax = axes[1]
    if flat["eval_rounds"]:
        ax.plot(flat["eval_rounds"], flat["eval_R"], "o-", color="C0",
                label="flat_grpo", linewidth=1.8, markersize=8)
    if tr["eval_rounds"]:
        ax.plot(tr["eval_rounds"], tr["eval_R"], "s-", color="C3",
                label="turnRD", linewidth=1.8, markersize=8)
    ax.set_xlabel("protocol round")
    ax.set_ylabel("eval avg_return")
    ax.set_title("Held-out eval (greedy K=1, tasks [6500, 6550))")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    all_rounds = sorted(set(flat["eval_rounds"]) | set(tr["eval_rounds"]))
    if all_rounds:
        ax.set_xticks(all_rounds)

    # --- Panel 3: held-out eval pct_success per round
    ax = axes[2]
    if flat["eval_rounds"]:
        ax.plot(flat["eval_rounds"], flat["eval_succ"], "o-", color="C0",
                label="flat_grpo", linewidth=1.8, markersize=8)
    if tr["eval_rounds"]:
        ax.plot(tr["eval_rounds"], tr["eval_succ"], "s-", color="C3",
                label="turnRD", linewidth=1.8, markersize=8)
    ax.set_xlabel("protocol round")
    ax.set_ylabel("eval pct_success")
    ax.set_title("Held-out eval — task-completion rate")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    if all_rounds:
        ax.set_xticks(all_rounds)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
