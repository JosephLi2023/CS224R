"""Plot the per-episode reward curve from a train_log.json artifact.

Usage:
    python scripts/plot_reward_curve.py /path/to/train_log.json [--out plot.png]

Reads train_log.json (the format written by infra/app_train_loop.py) and
emits a 2-panel figure: top = per-episode mean R with ±std band and a
moving average; bottom = per-episode KL coef + grad_norm + observed_kl
(stacked y-axes).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _moving_average(xs: list[float], window: int) -> list[float]:
    if window <= 1:
        return xs
    out = []
    for i in range(len(xs)):
        lo = max(0, i - window + 1)
        chunk = xs[lo: i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("train_log_json", type=str)
    ap.add_argument("--out", type=str, default="reward_curve.png")
    ap.add_argument("--ma-window", type=int, default=5,
                    help="Moving-average window for the smoothed R curve.")
    args = ap.parse_args()

    try:
        import matplotlib  # type: ignore[import-not-found]
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError:
        print("matplotlib not installed in this env; "
              "pip install matplotlib", file=sys.stderr)
        return 1

    p = Path(args.train_log_json)
    data = json.loads(p.read_text())
    rows = [r for r in data["rows"] if "mean_reward" in r]
    if not rows:
        print("No successful episodes in log.", file=sys.stderr)
        return 1
    eps = [r["episode"] for r in rows]
    R_mean = [r["mean_reward"] for r in rows]
    R_std = [r["std_reward"] for r in rows]
    kl_coef = [r["kl_coef"] for r in rows]
    obs_kl = [r["observed_kl"] for r in rows]
    gn = [r["grad_norm"] for r in rows]
    R_smooth = _moving_average(R_mean, args.ma_window)

    fig, (ax_r, ax_t) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    # --- Top: reward ---
    R_lo = [m - s for m, s in zip(R_mean, R_std)]
    R_hi = [m + s for m, s in zip(R_mean, R_std)]
    ax_r.fill_between(eps, R_lo, R_hi, alpha=0.15, color="C0", label="±1σ over K=4 trajectories")
    ax_r.plot(eps, R_mean, color="C0", marker="o", linewidth=1.0, markersize=4, alpha=0.5,
              label="per-episode mean R")
    ax_r.plot(eps, R_smooth, color="C0", linewidth=2.0,
              label=f"MA({args.ma_window}) over R")
    ax_r.axhline(0.0, color="gray", linewidth=0.5)
    ax_r.set_ylabel("Reward (final-step)")
    ax_r.set_title(
        f"Flat-GRPO from SFT init: {len(rows)}/{len(data['rows'])} episodes — "
        f"mean R = {sum(R_mean)/len(R_mean):.3f}"
    )
    ax_r.legend(loc="upper left", fontsize=9)
    ax_r.grid(alpha=0.3)

    # --- Bottom: training-signal health ---
    ax_t.plot(eps, kl_coef, color="C1", marker=".", label="kl_coef")
    ax_t.plot(eps, obs_kl, color="C2", marker=".", label="observed_kl")
    ax_t.set_ylabel("KL", color="C1")
    ax_t.set_yscale("log")
    ax_t.tick_params(axis="y", labelcolor="C1")
    ax_t.grid(alpha=0.3)
    ax_t.set_xlabel("Episode")
    ax_t.legend(loc="upper left", fontsize=9)

    ax_g = ax_t.twinx()
    ax_g.plot(eps, gn, color="C3", marker="x", linestyle="--", linewidth=0.8, label="grad_norm")
    ax_g.set_ylabel("grad_norm", color="C3")
    ax_g.tick_params(axis="y", labelcolor="C3")
    ax_g.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"Wrote {args.out}")
    print(f"  n_successful_episodes = {len(rows)}")
    print(f"  mean R                = {sum(R_mean)/len(R_mean):.4f}")
    print(f"  R range               = [{min(R_mean):.3f}, {max(R_mean):.3f}]")
    print(f"  final 10% mean R      = {sum(R_mean[-max(1, len(R_mean)//10):]) / max(1, len(R_mean)//10):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
