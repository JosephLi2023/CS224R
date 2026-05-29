#!/usr/bin/env python3
"""Plot per-round WebShop held-out eval success (100 eps, task IDs [6500, 6600)).

Reads ``reports/data/webshop_per_round_eval.json`` (from Modal manifest
``summary.json`` files). Re-fetch data with::

    python scripts/fetch_webshop_per_round_eval.py

Outputs:
  - reports/figures/webshop_per_round_success.png  (all methods)
  - reports/figures/webshop_per_round_success_grid.png  (2×2 panels)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "reports/data/webshop_per_round_eval.json"
FIG_DIR = REPO_ROOT / "reports/figures"

# Display order and colors (colorblind-friendly).
METHOD_STYLE: list[tuple[str, str]] = [
    ("Progress (HGPO-C)", "#2ca02c"),
    ("LLM Judge", "#9467bd"),
    ("Attention (TurnRD)", "#1f77b4"),
    ("flatGRPO", "#ff7f0e"),
]


def load_data(path: Path) -> dict[str, list[dict]]:
    with path.open() as f:
        raw = json.load(f)
    return {k: sorted(v, key=lambda r: r["round"]) for k, v in raw.items()}


def plot_combined(data: dict[str, list[dict]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    rounds = list(range(10))

    for label, color in METHOD_STYLE:
        if label not in data:
            continue
        series = data[label]
        y = [r["pct_success"] * 100 for r in series]
        x = [r["round"] for r in series]
        ax.plot(x, y, marker="o", linewidth=2, markersize=7, label=label, color=color)

    ax.set_xlabel("Training round", fontsize=12)
    ax.set_ylabel("Eval success (%)", fontsize=12)
    ax.set_title(
        "WebShop Phase 6: per-round held-out eval\n"
        "(100 episodes, greedy, tasks [6500, 6600))",
        fontsize=13,
    )
    ax.set_xticks(rounds)
    ax.set_ylim(40, 95)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_grid(data: dict[str, list[dict]], out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    rounds = list(range(10))

    for ax, (label, color) in zip(axes.flat, METHOD_STYLE):
        if label not in data:
            ax.set_visible(False)
            continue
        series = data[label]
        y = [r["pct_success"] * 100 for r in series]
        x = [r["round"] for r in series]
        ax.plot(x, y, marker="o", linewidth=2, markersize=6, color=color)
        ax.set_title(label, fontsize=11)
        ax.set_xticks(rounds)
        ax.grid(True, alpha=0.3)
        for xi, yi in zip(x, y):
            ax.annotate(
                f"{yi:.0f}",
                (xi, yi),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
            )

    fig.supxlabel("Training round", fontsize=12)
    fig.supylabel("Eval success (%)", fontsize=12)
    fig.suptitle(
        "WebShop per-round eval (100 eps, [6500, 6600), greedy)",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="JSON from fetch_webshop_per_round_eval.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=FIG_DIR,
        help="Directory for PNG outputs",
    )
    args = parser.parse_args()

    data = load_data(args.data)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    combined = args.out_dir / "webshop_per_round_success.png"
    grid = args.out_dir / "webshop_per_round_success_grid.png"
    plot_combined(data, combined)
    plot_grid(data, grid)
    print(f"Wrote {combined}")
    print(f"Wrote {grid}")


if __name__ == "__main__":
    main()
