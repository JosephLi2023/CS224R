"""F7: FiLM offline mechanism evidence (3-panel summary).

Panel A: γ/β disable → V-head output perturbation magnitude (39.7%)
Panel B: goal_emb shuffle → MSE delta (+3.16%) + % of samples worse-when-shuffled (57.6%)
Panel C: γ-β norm growth across rounds (compact version of F5)

Numbers from sota_R13_goalcondFiLM.md + film_eval_summary (offline test).

Output: reports/poster_paloalto/figures/F7_film_mechanism.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --- Panel A: perturbation magnitude (γ/β disable) ---
PERTURB_LABELS = ["FiLM\nactive", "FiLM\ndisabled (γ=0,β=0)"]
# Median V-head output magnitude (in arbitrary units, normalized to baseline = 1.0)
PERTURB_VALUES = [1.000, 1.0 - 0.397]  # 39.7% decrease when FiLM disabled
PERTURB_LABEL_PCT = 39.7

# --- Panel B: goal-shuffle ---
SHUFFLE_LABELS = ["Original\ngoal_emb", "Shuffled\ngoal_emb"]
# Mean V-head MSE relative to baseline (normalized to baseline = 1.0)
SHUFFLE_MSE = [1.0000, 1.0316]
SHUFFLE_PCT_DELTA = 3.16
SHUFFLE_FRAC_WORSE = 57.6

# --- Panel C: γ/β growth (compact, 3-point) ---
GROWTH_ROUNDS = [0, 4, 9, 12]
GAMMA_NORM = [0.341, 0.468, 0.781, 0.829]
BETA_NORM  = [0.330, 0.399, 0.773, 0.900]


def main(out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel A: perturbation
    ax = axes[0]
    bars = ax.bar(PERTURB_LABELS, PERTURB_VALUES,
                  color=["#8c1515", "#bbb"], edgecolor="black", linewidth=1.0)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("V-head output magnitude\n(relative)", fontsize=11)
    ax.set_title(f"(a) Perturbation: disabling γ/β\ndrops V-head output {PERTURB_LABEL_PCT:.1f}%",
                 fontsize=12, fontweight="bold")
    for bar, val in zip(bars, PERTURB_VALUES):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                f"{val:.2f}", ha="center", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    # Panel B: goal shuffle
    ax = axes[1]
    bars = ax.bar(SHUFFLE_LABELS, SHUFFLE_MSE,
                  color=["#175e54", "#bbb"], edgecolor="black", linewidth=1.0)
    ax.set_ylim(0.95, 1.10)
    ax.set_ylabel("V-head MSE\n(relative to original)", fontsize=11)
    ax.set_title(f"(b) Goal shuffle: +{SHUFFLE_PCT_DELTA:.2f}% MSE\n"
                 f"({SHUFFLE_FRAC_WORSE:.1f}% worse-when-shuffled)",
                 fontsize=12, fontweight="bold")
    for bar, val in zip(bars, SHUFFLE_MSE):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.4f}", ha="center", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    # Panel C: γ/β growth
    ax = axes[2]
    ax.plot(GROWTH_ROUNDS, GAMMA_NORM, "o-", lw=2.5, ms=9,
            label=r"$\|\gamma\|_F$", color="#8c1515")
    ax.plot(GROWTH_ROUNDS, BETA_NORM, "s-", lw=2.5, ms=9,
            label=r"$\|\beta\|_F$", color="#175e54")
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel(r"$\|\cdot\|_F$ (Frobenius)", fontsize=11)
    ax.set_title("(c) γ/β monotone growth\n(zero-init to meaningful)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(GROWTH_ROUNDS)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f">>> Wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        type=Path,
        default=Path("reports/poster_paloalto/figures/F7_film_mechanism.pdf"),
    )
    args = p.parse_args()
    main(args.out)
