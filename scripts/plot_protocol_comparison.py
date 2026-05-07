#!/usr/bin/env python3
"""Plot a head-to-head Method comparison from one or more train_log.json
artifacts.

Each `--method label=path` argument names a method and points at one of:
- A single `train_log.json` (Methods A/C: one file per protocol run).
- A directory containing multiple round dirs (Method B with the
  `--run-name method_b_orchestrated_seed{N}_round{NN}_<ts>`
  pattern; this script auto-aggregates them via the same logic as
  `merge_turnrd_round_logs.py`).
- A pre-merged train_log.json (output of `merge_turnrd_round_logs.py`).

Top panel: per-episode training reward (5-pt moving average) per method.
Mid panel: held-out eval `avg_return` markers — one dot per round per
method (assumed populated by `app_train_loop.py`'s eval block).
Bottom panel (optional, only when at least one method has TurnRD
diagnostics): cls_query_norm + alpha_var trajectories across episodes.

Usage:
    scripts/plot_protocol_comparison.py \\
        --method 'flat_grpo=/tmp/flat_grpo_sft_v3_train_log.json' \\
        --method 'method_b=/tmp/method_b_seed11_runs' \\
        --out /tmp/comparison_seed11.png

    # With TurnRD diagnostics panel:
    scripts/plot_protocol_comparison.py --turnrd-diagnostics \\
        --method 'flat_grpo=...' --method 'method_b=...' \\
        --out /tmp/comparison.png
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


_ROUND_DIR_RE = re.compile(r".*_round(\d+)_[0-9_]+$")


def _moving_average(xs: list[float], window: int) -> list[float]:
    if window <= 1 or not xs:
        return xs
    out = []
    for i in range(len(xs)):
        lo = max(0, i - window + 1)
        chunk = xs[lo: i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def _load_method(label: str, path_str: str) -> dict[str, Any]:
    """Load a method's data from a single train_log.json OR a dir of round dirs.

    Returns:
        {
          "label": str,
          "rows":  list[dict] (filtered for `mean_reward`),
          "evals": list[(global_episode_idx, avg_return)] for each round,
          "consistency_t": list[float] aligned with rows (0 when unavailable),
          "cls_query_norm": list[float] aligned with rows,
          "alpha_var": list[float] aligned with rows,
        }
    """
    p = Path(path_str)
    if p.is_file():
        # Single train_log.json (either a flat-GRPO log or a pre-merged one).
        with open(p) as fh:
            d = json.load(fh)
        rows = [r for r in d.get("rows", []) if "mean_reward" in r]
        evals: list[tuple[int, float]] = []
        if isinstance(d.get("eval"), dict) and "avg_return" in d["eval"]:
            # Single eval block at the end of one run.
            evals.append((len(rows), float(d["eval"]["avg_return"])))
        return _pack(label, rows, evals)

    if not p.is_dir():
        raise SystemExit(f"--method {label}: path {p} is neither a file nor a dir")

    # Directory: look for round dirs `<prefix>_round??_<ts>/train_log.json`.
    round_paths: list[tuple[int, Path]] = []
    for child in sorted(p.iterdir()):
        if not child.is_dir():
            # Could be a downloaded train_log.json that we routed here.
            if child.suffix == ".json" and child.name.startswith("round_"):
                # Filename pattern: `round_<ts>_train_log.json` (matches
                # the manual-aggregation downloads we did earlier).
                ts_match = re.match(r"round_([0-9_]+?)_train_log\.json", child.name)
                if ts_match:
                    round_paths.append((len(round_paths), child))
            continue
        m = _ROUND_DIR_RE.match(child.name)
        if m is None:
            continue
        log_path = child / "train_log.json"
        if log_path.is_file():
            round_paths.append((int(m.group(1)), log_path))

    # Sort by round_idx.
    round_paths.sort(key=lambda t: t[0])

    if not round_paths:
        raise SystemExit(
            f"--method {label}: no round dirs found under {p}. Expected "
            f"either `<prefix>_round??_<ts>/train_log.json` subdirs or "
            f"`round_<ts>_train_log.json` files (manual-aggregation output)."
        )

    all_rows: list[dict] = []
    evals: list[tuple[int, float]] = []
    global_ep = 0
    for round_idx, log_path in round_paths:
        with open(log_path) as fh:
            d = json.load(fh)
        round_rows = [r for r in d.get("rows", []) if "mean_reward" in r]
        for r in round_rows:
            r2 = dict(r)
            r2["episode"] = global_ep
            r2["round_idx"] = round_idx
            all_rows.append(r2)
            global_ep += 1
        # Collect eval marker IF present.
        if isinstance(d.get("eval"), dict) and "avg_return" in d["eval"]:
            evals.append((global_ep, float(d["eval"]["avg_return"])))
    return _pack(label, all_rows, evals)


def _pack(label: str, rows: list[dict], evals: list[tuple[int, float]]) -> dict[str, Any]:
    return {
        "label": label,
        "rows": rows,
        "evals": evals,
        "consistency_t": [r.get("consistency_t", 0.0) for r in rows],
        "cls_query_norm": [r.get("cls_query_norm", 0.0) for r in rows],
        "alpha_var": [r.get("alpha_var", 0.0) for r in rows],
    }


def _has_turnrd_signal(methods: list[dict]) -> bool:
    """True if at least one method has any non-zero TurnRD diagnostic."""
    for m in methods:
        if any(v > 0 for v in m["cls_query_norm"]):
            return True
        if any(v > 0 for v in m["alpha_var"]):
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--method",
        action="append",
        required=True,
        help="Method spec: 'label=path'. Path can be a single train_log.json "
             "or a dir containing per-round train_logs. Repeat for each method "
             "to compare.",
    )
    ap.add_argument(
        "--out",
        default="protocol_comparison.png",
        help="Output PNG path (default: protocol_comparison.png).",
    )
    ap.add_argument(
        "--ma-window",
        type=int,
        default=5,
        help="Moving-average window for the smoothed reward curve (default: 5).",
    )
    ap.add_argument(
        "--turnrd-diagnostics",
        action="store_true",
        help="Add a third panel showing cls_query_norm + alpha_var (only "
             "useful when at least one method has TurnRD diagnostic data).",
    )
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    try:
        import matplotlib  # type: ignore[import-not-found]
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError as e:
        raise SystemExit(f"matplotlib required for plotting: {e}") from e

    # Parse --method args.
    methods = []
    for spec in args.method:
        if "=" not in spec:
            raise SystemExit(f"--method must be 'label=path', got: {spec}")
        label, path_str = spec.split("=", 1)
        methods.append(_load_method(label.strip(), path_str.strip()))

    # Compute panel count.
    show_turnrd = args.turnrd_diagnostics and _has_turnrd_signal(methods)
    n_panels = 3 if show_turnrd else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 3.0 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    # Top panel: per-episode reward + MA per method.
    ax_R = axes[0]
    for m in methods:
        eps = [r["episode"] for r in m["rows"]]
        R = [r["mean_reward"] for r in m["rows"]]
        if not eps:
            continue
        ax_R.plot(eps, R, alpha=0.25, linewidth=1)
        ax_R.plot(
            eps,
            _moving_average(R, args.ma_window),
            linewidth=2,
            label=f"{m['label']} (n={len(R)})",
        )
    ax_R.set_ylabel("training mean R")
    ax_R.set_title(
        f"Method comparison (training reward, MA-{args.ma_window})"
    )
    ax_R.legend(loc="best")
    ax_R.grid(True, alpha=0.3)

    # Mid panel: held-out eval markers per round.
    ax_E = axes[1]
    has_eval = False
    for m in methods:
        if not m["evals"]:
            continue
        has_eval = True
        xs = [e[0] for e in m["evals"]]
        ys = [e[1] for e in m["evals"]]
        ax_E.plot(
            xs, ys, marker="o", linewidth=2, markersize=10,
            label=f"{m['label']} eval", linestyle="--",
        )
        # Annotate each marker with its value.
        for x, y in zip(xs, ys):
            ax_E.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                          xytext=(0, 8), ha="center", fontsize=8)
    if has_eval:
        ax_E.set_ylabel("held-out eval avg_return")
        ax_E.set_title("End-of-round held-out eval (greedy, K=1)")
        ax_E.legend(loc="best")
    else:
        ax_E.set_title(
            "(no eval data — re-run with --eval-episodes > 0 in the next protocol)"
        )
        ax_E.text(
            0.5, 0.5, "No eval blocks in any train_log.json",
            transform=ax_E.transAxes, ha="center", va="center",
            color="gray", fontsize=12,
        )
    ax_E.grid(True, alpha=0.3)

    # Optional bottom panel: TurnRD diagnostics.
    if show_turnrd:
        ax_T = axes[2]
        ax_T2 = ax_T.twinx()
        for m in methods:
            eps = [r["episode"] for r in m["rows"]]
            cqn = m["cls_query_norm"]
            av = m["alpha_var"]
            if any(v > 0 for v in cqn):
                ax_T.plot(eps, cqn, linewidth=1.5, label=f"{m['label']} cls_query_norm")
            if any(v > 0 for v in av):
                ax_T2.plot(
                    eps, av, linewidth=1.5, linestyle=":",
                    label=f"{m['label']} alpha_var",
                )
        ax_T.set_ylabel("‖cls_query‖₂")
        ax_T2.set_ylabel("alpha variance")
        ax_T.set_title("TurnRD diagnostics: cls_query L2 norm (solid) + alpha variance (dotted)")
        ax_T.legend(loc="upper left", fontsize=8)
        ax_T2.legend(loc="upper right", fontsize=8)
        ax_T.grid(True, alpha=0.3)

    axes[-1].set_xlabel("episode")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote: {args.out}", file=sys.stderr)

    # One-line summary per method.
    print("\nSummary:", file=sys.stderr)
    print(f"  {'method':<20} {'n_train':<8} {'mean_R':<8} {'last_R':<8} {'final_eval':<10}", file=sys.stderr)
    for m in methods:
        n = len(m["rows"])
        if n == 0:
            print(f"  {m['label']:<20} (no data)", file=sys.stderr)
            continue
        R = [r["mean_reward"] for r in m["rows"]]
        mean_R = sum(R) / n
        last_R = sum(R[-max(1, n // 5):]) / max(1, n // 5)
        final_eval = m["evals"][-1][1] if m["evals"] else float("nan")
        print(
            f"  {m['label']:<20} {n:<8} {mean_R:<8.4f} {last_R:<8.4f} {final_eval:<10.4f}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
