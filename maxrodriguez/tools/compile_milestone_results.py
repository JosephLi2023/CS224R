"""Compile Max milestone SFT/GRPO result summaries into CSV and Markdown.

Usage after the detached Modal jobs finish:

    modal volume get cs224r-hgpo-vol /manifests/maxrodriguez ./maxrodriguez/results/modal_manifests
    python maxrodriguez/tools/compile_milestone_results.py --input-root ./maxrodriguez/results/modal_manifests
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open() as f:
            value = json.load(f)
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _summary_rows(input_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in input_root.rglob("summary.json"):
        summary = _load_json(path)
        if not summary:
            continue
        row = dict(summary)
        row["_summary_path"] = str(path)
        rows.append(row)
    return rows


def _kind(row: dict[str, Any]) -> str:
    if "eval_type" in row:
        return "eval"
    if row.get("checkpoint_type") == "full" and "learning_rate" in row:
        return "sft_train"
    if "run_name" in row or "method" in row or "turn_reward_method" in row:
        return "grpo_train"
    return "other"


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows found._\n"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="maxrodriguez/results/modal_manifests")
    parser.add_argument("--output-dir", default="maxrodriguez/results/compiled")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    rows = _summary_rows(input_root)

    sft_rows = [r for r in rows if _kind(r) == "sft_train"]
    eval_rows = [r for r in rows if _kind(r) == "eval"]
    grpo_rows = [r for r in rows if _kind(r) == "grpo_train"]

    sft_columns = [
        "ckpt_dir",
        "epochs",
        "learning_rate",
        "max_seq_len",
        "micro_batch_size",
        "grad_accum",
        "use_dagger",
        "n_dagger_rows_total",
        "global_steps",
        "manifest_dir",
    ]
    eval_columns = [
        "adapter_path",
        "checkpoint_type",
        "eval_type",
        "split",
        "episodes",
        "n_success",
        "n_eval",
        "pct_success",
        "avg_return",
        "avg_turns",
        "run_dir",
    ]
    grpo_columns = [
        "run_name",
        "method",
        "turn_reward_method",
        "alpha",
        "learning_rate",
        "kl_coeff",
        "eval_seen_success",
        "eval_unseen_success",
        "_summary_path",
    ]

    _write_csv(output_dir / "sft_train_results.csv", sft_rows, sft_columns)
    _write_csv(output_dir / "eval_results.csv", eval_rows, eval_columns)
    _write_csv(output_dir / "grpo_results.csv", grpo_rows, grpo_columns)

    report = [
        "# Max Rodriguez Milestone Results",
        "",
        "## SFT Training Results",
        _markdown_table(sft_rows, sft_columns),
        "## Evaluation Results",
        _markdown_table(eval_rows, eval_columns),
        "## GRPO Results",
        _markdown_table(grpo_rows, grpo_columns),
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "milestone_results.md").write_text("\n".join(report))
    print(f"Wrote compiled results to {output_dir}")


if __name__ == "__main__":
    main()
