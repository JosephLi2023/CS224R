#!/usr/bin/env python3
"""Merge per-round Method-B (TurnRD) `train_log.json` artifacts into a
single contiguous reward curve.

Methods A and C produce one `train_log.json` per protocol run (200
episodes). Method B is dispatched through `scripts/run_turnrd_modal.py`
which produces one round dir per round (default: 5 rounds × 40 episodes
each, written to
`experiments/manifests/<run_name_prefix>_seed{N}_round{RR}_<ts>/train_log.json`).

This aggregator concatenates the per-round logs into a single
plotter-compatible artifact:
    {
      "rows": [...all episodes, with global cumulative `episode`...],
      "config": {... merged config + per-round provenance ...}
    }

The resulting file can be fed straight into `scripts/plot_reward_curve.py`,
`scripts/run_modal_eval.sh`-style aggregation, or any downstream tool
that expects the per-run shape.

Per-row enrichment:
- `episode` is rewritten to the GLOBAL cumulative episode count across
  rounds (0..199 for a default 5×40 run). The plotter uses this as
  the x-axis.
- `local_episode` preserves the original within-round episode number
  (0..(episodes_per_round - 1)) for round-level diagnostics.
- `round_idx` is added (0..rounds-1) so post-hoc analysis can split
  the curve by round.

Usage:
    # Default: merge all rounds for seed=11 into a single file.
    scripts/merge_turnrd_round_logs.py --seed 11 \
        --out experiments/manifests/method_hgpo_turnrd_seed11_merged/train_log.json

    # Merge a custom prefix (e.g. ad-hoc orchestrator runs):
    scripts/merge_turnrd_round_logs.py \
        --manifests-dir experiments/manifests \
        --prefix method_b_orchestrated_seed11 \
        --out merged.json

    # Print to stdout (useful for piping into jq):
    scripts/merge_turnrd_round_logs.py --seed 11 --out -
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


# Match `<prefix>_round<NN>_<timestamp>` where <NN> is a 2-digit round
# index. The orchestrator writes `..._round00_<ts>`, `..._round01_<ts>`,
# etc.
_ROUND_DIR_RE = re.compile(r".*_round(\d+)_[0-9_]+$")


@dataclass(frozen=True)
class RoundDir:
    """One per-round run directory + the parsed round index."""

    path: Path
    round_idx: int


def _find_round_dirs(manifests_dir: Path, prefix: str) -> list[RoundDir]:
    """Find all `<manifests_dir>/<prefix>_round??_<ts>/` dirs.

    Returns rounds sorted by `round_idx` so the merge produces
    chronologically-ordered episodes.
    """
    if not manifests_dir.is_dir():
        raise FileNotFoundError(
            f"manifests dir does not exist: {manifests_dir}"
        )
    out: list[RoundDir] = []
    for child in manifests_dir.iterdir():
        if not child.is_dir():
            continue
        if not child.name.startswith(f"{prefix}_round"):
            continue
        m = _ROUND_DIR_RE.match(child.name)
        if m is None:
            continue
        out.append(RoundDir(path=child, round_idx=int(m.group(1))))
    out.sort(key=lambda rd: rd.round_idx)
    return out


def _load_round_log(round_dir: RoundDir) -> dict[str, Any]:
    """Load `train_log.json` from a round dir; raises if missing."""
    log_path = round_dir.path / "train_log.json"
    if not log_path.is_file():
        raise FileNotFoundError(
            f"round {round_dir.round_idx} ({round_dir.path}) is missing "
            f"train_log.json — did the parent train_loop crash?"
        )
    with open(log_path) as fh:
        return json.load(fh)


def merge_rounds(round_dirs: Sequence[RoundDir]) -> dict[str, Any]:
    """Concatenate per-round logs into a single `{rows, config}` dict.

    - Episodes are renumbered into a global cumulative count.
    - `local_episode` preserves the original per-round number.
    - `round_idx` is stamped on every row.
    - The output `config` keeps the FIRST round's config as the base
      and adds `merged_rounds` (count) + `per_round_run_names` (list).
    """
    if not round_dirs:
        raise ValueError(
            "merge_rounds: empty round list — no train_log.json files found "
            "matching the prefix. Check --manifests-dir and --prefix."
        )

    # Detect duplicate round indices (would corrupt the global episode
    # numbering). The orchestrator should never produce these, but if a
    # caller manually re-ran a round it could happen.
    seen: set[int] = set()
    for rd in round_dirs:
        if rd.round_idx in seen:
            raise ValueError(
                f"duplicate round_idx={rd.round_idx} in {[r.path.name for r in round_dirs]}; "
                "remove or rename the older round dir before merging."
            )
        seen.add(rd.round_idx)

    merged_rows: list[dict[str, Any]] = []
    per_round_run_names: list[str] = []
    base_config: dict[str, Any] | None = None
    global_episode = 0

    for rd in round_dirs:
        log = _load_round_log(rd)
        rows = log.get("rows", []) or []
        cfg = log.get("config", {}) or {}
        per_round_run_names.append(str(cfg.get("run_name") or rd.path.name))

        if base_config is None:
            base_config = dict(cfg)

        # Skip "errored" rows that infra/app_train_loop.py emits when
        # an episode crashed (those have an "error" key but no
        # "mean_reward"). They still consume a global_episode slot so
        # downstream tooling sees the gap.
        for row in rows:
            new_row = dict(row)
            new_row["round_idx"] = rd.round_idx
            new_row["local_episode"] = int(row.get("episode", 0))
            new_row["episode"] = global_episode
            merged_rows.append(new_row)
            global_episode += 1

    assert base_config is not None  # narrowed by the empty-list guard above

    merged_config = dict(base_config)
    merged_config["merged_rounds"] = len(round_dirs)
    merged_config["per_round_run_names"] = per_round_run_names
    merged_config["total_episodes"] = len(merged_rows)
    # Drop per-round-specific knobs that don't apply to the merged log.
    merged_config.pop("task_id_offset", None)
    merged_config.pop("run_name", None)

    return {"rows": merged_rows, "config": merged_config}


def _write_output(merged: dict[str, Any], out_path: str) -> None:
    """Write merged dict to `out_path` (or stdout when out_path == '-')."""
    if out_path == "-":
        json.dump(merged, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(merged, fh, indent=2)
    print(
        f"Wrote {len(merged['rows'])} merged episodes from "
        f"{merged['config']['merged_rounds']} rounds to {out}",
        file=sys.stderr,
    )


def _build_prefix(args: argparse.Namespace) -> str:
    """Resolve --prefix from --seed (when --prefix not explicit)."""
    if args.prefix:
        return args.prefix
    if args.seed is not None:
        return f"{args.run_name_prefix}_seed{args.seed}"
    raise SystemExit(
        "ERROR: provide either --seed N (uses default run-name-prefix "
        f"'{args.run_name_prefix}') or --prefix <full-prefix>."
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=Path("experiments/manifests"),
        help="Root dir holding per-round run directories (default: experiments/manifests).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Protocol seed. When set, builds the prefix as "
             "'{run_name_prefix}_seed{seed}'. Mutually exclusive with --prefix.",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Explicit run-name prefix (e.g. 'method_b_orchestrated_seed11'). "
             "Overrides --seed-derived prefix.",
    )
    parser.add_argument(
        "--run-name-prefix",
        default="method_b_orchestrated",
        help="Base prefix matching scripts/run_turnrd_modal.py's default "
             "(default: method_b_orchestrated). Combined with --seed.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output path for the merged train_log.json. Use '-' for stdout.",
    )
    args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])

    prefix = _build_prefix(args)
    round_dirs = _find_round_dirs(args.manifests_dir, prefix)
    if not round_dirs:
        raise SystemExit(
            f"ERROR: no round dirs found matching '{prefix}_round??_*' under "
            f"{args.manifests_dir}. Did the orchestrator run?"
        )
    merged = merge_rounds(round_dirs)
    _write_output(merged, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
