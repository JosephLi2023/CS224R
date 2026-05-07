#!/usr/bin/env python3
"""Side-by-side comparison of flat_grpo vs method_b (turnRD) per-round eval.

Walks two glob patterns of `train_log.json` files (one per method), pulls
each round's `eval` block, and prints + writes a summary table.

Usage:
    python3 scripts/compare_flat_vs_turnrd.py \\
        --flat-glob 'experiments/manifests/_flat_grpo/flat_grpo_compare_seed11_round*_train_log.json' \\
        --turnrd-glob 'experiments/manifests/_baseline_turnrd/method_b_lean_seed11_round*_train_log.json' \\
        --out experiments/manifests/flat_grpo_vs_turnrd
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from pathlib import Path

ROUND_RE = re.compile(r"round(\d+)")


def _round_idx(p: str) -> int:
    m = ROUND_RE.search(Path(p).name)
    return int(m.group(1)) if m else -1


def _load_eval_rows(pattern: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(glob.glob(pattern), key=_round_idx):
        try:
            d = json.load(open(path))
        except Exception as exc:
            print(f"  skip {path}: {exc!r}")
            continue
        ev = d.get("eval")
        if not ev:
            continue
        rows.append({
            "round": _round_idx(path),
            "path": path,
            "avg_return": ev.get("avg_return"),
            "std_return": ev.get("std_return"),
            "pct_success": ev.get("pct_success"),
            "n_ok": ev.get("n_episodes_ok"),
            "n_attempted": ev.get("n_episodes_attempted"),
            "n_turns_avg": ev.get("n_turns_avg"),
        })
    return rows


def _agg(rows: list[dict]) -> dict:
    if not rows:
        return {"n_rounds": 0}
    rets = [r["avg_return"] for r in rows if r["avg_return"] is not None]
    succ = [r["pct_success"] for r in rows if r["pct_success"] is not None]
    return {
        "n_rounds": len(rows),
        "best_R": round(max(rets), 4) if rets else None,
        "last_R": round(rets[-1], 4) if rets else None,
        "mean_R": round(statistics.fmean(rets), 4) if rets else None,
        "best_succ": round(max(succ), 4) if succ else None,
        "last_succ": round(succ[-1], 4) if succ else None,
        "mean_succ": round(statistics.fmean(succ), 4) if succ else None,
    }


def _fmt(v: float | None, fmt: str = ".4f") -> str:
    return "—" if v is None else format(v, fmt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flat-glob", required=True)
    ap.add_argument("--turnrd-glob", required=True)
    ap.add_argument("--out", required=True, help="Output prefix (no extension)")
    args = ap.parse_args()

    flat_rows = _load_eval_rows(args.flat_glob)
    tr_rows = _load_eval_rows(args.turnrd_glob)
    flat_agg = _agg(flat_rows)
    tr_agg = _agg(tr_rows)

    out_json = Path(args.out + ".json")
    out_txt = Path(args.out + ".txt")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(
        {"flat_grpo": {"summary": flat_agg, "rounds": flat_rows},
         "turnrd": {"summary": tr_agg, "rounds": tr_rows}},
        indent=2,
    ))

    lines: list[str] = []
    lines.append("=" * 92)
    lines.append("Per-round held-out eval (greedy, K=1, task_id_base=6500, 50 eps each)")
    lines.append("=" * 92)
    lines.append(
        f"{'round':>5}  {'flat_R':>8} {'flat_succ':>10} {'flat_ok':>9}    "
        f"{'turnrd_R':>9} {'turnrd_succ':>12} {'turnrd_ok':>10}"
    )
    lines.append("-" * 92)
    n_rounds = max(len(flat_rows), len(tr_rows))
    for i in range(n_rounds):
        f = flat_rows[i] if i < len(flat_rows) else None
        t = tr_rows[i] if i < len(tr_rows) else None
        f_ok = f"{f['n_ok']}/{f['n_attempted']}" if f else "—"
        t_ok = f"{t['n_ok']}/{t['n_attempted']}" if t else "—"
        lines.append(
            f"{i:>5}  "
            f"{(_fmt(f['avg_return']) if f else '—'):>8} "
            f"{(_fmt(f['pct_success']) if f else '—'):>10} "
            f"{f_ok:>9}    "
            f"{(_fmt(t['avg_return']) if t else '—'):>9} "
            f"{(_fmt(t['pct_success']) if t else '—'):>12} "
            f"{t_ok:>10}"
        )
    lines.append("-" * 92)
    lines.append(
        f"{'mean':>5}  {_fmt(flat_agg.get('mean_R')):>8} {_fmt(flat_agg.get('mean_succ')):>10} "
        f"{'':>9}    {_fmt(tr_agg.get('mean_R')):>9} {_fmt(tr_agg.get('mean_succ')):>12}"
    )
    lines.append(
        f"{'best':>5}  {_fmt(flat_agg.get('best_R')):>8} {_fmt(flat_agg.get('best_succ')):>10} "
        f"{'':>9}    {_fmt(tr_agg.get('best_R')):>9} {_fmt(tr_agg.get('best_succ')):>12}"
    )
    lines.append(
        f"{'last':>5}  {_fmt(flat_agg.get('last_R')):>8} {_fmt(flat_agg.get('last_succ')):>10} "
        f"{'':>9}    {_fmt(tr_agg.get('last_R')):>9} {_fmt(tr_agg.get('last_succ')):>12}"
    )
    lines.append("=" * 92)
    txt = "\n".join(lines) + "\n"
    out_txt.write_text(txt)
    print(txt)
    print(f"wrote {out_txt}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
