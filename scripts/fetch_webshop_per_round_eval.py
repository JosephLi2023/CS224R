#!/usr/bin/env python3
"""Pull per-round eval_pct_success from Modal manifest summary.json files."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "reports/data/webshop_per_round_eval.json"

METHODS: dict[str, str] = {
    "Attention (TurnRD)": "webshop_attention_v1_seed11_round",
    "flatGRPO": "webshop_flatGRPO_v1_seed23_round",
    "LLM Judge": "webshop_LLMJudge_v1_seed31_round",
    "Progress (HGPO-C)": "webshop_Progress_v1_seed41_round",
}


def _latest_manifest(prefix: str, round_idx: int) -> str | None:
    out = subprocess.check_output(
        ["modal", "volume", "ls", "cs224r-hgpo-vol", "/manifests"],
        text=True,
    )
    candidates: list[str] = []
    for line in out.splitlines():
        if f"{prefix}{round_idx:02d}_" not in line:
            continue
        if "cloud" in line or "eval200" in line:
            continue
        m = re.search(r"(manifests/\S+)", line)
        if m:
            candidates.append(m.group(1))
    return sorted(candidates)[-1] if candidates else None


def fetch() -> dict[str, list[dict]]:
    tmp = REPO_ROOT / "reports/data/_summary_tmp.json"
    data: dict[str, list[dict]] = {}

    for label, prefix in METHODS.items():
        rounds: list[dict] = []
        for r in range(10):
            manifest = _latest_manifest(prefix, r)
            if manifest is None:
                print(f"WARN: no manifest for {label} round {r}")
                continue
            if tmp.exists():
                if tmp.is_dir():
                    shutil.rmtree(tmp)
                else:
                    tmp.unlink()
            subprocess.run(
                [
                    "modal",
                    "volume",
                    "get",
                    "cs224r-hgpo-vol",
                    f"{manifest}/summary.json",
                    str(tmp),
                    "--force",
                ],
                check=True,
                capture_output=True,
            )
            summary = json.loads(tmp.read_text())
            rounds.append(
                {
                    "round": r,
                    "pct_success": float(summary["eval_pct_success"]),
                    "avg_R": float(summary.get("eval_avg_return", 0.0)),
                    "manifest": manifest,
                }
            )
        data[label] = rounds
    if tmp.exists():
        tmp.unlink()
    return data


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = fetch()
    OUT_PATH.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Wrote {OUT_PATH}")
    for label, rounds in data.items():
        pcts = ", ".join(f"R{r['round']}:{r['pct_success']*100:.0f}%" for r in rounds)
        print(f"  {label}: {pcts}")


if __name__ == "__main__":
    main()
