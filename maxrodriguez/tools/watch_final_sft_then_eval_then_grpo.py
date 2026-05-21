from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        encoding="utf-8",
        errors="ignore",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "KMP_DUPLICATE_LIB_OK": "TRUE"},
    )


def _volume_ls(path: str) -> list[str]:
    cp = _run(["modal", "volume", "ls", "cs224r-hgpo-vol", path], check=False)
    if cp.returncode != 0:
        return []
    return [line.strip() for line in cp.stdout.splitlines() if line.strip()]


def _volume_get_json(src_path: str) -> dict | None:
    tmp = Path(os.environ["TEMP"]) / ("codex_" + str(abs(hash(src_path))) + ".json")
    cp = _run(["modal", "volume", "get", "cs224r-hgpo-vol", src_path, str(tmp)], check=False)
    if cp.returncode != 0 or not tmp.exists():
        return None
    try:
        return json.loads(tmp.read_text(encoding="utf-8"))
    except Exception:
        return None


def _latest_manifest_summary(prefix: str) -> tuple[str, dict] | tuple[None, None]:
    matches = [line for line in _volume_ls("/manifests/maxrodriguez") if prefix in line]
    if not matches:
        return None, None
    latest = sorted(matches)[-1]
    summary = _volume_get_json("/" + latest + "/summary.json")
    if summary is None:
        return None, None
    return latest, summary


def _append_status(log_path: Path, message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Wait for final SFT, run full eval, record results, then launch GRPO grid.")
    parser.add_argument("--run-tag", default="milestone_v8")
    parser.add_argument("--final-sft-checkpoint", required=True)
    parser.add_argument("--best-max-seq-len", type=int, default=1024)
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()

    results_dir = Path("maxrodriguez/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / f"watch_{args.run_tag}.log"
    result_json_path = results_dir / f"{args.run_tag}_final_sft_eval_results.json"

    final_sft_summary_path = args.final_sft_checkpoint.rstrip("/") + "/summary.json"
    _append_status(log_path, f"Watching final SFT summary at {final_sft_summary_path}")

    final_sft_summary = None
    while final_sft_summary is None:
        final_sft_summary = _volume_get_json(final_sft_summary_path)
        if final_sft_summary is None:
            _append_status(log_path, "Final SFT still running; sleeping.")
            time.sleep(args.poll_seconds)

    _append_status(log_path, "Final SFT summary detected.")

    seen_prefix = f"sfteval_{args.run_tag}_seen140_"
    unseen_prefix = f"sfteval_{args.run_tag}_unseen134_"
    seen_name, seen_summary = _latest_manifest_summary(seen_prefix)
    unseen_name, unseen_summary = _latest_manifest_summary(unseen_prefix)

    if seen_summary is None or unseen_summary is None:
        _append_status(log_path, "Launching full final SFT benchmark eval.")
        _run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                "maxrodriguez/scripts/run_milestone_gridsearch.ps1",
                "-Stage",
                "sft-final-eval",
                "-RunTag",
                args.run_tag,
                "-FinalEvalCheckpointPath",
                args.final_sft_checkpoint,
                "-FinalEvalCheckpointType",
                "full",
                "-BestMaxSeqLen",
                str(args.best_max_seq_len),
            ]
        )
    else:
        _append_status(log_path, "Final SFT eval already appears launched/completed; reusing existing summaries.")

    while True:
        seen_name, seen_summary = _latest_manifest_summary(seen_prefix)
        unseen_name, unseen_summary = _latest_manifest_summary(unseen_prefix)
        if seen_summary is not None and unseen_summary is not None:
            break
        _append_status(log_path, "Waiting for full final SFT eval summaries.")
        time.sleep(args.poll_seconds)

    payload = {
        "final_sft_checkpoint": args.final_sft_checkpoint,
        "final_sft_summary": final_sft_summary,
        "seen_manifest": seen_name,
        "seen_summary": seen_summary,
        "unseen_manifest": unseen_name,
        "unseen_summary": unseen_summary,
    }
    result_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _append_status(
        log_path,
        "Recorded final SFT eval results: "
        f"seen_success={seen_summary.get('pct_success')} "
        f"unseen_success={unseen_summary.get('pct_success')}",
    )

    _append_status(log_path, "Launching GRPO grid from final SFT checkpoint.")
    _run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            "maxrodriguez/scripts/run_milestone_gridsearch.ps1",
            "-Stage",
            "grpo-grid",
            "-RunTag",
            args.run_tag,
            "-BestSftCheckpoint",
            args.final_sft_checkpoint,
        ]
    )
    _append_status(log_path, "GRPO grid launched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
