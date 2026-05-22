from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The milestone GRPO sweep uses the local ALFWorld reward variants and does not
# call the OpenAI judge path, so we opt out of attaching the Modal secret.
os.environ.setdefault("CS224R_SKIP_OPENAI_SECRET", "1")

from maxrodriguez.grpo.app_alfworld_grpo import build_config_for_run, iter_grid_run_specs


VOLUME_NAME = "cs224r-hgpo-vol"
VOLUME_CONFIG_ROOT = "/maxrodriguez/generated_grpo"


def _write_ledger(local_ledger: Path, row: dict[str, object]) -> None:
    local_ledger.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(row, separators=(",", ":"))
    with local_ledger.open("a", encoding="utf-8") as f:
        f.write(payload + "\n")


def _already_submitted(run_name: str, submit_dir: Path) -> bool:
    return (submit_dir / f"{run_name}.out.log").exists()


def _launch_one_spec(
    tagged_spec,
    *,
    args,
    local_config_dir: Path,
    submit_dir: Path,
    local_ledger: Path,
    ledger_lock: threading.Lock,
) -> tuple[str, int]:
    local_config_path = local_config_dir / f"{tagged_spec.run_name}.json"
    with local_config_path.open("w", encoding="utf-8") as f:
        json.dump(build_config_for_run(tagged_spec), f, indent=2)
        f.write("\n")
    remote_config_path = f"{VOLUME_CONFIG_ROOT}/{tagged_spec.run_name}.json"
    container_config_path = f"/vol{remote_config_path}"

    put_result = subprocess.run(
        [
            "modal",
            "volume",
            "put",
            "--force",
            VOLUME_NAME,
            str(local_config_path),
            remote_config_path,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        creationflags=subprocess.CREATE_NO_WINDOW,
        env={
            **os.environ,
            "PYTHONIOENCODING": "utf-8",
            "KMP_DUPLICATE_LIB_OK": "TRUE",
            "CS224R_SKIP_OPENAI_SECRET": "1",
        },
        check=False,
    )
    if put_result.returncode != 0:
        raise RuntimeError(
            "modal volume put failed for "
            f"{tagged_spec.run_name}: {put_result.stderr or put_result.stdout}"
        )

    modal_args = [
        "run",
        "--detach",
        "infra/app_train_loop.py::train_loop_alfworld",
        "--n-episodes",
        str(tagged_spec.n_episodes),
        "--k",
        str(tagged_spec.k),
        "--max-turns",
        str(tagged_spec.max_turns),
        "--run-name",
        tagged_spec.run_name,
        "--sft-adapter",
        tagged_spec.sft_adapter,
        "--use-sft-as-ref",
        "--kl-warmup-episodes",
        str(tagged_spec.kl_warmup_episodes),
        "--gpu-mem-util",
        "0.20",
        "--config",
        container_config_path,
        "--eval-episodes",
        str(args.eval_episodes),
        "--eval-task-id-base",
        str(args.eval_task_id_base),
        "--train-task-id-stride",
        str(tagged_spec.task_id_stride),
        "--save-adapter-out",
        tagged_spec.save_adapter_out,
    ]

    stdout_path = submit_dir / f"{tagged_spec.run_name}.out.log"
    stderr_path = submit_dir / f"{tagged_spec.run_name}.err.log"
    with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
        result = subprocess.run(
            ["modal", *modal_args],
            stdout=stdout_f,
            stderr=stderr_f,
            creationflags=subprocess.CREATE_NO_WINDOW,
            env={
                **os.environ,
                "PYTHONIOENCODING": "utf-8",
                "KMP_DUPLICATE_LIB_OK": "TRUE",
                "CS224R_SKIP_OPENAI_SECRET": "1",
            },
            check=False,
        )

    row = {
        "launched_at": subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", "(Get-Date).ToUniversalTime().ToString('o')"],
            text=True,
            encoding="utf-8",
        ).strip(),
        "run_name": tagged_spec.run_name,
        "kind": "grpo-grid",
        "return_code": int(result.returncode),
        "command": "modal " + " ".join(modal_args),
    }
    with ledger_lock:
        _write_ledger(local_ledger, row)
    return tagged_spec.run_name, int(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the Max GRPO sweep directly via Modal function submits.")
    parser.add_argument("--sft-adapter", required=True)
    parser.add_argument("--run-tag", default="milestone")
    parser.add_argument("--max-specs", type=int, default=0)
    parser.add_argument("--eval-task-id-base", type=int, default=6500)
    parser.add_argument("--eval-episodes", type=int, default=0)
    parser.add_argument("--include-methods", default="")
    parser.add_argument("--signed-attention-transformer-ckpt", default="")
    parser.add_argument("--submit-concurrency", type=int, default=6)
    args = parser.parse_args()

    state_root = Path(os.environ["LOCALAPPDATA"]) / "CodexModalMilestone" / args.run_tag
    submit_dir = state_root / "submit_logs"
    local_config_dir = state_root / "generated_grpo"
    local_ledger = state_root / "milestone_launch_ledger.jsonl"
    submit_dir.mkdir(parents=True, exist_ok=True)
    local_config_dir.mkdir(parents=True, exist_ok=True)

    max_specs = args.max_specs if args.max_specs > 0 else None
    include_methods = [m.strip() for m in args.include_methods.split(",") if m.strip()] or None
    if include_methods is not None and "signed_attention" in include_methods and not args.signed_attention_transformer_ckpt:
        raise ValueError(
            "--signed-attention-transformer-ckpt is required when --include-methods includes signed_attention"
        )

    specs = iter_grid_run_specs(
        sft_adapter=args.sft_adapter,
        max_specs=max_specs,
        allowed_methods=include_methods,
        signed_attention_transformer_ckpt=args.signed_attention_transformer_ckpt,
    )

    tagged_specs = []
    for spec in specs:
        tagged_name = f"{spec.run_name}_{args.run_tag}"
        save_adapter_out = f"/vol/checkpoints/maxrodriguez_milestone/grpo_grid/{tagged_name}"
        if _already_submitted(tagged_name, submit_dir):
            print(f"skip {tagged_name} (already submitted)")
            continue
        tagged_specs.append(replace(spec, run_name=tagged_name, save_adapter_out=save_adapter_out))

    ledger_lock = threading.Lock()
    max_workers = max(1, int(args.submit_concurrency))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _launch_one_spec,
                tagged_spec,
                args=args,
                local_config_dir=local_config_dir,
                submit_dir=submit_dir,
                local_ledger=local_ledger,
                ledger_lock=ledger_lock,
            )
            for tagged_spec in tagged_specs
        ]
        for future in as_completed(futures):
            run_name, rc = future.result()
            print(f"launched {run_name} rc={rc}")

    print(f"submitted {len(tagged_specs)} new grpo runs for tag={args.run_tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
