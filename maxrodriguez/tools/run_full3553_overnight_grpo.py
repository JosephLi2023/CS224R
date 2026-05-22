from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal  # type: ignore[import-not-found]


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("CS224R_SKIP_OPENAI_SECRET", "1")

from infra.app_train_loop import app as train_loop_app
from maxrodriguez.grpo.app_alfworld_grpo import build_config_for_run, build_manual_run_spec
from maxrodriguez.grpo.app_signed_attention_transformer import (
    app as signed_attention_app,
    train_signed_attention_transformer_model,
)


app = modal.App("maxrodriguez-full3553-overnight-grpo")
app.include(signed_attention_app)
app.include(train_loop_app)

VOLUME_NAME = "cs224r-hgpo-vol"
VOLUME_CONFIG_ROOT = "/maxrodriguez/generated_grpo"
DEFAULT_SFT = "/vol/checkpoints/maxrodriguez_overnight/full_lr1e5_e10_v2"


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _local_state_root(run_tag: str) -> Path:
    return Path(os.environ["LOCALAPPDATA"]) / "CodexModalMilestone" / run_tag


def _run(
    cmd: list[str],
    *,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "CS224R_SKIP_OPENAI_SECRET": "1",
    }
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if stdout_path is None or stderr_path is None:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            creationflags=creationflags,
            check=False,
        )
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
        return subprocess.run(
            cmd,
            stdout=stdout_f,
            stderr=stderr_f,
            text=True,
            encoding="utf-8",
            env=env,
            creationflags=creationflags,
            check=False,
        )


def _upload_config(local_config_path: Path) -> str:
    remote_path = f"{VOLUME_CONFIG_ROOT}/{local_config_path.name}"
    result = _run(
        [
            "modal",
            "volume",
            "put",
            "--force",
            VOLUME_NAME,
            str(local_config_path),
            remote_path,
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    return f"/vol{remote_path}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
        f.write("\n")


def _launch_detached_grpo(
    *,
    spec,
    state_root: Path,
    eval_episodes: int,
    eval_task_id_base: int,
) -> dict[str, Any]:
    config_dir = state_root / "generated_grpo"
    submit_dir = state_root / "submit_logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    submit_dir.mkdir(parents=True, exist_ok=True)

    local_config_path = config_dir / f"{spec.run_name}.json"
    _write_json(local_config_path, build_config_for_run(spec))
    container_config_path = _upload_config(local_config_path)

    modal_args = [
        "modal",
        "run",
        "--detach",
        "infra/app_train_loop.py::train_loop_alfworld",
        "--n-episodes",
        str(spec.n_episodes),
        "--k",
        str(spec.k),
        "--max-turns",
        str(spec.max_turns),
        "--run-name",
        spec.run_name,
        "--sft-adapter",
        spec.sft_adapter,
        "--use-sft-as-ref",
        "--kl-warmup-episodes",
        str(spec.kl_warmup_episodes),
        "--gpu-mem-util",
        "0.20",
        "--config",
        container_config_path,
        "--eval-episodes",
        str(eval_episodes),
        "--eval-task-id-base",
        str(eval_task_id_base),
        "--train-task-id-stride",
        str(spec.task_id_stride),
        "--save-adapter-out",
        spec.save_adapter_out,
    ]
    stdout_path = submit_dir / f"{spec.run_name}.out.log"
    stderr_path = submit_dir / f"{spec.run_name}.err.log"
    result = _run(modal_args, stdout_path=stdout_path, stderr_path=stderr_path)
    row = {
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "run_name": spec.run_name,
        "return_code": int(result.returncode),
        "method": spec.turn_reward_method,
        "alpha": spec.alpha,
        "learning_rate": spec.learning_rate,
        "kl_coeff": spec.kl_coeff,
        "n_episodes": spec.n_episodes,
        "k": spec.k,
        "task_id_stride": spec.task_id_stride,
        "save_adapter_out": spec.save_adapter_out,
        "config_path": str(local_config_path),
        "container_config_path": container_config_path,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "command": " ".join(modal_args),
    }
    return row


@app.local_entrypoint()
def main(
    run_tag: str = "unlimited_full3553",
    sft_adapter: str = DEFAULT_SFT,
    signed_attention_train_trajectories: int = 3553,
    signed_attention_val_trajectories: int = 140,
    signed_attention_epochs: int = 8,
    signed_attention_learning_rate: float = 5e-5,
    hidden_size: int = 512,
    n_layers: int = 6,
    n_heads: int = 8,
    dropout: float = 0.05,
    grpo_episodes: int = 3553,
    grpo_k: int = 8,
    grpo_max_turns: int = 30,
    grpo_task_id_stride: int = 37,
    eval_episodes: int = 0,
    eval_task_id_base: int = 6500,
) -> None:
    started = time.time()
    run_tag = f"{run_tag}_{_stamp()}"
    state_root = _local_state_root(run_tag)
    output_root = f"/vol/checkpoints/maxrodriguez_overnight/full3553_grpo/{run_tag}"
    sat_output_dir = f"{output_root}/signed_attention_alltrain_h{hidden_size}_l{n_layers}"
    state_root.mkdir(parents=True, exist_ok=True)

    print(f">>> training signed-attention transformer: run_tag={run_tag}")
    sat_summary = train_signed_attention_transformer_model.remote(
        epochs=int(signed_attention_epochs),
        learning_rate=float(signed_attention_learning_rate),
        hidden_size=int(hidden_size),
        n_layers=int(n_layers),
        n_heads=int(n_heads),
        dropout=float(dropout),
        train_trajectories=int(signed_attention_train_trajectories),
        val_trajectories=int(signed_attention_val_trajectories),
        max_turns=int(grpo_max_turns),
        seed=42,
        run_name=f"sat_alltrain_h{hidden_size}_l{n_layers}_{run_tag}",
        output_dir=sat_output_dir,
        base_model_path="Qwen/Qwen2.5-1.5B-Instruct",
        validation_every_epochs=1,
    )
    sat_ckpt = str(sat_summary["checkpoint_path"])
    print(f">>> signed-attention checkpoint ready: {sat_ckpt}")

    choices = [
        ("trajectory_only", 0.0, 1e-6, 0.02),
        ("progress_delta", 0.1, 1e-6, 0.05),
        ("signed_attention", 0.1, 1e-6, 0.02),
        ("admissible_margin", 0.1, 1e-6, 0.05),
    ]
    launch_rows: list[dict[str, Any]] = []
    for method, alpha, lr, kl in choices:
        spec = build_manual_run_spec(
            sft_adapter=sft_adapter,
            alpha=float(alpha),
            turn_reward_method=method,
            learning_rate=float(lr),
            kl_coeff=float(kl),
            n_episodes=int(grpo_episodes),
            k=int(grpo_k),
            max_turns=int(grpo_max_turns),
            clip_eps=0.2,
            grad_accum_steps=1,
            max_tokens_per_microbatch=2048,
            kl_warmup_episodes=25,
            dataset_size_mode="full",
            eval_episodes=int(eval_episodes),
            run_name_suffix=f"{run_tag}_full3553_k{grpo_k}_a100",
            signed_attention_transformer_ckpt=sat_ckpt,
            task_id_stride=int(grpo_task_id_stride),
        )
        adapter_out = f"{output_root}/grpo/{spec.run_name}"
        spec = replace(spec, save_adapter_out=adapter_out)
        print(f">>> launching detached GRPO: {spec.run_name}")
        launch_rows.append(
            _launch_detached_grpo(
                spec=spec,
                state_root=state_root,
                eval_episodes=int(eval_episodes),
                eval_task_id_base=int(eval_task_id_base),
            )
        )

    summary = {
        "run_tag": run_tag,
        "elapsed_s": round(time.time() - started, 2),
        "gpu": "A100-80GB",
        "sft_adapter": sft_adapter,
        "signed_attention_summary": sat_summary,
        "grpo_launches": launch_rows,
        "state_root": str(state_root),
        "output_root": output_root,
        "note": "GRPO jobs are detached Modal runs; final seen/unseen evaluate_freeform_greedy should be run after adapters are written.",
    }
    local_summary = REPO_ROOT / "maxrodriguez" / "results" / f"full3553_overnight_launch_{run_tag}.json"
    appdata_summary = state_root / "summary.json"
    _write_json(local_summary, summary)
    _write_json(appdata_summary, summary)
    print(json.dumps(summary, indent=2, default=str))
