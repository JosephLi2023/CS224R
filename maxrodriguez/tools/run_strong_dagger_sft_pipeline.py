from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from maxrodriguez.grpo.app_alfworld_grpo import build_manual_run_spec, launch_grpo_run
from maxrodriguez.grpo.app_signed_attention_transformer import (
    app as signed_attention_app,
    train_signed_attention_transformer_model,
)
from maxrodriguez.supervised_FT.app_alfworld_sft_plus import (
    app as sft_app,
    evaluate_freeform_greedy,
    train_sft_plus,
)


app = modal.App("maxrodriguez-strong-dagger-alfworld")
app.include(sft_app)
app.include(signed_attention_app)
app.include(train_loop_app)


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _success_summary(result: dict[str, Any]) -> dict[str, Any]:
    post = result.get("post_train_eval", {}) or {}
    seen = post.get("seen", {}) or {}
    unseen = post.get("unseen", {}) or {}
    seen_success = int(seen.get("n_success", 0) or 0)
    seen_eval = int(seen.get("n_eval", 0) or 0)
    unseen_success = int(unseen.get("n_success", 0) or 0)
    unseen_eval = int(unseen.get("n_eval", 0) or 0)
    total_success = seen_success + unseen_success
    total_eval = seen_eval + unseen_eval
    return {
        "seen_success": seen_success,
        "seen_eval": seen_eval,
        "seen_rate": seen_success / max(1, seen_eval),
        "unseen_success": unseen_success,
        "unseen_eval": unseen_eval,
        "unseen_rate": unseen_success / max(1, unseen_eval),
        "total_success": total_success,
        "total_eval": total_eval,
        "total_rate": total_success / max(1, total_eval),
    }


def _sft_score(result: dict[str, Any]) -> tuple[float, float]:
    success = _success_summary(result)
    final_row = result.get("final_log_row", {}) or {}
    val = final_row.get("val", {}) or {}
    val_ce = float(val.get("per_token_ce", 999.0) or 999.0)
    return float(success["total_rate"]), -val_ce


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


@app.local_entrypoint()
def main(
    run_tag: str = "strong_dagger",
    data_path: str = "/vol/data/alfworld/sft_trajs_maxrodriguez_runtime500x10_plus_seq2seq_structured.jsonl",
    base_model_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
    sft_epochs: int = 2,
    sft_learning_rate: float = 1e-4,
    sft_max_seq_len: int = 1024,
    sft_grad_accum: int = 8,
    sft_max_examples: int = 20000,
    sft_eval_seen_episodes: int = 30,
    sft_eval_unseen_episodes: int = 30,
    dagger_episodes: int = 160,
    dagger_max_turns: int = 15,
    dagger_max_new_examples: int = 2500,
    dagger_mix_ratio: float = 0.25,
    success_threshold: float = 0.40,
    signed_attention_checkpoint: str = "",
    signed_attention_epochs: int = 2,
    signed_attention_train_trajectories: int = 128,
    signed_attention_val_trajectories: int = 32,
    grpo_episodes: int = 100,
    grpo_k: int = 4,
    grpo_max_turns: int = 30,
    grpo_task_id_stride: int = 37,
    grpo_eval_episodes: int = 30,
    max_parallel_grpo: int = 2,
    grpo_methods: str = "trajectory_only,progress_delta,signed_attention,admissible_margin",
    force_grpo: bool = False,
) -> None:
    """Run stronger LoRA SFT selection, then gated GRPO reruns if SFT clears the bar."""
    started = time.time()
    run_tag = f"{run_tag}_{_stamp()}"
    output_root = f"/vol/checkpoints/maxrodriguez_milestone/strong_dagger/{run_tag}"

    sft_common = {
        "epochs": int(sft_epochs),
        "learning_rate": float(sft_learning_rate),
        "min_reward": 1.0,
        "max_seq_len": int(sft_max_seq_len),
        "micro_batch_size": 1,
        "grad_accum": int(sft_grad_accum),
        "log_every": 50,
        "data_path": data_path,
        "val_fraction": 0.08,
        "max_examples": int(sft_max_examples),
        "sample_after_load": True,
        "full_finetune": False,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "base_model_path": base_model_path,
        "seed": 42,
        "post_eval_seen_episodes": int(sft_eval_seen_episodes),
        "post_eval_unseen_episodes": int(sft_eval_unseen_episodes),
        "post_eval_max_turns": 30,
        "post_eval_max_seq_len": int(sft_max_seq_len),
        "post_eval_task_id_base": 0,
    }

    sft_specs = [
        {
            **sft_common,
            "run_name": f"sftstrong_{run_tag}_nodagger",
            "output_dir": f"{output_root}/sft_nodagger",
            "use_dagger": False,
            "dagger_episodes": 0,
            "dagger_max_turns": int(dagger_max_turns),
            "dagger_max_new_examples": 0,
            "dagger_mix_ratio": 0.0,
            "dagger_start_epoch": 0,
            "dagger_every_n_epochs": 1,
            "dagger_task_id_base": 8000,
            "dagger_split": "train",
        },
        {
            **sft_common,
            "run_name": f"sftstrong_{run_tag}_dagger",
            "output_dir": f"{output_root}/sft_dagger",
            "use_dagger": True,
            "dagger_episodes": int(dagger_episodes),
            "dagger_max_turns": int(dagger_max_turns),
            "dagger_max_new_examples": int(dagger_max_new_examples),
            "dagger_mix_ratio": float(dagger_mix_ratio),
            "dagger_start_epoch": 0,
            "dagger_every_n_epochs": 1,
            "dagger_task_id_base": 8000,
            "dagger_split": "train",
        },
    ]

    sft_results: list[dict[str, Any]] = []
    for spec in sft_specs:
        print(f">>> training SFT candidate: {spec['run_name']}")
        result = train_sft_plus.remote(**spec)
        result["selection_success_summary"] = _success_summary(result)
        sft_results.append(result)
        print(
            ">>> SFT candidate done:",
            spec["run_name"],
            json.dumps(result["selection_success_summary"], sort_keys=True),
        )

    best_sft = max(sft_results, key=_sft_score)
    best_sft_path = str(best_sft["ckpt_dir"])
    best_success = _success_summary(best_sft)
    sft_gate_passed = bool(float(best_success["total_rate"]) >= float(success_threshold))
    print(f">>> selected SFT checkpoint: {best_sft_path}")
    print(f">>> SFT gate passed={sft_gate_passed}: {json.dumps(best_success, sort_keys=True)}")

    grpo_results: list[dict[str, Any]] = []
    sat_summary: dict[str, Any] = {}
    sat_ckpt = signed_attention_checkpoint

    if sft_gate_passed or bool(force_grpo):
        if sat_ckpt:
            sat_summary = {"checkpoint_path": sat_ckpt, "reused": True}
        else:
            sat_output_dir = f"{output_root}/signed_attention"
            sat_summary = train_signed_attention_transformer_model.remote(
                epochs=int(signed_attention_epochs),
                learning_rate=1e-4,
                hidden_size=128,
                n_layers=2,
                n_heads=4,
                dropout=0.0,
                train_trajectories=int(signed_attention_train_trajectories),
                val_trajectories=int(signed_attention_val_trajectories),
                max_turns=int(grpo_max_turns),
                seed=42,
                run_name=f"satstrong_{run_tag}",
                output_dir=sat_output_dir,
                base_model_path=base_model_path,
            )
            sat_ckpt = str(sat_summary["checkpoint_path"])
        print(f">>> signed-attention checkpoint: {sat_ckpt}")

        grpo_choices = [
            ("trajectory_only", 0.0, 1e-6, 0.02),
            ("progress_delta", 0.1, 1e-6, 0.05),
            ("signed_attention", 0.1, 1e-6, 0.02),
            ("admissible_margin", 0.1, 1e-6, 0.05),
        ]
        selected_methods = {
            item.strip() for item in grpo_methods.split(",") if item.strip()
        }
        if selected_methods:
            grpo_choices = [
                choice for choice in grpo_choices if choice[0] in selected_methods
            ]
            missing = selected_methods - {choice[0] for choice in grpo_choices}
            if missing:
                raise ValueError(f"unknown GRPO methods: {sorted(missing)}")

        def run_one_grpo(choice: tuple[str, float, float, float]) -> dict[str, Any]:
            method, alpha, lr, kl = choice
            spec = build_manual_run_spec(
                sft_adapter=best_sft_path,
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
                kl_warmup_episodes=5,
                dataset_size_mode="full",
                eval_episodes=0,
                run_name_suffix=f"{run_tag}_strong{grpo_episodes}",
                signed_attention_transformer_ckpt=sat_ckpt,
                task_id_stride=int(grpo_task_id_stride),
            )
            adapter_out = f"{output_root}/grpo/{spec.run_name}"
            spec = replace(spec, save_adapter_out=adapter_out)
            print(f">>> training GRPO: {spec.run_name}")
            train_summary = launch_grpo_run(spec)
            print(f">>> evaluating GRPO: {spec.run_name}")
            seen_eval = evaluate_freeform_greedy.remote(
                adapter_path=adapter_out,
                checkpoint_type="lora",
                episodes=int(grpo_eval_episodes),
                task_id_base=0,
                run_name=f"{spec.run_name}_seen{grpo_eval_episodes}",
                max_turns=int(grpo_max_turns),
                max_seq_len=int(sft_max_seq_len),
                split="eval_in_distribution",
            )
            unseen_eval = evaluate_freeform_greedy.remote(
                adapter_path=adapter_out,
                checkpoint_type="lora",
                episodes=int(grpo_eval_episodes),
                task_id_base=0,
                run_name=f"{spec.run_name}_unseen{grpo_eval_episodes}",
                max_turns=int(grpo_max_turns),
                max_seq_len=int(sft_max_seq_len),
                split="eval_out_of_distribution",
            )
            return {
                "method": method,
                "alpha": alpha,
                "learning_rate": lr,
                "kl_coeff": kl,
                "adapter_out": adapter_out,
                "train_summary": train_summary,
                "seen_eval": seen_eval,
                "unseen_eval": unseen_eval,
            }

        max_workers = max(1, int(max_parallel_grpo))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_one_grpo, choice) for choice in grpo_choices]
            for future in as_completed(futures):
                grpo_results.append(future.result())
    else:
        print(">>> skipping GRPO because the selected SFT did not clear the success gate")

    summary = {
        "run_tag": run_tag,
        "elapsed_s": round(time.time() - started, 2),
        "protocol": {
            "sft": "LoRA SFT on a trajectory-balanced sample from mixed runtime+official ALFWorld expert data; select by 30/30 free-form env success.",
            "dagger": "DAgger candidate aggregates expert corrections on states visited by the current policy after epoch 0.",
            "gate": f"Run GRPO only if selected SFT total seen+unseen success >= {success_threshold:.2f}, unless force_grpo is true.",
            "grpo": "Use stride-37 full-train task sampling and 30/30 evaluate_freeform_greedy seen/unseen claims.",
        },
        "sft_results": sft_results,
        "best_sft_path": best_sft_path,
        "best_sft_success": best_success,
        "sft_gate_passed": sft_gate_passed,
        "success_threshold": float(success_threshold),
        "signed_attention_summary": sat_summary,
        "grpo_results": grpo_results,
    }
    local_summary_path = REPO_ROOT / "maxrodriguez" / "results" / f"strong_dagger_pipeline_{run_tag}.json"
    _write_json(local_summary_path, summary)
    print(json.dumps({"summary_path": str(local_summary_path), **summary}, indent=2, default=str))
