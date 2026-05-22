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
from maxrodriguez.grpo.app_signed_attention_transformer import app as signed_attention_app
from maxrodriguez.supervised_FT.app_alfworld_sft_plus import (
    app as sft_app,
    evaluate_freeform_greedy,
    train_sft_plus,
)


app = modal.App("maxrodriguez-all-games-sft-alfworld")
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


@app.local_entrypoint()
def main(
    run_tag: str = "all_games_sft",
    data_path: str = "/vol/data/alfworld/sft_trajs_maxrodriguez_runtime500x10_plus_seq2seq_structured.jsonl",
    base_model_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
    sft_epochs: int = 1,
    sft_learning_rate: float = 5e-5,
    sft_max_seq_len: int = 1024,
    sft_grad_accum: int = 8,
    sft_eval_seen_episodes: int = 30,
    sft_eval_unseen_episodes: int = 30,
    success_threshold: float = 0.40,
    signed_attention_checkpoint: str = "",
    grpo_episodes: int = 100,
    grpo_k: int = 4,
    grpo_max_turns: int = 30,
    grpo_task_id_stride: int = 37,
    grpo_eval_episodes: int = 30,
    max_parallel_grpo: int = 2,
    grpo_methods: str = "trajectory_only,progress_delta,signed_attention,admissible_margin",
    force_grpo: bool = False,
) -> None:
    """Train one LoRA SFT pass over all available ALFWorld expert rows, then gate GRPO."""
    started = time.time()
    run_tag = f"{run_tag}_{_stamp()}"
    output_root = f"/vol/checkpoints/maxrodriguez_milestone/all_games_sft/{run_tag}"
    sft_output_dir = f"{output_root}/sft_all_games_1epoch"

    sft_result = train_sft_plus.remote(
        epochs=int(sft_epochs),
        learning_rate=float(sft_learning_rate),
        min_reward=1.0,
        max_seq_len=int(sft_max_seq_len),
        micro_batch_size=1,
        grad_accum=int(sft_grad_accum),
        log_every=200,
        run_name=f"sftall_{run_tag}",
        data_path=data_path,
        val_fraction=0.08,
        max_examples=0,
        output_dir=sft_output_dir,
        base_model_path=base_model_path,
        seed=42,
        use_dagger=False,
        dagger_episodes=0,
        dagger_max_turns=0,
        dagger_max_new_examples=0,
        dagger_mix_ratio=0.0,
        dagger_start_epoch=0,
        dagger_every_n_epochs=1,
        dagger_task_id_base=8000,
        dagger_split="train",
        post_eval_seen_episodes=int(sft_eval_seen_episodes),
        post_eval_unseen_episodes=int(sft_eval_unseen_episodes),
        post_eval_max_turns=30,
        post_eval_max_seq_len=int(sft_max_seq_len),
        post_eval_task_id_base=0,
        sample_after_load=False,
        full_finetune=False,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.0,
    )
    sft_success = _success_summary(sft_result)
    sft_gate_passed = bool(float(sft_success["total_rate"]) >= float(success_threshold))
    print(f">>> all-games SFT done: {json.dumps(sft_success, sort_keys=True)}")
    print(f">>> SFT gate passed={sft_gate_passed}")

    grpo_results: list[dict[str, Any]] = []
    if sft_gate_passed or bool(force_grpo):
        if not signed_attention_checkpoint and "signed_attention" in grpo_methods:
            raise ValueError("signed_attention GRPO requires --signed-attention-checkpoint")

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
                sft_adapter=str(sft_result["ckpt_dir"]),
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
                run_name_suffix=f"{run_tag}_allgames{grpo_episodes}",
                signed_attention_transformer_ckpt=signed_attention_checkpoint,
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

        with ThreadPoolExecutor(max_workers=max(1, int(max_parallel_grpo))) as executor:
            futures = [executor.submit(run_one_grpo, choice) for choice in grpo_choices]
            for future in as_completed(futures):
                grpo_results.append(future.result())
    else:
        print(">>> skipping GRPO because all-games SFT did not clear the success gate")

    summary = {
        "run_tag": run_tag,
        "elapsed_s": round(time.time() - started, 2),
        "protocol": {
            "sft": "One LoRA epoch over the full mixed ALFWorld expert dataset.",
            "grpo": "If gated, use the exact best hyperparameters from the compiled subset sweep.",
        },
        "sft_result": sft_result,
        "sft_success": sft_success,
        "sft_gate_passed": sft_gate_passed,
        "success_threshold": float(success_threshold),
        "grpo_results": grpo_results,
    }
    local_summary_path = REPO_ROOT / "maxrodriguez" / "results" / f"all_games_sft_pipeline_{run_tag}.json"
    _write_json(local_summary_path, summary)
    print(json.dumps({"summary_path": str(local_summary_path), **summary}, indent=2, default=str))
