"""Modal app for Max Rodriguez ALFWorld SFT experiments.

This file is intentionally a teaching scaffold now:

1. The ALFWorld evaluation paths still run and can score checkpoints.
2. The core SFT training code has been removed and replaced with TODOs.
3. The TODOs point to the exact pieces you should implement yourself.

The policy format is free-form ReAct: the model sees an ALFWorld text prompt
and learns to generate text containing a next action. ReAct is from Yao et al.,
"ReAct: Synergizing Reasoning and Acting in Language Models", 2022.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import modal  # type: ignore[import-not-found]
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

from infra.app_alfworld_sft_gen import _build_alfworld_config_dict, _extract_expert_plan, _extract_won
from infra.common import VOLUME_MOUNT, volume
from infra.image import ALFWORLD_DATA_DIR, alfworld_image, image
from src.datasets.sft_alfworld import (
    load_sft_examples_from_jsonl,
    summarize_sft_dataset,
    synthesize_sft_target,
)
from src.envs.alfworld_adapter import ALFWorldAdapter
from src.envs.prompts.react_alfworld import (
    parse_react_action,
    render_alfworld_turn_prompt,
)
from src.policy.lora_policy import LoRAPolicy, LoRAPolicyConfig

app = modal.App("maxrodriguez-alfworld-sft-plus")

DEFAULT_SFT_DATA_PATH = "/vol/data/alfworld/sft_trajs_maxrodriguez_500.jsonl"
DEFAULT_MANIFEST_ROOT = "/vol/manifests/maxrodriguez"
DEFAULT_EVAL_MAX_SEQ_LEN = 2048
DEFAULT_EVAL_LORA_R = 32
DEFAULT_EVAL_LORA_ALPHA = 64
DEFAULT_EVAL_LORA_DROPOUT = 0.0


# ---------- Planned SFT grid search ----------


# Milestone SFT selection grid. Hyperparameter selection always uses exactly
# one epoch; the winning settings are then reused for one final 3-epoch SFT
# checkpoint. DAgger is kept out of the primary sweep so it cannot dominate the
# BC baseline; compare minimal DAgger only after the base settings are chosen.
SFT_GRID_SEARCH_SPACE: dict[str, list[Any]] = {
    "epochs": [1],
    "learning_rate": [1.0e-4, 2.0e-5, 1.0e-5, 1.0e-6],
    "max_seq_len": [1024, 2048],
    "micro_batch_size": [1],
    "grad_accum": [4, 8],
    "log_every": [25],
    "min_reward": [1.0],
    "val_fraction": [0.08],
    "seed": [42],
    "use_dagger": [False],
    "dagger_episodes": [0],
    "dagger_max_turns": [10],
    "dagger_max_new_examples": [0],
    "dagger_mix_ratio": [0.0],
    "dagger_start_epoch": [0],
    "dagger_every_n_epochs": [1],
    "dagger_task_id_base": [8000],
    "dagger_split": ["train"],
}


SFT_FINAL_TRAINING_SPACE: dict[str, list[Any]] = {
    "epochs": [3],
    "use_dagger": [False],
}


SFT_MINIMAL_DAGGER_COMPARISON_SPACE: dict[str, list[Any]] = {
    "epochs": [3],
    "use_dagger": [False, True],
    "dagger_episodes": [5],
    "dagger_max_turns": [10],
    "dagger_max_new_examples": [100],
    "dagger_mix_ratio": [0.1],
    "dagger_start_epoch": [0],
    "dagger_every_n_epochs": [2],
    "dagger_task_id_base": [8000],
    "dagger_split": ["train"],
}


# Fill this after the sweep finishes. Keep values as None until a real run wins
# so the code does not silently treat guessed settings as final hyperparameters.
BEST_SFT_HYPERPARAMS_AFTER_SWEEP: dict[str, Any] = {
    "source_run_name": None,
    "selection_metric": None,
    "epochs": None,
    "learning_rate": None,
    "max_seq_len": None,
    "micro_batch_size": None,
    "grad_accum": None,
    "log_every": None,
    "min_reward": None,
    "val_fraction": None,
    "seed": None,
    "use_dagger": None,  # 0/False = do not use DAgger, 1/True = use DAgger.
    "dagger_episodes": None,
    "dagger_max_turns": None,
    "dagger_max_new_examples": None,
    "dagger_mix_ratio": None,
    "dagger_start_epoch": None,
    "dagger_every_n_epochs": None,
    "dagger_task_id_base": None,
    "dagger_split": None,
}


# ---------- Small shared utilities ----------


def _json_safe_float(x: object, default: float = 0.0) -> float:
    """Convert metric values to floats without crashing JSON/log handling."""
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _timestamp() -> str:
    """Return a UTC timestamp used to make run directories unique."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _write_json(path: str, payload: dict[str, Any]) -> None:
    """Write a JSON artifact and keep formatting stable for inspection."""
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _make_run_dir(run_name: str) -> str:
    """Create the Modal volume directory where eval/train artifacts are saved."""
    run_dir = f"{DEFAULT_MANIFEST_ROOT}/{run_name}_{_timestamp()}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


# ---------- SFT training scaffold ----------

# helper one to tokenize train and val rows
def tokenize_rows(split_examples: list[Any], tokenizer: Any, max_seq_len: int) -> tuple[list[dict[str, Any]], int]:
    """Convert ALFWorld SFT examples into [prompt tokens, action tokens] rows.

    The prompt is context, so TODO(core-sft-3) will mask it out of the loss.
    The action target is the supervised label, so truncation preserves it first.
    """
    rows: list[dict[str, Any]] = []
    n_truncated = 0

    for ex in split_examples:
        prompt_ids = tokenizer(ex.prompt, add_special_tokens=False).input_ids
        target_text = synthesize_sft_target(ex.action) + tokenizer.eos_token
        target_ids = tokenizer(target_text, add_special_tokens=False).input_ids

        if len(prompt_ids) + len(target_ids) > max_seq_len:
            n_truncated += 1

            if len(target_ids) >= max_seq_len:
                target_ids = target_ids[-max_seq_len:]
                prompt_ids = []
            else:
                keep_prompt_tokens = max_seq_len - len(target_ids)
                prompt_ids = prompt_ids[-keep_prompt_tokens:]

        rows.append(
            {
                "input_ids": prompt_ids + target_ids,
                "n_prompt": len(prompt_ids),
                "n_target": len(target_ids),
            }
        )

    return rows, n_truncated

def make_length_bucketed_batches(rows, micro_batch_size, bucket_size=128):
    # put similarly sized examples together so each batch needs less padding
    buckets = {}

    for row in rows:
        bucket_id = len(row["input_ids"]) // bucket_size
        buckets.setdefault(bucket_id, []).append(row)

    batches = []
    bucket_ids = list(buckets.keys())
    random.shuffle(bucket_ids)

    for bucket_id in bucket_ids:
        bucket_rows = buckets[bucket_id]
        random.shuffle(bucket_rows)

        for start in range(0, len(bucket_rows), micro_batch_size):
            batches.append(bucket_rows[start : start + micro_batch_size])

    random.shuffle(batches)
    return batches

def batch_loss(batch_rows, tokenizer, model, device):
    # get max length the gauge pad needed for each example
    max_len = max(len(row["input_ids"]) for row in batch_rows)

    pad_id = int(tokenizer.pad_token_id)

    batch_ids = []
    batch_attn = []

    for row in batch_rows:
        ids = list(row["input_ids"])

        pad = max_len - len(ids)

        batch_ids.append(ids + [pad_id] * pad)
        batch_attn.append([1] * len(ids) + [0] * pad)

    ids = torch.tensor(batch_ids, dtype=torch.long, device=device)
    attn = torch.tensor(batch_attn, dtype=torch.long, device=device)

    labels = ids.clone()

    # #magicnumberpreventionday
    mask_tok_val = -100
    labels[attn == 0] = mask_tok_val

    for i, row in enumerate(batch_rows):
        labels[i, : int(row["n_prompt"])] = mask_tok_val

    logits = model(ids, attention_mask=attn).logits.to(torch.float32)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    flat_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=mask_tok_val, # mask token value used above
        reduction="none"
    ).view(shift_labels.shape)

    mask = shift_labels != mask_tok_val

    per_row_tokens = mask.sum(dim=1).clamp_min(1)

    per_row_loss = (flat_loss * mask).sum(dim=1) / per_row_tokens

    loss = per_row_loss.mean()

    token_loss_sum = float((flat_loss * mask).sum().detach().item())
    n_tokens = int(mask.sum().item())

    return loss, token_loss_sum, n_tokens

# eval validation cross entropy helper
def eval_ce(rows, tokenizer, model, device, micro_batch_size):
    if not rows:
        return {"n_rows": 0, "per_token_ce": None}

    model.eval()
    loss_sum = 0.0
    tok_sum = 0

    with torch.no_grad():
        for start in range(0, len(rows), micro_batch_size):
            batch_rows = rows[start : start + micro_batch_size]

            _, batch_loss_sum, n_tok = batch_loss(
                batch_rows,
                tokenizer,
                model,
                device,
            )

            loss_sum += batch_loss_sum
            tok_sum += n_tok

    model.train()

    return {
        "n_rows": len(rows),
        "n_action_tokens": tok_sum,
        "per_token_ce": round(loss_sum / max(1, tok_sum), 6),
    }

def _valid_actions_for_example(ex):
    valid_actions = list(getattr(ex, "valid_actions", []) or [])

    if not valid_actions:
        for line in str(getattr(ex, "prompt", "")).splitlines():
            if line.startswith("Valid actions:"):
                raw_actions = line.split(":", 1)[1].strip()
                for raw_action in raw_actions.split(","):
                    action = raw_action.strip()
                    if not action or action.startswith("\u2026") or " more)" in action:
                        continue
                    valid_actions.append(action)
                break

    expert_action = str(getattr(ex, "action", "")).strip()
    if expert_action and expert_action not in valid_actions:
        valid_actions.append(expert_action)

    return valid_actions

def admissible_action_diagnostic(examples, tokenizer, model, max_seq_len, max_examples=64):
    # diagnostic only: checks whether expert action ranks above other valid actions
    model.eval()
    checked = 0
    skipped = 0
    top1 = 0
    margins = []

    with torch.no_grad():
        for ex in examples:
            if checked >= max_examples:
                break

            expert_action = str(getattr(ex, "action", "")).strip()
            valid_actions = _valid_actions_for_example(ex)
            if not expert_action or len(valid_actions) < 2:
                skipped += 1
                continue

            ranked = _score_valid_actions(
                model=model,
                tokenizer=tokenizer,
                prompt=ex.prompt,
                actions=valid_actions,
                max_seq_len=max_seq_len,
                score_normalization="mean",
            )
            if not ranked:
                skipped += 1
                continue

            expert_rows = [row for row in ranked if str(row["action"]).strip() == expert_action]
            wrong_rows = [row for row in ranked if str(row["action"]).strip() != expert_action]
            if not expert_rows or not wrong_rows:
                skipped += 1
                continue

            checked += 1
            if str(ranked[0]["action"]).strip() == expert_action:
                top1 += 1

            expert_score = _json_safe_float(expert_rows[0].get("mean_logprob"), float("-inf"))
            best_wrong_score = max(_json_safe_float(row.get("mean_logprob"), float("-inf")) for row in wrong_rows)
            margins.append(expert_score - best_wrong_score)

    model.train()

    return {
        "n_admissible_examples": checked,
        "n_skipped_examples": skipped,
        "admissible_top1": round(top1 / max(1, checked), 6),
        "mean_expert_margin": round(sum(margins) / max(1, len(margins)), 6) if margins else None,
    }

def collect_dagger_examples(
    model,
    tokenizer,
    episodes,
    task_id_base,
    max_turns,
    max_history_turns,
    max_seq_len,
    split,
):
    # DAgger: visit states with current policy, ask expert planner for correction
    adapter = _build_alfworld_adapter(max_turns=max_turns, split=split)
    device = next(model.parameters()).device
    model.eval()

    dagger_examples = []
    n_no_expert_plan = 0

    with torch.no_grad():
        for ep in range(episodes):
            state = adapter.reset(task_id=task_id_base + ep)
            history = []

            for step_idx in range(max_turns):
                prompt = render_alfworld_turn_prompt(
                    state,
                    history,
                    max_history_turns=max_history_turns,
                )

                expert_plan = _extract_expert_plan(getattr(state, "raw_info", {}))
                if expert_plan:
                    dagger_examples.append(
                        SimpleNamespace(
                            prompt=prompt,
                            action=expert_plan[0],
                            instruction=getattr(state, "observation_text", "") or "",
                            step_idx=step_idx,
                            trajectory_id=f"dagger_{task_id_base + ep}",
                            final_reward=1.0,
                        )
                    )
                else:
                    n_no_expert_plan += 1

                enc = tokenizer(
                    prompt,
                    add_special_tokens=False,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                ids = enc.input_ids[:, -max_seq_len:].to(device)
                attn = enc.attention_mask[:, -max_seq_len:].to(device)

                out = model.generate(
                    ids,
                    attention_mask=attn,
                    do_sample=False,
                    max_new_tokens=48,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                generation = tokenizer.decode(out[0, ids.shape[-1] :], skip_special_tokens=True)
                policy_action = parse_react_action(generation)

                history.append(
                    SimpleNamespace(
                        observation_text=getattr(state, "observation_text", "") or "",
                        action_text=policy_action,
                    )
                )

                next_state, _reward, done, _info = adapter.step(policy_action)
                state = next_state
                if done:
                    break

    model.train()

    return {
        "examples": dagger_examples,
        "n_examples": len(dagger_examples),
        "n_no_expert_plan": n_no_expert_plan,
    }

@app.function(image=image, gpu="A100-80GB", volumes={VOLUME_MOUNT: volume}, timeout=24 * 60 * 60)
def train_sft_plus(
    epochs: int,
    learning_rate: float,
    min_reward: float,
    max_seq_len: int,
    micro_batch_size: int,
    grad_accum: int,
    log_every: int,
    run_name: str,
    data_path: str,
    val_fraction: float,
    max_examples: int,
    output_dir: str,
    base_model_path: str,
    seed: int,
    use_dagger: bool,
    dagger_episodes: int,
    dagger_max_turns: int,
    dagger_max_new_examples: int,
    dagger_mix_ratio: float,
    dagger_start_epoch: int,
    dagger_every_n_epochs: int,
    dagger_task_id_base: int,
    dagger_split: str,
) -> dict:
    """SFT training entrypoint with the core implementation intentionally blank.

    Supervised fine-tuning here means behavior cloning on expert ALFWorld
    trajectories: maximize log probability of the expert action text under the
    prompt. This is the supervised pretraining stage used before RLHF-style
    optimization in Ouyang et al., "Training language models to follow
    instructions with human feedback", 2022.

    This is the full fine-tune path. LoRA is still supported by eval helpers
    for old checkpoints, but this training entrypoint does not train adapters.
    """
    sys.path.insert(0, "/workspace")

    # Reproducibility for the dataset split and PyTorch initialization.
    torch.manual_seed(seed)
    random.seed(seed)

    # Build the checkpoint directory. A full implementation should write model
    # weights here, plus train_log.json and summary.json.
    ckpt_dir = output_dir or f"/vol/checkpoints/{run_name}_{_timestamp()}"
    os.makedirs(ckpt_dir, exist_ok=True)

    # Load expert demonstrations. min_reward filters out failed trajectories so
    # the BC/SFT objective imitates successful behavior.
    examples = load_sft_examples_from_jsonl(
        data_path,
        min_reward=min_reward,
        max_examples=(max_examples or None),
    )
    dataset_summary = summarize_sft_dataset(examples)
    if not examples:
        raise RuntimeError(f"No usable SFT examples found at {data_path}")

    # Deterministic train/validation split. Validation CE should be your main
    # supervised metric before launching ALFWorld environment evaluations.
    order = list(range(len(examples)))
    random.shuffle(order)
    n_val = max(1, int(round(len(order) * val_fraction))) if len(order) >= 20 else 0
    val_ids = set(order[:n_val])
    train_examples = [ex for i, ex in enumerate(examples) if i not in val_ids]
    val_examples = [ex for i, ex in enumerate(examples) if i in val_ids]

    # TODO(core-sft-1): DONE. Load tokenizer and causal LM.
    # - Full SFT path: AutoModelForCausalLM.from_pretrained(...), all params trainable.
    # - LoRA path: attach low-rank adapters following Hu et al.,
    #   "LoRA: Low-Rank Adaptation of Large Language Models", 2021.
    # - If memory is tight, implement QLoRA with 4-bit quantization following
    #   Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", 2023.
    # - Set pad_token, disable use_cache during training, and consider gradient
    #   checkpointing for full fine-tuning.
    #
    model_source = base_model_path

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        cache_dir="/vol/hf_cache",
        use_fast=True
    )

    # set pad token to eos if none
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch.bfloat16, # semi optimzed
        cache_dir="/vol/hf_cache",
        device_map="cuda:0"
    )

    model.requires_grad_(True)

    # kv cache is for gen not for gradient based training
    model.config.use_cache = False

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    device = next(model.parameters()).device

    # TODO(core-sft-2): DONE. Tokenize each example into prompt tokens and target
    # action tokens.
    # - Prompt tokens come from ex.prompt.
    # - Target text should be synthesize_sft_target(ex.action) + eos_token.
    # - Truncate from the left of the prompt first so the action labels remain.
    # - Track how many rows were truncated at max_seq_len.
    #
    train_rows, n_train_trunc = tokenize_rows(train_examples, tokenizer, max_seq_len)
    val_rows, n_val_trunc = tokenize_rows(val_examples, tokenizer, max_seq_len)
    base_train_row_count = len(train_rows)
    n_dagger_rows_total = 0



    # TODO(core-sft-3): DONE. Implement the masked behavior-cloning loss.
    # - Run the causal LM on [prompt, target].
    # - Shift logits and labels by one position.
    # - Mask prompt and padding labels to -100.
    # - Cross entropy is computed only on target/action tokens:
    #     L_SFT(theta) = - mean_t log pi_theta(a_t^expert | prompt_t).
    #

    ############# done see helper above ################

    # TODO(core-sft-4): DONE. Implement the optimizer/training loop.
    # - AdamW is the standard baseline.
    # - Use micro_batch_size and grad_accum to form an effective batch.
    # - Clip grad norm, log train CE every log_every optimizer steps, and compute
    #   validation CE at the end of each epoch.
    #
    # training loop time

    trainable = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        weight_decay=0.01
    )

    micro_batch_size = max(1, int(micro_batch_size))

    grad_accum = max(1, int(grad_accum))

    log= []

    global_step = 0
    micro= 0
    t0 = time.time()

    optimizer.zero_grad(set_to_none=True)
    model.train()

    for epoch in range(epochs):
        batches = make_length_bucketed_batches(train_rows, micro_batch_size)

        ep_loss_sum = 0.0
        ep_tok_sum = 0

        for batch_rows in batches:
            loss, batch_loss_sum, n_tok = batch_loss(
                batch_rows,
                tokenizer,
                model,
                device
            )

            (loss / grad_accum).backward()
            micro += 1

            if micro % grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

                if global_step % log_every == 0:
                    row_log = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "train_loss_last_batch": round(float(loss.detach().item()), 6),
                        "grad_norm": round(float(grad_norm), 6),
                        "n_action_tokens": n_tok,
                        "n_rows": len(batch_rows),
                        "elapsed_s": round(time.time() - t0, 1),
                    }

                    log.append(row_log)
                    print(">>>", row_log)

            ep_loss_sum += batch_loss_sum
            ep_tok_sum += n_tok

        val_metrics = eval_ce(val_rows, tokenizer, model, device, micro_batch_size)
        admissible_metrics = admissible_action_diagnostic(
            val_examples,
            tokenizer,
            model,
            max_seq_len,
        )

        epoch_row = {
            "epoch": epoch,
            "epoch_summary": True,
            "train_per_token_ce": round(ep_loss_sum / max(1, ep_tok_sum), 6),
            "train_action_tokens": ep_tok_sum,
            "val": val_metrics,
            "admissible_action_diagnostic": admissible_metrics,
        }

        dagger_due = (
            use_dagger
            and dagger_episodes > 0
            and epoch < epochs - 1
            and epoch >= dagger_start_epoch
            and (epoch - dagger_start_epoch) % max(1, dagger_every_n_epochs) == 0
        )
        dagger_total_cap = int(base_train_row_count * max(0.0, dagger_mix_ratio))
        dagger_remaining_cap = max(0, dagger_total_cap - n_dagger_rows_total)
        dagger_add_cap = min(max(0, dagger_max_new_examples), dagger_remaining_cap)

        if dagger_due and dagger_add_cap > 0:
            dagger_result = collect_dagger_examples(
                model=model,
                tokenizer=tokenizer,
                episodes=dagger_episodes,
                task_id_base=dagger_task_id_base + epoch * dagger_episodes,
                max_turns=dagger_max_turns,
                max_history_turns=3,
                max_seq_len=max_seq_len,
                split=dagger_split,
            )
            dagger_examples = dagger_result["examples"]
            if len(dagger_examples) > dagger_add_cap:
                dagger_examples = dagger_examples[:dagger_add_cap]
            dagger_rows, n_dagger_trunc = tokenize_rows(dagger_examples, tokenizer, max_seq_len)
            train_examples.extend(dagger_examples)
            train_rows.extend(dagger_rows)
            n_train_trunc += n_dagger_trunc
            n_dagger_rows_total += len(dagger_rows)
            epoch_row["dagger"] = {
                "enabled": True,
                "n_added_examples": len(dagger_rows),
                "n_total_train_rows_after_add": len(train_rows),
                "n_dagger_rows_total": n_dagger_rows_total,
                "dagger_total_cap": dagger_total_cap,
                "dagger_add_cap": dagger_add_cap,
                "n_truncated_added_rows": n_dagger_trunc,
                "n_no_expert_plan": dagger_result["n_no_expert_plan"],
            }
        else:
            epoch_row["dagger"] = {
                "enabled": bool(use_dagger),
                "due": bool(dagger_due),
                "n_added_examples": 0,
                "n_dagger_rows_total": n_dagger_rows_total,
                "dagger_total_cap": dagger_total_cap,
                "dagger_add_cap": dagger_add_cap,
            }

        log.append(epoch_row)
        print(">>> END", epoch_row)

    if micro % grad_accum != 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        row_log = {
            "global_step": global_step,
            "leftover_microbatches_stepped": micro % grad_accum,
            "grad_norm": round(float(grad_norm), 6),
            "elapsed_s": round(time.time() - t0, 1),
        }
        log.append(row_log)
        print(">>>", row_log)


    # TODO(core-sft-5): DONE. Save checkpoints and logs.
    # - Full fine-tune: save model + tokenizer.
    # - Write train_log.json and summary.json into ckpt_dir.

    print(">>> Saving full SFT checkpoint to", ckpt_dir)

    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    final = {
        "ckpt_dir": ckpt_dir,
        "checkpoint_type": "full",
        "n_examples_total": len(examples),
        "n_train_examples": len(train_rows),
        "n_val_examples": len(val_rows),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "global_steps": global_step,
        "total_elapsed_s": round(time.time() - t0, 1),
        "final_log_row": log[-1] if log else None,
        "n_train_truncated_prompts": n_train_trunc,
        "n_val_truncated_prompts": n_val_trunc,
        "full_finetune": True,
        "max_seq_len": max_seq_len,
        "micro_batch_size": micro_batch_size,
        "grad_accum": grad_accum,
        "base_model_path": base_model_path,
        "use_dagger": use_dagger,
        "dagger_episodes": dagger_episodes,
        "dagger_max_turns": dagger_max_turns,
        "dagger_max_new_examples": dagger_max_new_examples,
        "dagger_mix_ratio": dagger_mix_ratio,
        "dagger_start_epoch": dagger_start_epoch,
        "dagger_every_n_epochs": dagger_every_n_epochs,
        "dagger_task_id_base": dagger_task_id_base,
        "dagger_split": dagger_split,
        "n_dagger_rows_total": n_dagger_rows_total,
    }
    manifest_dir = _make_run_dir(f"{run_name}_train")
    final["manifest_dir"] = manifest_dir

    with open(os.path.join(ckpt_dir, "train_log.json"), "w") as f:
        train_artifact = {
            "rows": log,
            "summary": final,
            "dataset_summary": dataset_summary,
            "config": {
                "data_path": data_path,
                "min_reward": min_reward,
                "seed": seed,
                "val_fraction": val_fraction,
                "max_examples": max_examples,
                "base_model_path": base_model_path,
                "use_dagger": use_dagger,
                "dagger_episodes": dagger_episodes,
                "dagger_max_turns": dagger_max_turns,
                "dagger_max_new_examples": dagger_max_new_examples,
                "dagger_mix_ratio": dagger_mix_ratio,
                "dagger_start_epoch": dagger_start_epoch,
                "dagger_every_n_epochs": dagger_every_n_epochs,
                "dagger_task_id_base": dagger_task_id_base,
                "dagger_split": dagger_split,
            },
        }
        json.dump(train_artifact, f, indent=2, default=str)

    with open(os.path.join(ckpt_dir, "summary.json"), "w") as f:
        json.dump(final, f, indent=2, default=str)
    with open(os.path.join(manifest_dir, "train_log.json"), "w") as f:
        json.dump(train_artifact, f, indent=2, default=str)
    with open(os.path.join(manifest_dir, "summary.json"), "w") as f:
        json.dump(final, f, indent=2, default=str)

    volume.commit()
    return final


# ---------- Evaluation model/env helpers ----------


def _load_eval_model(
    adapter_path: str,
    checkpoint_type: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> tuple[Any, Any]:
    """Load either a full checkpoint, a LoRA adapter, or the base model."""
    if checkpoint_type == "full" and adapter_path:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, cache_dir="/vol/hf_cache", use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            adapter_path,
            torch_dtype=torch.bfloat16,
            cache_dir="/vol/hf_cache",
            device_map="cuda:0",
        )
        model.eval()
        return tokenizer, model

    policy = LoRAPolicy(
        LoRAPolicyConfig(
            cache_dir="/vol/hf_cache",
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
    )
    if adapter_path:
        policy.load_adapter(adapter_path)
    policy.model.eval()
    return policy.tokenizer, policy.model


def _build_alfworld_adapter(max_turns: int, split: str) -> Any:
    """Construct the ALFWorld text environment for seen/unseen evaluation."""
    os.environ.setdefault("ALFWORLD_DATA", ALFWORLD_DATA_DIR)
    config = _build_alfworld_config_dict()
    config["dataset"]["num_eval_games"] = -1
    return ALFWorldAdapter(
        max_steps=max_turns,
        observation_mode="text",
        task_split=split,
        env_kwargs={"config": config},
    )


def _summarize_eval(
    rows: list[dict[str, Any]],
    run_dir: str,
    adapter_path: str,
    checkpoint_type: str,
    eval_type: str,
    episodes: int,
    task_id_base: int,
    split: str,
    max_turns: int,
    started_at: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate per-episode rows into the summary written beside eval logs."""
    successes = [1.0 if row["success"] else 0.0 for row in rows]
    rewards = [float(row["final_reward"]) for row in rows]
    summary = {
        "run_dir": run_dir,
        "adapter_path": adapter_path or "<base>",
        "checkpoint_type": checkpoint_type if adapter_path else "base",
        "eval_type": eval_type,
        "episodes": episodes,
        "task_id_base": task_id_base,
        "task_id_range": [task_id_base, task_id_base + episodes],
        "split": split,
        "max_turns": max_turns,
        "avg_return": round(sum(rewards) / max(1, len(rewards)), 4),
        "pct_success": round(sum(successes) / max(1, len(successes)), 4),
        "n_success": int(sum(successes)),
        "n_eval": len(rows),
        "avg_turns": round(sum(int(r["n_turns"]) for r in rows) / max(1, len(rows)), 2),
        "elapsed_s": round(time.time() - started_at, 2),
    }
    if extra:
        summary.update(extra)
    return summary


# ---------- Admissible-action rerank evaluation ----------


def _score_valid_actions(
    model: Any,
    tokenizer: Any,
    prompt: str,
    actions: list[str],
    max_seq_len: int,
    score_normalization: str,
) -> list[dict[str, Any]]:
    """Score each valid action by target-token log probability under the model."""
    rows: list[dict[str, Any]] = []
    for action in actions:
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        target = synthesize_sft_target(action) + tokenizer.eos_token
        target_ids = tokenizer(target, add_special_tokens=False).input_ids

        # Keep the full action label whenever possible; drop older prompt tokens
        # first because ALFWorld prompts can grow with history.
        if len(prompt_ids) + len(target_ids) > max_seq_len:
            if len(target_ids) >= max_seq_len:
                target_ids = target_ids[-max_seq_len:]
                prompt_ids = []
            else:
                prompt_ids = prompt_ids[-(max_seq_len - len(target_ids)) :]

        rows.append(
            {
                "action": action,
                "input_ids": prompt_ids + target_ids,
                "n_prompt": len(prompt_ids),
            }
        )

    device = next(model.parameters()).device
    max_len = max(len(row["input_ids"]) for row in rows)
    pad_id = int(tokenizer.pad_token_id)
    batch_ids = []
    batch_attn = []
    for row in rows:
        ids = list(row["input_ids"])
        pad = max_len - len(ids)
        batch_ids.append(ids + [pad_id] * pad)
        batch_attn.append([1] * len(ids) + [0] * pad)

    scores: list[dict[str, Any]] = []
    with torch.no_grad():
        ids_t = torch.tensor(batch_ids, dtype=torch.long, device=device)
        attn_t = torch.tensor(batch_attn, dtype=torch.long, device=device)
        logits = model(ids_t, attention_mask=attn_t).logits[:, :-1, :].to(torch.float32)
        labels = ids_t[:, 1:].contiguous()
        token_logp = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

        for i, row in enumerate(rows):
            real_len = int(sum(batch_attn[i]))
            label_len = max(0, real_len - 1)
            target_start = max(0, int(row["n_prompt"]) - 1)
            selected = token_logp[i, target_start:label_len]
            total = float(selected.sum().item()) if selected.numel() else float("-inf")
            mean = total / max(1, int(selected.numel()))
            scores.append(
                {
                    "action": row["action"],
                    "sum_logprob": total,
                    "mean_logprob": mean,
                    "n_tokens": int(selected.numel()),
                }
            )

    key = "mean_logprob" if score_normalization == "mean" else "sum_logprob"
    scores.sort(key=lambda row: _json_safe_float(row.get(key), float("-inf")), reverse=True)
    return scores


@app.function(image=alfworld_image, gpu="A100-80GB", volumes={VOLUME_MOUNT: volume}, timeout=2 * 60 * 60)
def evaluate_action_rerank(
    adapter_path: str = "",
    checkpoint_type: str = "lora",
    episodes: int = 50,
    seen_episodes: int = 140,
    unseen_episodes: int = 134,
    task_id_base: int = 6500,
    run_name: str = "maxrodriguez_rerank_eval",
    max_turns: int = 40,
    max_history_turns: int = 3,
    max_seq_len: int = 2048,
    score_normalization: str = "mean",
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.0,
    split: str = "eval_out_of_distribution",
) -> dict:
    """Evaluate by scoring ALFWorld valid actions and executing the best one."""
    sys.path.insert(0, "/workspace")

    run_dir = _make_run_dir(run_name)
    tokenizer, model = _load_eval_model(adapter_path, checkpoint_type, lora_r, lora_alpha, lora_dropout)
    adapter = _build_alfworld_adapter(max_turns=max_turns, split=split)

    rows: list[dict[str, Any]] = []
    started_at = time.time()
    for ep in range(episodes):
        task_id = task_id_base + ep
        ep_started_at = time.time()
        state = adapter.reset(task_id=task_id)
        history: list[SimpleNamespace] = []
        total_reward = 0.0
        done = False
        won = False
        turns: list[dict[str, Any]] = []

        # At each turn, ALFWorld gives admissible actions. This evaluator uses
        # the SFT model as a scorer over those actions instead of free generation.
        for turn_idx in range(max_turns):
            prompt = render_alfworld_turn_prompt(state, history, max_history_turns=max_history_turns)
            valid_actions = list(getattr(state, "valid_actions", []) or ["look", "inventory"])
            ranked = _score_valid_actions(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                actions=valid_actions,
                max_seq_len=max_seq_len,
                score_normalization=score_normalization,
            )
            action = str(ranked[0]["action"]) if ranked else "look"
            next_state, reward, done, info = adapter.step(action)
            won = won or _extract_won(info) or bool(float(reward) > 0)
            turns.append(
                {
                    "turn_idx": turn_idx,
                    "observation_head": (getattr(state, "observation_text", "") or "")[:500],
                    "n_valid_actions": len(valid_actions),
                    "action": action,
                    "reward": float(reward),
                    "done": bool(done),
                    "top5": ranked[:5],
                }
            )
            history.append(
                SimpleNamespace(
                    observation_text=getattr(state, "observation_text", "") or "",
                    action_text=action,
                )
            )
            total_reward += float(reward)
            state = next_state
            if done:
                break

        row = {
            "episode": ep,
            "task_id": task_id,
            "final_reward": total_reward,
            "success": bool(won or total_reward > 0.0),
            "n_turns": len(turns),
            "done": bool(done),
            "elapsed_s": round(time.time() - ep_started_at, 2),
            "turns": turns,
        }
        rows.append(row)
        print(f"ep={ep:03d} task={task_id} R={total_reward:.1f} success={row['success']}")
        _write_json(os.path.join(run_dir, "eval_log.json"), {"rows": rows})
        volume.commit()

    summary = _summarize_eval(
        rows=rows,
        run_dir=run_dir,
        adapter_path=adapter_path,
        checkpoint_type=checkpoint_type,
        eval_type="action_rerank",
        episodes=episodes,
        task_id_base=task_id_base,
        split=split,
        max_turns=max_turns,
        started_at=started_at,
        extra={
            "max_history_turns": max_history_turns,
            "score_normalization": score_normalization,
        },
    )
    _write_json(os.path.join(run_dir, "eval_log.json"), {"summary": summary, "rows": rows})
    _write_json(os.path.join(run_dir, "summary.json"), summary)
    volume.commit()
    return summary


# ---------- Free-form greedy evaluation ----------


@app.function(image=alfworld_image, gpu="A100-80GB", volumes={VOLUME_MOUNT: volume}, timeout=2 * 60 * 60)
def evaluate_freeform_greedy(
    adapter_path: str = "",
    checkpoint_type: str = "lora",
    episodes: int = 50,
    task_id_base: int = 6500,
    run_name: str = "maxrodriguez_freeform_eval",
    max_turns: int = 30,
    max_history_turns: int = 3,
    max_seq_len: int = 2048,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.0,
    split: str = "eval_out_of_distribution",
    stop_after_success: bool = False,
) -> dict:
    """Evaluate by greedily generating an action string at each ALFWorld turn."""
    sys.path.insert(0, "/workspace")

    run_dir = _make_run_dir(run_name)
    tokenizer, model = _load_eval_model(adapter_path, checkpoint_type, lora_r, lora_alpha, lora_dropout)
    adapter = _build_alfworld_adapter(max_turns=max_turns, split=split)
    device = next(model.parameters()).device

    rows: list[dict[str, Any]] = []
    started_at = time.time()
    for ep in range(episodes):
        task_id = task_id_base + ep
        ep_started_at = time.time()
        state = adapter.reset(task_id=task_id)
        history: list[SimpleNamespace] = []
        total_reward = 0.0
        done = False
        won = False
        turns: list[dict[str, Any]] = []

        # Free-form mode matches the policy's natural deployment: generate text,
        # parse the ReAct action, then send that action to ALFWorld.
        for turn_idx in range(max_turns):
            prompt = render_alfworld_turn_prompt(state, history, max_history_turns=max_history_turns)
            enc = tokenizer(
                prompt,
                add_special_tokens=False,
                return_tensors="pt",
                return_attention_mask=True,
            )
            ids = enc.input_ids[:, -max_seq_len:].to(device)
            attn = enc.attention_mask[:, -max_seq_len:].to(device)

            with torch.no_grad():
                out = model.generate(
                    ids,
                    attention_mask=attn,
                    do_sample=False,
                    max_new_tokens=48,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generation = tokenizer.decode(out[0, ids.shape[-1] :], skip_special_tokens=True)
            action = parse_react_action(generation)
            next_state, reward, done, info = adapter.step(action)
            won = won or _extract_won(info) or bool(float(reward) > 0)
            turns.append(
                {
                    "turn_idx": turn_idx,
                    "action": action,
                    "generation": generation[:500],
                    "reward": float(reward),
                    "done": bool(done),
                }
            )
            history.append(
                SimpleNamespace(
                    observation_text=getattr(state, "observation_text", "") or "",
                    action_text=action,
                )
            )
            total_reward += float(reward)
            state = next_state
            if done:
                break

        row = {
            "episode": ep,
            "task_id": task_id,
            "final_reward": total_reward,
            "success": bool(won or total_reward > 0.0),
            "n_turns": len(turns),
            "done": bool(done),
            "elapsed_s": round(time.time() - ep_started_at, 2),
            "turns": turns,
        }
        rows.append(row)
        print(f"ep={ep:03d} task={task_id} R={total_reward:.1f} success={row['success']}")
        _write_json(os.path.join(run_dir, "eval_log.json"), {"rows": rows})
        volume.commit()
        if stop_after_success and row["success"]:
            print(">>> stop_after_success=True; stopping after first successful episode")
            break

    summary = _summarize_eval(
        rows=rows,
        run_dir=run_dir,
        adapter_path=adapter_path,
        checkpoint_type=checkpoint_type,
        eval_type="freeform_greedy",
        episodes=episodes,
        task_id_base=task_id_base,
        split=split,
        max_turns=max_turns,
        started_at=started_at,
        extra={
            "max_history_turns": max_history_turns,
            "stop_after_success": stop_after_success,
        },
    )
    _write_json(os.path.join(run_dir, "eval_log.json"), {"summary": summary, "rows": rows})
    _write_json(os.path.join(run_dir, "summary.json"), summary)
    volume.commit()
    return summary


# ---------- Artifact reader and local dispatcher ----------


@app.function(image=image, volumes={VOLUME_MOUNT: volume}, timeout=5 * 60)
def read_json_artifact(path: str) -> dict:
    """Read a JSON artifact from the Modal volume for local reporting."""
    with open(path) as f:
        return json.load(f)


@app.local_entrypoint()
def main(
    action: str = "train",
    adapter_path: str = "",
    data_path: str = DEFAULT_SFT_DATA_PATH,
    epochs: int | None = None,
    learning_rate: float | None = None,
    min_reward: float | None = None,
    max_seq_len: int | None = None,
    micro_batch_size: int | None = None,
    grad_accum: int | None = None,
    log_every: int | None = None,
    lora_r: int | None = None,
    lora_alpha: int | None = None,
    lora_dropout: float | None = None,
    full_finetune: bool = False,
    seed: int | None = None,
    val_fraction: float | None = None,
    max_examples: int = 0,
    checkpoint_type: str = "lora",
    output_dir: str = "",
    base_model_path: str = "",
    episodes: int = 50,
    task_id_base: int = 6500,
    run_name: str = "",
    score_normalization: str = "mean",
    split: str = "eval_out_of_distribution",
    path: str = "",
    max_turns: int = 30,
    stop_after_success: bool = False,
    use_dagger: bool = False,
    dagger_episodes: int = 5,
    dagger_max_turns: int = 10,
    dagger_max_new_examples: int = 100,
    dagger_mix_ratio: float = 0.1,
    dagger_start_epoch: int = 1,
    dagger_every_n_epochs: int = 2,
    dagger_task_id_base: int = 8000,
    dagger_split: str = "train",
) -> None:
    """Dispatch one Modal action from the command line."""
    if action in {"train", "train_then_eval", "train_then_eval_both"}:
        missing = [
            name
            for name, value in {
                "epochs": epochs,
                "learning_rate": learning_rate,
                "min_reward": min_reward,
                "max_seq_len": max_seq_len,
                "micro_batch_size": micro_batch_size,
                "grad_accum": grad_accum,
                "log_every": log_every,
                "seed": seed,
                "val_fraction": val_fraction,
                "run_name": run_name,
                "base_model_path": base_model_path,
            }.items()
            if value is None
        ]
        if not run_name:
            missing.append("run_name")
        if not base_model_path:
            missing.append("base_model_path")
        if missing:
            raise ValueError(
                "action=train requires explicit values for "
                + ", ".join(missing)
                + ". Candidate sweep values live in SFT_GRID_SEARCH_SPACE."
            )

        res = train_sft_plus.remote(
            epochs=epochs,
            learning_rate=learning_rate,
            min_reward=min_reward,
            max_seq_len=max_seq_len,
            micro_batch_size=micro_batch_size,
            grad_accum=grad_accum,
            log_every=log_every,
            seed=seed,
            val_fraction=val_fraction,
            max_examples=max_examples,
            run_name=run_name,
            data_path=data_path,
            output_dir=output_dir,
            base_model_path=base_model_path,
            use_dagger=use_dagger,
            dagger_episodes=dagger_episodes,
            dagger_max_turns=dagger_max_turns,
            dagger_max_new_examples=dagger_max_new_examples,
            dagger_mix_ratio=dagger_mix_ratio,
            dagger_start_epoch=dagger_start_epoch,
            dagger_every_n_epochs=dagger_every_n_epochs,
            dagger_task_id_base=dagger_task_id_base,
            dagger_split=dagger_split,
        )
        if action == "train_then_eval":
            eval_res = evaluate_freeform_greedy.remote(
                adapter_path=res["ckpt_dir"],
                checkpoint_type="full",
                episodes=episodes,
                task_id_base=task_id_base,
                run_name=f"{run_name}_freeform_{split}",
                max_turns=max_turns,
                max_seq_len=max_seq_len or DEFAULT_EVAL_MAX_SEQ_LEN,
                lora_r=lora_r or DEFAULT_EVAL_LORA_R,
                lora_alpha=lora_alpha or DEFAULT_EVAL_LORA_ALPHA,
                lora_dropout=DEFAULT_EVAL_LORA_DROPOUT if lora_dropout is None else lora_dropout,
                split=split,
                stop_after_success=stop_after_success,
            )
            res = {"train": res, "eval": eval_res}
        elif action == "train_then_eval_both":
            seen_res = evaluate_freeform_greedy.remote(
                adapter_path=res["ckpt_dir"],
                checkpoint_type="full",
                episodes=seen_episodes,
                task_id_base=0,
                run_name=f"{run_name}_freeform_seen{seen_episodes}",
                max_turns=max_turns,
                max_seq_len=max_seq_len or DEFAULT_EVAL_MAX_SEQ_LEN,
                lora_r=lora_r or DEFAULT_EVAL_LORA_R,
                lora_alpha=lora_alpha or DEFAULT_EVAL_LORA_ALPHA,
                lora_dropout=DEFAULT_EVAL_LORA_DROPOUT if lora_dropout is None else lora_dropout,
                split="eval_in_distribution",
                stop_after_success=False,
            )
            unseen_res = evaluate_freeform_greedy.remote(
                adapter_path=res["ckpt_dir"],
                checkpoint_type="full",
                episodes=unseen_episodes,
                task_id_base=0,
                run_name=f"{run_name}_freeform_unseen{unseen_episodes}",
                max_turns=max_turns,
                max_seq_len=max_seq_len or DEFAULT_EVAL_MAX_SEQ_LEN,
                lora_r=lora_r or DEFAULT_EVAL_LORA_R,
                lora_alpha=lora_alpha or DEFAULT_EVAL_LORA_ALPHA,
                lora_dropout=DEFAULT_EVAL_LORA_DROPOUT if lora_dropout is None else lora_dropout,
                split="eval_out_of_distribution",
                stop_after_success=False,
            )
            res = {"train": res, "seen_eval": seen_res, "unseen_eval": unseen_res}
    elif action == "rerank_eval":
        res = evaluate_action_rerank.remote(
            adapter_path=adapter_path,
            checkpoint_type=checkpoint_type,
            episodes=episodes,
            task_id_base=task_id_base,
            run_name=run_name or "maxrodriguez_rerank_eval",
            max_turns=max_turns,
            max_seq_len=max_seq_len or DEFAULT_EVAL_MAX_SEQ_LEN,
            score_normalization=score_normalization,
            lora_r=lora_r or DEFAULT_EVAL_LORA_R,
            lora_alpha=lora_alpha or DEFAULT_EVAL_LORA_ALPHA,
            lora_dropout=DEFAULT_EVAL_LORA_DROPOUT if lora_dropout is None else lora_dropout,
            split=split,
        )
    elif action == "freeform_eval":
        res = evaluate_freeform_greedy.remote(
            adapter_path=adapter_path,
            checkpoint_type=checkpoint_type,
            episodes=episodes,
            task_id_base=task_id_base,
            run_name=run_name or "maxrodriguez_freeform_eval",
            max_turns=max_turns,
            max_seq_len=max_seq_len or DEFAULT_EVAL_MAX_SEQ_LEN,
            lora_r=lora_r or DEFAULT_EVAL_LORA_R,
            lora_alpha=lora_alpha or DEFAULT_EVAL_LORA_ALPHA,
            lora_dropout=DEFAULT_EVAL_LORA_DROPOUT if lora_dropout is None else lora_dropout,
            split=split,
            stop_after_success=stop_after_success,
        )
    elif action == "read_json":
        if not path:
            raise ValueError("--path is required for action=read_json")
        res = read_json_artifact.remote(path)
    else:
        raise ValueError(
            "action must be one of: train, train_then_eval, train_then_eval_both, "
            "rerank_eval, freeform_eval, read_json"
        )

    print(json.dumps(res, indent=2, default=str))
