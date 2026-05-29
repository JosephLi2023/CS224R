"""Modal A100 app: SFT warm-start of Qwen2.5-1.5B + LoRA on WebShop human trajectories.

  # Fire-and-forget (keeps running after local CLI exits):
  modal run --detach infra/app_sft_train.py::sft_train --epochs 6 --run-name my_run ...

  # Block until finished (local terminal streams logs):
  modal run infra/app_sft_train.py::main --wait --epochs 6 ...

  # Resume after a partial run:
  modal run --detach infra/app_sft_train.py::sft_train \
      --resume-from /vol/checkpoints/my_run_<ts> ...

Tokenizes (prompt, action) SFT examples loaded via src.datasets.sft_webshop,
runs masked cross-entropy (only action tokens contribute to the loss),
saves the LoRA adapter to /vol/checkpoints/<run_name>_<ts>/ with periodic
checkpoints under step_*/epoch_*/latest/.
"""
from __future__ import annotations

import modal  # type: ignore[import-not-found]

from infra.common import VOLUME_MOUNT, volume
from infra.image import image

app = modal.App("cs224r-hgpo-sft-train")

DEFAULT_SFT_DATA_PATH = "/vol/data/webshop/human_trajs"


@app.function(image=image, gpu="A100-80GB", volumes={VOLUME_MOUNT: volume}, timeout=8 * 60 * 60)
def sft_train(
    epochs: int = 3,
    learning_rate: float = 1e-4,
    min_reward: float = 0.5,
    max_seq_len: int = 1024,
    grad_accum: int = 4,
    log_every: int = 25,
    save_every_steps: int = 500,
    run_name: str = "sft_v1",
    data_path: str = DEFAULT_SFT_DATA_PATH,
    lora_rank: int = 16,
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj",
    resume_from: str = "",
) -> dict:
    import json
    import os
    import sys
    import time
    from datetime import datetime, timezone

    sys.path.insert(0, "/workspace")

    import torch
    from torch.nn import functional as F

    from src.datasets.sft_webshop import (
        load_sft_examples_from_directory,
        load_sft_examples_from_jsonl,
        summarize_sft_dataset,
        synthesize_sft_target,
    )
    from src.policy.lora_policy import LoRAPolicy, LoRAPolicyConfig

    def _write_json(path: str, obj: dict) -> None:
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, default=str)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ckpt_dir = f"/vol/checkpoints/{run_name}_{timestamp}"
    start_epoch = 0
    global_step = 0
    micro = 0
    adapter_path = ""

    if resume_from.strip():
        resume_from = resume_from.strip().rstrip("/")
        base = os.path.basename(resume_from)
        if base in {"latest",} or base.startswith(("step_", "epoch_")):
            ckpt_dir = os.path.dirname(resume_from)
            adapter_path = resume_from
        else:
            ckpt_dir = resume_from
            adapter_path = os.path.join(ckpt_dir, "latest")
            if not os.path.isdir(adapter_path):
                adapter_path = ckpt_dir
        state_path = os.path.join(ckpt_dir, "training_state.json")
        if not os.path.isfile(state_path):
            raise FileNotFoundError(
                f"No training_state.json under {ckpt_dir}; cannot resume."
            )
        with open(state_path) as f:
            train_state = json.load(f)
        start_epoch = int(train_state.get("epoch", 0))
        if train_state.get("epoch_finished"):
            start_epoch += 1
        global_step = int(train_state.get("global_step", 0))
        micro = int(train_state.get("micro", 0))
        run_name = str(train_state.get("run_name", run_name))
        print(
            f">>> Resuming from {adapter_path} "
            f"(epoch {start_epoch}/{epochs}, step {global_step})"
        )
    else:
        train_state = {
            "run_name": run_name,
            "ckpt_dir": ckpt_dir,
            "created_at": timestamp,
            "epoch": 0,
            "global_step": 0,
            "micro": 0,
            "epoch_finished": False,
            "epochs_total": epochs,
            "last_checkpoint": None,
        }

    os.makedirs(ckpt_dir, exist_ok=True)

    def _persist_checkpoint(tag: str, *, epoch_finished: bool) -> None:
        nonlocal train_state, epoch, global_step, micro
        train_state["epoch"] = epoch
        train_state["global_step"] = global_step
        train_state["micro"] = micro
        train_state["epoch_finished"] = epoch_finished
        train_state["last_checkpoint"] = tag
        for sub in (tag, "latest"):
            subdir = os.path.join(ckpt_dir, sub)
            os.makedirs(subdir, exist_ok=True)
            policy.save_adapter(subdir)
        _write_json(os.path.join(ckpt_dir, "training_state.json"), train_state)
        _write_json(
            os.path.join(ckpt_dir, "train_log.json"),
            {"rows": log, "config": train_config, "dataset_summary": summary},
        )
        print(f">>> Checkpoint saved: {os.path.join(ckpt_dir, tag)}/ (+ latest/)")
        volume.commit()

    print(">>> Loading SFT dataset (min_reward=", min_reward,
          ", data_path=", data_path, ")")
    if data_path.endswith(".jsonl"):
        examples = load_sft_examples_from_jsonl(data_path, min_reward=min_reward)
    else:
        examples = load_sft_examples_from_directory(data_path, min_reward=min_reward)
    summary = summarize_sft_dataset(examples)
    print(">>> Dataset summary:", summary)
    if not examples:
        raise RuntimeError(
            "No SFT examples after filtering — check that "
            f"`{data_path}` exists and that the SFT-gen app produced "
            "successful trajectories. Try lowering --min-reward."
        )

    _targets = [t.strip() for t in lora_target_modules.split(",") if t.strip()]
    print(f">>> LoRA arch: rank={lora_rank} target_modules={_targets}")
    policy = LoRAPolicy(LoRAPolicyConfig(
        cache_dir="/vol/hf_cache",
        lora_r=lora_rank,
        lora_alpha=2 * lora_rank,
        lora_target_modules=_targets,
    ))
    if resume_from.strip():
        policy.load_adapter(adapter_path)
    tokenizer = policy.tokenizer

    print(">>> Tokenizing", len(examples), "examples")
    rows: list[dict] = []
    n_truncated_prompts = 0
    for ex in examples:
        prompt_ids = tokenizer(ex.prompt, add_special_tokens=False).input_ids
        target_str = synthesize_sft_target(ex.action) + tokenizer.eos_token
        action_ids = tokenizer(target_str, add_special_tokens=False).input_ids
        if len(prompt_ids) + len(action_ids) > max_seq_len:
            keep = max(0, max_seq_len - len(action_ids))
            prompt_ids = prompt_ids[-keep:]
            n_truncated_prompts += 1
        rows.append({
            "input_ids": prompt_ids + action_ids,
            "n_prompt": len(prompt_ids),
        })
    if n_truncated_prompts:
        print(f">>> WARNING: {n_truncated_prompts}/{len(rows)} prompts were "
              f"left-truncated to fit max_seq_len={max_seq_len}.")

    trainable = list(policy.trainable_parameters())
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    device = next(policy.model.parameters()).device

    log: list[dict] = []
    if resume_from.strip():
        log_path = os.path.join(ckpt_dir, "train_log.json")
        if os.path.isfile(log_path):
            with open(log_path) as f:
                log = json.load(f).get("rows", [])
    train_config = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "min_reward": min_reward,
        "max_seq_len": max_seq_len,
        "grad_accum": grad_accum,
        "log_every": log_every,
        "save_every_steps": save_every_steps,
        "run_name": run_name,
        "data_path": data_path,
        "lora_rank": lora_rank,
        "lora_target_modules": _targets,
        "resume_from": resume_from.strip() or None,
    }
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    if start_epoch >= epochs:
        print(f">>> Already completed {epochs} epochs; saving final manifest.")
        policy.save_adapter(ckpt_dir)
        volume.commit()
        return {"ckpt_dir": ckpt_dir, "global_steps": global_step, "resumed": True}

    for epoch in range(start_epoch, epochs):
        order = torch.randperm(len(rows)).tolist()
        ep_loss_sum, ep_tok_sum = 0.0, 0
        for ri in order:
            r = rows[ri]
            ids = torch.tensor([r["input_ids"]], dtype=torch.long, device=device)
            attn = torch.ones_like(ids)
            labels = ids.clone()
            labels[:, : r["n_prompt"]] = -100
            logits = policy.model(ids, attention_mask=attn).logits.to(torch.float32)
            shift_lg = logits[:, :-1, :].contiguous()
            shift_lb = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_lg.view(-1, shift_lg.size(-1)),
                shift_lb.view(-1),
                ignore_index=-100,
            )
            n_tok = int((shift_lb != -100).sum().item())
            (loss / max(1, grad_accum)).backward()
            micro += 1
            if micro % grad_accum == 0:
                gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % log_every == 0:
                    row_log = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "loss": round(float(loss.detach().item()), 4),
                        "grad_norm": round(float(gn), 4),
                        "n_action_tokens": n_tok,
                        "elapsed_s": round(time.time() - t0, 1),
                    }
                    log.append(row_log)
                    print(
                        f"ep={epoch} step={global_step:04d} "
                        f"loss={row_log['loss']:.4f} "
                        f"gn={row_log['grad_norm']:.3f} "
                        f"t={row_log['elapsed_s']}s"
                    )
                if (
                    save_every_steps > 0
                    and global_step > 0
                    and global_step % save_every_steps == 0
                ):
                    _persist_checkpoint(
                        f"step_{global_step:05d}",
                        epoch_finished=False,
                    )
            ep_loss_sum += float(loss.detach().item()) * n_tok
            ep_tok_sum += n_tok

        per_tok = ep_loss_sum / max(1, ep_tok_sum)
        print(f">>> END epoch={epoch} per_token_ce={per_tok:.4f}")
        log.append({
            "epoch": epoch,
            "epoch_summary": True,
            "per_token_ce": round(per_tok, 4),
            "n_action_tokens": ep_tok_sum,
        })
        _persist_checkpoint(f"epoch_{epoch}", epoch_finished=True)

    if micro % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

    print(">>> Saving final adapter to", ckpt_dir)
    policy.save_adapter(ckpt_dir)
    train_state["epoch"] = epochs
    train_state["global_step"] = global_step
    train_state["epoch_finished"] = True
    train_state["last_checkpoint"] = "final"
    train_config["n_truncated_prompts"] = n_truncated_prompts
    _write_json(os.path.join(ckpt_dir, "training_state.json"), train_state)
    _write_json(
        os.path.join(ckpt_dir, "train_log.json"),
        {"rows": log, "config": train_config, "dataset_summary": summary},
    )

    final = {
        "ckpt_dir": ckpt_dir,
        "latest_checkpoint": os.path.join(ckpt_dir, "latest"),
        "n_examples": len(rows),
        "epochs": epochs,
        "global_steps": global_step,
        "final_log_row": log[-1] if log else None,
        "total_elapsed_s": round(time.time() - t0, 1),
        "n_truncated_prompts": n_truncated_prompts,
        "resumed": bool(resume_from.strip()),
    }
    with open(os.path.join(ckpt_dir, "summary.json"), "w") as f:
        json.dump(final, f, indent=2, default=str)
    volume.commit()
    return final


@app.local_entrypoint()
def main(
    epochs: int = 3,
    learning_rate: float = 1e-4,
    min_reward: float = 0.5,
    max_seq_len: int = 1024,
    grad_accum: int = 4,
    log_every: int = 25,
    save_every_steps: int = 500,
    run_name: str = "sft_v1",
    data_path: str = DEFAULT_SFT_DATA_PATH,
    lora_rank: int = 16,
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj",
    resume_from: str = "",
    wait: bool = False,
) -> None:
    """Local entrypoint. Prefer `modal run --detach ...::sft_train` for long jobs."""
    import json as _json

    kwargs = dict(
        epochs=epochs,
        learning_rate=learning_rate,
        min_reward=min_reward,
        max_seq_len=max_seq_len,
        grad_accum=grad_accum,
        log_every=log_every,
        save_every_steps=save_every_steps,
        run_name=run_name,
        data_path=data_path,
        lora_rank=lora_rank,
        lora_target_modules=lora_target_modules,
        resume_from=resume_from,
    )
    if wait:
        res = sft_train.remote(**kwargs)
        print(_json.dumps(res, indent=2, default=str))
        return

    call = sft_train.spawn(**kwargs)
    print(_json.dumps({
        "submitted": True,
        "function_call_id": call.object_id,
        "run_name": run_name,
        "hint": "Use: modal run --detach infra/app_sft_train.py::sft_train ...",
    }, indent=2))
