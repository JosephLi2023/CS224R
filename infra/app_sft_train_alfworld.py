"""Modal A100 app: SFT warm-start of Qwen2.5-1.5B + LoRA on ALFWorld expert trajectories.

  modal run --detach infra/app_sft_train_alfworld.py --epochs 3 --min-reward 0.5

Tokenizes (prompt, action) SFT examples loaded via
`src.datasets.sft_alfworld.load_sft_examples_from_jsonl`, runs masked
cross-entropy (only action tokens contribute to the loss), saves the
LoRA adapter to /vol/checkpoints/<run_name>_<ts>/. The adapter can
then be loaded by `infra/app_train_loop.py::train_loop_alfworld` via
`LoRAPolicy.load_adapter(...)` to seed GRPO with a non-trivial init.

This is a near line-for-line clone of `infra/app_sft_train.py` with:
  - dataset import: `src.datasets.sft_alfworld` instead of `sft_webshop`
  - data path: `/vol/data/alfworld/sft_trajs.jsonl` (single JSONL) instead
    of `/vol/data/webshop/human_trajs/` (directory of JSONLs)
  - loader: `load_sft_examples_from_jsonl(path)` instead of
    `load_sft_examples_from_directory(dir)`
  - default `run_name = "sft_alfworld_v1"`
  - image: `image` (NOT `alfworld_image`); the trainer doesn't need
    AlfWorld at runtime — the SFT data file is fully pre-rendered by
    the SFT-gen app, so the trainer is purely a tokenize+CE loop.

Cost: ~$0.50, ~30 min for 3 epochs over ~2000+ examples on A100.
"""
from __future__ import annotations

import modal  # type: ignore[import-not-found]

from infra.common import VOLUME_MOUNT, volume
from infra.image import image

app = modal.App("cs224r-hgpo-sft-train-alfworld")


# Default location of the AlfWorld SFT JSONL on the shared Volume —
# matches `infra/app_alfworld_sft_gen.py::SFT_OUTPUT_PATH`.
DEFAULT_SFT_DATA_PATH = "/vol/data/alfworld/sft_trajs.jsonl"


@app.function(image=image, gpu="A100-80GB", volumes={VOLUME_MOUNT: volume}, timeout=60 * 60)
def sft_train(
    epochs: int = 3,
    learning_rate: float = 1e-4,
    min_reward: float = 0.5,
    max_seq_len: int = 1024,
    grad_accum: int = 4,
    log_every: int = 25,
    run_name: str = "sft_alfworld_v1",
    data_path: str = DEFAULT_SFT_DATA_PATH,
) -> dict:
    import json
    import os
    import sys
    import time
    from datetime import datetime, timezone

    sys.path.insert(0, "/workspace")

    import torch
    from torch.nn import functional as F

    from src.datasets.sft_alfworld import (
        load_sft_examples_from_jsonl,
        summarize_sft_dataset,
        synthesize_sft_target,
    )
    from src.policy.lora_policy import LoRAPolicy, LoRAPolicyConfig

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ckpt_dir = f"/vol/checkpoints/{run_name}_{timestamp}"
    os.makedirs(ckpt_dir, exist_ok=True)

    print(">>> Loading SFT dataset (min_reward=", min_reward, ")")
    examples = load_sft_examples_from_jsonl(data_path, min_reward=min_reward)
    summary = summarize_sft_dataset(examples)
    print(">>> Dataset summary:", summary)
    if not examples:
        raise RuntimeError(
            "No SFT examples after filtering — check that "
            f"`{data_path}` exists and that the SFT-gen app produced "
            "successful (won) trajectories. Try lowering --min-reward."
        )

    print(">>> Loading LoRAPolicy")
    policy = LoRAPolicy(LoRAPolicyConfig(cache_dir="/vol/hf_cache"))
    tokenizer = policy.tokenizer

    print(">>> Tokenizing", len(examples), "examples")
    rows: list[dict] = []
    n_truncated_prompts = 0
    for ex in examples:
        prompt_ids = tokenizer(ex.prompt, add_special_tokens=False).input_ids
        # Target is the full ReAct emission ` <thought>\nAction: <body>` so
        # the SFT model learns to produce Thought + Action together — the
        # exact format the runtime ReAct loop expects.
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
              f"left-truncated to fit max_seq_len={max_seq_len}. AlfWorld "
              "observations grow with room size; if this is >10% bump "
              "--max-seq-len to 2048.")

    trainable = list(policy.trainable_parameters())
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    device = next(policy.model.parameters()).device

    log: list[dict] = []
    t0 = time.time()
    global_step = 0
    micro = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(epochs):
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
            ep_loss_sum += float(loss.detach().item()) * n_tok
            ep_tok_sum += n_tok

        per_tok = ep_loss_sum / max(1, ep_tok_sum)
        print(f">>> END epoch={epoch} per_token_ce={per_tok:.4f}")
        log.append({"epoch": epoch, "epoch_summary": True,
                    "per_token_ce": round(per_tok, 4),
                    "n_action_tokens": ep_tok_sum})

    if micro % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

    print(">>> Saving adapter to", ckpt_dir)
    policy.save_adapter(ckpt_dir)
    with open(os.path.join(ckpt_dir, "train_log.json"), "w") as f:
        json.dump({"rows": log, "config": {
            "epochs": epochs, "learning_rate": learning_rate,
            "min_reward": min_reward, "max_seq_len": max_seq_len,
            "grad_accum": grad_accum, "log_every": log_every,
            "run_name": run_name, "data_path": data_path,
            "n_truncated_prompts": n_truncated_prompts,
        }, "dataset_summary": summary}, f, indent=2, default=str)

    final = {
        "ckpt_dir": ckpt_dir,
        "n_examples": len(rows),
        "epochs": epochs,
        "global_steps": global_step,
        "final_log_row": log[-1] if log else None,
        "total_elapsed_s": round(time.time() - t0, 1),
        "n_truncated_prompts": n_truncated_prompts,
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
    run_name: str = "sft_alfworld_v1",
    data_path: str = DEFAULT_SFT_DATA_PATH,
) -> None:
    import json as _json
    res = sft_train.remote(
        epochs=epochs, learning_rate=learning_rate, min_reward=min_reward,
        max_seq_len=max_seq_len, grad_accum=grad_accum,
        log_every=log_every, run_name=run_name, data_path=data_path,
    )
    print(_json.dumps(res, indent=2, default=str))
