"""Modal A100 app: SFT warm-start of Qwen2.5-1.5B + LoRA on WebShop human trajectories.

  modal run --detach infra/app_sft_train.py --epochs 3 --min-reward 0.5

Tokenizes (prompt, action) SFT examples loaded via src.datasets.sft_webshop,
runs masked cross-entropy (only action tokens contribute to the loss),
saves the LoRA adapter to /vol/checkpoints/<run_name>_<ts>/. The adapter
can then be loaded by infra/app_train_loop.py via
LoRAPolicy.load_adapter(...) to seed GRPO with a non-trivial init.

Cost: ~$0.50 for 3 epochs over ~745 examples on A100.
"""
from __future__ import annotations

import modal  # type: ignore[import-not-found]

from infra.common import VOLUME_MOUNT, volume
from infra.image import image

app = modal.App("cs224r-hgpo-sft-train")


# Legacy default data path. Directory of upstream WebShop gdown human
# trajectories — consumed by `load_sft_examples_from_directory`. The
# new oracle-gen single-JSONL path is
# `/vol/data/webshop/oracle_trajs.jsonl`; dispatch happens on suffix
# in the loader site below.
DEFAULT_SFT_DATA_PATH = "/vol/data/webshop/human_trajs"


@app.function(image=image, gpu="A100-80GB", volumes={VOLUME_MOUNT: volume}, timeout=4 * 60 * 60)
def sft_train(
    epochs: int = 3,
    learning_rate: float = 1e-4,
    min_reward: float = 0.5,
    max_seq_len: int = 1024,
    grad_accum: int = 4,
    log_every: int = 25,
    run_name: str = "sft_v1",
    data_path: str = DEFAULT_SFT_DATA_PATH,
    lora_rank: int = 16,
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj",
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

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ckpt_dir = f"/vol/checkpoints/{run_name}_{timestamp}"
    os.makedirs(ckpt_dir, exist_ok=True)

    # Dispatch loader on the data path's suffix. A single `.jsonl` file
    # is the AlfWorld-style pre-rendered format produced by the new
    # `infra/app_webshop_sft_gen.py` oracle generator; a directory is
    # the legacy upstream-gdown human-trajs layout consumed by
    # `load_sft_examples_from_directory`. Default keeps legacy behavior
    # byte-identical when `--data-path` is not passed.
    print(">>> Loading SFT dataset (min_reward=", min_reward,
          ", data_path=", data_path, ")")
    if data_path.endswith(".jsonl"):
        examples = load_sft_examples_from_jsonl(
            data_path, min_reward=min_reward
        )
    else:
        examples = load_sft_examples_from_directory(
            data_path, min_reward=min_reward
        )
    summary = summarize_sft_dataset(examples)
    print(">>> Dataset summary:", summary)
    if not examples:
        raise RuntimeError(
            "No SFT examples after filtering — check that "
            f"`{data_path}` exists and that the SFT-gen app produced "
            "successful trajectories. Try lowering --min-reward."
        )

    # LoRA arch knobs propagated from CLI. Defaults preserve v1
    # behavior (rank 16, attention-only). The 2:1 alpha:rank convention
    # is kept in lockstep with `infra/app_sft_train_alfworld.py` so the
    # two trainers stay binary-compatible against the same
    # `LoRAPolicyConfig` dataclass defaults (16/32) and the
    # `infra/app_train_loop.py` plumb-through.
    _targets = [t.strip() for t in lora_target_modules.split(",") if t.strip()]
    print(f">>> LoRA arch: rank={lora_rank} target_modules={_targets}")
    policy = LoRAPolicy(LoRAPolicyConfig(
        cache_dir="/vol/hf_cache",
        lora_r=lora_rank,
        lora_alpha=2 * lora_rank,
        lora_target_modules=_targets,
    ))
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
              f"left-truncated to fit max_seq_len={max_seq_len}. WebShop "
              "item-page observations grow with option count; if this "
              "is >10% bump --max-seq-len to 2048.")

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
            "lora_rank": lora_rank,
            "lora_target_modules": _targets,
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
    run_name: str = "sft_v1",
    data_path: str = DEFAULT_SFT_DATA_PATH,
    lora_rank: int = 16,
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj",
) -> None:
    import json as _json
    res = sft_train.remote(
        epochs=epochs, learning_rate=learning_rate, min_reward=min_reward,
        max_seq_len=max_seq_len, grad_accum=grad_accum,
        log_every=log_every, run_name=run_name, data_path=data_path,
        lora_rank=lora_rank, lora_target_modules=lora_target_modules,
    )
    print(_json.dumps(res, indent=2, default=str))
