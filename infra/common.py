"""Shared Modal infrastructure: Volume + Secrets.

Volume layout:
  /vol/cache/judge.sqlite          — judge cache (shared across runs)
  /vol/cache/turnrd_replay.jsonl   — TurnRD replay buffer
  /vol/data/webshop/               — baked WebShop product index
  /vol/data/alfworld/              — ALFWorld task data
  /vol/checkpoints/<run>/          — per-run LoRA adapter snapshots
  /vol/manifests/<run>/            — train_log.json, eval_log.json, config_snapshot.json
  /vol/hf_cache/                   — huggingface model + tokenizer cache (Qwen weights)
"""

from __future__ import annotations

import modal  # type: ignore[import-not-found]

# Single shared Volume for all apps. `create_if_missing=True` so first deploy
# auto-creates it.
VOLUME_NAME = "cs224r-hgpo-vol"
volume: modal.Volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Mount point inside containers.
VOLUME_MOUNT = "/vol"

# Secrets we'll reference (none required for the baseline path).
# Created via: modal secret create openai-key OPENAI_API_KEY=sk-...
def maybe_openai_secret() -> list[modal.Secret]:
    """Returns [Secret] if OPENAI_API_KEY is configured, else []."""
    try:
        return [modal.Secret.from_name("openai-key")]
    except Exception:
        return []
