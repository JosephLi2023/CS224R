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

import logging
import os

import modal  # type: ignore[import-not-found]

_logger = logging.getLogger(__name__)

# Single shared Volume for all apps. `create_if_missing=True` so first deploy
# auto-creates it.
VOLUME_NAME = "cs224r-hgpo-vol"
volume: modal.Volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Mount point inside containers.
VOLUME_MOUNT = "/vol"

# Env var to opt OUT of attaching the OpenAI Secret. Set to "1" when you
# haven't created `openai-secret` yet but still want to deploy the
# baseline / vLLM-judge path. Default behavior includes the secret so
# Method A (judge.backend=openai) just works.
OPENAI_SECRET_OPT_OUT_ENV = "CS224R_SKIP_OPENAI_SECRET"

# The Modal Secret name and the env-var key it injects into the container.
OPENAI_SECRET_NAME = "openai-secret"
OPENAI_SECRET_REQUIRED_KEYS = ["OPENAI_API_KEY"]


def maybe_openai_secret() -> list[modal.Secret]:
    """Return `[Secret]` referencing `openai-secret`, or `[]` if opted out.

    Important: `modal.Secret.from_name(...)` is **lazy** — it returns a
    reference and never raises here even if the secret is missing on the
    workspace. The actual lookup happens at function-deploy / invocation
    time, which means a missing secret surfaces as Modal's
    `NotFoundError` ("Secret 'openai-secret' not found …") with a clear
    message — not a silent failure inside `OpenAIJudge.score_turns`.

    `required_keys=["OPENAI_API_KEY"]` makes Modal additionally validate
    that the secret payload contains the expected key, so a misnamed
    secret (e.g. user typed `OPENAI_KEY` instead of `OPENAI_API_KEY`)
    fails fast at deploy with a clear error.

    To deploy WITHOUT attaching the secret (e.g. for the baseline path
    or while the key isn't provisioned yet), set the env var
    `CS224R_SKIP_OPENAI_SECRET=1` before invoking `modal run`/`modal deploy`.
    """
    if os.getenv(OPENAI_SECRET_OPT_OUT_ENV, "0") == "1":
        _logger.warning(
            "[infra.common] %s=1 set; skipping openai-secret attachment. "
            "judge.backend=openai will fail at runtime if reached.",
            OPENAI_SECRET_OPT_OUT_ENV,
        )
        return []
    return [
        modal.Secret.from_name(
            OPENAI_SECRET_NAME, required_keys=OPENAI_SECRET_REQUIRED_KEYS
        )
    ]
