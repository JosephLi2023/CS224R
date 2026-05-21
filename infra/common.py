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

# Env vars controlling whether we attach the OpenAI Secret. For the current
# Max milestone pipeline we do not use the OpenAI judge backend, so the stable
# default is to SKIP attaching the secret unless a caller explicitly opts in.
OPENAI_SECRET_OPT_OUT_ENV = "CS224R_SKIP_OPENAI_SECRET"
OPENAI_SECRET_OPT_IN_ENV = "CS224R_USE_OPENAI_SECRET"

# The Modal Secret name and the env-var key it injects into the container.
OPENAI_SECRET_NAME = "openai-secret"
OPENAI_SECRET_REQUIRED_KEYS = ["OPENAI_API_KEY"]


def maybe_openai_secret() -> list[modal.Secret]:
    """Return `[Secret]` referencing `openai-secret`, or `[]` by default.

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

    The current default is to SKIP attaching the secret so local submit and
    remote container import stay consistent for non-OpenAI training paths.
    To opt in explicitly, set `CS224R_USE_OPENAI_SECRET=1` before invoking
    `modal run`/`modal deploy`. `CS224R_SKIP_OPENAI_SECRET=1` still forces
    the secret off even if the opt-in env var is present.
    """
    opt_in = os.getenv(OPENAI_SECRET_OPT_IN_ENV, "0") == "1"
    opt_out = os.getenv(OPENAI_SECRET_OPT_OUT_ENV, "0") == "1"
    if (not opt_in) or opt_out:
        _logger.warning(
            "[infra.common] skipping openai-secret attachment "
            "(%s=%s, %s=%s). judge.backend=openai will fail at runtime if reached.",
            OPENAI_SECRET_OPT_IN_ENV,
            int(opt_in),
            OPENAI_SECRET_OPT_OUT_ENV,
            int(opt_out),
        )
        return []
    return [
        modal.Secret.from_name(
            OPENAI_SECRET_NAME, required_keys=OPENAI_SECRET_REQUIRED_KEYS
        )
    ]
