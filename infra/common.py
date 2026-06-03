"""Shared Modal infrastructure: the project Volume + OpenAI Secret.

Volume `/vol` holds caches, datasets, checkpoints, manifests, and the HF cache.
"""

from __future__ import annotations

import logging
import os

import modal  # type: ignore[import-not-found]

_logger = logging.getLogger(__name__)

# Single shared Volume for all apps (auto-created on first deploy).
VOLUME_NAME = "cs224r-hgpo-vol"
volume: modal.Volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Mount point inside containers.
VOLUME_MOUNT = "/vol"

# Set to "1" to deploy without attaching the OpenAI Secret (baseline /
# vLLM-judge path); default attaches it for Method A.
OPENAI_SECRET_OPT_OUT_ENV = "CS224R_SKIP_OPENAI_SECRET"

# The Modal Secret name and the env-var key it injects into the container.
OPENAI_SECRET_NAME = "openai-secret"
OPENAI_SECRET_REQUIRED_KEYS = ["OPENAI_API_KEY"]


def maybe_openai_secret() -> list[modal.Secret]:
    """Return `[Secret]` referencing `openai-secret`, or `[]` if opted out.

    `from_name` is lazy, so a missing/misnamed secret fails fast at
    deploy/invocation (not silently inside the judge). Opt out with
    `CS224R_SKIP_OPENAI_SECRET=1`.
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
