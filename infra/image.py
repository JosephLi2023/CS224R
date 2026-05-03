"""Shared Modal Image for CS224R H-GRPO.

Built once and cached by Modal; all apps import `image` from here so we get
a single rebuild whenever a dep changes.

Heavy stack pinned to 2025/early-2026 versions known to play well together:
- torch 2.4.1
- transformers 4.45.2
- peft 0.13.2
- trl 0.11.4
- vllm 0.6.3.post1
"""

from __future__ import annotations

import modal  # type: ignore[import-not-found]

PYTHON_VERSION = "3.11"

# Heavy ML deps — see also requirements/modal.txt for the canonical list.
_PIP_PACKAGES = [
    # Core deep learning
    "torch==2.4.0",
    "transformers==4.45.2",
    "accelerate==0.34.2",
    "datasets==3.0.1",
    "peft==0.13.2",
    "trl==0.11.4",
    # Inference
    "vllm==0.6.3.post1",
    # Judge backends
    "openai>=1.40",
    "fastapi>=0.115",
    "uvicorn>=0.30",
    "httpx>=0.27",
    # Utilities
    "pydantic>=2.6",
    "sentencepiece>=0.2",
    "tiktoken>=0.7",
    "numpy>=1.26",
    "PyYAML>=6.0",
    "tqdm>=4.66",
    "matplotlib>=3.9",
    # Data acquisition (used by infra/app_data.py to pull WebShop JSONs)
    "gdown>=5.2",
]

# WebShop install — bake from a pinned commit. Kept as a separate layer so
# Modal cache hits even when we tweak the python deps above.
# TODO Day 1: pin actual commit; this is a placeholder while we verify the
# install works.
_WEBSHOP_INSTALL = [
    "git",
    "build-essential",
]

image: modal.Image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install(*_WEBSHOP_INSTALL)
    .pip_install(*_PIP_PACKAGES)
    # Expose /workspace on PYTHONPATH so `from src.* import ...` and
    # `from infra.* import ...` both work inside the container (Modal CLI
    # copies the entrypoint file to /root/ so we can't rely on the file's
    # parent dir). Also point HuggingFace's cache at the shared Volume so
    # Qwen2.5-1.5B downloads (~3 GB) survive container restarts. Must come
    # BEFORE add_local_dir per Modal's image rules.
    .env({"PYTHONPATH": "/workspace", "HF_HOME": "/vol/hf_cache"})
    # add_local_* must be the last step in the image chain.
    .add_local_dir(
        local_path=".",
        remote_path="/workspace",
        ignore=[
            ".git",
            ".venv",
            "**/__pycache__",
            "experiments/manifests",  # large run artifacts, served from Volume instead
            "*.pdf",
        ],
    )
)
