"""Shared Modal Images for CS224R H-GRPO, all built off one heavy ML base:
`image` (trainer/policy/judge), `webshop_image` (+ Java/pyserini/spaCy), and
`alfworld_image` (+ AlfWorld/TextWorld). WebShop's legacy pins are kept out by
installing only its extra deps here and `pip install -e --no-deps` at runtime.
"""
from __future__ import annotations

import modal  # type: ignore[import-not-found]

PYTHON_VERSION = "3.11"

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
    # Data acquisition
    "gdown>=5.2",
]

_BASE_APT = ["git", "build-essential"]

# WebShop runtime deps on top of the modern stack (torch/transformers/numpy
# stay at our versions).
_WEBSHOP_PIP_EXTRAS = [
    "pyserini==0.22.1",        # newer than WebShop's 0.17.0; works with Java 21
    "faiss-cpu>=1.8",          # pyserini imports faiss at module load
    "rank_bm25==0.2.2",
    "beautifulsoup4>=4.11",
    "Flask>=2.2",
    "gym==0.24.0",
    "spacy>=3.7,<3.8",
    "thefuzz>=0.20",
    "cleantext==1.1.4",
    "rich>=12",
    "scikit-learn>=1.3",
    "selenium==4.2.0",         # web_agent_site/envs/__init__.py imports WebAgentSiteEnv which uses selenium
]

_WEBSHOP_APT = ["default-jdk", "default-jre"]


# AlfWorld runtime deps (alfworld + textworld + gym; no Java/pyserini/spaCy).
_ALFWORLD_PIP_EXTRAS = [
    "alfworld>=0.3.5",
    "textworld>=1.6",
    # Pin gym to WebShop's version so the two images stay compatible.
    "gym==0.24.0",
]

_ALFWORLD_APT = ["libffi-dev"]

# Where the install app downloads ALFWorld data (matches configs'
# `$ALFWORLD_DATA` interpolation).
ALFWORLD_DATA_DIR = "/vol/data/alfworld"


def _make_base_image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .apt_install(*_BASE_APT)
        .pip_install(*_PIP_PACKAGES)
    )


def _add_workspace(img: modal.Image) -> modal.Image:
    return (
        img.env({
            "PYTHONPATH": "/workspace",
            "HF_HOME": "/vol/hf_cache",
            # Use PyTorch's expandable-segments allocator so mem_get_info()
            # reports OS-level free memory honestly across empty_cache cycles
            # (works around vLLM 0.6.3.post1's `peak_memory > 0` profile assert
            # on the train->eval handoff).
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        })
        .add_local_dir(
            local_path=".",
            remote_path="/workspace",
            ignore=[".git", ".venv", "**/__pycache__", "experiments/manifests", "*.pdf"],
        )
    )


# Public images.
image: modal.Image = _add_workspace(_make_base_image())

webshop_image: modal.Image = _add_workspace(
    _make_base_image()
    .apt_install(*_WEBSHOP_APT)
    .pip_install(*_WEBSHOP_PIP_EXTRAS)
    .run_commands(
        "python -m spacy download en_core_web_sm",
        "python -m spacy download en_core_web_lg",
    )
)

alfworld_image: modal.Image = _add_workspace(
    _make_base_image()
    .apt_install(*_ALFWORLD_APT)
    .pip_install(*_ALFWORLD_PIP_EXTRAS)
    .env({"ALFWORLD_DATA": ALFWORLD_DATA_DIR})
)
