"""Shared Modal Images for CS224R H-GRPO.

Two images, both built off the same heavy ML base:
- `image`         — trainer + policy + judge (default).
- `webshop_image` — same base + Java JDK + WebShop env runtime deps
                    (pyserini, spaCy, rank_bm25, Flask, gym, ...). Used by
                    `infra/app_webshop_install.py` and any app that
                    instantiates `WebAgentTextEnv` directly.

WebShop's own `requirements.txt` pins legacy torch/transformers/numpy that
would clobber our modern stack — we install its EXTRA deps explicitly here
and `pip install -e webshop --no-deps` at runtime in the install app.
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

# WebShop runtime deps — installed ON TOP of the modern stack. We
# deliberately pin only what WebShop genuinely needs at env-reset/step
# time and leave torch/transformers/numpy at our modern versions.
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


def _make_base_image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .apt_install(*_BASE_APT)
        .pip_install(*_PIP_PACKAGES)
    )


def _add_workspace(img: modal.Image) -> modal.Image:
    return (
        img.env({"PYTHONPATH": "/workspace", "HF_HOME": "/vol/hf_cache"})
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
