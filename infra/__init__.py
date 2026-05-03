"""Modal apps for the CS224R H-GRPO project.

Layout:
  infra/image.py    — shared Image with torch/transformers/peft/vllm/trl
  infra/common.py   — shared Volume + Secrets used by every app
  infra/app_train.py — trainer (A100-80GB)
  infra/app_judge.py — vLLM judge server (A10G), only deployed when needed
  infra/app_eval.py  — eval-only against a checkpoint

Note: this package is named `infra` (not `modal`) to avoid colliding with the
PyPI `modal` library. Modal CLI invocation uses the file path:

  modal deploy infra/app_train.py
  modal run infra/app_train.py::hello   # smoke test, no GPU
  modal run infra/app_train.py::train   # real training, A100
"""
