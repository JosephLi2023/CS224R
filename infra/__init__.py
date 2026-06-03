"""Modal apps for the CS224R H-GRPO project.

Layout:
  infra/image.py              - shared Image with torch/transformers/peft/vllm/trl
  infra/common.py             - shared Volume + Secrets used by every app
  infra/app_webshop_install.py / app_alfworld_install.py - install env data
  infra/app_webshop_sft_gen.py / app_alfworld_sft_gen.py - generate SFT data
  infra/app_sft_train.py / app_sft_train_alfworld.py     - SFT training
  infra/app_train_loop.py     - eval + RL train loop
  infra/app_train_turnrd.py   - standalone TurnRD fit
  infra/app_orchestrator.py   - env-agnostic round orchestrator

Note: this package is named `infra` (not `modal`) to avoid colliding with the
PyPI `modal` library. Modal CLI invocation uses the file path:

  modal run infra/app_train_loop.py --env-name webshop --n-episodes 50
"""
