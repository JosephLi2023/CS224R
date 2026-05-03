#!/usr/bin/env bash
set -euo pipefail

# Local fallback command; in Modal, wrap this in your modal entrypoint.
PYTHONPATH=. python3 -m src.trainers.train \
  --env-config configs/env_toy.json \
  --train-config "${1:-configs/baseline_train.json}" \
  --eval-config configs/eval.json
