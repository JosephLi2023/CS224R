#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <train_config> <checkpoint_path>"
  exit 1
fi

TRAIN_CONFIG="$1"
CHECKPOINT="$2"

PYTHONPATH=. python3 -m src.trainers.eval \
  --env-config configs/env_toy.json \
  --train-config "${TRAIN_CONFIG}" \
  --eval-config configs/eval.json \
  --checkpoint "${CHECKPOINT}"
