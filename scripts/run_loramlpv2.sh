#!/usr/bin/env bash
# Launch the loraMLPv2 bake-off (MLP-LoRA + rank=32 + LR=5e-6).
#
# Prereq: Modal auth + `cs224r-hgpo-vol` volume + SFT adapter at
#   /vol/checkpoints/sft_alfworld_v1_20260507_165617 on your account.
#
# Usage:  ./scripts/run_loramlpv2.sh [SEED]
# Example: ./scripts/run_loramlpv2.sh 22
set -e

SEED="${1:-11}"
cd "$(dirname "$0")/.."

nohup python scripts/run_turnrd_modal.py \
  --config configs/TurnRDV2_alfworld_a050_loraMLPv2.json \
  --env-name alfworld --rounds 5 --episodes-per-round 40 \
  --turnrd-epochs 3 --seed "$SEED" \
  --sft-adapter /vol/checkpoints/sft_alfworld_v1_20260507_165617 \
  --replay-path /vol/cache/TurnRDV2_alfworld_a050_loraMLPv2/replay.jsonl \
  --ckpt-path /vol/cache/TurnRDV2_alfworld_a050_loraMLPv2/ckpt.pt \
  --eval-episodes 100 \
  --run-name-prefix TurnRDV2_a050_loraMLPv2 \
  --carry-policy-across-rounds \
  --adapter-dir /vol/checkpoints \
  > "/tmp/turnrd_loraMLPv2_seed${SEED}.log" 2>&1 &
disown $!
echo "Launched loraMLPv2 (seed=$SEED). Log: /tmp/turnrd_loraMLPv2_seed${SEED}.log"
echo "Monitor: grep 'Eval done' /tmp/turnrd_loraMLPv2_seed${SEED}.log"
