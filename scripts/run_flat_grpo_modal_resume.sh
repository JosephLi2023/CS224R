#!/usr/bin/env bash
# Re-launch flat_grpo rounds 3 and 4 (round 2 partially completed earlier;
# round 2 may also be optionally re-run by changing R_START to 2).
# Same protocol as scripts/run_flat_grpo_modal.sh: K=4, max_turns=6,
# eval [6500, 6550), seed 11, base task offset = 2200.
#
# Run under nohup so it outlives the shell:
#   nohup bash scripts/run_flat_grpo_modal_resume.sh > /tmp/flat_grpo_resume.log 2>&1 &
set -euo pipefail

cd "$(dirname "$0")/.."

SEED=11
ROUNDS=5
EPS_PER_ROUND=40
K=4
MAX_TURNS=6
CONFIG="configs/method_flat_grpo.json"
SFT_ADAPTER="/vol/checkpoints/sft_v3_20260504_154752"
RUN_PREFIX="flat_grpo_compare_seed${SEED}"
EVAL_EPS=50
EVAL_BASE=6500

R_START=3
R_END=4    # inclusive
BASE_OFFSET=$(( SEED * ROUNDS * EPS_PER_ROUND ))   # = 2200 for seed=11

for r in $(seq "$R_START" "$R_END"); do
  OFFSET=$(( BASE_OFFSET + r * EPS_PER_ROUND ))
  printf '\n=== flat_grpo round %02d (task_offset=%d) ===\n' "$r" "$OFFSET"
  modal run infra/app_train_loop.py \
    --config "/workspace/${CONFIG}" \
    --n-episodes "$EPS_PER_ROUND" \
    --k "$K" \
    --max-turns "$MAX_TURNS" \
    --task-id-offset "$OFFSET" \
    --run-name "${RUN_PREFIX}_round$(printf '%02d' "$r")" \
    --round-idx "$r" \
    --sft-adapter "$SFT_ADAPTER" \
    --eval-episodes "$EVAL_EPS" \
    --eval-task-id-base "$EVAL_BASE" \
    --gpu-mem-util 0.30
done

echo "=== flat_grpo seed=${SEED} resume (rounds ${R_START}-${R_END}) complete ==="
