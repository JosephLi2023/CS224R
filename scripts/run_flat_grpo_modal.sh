#!/usr/bin/env bash
# Launch flat-GRPO on Modal in the same 5-round x 40-ep protocol shape that
# method_b_lean (turnRD) uses, so per-round eval blocks are directly
# comparable. Same base task offset (seed * rounds * eps_per_round = 11*5*40
# = 2200), same eval task range [6500, 6550), same SFT warm-start, same K=4.
#
# Each call is foreground (no --detach) so this script blocks until all 5
# rounds finish. Run under nohup if you want it to outlive your shell:
#   nohup bash scripts/run_flat_grpo_modal.sh > /tmp/flat_grpo_seed11.log 2>&1 &
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

BASE_OFFSET=$(( SEED * ROUNDS * EPS_PER_ROUND ))   # = 2200 for seed=11

for r in $(seq 0 $((ROUNDS - 1))); do
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

echo "=== flat_grpo seed=${SEED} sweep complete ==="
