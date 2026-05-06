#!/usr/bin/env bash
# Launch the CounterFactualDecomposer (Method D) on Modal in the same
# 5-round × 40-episode protocol shape that flat_grpo and Method-B (TurnRD)
# use, so per-round eval blocks are directly comparable to the existing
# 4method_comparison.txt results.
#
# CF re-samples N=2 alt actions per turn from the policy and completes
# short greedy rollouts on the SAME vLLM runner the collector uses.
# Expect ~3-5× wall-clock per group vs flat_grpo because of the extra
# env steps + alt-action LLM calls (see counterfactual.py module
# docstring "Cost" section). Single round = ~3 GPU-hr; 5 rounds = ~15
# GPU-hr; budget ≈ $50 on A100-80GB.
#
# Same base task offset (seed * rounds * eps_per_round = 11*5*40 = 2200),
# same eval task range [6500, 6550), same SFT warm-start, same K=4
# trajectories so per-method comparison is apples-to-apples.
#
# Each call is foreground (no --detach) so this script blocks until all
# 5 rounds finish. Run under nohup if you want it to outlive your shell:
#   nohup bash scripts/run_counterfactual_modal.sh > /tmp/cf_seed11.log 2>&1 &
set -euo pipefail

cd "$(dirname "$0")/.."

SEED=11
ROUNDS=5
EPS_PER_ROUND=40
K=4
MAX_TURNS=6
CONFIG="configs/method_hgpo_counterfactual.json"
SFT_ADAPTER="/vol/checkpoints/sft_v3_20260504_154752"
RUN_PREFIX="cf_compare_seed${SEED}"
EVAL_EPS=50
EVAL_BASE=6500

BASE_OFFSET=$(( SEED * ROUNDS * EPS_PER_ROUND ))   # = 2200 for seed=11

for r in $(seq 0 $((ROUNDS - 1))); do
  OFFSET=$(( BASE_OFFSET + r * EPS_PER_ROUND ))
  printf '\n=== counterfactual round %02d (task_offset=%d) ===\n' "$r" "$OFFSET"
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

echo "=== counterfactual seed=${SEED} sweep complete ==="
