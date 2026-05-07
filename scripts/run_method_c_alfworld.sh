#!/usr/bin/env bash
# Method C (progress decomposer) sweep on AlfWorld — 5 rounds × 40 episodes.
#
# The orchestrator (`scripts/run_turnrd_modal.py`) requires a turnrd
# config block, so Method C bypasses it via 5 direct `modal run` calls,
# one per round. Each round shares the same WebShop-style task-id math
# the orchestrator would use: `task_id_offset = base + round_idx * 40`.
#
# Run from the repo root. Logs land at /tmp/method_c_alfworld_round{N}.log.
# Each round runs detached on Modal so the local CLI can be killed
# without losing the cloud run.
#
# Usage: bash scripts/run_method_c_alfworld.sh

set -euo pipefail

BASE_TASK_OFFSET=${BASE_TASK_OFFSET:-2200}   # seed 11 × 5 × 40 = 2200 (matches B's seed-11 base)
EPS_PER_ROUND=${EPS_PER_ROUND:-40}
EVAL_EPS=${EVAL_EPS:-50}
EVAL_TASK_BASE=${EVAL_TASK_BASE:-6500}
N_ROUNDS=${N_ROUNDS:-5}
RUN_PREFIX=${RUN_PREFIX:-method_c_alfworld_seed11}
CONFIG=${CONFIG:-configs/method_hgpo_progress_alfworld.json}

for round_idx in $(seq 0 $((N_ROUNDS - 1))); do
    task_offset=$((BASE_TASK_OFFSET + round_idx * EPS_PER_ROUND))
    run_name="${RUN_PREFIX}_round$(printf '%02d' "$round_idx")"
    log_path="/tmp/${run_name}.log"
    echo "═══ Round ${round_idx} ═══"
    echo "  task_id_offset = ${task_offset}"
    echo "  run_name       = ${run_name}"
    echo "  log            = ${log_path}"
    modal run --detach infra/app_train_loop.py::train_loop_alfworld \
        --config "/workspace/${CONFIG}" \
        --n-episodes "${EPS_PER_ROUND}" \
        --k 4 \
        --max-turns 30 \
        --task-id-offset "${task_offset}" \
        --run-name "${run_name}" \
        --round-idx "${round_idx}" \
        --gpu-mem-util 0.20 \
        --eval-episodes "${EVAL_EPS}" \
        --eval-task-id-base "${EVAL_TASK_BASE}" \
        2>&1 | tee "${log_path}"
done

echo "═══ All ${N_ROUNDS} rounds submitted ═══"
