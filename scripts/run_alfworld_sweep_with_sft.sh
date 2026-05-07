#!/usr/bin/env bash
# AlfWorld 3-method sweep WITH SFT-warm-started LoRA adapter.
#
#   bash scripts/run_alfworld_sweep_with_sft.sh /vol/checkpoints/sft_alfworld_v1_<ts>
#
# Launches three methods in parallel, each in its own nohup process so
# the local terminal can be closed without killing the orchestration:
#
#   1. Method B v2     (TurnRD v2 progress decomposer)
#      → scripts/run_turnrd_modal.py with method_hgpo_turnrd_v2_alfworld.json
#   2. Method B lean   (TurnRD v2 reduced losses; ablation)
#      → scripts/run_turnrd_modal.py with method_hgpo_turnrd_lean_alfworld.json
#   3. Method C        (progress decomposer; baseline)
#      → 5 direct `modal run` calls (no turnrd block, can't go through
#        run_turnrd_modal.py); pattern mirrors run_method_c_alfworld.sh
#        but is INLINED here so we can drop `set -euo pipefail` and
#        use `|| true` on transient Modal CLI exit-1 (root cause of
#        the previous Method C dying mid-sweep — see plan
#        `alfworld_sft_warm_start.plan.md` Risks).
#
# All three methods get the same `--sft-adapter` argument so they
# train from a non-trivial init that knows AlfWorld's verb surface.
#
# Per-method logs: /tmp/alfworld_sft_sweep_<method>.log
# Wall-clock budget: ~3-6 hr; ~$30 total.
#
# DELIBERATELY does NOT use `set -e`. Method C's loop in particular
# experienced transient `modal CLI exit 1` mid-sweep last time even
# though the cloud function completed — the orchestrator dying killed
# the remaining rounds. Each per-round Modal call is now wrapped with
# `|| true` so a transient CLI hiccup doesn't propagate.

# Usage check.
if [[ "${1:-}" == "" ]]; then
    echo "Usage: $0 <sft_adapter_path>" >&2
    echo "Example: $0 /vol/checkpoints/sft_alfworld_v1_20260507_180000" >&2
    exit 2
fi

SFT_ADAPTER="$1"

# Shared protocol knobs — match the WebShop sweep so cross-env comparisons
# work. Exposed as env-vars for ad-hoc overrides.
N_ROUNDS=${N_ROUNDS:-5}
EPS_PER_ROUND=${EPS_PER_ROUND:-40}
TURNRD_EPOCHS=${TURNRD_EPOCHS:-3}
SEED=${SEED:-11}
EVAL_EPS=${EVAL_EPS:-50}
EVAL_TASK_BASE=${EVAL_TASK_BASE:-6500}

# Method C task-id math: matches what scripts/run_turnrd_modal.py would
# compute internally. seed × N_ROUNDS × EPS_PER_ROUND.
METHOD_C_TASK_OFFSET=$((SEED * N_ROUNDS * EPS_PER_ROUND))

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

LOG_DIR=${LOG_DIR:-/tmp}

echo "═══════════════════════════════════════════════════════════════"
echo "AlfWorld 3-method sweep with SFT warm-start"
echo "  SFT adapter         : ${SFT_ADAPTER}"
echo "  rounds              : ${N_ROUNDS}"
echo "  episodes per round  : ${EPS_PER_ROUND}"
echo "  turnrd epochs       : ${TURNRD_EPOCHS}"
echo "  seed                : ${SEED}"
echo "  eval episodes       : ${EVAL_EPS}"
echo "  eval task base      : ${EVAL_TASK_BASE}"
echo "  method-C task offset: ${METHOD_C_TASK_OFFSET}"
echo "  log dir             : ${LOG_DIR}"
echo "═══════════════════════════════════════════════════════════════"

# -------------------------------------------------------------------
# Method B v2 — TurnRD v2 progress decomposer
# -------------------------------------------------------------------
B_V2_LOG="${LOG_DIR}/alfworld_sft_sweep_method_b_v2.log"
echo "→ Launching Method B v2 (log: ${B_V2_LOG})"
nohup python3 scripts/run_turnrd_modal.py \
    --env-name alfworld \
    --config configs/method_hgpo_turnrd_v2_alfworld.json \
    --rounds "${N_ROUNDS}" \
    --episodes-per-round "${EPS_PER_ROUND}" \
    --turnrd-epochs "${TURNRD_EPOCHS}" \
    --seed "${SEED}" \
    --eval-episodes "${EVAL_EPS}" \
    --eval-task-id-base "${EVAL_TASK_BASE}" \
    --replay-path /vol/cache/method_b_v2_alfworld/replay.jsonl \
    --ckpt-path  /vol/cache/method_b_v2_alfworld/ckpt.pt \
    --run-name-prefix method_b_v2_alfworld_sft \
    --sft-adapter "${SFT_ADAPTER}" \
    > "${B_V2_LOG}" 2>&1 &
B_V2_PID=$!
echo "   PID: ${B_V2_PID}"

# -------------------------------------------------------------------
# Method B lean — TurnRD v2 reduced losses (ablation)
# -------------------------------------------------------------------
B_LEAN_LOG="${LOG_DIR}/alfworld_sft_sweep_method_b_lean.log"
echo "→ Launching Method B lean (log: ${B_LEAN_LOG})"
nohup python3 scripts/run_turnrd_modal.py \
    --env-name alfworld \
    --config configs/method_hgpo_turnrd_lean_alfworld.json \
    --rounds "${N_ROUNDS}" \
    --episodes-per-round "${EPS_PER_ROUND}" \
    --turnrd-epochs "${TURNRD_EPOCHS}" \
    --seed "${SEED}" \
    --eval-episodes "${EVAL_EPS}" \
    --eval-task-id-base "${EVAL_TASK_BASE}" \
    --replay-path /vol/cache/method_b_lean_alfworld/replay.jsonl \
    --ckpt-path  /vol/cache/method_b_lean_alfworld/ckpt.pt \
    --run-name-prefix method_b_lean_alfworld_sft \
    --sft-adapter "${SFT_ADAPTER}" \
    > "${B_LEAN_LOG}" 2>&1 &
B_LEAN_PID=$!
echo "   PID: ${B_LEAN_PID}"

# -------------------------------------------------------------------
# Method C — progress decomposer (baseline). 5 direct `modal run` calls.
# Wrapped in nohup so the loop continues if the parent shell dies. Each
# inner `modal run` uses `|| true` so a transient Modal CLI exit-1 on
# round K doesn't abort rounds K+1..N.
# -------------------------------------------------------------------
METHOD_C_LOG="${LOG_DIR}/alfworld_sft_sweep_method_c.log"
echo "→ Launching Method C (log: ${METHOD_C_LOG})"
nohup bash -c "
    RUN_PREFIX='method_c_alfworld_sft_seed${SEED}'
    CONFIG='configs/method_hgpo_progress_alfworld.json'
    for round_idx in \$(seq 0 \$((${N_ROUNDS} - 1))); do
        task_offset=\$((${METHOD_C_TASK_OFFSET} + round_idx * ${EPS_PER_ROUND}))
        run_name=\"\${RUN_PREFIX}_round\$(printf '%02d' \"\${round_idx}\")\"
        echo \"═══ Method C Round \${round_idx} ═══\"
        echo \"  task_id_offset = \${task_offset}\"
        echo \"  run_name       = \${run_name}\"
        modal run --detach infra/app_train_loop.py::train_loop_alfworld \
            --config \"/workspace/\${CONFIG}\" \
            --n-episodes ${EPS_PER_ROUND} \
            --k 4 \
            --max-turns 30 \
            --task-id-offset \"\${task_offset}\" \
            --run-name \"\${run_name}\" \
            --round-idx \"\${round_idx}\" \
            --gpu-mem-util 0.20 \
            --sft-adapter \"${SFT_ADAPTER}\" \
            --eval-episodes ${EVAL_EPS} \
            --eval-task-id-base ${EVAL_TASK_BASE} \
            || echo \"⚠ Method C round \${round_idx} modal CLI exited non-zero; cloud job likely still running, continuing to next round.\"
    done
    echo '═══ Method C: all rounds submitted ═══'
" > "${METHOD_C_LOG}" 2>&1 &
METHOD_C_PID=$!
echo "   PID: ${METHOD_C_PID}"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "All 3 methods launched."
echo "  Method B v2  : PID ${B_V2_PID}    log ${B_V2_LOG}"
echo "  Method B lean: PID ${B_LEAN_PID}  log ${B_LEAN_LOG}"
echo "  Method C     : PID ${METHOD_C_PID}  log ${METHOD_C_LOG}"
echo ""
echo "Tail progress with:"
echo "  tail -f ${B_V2_LOG} ${B_LEAN_LOG} ${METHOD_C_LOG}"
echo "Or use scripts/monitor_alfworld_sweep.sh."
echo "═══════════════════════════════════════════════════════════════"
