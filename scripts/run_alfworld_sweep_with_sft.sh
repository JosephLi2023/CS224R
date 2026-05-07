#!/usr/bin/env bash
# AlfWorld 5-method sweep WITH SFT-warm-started LoRA adapter,
# CANONICAL run-name vocabulary, and per-round POLICY CARRY-ACROSS.
#
#   bash scripts/run_alfworld_sweep_with_sft.sh /vol/checkpoints/sft_alfworld_v1_<ts>
#
# Methods (canonical names — see docs/method_naming.md):
#   1. TurnRDV2     -> scripts/run_turnrd_modal.py with method_hgpo_turnrd_v2_alfworld.json
#                      (--carry-policy-across-rounds)
#   2. TurnRDV1     -> scripts/run_turnrd_modal.py with method_hgpo_turnrd_lean_alfworld.json
#                      (--carry-policy-across-rounds)
#   3. Progressive  -> 5 direct `modal run` calls (no turnrd block, can't go through
#                      run_turnrd_modal.py); inline loop now threads per-round adapter
#                      save/load so the policy carries across rounds.
#   4. SFTOnly      -> 1 direct `modal run` call (n_episodes=0; eval-only baseline).
#                      Mirrors the canonical SFTOnly dispatch in run_methods_protocol.sh.
#   5. flatGRPO     -> 5 direct `modal run` calls; same per-round inline loop pattern as
#                      Progressive (carry-policy threaded), pointed at flatGRPO_alfworld.json.
#
# All five methods get the same `--sft-adapter` argument for round 0 so they
# train from a non-trivial init that knows AlfWorld's verb surface; rounds 1+
# of TurnRDV2/TurnRDV1/Progressive/flatGRPO load the previous round's adapter
# instead of resetting back to SFT.
#
# Per-method logs: /tmp/alfworld_sft_sweep_<canonical_name>.log
# Wall-clock budget: ~3-6 hr; ~$33 total ($30 RL + $3 SFTOnly).
#
# DELIBERATELY does NOT use `set -e`. The Progressive/flatGRPO loops in particular
# experienced transient `modal CLI exit 1` mid-sweep last time even though the
# cloud function completed — the orchestrator dying killed the remaining rounds.
# Each per-round Modal call is wrapped with `|| true` so a transient CLI hiccup
# doesn't propagate.

# Usage check.
if [[ "${1:-}" == "" ]]; then
    echo "Usage: $0 <sft_adapter_path>" >&2
    echo "Example: $0 /vol/checkpoints/sft_alfworld_v1_20260507_165617" >&2
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

# Per-round task-id math for the inline-loop methods (Progressive, flatGRPO):
# matches what scripts/run_turnrd_modal.py computes internally.
INLINE_BASE_OFFSET=$((SEED * N_ROUNDS * EPS_PER_ROUND))

# Adapter dir on the Modal volume — must match the orchestrator's default
# (scripts/run_turnrd_modal.py::OrchestrationConfig.adapter_dir = "/vol/checkpoints").
ADAPTER_DIR="/vol/checkpoints"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

LOG_DIR=${LOG_DIR:-/tmp}

echo "═══════════════════════════════════════════════════════════════"
echo "AlfWorld 5-method sweep with SFT warm-start + carry-policy"
echo "  SFT adapter         : ${SFT_ADAPTER}"
echo "  rounds              : ${N_ROUNDS}"
echo "  episodes per round  : ${EPS_PER_ROUND}"
echo "  turnrd epochs       : ${TURNRD_EPOCHS}"
echo "  seed                : ${SEED}"
echo "  eval episodes       : ${EVAL_EPS}"
echo "  eval task base      : ${EVAL_TASK_BASE}"
echo "  inline task offset  : ${INLINE_BASE_OFFSET}"
echo "  adapter dir         : ${ADAPTER_DIR}"
echo "  log dir             : ${LOG_DIR}"
echo "═══════════════════════════════════════════════════════════════"

# -------------------------------------------------------------------
# TurnRDV2 — TurnRD v2 progress decomposer (orchestrator-driven)
# -------------------------------------------------------------------
TURNRDV2_LOG="${LOG_DIR}/alfworld_sft_sweep_TurnRDV2.log"
echo "→ Launching TurnRDV2 (log: ${TURNRDV2_LOG})"
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
    --run-name-prefix TurnRDV2_alfworld \
    --sft-adapter "${SFT_ADAPTER}" \
    --carry-policy-across-rounds \
    > "${TURNRDV2_LOG}" 2>&1 &
TURNRDV2_PID=$!
echo "   PID: ${TURNRDV2_PID}"

# -------------------------------------------------------------------
# TurnRDV1 — TurnRD v1 lean (CLS-bottlenecked causal encoder, no aux losses)
# -------------------------------------------------------------------
TURNRDV1_LOG="${LOG_DIR}/alfworld_sft_sweep_TurnRDV1.log"
echo "→ Launching TurnRDV1 (log: ${TURNRDV1_LOG})"
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
    --run-name-prefix TurnRDV1_alfworld \
    --sft-adapter "${SFT_ADAPTER}" \
    --carry-policy-across-rounds \
    > "${TURNRDV1_LOG}" 2>&1 &
TURNRDV1_PID=$!
echo "   PID: ${TURNRDV1_PID}"

# -------------------------------------------------------------------
# Progressive — progress decomposer (parameter-free baseline).
# 5 direct `modal run` calls; carry-policy threaded inline.
#
# Round 0 loads ${SFT_ADAPTER}; rounds N>=1 load
#   ${ADAPTER_DIR}/Progressive_alfworld_seed${SEED}_round{N-1:02d}_adapter
# Every round saves
#   ${ADAPTER_DIR}/Progressive_alfworld_seed${SEED}_round{N:02d}_adapter
# so the next round can pick it up. Mirrors the orchestrator's logic in
# scripts/run_turnrd_modal.py::_train_loop_cmd (carry-policy branch).
# -------------------------------------------------------------------
PROGRESSIVE_LOG="${LOG_DIR}/alfworld_sft_sweep_Progressive.log"
echo "→ Launching Progressive (log: ${PROGRESSIVE_LOG})"
nohup bash -c "
    RUN_PREFIX='Progressive_alfworld_seed${SEED}'
    CONFIG='configs/method_hgpo_progress_alfworld.json'
    for round_idx in \$(seq 0 \$((${N_ROUNDS} - 1))); do
        task_offset=\$((${INLINE_BASE_OFFSET} + round_idx * ${EPS_PER_ROUND}))
        round_pad=\$(printf '%02d' \"\${round_idx}\")
        run_name=\"\${RUN_PREFIX}_round\${round_pad}\"
        save_adapter_out=\"${ADAPTER_DIR}/\${run_name}_adapter\"
        if [[ \"\${round_idx}\" -eq 0 ]]; then
            load_adapter='${SFT_ADAPTER}'
        else
            prev_pad=\$(printf '%02d' \$((round_idx - 1)))
            load_adapter=\"${ADAPTER_DIR}/\${RUN_PREFIX}_round\${prev_pad}_adapter\"
        fi
        echo \"═══ Progressive Round \${round_idx} ═══\"
        echo \"  task_id_offset   = \${task_offset}\"
        echo \"  run_name         = \${run_name}\"
        echo \"  load_adapter     = \${load_adapter}\"
        echo \"  save_adapter_out = \${save_adapter_out}\"
        modal run --detach infra/app_train_loop.py::train_loop_alfworld \
            --config \"/workspace/\${CONFIG}\" \
            --n-episodes ${EPS_PER_ROUND} \
            --k 4 \
            --max-turns 30 \
            --task-id-offset \"\${task_offset}\" \
            --run-name \"\${run_name}\" \
            --round-idx \"\${round_idx}\" \
            --gpu-mem-util 0.20 \
            --sft-adapter \"\${load_adapter}\" \
            --save-adapter-out \"\${save_adapter_out}\" \
            --eval-episodes ${EVAL_EPS} \
            --eval-task-id-base ${EVAL_TASK_BASE} \
            || echo \"⚠ Progressive round \${round_idx} modal CLI exited non-zero; cloud job likely still running, continuing to next round.\"
    done
    echo '═══ Progressive: all rounds submitted ═══'
" > "${PROGRESSIVE_LOG}" 2>&1 &
PROGRESSIVE_PID=$!
echo "   PID: ${PROGRESSIVE_PID}"

# -------------------------------------------------------------------
# flatGRPO — alpha=1.0, decomposer inert (no-decomposer baseline).
# Same per-round inline loop pattern as Progressive (carry-policy threaded).
# -------------------------------------------------------------------
FLATGRPO_LOG="${LOG_DIR}/alfworld_sft_sweep_flatGRPO.log"
echo "→ Launching flatGRPO (log: ${FLATGRPO_LOG})"
nohup bash -c "
    RUN_PREFIX='flatGRPO_alfworld_seed${SEED}'
    CONFIG='configs/flatGRPO_alfworld.json'
    for round_idx in \$(seq 0 \$((${N_ROUNDS} - 1))); do
        task_offset=\$((${INLINE_BASE_OFFSET} + round_idx * ${EPS_PER_ROUND}))
        round_pad=\$(printf '%02d' \"\${round_idx}\")
        run_name=\"\${RUN_PREFIX}_round\${round_pad}\"
        save_adapter_out=\"${ADAPTER_DIR}/\${run_name}_adapter\"
        if [[ \"\${round_idx}\" -eq 0 ]]; then
            load_adapter='${SFT_ADAPTER}'
        else
            prev_pad=\$(printf '%02d' \$((round_idx - 1)))
            load_adapter=\"${ADAPTER_DIR}/\${RUN_PREFIX}_round\${prev_pad}_adapter\"
        fi
        echo \"═══ flatGRPO Round \${round_idx} ═══\"
        echo \"  task_id_offset   = \${task_offset}\"
        echo \"  run_name         = \${run_name}\"
        echo \"  load_adapter     = \${load_adapter}\"
        echo \"  save_adapter_out = \${save_adapter_out}\"
        modal run --detach infra/app_train_loop.py::train_loop_alfworld \
            --config \"/workspace/\${CONFIG}\" \
            --n-episodes ${EPS_PER_ROUND} \
            --k 4 \
            --max-turns 30 \
            --task-id-offset \"\${task_offset}\" \
            --run-name \"\${run_name}\" \
            --round-idx \"\${round_idx}\" \
            --gpu-mem-util 0.20 \
            --sft-adapter \"\${load_adapter}\" \
            --save-adapter-out \"\${save_adapter_out}\" \
            --eval-episodes ${EVAL_EPS} \
            --eval-task-id-base ${EVAL_TASK_BASE} \
            || echo \"⚠ flatGRPO round \${round_idx} modal CLI exited non-zero; cloud job likely still running, continuing to next round.\"
    done
    echo '═══ flatGRPO: all rounds submitted ═══'
" > "${FLATGRPO_LOG}" 2>&1 &
FLATGRPO_PID=$!
echo "   PID: ${FLATGRPO_PID}"

# -------------------------------------------------------------------
# SFTOnly — eval-only baseline (n_episodes=0, single one-shot Modal call).
# Mirrors the canonical SFTOnly dispatch in scripts/run_methods_protocol.sh.
# Cost ~$3, ~10 min.
# -------------------------------------------------------------------
SFTONLY_LOG="${LOG_DIR}/alfworld_sft_sweep_SFTOnly.log"
echo "→ Launching SFTOnly (log: ${SFTONLY_LOG})"
nohup modal run infra/app_train_loop.py::train_loop_alfworld \
    --config /workspace/configs/SFTOnly_alfworld.json \
    --n-episodes 0 \
    --k 4 \
    --max-turns 30 \
    --task-id-offset 0 \
    --run-name "SFTOnly_alfworld_seed${SEED}_round00" \
    --round-idx 0 \
    --sft-adapter "${SFT_ADAPTER}" \
    --eval-episodes "${EVAL_EPS}" \
    --eval-task-id-base "${EVAL_TASK_BASE}" \
    --gpu-mem-util 0.30 \
    > "${SFTONLY_LOG}" 2>&1 &
SFTONLY_PID=$!
echo "   PID: ${SFTONLY_PID}"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "All 5 methods launched."
echo "  TurnRDV2    : PID ${TURNRDV2_PID}     log ${TURNRDV2_LOG}"
echo "  TurnRDV1    : PID ${TURNRDV1_PID}     log ${TURNRDV1_LOG}"
echo "  Progressive : PID ${PROGRESSIVE_PID}  log ${PROGRESSIVE_LOG}"
echo "  flatGRPO    : PID ${FLATGRPO_PID}     log ${FLATGRPO_LOG}"
echo "  SFTOnly     : PID ${SFTONLY_PID}      log ${SFTONLY_LOG}"
echo ""
echo "Tail progress with:"
echo "  tail -f ${TURNRDV2_LOG} ${TURNRDV1_LOG} ${PROGRESSIVE_LOG} ${FLATGRPO_LOG} ${SFTONLY_LOG}"
echo "Or use scripts/monitor_alfworld_sweep.sh."
echo "═══════════════════════════════════════════════════════════════"
