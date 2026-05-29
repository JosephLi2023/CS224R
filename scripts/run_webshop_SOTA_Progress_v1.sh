#!/usr/bin/env bash
# WebShop SOTA — HGPO-Progress (Method C) v1.
#
# Recipe-transplant from AlfWorld SOTA (rank-32 + MLP target modules,
# K=8, kl=0.04, lr=5e-6, T=1.0) PLUS the new WebShop-specific
# attribute-progress dense signal (env.use_attribute_progress_intermediate_reward=true).
# alpha=0.5 (per-turn signal CONTRIBUTES), decomposer=progress
# (reads TurnRecord.raw_env_reward, which is populated with the dense
# IR signal). See plan ~/.llms/plans/webshop_sft_mlpr32_oracle_baseline.plan.md.
#
# Method C is the PARAMETER-FREE per-turn-supervision baseline:
# TurnRDv2 learns a model to attribute credit; Progress just uses the
# raw env-progress signal as the per-turn credit. With the
# goal_options + user_sessions adapter fixes landed (verified via
# infra/app_webshop_sft_gen.py::validate_dense_signal:
# mean_ir_by_action_kind[click_option]=0.0384,
# pearson_r(cum_IR, final_reward)=0.9651), this is now a real
# per-turn-supervision experiment on WebShop — previously the dense
# signal degenerated to a 1-bit ASIN-landing indicator so Method C
# was equivalent to flatGRPO on WebShop.
#
# Geometry (seed=41, rounds=10, eps=80)
# =====================================
#   base_task_id_offset = 41 × 10 × 80 = 32800
#   train task range    = [32800, 33600)
#   eval task range     = [6500, 6600)   (disjoint)
#   disjoint from attention v1 (seed=11), flatGRPO v1 (seed=23),
#   and LLMJudge v1 (seed=31) ✓
#
# Compute envelope
# ================
# ~2.5-3 hours wall-clock, ~$15. K=8 + 100-eps eval, 10 rounds × 80 eps.
# Same cost profile as flatGRPO (no TurnRD trainer step per round;
# decomposer=progress is zero additional training/inference cost).
set -euo pipefail
cd "$(dirname "$0")/.."

# Warm-start adapter at R0. Default points at the WebShop SFT v3
# rank-32 + 7-MLP-target adapter produced by
# `scripts/run_webshop_sft_v3_mlpr32.sh` (see
# ~/.llms/plans/webshop_sft_mlpr32_oracle_baseline.plan.md).
#
# PLACEHOLDER — replace `20260527_083426_20260527_153446` with the
# timestamp suffix of the actual adapter dir produced by Phase 4,
# or override at invocation time:
#   SFT_ADAPTER=/vol/checkpoints/sft_webshop_v3_mlpr32_<ts> \
#     bash scripts/run_webshop_SOTA_Progress_v1.sh
SFT_ADAPTER="${SFT_ADAPTER:-/vol/checkpoints/sft_webshop_v3_mlpr32_20260527_083426_20260527_153446}"

CONFIG=${CONFIG:-configs/Progress_webshop_SOTA_10round_mlpr32_v1.json}
RUN_PREFIX=${RUN_PREFIX:-webshop_Progress_v1}
ROUNDS=${ROUNDS:-10}
START_ROUND=${START_ROUND:-0}
EPS_PER_ROUND=${EPS_PER_ROUND:-80}
EVAL_EPS=${EVAL_EPS:-100}
SEED=${SEED:-41}
ROLLOUT_TEMP=${ROLLOUT_TEMP:-1.0}
ADAPTER_DIR=${ADAPTER_DIR:-/vol/checkpoints}
EVAL_TASK_BASE=${EVAL_TASK_BASE:-6500}

LOG=/tmp/webshop_Progress_v1.log
PIDFILE=/tmp/webshop_Progress_v1.pid

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "ERROR: launcher already running with PID $(cat "$PIDFILE"). Refusing to relaunch."
  exit 1
fi

BASE_OFFSET=$(( SEED * ROUNDS * EPS_PER_ROUND ))
TRAIN_HI=$(( BASE_OFFSET + ROUNDS * EPS_PER_ROUND ))

# Read K_trajectories_per_task + gpu_mem_util + sync_every from the JSON
# config so we mirror the orchestrator's flag-derivation logic for
# train_loop_webshop. (Flat GRPO has no TurnRD trainer to coordinate, so
# we invoke `modal run infra/app_train_loop.py::train_loop_webshop`
# directly in a per-round bash for-loop. This matches the orchestrator's
# error-message guidance: "non-TurnRD configs should run directly via
# `modal run infra/app_train_loop.py --config ...`.")
read -r K_PER_TASK GPU_MEM_UTIL SYNC_EVERY MAX_TURNS_CFG < <(python -c "
import json
with open('${CONFIG}') as f: c = json.load(f)
tr = c.get('train', {}) or {}
env = c.get('env', {}) or {}
k = int(tr.get('K_trajectories_per_task', 4))
gmu = tr.get('gpu_mem_util')
se = tr.get('sync_every')
ms = int(env.get('max_steps', 15))
print(k, '' if gmu is None else float(gmu), '' if se is None else int(se), ms)
")
MAX_TURNS=${MAX_TURNS:-${MAX_TURNS_CFG}}

echo "═══════════════════════════════════════════════════════════════════════"
echo "WebShop SOTA — HGPO-Progress (Method C) v1 (seed=${SEED}, eps=${EPS_PER_ROUND}, T=${ROLLOUT_TEMP})"
echo "  config             : ${CONFIG}"
echo "  sft_adapter (R0)   : ${SFT_ADAPTER}"
echo "  run-name-prefix    : ${RUN_PREFIX}"
echo "  rounds             : ${ROUNDS} (start at R${START_ROUND})"
echo "  episodes/round     : ${EPS_PER_ROUND}"
echo "  eval episodes      : ${EVAL_EPS}"
echo "  seed               : ${SEED}"
echo "  K per task         : ${K_PER_TASK}    (from config)"
echo "  gpu_mem_util       : ${GPU_MEM_UTIL:-<unset>}    (from config)"
echo "  sync_every         : ${SYNC_EVERY:-<unset>}    (from config)"
echo "  max_turns          : ${MAX_TURNS}    (from config env.max_steps; forwarded to train_loop)"
echo "  rollout temperature: ${ROLLOUT_TEMP}"
echo "  base_task_id_offset: ${BASE_OFFSET}"
echo "  train task range   : [${BASE_OFFSET}, ${TRAIN_HI})"
echo "  eval task range    : [${EVAL_TASK_BASE}, $((EVAL_TASK_BASE + EVAL_EPS)))   (disjoint)"
echo "  adapter dir (vol)  : ${ADAPTER_DIR}"
echo "  log                : ${LOG}"
echo "  pidfile            : ${PIDFILE}"
echo ""
echo "  Dense signal       : ENABLED via env.use_attribute_progress_intermediate_reward=true"
echo "                       (Method C consumes it as the per-turn reward via"
echo "                       decomposer='progress' + alpha=0.5. The per-turn IR"
echo "                       was validated at mean=0.0384 on click_option turns"
echo "                       and Pearson r=0.9651 vs final reward; see"
echo "                       infra/app_webshop_sft_gen.py::validate_dense_signal.)"
echo "  Driver             : per-round \`modal run\` loop (no TurnRD orchestrator;"
echo "                       decomposer='progress' is zero-parameter — no separate"
echo "                       trainer step per round)."
echo "  Parallel-safe      : YES — disjoint task ranges + adapter run-name prefix."
echo "═══════════════════════════════════════════════════════════════════════"

# In-bash per-round driver. Adapter chaining mirrors run_turnrd_modal.py's
# carry-policy mode: R0 loads ${SFT_ADAPTER}; R_N>0 loads R_{N-1}'s saved
# adapter. Each round saves to ${ADAPTER_DIR}/<prefix>_seed<S>_round<NN>_adapter
# so the chain is deterministic + recoverable.
run_rounds () {
    local rc
    for round_idx in $(seq "${START_ROUND}" $((ROUNDS - 1))); do
        local task_offset=$((BASE_OFFSET + round_idx * EPS_PER_ROUND))
        local round_pad=$(printf "%02d" "${round_idx}")
        local prev_pad=$(printf "%02d" $((round_idx - 1)))
        local run_name="${RUN_PREFIX}_seed${SEED}_round${round_pad}"
        local save_adapter="${ADAPTER_DIR}/${run_name}_adapter"
        local load_adapter
        if [[ "${round_idx}" -eq 0 ]]; then
            load_adapter="${SFT_ADAPTER}"
        else
            load_adapter="${ADAPTER_DIR}/${RUN_PREFIX}_seed${SEED}_round${prev_pad}_adapter"
        fi

        # Build the modal run command. `_to_container_path` analog: the
        # local repo gets mounted at /workspace inside the container, so
        # `configs/foo.json` → `/workspace/configs/foo.json`.
        local config_container="/workspace/${CONFIG}"
        local cmd=(
            modal run --detach
            "infra/app_train_loop.py::train_loop_webshop"
            --config "${config_container}"
            --n-episodes "${EPS_PER_ROUND}"
            --k "${K_PER_TASK}"
            --task-id-offset "${task_offset}"
            --run-name "${run_name}"
            --round-idx "${round_idx}"
            --sft-adapter "${load_adapter}"
            --save-adapter-out "${save_adapter}"
            --eval-episodes "${EVAL_EPS}"
            --eval-task-id-base "${EVAL_TASK_BASE}"
            --max-turns "${MAX_TURNS}"
            --rollout-temperature "${ROLLOUT_TEMP}"
        )
        if [[ -n "${GPU_MEM_UTIL}" ]]; then
            cmd+=(--gpu-mem-util "${GPU_MEM_UTIL}")
        fi
        if [[ -n "${SYNC_EVERY}" ]]; then
            cmd+=(--sync-every "${SYNC_EVERY}")
        fi

        echo ""
        echo "┌── Round ${round_idx}/${ROUNDS} (offset=${task_offset}, load=${load_adapter##*/}, save=${save_adapter##*/})"
        echo "│  $ ${cmd[*]}"
        "${cmd[@]}"
        rc=$?
        echo "└── Round ${round_idx} exit=${rc}"
        if [[ ${rc} -ne 0 ]]; then
            echo "ERROR: round ${round_idx} exited ${rc}. Aborting; restart with START_ROUND=${round_idx}." >&2
            return ${rc}
        fi
    done
    return 0
}

nohup bash -c "
set -uo pipefail
$(declare -f run_rounds)
ADAPTER_DIR='${ADAPTER_DIR}' SFT_ADAPTER='${SFT_ADAPTER}' \
RUN_PREFIX='${RUN_PREFIX}' SEED='${SEED}' \
ROUNDS=${ROUNDS} START_ROUND=${START_ROUND} \
EPS_PER_ROUND=${EPS_PER_ROUND} EVAL_EPS=${EVAL_EPS} \
EVAL_TASK_BASE=${EVAL_TASK_BASE} BASE_OFFSET=${BASE_OFFSET} \
CONFIG='${CONFIG}' K_PER_TASK='${K_PER_TASK}' \
GPU_MEM_UTIL='${GPU_MEM_UTIL}' SYNC_EVERY='${SYNC_EVERY}' \
MAX_TURNS='${MAX_TURNS}' ROLLOUT_TEMP='${ROLLOUT_TEMP}' \
run_rounds
" > "$LOG" 2>&1 < /dev/null &
ORCH_PID=$!
disown $ORCH_PID
echo $ORCH_PID > "$PIDFILE"
echo "WebShop Progress v1 launched: PID=$ORCH_PID"
echo "  Log: $LOG"
echo "  PID: $PIDFILE"
echo "  ETA: ~2.5-3 hours, ~\$15"
echo ""
echo "Tail the log with:   tail -f $LOG"
echo "Pass criterion       : final-round eval ≥ 0.40 (matches flatGRPO target; if Method C wins"
echo "                        meaningfully — e.g. ≥ 0.45 — the dense signal is doing real"
echo "                        per-turn credit work and validates the WebShop adapter fixes)."
