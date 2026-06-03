#!/usr/bin/env bash
# AlfWorld SOTA - Flat GRPO baseline (alpha=1.0) to quantify credit-assignment lift.
# Flips two knobs vs the SOTA recipe: hgpo.alpha 0.5->1.0 and decomposer turnrd->progress
# (inert at alpha=1.0). seed=41, rounds=10, eps=80; everything else matches v3 SOTA.
# Override via env vars: SFT_ADAPTER (required), CONFIG, RUN_PREFIX, ROUNDS, etc.
set -euo pipefail
cd "$(dirname "$0")/.."

# Same SFT adapter family as v3 SOTA (rank-32 + MLP target modules).
SFT_ADAPTER="${SFT_ADAPTER:-}"
if [[ -z "${SFT_ADAPTER}" ]]; then
  echo "ERROR: SFT_ADAPTER env var is required."
  echo "       Set it to the same rank-32 MLP-target SFT adapter used by"
  echo "       scripts/run_alfworld_SOTA_10round_mlpr32_v3.sh, e.g.:"
  echo "         SFT_ADAPTER=/vol/checkpoints/sft_alfworld_v2_e3_20260521_145134_20260521_145149 \\"
  echo "           bash scripts/run_flatGRPO_alfworld_SOTA_a100.sh"
  exit 1
fi

CONFIG=${CONFIG:-configs/flatGRPO_alfworld_SOTA_10round_mlpr32_baseline.json}
RUN_PREFIX=${RUN_PREFIX:-flatGRPO_alfworld_SOTA_baseline_seed31}
ROUNDS=${ROUNDS:-10}
START_ROUND=${START_ROUND:-0}
EPS_PER_ROUND=${EPS_PER_ROUND:-80}
EVAL_EPS=${EVAL_EPS:-100}
SEED=${SEED:-31}
ROLLOUT_TEMP=${ROLLOUT_TEMP:-1.0}
ADAPTER_DIR=${ADAPTER_DIR:-/vol/checkpoints}
EVAL_TASK_BASE=${EVAL_TASK_BASE:-6500}

LOG=/tmp/flatGRPO_alfworld_SOTA_a100.log
PIDFILE=/tmp/flatGRPO_alfworld_SOTA_a100.pid

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "ERROR: launcher already running with PID $(cat "$PIDFILE"). Refusing to relaunch."
  exit 1
fi

BASE_OFFSET=$(( SEED * ROUNDS * EPS_PER_ROUND ))
TRAIN_HI=$(( BASE_OFFSET + ROUNDS * EPS_PER_ROUND ))

# Read K/gpu_mem_util/sync_every from the JSON config (Flat GRPO runs
# app_train_loop.py per round directly; no TurnRD trainer to coordinate).
# Capture stdout first so a JSON/config error fails via set -e.
_CFG_PROBE=$(python -c "
import json, sys
with open('${CONFIG}') as f: c = json.load(f)
tr = c.get('train', {}) or {}
k = int(tr.get('K_trajectories_per_task', 4))
gmu = tr.get('gpu_mem_util')
se = tr.get('sync_every')
print(k, '' if gmu is None else float(gmu), '' if se is None else int(se))
") || { echo "ERROR: failed to read K/gpu_mem_util/sync_every from ${CONFIG}"; exit 1; }
read -r K_PER_TASK GPU_MEM_UTIL SYNC_EVERY <<< "${_CFG_PROBE}"
if [[ -z "${K_PER_TASK}" ]]; then
  echo "ERROR: K_trajectories_per_task probe returned empty for ${CONFIG}"; exit 1
fi

# Read env.max_steps from the config; passed to the entrypoint via --max-turns.
MAX_TURNS=$(python -c "
import json
with open('${CONFIG}') as f: c = json.load(f)
print(int(c.get('env', {}).get('max_steps', 40)))
") || { echo "ERROR: failed to read env.max_steps from ${CONFIG}"; exit 1; }
if [[ -z "${MAX_TURNS}" ]]; then
  echo "ERROR: env.max_steps probe returned empty for ${CONFIG}"; exit 1
fi

echo "========================================"
echo "AlfWorld SOTA - Flat GRPO baseline alpha=1.0 (seed=${SEED}, eps=${EPS_PER_ROUND}, T=${ROLLOUT_TEMP})"
echo "  config             : ${CONFIG}"
echo "  sft_adapter (R0)   : ${SFT_ADAPTER}"
echo "  run-name-prefix    : ${RUN_PREFIX}"
echo "  rounds             : ${ROUNDS} (start at R${START_ROUND})"
echo "  episodes/round     : ${EPS_PER_ROUND}"
echo "  eval episodes      : ${EVAL_EPS}"
echo "  seed               : ${SEED}  (!= v1=11, v2=23, v3=31 - disjoint task slice)"
echo "  K per task         : ${K_PER_TASK}    (from config)"
echo "  max_turns          : ${MAX_TURNS}    (from config env.max_steps)"
echo "  gpu_mem_util       : ${GPU_MEM_UTIL:-<unset>}    (from config)"
echo "  sync_every         : ${SYNC_EVERY:-<unset>}    (from config)"
echo "  rollout temperature: ${ROLLOUT_TEMP}"
echo "  base_task_id_offset: ${BASE_OFFSET}"
echo "  train task range   : [${BASE_OFFSET}, ${TRAIN_HI})"
echo "  eval task range    : [${EVAL_TASK_BASE}, $((EVAL_TASK_BASE + EVAL_EPS)))   (disjoint)"
echo "  adapter dir (vol)  : ${ADAPTER_DIR}"
echo "  log                : ${LOG}"
echo "  pidfile            : ${PIDFILE}"
echo ""
echo "  H-GRPO blend       : alpha=1.0 (pure trajectory advantage; per-turn signal OFF)"
echo "  Decomposer         : 'progress' (inert at alpha=1.0; no TurnRD model built)"
echo "  Driver             : per-round \`modal run\` loop (run_turnrd_modal.py"
echo "                       requires decomposer='turnrd'; not applicable here)."
echo "  Parallel-safe      : YES - disjoint task ranges + disjoint run-name prefix."
echo "========================================"

# Per-round driver: R0 loads the SFT adapter; R_N>0 loads R_{N-1}'s saved
# adapter. Restart on partial failure with START_ROUND=<last_completed+1>.
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

        # Modal mounts the repo at /workspace; relative paths become /workspace/<rel>.
        local config_container="/workspace/${CONFIG}"
        local cmd=(
            modal run --detach
            "infra/app_train_loop.py::train_loop_alfworld"
            --config "${config_container}"
            --n-episodes "${EPS_PER_ROUND}"
            --k "${K_PER_TASK}"
            --max-turns "${MAX_TURNS}"
            --task-id-offset "${task_offset}"
            --run-name "${run_name}"
            --round-idx "${round_idx}"
            --sft-adapter "${load_adapter}"
            --save-adapter-out "${save_adapter}"
            --eval-episodes "${EVAL_EPS}"
            --eval-task-id-base "${EVAL_TASK_BASE}"
            --rollout-temperature "${ROLLOUT_TEMP}"
        )
        if [[ -n "${GPU_MEM_UTIL}" ]]; then
            cmd+=(--gpu-mem-util "${GPU_MEM_UTIL}")
        fi
        if [[ -n "${SYNC_EVERY}" ]]; then
            cmd+=(--sync-every "${SYNC_EVERY}")
        fi

        echo ""
        echo "-- Round ${round_idx}/${ROUNDS} (offset=${task_offset}, load=${load_adapter##*/}, save=${save_adapter##*/})"
        echo "   $ ${cmd[*]}"
        "${cmd[@]}"
        rc=$?
        echo "-- Round ${round_idx} exit=${rc}"
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
MAX_TURNS='${MAX_TURNS}' \
ROLLOUT_TEMP='${ROLLOUT_TEMP}' \
run_rounds
" > "$LOG" 2>&1 < /dev/null &
ORCH_PID=$!
disown $ORCH_PID
echo $ORCH_PID > "$PIDFILE"
echo "AlfWorld flatGRPO alpha=1.0 baseline launched: PID=$ORCH_PID"
echo "  Log: $LOG"
echo "  PID: $PIDFILE"
echo "  ETA: ~4-5 hours"
echo ""
echo "Tail the log with:   tail -f $LOG"
echo ""
echo "Analysis after completion:"
echo "  1. Each round's eval pct_success is in the round's train_log.json on"
echo "     /vol/manifests/${RUN_PREFIX}_seed${SEED}_round<NN>_<ts>/."
echo "  2. Compare against TurnRDV2_alfworld_SOTA_10round_mlpr32_v3 (alpha=0.5)"
echo "     and the existing alpha-sweep mean (alpha=0.5: 0.540, alpha=0.75: 0.518,"
echo "     alpha=0.25: 0.514). Expected alpha=1.0 ~ alpha=0.75 ~ 0.52."
echo "  3. Credit-assignment lift estimate ="
echo "       (v3 SOTA final-round eval) - (this run's final-round eval)."
echo "     Lift > noise floor (+/-5pp at n=100) confirms TurnRD adds signal;"
echo "     lift <= noise floor confirms the noise-analysis prediction that"
echo "     the per-turn channel contributes <2pp."
