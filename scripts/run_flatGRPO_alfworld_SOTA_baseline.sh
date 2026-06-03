#!/usr/bin/env bash
# AlfWorld SOTA - Flat GRPO baseline (alpha=1.0) to quantify credit-assignment lift.
#
# The alpha-sweep (reports/alfworld_alpha_sweep_README.md) tested
# alpha in {0.25, 0.50, 0.75} but never alpha=1.0, so we can't isolate how
# much of the TurnRDV2 SOTA gain is the per-turn signal vs the Tier-2
# fix-package (lr=5e-6, K=8, dead-K guard).
#
# This run flips two knobs vs the SOTA recipe:
#   1. hgpo.alpha          : 0.5 -> 1.0   (drop per-turn signal; A_H = A_traj)
#   2. hgpo.decomposer     : turnrd -> progress
#                            (decomposer is inert at alpha=1.0; switching to
#                             'progress' removes the TurnRD model + replay
#                             buffer + per-round train_turnrd cycle)
# Everything else (env, K=8, lr=5e-6, kl=0.04, rank-32 LoRA + MLP targets,
# num_train_games=400, dense intermediate_reward signals, eval pool, warm-start
# adapter family, rounds=10, eps/round=80, T=1.0) is identical to
# scripts/run_alfworld_SOTA_10round_mlpr32_v3.sh, so the eval delta isolates
# the per-turn signal's contribution.
#
# decomposer='progress' (not 'turnrd') at alpha=1.0: the combined advantage
# is A_H = 1.0*A_traj + 0.0*A_turn, so A_turn is multiplied by zero regardless
# of the decomposer (see combine in src/algorithms/grpo/advantage.py:130).
# 'progress' is pure-Python and parameter-free, so it avoids building a V-head
# and running train_turnrd per round. This is NOT Progress-HGPO (Method C),
# which runs decomposer='progress' at alpha<1.0 so the per-turn signal flows.
#
# Geometry (seed=41, rounds=10, eps=80):
#   base_task_id_offset = 41 * 10 * 80 = 32800
#   train task range    = [32800, 33600)
#   eval task range     = [6500, 6600)   (disjoint, identical to v3 SOTA)
#   disjoint from v1 (seed=11 -> [5280, 5760))
#   disjoint from v2 (seed=23 -> [14720, 15360))
#   disjoint from v3 (seed=31 -> [24800, 25600))
#
# ~4-5 hours wall-clock. Faster than v3 SOTA because there's no train_turnrd
# per round and no /vol/cache/turnrd_* IO. Can run in parallel with v3 SOTA -
# disjoint task slice and run-name prefix.
#
# Pass criterion: expected alpha=1.0 mean pct_success should land near
# alpha=0.75 mean (0.518) from the alpha-sweep. The credit-assignment lift is
# quantified as: lift = (v3 SOTA alpha=0.5 best mean) - (this run alpha=1.0
# best mean). A null result (this run >= v3 SOTA mean - noise floor) would
# imply the per-turn signal contributes <2pp lift, consistent with
# reports/turn_adv_noise_analysis.md.
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

# Read K_trajectories_per_task + gpu_mem_util + sync_every from the JSON
# config so we mirror the orchestrator's flag-derivation logic for
# train_loop_alfworld. (Flat GRPO has no TurnRD trainer to coordinate,
# so we invoke `modal run infra/app_train_loop.py::train_loop_alfworld`
# directly in a per-round bash for-loop. This matches the WebShop
# flatGRPO launcher and the orchestrator's error message: non-TurnRD
# configs should run directly via `modal run infra/app_train_loop.py`.)
#
# Capture python stdout into a variable first so an exception (missing
# config, malformed JSON) fails the script via `set -e` instead of
# silently producing empty values that get spliced into `modal run`.
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

# AlfWorld max_steps lives under env.max_steps in the config; the modal
# entrypoint takes it via `--max-turns`. Pull it from the same source
# of truth so the two stay in sync.
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

# In-bash per-round driver. Adapter chaining mirrors run_turnrd_modal.py's
# carry-policy mode: R0 loads ${SFT_ADAPTER}; R_N>0 loads R_{N-1}'s saved
# adapter. Each round saves to ${ADAPTER_DIR}/<prefix>_seed<S>_round<NN>_adapter
# so the chain is deterministic + recoverable on partial failure (just
# restart with START_ROUND=<last_completed+1>).
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

        # Modal mounts the local repo at /workspace inside the container,
        # so relative repo paths become /workspace/<rel>.
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
