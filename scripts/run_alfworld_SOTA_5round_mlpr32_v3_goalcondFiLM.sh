#!/usr/bin/env bash
# AlfWorld goal-conditioned (FiLM-only) 5-round smoke test.
# A/B vs v3 SOTA with goal_conditioned_value_head + emit_goal_text/emit_goal_emb
# enabled and no per-turn supervision. seed=31, eps=80, rounds=5.
# Override via env vars: SFT_ADAPTER (required), CONFIG, RUN_PREFIX, ROUNDS, etc.

set -euo pipefail
cd "$(dirname "$0")/.."

# Required: same rank-32 MLP-target SFT adapter as v3 SOTA.
SFT_ADAPTER="${SFT_ADAPTER:-}"
if [[ -z "${SFT_ADAPTER}" ]]; then
  echo "ERROR: SFT_ADAPTER env var is required."
  echo "       Set it to the v3-baseline SFT adapter, e.g.:"
  echo "         SFT_ADAPTER=/vol/checkpoints/sft_alfworld_v2_e3_20260521_145134_20260521_145149 \\"
  echo "           bash scripts/run_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM.sh"
  exit 1
fi

CONFIG=${CONFIG:-configs/TurnRDV2_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM.json}
RUN_PREFIX=${RUN_PREFIX:-TurnRDV2_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM}
REPLAY=${REPLAY:-/vol/cache/TurnRDV2_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM/replay.jsonl}
CKPT=${CKPT:-/vol/cache/TurnRDV2_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM/ckpt.pt}
ROUNDS=${ROUNDS:-5}
START_ROUND=${START_ROUND:-0}
EPS_PER_ROUND=${EPS_PER_ROUND:-80}
EVAL_EPS=${EVAL_EPS:-100}
TURNRD_EPOCHS=${TURNRD_EPOCHS:-5}
SEED=${SEED:-31}
ROLLOUT_TEMP=${ROLLOUT_TEMP:-1.0}

LOG=/tmp/turnrd_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM.log
PIDFILE=/tmp/turnrd_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM.pid

# Refuse to clobber an already-running goalcondFiLM orchestrator.
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "ERROR: orchestrator already running with PID $(cat "$PIDFILE"). Refusing to relaunch."
  exit 1
fi

BASE_OFFSET=$(( SEED * ROUNDS * EPS_PER_ROUND ))
TRAIN_HI=$(( BASE_OFFSET + ROUNDS * EPS_PER_ROUND ))

echo "========================================"
echo "AlfWorld SOTA 5-round MLP-r32 - v3 + goalcond FiLM-only smoke (seed=${SEED}, eps=${EPS_PER_ROUND}, T=${ROLLOUT_TEMP})"
echo "  config             : ${CONFIG}"
echo "  sft_adapter        : ${SFT_ADAPTER}  (loaded at R0; carry-policy thereafter)"
echo "  run-name-prefix    : ${RUN_PREFIX}"
echo "  rounds             : ${ROUNDS} (start at R${START_ROUND})"
echo "  episodes/round     : ${EPS_PER_ROUND}"
echo "  eval episodes      : ${EVAL_EPS}"
echo "  turnrd epochs      : ${TURNRD_EPOCHS}"
echo "  seed               : ${SEED}  (same as v3; different train slice from v3 because rounds=5)"
echo "  rollout temperature: ${ROLLOUT_TEMP}"
echo "  base_task_id_offset: ${BASE_OFFSET}"
echo "  train task range   : [${BASE_OFFSET}, ${TRAIN_HI})"
echo "  eval task range    : [6500, 6600)   (SAME pool as v3 - eval is comparable)"
echo "  replay (vol)       : ${REPLAY}   (fresh prefix - starts empty)"
echo "  ckpt   (vol)       : ${CKPT}"
echo "  log                : ${LOG}"
echo "  pidfile            : ${PIDFILE}"
echo ""
echo "  FiLM goal-cond     : ENABLED via config (goal_conditioned_value_head=true,"
echo "                       emit_goal_text=true, emit_goal_emb=true). No per-turn"
echo "                       supervision - goal_match_blend/goal_shaping_coef are"
echo "                       removed from the codebase."
echo "  Parallel-safe      : YES - different RUN_PREFIX + cache dir than v3."
echo "========================================"

# Python-based new-session wrapper (portable replacement for setsid on macOS).
nohup python -c '
import os, sys
os.setsid()
os.execvp(sys.argv[1], sys.argv[1:])
' python scripts/run_turnrd_modal.py \
  --config "${CONFIG}" \
  --env-name alfworld --rounds "${ROUNDS}" --start-round "${START_ROUND}" \
  --episodes-per-round "${EPS_PER_ROUND}" \
  --turnrd-epochs "${TURNRD_EPOCHS}" --seed "${SEED}" \
  --sft-adapter "${SFT_ADAPTER}" \
  --replay-path "${REPLAY}" \
  --ckpt-path "${CKPT}" \
  --eval-episodes "${EVAL_EPS}" \
  --run-name-prefix "${RUN_PREFIX}" \
  --carry-policy-across-rounds \
  --adapter-dir /vol/checkpoints \
  --rollout-temperature "${ROLLOUT_TEMP}" \
  > "$LOG" 2>&1 < /dev/null &
ORCH_PID=$!
disown $ORCH_PID
echo $ORCH_PID > "$PIDFILE"
echo "AlfWorld SOTA 5round goalcond FiLM-only smoke launched: PID=$ORCH_PID"
echo "  Log: $LOG"
echo "  PID: $PIDFILE"
echo "  ETA: ~3-4 hours"
echo ""
echo "Tail the log with:   tail -f $LOG"
echo "Smoke-test sanity checks (in the R0 train_log.json):"
echo "  1. 'goal_aware_supervision' key MUST NOT be present (removed with per-turn supervision)."
echo "  2. mean_reward should be in [0.5, 0.7] - NOT inflated to ~1.0 like the prior shaped run."
echo "  3. R0 eval pct_success should be >= 0.58."
echo "  4. R4 eval should match or exceed v3 R4 (~0.70). Hard floor: 0.65."
