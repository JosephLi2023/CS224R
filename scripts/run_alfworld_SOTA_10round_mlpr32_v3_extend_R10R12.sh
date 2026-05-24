#!/usr/bin/env bash
# AlfWorld SOTA mlpr32 v3 — EXTEND R10-R12 (3 more rounds beyond the
# completed 10-round v3 baseline).
#
# Why this exists
# ===============
# v3 finished R0-R9 with a clean monotone climb 0.63 → 0.71 and a late-round
# linear slope of +0.009 eval/round (R² = 0.81 over the last 5 rounds).
# Linear extrapolation predicts R10 ≈ 0.717, R11 ≈ 0.726, R12 ≈ 0.735 —
# i.e., the policy hasn't converged. This launcher runs 3 more rounds to test
# whether the trend continues or saturates.
#
# Geometry note
# =============
# The orchestrator computes `base_task_id_offset = seed × rounds × eps`. With
# the new ROUNDS=13 (vs original 10), the new base = 31 × 13 × 80 = 32240, so
# R10-R12 train on task_id slice [32240, 33280) — DIFFERENT from R0-R9's
# slice [24800, 25600). On AlfWorld this is fine: tasks wrap via
# `% num_train_games`, so the actual games are still drawn from the
# 400-game pool. The carry-policy chain auto-loads R9's saved adapter
# (`..._round09_adapter`) regardless of the offset change.
#
#   train slice (R10-R12) : [32240, 33280)  — 1040 task IDs
#   eval slice            : [6500, 6600)   — disjoint ✓
#   no collision with v1, v2, or v3-R0-R9's slices
#
# Cost
# ====
# ~2-3 hours wall, ~$10-15 compute (plus possible Modal preemption retries).
set -euo pipefail
cd "$(dirname "$0")/.."

SFT_ADAPTER="${SFT_ADAPTER:-}"
if [[ -z "${SFT_ADAPTER}" ]]; then
  echo "ERROR: SFT_ADAPTER env var is required (orchestrator preflight rejects empty)."
  echo "       Carry-policy ignores it for N>=1; only fed to round 0 (which we skip)."
  echo "       Set it to the same path used by the original v3 launch:"
  echo "         SFT_ADAPTER=/vol/checkpoints/sft_alfworld_v2_e3_20260521_145134_20260521_145149 \\"
  echo "           bash scripts/run_alfworld_SOTA_10round_mlpr32_v3_extend_R10R12.sh"
  exit 1
fi

CONFIG=${CONFIG:-configs/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3.json}
RUN_PREFIX=${RUN_PREFIX:-TurnRDV2_alfworld_SOTA_10round_mlpr32_v3}
REPLAY=${REPLAY:-/vol/cache/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3/replay.jsonl}
CKPT=${CKPT:-/vol/cache/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3/ckpt.pt}
ROUNDS=${ROUNDS:-13}
START_ROUND=${START_ROUND:-10}
EPS_PER_ROUND=${EPS_PER_ROUND:-80}
EVAL_EPS=${EVAL_EPS:-100}
TURNRD_EPOCHS=${TURNRD_EPOCHS:-5}
SEED=${SEED:-31}
ROLLOUT_TEMP=${ROLLOUT_TEMP:-1.0}

LOG=/tmp/turnrd_alfworld_SOTA_10round_mlpr32_v3_extend_R10R12.log
PIDFILE=/tmp/turnrd_alfworld_SOTA_10round_mlpr32_v3_extend_R10R12.pid

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "ERROR: extend orchestrator already running PID $(cat "$PIDFILE"). Refusing to relaunch."
  exit 1
fi
# Guard against the base v3 orchestrator still being alive (would double-write cache)
BASE_PIDFILE=/tmp/turnrd_alfworld_SOTA_10round_mlpr32_v3.pid
if [[ -f "$BASE_PIDFILE" ]] && kill -0 "$(cat "$BASE_PIDFILE")" 2>/dev/null; then
  echo "ERROR: base v3 orchestrator still running PID $(cat "$BASE_PIDFILE"). Kill it first."
  exit 1
fi

BASE_OFFSET=$(( SEED * ROUNDS * EPS_PER_ROUND ))
TRAIN_HI=$(( BASE_OFFSET + ROUNDS * EPS_PER_ROUND ))

echo "═══════════════════════════════════════════════════════════════════════"
echo "AlfWorld SOTA mlpr32 v3 EXTEND — R${START_ROUND}-R$((ROUNDS-1))"
echo "  config             : ${CONFIG}"
echo "  sft_adapter        : ${SFT_ADAPTER}  (preflight-required; ignored by carry-policy at N>=1)"
echo "  run-name-prefix    : ${RUN_PREFIX}  (same family as original v3)"
echo "  rounds             : ${ROUNDS} (start at R${START_ROUND}; runs R${START_ROUND}..R$((ROUNDS-1)))"
echo "  episodes/round     : ${EPS_PER_ROUND}"
echo "  eval episodes      : ${EVAL_EPS}"
echo "  turnrd epochs      : ${TURNRD_EPOCHS}"
echo "  seed               : ${SEED}"
echo "  rollout temperature: ${ROLLOUT_TEMP}"
echo "  base_task_id_offset: ${BASE_OFFSET}  (geometry changed since rounds bumped to ${ROUNDS})"
echo "  train task range   : [${BASE_OFFSET}, ${TRAIN_HI})  — disjoint from R0-R9's slice"
echo "  eval task range    : [6500, 6600)   (disjoint)"
echo "  carry-policy from  : /vol/checkpoints/${RUN_PREFIX}_seed${SEED}_round$(printf %02d $((START_ROUND - 1)))_adapter"
echo "  replay (vol)       : ${REPLAY}  (NOT trimmed; recency-decay still active via config)"
echo "  ckpt   (vol)       : ${CKPT}"
echo "  log                : ${LOG}"
echo "  pidfile            : ${PIDFILE}"
echo ""
echo "  Recency decay      : ENABLED (half_life=4 rounds, from config). At R12 the"
echo "                       R0 data weighs 0.5^(12/4) = 0.125; R9 data weighs"
echo "                       0.5^(3/4) ≈ 0.594. Older rows naturally fade."
echo "═══════════════════════════════════════════════════════════════════════"

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
echo "v3 EXTEND R${START_ROUND}-R$((ROUNDS-1)) launched: PID=$ORCH_PID"
echo "  Log: $LOG"
echo "  PID: $PIDFILE"
echo "  ETA: ~2-3 hours, ~\$10-15"
echo ""
echo "Tail the log with:   tail -f $LOG"
echo "Sanity checks for first R${START_ROUND} episode:"
echo "  1. Expect: '>>> Loading SFT-warm-started LoRA adapter from /vol/checkpoints/${RUN_PREFIX}_seed${SEED}_round$(printf %02d $((START_ROUND - 1)))_adapter'"
echo "  2. R${START_ROUND} eval lands in ~60-80 min."
echo "  3. R$((ROUNDS-1)) eval lands in ~3 h total."
