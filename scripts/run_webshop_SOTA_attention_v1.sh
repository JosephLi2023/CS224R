#!/usr/bin/env bash
# WebShop SOTA - HGPO-Attention (TurnRDv2) v1.
# Recipe-transplant from AlfWorld's 73% SOTA + WebShop attribute-progress dense
# signal (env.use_attribute_progress_intermediate_reward=true). seed=11, rounds=10, eps=80.
# Override via env vars: SFT_ADAPTER, CONFIG, RUN_PREFIX, REPLAY, CKPT, ROUNDS, etc.
set -euo pipefail
cd "$(dirname "$0")/.."

# Warm-start adapter at R0 (WebShop SFT v3 rank-32 + 7-MLP, from
# scripts/run_webshop_sft_v3_mlpr32.sh). Replace REPLACE_WITH_TS_FROM_PHASE4 or
# override via SFT_ADAPTER=/vol/checkpoints/sft_webshop_v3_mlpr32_<ts>. Do NOT
# default to the legacy rank-16 adapter (arch mismatch the new SFT fixes).
SFT_ADAPTER="${SFT_ADAPTER:-/vol/checkpoints/sft_webshop_v3_mlpr32_REPLACE_WITH_TS_FROM_PHASE4}"

CONFIG=${CONFIG:-configs/TurnRDV2_webshop_SOTA_10round_mlpr32_v1.json}
RUN_PREFIX=${RUN_PREFIX:-webshop_attention_v1}
# REPLAY / CKPT defaults must match the config's turnrd.replay_buffer_path /
# turnrd.ckpt_path (run_turnrd_modal.py enforces this at pre-flight).
REPLAY=${REPLAY:-/vol/cache/TurnRDV2_webshop_SOTA_10round_mlpr32_v1/replay.jsonl}
CKPT=${CKPT:-/vol/cache/TurnRDV2_webshop_SOTA_10round_mlpr32_v1/ckpt.pt}
ROUNDS=${ROUNDS:-10}
START_ROUND=${START_ROUND:-0}
EPS_PER_ROUND=${EPS_PER_ROUND:-80}
EVAL_EPS=${EVAL_EPS:-100}
TURNRD_EPOCHS=${TURNRD_EPOCHS:-5}
SEED=${SEED:-11}
ROLLOUT_TEMP=${ROLLOUT_TEMP:-1.0}

LOG=/tmp/webshop_attention_v1.log
PIDFILE=/tmp/webshop_attention_v1.pid

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "ERROR: orchestrator already running with PID $(cat "$PIDFILE"). Refusing to relaunch."
  exit 1
fi

BASE_OFFSET=$(( SEED * ROUNDS * EPS_PER_ROUND ))
TRAIN_HI=$(( BASE_OFFSET + ROUNDS * EPS_PER_ROUND ))

echo "========================================"
echo "WebShop SOTA - HGPO-Attention (TurnRDv2) v1 (seed=${SEED}, eps=${EPS_PER_ROUND}, T=${ROLLOUT_TEMP})"
echo "  config             : ${CONFIG}"
echo "  sft_adapter        : ${SFT_ADAPTER}  (loaded at R0; carry-policy thereafter)"
echo "  run-name-prefix    : ${RUN_PREFIX}"
echo "  rounds             : ${ROUNDS} (start at R${START_ROUND})"
echo "  episodes/round     : ${EPS_PER_ROUND}"
echo "  eval episodes      : ${EVAL_EPS}"
echo "  turnrd epochs      : ${TURNRD_EPOCHS}"
echo "  seed               : ${SEED}"
echo "  rollout temperature: ${ROLLOUT_TEMP}"
echo "  base_task_id_offset: ${BASE_OFFSET}"
echo "  train task range   : [${BASE_OFFSET}, ${TRAIN_HI})"
echo "  eval task range    : [6500, 6600)   (disjoint)"
echo "  replay (vol)       : ${REPLAY}"
echo "  ckpt   (vol)       : ${CKPT}"
echo "  log                : ${LOG}"
echo "  pidfile            : ${PIDFILE}"
echo ""
echo "  Dense signal       : ENABLED via env.use_attribute_progress_intermediate_reward=true"
echo "                       (per-step attribute-progress + ASIN-landing IR, plumbed into"
echo "                       progress_signal via the standard collector + dataset gates)."
echo "  Recency decay      : ENABLED, half_life=4 rounds."
echo "  Parallel-safe      : YES - different prefix + cache dir than flatGRPO/LLMJudge v1."
echo "========================================"

nohup python -c '
import os, sys
os.setsid()
os.execvp(sys.argv[1], sys.argv[1:])
' python scripts/run_turnrd_modal.py \
  --config "${CONFIG}" \
  --env-name webshop --rounds "${ROUNDS}" --start-round "${START_ROUND}" \
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
echo "WebShop attention v1 launched: PID=$ORCH_PID"
echo "  Log: $LOG"
echo "  PID: $PIDFILE"
echo "  ETA: ~3-4 hours"
echo ""
echo "Tail the log with:   tail -f $LOG"
echo "Pass criterion       : final-round eval >= 0.55 (current 0.46 baseline + meaningful lift)."
