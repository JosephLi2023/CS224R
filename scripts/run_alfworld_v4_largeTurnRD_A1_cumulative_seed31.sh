#!/usr/bin/env bash
# v4 largeTurnRD A1 — CUMULATIVE warm-start arm (plan: turnrd_v2_continual_larger)
#
# What this launcher does
# =======================
# 10-round AlfWorld run with:
#   - Larger TurnRDv2: hidden_size=256, layers=3, n_heads=4 (head_dim=64),
#     dropout=0.15, lr=2e-4 (~1.6M params vs current 862K)
#   - Cumulative warm-start: each round's train_turnrd loads the prior
#     round's ckpt with strict=False (compounds optimizer steps)
#   - LR warmup_cosine: 100-step warmup → cosine decay to 0 over total steps
#   - Fresh-emphasis pass: after main loop, 2 extra epochs on last 2 rounds
#     of records (no recency decay applied during this pass)
#   - Seed=31, K=8 (canonical) — apples-to-apples vs goalcondFiLM
#
# A1 vs A2 (ablation)
# ===================
#   A1 (this launcher): cumulative_train: TRUE  — warm-start each round
#   A2 (sibling):       cumulative_train: FALSE — cold-restart each round
# Both share identical architecture, fresh-emphasis, and LR schedule.
# A1 − A2 isolates the pure cumulative-training contribution.
#
# Geometry
# ========
#   seed=31, rounds=10, episodes_per_round=80, K=8
#   base_task_id_offset = 31 × 10 × 80 = 24800
#   train range = [24800, 25600)   ← SAME as v3 SOTA and goalcondFiLM
#   eval pool   = [6500, 6600)     ← SAME as ALL prior seed-31 runs
#
# Compute envelope
# ================
#   ~10-12h wall-clock, ~$35-50 (matches K=8 goalcondFiLM cadence;
#   architecture change adds ~2× wall-clock to train_turnrd but
#   train_turnrd is only ~3-5% of round wall-clock).
set -euo pipefail
cd "$(dirname "$0")/.."

SFT_ADAPTER="${SFT_ADAPTER:-}"
if [[ -z "${SFT_ADAPTER}" ]]; then
  echo "ERROR: SFT_ADAPTER env var is required."
  echo "  SFT_ADAPTER=/vol/checkpoints/sft_alfworld_v2_e3_20260521_145134_20260521_145149 \\"
  echo "    bash scripts/run_alfworld_v4_largeTurnRD_A1_cumulative_seed31.sh"
  exit 1
fi

CONFIG=${CONFIG:-configs/TurnRDV2_alfworld_SOTA_v4_largeTurnRD_A1_cumulative_seed31.json}
RUN_PREFIX=${RUN_PREFIX:-TurnRDV2_alfworld_SOTA_v4_largeTurnRD_A1}
REPLAY=${REPLAY:-/vol/cache/TurnRDV2_alfworld_SOTA_v4_largeTurnRD_A1/replay.jsonl}
CKPT=${CKPT:-/vol/cache/TurnRDV2_alfworld_SOTA_v4_largeTurnRD_A1/ckpt.pt}

ROUNDS=${ROUNDS:-10}
START_ROUND=${START_ROUND:-0}
EPS_PER_ROUND=${EPS_PER_ROUND:-80}
EVAL_EPS=${EVAL_EPS:-100}
# n_epochs_main = 3 per plan (cumulative warm-start needs fewer per-round epochs;
# the fresh-emphasis pass adds ~2 effective epochs on top, total ~5 effective).
TURNRD_EPOCHS=${TURNRD_EPOCHS:-3}
SEED=${SEED:-31}
ROLLOUT_TEMP=${ROLLOUT_TEMP:-1.0}

LOG=/tmp/turnrd_alfworld_v4_largeTurnRD_A1_cumulative_seed31.log
PIDFILE=/tmp/turnrd_alfworld_v4_largeTurnRD_A1_cumulative_seed31.pid

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "ERROR: A1 orchestrator already running PID $(cat "$PIDFILE")."
  exit 1
fi

BASE_OFFSET=$(( SEED * ROUNDS * EPS_PER_ROUND ))

echo "═══════════════════════════════════════════════════════════════════════"
echo "v4 largeTurnRD A1 — cumulative warm-start, 10 rounds, seed=${SEED}"
echo "  config             : ${CONFIG}"
echo "  sft_adapter (R0)   : ${SFT_ADAPTER}"
echo "  run-name-prefix    : ${RUN_PREFIX}"
echo "  rounds             : ${ROUNDS} (R${START_ROUND}..R$((ROUNDS-1)))"
echo "  episodes/round     : ${EPS_PER_ROUND}"
echo "  K (rollouts/task)  : 8 (canonical)"
echo "  eval episodes      : ${EVAL_EPS}"
echo "  turnrd epochs/main : ${TURNRD_EPOCHS} (+ 2 fresh-emphasis = ~5 effective)"
echo "  seed               : ${SEED}"
echo "  rollout temperature: ${ROLLOUT_TEMP}"
echo "  base_task_id_offset: ${BASE_OFFSET}"
echo "  train range        : [${BASE_OFFSET}, $((BASE_OFFSET + ROUNDS * EPS_PER_ROUND)))"
echo "  eval task range    : [6500, 6600)"
echo "  replay (vol)       : ${REPLAY}"
echo "  ckpt   (vol)       : ${CKPT}"
echo "  log                : ${LOG}"
echo "  pidfile            : ${PIDFILE}"
echo ""
echo "  Architecture       : hidden=256, layers=3, n_heads=4 (head_dim=64), dropout=0.15"
echo "  LR                 : 2e-4 with warmup_cosine (warmup_steps=100)"
echo "  Cumulative warm    : ON (ckpt-in from prior round for N>=1)"
echo "  Fresh-emphasis     : window=2 rounds, n_epochs=2"
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
echo "v4 A1 cumulative launched: PID=$ORCH_PID"
echo "  Log: $LOG"
echo "  PID: $PIDFILE"
echo "  ETA: ~10-12 hours, ~\$35-50"
echo ""
echo "Tail the log with:   tail -f $LOG"
echo "Sanity checks:"
echo "  1. modal-run cmd should contain '--ckpt-in <ckpt path>' for round_idx >= 1"
echo "  2. R0 train_turnrd should log 'LR schedule = warmup_cosine: warmup_steps=100'"
echo "  3. Each round's train_turnrd should log 'fresh-emphasis pass: <N> rows'"
echo "  4. R0 eval should match seed-31 goalcondFiLM R0 (~60%) since both load same SFT"
