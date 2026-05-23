#!/usr/bin/env bash
# AlfWorld SOTA 10-round MLP-r32 — v3: clean buffer, seed=31, eps=80, T=0.7,
# soft per-batch recency decay (half-life=4 rounds).
#
# Why this exists
# ===============
# Builds on the v2 fresh experiment with two changes:
#   1. rounds=10 (instead of 8) — gives the policy more carry-policy
#      iterations to converge AND tests whether the late-round regression
#      we observed in v1 reappears in a longer schedule with decay.
#   2. turnrd.recency_decay_half_life=4.0 — soft per-batch loss weighting
#      via 0.5^((max_round - round_idx) / 4). Stale rows are KEPT in the
#      buffer (so information isn't thrown away) but contribute less to
#      the TurnRD value-head fit as the policy drifts. See
#      `src/turnrd/train.py::train_turnrd` for the algorithm.
#
# Decay schedule at a glance (half_life=4):
#   round age | weight | semantics
#   ----------+--------+----------
#       0     | 1.000  | fresh data
#       2     | 0.707  | recent
#       4     | 0.500  | half-strength
#       8     | 0.250  | quarter-strength
#      10     | 0.177  | still informative, but a fifth of fresh
#
# Geometry (seed=31, rounds=10, eps=80)
# =====================================
#   base_task_id_offset = 31 × 10 × 80 = 24800
#   train task range    = [24800, 25600)
#   per-round slice     = 80 task IDs starting at base + round × 80
#   eval task range     = [6500, 6600) — disjoint from training
#   disjoint from v1 (seed=11 → [5280, 5760)) ✓
#   disjoint from v2 (seed=23 → [14720, 15360)) ✓
#
# Compute envelope
# ================
# ~6-8 hours wall-clock, ~$40-55. Heavier than v2 (+2 rounds at 80 eps each).
# Can be launched in parallel with v1/v2 — different prefix, different cache
# dir, no Modal app or volume-path collision.
#
# Recency-window note
# ===================
# Unlike v1 (which required `scripts/trim_replay_recency.py` operator-driven
# trim mid-run), v3's decay is BUILT IN at the loss-scaling layer. No manual
# trimming is needed, and you can still trim manually if you want to compare
# soft-decay vs hard-cutoff regimes.
set -euo pipefail
cd "$(dirname "$0")/.."

# Required: same SFT adapter family as v1/v2 (seed-agnostic warm-start).
SFT_ADAPTER="${SFT_ADAPTER:-}"
if [[ -z "${SFT_ADAPTER}" ]]; then
  echo "ERROR: SFT_ADAPTER env var is required."
  echo "       Set it to the path of the rank-32 MLP-target SFT adapter, e.g.:"
  echo "         SFT_ADAPTER=/vol/checkpoints/sft_alfworld_v2_e3_20260521_145134_20260521_145149 \\"
  echo "           bash scripts/run_alfworld_SOTA_10round_mlpr32_v3.sh"
  exit 1
fi

CONFIG=${CONFIG:-configs/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3.json}
RUN_PREFIX=${RUN_PREFIX:-TurnRDV2_alfworld_SOTA_10round_mlpr32_v3}
REPLAY=${REPLAY:-/vol/cache/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3/replay.jsonl}
CKPT=${CKPT:-/vol/cache/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3/ckpt.pt}
ROUNDS=${ROUNDS:-10}
START_ROUND=${START_ROUND:-0}
EPS_PER_ROUND=${EPS_PER_ROUND:-80}
EVAL_EPS=${EVAL_EPS:-100}
TURNRD_EPOCHS=${TURNRD_EPOCHS:-5}
SEED=${SEED:-31}
ROLLOUT_TEMP=${ROLLOUT_TEMP:-1.0}

LOG=/tmp/turnrd_alfworld_SOTA_10round_mlpr32_v3.log
PIDFILE=/tmp/turnrd_alfworld_SOTA_10round_mlpr32_v3.pid

# Refuse to clobber an already-running v3 orchestrator.
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "ERROR: orchestrator already running with PID $(cat "$PIDFILE"). Refusing to relaunch."
  exit 1
fi

BASE_OFFSET=$(( SEED * ROUNDS * EPS_PER_ROUND ))
TRAIN_HI=$(( BASE_OFFSET + ROUNDS * EPS_PER_ROUND ))

echo "═══════════════════════════════════════════════════════════════════════"
echo "AlfWorld SOTA 10-round MLP-r32 — v3 (decay half-life=4, seed=${SEED}, eps=${EPS_PER_ROUND}, T=${ROLLOUT_TEMP})"
echo "  config             : ${CONFIG}"
echo "  sft_adapter        : ${SFT_ADAPTER}  (loaded at R0; carry-policy thereafter)"
echo "  run-name-prefix    : ${RUN_PREFIX}"
echo "  rounds             : ${ROUNDS} (start at R${START_ROUND})"
echo "  episodes/round     : ${EPS_PER_ROUND}"
echo "  eval episodes      : ${EVAL_EPS}"
echo "  turnrd epochs      : ${TURNRD_EPOCHS}"
echo "  seed               : ${SEED}  (\u2260 v1's 11, v2's 23 \u2014 disjoint task slice)"
echo "  rollout temperature: ${ROLLOUT_TEMP}"
echo "  base_task_id_offset: ${BASE_OFFSET}"
echo "  train task range   : [${BASE_OFFSET}, ${TRAIN_HI})"
echo "  eval task range    : [6500, 6600)   (disjoint)"
echo "  replay (vol)       : ${REPLAY}   (starts empty \u2014 fresh prefix)"
echo "  ckpt   (vol)       : ${CKPT}"
echo "  log                : ${LOG}"
echo "  pidfile            : ${PIDFILE}"
echo ""
echo "  Recency decay      : ENABLED, half_life=4 rounds (configured in"
echo "                       ${CONFIG}::turnrd.recency_decay_half_life)."
echo "                       Buffer is append-only \u2014 stale rows are kept and"
echo "                       downweighted at loss time. No manual trim needed."
echo "  Parallel-safe      : YES \u2014 different prefix + cache dir than v1/v2."
echo "═══════════════════════════════════════════════════════════════════════"

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
echo "AlfWorld SOTA mlpr32 v3 launched: PID=$ORCH_PID"
echo "  Log: $LOG"
echo "  PID: $PIDFILE"
echo "  ETA: ~6-8 hours, ~\$40-55"
echo ""
echo "Tail the log with:   tail -f $LOG"
echo "Sanity checks:"
echo "  1. Modal app logs for R0 train_turnrd should show:"
echo "       '[turnrd train] recency-decay enabled: half_life=4.0 rounds, max_round_idx=0, N rows w/ round_idx, 0 legacy rows ...'"
echo "  2. Each round's train_log.json will include a 'recency_decay' block with"
echo "       batch_weight_mean / batch_weight_min / batch_weight_max stats."
echo "  3. Watch eval trend across R0-R9. With decay enabled we expect no late-"
echo "     round regression like v1's R8-R11 dip (since stale rows are softly"
echo "     attenuated rather than dominating the value-head fit)."
