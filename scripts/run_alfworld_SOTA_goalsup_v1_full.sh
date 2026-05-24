#!/usr/bin/env bash
# Goal-aware-supervision RL-loop integration: FULL 10-round run.
#
# Plan: turnrd_goalsup_rl_loop_integration. This is Phase 4 (full SoTA
# schedule) — same recipe as the 3-round smoke that already passed all
# verification criteria, just extended to the full 10-round
# carry-policy schedule.
#
# Config tweaks vs the smoke run (configs/TurnRDV2_alfworld_SOTA_goalsup_v1.json):
#   - num_eval_games: 50 -> 127 (max valid_seen, more stable easy-type eval)
#   - progress_prior_strength: 1.0 -> 0.5 (let goal-aware target dominate)
#   - β = 0.5, emit_goal_text=true (unchanged from smoke)
#
# Geometry (seed=42, rounds=10, eps=80):
#   base_task_id_offset = 42 × 10 × 80 = 33600
#   train task range    = [33600, 34400)   (disjoint from smoke's [9840, 10080))
#   eval task range     = [6500, 6600)     (disjoint from training)
#   disjoint from v1 (seed=11), v2 (seed=23), v3 (seed=31), goalsup-smoke (seed=41) ✓
#
# Run name + replay/ckpt paths use a fresh `goalsup_v1_full` prefix so
# the smoke replay isn't mixed into the full run.
#
# Compute envelope: ~$25-30 + 10-12 hours on A100. Matches the v3 SoTA
# envelope (the goal-aware-supervision blend adds negligible per-round
# overhead).
set -euo pipefail
cd "$(dirname "$0")/.."

SFT_ADAPTER="${SFT_ADAPTER:-}"
if [[ -z "${SFT_ADAPTER}" ]]; then
  echo "ERROR: SFT_ADAPTER env var is required."
  echo "       Set it to the path of the rank-32 MLP-target SFT adapter, e.g.:"
  echo "         SFT_ADAPTER=/vol/checkpoints/sft_alfworld_v2_e3_20260521_145134_20260521_145149 \\"
  echo "           bash scripts/run_alfworld_SOTA_goalsup_v1_full.sh"
  exit 1
fi

CONFIG=${CONFIG:-configs/TurnRDV2_alfworld_SOTA_goalsup_v1.json}
RUN_PREFIX=${RUN_PREFIX:-TurnRDV2_alfworld_SOTA_goalsup_v1_full}
REPLAY=${REPLAY:-/vol/cache/TurnRDV2_alfworld_SOTA_goalsup_v1_full/replay.jsonl}
CKPT=${CKPT:-/vol/cache/TurnRDV2_alfworld_SOTA_goalsup_v1_full/ckpt.pt}
ROUNDS=${ROUNDS:-10}
START_ROUND=${START_ROUND:-0}
EPS_PER_ROUND=${EPS_PER_ROUND:-80}
EVAL_EPS=${EVAL_EPS:-100}
TURNRD_EPOCHS=${TURNRD_EPOCHS:-5}
SEED=${SEED:-42}
ROLLOUT_TEMP=${ROLLOUT_TEMP:-1.0}

LOG=/tmp/turnrd_alfworld_SOTA_goalsup_v1_full.log
PIDFILE=/tmp/turnrd_alfworld_SOTA_goalsup_v1_full.pid

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "ERROR: orchestrator already running with PID $(cat "$PIDFILE"). Refusing to relaunch."
  exit 1
fi

BASE_OFFSET=$(( SEED * ROUNDS * EPS_PER_ROUND ))
TRAIN_HI=$(( BASE_OFFSET + ROUNDS * EPS_PER_ROUND ))

# The split-brain check in run_turnrd_modal.py requires the orchestrator's
# --replay-path and the config's turnrd.replay_buffer_path to match. The
# config defaults to .../goalsup_v1/replay.jsonl but the full run wants
# .../goalsup_v1_full/replay.jsonl. We override the config path on the fly
# via a small JSON patch right before invocation.
# IMPORTANT: the patched config MUST live inside REPO_ROOT (the orchestrator
# refuses configs in /var/folders/ etc. because Modal can't map them to
# /workspace). Use a stable in-repo path so the patched config is
# overwritten each launch instead of accumulating.
TMP_CONFIG="configs/_goalsup_v1_full_generated.json"
python -c "
import json, sys
c = json.load(open('${CONFIG}'))
c['turnrd']['replay_buffer_path'] = '${REPLAY}'
c['turnrd']['ckpt_path'] = '${CKPT}'
json.dump(c, open('${TMP_CONFIG}','w'), indent=2)
"
CONFIG="${TMP_CONFIG}"

echo "═══════════════════════════════════════════════════════════════════════"
echo "AlfWorld SOTA — goal-aware-supervision FULL (seed=${SEED}, rounds=${ROUNDS}, eps=${EPS_PER_ROUND}, T=${ROLLOUT_TEMP})"
echo "  config             : ${CONFIG}  (in-place patched paths)"
echo "  sft_adapter        : ${SFT_ADAPTER}  (loaded at R0; carry-policy thereafter)"
echo "  run-name-prefix    : ${RUN_PREFIX}"
echo "  rounds             : ${ROUNDS} (start at R${START_ROUND})"
echo "  episodes/round     : ${EPS_PER_ROUND}"
echo "  eval episodes      : ${EVAL_EPS}"
echo "  turnrd epochs      : ${TURNRD_EPOCHS}"
echo "  seed               : ${SEED}  (disjoint from v1/v2/v3/goalsup-smoke)"
echo "  rollout temperature: ${ROLLOUT_TEMP}"
echo "  base_task_id_offset: ${BASE_OFFSET}"
echo "  train task range   : [${BASE_OFFSET}, ${TRAIN_HI})"
echo "  eval task range    : [6500, 6600)"
echo "  replay (vol)       : ${REPLAY}"
echo "  ckpt   (vol)       : ${CKPT}"
echo "  log                : ${LOG}"
echo "  pidfile            : ${PIDFILE}"
echo ""
echo "  Goal-aware-sup     : ENABLED (β=0.5, emit_goal_text=true)"
echo "  Tweaks vs smoke    : num_eval_games 50→127, progress_prior_strength 1.0→0.5"
echo "  Compute envelope   : ~\$25-30 + 10-12 hr on A100."
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
echo "goalsup FULL launched: PID=$ORCH_PID"
echo "  Log: $LOG"
echo "  PID: $PIDFILE"
echo "  Tail with:   tail -f $LOG"
echo ""
echo "Expected milestones:"
echo "  - R0 eval pct_success ~0.61 (matches smoke + within 1pp of v3)"
echo "  - R9 eval pct_success > 0.71 (target: beat v3 R9's 0.71)"
echo "  - Each round emits 'goal_aware_supervision' block in train_log.json"
echo "    with n_batches_with_goal_match == n_batches_total (full coverage)"
echo ""
echo "Note: held-out per-type hard-task-type probe runs SEPARATELY after R9"
echo "completes — that probe is not part of this script (use the existing"
echo "per_type_eval probe app pointed at the round 9 adapter)."
