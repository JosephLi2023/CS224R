#!/usr/bin/env bash
# ============================================================================
# [FALSIFIED 2026-05-27] DO NOT RELAUNCH.
#
# Result of the seed31 run (10 rounds): R0=0.61, R3=0.66, R4=0.70, R5=0.70,
# R6=0.68, R7=0.70, R8=0.69 — flat plateau identical to v3 baseline (0.69-0.71).
# Hit the config's pre-registered abort criterion at R5 (R5=0.70 < 0.75).
# The "ceiling is step-budget-bound" hypothesis is ruled out by the data; the
# AlfWorld ceiling on this slice is NOT truncation-bound. Pivot to the
# goalcond A/B (supervision-mismatch hypothesis) — see
# scripts/run_alfworld_SOTA_10round_mlpr32_v3_goalcond.sh.
#
# Launcher kept for negative-evidence reproducibility. The replay buffer and
# per-round adapters at /vol/cache/..._xlbudget/ and /vol/checkpoints/
# ..._xlbudget_cloud_seed31_round*_adapter are preserved as a baseline data
# source for the goalcond comparison.
# ============================================================================
#
# AlfWorld SOTA 10-round MLP-r32 — v3 + xlbudget: lift the per-episode step
# cap that the v3 R9 failure-mode probe identified as the binding ceiling.
#
# Why this exists
# ===============
# v3 (this script's parent) hit a 0.71-0.73 pct_success ceiling that
# oscillated across rounds (R6=0.70 → R10=0.69 → R12=0.73). The failure-
# mode probe `reports/turnrd_credit_assignment_demo/v3_R9_probe.summary.json`
# is decisive: 100% of probed eval failures truncate at n_turns ==
# max_turns. Every credit-assignment knob has been exhausted (rank-32 MLP
# LoRA, recency-decay replay, dense intermediate rewards, goal-aware
# supervision). The ceiling is STRUCTURAL — step-budget-bound, not
# credit-assignment-bound.
#
# This launcher runs the same v3 recipe with two budget knobs lifted:
#   - env.max_steps         40 -> 60   (TextWorld adapter step cap headroom)
#   - env.max_turns         30 -> 50   (rollout-collector turn cap; new
#                                       JSON field plumbed in
#                                       infra/app_train_loop.py per plan
#                                       `alfworld_xlbudget_break_ceiling`)
#
# Deliberately NOT changed: train/eval max_action_tokens stays at the
# hardcoded 48. The v3 R9 failure-mode probe shows max action length is
# 6 words (~10 tokens) across all 11 failure trajectories; lifting the
# per-action token cap would be a no-op confounder.
#
# Everything else is byte-for-byte v3: rank-32 MLP+attn LoRA, alpha=0.5,
# v_projection=True, recency_decay_half_life=4.0, num_train_games=400,
# lambda_value=1.0, TURNRD_EPOCHS=5, SEED=31, ROUNDS=10, EPS_PER_ROUND=80,
# EVAL_EPS=100, ROLLOUT_TEMP=1.0, same SFT warm-start adapter.
#
# Geometry (seed=31, rounds=10, eps=80)
# =====================================
#   base_task_id_offset = 31 × 10 × 80 = 24800  (same as v3)
#   train task range    = [24800, 25600)        (same as v3)
#   eval task range     = [6500, 6600)          (disjoint from train)
#
# Compute envelope
# ================
# ~7-10 hours wall-clock, ~$25-35. Per-round wall-clock grows ~15-30% vs
# v3 because each rollout can use up to 50 turns (vs 30) at the unchanged
# 48-token action cap. Total run length cap is `--rounds 10`.
#
# Parallel-safe vs v3
# ===================
# Different RUN_PREFIX, different REPLAY/CKPT paths, different PIDFILE.
# Per-round adapters are written under `_xlbudget_round{00..09}_adapter/`
# so the existing v3 R0-R9 adapters at
# `/vol/checkpoints/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3_seed31_round*_adapter/`
# are NOT overwritten and the old-vs-new comparison stays apples-to-apples.

set -euo pipefail
cd "$(dirname "$0")/.."

# Required: same SFT adapter family as v1/v2/v3 (seed-agnostic warm-start).
SFT_ADAPTER="${SFT_ADAPTER:-}"
if [[ -z "${SFT_ADAPTER}" ]]; then
  echo "ERROR: SFT_ADAPTER env var is required."
  echo "       Set it to the path of the rank-32 MLP-target SFT adapter, e.g.:"
  echo "         SFT_ADAPTER=/vol/checkpoints/sft_alfworld_v2_e3_20260521_145134_20260521_145149 \\"
  echo "           bash scripts/run_alfworld_SOTA_10round_mlpr32_v3_xlbudget.sh"
  exit 1
fi

CONFIG=${CONFIG:-configs/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3_xlbudget.json}
RUN_PREFIX=${RUN_PREFIX:-TurnRDV2_alfworld_SOTA_10round_mlpr32_v3_xlbudget}
REPLAY=${REPLAY:-/vol/cache/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3_xlbudget/replay.jsonl}
CKPT=${CKPT:-/vol/cache/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3_xlbudget/ckpt.pt}
ROUNDS=${ROUNDS:-10}
START_ROUND=${START_ROUND:-0}
EPS_PER_ROUND=${EPS_PER_ROUND:-80}
EVAL_EPS=${EVAL_EPS:-100}
TURNRD_EPOCHS=${TURNRD_EPOCHS:-5}
SEED=${SEED:-31}
ROLLOUT_TEMP=${ROLLOUT_TEMP:-1.0}

LOG=/tmp/turnrd_alfworld_SOTA_10round_mlpr32_v3_xlbudget.log
PIDFILE=/tmp/turnrd_alfworld_SOTA_10round_mlpr32_v3_xlbudget.pid

# Refuse to clobber an already-running xlbudget orchestrator.
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "ERROR: orchestrator already running with PID $(cat "$PIDFILE"). Refusing to relaunch."
  exit 1
fi

BASE_OFFSET=$(( SEED * ROUNDS * EPS_PER_ROUND ))
TRAIN_HI=$(( BASE_OFFSET + ROUNDS * EPS_PER_ROUND ))

echo "═══════════════════════════════════════════════════════════════════════"
echo "AlfWorld SOTA 10-round MLP-r32 — v3 + xlbudget (seed=${SEED}, eps=${EPS_PER_ROUND}, T=${ROLLOUT_TEMP})"
echo "  config             : ${CONFIG}"
echo "  sft_adapter        : ${SFT_ADAPTER}  (loaded at R0; carry-policy thereafter)"
echo "  run-name-prefix    : ${RUN_PREFIX}"
echo "  rounds             : ${ROUNDS} (start at R${START_ROUND})"
echo "  episodes/round     : ${EPS_PER_ROUND}"
echo "  eval episodes      : ${EVAL_EPS}"
echo "  turnrd epochs      : ${TURNRD_EPOCHS}"
echo "  seed               : ${SEED}  (same as v3; xlbudget is in-place lift, not a new seed)"
echo "  rollout temperature: ${ROLLOUT_TEMP}"
echo "  base_task_id_offset: ${BASE_OFFSET}"
echo "  train task range   : [${BASE_OFFSET}, ${TRAIN_HI})"
echo "  eval task range    : [6500, 6600)   (disjoint)"
echo "  replay (vol)       : ${REPLAY}   (starts empty — fresh xlbudget prefix)"
echo "  ckpt   (vol)       : ${CKPT}"
echo "  log                : ${LOG}"
echo "  pidfile            : ${PIDFILE}"
echo ""
echo "  Budget lift        : env.max_steps 40->60, env.max_turns 30->50."
echo "                       max_action_tokens stays at 48 (probe shows"
echo "                       max action length is 6 words; lifting would"
echo "                       be a no-op confounder). See ${CONFIG} _notes."
echo "  Parallel-safe      : YES — different RUN_PREFIX + cache dir than v3."
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
echo "AlfWorld SOTA mlpr32 v3 xlbudget launched: PID=$ORCH_PID"
echo "  Log: $LOG"
echo "  PID: $PIDFILE"
echo "  ETA: ~7-10 hours, ~\$25-35"
echo ""
echo "Tail the log with:   tail -f $LOG"
echo "Sanity checks (R0 eval_block must show):"
echo "  1. eval.max_turns == 50 (new JSON plumbing reached the rollout collector)"
echo "  2. eval.max_action_tokens == 48 (unchanged; probe-justified no-op skip)"
echo "  3. eval.empty_outputs field is present (value can be 0)"
echo "  4. at least one rollout has n_turns > 30 OR a rollout that previously"
echo "     truncated at n_turns=30 now completes with success=True"
echo ""
echo "Abort criteria:"
echo "  - R0 pct_success < 0.62 (budget lift HURT R0) -> abort"
echo "  - R5 pct_success < 0.75 (ceiling wasn't truncation-bound) -> abort"
echo ""
echo "Realistic target (probe-derived): 0.78-0.85 pct_success at R9."
echo "  Hard floor: 3 of 11 probed failures are policy dead-zones (IR=0,"
echo "  repeated actions) that the budget lift cannot fix. Floor ~ 0.94."
echo ""
echo "Recommended wrapper while orchestrator is alive (mitigates Modal CLI"
echo "poll-timeout that hit goalsup R2):"
echo "  caffeinate -dimsu &"
