#!/usr/bin/env bash
# Continuation runner for TurnRDv2: extends the existing 4-round
# protocol by running rounds 5 and 6 (zero-indexed). Reuses the
# accumulated replay + ckpt at /vol/cache/method_b_v2/ that the
# original 4-round run left behind.
#
# Original 4-round protocol used `scripts/run_turnrd_modal.py
# --config configs/method_hgpo_turnrd_v2.json --rounds 5 ...`,
# so base_task_id_offset = 11*5*40 = 2200. Round N uses
# tasks [2200 + 40*N, 2200 + 40*(N+1)).
#
#   round 0..3: covered by the original run (rounds=5, but round 4
#               eval crashed mid-pass)
#   round 4   : already trained (per the comparison _notes), but
#               eval crashed → re-eval would need a fresh round
#   round 5   : THIS SCRIPT — tasks [2400, 2440)
#   round 6   : THIS SCRIPT — tasks [2440, 2480)
#
# Each round = train_loop (40 eps + 50 eval) + standalone TurnRDv2 fit
# (3 epochs over the cumulative replay).
#
# Usage:
#   nohup bash scripts/run_turnrdv2_more_rounds.sh > /tmp/v2_more_rounds.log 2>&1 &
#   tail -f /tmp/v2_more_rounds.log
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG_PATH="/workspace/configs/method_hgpo_turnrd_v2.json"
SFT_ADAPTER="/vol/checkpoints/sft_v3_20260504_154752"
RUN_PREFIX="method_b_v2_seed11"
EPS=40
K=4
MAX_TURNS=6
EVAL_EPS=50
EVAL_BASE=6500
GPU_MEM_UTIL=0.20

# Standalone TurnRDv2 fit knobs (must match configs/method_hgpo_turnrd_v2.json's
# `turnrd` block so the fitter constructs the same architecture as the trainer
# refresh hook expects).
REPLAY_PATH="/vol/cache/method_b_v2/replay.jsonl"
CKPT_PATH="/vol/cache/method_b_v2/ckpt.pt"
TURNRD_EPOCHS=3
TURNRD_VERSION="v2"
TURNRD_LAYERS=2
TURNRD_HIDDEN=128
TURNRD_HEADS=4
TURNRD_MAXTURNS=16
TURNRD_DROPOUT=0.1
TURNRD_LR=0.0005
LAMBDA_VALUE=0.0
LAMBDA_RANK=0.1
LAMBDA_PROGRESS=0.01
RANK_MARGIN=0.1
PROGRESS_PRIOR_STRENGTH=1.0

for r in 5 6; do
  OFFSET=$(( 2200 + r * EPS ))
  printf '\n=== TurnRDv2 round %02d (task_offset=%d) ===\n' "$r" "$OFFSET"

  # --- train_loop (foreground; this is the slow step ~30 min) ---
  printf '\n--- round %02d: train_loop ---\n' "$r"
  modal run infra/app_train_loop.py \
    --config "$CONFIG_PATH" \
    --n-episodes "$EPS" \
    --k "$K" \
    --max-turns "$MAX_TURNS" \
    --task-id-offset "$OFFSET" \
    --run-name "${RUN_PREFIX}_round$(printf '%02d' "$r")" \
    --round-idx "$r" \
    --sft-adapter "$SFT_ADAPTER" \
    --eval-episodes "$EVAL_EPS" \
    --eval-task-id-base "$EVAL_BASE" \
    --gpu-mem-util "$GPU_MEM_UTIL"

  # --- standalone TurnRDv2 fit (foreground; ~30s) ---
  printf '\n--- round %02d: standalone TurnRDv2 fit ---\n' "$r"
  modal run infra/app_train_turnrd.py \
    --replay "$REPLAY_PATH" \
    --ckpt-out "$CKPT_PATH" \
    --n-epochs "$TURNRD_EPOCHS" \
    --version "$TURNRD_VERSION" \
    --layers "$TURNRD_LAYERS" \
    --hidden-size "$TURNRD_HIDDEN" \
    --n-heads "$TURNRD_HEADS" \
    --max-turns "$TURNRD_MAXTURNS" \
    --dropout "$TURNRD_DROPOUT" \
    --lr "$TURNRD_LR" \
    --no-causal \
    --progress-prior-strength "$PROGRESS_PRIOR_STRENGTH" \
    --lambda-value "$LAMBDA_VALUE" \
    --lambda-rank "$LAMBDA_RANK" \
    --lambda-progress "$LAMBDA_PROGRESS" \
    --rank-margin "$RANK_MARGIN"

  printf '\n=== round %02d complete ===\n' "$r"
done

echo "=== TurnRDv2 rounds 5+6 protocol complete ==="
