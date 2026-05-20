#!/usr/bin/env bash
# AlfWorld SOTA extended: 8 rounds × 40 episodes + 200 eval episodes/round.
#
# Same recipe as SOTA (vProjection):
#   - lambda_value=0.5 (V supervision against dense facts_diff)
#   - use_v_projection_for_decomposition=true
#   - K=8, all other knobs identical to SOTA config
#
# Differences vs SOTA reference:
#   - rounds: 5 → 8 (more training)
#   - eval-episodes: 100 → 200 (better statistical power, ~±7pp CI vs ±10pp)
#
# Reference: SOTA = 0.59 at 5×100. Target: 0.60-0.63 with extended protocol.
#
# Wall-clock: ~8-10 hours, ~$45-55.
# Log: /tmp/turnrd_alfworld_SOTA_extended.log
set -euo pipefail
cd "$(dirname "$0")/.."

nohup python scripts/run_turnrd_modal.py \
  --config configs/TurnRDV2_alfworld_SOTA_extended.json \
  --env-name alfworld --rounds 8 --episodes-per-round 40 \
  --turnrd-epochs 3 --seed 11 \
  --sft-adapter /vol/checkpoints/sft_alfworld_v1_20260507_165617 \
  --replay-path /vol/cache/TurnRDV2_alfworld_SOTA_extended/replay.jsonl \
  --ckpt-path /vol/cache/TurnRDV2_alfworld_SOTA_extended/ckpt.pt \
  --eval-episodes 200 \
  --run-name-prefix TurnRDV2_alfworld_SOTA_extended \
  --carry-policy-across-rounds \
  --adapter-dir /vol/checkpoints \
  > /tmp/turnrd_alfworld_SOTA_extended.log 2>&1 &
ORCH_PID=$!
disown $ORCH_PID
echo $ORCH_PID > /tmp/turnrd_alfworld_SOTA_extended.pid
echo "AlfWorld SOTA-extended launched: PID=$ORCH_PID"
echo "  Log:        /tmp/turnrd_alfworld_SOTA_extended.log"
echo "  Cache:      /vol/cache/TurnRDV2_alfworld_SOTA_extended/"
echo ""
echo "Reference:"
echo "  SOTA (5×100):    R0=0.48, R1=0.50, R2=0.52, R3=0.55, R4=0.59"
echo "  Target (8×200):  ≥ 0.59 by R4, ideally 0.60-0.63 by R7"
echo ""
echo "Kill triggers: R0 < 0.42, R3 < R0, CUDA error"
echo "Monitor: grep -E 'Eval done|Round [0-9]+|Error' /tmp/turnrd_alfworld_SOTA_extended.log"
