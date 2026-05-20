#!/usr/bin/env bash
# Pure K=8 test on WebShop (V supervision OFF — matches winning V2 baseline).
#
# Single-knob change vs TurnRDV2_repro: K_trajectories_per_task 4 → 8.
# lambda_value stays at 0.0 (V disabled — data shows WebShop's sparse
# progress signal makes V supervision actively harmful).
#
# Built on the max_model_len=4096 patch in app_train_loop.py (fixes the
# prompt-truncation CUDA crash that hit the previous K=8 attempt).
#
# Wall-clock: ~6-8 hours, ~$40-50 (K=8 + 100-eps eval).
# Log: /tmp/turnrd_TurnRDV2_K8_v2.log
set -euo pipefail
cd "$(dirname "$0")/.."

nohup python scripts/run_turnrd_modal.py \
  --config configs/TurnRDV2_K8_v2.json \
  --env-name webshop --rounds 5 --episodes-per-round 40 \
  --turnrd-epochs 3 --seed 11 \
  --sft-adapter /vol/checkpoints/sft_v3_20260504_154752 \
  --replay-path /vol/cache/TurnRDV2_K8_v2/replay.jsonl \
  --ckpt-path /vol/cache/TurnRDV2_K8_v2/ckpt.pt \
  --eval-episodes 100 \
  --run-name-prefix TurnRDV2_K8_v2 \
  --carry-policy-across-rounds \
  --adapter-dir /vol/checkpoints \
  > /tmp/turnrd_TurnRDV2_K8_v2.log 2>&1 &
ORCH_PID=$!
disown $ORCH_PID
echo $ORCH_PID > /tmp/turnrd_TurnRDV2_K8_v2.pid
echo "K8_v2 (V_OFF) launched: PID=$ORCH_PID"
echo "  Log:        /tmp/turnrd_TurnRDV2_K8_v2.log"
echo "  Cache:      /vol/cache/TurnRDV2_K8_v2/"
echo ""
echo "Single-knob test: K=4 → K=8 (V supervision stays OFF, matching V2 baseline)."
echo ""
echo "Reference (100-eps eval):"
echo "  TurnRDV2_repro (K=4): R0..R4 = 24% → 32% → 22% → 32% → 36% (best 36%)"
echo "  Progressive (no V):   best 46%"
echo ""
echo "Kill triggers: R0<0.20, R3<R0, CUDA error"
