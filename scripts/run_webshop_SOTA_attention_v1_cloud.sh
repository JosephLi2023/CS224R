#!/usr/bin/env bash
# WebShop SOTA — HGPO-Attention (TurnRDv2) v1 — CLOUD-ORCHESTRATED variant.
#
# Difference vs. run_webshop_SOTA_attention_v1.sh
# ===============================================
# The sibling script runs the round-loop driver locally on the laptop
# (a nohup'd `scripts/run_turnrd_modal.py` Python loop). If the laptop
# sleeps or shuts down mid-run, only the in-flight round's `--detach`
# Modal job finishes; remaining rounds never launch.
#
# This script instead deploys all 3 required apps once, then submits
# infra/app_orchestrator.py::orchestrate_rl_with_turnrd as a single
# `modal run --detach` call. The round-loop runs INSIDE a Modal
# container with a 24-hour timeout. Laptop free to shut down after submit.
#
# Workflow
# ========
# 1. `modal deploy infra/app_train_loop.py`
# 2. `modal deploy infra/app_train_turnrd.py`
# 3. `modal deploy infra/app_orchestrator.py`
# 4. `modal run --detach infra/app_orchestrator.py::orchestrate_rl_with_turnrd \
#       --env-name webshop --config /workspace/<CONFIG> ...`
# 5. Exit.
#
# Cost / envelope
# ===============
# Same as the local sibling (~3-4 hours, ~$20) + ~$0.05 for the
# orchestrator container's CPU+RAM over the full run.
#
# Resume support
# ==============
# The orchestrator scans for per-round "done" sentinels on the volume
# and skips completed rounds. Manual override: pass --no-auto-resume
# (= `--auto-resume=false`) or START_ROUND=<n> to force re-execution.
#
# Parallel-safe vs the local-orchestrated sibling
# ===============================================
# Different RUN_PREFIX suffix `_cloud` so per-round adapter dirs and the
# replay buffer + ckpt path stay disjoint. Reuses the same JSON config
# (no config-side difference; only the round-loop driver differs).
set -euo pipefail
cd "$(dirname "$0")/.."

SFT_ADAPTER="${SFT_ADAPTER:-}"
if [[ -z "${SFT_ADAPTER}" ]]; then
  echo "ERROR: SFT_ADAPTER env var is required."
  echo "       Set it to the rank-32 + 7-MLP-target SFT adapter produced by"
  echo "       scripts/run_webshop_sft_v3_mlpr32_cloud.sh, e.g.:"
  echo "         SFT_ADAPTER=/vol/checkpoints/sft_webshop_v3_mlpr32_<ts> \\"
  echo "           bash scripts/run_webshop_SOTA_attention_v1_cloud.sh"
  exit 1
fi

CONFIG=${CONFIG:-configs/TurnRDV2_webshop_SOTA_10round_mlpr32_v1.json}
RUN_PREFIX=${RUN_PREFIX:-webshop_attention_v1_cloud}
ROUNDS=${ROUNDS:-10}
START_ROUND=${START_ROUND:-0}
EPS_PER_ROUND=${EPS_PER_ROUND:-80}
EVAL_EPS=${EVAL_EPS:-100}
EVAL_TASK_BASE=${EVAL_TASK_BASE:-6500}
TURNRD_EPOCHS=${TURNRD_EPOCHS:-5}
SEED=${SEED:-11}
ROLLOUT_TEMP=${ROLLOUT_TEMP:-1.0}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.30}
SYNC_EVERY=${SYNC_EVERY:-8}

BASE_OFFSET=$(( SEED * ROUNDS * EPS_PER_ROUND ))
TRAIN_HI=$(( BASE_OFFSET + ROUNDS * EPS_PER_ROUND ))

echo "═══════════════════════════════════════════════════════════════════════"
echo "WebShop SOTA — HGPO-Attention (TurnRDv2) v1 (CLOUD orchestrator)"
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
echo "  eval task range    : [${EVAL_TASK_BASE}, $((EVAL_TASK_BASE + EVAL_EPS)))"
echo "  replay (vol)       : (sourced from cfg.turnrd.replay_buffer_path in ${CONFIG})"
echo "  ckpt   (vol)       : (sourced from cfg.turnrd.ckpt_path in ${CONFIG})"
echo ""
echo "  Mode               : CLOUD orchestrator (24h Modal container)."
echo "                       Laptop is free to shut down after submit."
echo "═══════════════════════════════════════════════════════════════════════"

echo ""
echo ">>> Step 1/4: deploying cs224r-hgpo-train-loop ..."
modal deploy infra/app_train_loop.py

echo ""
echo ">>> Step 2/4: deploying cs224r-hgpo-train-turnrd ..."
modal deploy infra/app_train_turnrd.py

echo ""
echo ">>> Step 3/4: deploying cs224r-hgpo-orchestrator ..."
modal deploy infra/app_orchestrator.py

echo ""
echo ">>> Step 4/4: submitting cs224r-hgpo-orchestrator (detached) ..."
modal run --detach infra/app_orchestrator.py::orchestrate_rl_with_turnrd \
  --config "/workspace/${CONFIG}" \
  --sft-adapter "${SFT_ADAPTER}" \
  --env-name webshop \
  --rounds "${ROUNDS}" \
  --start-round "${START_ROUND}" \
  --episodes-per-round "${EPS_PER_ROUND}" \
  --eval-episodes "${EVAL_EPS}" \
  --eval-task-id-base "${EVAL_TASK_BASE}" \
  --seed "${SEED}" \
  --turnrd-epochs "${TURNRD_EPOCHS}" \
  --turnrd-mode 1 \
  --turnrd-batch-size 16 \
  --rollout-temperature "${ROLLOUT_TEMP}" \
  --run-name-prefix "${RUN_PREFIX}" \
  --adapter-dir /vol/checkpoints \
  --gpu-mem-util "${GPU_MEM_UTIL}" \
  --sync-every "${SYNC_EVERY}"

echo ""
echo ">>> Submitted. Laptop is now free to disconnect / shut down."
echo ""
echo "Track progress:"
echo "  modal app logs <orchestrator-app-id>   # see the URL printed above"
echo ""
echo "Pull per-round eval summary after completion:"
echo "  modal volume get cs224r-hgpo-vol \\"
echo "    manifests/${RUN_PREFIX}_seed${SEED}_orchestrator_summary/summary.json ./"
echo ""
echo "Resume from a partial run (orchestrator container died):"
echo "  START_ROUND=<n> SFT_ADAPTER=... \\"
echo "    bash scripts/run_webshop_SOTA_attention_v1_cloud.sh"
