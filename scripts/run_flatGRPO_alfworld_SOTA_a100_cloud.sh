#!/usr/bin/env bash
# AlfWorld SOTA — Flat GRPO (α=1.0) baseline — CLOUD-ORCHESTRATED variant.
#
# Difference vs. run_flatGRPO_alfworld_SOTA_a100.sh
# =================================================
# The sibling runs a per-round bash for-loop locally. This script
# instead submits `orchestrate_rl_no_turnrd --env-name alfworld` as a
# single Modal job that drives all rounds cloud-side. Per-round =
# train_loop_alfworld only (decomposer='progress' at α=1.0 is inert →
# no TurnRD trainer step needed).
#
# Workflow
# ========
# 1. `modal deploy infra/app_train_loop.py`
# 2. `modal deploy infra/app_orchestrator.py`
# 3. `modal run --detach infra/app_orchestrator.py::orchestrate_rl_no_turnrd \
#       --env-name alfworld --config /workspace/<CONFIG> ...`
# 4. Exit. Laptop free to shut down.
#
# Cost / envelope
# ===============
# Same as the local sibling (~4-5 hours, ~$25) + ~$0.05 for the
# orchestrator container's CPU+RAM.
#
# Resume support
# ==============
# Auto-resume via per-round sentinels under
# `${ADAPTER_DIR}/${RUN_PREFIX}_seed${SEED}_round{NN}_done.json`.
set -euo pipefail
cd "$(dirname "$0")/.."

SFT_ADAPTER="${SFT_ADAPTER:-}"
if [[ -z "${SFT_ADAPTER}" ]]; then
  echo "ERROR: SFT_ADAPTER env var is required."
  echo "       Set it to the same rank-32 MLP-target SFT adapter used by v3 SOTA, e.g.:"
  echo "         SFT_ADAPTER=/vol/checkpoints/sft_alfworld_v2_e3_<ts> \\"
  echo "           bash scripts/run_flatGRPO_alfworld_SOTA_a100_cloud.sh"
  exit 1
fi

CONFIG=${CONFIG:-configs/flatGRPO_alfworld_SOTA_10round_mlpr32_a100.json}
RUN_PREFIX=${RUN_PREFIX:-flatGRPO_alfworld_SOTA_a100_cloud}
ROUNDS=${ROUNDS:-10}
START_ROUND=${START_ROUND:-0}
EPS_PER_ROUND=${EPS_PER_ROUND:-80}
EVAL_EPS=${EVAL_EPS:-100}
EVAL_TASK_BASE=${EVAL_TASK_BASE:-6500}
SEED=${SEED:-41}
ROLLOUT_TEMP=${ROLLOUT_TEMP:-1.0}
ADAPTER_DIR=${ADAPTER_DIR:-/vol/checkpoints}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.40}
SYNC_EVERY=${SYNC_EVERY:-8}

BASE_OFFSET=$(( SEED * ROUNDS * EPS_PER_ROUND ))
TRAIN_HI=$(( BASE_OFFSET + ROUNDS * EPS_PER_ROUND ))

echo "═══════════════════════════════════════════════════════════════════════"
echo "AlfWorld SOTA — Flat GRPO α=1.0 baseline (CLOUD orchestrator)"
echo "  config             : ${CONFIG}"
echo "  sft_adapter (R0)   : ${SFT_ADAPTER}"
echo "  run-name-prefix    : ${RUN_PREFIX}"
echo "  rounds             : ${ROUNDS} (start at R${START_ROUND})"
echo "  episodes/round     : ${EPS_PER_ROUND}"
echo "  eval episodes      : ${EVAL_EPS}"
echo "  seed               : ${SEED}"
echo "  rollout temperature: ${ROLLOUT_TEMP}"
echo "  base_task_id_offset: ${BASE_OFFSET}"
echo "  train task range   : [${BASE_OFFSET}, ${TRAIN_HI})"
echo "  eval task range    : [${EVAL_TASK_BASE}, $((EVAL_TASK_BASE + EVAL_EPS)))"
echo "  adapter dir (vol)  : ${ADAPTER_DIR}"
echo ""
echo "  H-GRPO blend       : α=1.0 (pure trajectory advantage; per-turn signal OFF)"
echo "  Decomposer         : 'progress' (inert at α=1.0; no TurnRD model built)"
echo "  Mode               : CLOUD orchestrator (24h Modal container, no TurnRD step)."
echo "═══════════════════════════════════════════════════════════════════════"

echo ""
echo ">>> Step 1/3: deploying cs224r-hgpo-train-loop ..."
modal deploy infra/app_train_loop.py

echo ""
echo ">>> Step 2/3: deploying cs224r-hgpo-orchestrator ..."
modal deploy infra/app_orchestrator.py

echo ""
echo ">>> Step 3/3: submitting cs224r-hgpo-orchestrator (detached) ..."
modal run --detach infra/app_orchestrator.py::orchestrate_rl_no_turnrd \
  --config "/workspace/${CONFIG}" \
  --sft-adapter "${SFT_ADAPTER}" \
  --env-name alfworld \
  --rounds "${ROUNDS}" \
  --start-round "${START_ROUND}" \
  --episodes-per-round "${EPS_PER_ROUND}" \
  --eval-episodes "${EVAL_EPS}" \
  --eval-task-id-base "${EVAL_TASK_BASE}" \
  --seed "${SEED}" \
  --rollout-temperature "${ROLLOUT_TEMP}" \
  --run-name-prefix "${RUN_PREFIX}" \
  --adapter-dir "${ADAPTER_DIR}" \
  --gpu-mem-util "${GPU_MEM_UTIL}" \
  --sync-every "${SYNC_EVERY}"

echo ""
echo ">>> Submitted. Laptop is now free to disconnect / shut down."
echo ""
echo "Track progress:"
echo "  modal app logs <orchestrator-app-id>"
echo ""
echo "Pull per-round eval summary after completion:"
echo "  modal volume get cs224r-hgpo-vol \\"
echo "    manifests/${RUN_PREFIX}_seed${SEED}_orchestrator_summary/summary.json ./"
echo ""
echo "Resume from a partial run:"
echo "  START_ROUND=<n> SFT_ADAPTER=... \\"
echo "    bash scripts/run_flatGRPO_alfworld_SOTA_a100_cloud.sh"
