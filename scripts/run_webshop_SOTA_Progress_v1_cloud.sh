#!/usr/bin/env bash
# WebShop SOTA — HGPO-Progress (Method C) v1 — CLOUD-ORCHESTRATED variant.
#
# Difference vs. run_webshop_SOTA_Progress_v1.sh
# ==============================================
# The sibling runs a per-round bash for-loop locally. This script
# instead submits `orchestrate_rl_no_turnrd` as a single Modal job that
# drives all rounds cloud-side. Per-round = train_loop only (Method C
# 'progress' decomposer is parameter-free; no TurnRD trainer step).
#
# Workflow
# ========
# 1. `modal deploy infra/app_train_loop.py`
# 2. `modal deploy infra/app_orchestrator.py`
# 3. `modal run --detach infra/app_orchestrator.py::orchestrate_rl_no_turnrd \
#       --env-name webshop --config /workspace/<CONFIG> ...`
# 4. Exit. Laptop free to shut down.
#
# Cost / envelope
# ===============
# Same as the local sibling (~2.5-3 hours, ~$15) + ~$0.05 for the
# orchestrator container.
#
# Resume support
# ==============
# Per-round sentinels under `${ADAPTER_DIR}/${RUN_PREFIX}_seed${SEED}_round{NN}_done.json`.
set -euo pipefail
cd "$(dirname "$0")/.."

SFT_ADAPTER="${SFT_ADAPTER:-}"
if [[ -z "${SFT_ADAPTER}" ]]; then
  echo "ERROR: SFT_ADAPTER env var is required."
  echo "       Set it to the rank-32 + 7-MLP-target SFT adapter, e.g.:"
  echo "         SFT_ADAPTER=/vol/checkpoints/sft_webshop_v3_mlpr32_<ts> \\"
  echo "           bash scripts/run_webshop_SOTA_Progress_v1_cloud.sh"
  exit 1
fi

CONFIG=${CONFIG:-configs/Progress_webshop_SOTA_10round_mlpr32_v1.json}
RUN_PREFIX=${RUN_PREFIX:-webshop_Progress_v1_cloud}
ROUNDS=${ROUNDS:-10}
START_ROUND=${START_ROUND:-0}
EPS_PER_ROUND=${EPS_PER_ROUND:-80}
EVAL_EPS=${EVAL_EPS:-100}
EVAL_TASK_BASE=${EVAL_TASK_BASE:-6500}
SEED=${SEED:-41}
ROLLOUT_TEMP=${ROLLOUT_TEMP:-1.0}
ADAPTER_DIR=${ADAPTER_DIR:-/vol/checkpoints}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.30}
SYNC_EVERY=${SYNC_EVERY:-8}

BASE_OFFSET=$(( SEED * ROUNDS * EPS_PER_ROUND ))
TRAIN_HI=$(( BASE_OFFSET + ROUNDS * EPS_PER_ROUND ))

echo "═══════════════════════════════════════════════════════════════════════"
echo "WebShop SOTA — HGPO-Progress (Method C) v1 (CLOUD orchestrator)"
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
  --env-name webshop \
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
echo "    bash scripts/run_webshop_SOTA_Progress_v1_cloud.sh"
