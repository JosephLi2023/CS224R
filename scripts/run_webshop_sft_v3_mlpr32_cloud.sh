#!/usr/bin/env bash
# WebShop SFT v3 pipeline (rank-32 + MLP targets) — CLOUD-ORCHESTRATED variant.
#
# Difference vs. run_webshop_sft_v3_mlpr32.sh
# ===========================================
# The sibling runs the 4 pipeline stages locally on the laptop, calling
# `modal run [--detach]` for each. The non-detached install/gen/eval
# steps require the laptop to stay alive; only step 3 (sft_train) is
# `--detach`. If the laptop sleeps before step 4 finishes, the eval
# never runs.
#
# This script instead deploys all required apps once, then submits
# infra/app_orchestrator.py::orchestrate_sft_pipeline as a single
# `modal run --detach` call. The pipeline runs INSIDE a Modal container
# with a 24-hour timeout. Per-stage sentinels enable auto-resume on
# preemption-restart.
#
# Workflow
# ========
# 1. `modal deploy infra/app_webshop_install.py`     (stages 1a/1b/1c)
# 2. `modal deploy infra/app_webshop_sft_gen.py`     (stage 2)
# 3. `modal deploy infra/app_sft_train.py`           (stage 3)
# 4. `modal deploy infra/app_train_loop.py`          (stage 4 eval)
# 5. `modal deploy infra/app_orchestrator.py`        (the driver)
# 6. `modal run --detach infra/app_orchestrator.py::orchestrate_sft_pipeline \
#       --run-name <name> --mode <full|skip-install|skip-gen|train-only|eval-only> ...`
#
# Cost / envelope
# ===============
# Same as the local sibling (~$15-30 train + ~$1 gen + ~$3-5 eval) +
# ~$0.05 for the orchestrator container's CPU+RAM over the full ~4-6 hr.
#
# Resume support
# ==============
# Auto-resume via per-stage sentinels under
# `/vol/checkpoints/sft_pipeline_${RUN_NAME}_stage{1a,1b,1c,2,3,4}_done.json`.
# To force re-execution of a completed stage, delete its sentinel or
# re-run with --auto-resume=false.
set -uo pipefail
cd "$(dirname "$0")/.."

MODE_FLAG="${1:---full}"

case "${MODE_FLAG}" in
    --full)         MODE="full" ;;
    --skip-install) MODE="skip-install" ;;
    --skip-gen)     MODE="skip-gen" ;;
    --train-only)   MODE="train-only" ;;
    --eval-only)    MODE="eval-only" ;;
    *)
        echo "ERROR: unrecognized mode '${MODE_FLAG}'." >&2
        echo "Valid: --full | --skip-install | --skip-gen | --train-only | --eval-only" >&2
        exit 2
        ;;
esac

# ---------- Tunable knobs (env-var overrides; same names as local sibling) ----
N_SESSIONS=${N_SESSIONS:-2000}
INCLUDE_HUMAN_TRAJS=${INCLUDE_HUMAN_TRAJS:-true}
case "${INCLUDE_HUMAN_TRAJS}" in
    0|false|False|FALSE|no|No|NO) INCLUDE_HUMAN_TRAJS_FLAG=(--no-include-human-trajs) ;;
    *) INCLUDE_HUMAN_TRAJS_FLAG=(--include-human-trajs) ;;
esac
HUMAN_TRAJS_MIN_REWARD=${HUMAN_TRAJS_MIN_REWARD:-0.5}
MAX_RESULT_PAGES=${MAX_RESULT_PAGES:-5}
MAX_STEPS_PER_EP=${MAX_STEPS_PER_EP:-25}
REWARD_THRESHOLD=${REWARD_THRESHOLD:-0.99}

EPOCHS=${EPOCHS:-6}
LR=${LR:-5e-5}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-2048}
GRAD_ACCUM=${GRAD_ACCUM:-8}
MIN_REWARD=${MIN_REWARD:-0.5}
EVAL_EPS=${EVAL_EPS:-200}
EVAL_TASK_BASE=${EVAL_TASK_BASE:-6500}

LORA_RANK=${LORA_RANK:-32}
LORA_TARGETS=${LORA_TARGETS:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}

RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
RUN_NAME=${RUN_NAME:-sft_webshop_v3_mlpr32_cloud_${RUN_TS}}
DATA_PATH=${DATA_PATH:-/vol/data/webshop/oracle_trajs.jsonl}
ADAPTER_PATH=${ADAPTER_PATH:-}
SENTINEL_DIR=${SENTINEL_DIR:-/vol/checkpoints}
EVAL_CONFIG=${EVAL_CONFIG:-/workspace/configs/SFTOnly_webshop_mlpr32.json}
EVAL_GPU_MEM_UTIL=${EVAL_GPU_MEM_UTIL:-0.30}

# ---------- Pre-flight checks --------------------------------------------------
if ! command -v modal >/dev/null 2>&1; then
    echo "ERROR: modal CLI not on PATH." >&2
    echo "  Install with:  pip install modal" >&2
    echo "  Authenticate:  modal token new" >&2
    exit 2
fi

if ! modal app list >/dev/null 2>&1; then
    echo "ERROR: modal CLI not authenticated. Run: modal token new" >&2
    exit 2
fi

for f in \
    infra/app_webshop_install.py \
    infra/app_webshop_sft_gen.py \
    infra/app_sft_train.py \
    infra/app_train_loop.py \
    infra/app_orchestrator.py \
    configs/SFTOnly_webshop_mlpr32.json
do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: required file missing: ${f}" >&2
        echo "       Are you in the repo root and on the right branch?" >&2
        exit 2
    fi
done

# ---------- Header --------------------------------------------------------------
echo "═══════════════════════════════════════════════════════════════════════"
echo "WebShop SFT v3 pipeline (rank-${LORA_RANK} + MLP targets) — CLOUD orchestrator"
echo "  mode               : ${MODE}"
echo "  run_name           : ${RUN_NAME}"
echo "  data_path  (vol)   : ${DATA_PATH}"
echo "  adapter_path arg   : ${ADAPTER_PATH:-<auto-resolve>}"
echo "  sentinel_dir       : ${SENTINEL_DIR}"
echo "  eval_config        : ${EVAL_CONFIG}"
echo "  ─── gen ───────────────────────"
echo "  n_sessions         : ${N_SESSIONS}"
echo "  include_human_trajs: ${INCLUDE_HUMAN_TRAJS}"
echo "  max_result_pages   : ${MAX_RESULT_PAGES}"
echo "  max_steps_per_ep   : ${MAX_STEPS_PER_EP}"
echo "  reward_threshold   : ${REWARD_THRESHOLD}"
echo "  ─── train ─────────────────────"
echo "  epochs             : ${EPOCHS}"
echo "  learning_rate      : ${LR}"
echo "  max_seq_len        : ${MAX_SEQ_LEN}"
echo "  grad_accum         : ${GRAD_ACCUM}"
echo "  min_reward         : ${MIN_REWARD}"
echo "  ─── LoRA arch ─────────────────"
echo "  lora_rank          : ${LORA_RANK}"
echo "  lora_target_mods   : ${LORA_TARGETS}"
echo "  ─── eval ──────────────────────"
echo "  eval_episodes      : ${EVAL_EPS}"
echo "  eval_task_id_base  : ${EVAL_TASK_BASE}"
echo "  eval_gpu_mem_util  : ${EVAL_GPU_MEM_UTIL}"
echo ""
echo "  Mode               : CLOUD orchestrator (24h Modal container)."
echo "                       Laptop is free to shut down after submit."
echo "═══════════════════════════════════════════════════════════════════════"

echo ""
echo ">>> Step 1/6: deploying cs224r-hgpo-webshop-install ..."
modal deploy infra/app_webshop_install.py

echo ""
echo ">>> Step 2/6: deploying cs224r-hgpo-webshop-sft-gen ..."
modal deploy infra/app_webshop_sft_gen.py

echo ""
echo ">>> Step 3/6: deploying cs224r-hgpo-sft-train ..."
modal deploy infra/app_sft_train.py

echo ""
echo ">>> Step 4/6: deploying cs224r-hgpo-train-loop ..."
modal deploy infra/app_train_loop.py

echo ""
echo ">>> Step 5/6: deploying cs224r-hgpo-orchestrator ..."
modal deploy infra/app_orchestrator.py

echo ""
echo ">>> Step 6/6: submitting cs224r-hgpo-orchestrator::orchestrate_sft_pipeline (detached) ..."
modal run --detach infra/app_orchestrator.py::orchestrate_sft_pipeline \
  --run-name "${RUN_NAME}" \
  --mode "${MODE}" \
  --n-sessions "${N_SESSIONS}" \
  "${INCLUDE_HUMAN_TRAJS_FLAG[@]}" \
  --human-trajs-min-reward "${HUMAN_TRAJS_MIN_REWARD}" \
  --max-result-pages "${MAX_RESULT_PAGES}" \
  --max-steps-per-episode "${MAX_STEPS_PER_EP}" \
  --reward-threshold "${REWARD_THRESHOLD}" \
  --epochs "${EPOCHS}" \
  --learning-rate "${LR}" \
  --max-seq-len "${MAX_SEQ_LEN}" \
  --grad-accum "${GRAD_ACCUM}" \
  --min-reward "${MIN_REWARD}" \
  --lora-rank "${LORA_RANK}" \
  --lora-target-modules "${LORA_TARGETS}" \
  --eval-episodes "${EVAL_EPS}" \
  --eval-task-id-base "${EVAL_TASK_BASE}" \
  --data-path "${DATA_PATH}" \
  --adapter-path "${ADAPTER_PATH}" \
  --sentinel-dir "${SENTINEL_DIR}" \
  --eval-config "${EVAL_CONFIG}" \
  --eval-gpu-mem-util "${EVAL_GPU_MEM_UTIL}"

echo ""
echo ">>> Submitted. Laptop is now free to disconnect / shut down."
echo ""
echo "Track progress:"
echo "  modal app logs <orchestrator-app-id>   # see the URL printed above"
echo ""
echo "Per-stage sentinels (delete to force re-execution of a stage):"
echo "  ${SENTINEL_DIR}/sft_pipeline_${RUN_NAME}_stage{1a,1b,1c,2,3,4}_done.json"
echo ""
echo "Pull pipeline summary after completion:"
echo "  modal volume get cs224r-hgpo-vol \\"
echo "    manifests/sft_pipeline_${RUN_NAME}_orchestrator_summary/summary.json ./"
echo ""
echo "Resume after partial failure:"
echo "  RUN_NAME=${RUN_NAME} bash scripts/run_webshop_sft_v3_mlpr32_cloud.sh --full"
echo "    (already-completed stages skipped via sentinel scan; same run_name reuses adapter dir)"
