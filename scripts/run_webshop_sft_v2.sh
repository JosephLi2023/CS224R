#!/usr/bin/env bash
# run_webshop_sft_v2.sh — Train an improved WebShop SFT baseline with the new
# template-drift-fixed prompt renderer.
#
# Self-contained pipeline for a teammate running on a FRESH Modal account:
#   1. Download WebShop env data (products + sampled goals) — one-time.
#   2. Download WebShop human-trajectory SFT data (~50 trajs, ~745 examples).
#   3. Train SFT v2 with longer training + larger context window.
#   4. Eval on 200 held-out tasks.
#
# Recipe (v1 → v2):
#   epochs             3   → 6      (2×; pairs with the lower LR)
#   learning_rate      1e-4 → 5e-5  (½; pairs with the higher epoch count)
#   max_seq_len        1024 → 2048  (2×; WebShop product pages can be long)
#   grad_accum         4   → 8      (2×; compensates for 2× attention memory)
#   eval_episodes      ~50 → 200    (4×; tightens 95% CI from ±13.9pp to ±6.9pp)
#
# IMPORTANT: WebShop's SFT data is a FIXED set of ~50 human trajectories
# (~745 SFT examples after URL-diff parsing). Unlike AlfWorld, there is no
# expert collector to scale this up. The realistic SFT ceiling on Qwen2.5-1.5B
# with this dataset is ~0.45-0.55, NOT 0.60+. The headline gain over v1
# comes from (a) the template-drift fix already shipped to
# `src/datasets/sft_webshop.py` (eliminates the v3 R=0 root cause), and
# (b) longer training with a smaller LR.
#
# Cost estimate: ~$15-30, ~5-7 hr wall-clock end-to-end.
#   Step 1 (env data):   ~$1, ~10 min CPU
#   Step 2 (human trajs):free, ~5 min (gdown from Google Drive)
#   Step 3 (train):      ~$10-20 on A100-80GB, ~2-4 hr at 6 epochs × ~745 examples
#   Step 4 (eval):       ~$3-5, ~30 min for 200 episodes
#
# Usage:
#   bash scripts/run_webshop_sft_v2.sh [--dry-run|--full|--skip-data|--train-only|--eval-only]
#
# Modes:
#   --dry-run     : print every modal command without running anything (free).
#   --full        : run all 4 steps end-to-end (default).
#   --skip-data   : assume both data sources already downloaded; start at step 3.
#   --train-only  : only run step 3 (assumes data already present).
#   --eval-only   : only run step 4 (assumes adapter exists at $ADAPTER_PATH; set ADAPTER_PATH env var).
#
# Override knobs via env vars (defaults shown):
#   EPOCHS=6  LR=5e-5  MAX_SEQ_LEN=2048  GRAD_ACCUM=8  MIN_REWARD=0.5
#   EVAL_EPS=200  EVAL_TASK_BASE=6500
#   RUN_TS=$(date +%Y%m%d_%H%M%S)  RUN_NAME=sft_v2_${RUN_TS}
#   SMALL_DATA=True  (use the small WebShop product split; recommended)
#
# Resume after partial failure:
#   - Data download crashed: re-run --full (downloads are idempotent)
#   - Train crashed:         re-run with --train-only
#   - Eval crashed:          re-run with --eval-only ADAPTER_PATH=/vol/checkpoints/<run_name>

set -uo pipefail

MODE="${1:---full}"

case "${MODE}" in
    --dry-run|--full|--skip-data|--train-only|--eval-only) ;;
    *)
        echo "ERROR: unrecognized mode '${MODE}'." >&2
        echo "Valid: --dry-run | --full | --skip-data | --train-only | --eval-only" >&2
        exit 2
        ;;
esac

# ---------- Tunable knobs (env-var overrides) ---------------------------------
EPOCHS=${EPOCHS:-6}
LR=${LR:-5e-5}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-2048}
GRAD_ACCUM=${GRAD_ACCUM:-8}
MIN_REWARD=${MIN_REWARD:-0.5}
EVAL_EPS=${EVAL_EPS:-200}
EVAL_TASK_BASE=${EVAL_TASK_BASE:-6500}
SMALL_DATA=${SMALL_DATA:-True}

RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
RUN_NAME=${RUN_NAME:-sft_v2_${RUN_TS}}
ADAPTER_PATH=${ADAPTER_PATH:-/vol/checkpoints/${RUN_NAME}}
LOG_DIR=${LOG_DIR:-/tmp}
LOG_FILE="${LOG_DIR}/sft_webshop_v2_${RUN_TS}.log"

# ---------- Pre-flight checks --------------------------------------------------
if ! command -v modal >/dev/null 2>&1; then
    echo "ERROR: modal CLI not on PATH." >&2
    echo "  Install with:  pip install modal" >&2
    echo "  Authenticate:  modal token new" >&2
    exit 2
fi

if [[ "${MODE}" != "--dry-run" ]]; then
    if ! modal app list >/dev/null 2>&1; then
        echo "ERROR: modal CLI not authenticated. Run: modal token new" >&2
        exit 2
    fi
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}" || { echo "ERROR: could not cd to ${REPO_ROOT}" >&2; exit 1; }

for app in \
    infra/app_data.py \
    infra/app_sft_train.py \
    infra/app_train_loop.py \
    configs/SFTOnly.json
do
    if [[ ! -f "${app}" ]]; then
        echo "ERROR: required file missing: ${app}" >&2
        echo "       Are you in the repo root and on the right branch?" >&2
        exit 2
    fi
done

mkdir -p "${LOG_DIR}"

# ---------- Header --------------------------------------------------------------
echo "═══════════════════════════════════════════════════════════════════════"
echo "WebShop SFT v2 boost  (template-drift-fixed; target 0.45-0.55)"
echo "  mode               : ${MODE}"
echo "  run_name           : ${RUN_NAME}"
echo "  adapter_path (vol) : ${ADAPTER_PATH}"
echo "  log file (local)   : ${LOG_FILE}"
echo "  ─── data ──────────────────────"
echo "  SMALL_DATA         : ${SMALL_DATA}   (small WebShop split is recommended)"
echo "  ─── train ─────────────────────"
echo "  epochs             : ${EPOCHS}        (v1: 3)"
echo "  learning_rate      : ${LR}     (v1: 1e-4)"
echo "  max_seq_len        : ${MAX_SEQ_LEN}     (v1: 1024)"
echo "  grad_accum         : ${GRAD_ACCUM}        (v1: 4)"
echo "  min_reward         : ${MIN_REWARD}      (v1: 0.5)"
echo "  ─── eval ──────────────────────"
echo "  eval_episodes      : ${EVAL_EPS}      (v1: ~50)"
echo "  eval_task_id_base  : ${EVAL_TASK_BASE}     (slice [${EVAL_TASK_BASE}, $((EVAL_TASK_BASE + EVAL_EPS))))"
echo "═══════════════════════════════════════════════════════════════════════"

run_step () {
    local label="$1"; shift
    if [[ "${MODE}" == "--dry-run" ]]; then
        echo ""
        echo "[DRY-RUN] ${label}"
        echo "  $*"
        return 0
    fi
    echo ""
    echo "▶ ${label}  (start: $(date +%H:%M:%S))"
    echo "  $*"
    "$@" 2>&1 | tee -a "${LOG_FILE}"
    local rc=${PIPESTATUS[0]}
    echo "▶ ${label}  (end: $(date +%H:%M:%S), exit=${rc})"
    if [[ ${rc} -ne 0 ]]; then
        echo "ERROR: ${label} exited ${rc}. Aborting." >&2
        echo "       Full log: ${LOG_FILE}" >&2
        return ${rc}
    fi
    return 0
}

# ---------- Step 1: download WebShop env data (products + goals) --------------
if [[ "${MODE}" == "--full" || "${MODE}" == "--dry-run" ]]; then
    echo ""
    echo "═══ Step 1/4: download WebShop env data ═══"
    echo "  (Products, attributes, sampled goals. Idempotent on re-runs.)"
    run_step "step1_env_data" \
        modal run infra/app_data.py --action download --small "${SMALL_DATA}" \
        || exit $?
fi

# ---------- Step 2: download WebShop human SFT trajectories -------------------
if [[ "${MODE}" == "--full" || "${MODE}" == "--dry-run" ]]; then
    echo ""
    echo "═══ Step 2/4: download WebShop human SFT trajectories ═══"
    echo "  (~50 trajectories from the WebShop paper's released set, via gdown."
    echo "   Yields ~745 SFT examples after URL-diff parsing.)"
    run_step "step2_sft_data" \
        modal run infra/app_data.py --action download_human_trajs \
        || exit $?
fi

# ---------- Step 3: train SFT v2 ----------------------------------------------
if [[ "${MODE}" == "--full" || "${MODE}" == "--skip-data" || "${MODE}" == "--train-only" || "${MODE}" == "--dry-run" ]]; then
    echo ""
    echo "═══ Step 3/4: train SFT v2 (${EPOCHS} epochs, lr=${LR}, seq_len=${MAX_SEQ_LEN}) ═══"
    echo "  Writes adapter to /vol/checkpoints/${RUN_NAME} on the Modal volume."
    echo "  Wall-clock ~2-4 hr on A100-80GB."
    echo "  NOTE: this run uses the FIXED prompt renderer in src/datasets/sft_webshop.py"
    echo "        (template-drift bug eliminated; v1 ckpts predate this fix)."
    run_step "step3_train" \
        modal run --detach infra/app_sft_train.py \
            --epochs "${EPOCHS}" \
            --learning-rate "${LR}" \
            --min-reward "${MIN_REWARD}" \
            --max-seq-len "${MAX_SEQ_LEN}" \
            --grad-accum "${GRAD_ACCUM}" \
            --run-name "${RUN_NAME}" \
        || exit $?
fi

# ---------- Step 4: eval the new SFT v2 ckpt ----------------------------------
if [[ "${MODE}" == "--full" || "${MODE}" == "--skip-data" || "${MODE}" == "--train-only" || "${MODE}" == "--eval-only" || "${MODE}" == "--dry-run" ]]; then
    echo ""
    echo "═══ Step 4/4: eval SFT v2 on ${EVAL_EPS} held-out WebShop tasks ═══"
    echo "  total_episodes=0 in SFTOnly.json skips RL and goes straight to"
    echo "  the held-out eval pass on slice [${EVAL_TASK_BASE}, $((EVAL_TASK_BASE + EVAL_EPS)))."
    echo "  Wall-clock ~30 min."
    run_step "step4_eval" \
        modal run infra/app_train_loop.py::train_loop_webshop \
            --config "/workspace/configs/SFTOnly.json" \
            --n-episodes 0 \
            --eval-episodes "${EVAL_EPS}" \
            --eval-task-id-base "${EVAL_TASK_BASE}" \
            --sft-adapter "${ADAPTER_PATH}" \
            --gpu-mem-util 0.30 \
            --run-name "SFTOnly_webshop_v2_eval_${RUN_TS}" \
        || exit $?
fi

# ---------- Footer -------------------------------------------------------------
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
if [[ "${MODE}" == "--dry-run" ]]; then
    echo "Dry-run complete. Re-invoke without --dry-run to actually launch."
else
    echo "Done.  log: ${LOG_FILE}"
    echo ""
    echo "Outputs on Modal volume:"
    echo "  human SFT data : /vol/data/webshop/human_trajs/   (~50 trajs)"
    echo "  adapter        : ${ADAPTER_PATH}"
    echo "  eval           : /vol/manifests/SFTOnly_webshop_v2_eval_${RUN_TS}_*/train_log.json"
    echo ""
    echo "Compare against v1 (sft_v3_20260504_154752, eval=null in current manifest)."
    echo ""
    echo "Realistic decision tree (Qwen2.5-1.5B + ~745 SFT examples):"
    echo "  ≥ 0.55  → strong win; the template-drift fix + longer training paid off"
    echo "  ∈ [0.45, 0.55) → expected range; use as new baseline"
    echo "  ∈ [0.35, 0.45) → marginal; data ceiling is the bottleneck (only 745 examples)"
    echo "  < 0.35  → investigate (check that the fixed prompt renderer is on PATH)"
    echo ""
    echo "NOTE: WebShop SFT-only on Qwen2.5-1.5B is data-bottlenecked (~745 examples)."
    echo "      Reaching 0.60+ likely requires self-distillation from a stronger policy"
    echo "      OR a bigger base model (3B/7B). The 60% target is aspirational."
    echo ""
    echo "To download the eval log locally:"
    echo "  modal volume get cs224r-hgpo-vol \\"
    echo "    /manifests/SFTOnly_webshop_v2_eval_${RUN_TS}_*/train_log.json eval.json"
fi
echo "═══════════════════════════════════════════════════════════════════════"
