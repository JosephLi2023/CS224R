#!/usr/bin/env bash
# run_webshop_sft_v3_mlpr32.sh — WebShop SFT v3 baseline at LoRA rank-32 + 7 MLP targets.
#
# Background
# ==========
# The 3 WebShop SoTA RL launchers (run_webshop_SOTA_attention_v1.sh,
# run_webshop_SOTA_flatGRPO_v1.sh, run_webshop_SOTA_LLMJudge_v1.sh) all
# declare a rank-32 + 7-MLP-target LoRA arch (the AlfWorld SOTA recipe
# transplant) but warm-start from `/vol/checkpoints/sft_v3_20260504_154752`
# which was trained at rank-16 attention-only. The mismatched
# `policy.load_adapter(...)` silently keeps the attention slice and
# leaves the MLP LoRAs randomly-initialised. This launcher produces a
# rank-32 + 7-MLP SFT adapter trained on a larger oracle-generated
# corpus so the warm-start actually matches the RL arch.
#
# Pipeline (4 steps)
# ==================
#   1. install — WebShop env (pip-install editable + spaCy + BM25 index).
#                Idempotent on the Modal volume; ~20 min the first time.
#   2. gen     — `infra/app_webshop_sft_gen.py` runs a deterministic
#                oracle over N_SESSIONS sessions, optionally
#                concatenating the upstream ~50 gdown human trajectories.
#                ~30 min CPU, ~$0.50.
#   3. train   — `infra/app_sft_train.py --data-path <jsonl> --lora-rank 32
#                --lora-target-modules <7 modules>`. ~3-5 hr A100, ~$15-30.
#   4. eval    — smoke eval against `configs/SFTOnly_webshop_mlpr32.json`
#                on `[6500, 6600)`. Confirms the adapter loads cleanly
#                into the rank-32 + MLP policy AND has non-degenerate
#                eval pct_success. ~30 min A100, ~$3-5.
#
# Usage
# =====
#   bash scripts/run_webshop_sft_v3_mlpr32.sh [--dry-run|--full|--skip-install|--skip-gen|--train-only|--eval-only]
#
# Override knobs via env vars (defaults shown):
#   N_SESSIONS=2000                       # oracle generator sessions to attempt
#   INCLUDE_HUMAN_TRAJS=true              # also concat upstream gdown human trajs
#   EPOCHS=6
#   LR=5e-5
#   MAX_SEQ_LEN=2048
#   GRAD_ACCUM=8
#   MIN_REWARD=0.5
#   EVAL_EPS=200
#   EVAL_TASK_BASE=6500
#   LORA_RANK=32
#   LORA_TARGETS=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
#   RUN_TS=$(date +%Y%m%d_%H%M%S)
#   RUN_NAME=sft_webshop_v3_mlpr32_${RUN_TS}
#   DATA_PATH=/vol/data/webshop/oracle_trajs.jsonl
#   LOG_DIR=/tmp
#
# Resume after partial failure
# ============================
#   - Install crashed:    re-run --full (install is idempotent on the volume).
#   - Gen crashed:        re-run --skip-install (gen overwrites the JSONL fresh).
#   - Train crashed:      re-run --train-only (data file is preserved).
#   - Eval crashed:       re-run --eval-only ADAPTER_PATH=/vol/checkpoints/<run_name>.

set -uo pipefail

MODE="${1:---full}"

case "${MODE}" in
    --dry-run|--full|--skip-install|--skip-gen|--train-only|--eval-only) ;;
    *)
        echo "ERROR: unrecognized mode '${MODE}'." >&2
        echo "Valid: --dry-run | --full | --skip-install | --skip-gen | --train-only | --eval-only" >&2
        exit 2
        ;;
esac

# ---------- Tunable knobs (env-var overrides) ---------------------------------
N_SESSIONS=${N_SESSIONS:-2000}
INCLUDE_HUMAN_TRAJS=${INCLUDE_HUMAN_TRAJS:-true}
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

# LoRA arch knobs. Defaults match the rank-32 + 7-MLP-target recipe used
# by the 3 WebShop SOTA RL launchers (transplant from AlfWorld SOTA).
LORA_RANK=${LORA_RANK:-32}
LORA_TARGETS=${LORA_TARGETS:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}

RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
RUN_NAME=${RUN_NAME:-sft_webshop_v3_mlpr32_${RUN_TS}}
DATA_PATH=${DATA_PATH:-/vol/data/webshop/oracle_trajs.jsonl}
ADAPTER_PATH=${ADAPTER_PATH:-/vol/checkpoints/${RUN_NAME}}
LOG_DIR=${LOG_DIR:-/tmp}
LOG_FILE="${LOG_DIR}/sft_webshop_v3_mlpr32_${RUN_TS}.log"

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

# Verify the apps we'll invoke exist (catches stale-clone / wrong-branch issues).
for app in \
    infra/app_webshop_install.py \
    infra/app_webshop_sft_gen.py \
    infra/app_sft_train.py \
    infra/app_train_loop.py \
    configs/SFTOnly_webshop_mlpr32.json
do
    if [[ ! -f "${app}" ]]; then
        echo "ERROR: required file missing: ${app}" >&2
        echo "       Are you in the repo root and on the right branch?" >&2
        exit 2
    fi
done

# ---------- Header --------------------------------------------------------------
mkdir -p "${LOG_DIR}"

echo "═══════════════════════════════════════════════════════════════════════"
echo "WebShop SFT v3 (rank-${LORA_RANK} + MLP targets)"
echo "  mode               : ${MODE}"
echo "  run_name           : ${RUN_NAME}"
echo "  data_path  (vol)   : ${DATA_PATH}"
echo "  adapter_path (vol) : ${ADAPTER_PATH}"
echo "  log file (local)   : ${LOG_FILE}"
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

# ---------- Step 1: install WebShop env (BM25 index, spaCy, editable repo) ----
if [[ "${MODE}" == "--full" || "${MODE}" == "--dry-run" ]]; then
    echo ""
    echo "═══ Step 1/4: install WebShop env onto Modal volume ═══"
    echo "  (Idempotent. Takes ~20 min the first time; near-instant on re-runs.)"
    run_step "step1a_pip_install_webshop" \
        modal run infra/app_webshop_install.py --action pip_install \
        || exit $?
    run_step "step1b_download_spacy" \
        modal run infra/app_webshop_install.py --action download_spacy \
        || exit $?
    run_step "step1c_build_index_1k" \
        modal run infra/app_webshop_install.py --action build_index_1k \
        || exit $?
fi

# ---------- Step 2: generate oracle SFT trajectories --------------------------
if [[ "${MODE}" == "--full" || "${MODE}" == "--skip-install" || "${MODE}" == "--dry-run" ]]; then
    echo ""
    echo "═══ Step 2/4: generate ${N_SESSIONS} oracle SFT trajectories ═══"
    echo "  Writes JSONL to ${DATA_PATH} on the Modal volume."
    echo "  Wall-clock ~30-60 min (CPU-only, deterministic oracle)."
    run_step "step2_gen" \
        modal run infra/app_webshop_sft_gen.py::main \
            --action generate \
            --n-sessions "${N_SESSIONS}" \
            --output-path "${DATA_PATH}" \
            --max-result-pages "${MAX_RESULT_PAGES}" \
            --max-steps-per-episode "${MAX_STEPS_PER_EP}" \
            --reward-threshold "${REWARD_THRESHOLD}" \
            --include-human-trajs "${INCLUDE_HUMAN_TRAJS}" \
            --human-trajs-min-reward "${HUMAN_TRAJS_MIN_REWARD}" \
        || exit $?
fi

# ---------- Step 3: train SFT v3 ----------------------------------------------
if [[ "${MODE}" == "--full" || "${MODE}" == "--skip-install" || "${MODE}" == "--skip-gen" || "${MODE}" == "--train-only" || "${MODE}" == "--dry-run" ]]; then
    echo ""
    echo "═══ Step 3/4: train SFT v3 (${EPOCHS} epochs, lr=${LR}, seq_len=${MAX_SEQ_LEN}) ═══"
    echo "  Writes adapter to ${ADAPTER_PATH} on the Modal volume."
    echo "  Wall-clock ~3-5 hr on A100-80GB."
    run_step "step3_train" \
        modal run --detach infra/app_sft_train.py \
            --epochs "${EPOCHS}" \
            --learning-rate "${LR}" \
            --min-reward "${MIN_REWARD}" \
            --max-seq-len "${MAX_SEQ_LEN}" \
            --grad-accum "${GRAD_ACCUM}" \
            --run-name "${RUN_NAME}" \
            --data-path "${DATA_PATH}" \
            --lora-rank "${LORA_RANK}" \
            --lora-target-modules "${LORA_TARGETS}" \
        || exit $?
fi

# ---------- Step 4: eval the new SFT v3 ckpt ----------------------------------
if [[ "${MODE}" == "--full" || "${MODE}" == "--skip-install" || "${MODE}" == "--skip-gen" || "${MODE}" == "--train-only" || "${MODE}" == "--eval-only" || "${MODE}" == "--dry-run" ]]; then
    echo ""
    echo "═══ Step 4/4: eval SFT v3 on ${EVAL_EPS} held-out sessions ═══"
    echo "  total_episodes=0 in SFTOnly_webshop_mlpr32.json skips RL and goes"
    echo "  straight to the held-out eval pass."
    echo "  Wall-clock ~30 min."
    run_step "step4_eval" \
        modal run infra/app_train_loop.py::train_loop_webshop \
            --config "/workspace/configs/SFTOnly_webshop_mlpr32.json" \
            --n-episodes 0 \
            --eval-episodes "${EVAL_EPS}" \
            --eval-task-id-base "${EVAL_TASK_BASE}" \
            --sft-adapter "${ADAPTER_PATH}" \
            --gpu-mem-util 0.30 \
            --run-name "SFTOnly_webshop_mlpr32_eval_${RUN_TS}" \
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
    echo "  data    : ${DATA_PATH}"
    echo "  adapter : ${ADAPTER_PATH}"
    echo "  eval    : /vol/manifests/SFTOnly_webshop_mlpr32_eval_${RUN_TS}_*/train_log.json"
    echo ""
    echo "Compare against current sft_v3_20260504_154752 baseline (rank-16, ~0.45 @ [6500,6600))."
    echo "Decision tree:"
    echo "  ≥ 0.50  → strong win; bump SFT_ADAPTER default in the 3 RL launchers"
    echo "  ∈ [0.42, 0.50) → marginal; bump default if pre-RL R0 lift is positive"
    echo "  < 0.42  → halt; revisit oracle quality / epoch count before Phase 5-6"
    echo ""
    echo "To download the eval log locally for inspection:"
    echo "  modal volume get cs224r-hgpo-vol \\"
    echo "    /manifests/SFTOnly_webshop_mlpr32_eval_${RUN_TS}_*/train_log.json eval.json"
    echo ""
    echo "Phase 6 (after eval passes): flip SFT_ADAPTER default in"
    echo "  scripts/run_webshop_SOTA_attention_v1.sh:24"
    echo "  scripts/run_webshop_SOTA_flatGRPO_v1.sh:31"
    echo "  scripts/run_webshop_SOTA_LLMJudge_v1.sh:28"
    echo "  → to: ${ADAPTER_PATH}"
fi
echo "═══════════════════════════════════════════════════════════════════════"
