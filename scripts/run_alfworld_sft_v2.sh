#!/usr/bin/env bash
# run_alfworld_sft_v2.sh - Train an improved AlfWorld SFT baseline (target >=0.55).
# Pipeline: install AlfWorld data -> generate expert SFT trajs -> train SFT v2 -> eval.
#
# Usage:
#   bash scripts/run_alfworld_sft_v2.sh [--dry-run|--full|--skip-install|--skip-gen|--train-only|--eval-only]
#
# Modes:
#   --dry-run        : print every modal command without running anything.
#   --full (default) : run all 4 steps end-to-end.
#   --skip-install   : AlfWorld data already on the volume; start from step 2.
#   --skip-gen       : SFT data file already exists; start from step 3.
#   --train-only     : only run step 3 (assumes data file is at $DATA_PATH).
#   --eval-only      : only run step 4 (set ADAPTER_PATH env var).
#
# Override knobs via env vars: N_GAMES EPOCHS LR MAX_SEQ_LEN MAX_HISTORY_TURNS
#   GRAD_ACCUM MIN_REWARD EVAL_EPS EVAL_TASK_BASE RUN_NAME DATA_PATH LOG_DIR.

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

# Tunable knobs (env-var overrides)
N_GAMES=${N_GAMES:-1000}
EPOCHS=${EPOCHS:-6}
LR=${LR:-5e-5}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-2048}
MAX_HISTORY_TURNS=${MAX_HISTORY_TURNS:-6}
MAX_STEPS_PER_EP=${MAX_STEPS_PER_EP:-50}
GRAD_ACCUM=${GRAD_ACCUM:-8}
MIN_REWARD=${MIN_REWARD:-0.5}
EVAL_EPS=${EVAL_EPS:-200}
EVAL_TASK_BASE=${EVAL_TASK_BASE:-6500}
# LoRA arch knobs. Defaults: rank-16 attention-only; override LORA_RANK/LORA_TARGETS
# for the rank-32 MLP+attn experiment.
LORA_RANK=${LORA_RANK:-16}
LORA_TARGETS=${LORA_TARGETS:-q_proj,k_proj,v_proj,o_proj}

RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
RUN_NAME=${RUN_NAME:-sft_alfworld_v2_${RUN_TS}}
DATA_PATH=${DATA_PATH:-/vol/data/alfworld/sft_trajs_v2.jsonl}
ADAPTER_PATH=${ADAPTER_PATH:-/vol/checkpoints/${RUN_NAME}}
LOG_DIR=${LOG_DIR:-/tmp}
LOG_FILE="${LOG_DIR}/sft_alfworld_v2_${RUN_TS}.log"

# Pre-flight checks
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
    infra/app_alfworld_install.py \
    infra/app_alfworld_sft_gen.py \
    infra/app_sft_train_alfworld.py \
    infra/app_train_loop.py \
    configs/SFTOnly_alfworld_mlpr32.json
do
    if [[ ! -f "${app}" ]]; then
        echo "ERROR: required file missing: ${app}" >&2
        echo "       Are you in the repo root and on the right branch?" >&2
        exit 2
    fi
done

# Header
mkdir -p "${LOG_DIR}"

echo "========================================"
echo "AlfWorld SFT v2 boost  (v1 baseline @ 0.40, target >=0.55)"
echo "  mode               : ${MODE}"
echo "  run_name           : ${RUN_NAME}"
echo "  data_path  (vol)   : ${DATA_PATH}"
echo "  adapter_path (vol) : ${ADAPTER_PATH}"
echo "  log file (local)   : ${LOG_FILE}"
echo "  --- data ---"
echo "  n_games            : ${N_GAMES}      (v1: 200)"
echo "  max_history_turns  : ${MAX_HISTORY_TURNS}        (v1: 3)"
echo "  max_steps_per_ep   : ${MAX_STEPS_PER_EP}       (v1: 50)"
echo "  --- train ---"
echo "  epochs             : ${EPOCHS}        (v1: 3)"
echo "  learning_rate      : ${LR}     (v1: 1e-4)"
echo "  max_seq_len        : ${MAX_SEQ_LEN}     (v1: 1024)"
echo "  grad_accum         : ${GRAD_ACCUM}        (v1: 4)"
echo "  min_reward         : ${MIN_REWARD}      (v1: 0.5)"
echo "  --- LoRA arch ---"
echo "  lora_rank          : ${LORA_RANK}       (default: 16)"
echo "  lora_target_mods   : ${LORA_TARGETS}"
echo "                       (default: q_proj,k_proj,v_proj,o_proj)"
echo "  --- eval ---"
echo "  eval_episodes      : ${EVAL_EPS}      (v1: 50)"
echo "  eval_task_id_base  : ${EVAL_TASK_BASE}"
echo "========================================"

run_step () {
    local label="$1"; shift
    if [[ "${MODE}" == "--dry-run" ]]; then
        echo ""
        echo "[DRY-RUN] ${label}"
        echo "  $*"
        return 0
    fi
    echo ""
    echo "${label}  (start: $(date +%H:%M:%S))"
    echo "  $*"
    "$@" 2>&1 | tee -a "${LOG_FILE}"
    local rc=${PIPESTATUS[0]}
    echo "${label}  (end: $(date +%H:%M:%S), exit=${rc})"
    if [[ ${rc} -ne 0 ]]; then
        echo "ERROR: ${label} exited ${rc}. Aborting." >&2
        echo "       Full log: ${LOG_FILE}" >&2
        return ${rc}
    fi
    return 0
}

# Step 1: install AlfWorld game data
if [[ "${MODE}" == "--full" || "${MODE}" == "--dry-run" ]]; then
    echo ""
    echo "=== Step 1/4: install AlfWorld game data ==="
    echo "  (This downloads the AlfWorld json_2.1.1 train+valid splits onto your"
    echo "   Modal volume. Takes ~20 min the first time; idempotent on re-runs.)"
    run_step "step1_install" \
        modal run infra/app_alfworld_install.py --action download \
        || exit $?
fi

# Step 2: generate 1000 expert SFT trajectories
if [[ "${MODE}" == "--full" || "${MODE}" == "--skip-install" || "${MODE}" == "--dry-run" ]]; then
    echo ""
    echo "=== Step 2/4: generate ${N_GAMES} expert SFT trajectories ==="
    echo "  Writes JSONL to ${DATA_PATH} on the Modal volume."
    echo "  Wall-clock ~3-5 hr (CPU-only; handcoded PDDL expert)."
    run_step "step2_gen" \
        modal run infra/app_alfworld_sft_gen.py::main \
            --n-games "${N_GAMES}" \
            --output-path "${DATA_PATH}" \
            --max-history-turns "${MAX_HISTORY_TURNS}" \
            --max-steps-per-episode "${MAX_STEPS_PER_EP}" \
        || exit $?
fi

# Step 3: train SFT v2
if [[ "${MODE}" == "--full" || "${MODE}" == "--skip-install" || "${MODE}" == "--skip-gen" || "${MODE}" == "--train-only" || "${MODE}" == "--dry-run" ]]; then
    echo ""
    echo "=== Step 3/4: train SFT v2 (${EPOCHS} epochs, lr=${LR}, seq_len=${MAX_SEQ_LEN}) ==="
    echo "  Writes adapter to /vol/checkpoints/${RUN_NAME} on the Modal volume."
    echo "  Wall-clock ~3-5 hr on A100-80GB."
    run_step "step3_train" \
        modal run --detach infra/app_sft_train_alfworld.py \
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

# Step 4: eval the new SFT v2 ckpt
if [[ "${MODE}" == "--full" || "${MODE}" == "--skip-install" || "${MODE}" == "--skip-gen" || "${MODE}" == "--train-only" || "${MODE}" == "--eval-only" || "${MODE}" == "--dry-run" ]]; then
    echo ""
    echo "=== Step 4/4: eval SFT v2 on ${EVAL_EPS} held-out games ==="
    echo "  total_episodes=0 in SFTOnly_alfworld_mlpr32.json skips RL and goes"
    echo "  straight to the held-out eval pass (valid_seen + valid_unseen)."
    echo "  Wall-clock ~30 min."
    run_step "step4_eval" \
        modal run infra/app_train_loop.py::train_loop_alfworld \
            --config "/workspace/configs/SFTOnly_alfworld_mlpr32.json" \
            --n-episodes 0 \
            --eval-episodes "${EVAL_EPS}" \
            --eval-task-id-base "${EVAL_TASK_BASE}" \
            --sft-adapter "${ADAPTER_PATH}" \
            --gpu-mem-util 0.30 \
            --run-name "SFTOnly_alfworld_v2_eval_${RUN_TS}" \
        || exit $?
fi

# Footer
echo ""
echo "========================================"
if [[ "${MODE}" == "--dry-run" ]]; then
    echo "Dry-run complete. Re-invoke without --dry-run to actually launch."
else
    echo "Done.  log: ${LOG_FILE}"
    echo ""
    echo "Outputs on Modal volume:"
    echo "  data    : ${DATA_PATH}"
    echo "  adapter : ${ADAPTER_PATH}"
    echo "  eval    : /vol/manifests/SFTOnly_alfworld_v2_eval_${RUN_TS}_*/train_log.json"
    echo ""
    echo "Compare against v1 baseline (sft_alfworld_v1_20260507_165617, eval=0.40 @ N=50)."
    echo "Decision tree:"
    echo "  >= 0.60  -> strong win; use as new SFT for downstream RL methods"
    echo "  [0.50, 0.60) -> modest win; worth using as new baseline"
    echo "  [0.45, 0.50) -> marginal; consider tweaking knobs (more epochs, bigger LoRA)"
    echo "  < 0.45  -> no real lift; investigate (overfitting? data quality?)"
    echo ""
    echo "To download the eval log locally for inspection:"
    echo "  modal volume get cs224r-hgpo-vol \\"
    echo "    /manifests/SFTOnly_alfworld_v2_eval_${RUN_TS}_*/train_log.json eval.json"
fi
echo "========================================"
