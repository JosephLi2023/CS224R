#!/usr/bin/env bash
# Canonical-vocabulary launcher for the 7-method WebShop bake-off.
#
# Replaces scripts/run_webshop_protocol.sh for FUTURE runs. The legacy
# script is kept on disk because prior protocols still reference its
# method names. See docs/method_naming.md for the old<->new mapping.
#
# Methods (one per row of experiments/manifests/methods_comparison.json):
#   SFTOnly         eval-only (n_episodes=0; just runs the held-out eval pass)
#   flatGRPO        H-GRPO with alpha=1.0 (drops per-turn signal)
#   TurnRDV1        original learned attention decomposer (lean variant)
#   TurnRDV2        bidirectional + sigma alpha.v identifiable + carry-policy
#   Progressive     parameter-free progress decomposer (env raw_env_reward delta)
#   LLMJudge        OpenAI gpt-4o-mini judge (currently deferred; see notes)
#   Counterfactual  replay-based CF rollouts (N=2 alts, 3-turn completions)
#
# Each method dispatches to the appropriate Modal entrypoint:
#   - TurnRDV1 / TurnRDV2  -> scripts/run_turnrd_modal.py (producer<->trainer
#                              orchestration; TurnRDV2 adds --carry-policy-across-rounds)
#   - everything else      -> infra/app_train_loop.py::train_loop_webshop
#                              (per-round modal run, same per-round shape that
#                              matches scripts/run_flat_grpo_modal.sh and
#                              scripts/run_counterfactual_modal.sh)
#
# Usage:
#   bash scripts/run_methods_protocol.sh --seed 11
#   bash scripts/run_methods_protocol.sh --seed 11 --methods flatGRPO,TurnRDV2
#   bash scripts/run_methods_protocol.sh --dry-run
#
# Each method blocks until its rounds finish. Run under nohup if you want
# the launcher to outlive your shell:
#   nohup bash scripts/run_methods_protocol.sh --seed 11 \
#     > /tmp/methods_seed11.log 2>&1 &
set -euo pipefail

cd "$(dirname "$0")/.."

# ---- defaults (override via CLI flags) -----------------------------------
SEED=11
ROUNDS=5
EPS_PER_ROUND=40
K=4
MAX_TURNS=6
EVAL_EPS=50
EVAL_BASE=6500
SFT_ADAPTER="/vol/checkpoints/sft_v3_20260504_154752"
DRY_RUN="0"
METHODS_CSV="SFTOnly,flatGRPO,TurnRDV1,TurnRDV2,Progressive,LLMJudge,Counterfactual"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED="$2"; shift 2 ;;
    --rounds) ROUNDS="$2"; shift 2 ;;
    --eps-per-round) EPS_PER_ROUND="$2"; shift 2 ;;
    --sft-adapter) SFT_ADAPTER="$2"; shift 2 ;;
    --methods) METHODS_CSV="$2"; shift 2 ;;
    --dry-run) DRY_RUN="1"; shift 1 ;;
    -h|--help)
      sed -n '1,40p' "$0"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

IFS=',' read -ra METHODS <<< "$METHODS_CSV"

BASE_OFFSET=$(( SEED * ROUNDS * EPS_PER_ROUND ))   # disjoint per-seed slice

echo "=== methods_protocol seed=${SEED} rounds=${ROUNDS} eps_per_round=${EPS_PER_ROUND} ==="
echo "=== methods: ${METHODS[*]} ==="
echo "=== sft_adapter: ${SFT_ADAPTER} ==="
echo

# Foreground modal-run wrapper: respects --dry-run.
_run() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY RUN: $*"
  else
    "$@"
  fi
}

for METHOD in "${METHODS[@]}"; do
  CONFIG="configs/${METHOD}.json"
  if [[ ! -f "${CONFIG}" ]]; then
    echo "missing config: ${CONFIG}" >&2
    exit 1
  fi
  RUN_PREFIX="${METHOD}_seed${SEED}"
  echo
  echo "######################################################################"
  echo "# ${METHOD}"
  echo "######################################################################"

  case "${METHOD}" in
    # --- TurnRD methods: producer<->trainer orchestrator ---
    TurnRDV1)
      _run scripts/run_turnrd_modal.py \
        --config "${CONFIG}" \
        --seed "${SEED}" \
        --rounds "${ROUNDS}" \
        --episodes-per-round "${EPS_PER_ROUND}" \
        --turnrd-epochs 3 \
        --replay-path "/vol/cache/TurnRDV1/replay.jsonl" \
        --ckpt-path  "/vol/cache/TurnRDV1/ckpt.pt" \
        --run-name-prefix "TurnRDV1" \
        --sft-adapter "${SFT_ADAPTER}" \
        --eval-episodes "${EVAL_EPS}" \
        --eval-task-id-base "${EVAL_BASE}"
      ;;
    TurnRDV2)
      _run scripts/run_turnrd_modal.py \
        --config "${CONFIG}" \
        --seed "${SEED}" \
        --rounds "${ROUNDS}" \
        --episodes-per-round "${EPS_PER_ROUND}" \
        --turnrd-epochs 3 \
        --replay-path "/vol/cache/TurnRDV2/replay.jsonl" \
        --ckpt-path  "/vol/cache/TurnRDV2/ckpt.pt" \
        --run-name-prefix "TurnRDV2" \
        --sft-adapter "${SFT_ADAPTER}" \
        --eval-episodes "${EVAL_EPS}" \
        --eval-task-id-base "${EVAL_BASE}" \
        --carry-policy-across-rounds
      ;;
    SFTOnly)
      # Eval-only: total_episodes=0 makes _train_loop_impl skip the
      # `for ep in range(n_episodes)` body and go straight to the
      # held-out eval block. Single Modal call, no per-round loop.
      _run modal run infra/app_train_loop.py::train_loop_webshop \
        --config "/workspace/${CONFIG}" \
        --n-episodes 0 \
        --k "${K}" \
        --max-turns "${MAX_TURNS}" \
        --task-id-offset "${BASE_OFFSET}" \
        --run-name "${RUN_PREFIX}" \
        --round-idx 0 \
        --sft-adapter "${SFT_ADAPTER}" \
        --eval-episodes "${EVAL_EPS}" \
        --eval-task-id-base "${EVAL_BASE}" \
        --gpu-mem-util 0.30
      ;;
    # --- Everything else: per-round modal-run train_loop_webshop ---
    flatGRPO|Progressive|LLMJudge|Counterfactual)
      for r in $(seq 0 $((ROUNDS - 1))); do
        OFFSET=$(( BASE_OFFSET + r * EPS_PER_ROUND ))
        printf '\n=== %s round %02d (task_offset=%d) ===\n' "${METHOD}" "$r" "${OFFSET}"
        _run modal run infra/app_train_loop.py::train_loop_webshop \
          --config "/workspace/${CONFIG}" \
          --n-episodes "${EPS_PER_ROUND}" \
          --k "${K}" \
          --max-turns "${MAX_TURNS}" \
          --task-id-offset "${OFFSET}" \
          --run-name "${RUN_PREFIX}_round$(printf '%02d' "$r")" \
          --round-idx "$r" \
          --sft-adapter "${SFT_ADAPTER}" \
          --eval-episodes "${EVAL_EPS}" \
          --eval-task-id-base "${EVAL_BASE}" \
          --gpu-mem-util 0.30
      done
      ;;
    *)
      echo "no dispatch rule for method: ${METHOD}" >&2
      exit 1
      ;;
  esac
done

echo
echo "=== methods_protocol seed=${SEED} sweep complete ==="
