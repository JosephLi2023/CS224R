#!/usr/bin/env bash
# Launches the proposal §3.3 WebShop protocol: 6 methods × 1 seed = 6 runs.
# Run twice (seed=11 then seed=23) for the full 12-run protocol.
#
# Usage:
#   bash scripts/run_webshop_protocol.sh --seed 11
#   bash scripts/run_webshop_protocol.sh --seed 23
#
# Each method config is merged with configs/env_webshop_llm.yaml + configs/eval.json
# and dispatched as a Modal job (or local fallback). Run dirs land under
# experiments/manifests/<method>_webshop_seed<seed>_<ts>/.
set -euo pipefail

SEED="11"
DRY_RUN="0"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED="$2"; shift 2 ;;
    --dry-run) DRY_RUN="1"; shift 1 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

METHODS=(
  "method_react_eval"
  "method_flat_grpo"
  "method_hgpo_progress"
  "method_hgpo_judge"
  "method_hgpo_turnrd"
  "method_archer"
)

cd "$(dirname "$0")/.."
mkdir -p experiments/manifests

for METHOD in "${METHODS[@]}"; do
  CONFIG="configs/${METHOD}.json"
  if [[ ! -f "${CONFIG}" ]]; then
    echo "missing config: ${CONFIG}" >&2
    exit 1
  fi
  echo "=== launching ${METHOD} seed=${SEED} ==="
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY RUN: would invoke train with --train-config ${CONFIG} --env-config configs/env_webshop.json --seed ${SEED}"
    continue
  fi
  PYTHONPATH=. python3 -m src.trainers.train \
    --env-config configs/env_webshop.json \
    --train-config "${CONFIG}" \
    --eval-config configs/eval.json
done

echo "=== seed=${SEED} sweep complete ==="
