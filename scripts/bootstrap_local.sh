#!/usr/bin/env bash
# Bootstrap the local dev environment for CS224R H-GRPO.
#
# Creates a project-local venv on top of /Users/shoupeili/miniconda3 Python 3.11,
# installs the lightweight requirements, runs the test suite to confirm.
#
# Heavy ML deps (torch, transformers, vllm, ...) intentionally NOT installed
# locally — they live inside the Modal image (infra/image.py).
#
# Usage:
#   bash scripts/bootstrap_local.sh
#
# Idempotent: safe to re-run.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON="${PYTHON:-/Users/shoupeili/miniconda3/bin/python}"

echo "==> Project root: ${PROJECT_ROOT}"
echo "==> Using Python: ${PYTHON}"
"${PYTHON}" --version

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "==> Creating venv at ${VENV_DIR}"
  "${PYTHON}" -m venv "${VENV_DIR}"
else
  echo "==> Reusing existing venv at ${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r "${PROJECT_ROOT}/requirements/local.txt"

echo
echo "==> Installed local toolchain:"
python - <<'PY'
import importlib.metadata, sys
mods = ['pytest', 'modal', 'openai', 'httpx', 'pydantic', 'ruff']
for m in mods:
    try:
        v = importlib.metadata.version(m)
    except importlib.metadata.PackageNotFoundError as e:
        v = f'MISSING ({e})'
    print(f'  {m:<10} {v}')
print(f'  python    {sys.version.split()[0]}')
PY

echo
echo "==> Running test suite (29 tests expected to pass)"
PYTHONPATH="${PROJECT_ROOT}" python -m pytest "${PROJECT_ROOT}/tests/unit" "${PROJECT_ROOT}/tests/integration" -q

echo
echo "==> Bootstrap complete. Activate with: source ${VENV_DIR}/bin/activate"
echo "==> Next: follow docs/MODAL_SETUP.md to provision your Modal account and run the first smoke test."
