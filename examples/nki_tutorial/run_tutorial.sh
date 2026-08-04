#!/usr/bin/env bash
# =============================================================================
#  NKI trace tutorial - one-shot runner
# -----------------------------------------------------------------------------
#  Purpose: go from nothing to a working trace with a single command. It will:
#    1. Locate the repository root.
#    2. Create the .venv virtual environment if it does not exist.
#    3. Install triton-viz + nki dependencies into .venv (first run only; later
#       runs skip this).
#    4. If Python.h is missing, try to install it (Triton needs it to compile a
#       small helper module).
#    5. Activate the venv and run the NKI tutorial script to generate traces.
#
#  Usage:
#    bash examples/nki_tutorial/run_tutorial.sh
#
#  Note: this script is idempotent; once dependencies are installed it jumps
#  quickly to the final step.
# =============================================================================
set -euo pipefail

# ---- 1. Locate the repo root (this script lives in <repo>/examples/nki_tutorial/) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
echo "[1/5] repo root: ${REPO_ROOT}"

VENV_DIR="${REPO_ROOT}/.venv"
PY_INDEX_URL="https://pip.repos.neuron.amazonaws.com"

# ---- 2. Create the virtual environment (if it does not exist) ----
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[2/5] no .venv found, creating a virtual environment..."
  python3 -m venv "${VENV_DIR}"
else
  echo "[2/5] virtual environment already exists: ${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
echo "      activated: $(python --version) @ $(which python)"

# ---- 3. Install dependencies (only when nki is not yet importable, to avoid
#         re-downloading on every run) ----
if python -c "import nki, triton_viz, neuronxcc" >/dev/null 2>&1; then
  echo "[3/5] dependencies ready (nki / triton-viz / neuronxcc all importable), skipping install."
else
  echo "[3/5] installing dependencies (first run downloads a lot, please be patient)..."
  python -m pip install -U pip setuptools wheel
  # -e '.[test,nki]' installs this repo in editable mode with the nki extra.
  python -m pip install -e '.[test,nki]' --extra-index-url "${PY_INDEX_URL}"
fi

# ---- 4. Ensure Python.h is present (Triton needs it to compile a CUDA/CPU helper) ----
#     Without it, some environments fail on first kernel run with
#     "Python.h: No such file".
PY_TAG="$(python -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if ! python -c "import sysconfig,os,sys; p=os.path.join(sysconfig.get_paths()['include'],'Python.h'); sys.exit(0 if os.path.exists(p) else 1)"; then
  echo "[4/5] Python.h missing, trying to install python${PY_TAG}-dev ..."
  if command -v sudo >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y && sudo apt-get install -y "python${PY_TAG}-dev" || \
      echo "      (auto-install failed; if a Python.h error appears later, run: sudo apt-get install python${PY_TAG}-dev)"
  else
    echo "      (no sudo/apt-get; if a Python.h error appears, install python${PY_TAG}-dev manually)"
  fi
else
  echo "[4/5] Python.h already present, nothing to do."
fi

# ---- 5. Run the tutorial script ----
echo "[5/5] running the NKI tutorial script to generate traces ..."
echo "-----------------------------------------------------------------------"
python "${SCRIPT_DIR}/nki_trace_tutorial.py"
echo "-----------------------------------------------------------------------"
echo "Done! Generated traces / event stream / timeline are in: ${REPO_ROOT}/nki_tutorial_out/"
ls -la "${REPO_ROOT}/nki_tutorial_out/" 2>/dev/null || true
