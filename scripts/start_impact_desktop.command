#!/bin/zsh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
ENV_NAME="${IMPACT_CONDA_ENV:-impact-synergy-clean}"

# A Finder double-click runs this script non-interactively, so `conda init`
# from ~/.zshrc may not have run. Look for conda explicitly.
CONDA_BIN="${IMPACT_CONDA_EXE:-${CONDA_EXE:-}}"
if [[ -z "${CONDA_BIN}" ]]; then
  CONDA_BIN="$(command -v conda 2>/dev/null || true)"
fi
if [[ -z "${CONDA_BIN}" ]]; then
  for candidate in \
    "${HOME}/miniforge3/bin/conda" \
    "${HOME}/mambaforge/bin/conda" \
    "${HOME}/miniconda3/bin/conda" \
    "${HOME}/anaconda3/bin/conda" \
    "/opt/homebrew/Caskroom/miniforge/base/bin/conda" \
    "/opt/miniconda3/bin/conda" \
    "/opt/anaconda3/bin/conda"; do
    if [[ -x "${candidate}" ]]; then
      CONDA_BIN="${candidate}"
      break
    fi
  done
fi
if [[ -z "${CONDA_BIN}" ]]; then
  echo "conda was not found. Set IMPACT_CONDA_EXE=/path/to/conda (or run 'conda init zsh') and try again."
  read -r "?Press Return to close."
  exit 1
fi

exec "${CONDA_BIN}" run --no-capture-output -n "${ENV_NAME}" python scripts/impact_desktop_app.py
