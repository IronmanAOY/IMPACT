#!/bin/bash
# hunter_pbs_setup.sh -- site setup sourced by every generated IMPaCT PBS job on HLRS Hunter.
#
# Usage (once, on a Hunter login node, before building a campaign):
#   cp scripts/hunter/hunter_pbs_setup.sh "$WS/hunter_pbs_setup.sh"   # edit the copy if needed
#   export IMPACT_HUNTER_SETUP_FILE="$WS/hunter_pbs_setup.sh"
#   source "$IMPACT_HUNTER_SETUP_FILE"            # also sets IMPACT_HUNTER_PYTHON for the build
#   python3 run_pipeline.py --execution-mode hunter --hunter-stage build-campaign ...
#
# The generated job scripts start with #!/bin/bash (module needs bash), run `set -eo pipefail`,
# source this file, and only then enable `set -u` (Lmod under nounset is unverified).
#
# Tags: [KB] stated on kb.hlrs.de Hunter pages; [VENDOR] HPE CPE / CuPy / AMD documentation;
#       [UNVERIFIED] not confirmed for Hunter -- check with `module avail`, `module show` or HLRS staff.
# Every value can be overridden by exporting the variable before sourcing.

# ---- 1. Software stack --------------------------------------------------------------------------
# Default since 2026-03-30: HLRS/APU/2026.1 = CPE 25.09, ROCm 6.4.1, cray-python 3.11.7 (numpy 1.24.4).  [KB]
# Alternative: HLRS/APU/testing-2026.2 = CPE 26.03, ROCm 7.0.2, cray-python 3.12.12 (numpy 2.3.5).       [KB]
IMPACT_HUNTER_STACK="${IMPACT_HUNTER_STACK:-HLRS/APU/2026.1}"
IMPACT_HUNTER_ROCM_MODULE="${IMPACT_HUNTER_ROCM_MODULE:-rocm}"      # [UNVERIFIED] exact name/version
export LMOD_PAGER=None
module load "${IMPACT_HUNTER_STACK}"                                # [KB]
module load cray-python                                             # [KB name][VENDOR versions]
if [ -n "${IMPACT_HUNTER_ROCM_MODULE}" ]; then
  module load "${IMPACT_HUNTER_ROCM_MODULE}"                        # [UNVERIFIED] may already be loaded
fi
module list 2>&1                                                    # record the environment in the job log

# ---- 2. Workspace (all job I/O on Lustre; HOME is not for compute jobs) ---------------------------  [KB Storage_(Hunter)]
IMPACT_WS_NAME="${IMPACT_WS_NAME:-impact}"                          # created once: ws_allocate impact 60  [KB]
IMPACT_WS="${IMPACT_WS:-$(ws_find "${IMPACT_WS_NAME}")}"            # [KB ws_find]
if [ -z "${IMPACT_WS}" ] || [ ! -d "${IMPACT_WS}" ]; then
  echo "ERROR: workspace '${IMPACT_WS_NAME}' not found (see ws_list)" >&2
  return 1 2>/dev/null || exit 1   # sourced: do not close an interactive login shell
fi
export IMPACT_WS

# ---- 3. Python: venv on top of cray-python (created once with --system-site-packages) -------------  [KB Python_Virtual_Environments_and_Packages]
IMPACT_VENV="${IMPACT_VENV:-${IMPACT_WS}/venvs/impact-hunter}"
if [ ! -f "${IMPACT_VENV}/bin/activate" ]; then
  echo "ERROR: venv '${IMPACT_VENV}' missing; run scripts/hunter/install_hunter_env.sh first" >&2
  return 1 2>/dev/null || exit 1   # sourced: do not close an interactive login shell
fi
# shellcheck disable=SC1091
source "${IMPACT_VENV}/bin/activate"
export IMPACT_HUNTER_PYTHON="${IMPACT_VENV}/bin/python3"           # HLRS: invoke as python3  [KB]
# run_pipeline.py checks for the workstation environment name by default; the venv above pins the
# interpreter here, so the check is skipped for login-node commands too (build-campaign, status).
export IMPACT_SKIP_ENV_CHECK="${IMPACT_SKIP_ENV_CHECK:-1}"
export PYTHONNOUSERSITE=1                                           # keep ~/.local out of sys.path  [UNVERIFIED on Hunter]
# Deployments without .git metadata (e.g. a copied tarball) should record their version
# for provenance, e.g.: export IMPACT_CODE_VERSION=1.1.0+<commit>

# ---- 4. Caches/data off HOME ------------------------------------------------------------------------
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-${IMPACT_WS}/cache/cupy}"            # CuPy default: ${HOME}/.cupy  [VENDOR]
export MPLCONFIGDIR="${MPLCONFIGDIR:-${IMPACT_WS}/cache/matplotlib}"          # [UNVERIFIED vs HLRS docs]
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${IMPACT_WS}/cache/numba}"         # [UNVERIFIED]
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${IMPACT_WS}/cache/xdg}"             # [UNVERIFIED]
export MNE_DATA="${MNE_DATA:-${IMPACT_WS}/data/mne}"                          # [UNVERIFIED]
export NILEARN_DATA="${NILEARN_DATA:-${IMPACT_WS}/data/nilearn}"              # [UNVERIFIED]
mkdir -p "${CUPY_CACHE_DIR}" "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}" "${XDG_CACHE_HOME}" "${MNE_DATA}" "${NILEARN_DATA}"
# IIM SQLite kernel caches are node-local by default (/localscratch/$PBS_JOBID on localscratch
# nodes, else $TMPDIR = RAM disk). Uncomment to force a location:
# export IMPACT_IIM_CACHE_DIR="${TMPDIR:-/tmp}"

# ---- 5. ROCm / CuPy -----------------------------------------------------------------------------------
export ROCM_HOME="${ROCM_HOME:-${ROCM_PATH:-/opt/rocm}}"            # CuPy uses ROCM_HOME [VENDOR]; ROCM_PATH [UNVERIFIED]
# CUPY_ACCELERATORS defaults to "" on HIP. [VENDOR]
# cupy.linalg.eigh on ROCm is checked against NumPy at first use and falls back to the CPU;
# force a policy with IMPACT_EIGH_BACKEND=cpu|device|auto.
# IIM Psi kernel: on --hardware-target hunter-apu the shards run the array-module (CuPy) kernel on
# the APU (HLRS: GPU use is mandatory). Check it first with
#   python3 -m impact_pipeline.hardware_selftest --target hunter-apu   (case iim_psi_xp_parity)
# IMPACT_IIM_PSI_KERNEL=numba forces the host kernel (debugging only); IMPACT_IIM_XP_MAX_ELEMENTS
# bounds the largest temporary array of the device kernel (default 8388608 float64 = 64 MB).
# export IMPACT_IIM_XP_MAX_ELEMENTS=33554432
# The generated jobs export IMPACT_REPO_ROOT (the checkout they run); set it here too if the package
# was pip-installed non-editably and login-node commands must find the checkout:
# export IMPACT_REPO_ROOT="${IMPACT_WS}/impact-synergy-pipeline"

# ---- 6. Threads (one rank per APU = 24 cores; the job scripts set 1 thread per worker process) ---  [KB Affinity_and_Pinning]
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-22}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"

# ---- 7. No internet on compute nodes: make accidental downloads fail fast ----------------------------  [KB SSH_Tunnel_with_Proxy]
if [ -n "${PBS_JOBID:-}" ]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
fi
