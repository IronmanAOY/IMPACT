#!/bin/bash
# install_hunter_env.sh -- one-time IMPaCT environment setup on an HLRS Hunter LOGIN node.
#
# Default: print the steps (dry run). With --run the steps are executed in this shell.
# Nothing here submits jobs. Every step is [UNVERIFIED] on Hunter unless tagged [KB]
# (kb.hlrs.de) or [VENDOR] (HPE CPE / CuPy / AMD docs).
#
# Prerequisites:
#   * Internet for pip exists only through the HLRS reverse SOCKS tunnel [KB SSH_Tunnel_with_Proxy]:
#       local>  MY_PROXY_PORT=<10000-60000>; ssh -R localhost:$MY_PROXY_PORT hunter.hww.hlrs.de
#       hunter> export https_proxy=socks5://localhost:$MY_PROXY_PORT http_proxy=socks5://localhost:$MY_PROXY_PORT
#   * The repository is cloned into a workspace (HOME must not be used for job I/O) [KB].
#   * Login nodes have a 2 h CPU-time limit: heavy builds are prefixed with `nice -n 19` [KB];
#     a CuPy source build may need an interactive mi300a job instead.
#
# Options:
#   --run                      execute the steps (default: print them)
#   --cupy none|source-13.6|wheel-rocm7
#                              none (default); source-13.6 = CuPy 13.6.0 from source for ROCm 6.4.1
#                              (default stack HLRS/APU/2026.1); wheel-rocm7 = cupy-rocm-7-0>=14.1
#                              (testing stack HLRS/APU/testing-2026.2, ROCm 7.0.2) [VENDOR]
#   --ws-name NAME             workspace name (default: impact)
#   --venv DIR                 venv location (default: <workspace>/venvs/impact-hunter)
set -eo pipefail

MODE=print
CUPY=none
WS_NAME="${IMPACT_WS_NAME:-impact}"
VENV=""
STACK="${IMPACT_HUNTER_STACK:-HLRS/APU/2026.1}"
ROCM_MODULE="${IMPACT_HUNTER_ROCM_MODULE:-rocm}"
PYSOCKS_WHEEL="/sw/general/x86_64/development/python/share/PySocks-1.7.1-py3-none-any.whl"

usage() { sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --run) MODE=run ;;
    --cupy) CUPY="${2:?--cupy needs a value}"; shift ;;
    --ws-name) WS_NAME="${2:?}"; shift ;;
    --venv) VENV="${2:?}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
case "${CUPY}" in none|source-13.6|wheel-rocm7) ;; *) echo "Invalid --cupy ${CUPY}" >&2; exit 2 ;; esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

step() {
  echo "+ $*"
  if [ "${MODE}" = "run" ]; then
    eval "$@"
  fi
}

echo "# IMPaCT Hunter environment setup (mode: ${MODE}; repo: ${repo_root})"
echo "# 1. Software stack [KB]"
step "module load ${STACK} cray-python ${ROCM_MODULE}"
echo "# 2. Workspace [KB ws_allocate/ws_find]"
step "WS=\$(ws_find ${WS_NAME} || true); if [ -z \"\$WS\" ]; then WS=\$(ws_allocate ${WS_NAME} 60); fi; export WS"
VENV_EXPR="${VENV:-\$WS/venvs/impact-hunter}"
echo "# 3. venv on top of cray-python, reusing the Cray-linked numpy/scipy [KB]"
step "python3 -m venv --system-site-packages \"${VENV_EXPR}\""
step "source \"${VENV_EXPR}/bin/activate\""
echo "# 4. pip through the SOCKS tunnel needs PySocks from the local wheel [KB]"
step "python3 -m pip install ${PYSOCKS_WHEEL}"
echo "# 5. Pinned runtime packages; constraints keep numpy 1.24.4 / scipy 1.10.1 of cray-python"
step "nice -n 19 python3 -m pip install -c \"${repo_root}/constraints-hunter.txt\" -r \"${repo_root}/requirements-hunter.txt\""
echo "# 6. CuPy for ROCm (${CUPY}) [VENDOR]"
case "${CUPY}" in
  source-13.6)
    step "export CUPY_INSTALL_USE_HIP=1 ROCM_HOME=\${ROCM_PATH:-/opt/rocm} HCC_AMDGPU_TARGET=gfx942"
    step "nice -n 19 python3 -m pip install -c \"${repo_root}/constraints-hunter.txt\" cupy==13.6.0"
    ;;
  wheel-rocm7)
    echo "#    (testing stack only: use numpy==2.3.5 / scipy==1.16.3 constraints and numba>=0.62)"
    step "python3 -m pip install 'cupy-rocm-7-0>=14.1'"
    ;;
  *)
    echo "#    skipped (--cupy none): --hardware-target hunter-apu needs CuPy; cpu/auto work without it"
    ;;
esac
echo "# 7. Import and CPU self-test on the login node (the APU test must run on a compute node)"
step "PYTHONPATH=\"${repo_root}/src\" python3 -m impact_pipeline.hardware_selftest --target cpu --size 64"
cat <<EOF
# 8. Next steps (run by a person; job submission stays under human control):
#    cp "${repo_root}/scripts/hunter/hunter_pbs_setup.sh" "\$WS/hunter_pbs_setup.sh"
#    export IMPACT_HUNTER_SETUP_FILE="\$WS/hunter_pbs_setup.sh"; source "\$IMPACT_HUNTER_SETUP_FILE"
#    bash "${repo_root}/scripts/hunter/hunter_smoke_test.sh" --bids-root <BIDS> --out-dir <OUT> --subjects <1-2 IDs> --submit
#    On the node the smoke job runs: python3 -m impact_pipeline.hardware_selftest --target hunter-apu
EOF
