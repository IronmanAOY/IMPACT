# HLRS Hunter Runbook: IMPaCT Synergy Pipeline 1.1.0

This runbook is for an HLRS colleague who runs the IMPaCT IIM campaign on Hunter
(HPE Cray EX4000, AMD MI300A APUs, Altair PBS Pro) without the author being
present. Follow it from top to bottom. It also works as a reference for
researchers who run their own campaigns.

- Pipeline: IMPaCT Synergy Pipeline 1.1.0 (`run_pipeline.py`, package
  `impact_pipeline`).
- Written: 2026-09-28. HLRS facts are taken from the HLRS knowledge base
  (kb.hlrs.de) and vendor documentation as read on 2026-09-27.
- Contact for scientific questions: the author (see `CITATION.cff`). Contact
  for Hunter questions: `rt-platform-hunter@hlrs.de` or the HLRS
  trouble-ticket form. HLRS asks users not to contact staff individually.

> **Size the IIM configuration before the full campaign
> ([section 10a](#10a-size-the-iim-configuration-before-the-full-campaign)).**
> Exhaustive IIM grows roughly 30x per added subsystem node (one CPU core,
> T = 1000: 1.7 s at 4 nodes, 46 s at 5, about 27 min at 6). Without
> `--iim-max-nodes` the subsystem has 10 nodes, which is infeasible
> exhaustively (about 7 core-years per IIM run). The build prints the
> estimated number of Ψ evaluations and an order-of-magnitude runtime, and it
> refuses configurations above a ceiling (`--hunter-max-psi-evals`,
> `IMPACT_HUNTER_MAX_PSI_EVALS`, default 1e11) unless `--hunter-allow-large` is
> given. Measure the time per evaluation with the smoke test first, and let the
> author choose the IIM sizes from your measurements.

## How to read this document

Commands are labelled in one of two ways.

| Label | Meaning |
|---|---|
| **[checked]** | Run for this document on a workstation (macOS, conda env `impact-synergy-clean`, Python 3.10, CPU only). Any output shown comes from that run. |
| **[to be verified on Hunter]** | Taken from the HLRS knowledge base or from vendor documentation. It has **not** been run on Hunter. Module names, paths and limits may differ. Check them, and write down what you find (section 17). |

Placeholders use angle brackets, for example `<group>`. Shell variables such as
`$WS` are set in the steps where they first appear.

## Contents

1. [What runs on Hunter](#1-what-runs-on-hunter)
2. [Account, 2FA and login nodes](#2-account-2fa-and-login-nodes)
3. [Workspace (never HOME)](#3-workspace-never-home)
4. [Getting the code](#4-getting-the-code)
5. [Python environment](#5-python-environment)
6. [CuPy for ROCm](#6-cupy-for-rocm)
7. [Hardware self-test (run this first)](#7-hardware-self-test-run-this-first)
8. [Data staging](#8-data-staging)
9. [Building the campaign (login node)](#9-building-the-campaign-login-node)
10. [Smoke test on the test queue](#10-smoke-test-on-the-test-queue)
    - 10a. [Size the IIM configuration before the full campaign](#10a-size-the-iim-configuration-before-the-full-campaign)
11. [Full submission](#11-full-submission)
12. [Monitoring](#12-monitoring)
13. [Resuming and re-running](#13-resuming-and-re-running)
14. [Collecting results](#14-collecting-results)
15. [Resources and accounting](#15-resources-and-accounting)
16. [Troubleshooting](#16-troubleshooting)
17. [What to send back to the author](#17-what-to-send-back-to-the-author)
18. [Open questions for HLRS staff](#18-open-questions-for-hlrs-staff)
19. [Reference: files, variables and stages](#19-reference-files-variables-and-stages)

---

## 1. What runs on Hunter

Only the **integrated-information (IIM) campaign** is distributed across nodes.
IIM is the only expensive estimator: it evaluates the integration mass Ψ over
all mechanism/purview bipartitions of a small subsystem and then over every
system cut. The other four estimators (RAM, PDI, NAS, SRPI), the evidence
layer, the statistics and the Word report run afterwards, in a single job on
one node.

```text
 login node                          compute nodes (mi300a, 4 APUs each)
 ----------                          -----------------------------------
 build-campaign  ──qsub──►  01_phase1_shards.pbs  (PBS array; 4 shards/node, Ψ of the intact TPM)
 (CPU, minutes)  ──qsub──►  02_cut_shards.pbs     (PBS array; 4 shards/node, Ψ of every system cut)
                                 │ afterok (both whole arrays)
                                 ▼
                            03_reduce_finalize.pbs (1 node: reduce-all, then finalize-pipeline:
                                                    RAM/PDI/NAS/SRPI, legacy CI, MPC verdicts,
                                                    statistics, report)
```

- IIM is computed at lag 1 with 3 bins per node on a subsystem of at most
  `--iim-max-nodes` nodes (`docs/metrics.md`, section 5).
- Phase-1 and cut shards are independent and run concurrently.
- Surrogate-null calibration of IIM (`--hunter-iim-null-surrogates K`) adds
  K surrogate runs per real run, and the sampling SE of IIM
  (`--hunter-iim-bootstrap-se B`) adds B block-bootstrap replicate runs per
  real run. Both are sharded like real runs, so the campaign costs about
  (K+B+1) times as much. Without them the IIM evidence has no null
  (`NO_NULL_CALIBRATION:IIM`) or no sampling SE (`NO_SAMPLING_SE:IIM`), and
  every verdict with IIM in the necessity set is UNDETERMINED.
- Nothing in the campaign downloads anything. Compute nodes have no internet.
- Not done on Hunter: dataset downloads, fMRIPrep (Docker is not available on
  Hunter), and normally the time-series extraction ("preprocessing"). See
  section 8.

A complete run has these phases. The times are rough estimates for a person who
has done it before, not measurements.

| Phase | Where | Time |
|---|---|---|
| Account, key, 2FA (section 2) | HLRS portal, your machine | 1-2 days (key activated next day) |
| Workspace, code, venv, CuPy (sections 3-6) | login node, interactive job | 1-3 h |
| Hardware self-test (section 7) | interactive or `test` job | 10 min |
| Data transfer (section 8) | your machine to the workspace | depends on size |
| Campaign build and smoke test (sections 9-10) | login node, `test` queue | 30-60 min |
| IIM sizing: calibration smoke test, report to the author, their decision (section 10a) | `test` queue, e-mail | 1 h plus the author's reply |
| Production campaign (section 11) | batch queue | hours to days (queue waits) |
| Collect and send back (sections 14, 17) | login node, your machine | 30 min |

## 2. Account, 2FA and login nodes

**[to be verified on Hunter]** Steps from the HLRS knowledge base (pages
`Reactivating_Your_HLRS_User_Account`, `Hunter_access`, `Secure_Shell_ssh`):

1. Create an ed25519 SSH key with a passphrase on your machine
   (`ssh-keygen -t ed25519`).
2. The project manager registers the public key and your source IP address in
   the HLRS project portal. The key becomes active the next day. Users without
   a static IP use the HLRS VPN (FortiClient/openfortivpn to
   `rmgw.hww.hlrs.de`; the project supervisor must add the `vpn-hww`
   resource).
3. Log in once to `initial.hww.hlrs.de` and run `generate_2fa_otp.sh` to create
   the TOTP secret. Access propagates within about 1 h.
4. Log in with `ssh hunter.hww.hlrs.de` (load balancer; IPv6:
   `hunter6.hww.hlrs.de`) or directly to `hunter-login03/04/05`.

Notes:

- Password login is disabled, TOTP 2FA is mandatory, and `~/.ssh/authorized_keys`
  is not used.
- A 2FA prompt is valid for 1 h per source/target pair. The round-robin alias
  can cause repeated prompts, so pin one login node in `~/.ssh/config`:

  ```text
  Host hunter
      HostName hunter-login04.hww.hlrs.de
      User <hlrs-user>
      IdentityFile ~/.ssh/id_ed25519
  ```

- Login nodes are for editing, compiling, data movement and job submission.
  They have a **2 h CPU-time limit** per process. Prefix heavy commands with
  `nice -n 19`.
- 2FA is mandatory, and HLRS's planned 2FA-free workflow hosts explicitly
  forbid scripts that generate and submit jobs. So `00_submit_all.sh` must be
  run by a person on a normal login node.
- HLRS guidance for AI agents (kb `AI_Agents_on_hww_clusters`, 2026-09-01):
  loading modules, submitting and cancelling jobs stay under direct human
  control. Do not give an agent a persistent SSH session.

## 3. Workspace (never HOME)

HOME is NFS with a 50 GB quota, and HLRS states it "is not intended for use in
any compute jobs". Everything the jobs read or write goes into a Lustre
workspace: code, venv, caches, data and outputs. There is no backup of any
file system.

**[to be verified on Hunter]**

```bash
ws_allocate impact 60          # name "impact", 60 days (the maximum); prints the path
WS=$(ws_find impact)           # e.g. /lustre/hpe/ws13/ws13.a/ws/<user>-impact
ws_list -l                     # remaining lifetime
ws_extend impact 60            # at most 3 extensions, then automatic deletion
ws_allocate -r 7 -m <mail> ... # optional: reminder e-mails before expiry
ws_quota                       # capacity and file-count quota of the project
ws_share share impact <user>   # read access for a colleague (e.g. the author)
```

- `ws13` is the default file system for existing projects. On the other one
  (`ws12`) a project gets only 1/5 of its quota.
- If a user or group exceeds its quota, "no further jobs will be executed".
  The pipeline writes many small JSON files per shard. Watch `ws_quota` during
  large campaigns.
- Workspaces expire. Copy results off-site before expiry (section 14).

Use this layout (the rest of the runbook assumes it):

```text
$WS/
  impact-synergy-pipeline/        code: clone of the release tag (section 4)
  venvs/impact-hunter/            cray-python venv (section 5)
  cache/                          CuPy, numba, matplotlib, XDG caches (set by the setup file)
  hunter_pbs_setup.sh             job setup file (section 5.3)
  data/<dataset>/                 BIDS dataset with events.tsv and sidecars (section 8)
  outputs/<dataset>/preprocessed/ extracted time series: the campaign input (section 8)
  outputs/<dataset>/cache/        campaign, step-2 tables, provenance (sections 9-14)
```

## 4. Getting the code

Compute nodes have no internet. Login nodes reach the internet only through a
reverse SOCKS tunnel that **you** open from your own machine.

The release to run is tagged by the author. At the time of writing the planned
tag is `v1.1.0`; use the tag the author gives you.

### 4.1 Option A: clone on a login node through the SOCKS tunnel

**[to be verified on Hunter]** (kb `SSH_Tunnel_with_Proxy`, 2026-05-22)

```bash
# on your machine: pick a free port between 10000 and 60000
MY_PROXY_PORT=<port>
ssh -R localhost:$MY_PROXY_PORT hunter

# on the Hunter login node, in that SSH session
export https_proxy=socks5://localhost:<port> http_proxy=socks5://localhost:<port>
cd "$WS"
git clone --branch v1.1.0 --depth 1 https://github.com/IronmanAOY/IMPACT.git impact-synergy-pipeline
```

Caveats:

- Every user on Hunter can use an open tunnel into your network. Close the SSH
  session as soon as the downloads are done.
- The page `Secure_Shell_ssh` (2025-08-26) carries a notice that "reverse
  connections to the user network are no longer permitted", while
  `SSH_Tunnel_with_Proxy` (2026-05-22) still prescribes this tunnel. Ask HLRS
  which applies (open question 10).

### 4.2 Option B: clone on your machine and copy it

This needs no tunnel. Copy the whole clone including `.git`, so that the
provenance records the exact commit.

```bash
# on your machine (the tag must have been pushed by the author)
git clone --branch v1.1.0 https://github.com/IronmanAOY/IMPACT.git impact-synergy-pipeline
rsync -a impact-synergy-pipeline/ hunter:<WS path>/impact-synergy-pipeline/
```

**[to be verified on Hunter]**: transfers must be started from your machine
(login nodes cannot connect back), and large `scp`/`rsync` transfers through a
login node may hit its CPU limit (section 8.3). `<WS path>` is the output of
`ws_find impact` on Hunter.

If you deploy a copy without `.git` (for example a tarball), record the version
for provenance in the setup file: `export IMPACT_CODE_VERSION=<commit>` (the
commit only; it is recorded as `1.1.0+<commit>`). From a git checkout, recorded
code versions look like `1.1.0+g<40-hex commit>`, with `.dirty` when the
checkout has local changes.

## 5. Python environment

HLRS's documented Python route on Hunter is `cray-python` plus a venv created
with `--system-site-packages`, so that the Cray LibSci-linked numpy and scipy
are reused. Conda is not documented for Hunter (open question 4). The pipeline
runs from the checkout (`run_pipeline.py` puts `<repo>/src` on `sys.path`). It
is **not** pip-installed on Hunter, because `pyproject.toml` asks for
`scipy>=1.11` and the default stack ships scipy 1.10.1.

### 5.1 Software stacks

| Stack | CPE | ROCm | cray-python | numpy / scipy | Status |
|---|---|---|---|---|---|
| `HLRS/APU/2026.1` | 25.09 | 6.4.1 | 3.11.7 | 1.24.4 / 1.10.1 | default since 2026-03-30; **recommended** |
| `HLRS/APU/testing-2026.2` | 26.03 | 7.0.2 | 3.12.12 | 2.3.5 / 1.16.3 | testing stack (CPE components only) |

`requirements-hunter.txt` and `constraints-hunter.txt` in the repository target
the default stack. The constraints file pins numpy 1.24.4 and scipy 1.10.1 so
that pip does not shadow the Cray builds.

### 5.2 Install (login node, once)

`scripts/hunter/install_hunter_env.sh` prints every step by default and runs
them only with `--run`. **[checked]**: the dry run prints the steps (tested by
`tests/test_hunter_scripts.py`). **[to be verified on Hunter]**: the steps
themselves.

```bash
cd "$WS/impact-synergy-pipeline"
bash scripts/hunter/install_hunter_env.sh --cupy source-13.6          # print the steps
# open the SOCKS tunnel (section 4.1) and export https_proxy/http_proxy, then:
bash scripts/hunter/install_hunter_env.sh --cupy source-13.6 --run    # execute them
```

What the script runs, in order (you can also type these by hand):

```bash
module load HLRS/APU/2026.1 cray-python rocm      # "rocm" module name: to be verified on Hunter
WS=$(ws_find impact || true); if [ -z "$WS" ]; then WS=$(ws_allocate impact 60); fi; export WS
python3 -m venv --system-site-packages "$WS/venvs/impact-hunter"
source "$WS/venvs/impact-hunter/bin/activate"
python3 -m pip install /sw/general/x86_64/development/python/share/PySocks-1.7.1-py3-none-any.whl
nice -n 19 python3 -m pip install -c constraints-hunter.txt -r requirements-hunter.txt
# CuPy: section 6
PYTHONPATH="$PWD/src" python3 -m impact_pipeline.hardware_selftest --target cpu --size 64
```

- pip needs PySocks to use the SOCKS tunnel. HLRS provides the wheel at the
  path above.
- Options: `--ws-name NAME` (default `impact`), `--venv DIR` (default
  `$WS/venvs/impact-hunter`), `--cupy none|source-13.6|wheel-rocm7`.
- Check that pip kept the Cray numpy/scipy:

  ```bash
  python3 -c "import numpy, scipy; print(numpy.__version__, numpy.__file__, scipy.__version__)"
  ```

  You expect 1.24.4 and 1.10.1 from the cray-python installation, not from the
  venv (to be verified on Hunter).

### 5.3 The job setup file

Every generated job script starts with `#!/bin/bash` (the `module` command
needs bash), runs `set -eo pipefail`, sources the setup file, and only then
enables `set -u`, because it is not known whether Lmod works under `nounset`
(open question 13).

```bash
cp scripts/hunter/hunter_pbs_setup.sh "$WS/hunter_pbs_setup.sh"   # edit the copy if needed
export IMPACT_HUNTER_SETUP_FILE="$WS/hunter_pbs_setup.sh"
source "$IMPACT_HUNTER_SETUP_FILE"
```

The template (`scripts/hunter/hunter_pbs_setup.sh`) does the following. Every
value can be overridden by exporting the variable before sourcing.

1. `module load HLRS/APU/2026.1`, `cray-python` and `rocm`
   (`IMPACT_HUNTER_STACK`, `IMPACT_HUNTER_ROCM_MODULE`).
2. Finds the workspace (`IMPACT_WS_NAME`, default `impact`) and fails with a
   clear message if it is missing.
3. Activates the venv (`IMPACT_VENV`) and exports `IMPACT_HUNTER_PYTHON`,
   `IMPACT_SKIP_ENV_CHECK=1` (the pipeline otherwise insists on the conda env
   name `impact-synergy-clean`) and `PYTHONNOUSERSITE=1`.
4. Points `CUPY_CACHE_DIR`, `MPLCONFIGDIR`, `NUMBA_CACHE_DIR`, `XDG_CACHE_HOME`,
   `MNE_DATA` and `NILEARN_DATA` into `$WS` (CuPy's default kernel cache is
   under HOME).
5. Sets `ROCM_HOME` (from `ROCM_PATH`, else `/opt/rocm`).
6. Sets `OMP_NUM_THREADS=22` and `OMP_PROC_BIND=close` (one rank per APU = 24
   cores, 2 of them left for the OS). The shard jobs themselves use one thread
   per worker process.
7. Inside a job, unsets `http(s)_proxy` so that an accidental download fails
   immediately.

**Important:** `qsub` does not forward your login shell's environment to the
jobs (the generated scripts do not use `-V`). Any variable that the jobs need
at run time (`IMPACT_HUNTER_FORCE`, `IMPACT_IIM_*`, `IMPACT_EIGH_BACKEND`,
`IMPACT_CODE_VERSION`) belongs in `$WS/hunter_pbs_setup.sh`. Variables that
shape the generated scripts (`IMPACT_HUNTER_*` in section 19) are read when the
campaign is **built**, so export them before the build.

### 5.4 Alternative: testing stack 2026.2

**[to be verified on Hunter]**. Only if HLRS confirms that production jobs may
use `HLRS/APU/testing-2026.2` (open question 7):

- export `IMPACT_HUNTER_STACK=HLRS/APU/testing-2026.2` before sourcing the setup
  file and before running the install script;
- in `constraints-hunter.txt` use `numpy==2.3.5` and `scipy==1.16.3`; in
  `requirements-hunter.txt` use `numba==0.62.1`;
- install CuPy with `--cupy wheel-rocm7` (`cupy-rocm-7-0>=14.1`);
- `mne==1.6.1` predates NumPy 2. The reference environment runs it on NumPy
  2.2 for the calls the pipeline uses (EEG reading, filtering, resampling), but
  not on NumPy 2.3.5, and the pipeline has **not** been validated with
  `mne>=1.7`. Tell the author if you use this route.

## 6. CuPy for ROCm

`--hardware-target hunter-apu` requires CuPy built for ROCm/HIP. HLRS provides
no CuPy module that we know of (open question 5). HLRS states that using the
GPU cores is mandatory on Hunter, so the campaign's Ψ kernels run on the APUs.

### 6.1 Default stack 2026.1 (ROCm 6.4.1): CuPy 13.6.0 from source

**[to be verified on Hunter]**

```bash
source "$WS/venvs/impact-hunter/bin/activate"
export https_proxy=socks5://localhost:<port> http_proxy=socks5://localhost:<port>   # tunnel open
export CUPY_INSTALL_USE_HIP=1
export ROCM_HOME=${ROCM_PATH:-/opt/rocm}
export HCC_AMDGPU_TARGET=gfx942          # MI300A; without it the build targets the build host's GPUs
nice -n 19 python3 -m pip install -c constraints-hunter.txt cupy==13.6.0
```

- CuPy 14 dropped ROCm 6, so 13.6.0 is the last line for this stack.
- The source build needs these ROCm libraries: hipblas, hipsparse, rocsparse,
  rocrand, hiprand, rocthrust, rocsolver, rocfft, hipfft, hipcub, rocprim, rccl
  and roctracer. It is not confirmed that Hunter's `rocm` module has all of them
  (open question 6).
- The build may exceed the login node's 2 h CPU-time limit. If it is killed,
  ask HLRS where to build (open question 5). One option to discuss with them is
  to download the source distribution on the login node
  (`python3 -m pip download --no-deps --no-binary=:all: cupy==13.6.0 -d "$WS/wheels"`)
  and build it in an interactive job, which has no internet.
- `requirements-hunter.txt` pins `fastrlock` (CuPy's runtime dependency), so a
  locally built CuPy wheel can be installed with `--no-deps`.

### 6.2 Testing stack 2026.2 (ROCm 7.0.2): official wheel

**[to be verified on Hunter]**

```bash
python3 -m pip install 'cupy-rocm-7-0>=14.1'
```

Use 14.1.0 or newer: 14.0.x emitted 32-bit warp-shuffle masks that broke hiprtc
kernels on ROCm 7 (fixed upstream in CuPy 14.1.0).

### 6.3 Check CuPy on a compute node

**[to be verified on Hunter]** Login nodes have no MI300A, so CuPy can only be
checked on a compute node:

```bash
qsub -I -v TERM -l select=1:node_type=mi300a -l walltime=01:00:00   # interactive job (queue "interactive", max 8 h)
source "$WS/hunter_pbs_setup.sh"
python3 -c "import cupy as cp; print(cp.cuda.runtime.is_hip, cp.cuda.runtime.getDeviceCount()); cp.show_config()"
```

You expect `True 4`. A known HLRS issue (2025-12-06) says GPUs may appear with
IDs 4-7 in `rocm-smi` (open question 9).

Known CuPy-on-ROCm limitations that matter here:

- `cupy.linalg.eigh` is listed as "not yet supported" on ROCm, but CuPy routes it
  to hipSOLVER's Jacobi solver. The pipeline uses the device `eigh` only after a
  NumPy parity check and otherwise falls back to the CPU
  (`IMPACT_EIGH_BACKEND=auto|cpu|device`).
- HPE warns that `cray-libsci_acc` "may generate wrong numerical results on AMD
  GPUs with ROCm 6.4.0 and 6.4.1" (rocBLAS). The self-test compares matmul,
  SVD, pinv and the IIM kernels with NumPy for exactly this reason.

## 7. Hardware self-test (run this first)

Run the self-test on a compute node **before any campaign**, and again whenever
the stack, the venv or CuPy changes. The campaign shards run their Ψ work
through exactly the kernel that the case `iim_psi_xp_parity` checks.

**[to be verified on Hunter]** in an interactive job (section 6.3) or a short
`test`-queue job:

```bash
cd "$WS/impact-synergy-pipeline"
source "$WS/hunter_pbs_setup.sh"
PYTHONPATH="$PWD/src" python3 -m impact_pipeline.hardware_selftest \
    --target hunter-apu --json "$WS/outputs/hardware_selftest_$(date +%Y%m%d).json"
echo "exit code: $?"
```

The smoke job of section 10 runs the same test as its first step and writes
`<campaign>/hardware_selftest.json`.

**[checked]** The CPU variant, and the output format (workstation, NumPy against
NumPy):

```text
$ PYTHONPATH=src python -m impact_pipeline.hardware_selftest --target cpu --size 64
IMPaCT hardware self-test: cpu->cpu
  code version: 1.1.0+gf44f5390f393aa5a29ef5d223c48c9a0fded99dd; IIM Psi kernel for this target: numba
  matmul             OK   rel_err=0.00e+00 t=0.000s
  eigh               OK   rel_err=0.00e+00 t=0.001s
  ufunc_add_at       OK   rel_err=0.00e+00 t=0.000s
  bincount_weighted  OK   rel_err=0.00e+00 t=0.000s
  svd_values         OK   rel_err=0.00e+00 t=0.000s
  pinv               OK   rel_err=0.00e+00 t=0.000s
  psd_invsqrt        OK   rel_err=0.00e+00 t=0.001s
  iim_tpm_kernels    OK   rel_err=0.00e+00 t=0.408s
  iim_psi_xp_parity  OK   rel_err=7.89e-16 t=0.580s
  eigh on device: False (CPU fallback otherwise)
  overall: OK
```

On a workstation without CuPy, `--target hunter-apu` prints
`Hardware target unavailable: ... requires CuPy ...` and exits with code 2
**[checked]**.

How to read the output on Hunter:

- The first line should read
  `IMPaCT hardware self-test: hunter-apu->hunter-apu runtime=rocm devices=4 device0=...`,
  and the `device[i]` lines should list the MI300A devices (`gfx942`). The IIM Ψ
  kernel for this target must be `xp`; `numba` means that the accelerator was
  not resolved. A `visible-device env` line shows `HIP_VISIBLE_DEVICES`,
  `ROCR_VISIBLE_DEVICES`, `PMI_LOCAL_RANK` and the PBS IDs when they are set.
- Each case compares the device result with NumPy on the host. `rel_err` is the
  maximum absolute difference divided by the largest absolute reference value.
  A case passes when `rel_err <= rtol` (default `1e-8`; `--rtol` changes it).
- `iim_psi_xp_parity` compares phase-1 Ψ, the bidirectional and directional cut
  TPMs and their Ψ from the device kernel with the host (numba) reference
  kernel. On the CPU they agree to about 1e-15. A failure here means the
  campaign would compute wrong IIM values: **do not run a campaign**.
- `eigh FAIL` alone does not fail the test. The pipeline then uses the CPU
  `eigh`, and `eigh on device: False` confirms it.

| Exit code | Meaning | Action |
|---|---|---|
| 0 | every required case passed | continue |
| 1 | a comparison failed | stop; send the JSON report to the author (section 17) |
| 2 | no accelerator visible (CuPy missing, no HIP device) | check the venv, CuPy (section 6.3), and that you are on a compute node |

CPU/GPU parity in the campaign itself: IIM values computed on the APU should
equal a CPU computation of the same problem up to rounding. If the author asks
for a parity check, build a second tiny campaign with `--hardware-target cpu` on
the same data (section 10) and send both `iim_results.csv` files.

## 8. Data staging

The campaign needs two inputs in the workspace:

1. **The BIDS dataset** (`--bids-root`): at least `dataset_description.json`,
   the `events.tsv` files and the JSON sidecars (TR). RAM and SRPI read the
   events in the finalize job. The NIfTI/BrainVision data files are needed only
   if you run the preprocessing on Hunter.
2. **The extracted time series** under `<out-dir>/preprocessed/`:

   ```text
   <out-dir>/preprocessed/<subject>/<session>/<condition>/<subject>_run-<k>_<atlas>_ts.npy   (time x regions)
   <out-dir>/preprocessed/<subject>/<session>/rest/...                                      (PDI baselines)
   ```

   For ds003171: condition `audio`, atlas `schaefer400`, sessions `awake` and
   `deep`. For ds005620 (EEG): condition `eeg`, atlas key `eeg64`.

All paths are written into the generated job scripts when the campaign is
built. **Stage the data at its final workspace location first, then build the
campaign on Hunter.**

### 8.1 Download (on a machine with internet)

Pinned OpenNeuro snapshots through DataLad or git-annex (usage as documented in
the script header, `bash scripts/download_data.sh --help`; the downloads were
not repeated for this document):

```bash
bash scripts/download_data.sh ds003171 2.0.1 /data/openneuro/ds003171    # fMRI
bash scripts/download_data.sh ds005620 1.0.0 /data/openneuro/ds005620    # EEG
```

### 8.2 fMRIPrep and preprocessing (on a machine with Docker, or as the author says)

fMRI time series are extracted from fMRIPrep derivatives, which the pipeline
expects at `<bids-root>/derivatives/fmriprep` (or `--fmriprep-dir`). fMRIPrep
runs in Docker (`scripts/fetch_fmriprep.sh`, image `nipreps/fmriprep:25.1.3`),
which Hunter does not offer. Run it off-cluster, or obtain the derivatives from
the author.

Preprocessing (atlas time series from fMRIPrep derivatives; EEG channel time
series from BrainVision) runs before the campaign build in the same command
when you add `--run-preprocessing`. The cheapest way to produce only the
`preprocessed/` tree is a campaign build on your workstation with
`--run-preprocessing`: it preprocesses, prepares a campaign that you then
discard, and does not compute the step-2 metrics. Keep the IIM subsystem of
this throw-away campaign small (`--iim-max-nodes 4`). Without it the build
uses the 10-node default, and the cost guard of section 10a stops the command
with `HunterCostError` after preprocessing has finished (the `preprocessed/`
tree is then complete, but the command exits with an error).

```bash
# on a workstation with the conda env (README.md), fMRIPrep derivatives present
python run_pipeline.py --execution-mode hunter --hunter-stage build-campaign \
    --hardware-target cpu --run-preprocessing \
    --dataset-id ds003171 --bids-root /data/openneuro/ds003171 \
    --out-dir /data/impact_out/ds003171 --mpc-metrics RAM PDI NAS IIM SRPI \
    --iim-max-nodes 4
```

**[checked]**: this command form (build-campaign on a workstation) was run for
this document on a tiny synthetic layout without `--run-preprocessing`. The
preprocessing step itself is the one of every local run. Preprocessing on
Hunter is possible in an interactive job (not on a login node: 2 h CPU limit),
but it has not been tried there.

Without a plausible TR in the NIfTI header or BIDS sidecar, preprocessing stops
with an error. It never falls back to a default TR silently. Pass
`--assume-tr <seconds>` only if the author tells you to.

### 8.3 Transfer into the workspace

**[to be verified on Hunter]** (kb `Data_Transfer`, `SSH_Tunnel_with_Proxy`)

```bash
# from your machine; transfers must be started from your side
rsync -a --info=progress2 /data/openneuro/ds003171/ hunter:<WS path>/data/ds003171/
rsync -a /data/impact_out/ds003171/preprocessed/ hunter:<WS path>/outputs/ds003171/preprocessed/
```

- HLRS warns that `scp` through login nodes can fail on the 2 h CPU limit. For
  large data (tens of GB and more) use UFTP (`gridftp-fr1.hww.hlrs.de:9000`;
  send your SSH public key via ticket; add `--encrypt` with
  `UFTP_ENCRYPTION_ALGORITHM=AES`) or GridFTP (needs a registered X.509
  certificate). Ask HLRS for the recommended path (open question 11).
- Lustre handles many small files poorly. Pack large trees into a tar file,
  transfer it, and unpack it in the workspace.
- DataLad/git-annex datasets contain symlinks into `.git/annex`. Use
  `rsync -aL` (or `datalad get` first and copy the resolved files) so the
  workspace holds real files.
- Check what arrived: `find "$WS/outputs/ds003171/preprocessed" -name '*_ts.npy' | wc -l`.

## 9. Building the campaign (login node)

### 9.1 Parameters from the author

Fill in this table from the author's instructions before you build. Do not
guess IIM sizes: the cost of IIM grows steeply with the number of nodes.

| Setting | Flag | Value |
|---|---|---|
| Dataset | `--dataset-id` | e.g. `ds003171` |
| Subjects (optional) | `--subjects` | all, or a list |
| Metrics | `--mpc-metrics` | `RAM PDI NAS IIM SRPI` (IIM is required for a campaign) |
| IIM subsystem size | `--iim-max-nodes` | as given by the author. The pipeline uses 3 bins per node (3^N states; above 1500 states the bins drop to 2). Cost grows roughly 30x per node; unset means 10 nodes, which the build refuses (section 10a) |
| IIM cut sample | `--iim-n-parts` | unset = exhaustive |
| IIM mechanism/purview sizes | `--iim-max-mechanism-size`, `--iim-max-purview-size` | unset = all |
| Size guard | `--hunter-max-psi-evals`, `--hunter-allow-large`, `--hunter-seconds-per-psi-eval` | default ceiling 1e11 Ψ evaluations; the measured rate from the calibration smoke test (section 10a) |
| MPC protocol | `--protocol` | the protocol file the author gives you: `protocols/mpc_default_v1.json` (the preregistered default for real data, used when `--protocol` is not given: estimator modes, cutoffs, null families, reference) with the NAS hub declared for the dataset's grain, e.g. `protocols/examples/mpc_default_v1_schaefer400_7networks_hub.json` for `schaefer400` (an example; the hub has to be preregistered before confirmatory use). v1 itself declares no hub: without a derived protocol NAS is UNDEFINED (`NO_DECLARED_WORKSPACE`) in every run. The hash is recorded, and the campaign computes IIM with the protocol's IIM options. Do not use `protocols/mpc_behavioural_ram_v1.json` (opt-in: RAM can be ABSENT on behavioural evidence alone; an ABSENT RAM then means no responsiveness-and-adaptation above the null in the recorded behaviour, not absence of responsiveness) unless the author asks for it |
| IIM surrogate runs per real run | `--hunter-iim-null-surrogates K` | e.g. 19 (for `IIM_null_p`, K >= 19) |
| IIM bootstrap replicate runs per real run | `--hunter-iim-bootstrap-se B` | as given by the author (e.g. the same B as `--bootstrap-se`); 0 = IIM has no sampling SE |
| Null surrogates for the evidence layer | `--null-surrogates K` | e.g. 19; 0 = every legacy-mode component `NO_NULL_CALIBRATION` |
| Bootstrap SE for the evidence layer | `--bootstrap-se B`, `--bootstrap-block-len L` | e.g. 100; 0 = every verdict UNDETERMINED (`NO_SAMPLING_SE`); block default ceil(sqrt(n_time)) samples |
| Necessity set | `--necessity-set` | default all five (must match `--protocol`) |
| Applicability registry | `--applicability-registry` | JSON path, if given |
| CI reference | `--ci-reference` | legacy CI only; default `cohort_high_state` |
| Walltimes | `IMPACT_HUNTER_PHASE1_TIME`, `..._CUT_TIME`, `..._REDUCE_TIME` | defaults 24 h / 24 h / 4 h |
| Shards per run | `--hunter-phase1-shards-per-run`, `--hunter-cut-shards-per-run` | defaults 16 / 256 |

With `--null-surrogates K > 0` and `--bootstrap-se B > 0` the finalize job
also computes K surrogate and B bootstrap evaluations of RAM, PDI, NAS and SRPI
per run on one node (the IIM ones come from the campaign). Raise
`IMPACT_HUNTER_REDUCE_TIME` (at most 24 h) if the author expects this to be
long. The finalize job uses the evidence options of the build (they are stored
in the campaign manifest), so its command line does not repeat them.

### 9.2 Build

**[to be verified on Hunter]** for the environment. **[checked]** for the
command itself: the command below (with a placeholder setup file and
interpreter path) was run on a workstation against a tiny synthetic layout (2
subjects x 2 sessions, 6 regions, 120 time points).

```bash
cd "$WS/impact-synergy-pipeline"
export IMPACT_HUNTER_SETUP_FILE="$WS/hunter_pbs_setup.sh"
source "$IMPACT_HUNTER_SETUP_FILE"               # also exports IMPACT_HUNTER_PYTHON
# optional build-time settings, e.g.:
# export IMPACT_HUNTER_PHASE1_TIME=12:00:00 IMPACT_HUNTER_CUT_TIME=12:00:00
# export IMPACT_HUNTER_PBS_GROUP_LIST=<group>    # only for a non-default project (id -Gn)

nice -n 19 python3 run_pipeline.py --execution-mode hunter --hunter-stage build-campaign \
    --hardware-target hunter-apu \
    --dataset-id ds003171 \
    --bids-root "$WS/data/ds003171" \
    --out-dir "$WS/outputs/ds003171" \
    --mpc-metrics RAM PDI NAS IIM SRPI \
    --iim-max-nodes <N> \
    --protocol <PROTOCOL> \
    --hunter-iim-null-surrogates <K> --null-surrogates <K> \
    --hunter-iim-bootstrap-se <B> --bootstrap-se <B>
```

`<PROTOCOL>` is the protocol file from the table in 9.1 (derived from
`protocols/mpc_default_v1.json`, with the declared NAS hub). Check the
`MPC protocol: <path> (hash <12 hex digits>)` line of the log against the
hash the author gives you. The PBS jobs of the campaign run without
`--protocol` and use the protocol recorded in the campaign, so their logs
show the same line, ending in `; the campaign's protocol`, followed by a
note that the command-line default is not used. In the PBS jobs this line
comes before the dataset, hardware and provenance setup.

What happens:

- The login node has no APU, so you see a warning that `hunter-apu` is not
  available on the build host and the campaign is prepared on the CPU. The TPMs
  are numerically identical; the jobs still request `hunter-apu`. This warning
  is expected.
- Before it prepares anything, the build counts the IIM work in Ψ evaluations
  and prints an order-of-magnitude runtime **[checked]**; above the ceiling it
  stops with `HunterCostError: Hunter IIM campaign refused: ...` and writes
  nothing (section 10a). For 2 runs at 6 nodes with 2 surrogate runs each:

  ```text
  INFO:impact_pipeline.hunter_iim:IIM cost estimate (closed form before preparation; upper bound): 2 real run(s) x (1 + 2 surrogate + 0 bootstrap) = 6 IIM runs; 1.74e+07 Psi evaluations in 192 TPM evaluations (unit: ...)
  INFO:impact_pipeline.hunter_iim:  2 real run(s) with 6 nodes, 3 bins: 63 mechanisms (301 bipartitions) x 63 purviews (301) = 90,601 per TPM x 32 TPMs (intact + 31 of 31 cuts) = 2,899,232 per IIM run.
  INFO:impact_pipeline.hunter_iim:  Runtime (order of magnitude only; reference rate 0.00056 s per evaluation (...), divided by 22 worker(s) per shard; not measured on Hunter): 2.71 single-core hours at the reference rate; about 0.0307 node-hours with 4 shard(s) per node; longest phase-1 shard 0.47 s, longest cut shard 2.3 s.
  INFO:impact_pipeline.hunter_iim:  Ceiling: 1e+11 Psi evaluations (default): within the ceiling.
  ```

- The build estimates one TPM per run, per surrogate run and per bootstrap
  replicate run. This is far cheaper than the campaign, but it grows with the
  number of runs and with K and B (the tiny checked build took seconds). If it approaches the login node's 2 h CPU limit,
  run the same command in an interactive job.
- The last log lines name the campaign directory and the submit command:

  ```text
  INFO:pipeline:Hunter campaign prepared: runs=12 phase1_tasks=180 cut_tasks=36 scheduler=pbs shards_per_node=4 workers_per_task=22 dir=<out-dir>/cache/hunter_iim_campaign
  INFO:pipeline:Submit on a Hunter login node with: bash <out-dir>/cache/hunter_iim_campaign/pbs/00_submit_all.sh
  ```

  (from the checked run: 4 real runs + 2 surrogate runs each = 12 runs).

### 9.3 What the build writes

```text
<out-dir>/cache/hunter_iim_campaign/
  campaign_manifest.json        runs, tasks, settings, step-2 context (paths, evidence options),
                                iim_cost_estimate (Ψ evaluations, runtime estimate, ceiling decision)
  runs/<run-key>/               per run: tpm_full.npy, states_full.npy, curr_obs.npy, meta.json,
                                mechanisms.json, purviews.json, cuts.json
  runs/<run-key>__nullNNN/      surrogate runs (with surrogate_ts.npy)
  runs/<run-key>__bootNNN/      bootstrap replicate runs (with bootstrap_ts.npy)
  pbs/00_submit_all.sh          submits the three jobs with dependencies
  pbs/01_phase1_shards.pbs      array: phase-1 Ψ shards
  pbs/02_cut_shards.pbs         array: cut Ψ shards
  pbs/03_reduce_finalize.pbs    one job: reduce-all, then finalize-pipeline
  pbs/90_smoke_all_in_one.pbs   everything in one job for the test queue
  pbs/campaign_plan.json        tasks, subjobs, walltimes, node-hour upper bounds, IIM cost estimate
  pbs/logs/                     PBS output files land here
<out-dir>/cache/provenance_manifest.json   status "hunter_campaign_built"
```

Check the generated scripts before you submit **[checked: content of the
generated files]**. `01_phase1_shards.pbs` from the checked build:

```bash
#!/bin/bash
#PBS -N impact_p1
#PBS -l select=1:node_type=mi300a:mpiprocs=4
#PBS -l walltime=24:00:00
#PBS -l ws13=True
#PBS -j oe
#PBS -J 0-44
#PBS -r y
# IMPaCT IIM phase-1 shards: 180 tasks, 4 per node, 45 node(s).
set -eo pipefail
# Site setup is sourced before `set -u` (Lmod under nounset is unverified).
source <WS>/hunter_pbs_setup.sh
set -u
cd <WS>/impact-synergy-pipeline
export IMPACT_REPO_ROOT=<WS>/impact-synergy-pipeline
# The setup above pins the interpreter; skip the conda check.
export IMPACT_SKIP_ENV_CHECK=1
# One BLAS/OpenMP thread per worker process (workers are processes).
export OMP_NUM_THREADS=1
...
array_index="${PBS_ARRAY_INDEX:-0}"
mpiexec -n 4 --ppn 4 --cpu-bind list:0-23:24-47:48-71:72-95 --gpu-bind list:0:1:2:3 \
  <venv>/bin/python3 <WS>/impact-synergy-pipeline/run_pipeline.py --execution-mode hunter \
  --hunter-campaign-dir <out-dir>/cache/hunter_iim_campaign --hardware-target hunter-apu \
  --dataset-id ds003171 --data-origin real --out-dir <out-dir> \
  --hunter-stage phase1-shard --hunter-array-index "${array_index}" --hunter-shards-per-node 4
```

Check in particular:

- the `source` line points at your setup file, and `cd` at your checkout in the
  workspace (never HOME);
- the interpreter is the venv's `python3`;
- `--hardware-target hunter-apu` appears in **both** shard scripts;
- `pbs/campaign_plan.json`: `node_hours_upper_bound` per stage is subjobs x
  walltime, the worst case that can be charged. Compare it with the project's
  budget before submitting (section 15). `iim_cost_estimate.node_hours_estimate`
  is the expected work (order of magnitude; section 10a).

## 10. Smoke test on the test queue

The `test` queue allows 25 min per job, 1 job per user and no production runs.
The smoke test builds a tiny campaign (4 IIM nodes, mechanisms and purviews of
at most 2 nodes, 4 cuts, one shard of each kind per run) and submits
`pbs/90_smoke_all_in_one.pbs`, which runs, in one job: the hardware self-test,
every phase-1 and cut shard (4 per node through PALS `mpiexec`), the merged
reduce and finalize.

**[checked]**: the helper builds the smoke campaign (tested by
`tests/test_hunter_scripts.py`), and its build command with `--data-origin real`
and a symlinked `preprocessed/` tree was run for this document; the generated
`90_smoke_all_in_one.pbs` carries `-q test` and `walltime=00:25:00`.
**[to be verified on Hunter]**: `--submit` and the job itself.

With real data, give the smoke test its own output directory that links to the
real `preprocessed/` tree, because finalize overwrites `<out-dir>/cache/step2_*`:

```bash
cd "$WS/impact-synergy-pipeline"
source "$WS/hunter_pbs_setup.sh"
mkdir -p "$WS/outputs/smoke_ds003171"
ln -s "$WS/outputs/ds003171/preprocessed" "$WS/outputs/smoke_ds003171/preprocessed"
bash scripts/hunter/hunter_smoke_test.sh \
    --dataset-id ds003171 --data-origin real \
    --bids-root "$WS/data/ds003171" --out-dir "$WS/outputs/smoke_ds003171" \
    --subjects "<one or two subject IDs>" \
    --hardware-target hunter-apu --null-surrogates 2 --submit
```

- Without `--submit` the helper only prints the `qsub` command.
- Restrict `--subjects` to one or two subjects (IDs without the `sub-` prefix,
  as in `preprocessed/`), otherwise the job may not fit into 25 min.
- `--null-surrogates 2` exercises the IIM surrogate calibration (two extra
  surrogate runs per run).
- With the default `--data-origin dummy` the helper expects the synthetic
  smoke-test objects (`docs/synthetic_data.md`) and writes under
  `test_objects/`.

After the job has finished, check **[to be verified on Hunter]**:

```bash
C="$WS/outputs/smoke_ds003171/cache/hunter_iim_smoke"
ls "$C/pbs/logs"                                   # impact_smoke.o<jobid>: read it end to end
python3 -m json.tool "$C/hardware_selftest.json" | grep '"all_ok"'
python3 run_pipeline.py --execution-mode hunter --hunter-stage status \
    --hunter-campaign-dir "$C" --dataset-id ds003171 --data-origin real \
    --out-dir "$WS/outputs/smoke_ds003171"
head -3 "$C/iim_results.csv"                       # psi_kernel column should read "xp"
python3 -m json.tool "$C/timing_summary.json"
```

The smoke test passes when the log ends without an error, `all_ok` is true,
`status` reports no missing shards, `iim_results.csv` has one row per real run
with `psi_kernel` equal to `xp`, and `<out-dir>/cache/step2_df.csv` exists.

Every shard also records its wall time and the number of Ψ evaluations it did
(`timing/<stage>/<task>.json`), and the smoke job's `reduce-all` writes
`cost_calibration.json` from them. The tiny smoke campaign is too small to
measure the speed; section 10a uses a larger calibration run.

## 10a. Size the IIM configuration before the full campaign

**Do this before every production campaign, and do not submit one whose IIM
configuration the author has not confirmed from your measurements.**

### Why

IIM evaluates the integration mass Ψ over every (mechanism, purview) pair and
every pair of their bipartitions, once for the intact TPM and once for every
system cut. The number of these terms grows roughly **30x per added subsystem
node**. Exhaustive IIM (all mechanism and purview sizes, all cuts) with
T = 1000 time points took, on one CPU core of a workstation (numba kernel):

| Nodes | Ψ evaluations per IIM run | Measured, one core | Reference estimate, one core |
|---|---|---|---|
| 4 | 5,000 | 1.7 s | 2.8 s |
| 5 | 129,600 | 46 s | 73 s |
| 6 | 2,899,232 | about 27 min | 27 min |
| 7 | 5.97e7 | | 9.3 h |
| 8 | 1.17e9 | | 7.6 days |
| 9 | 2.23e10 | | 140 days |
| 10 | 4.16e11 | | 7.4 years |

(Reference estimate: 5.6e-4 s per Ψ evaluation, the 6-node measurement. The
table comes from `PYTHONPATH=src python3 -m impact_pipeline.hunter_cost
--nodes 4-10`, run in the checkout **[checked]**.)

Every real run also brings K surrogate runs and B bootstrap replicate runs,
each a full IIM run, so the campaign costs (1 + K + B) times the real runs.
Without `--iim-max-nodes` the subsystem is as large as the state budget
allows: 3 bins give 3^N <= 1500 states, so under the default state-budget
policy (`reduce_bins_first`) the bins drop to 2 and N = 10 nodes (with
`reduce_nodes_first` it is 6 nodes at 3 bins). **The 10-node default is
infeasible exhaustively**: about 7 core-years,
or roughly 700 node-hours with perfect packing, per IIM run.

### What the build does

- It counts the work in **Ψ evaluations**: one (mechanism, purview, mechanism
  bipartition, purview bipartition) term of Ψ under one TPM, i.e. mechanisms x
  purviews x bipartitions x (1 + cuts) x runs x (1 + K + B). Before it prepares
  anything it uses closed-form counts from each run's region count (an upper
  bound). After preparation it counts again from the enumerated problems.
- It prints both counts and an **order-of-magnitude runtime** (section 9.2).
  Without a measured rate it uses the reference single-core rate divided by the
  workers per shard. That rate says nothing about the APU kernel, which is why
  you measure it (below).
- It writes both estimates and the decision to `campaign_manifest.json`
  (`iim_cost_estimate`, with `preflight`) and a summary to
  `pbs/campaign_plan.json`. It warns when the longest shard is expected to
  exceed its walltime.
- **It refuses campaigns above the ceiling** and then writes nothing:
  `HunterCostError: Hunter IIM campaign refused: an estimated ... Psi
  evaluations exceed the ceiling of ...`. The default ceiling is 1e11 Ψ
  evaluations, about 160 node-hours at the reference rate. Change it with
  `--hunter-max-psi-evals N` (or `IMPACT_HUNTER_MAX_PSI_EVALS`; `1e12` style is
  accepted), or build anyway with `--hunter-allow-large`. Use either only after
  the author has confirmed the configuration and the budget (section 15).

### Step 1: measure (calibration smoke test)

**[to be verified on Hunter]** Run the plumbing smoke test of section 10
first. Then run a second smoke test that is large enough to time: one subject,
6 nodes, exhaustive, no surrogates, in its own output directory:

```bash
mkdir -p "$WS/outputs/calib_ds003171"
ln -s "$WS/outputs/ds003171/preprocessed" "$WS/outputs/calib_ds003171/preprocessed"
bash scripts/hunter/hunter_smoke_test.sh \
    --dataset-id ds003171 --data-origin real \
    --bids-root "$WS/data/ds003171" --out-dir "$WS/outputs/calib_ds003171" \
    --subjects "<one subject ID>" --hardware-target hunter-apu \
    --iim-max-nodes 6 --iim-max-mechanism-size all --iim-max-purview-size all \
    --iim-n-parts all --submit
```

- With 6 nodes an IIM run is about 2.9 million Ψ evaluations (27 min on one
  CPU core; each shard uses the APU or 22 worker processes). It is expected to
  fit into the 25 min of the `test` queue, but that is not verified. If the job
  hits the walltime, repeat with `--iim-max-nodes 5` and say so in your report.
- After the job, read the measured rate:

  ```bash
  C="$WS/outputs/calib_ds003171/cache/hunter_iim_smoke"
  python3 -m json.tool "$C/cost_calibration.json"
  ```

  The key fields are `shard_wall_seconds_per_psi_evaluation` (the rate),
  `psi_evaluations`, `wall_seconds`, `stages` (phase-1 and cut shards
  separately), `per_shard_wall_seconds_per_psi_evaluation` (spread),
  `overhead_dominated` and `context` (Ψ kernel, backend, hosts, workers,
  packing and the IIM sizes measured). `--hunter-stage status` prints the same
  rate.
- `overhead_dominated: true` means fewer than 1e6 evaluations were timed, so
  start-up costs (imports, compilation, worker pools) dominate and the rate is
  too high. This is always the case for the tiny smoke test of section 10.

### Step 2: extrapolate

- Table for candidate configurations, from the checkout:

  ```bash
  PYTHONPATH=src python3 -m impact_pipeline.hunter_cost --nodes 5-10 \
      --runs <real runs> --null <K> --boot <B> \
      --seconds-per-psi-eval <shard_wall_seconds_per_psi_evaluation>
  # optionally with --max-mechanism-size S --max-purview-size S --n-parts P
  ```

  It prints, per node count, the Ψ evaluations per TPM and per IIM run, the
  campaign total, node-hours at the measured rate and whether the default
  ceiling is exceeded.
- For a specific configuration, add `--hunter-seconds-per-psi-eval <rate>` to
  the build command of section 9.2. The log, `iim_cost_estimate` and
  `pbs/campaign_plan.json` then give node-hours and the longest shard at the
  measured rate. If the build refuses, the configuration is above the ceiling.
- Extrapolate at most one or two nodes beyond the measured size. The time per
  evaluation changes with the number of bins (3 bins up to 6 nodes, 2 above),
  the purview sizes and the device memory. Re-measure at the production size
  when it is far from 6 nodes.

### Step 3: choose, with the author

The IIM sizes are scientific choices. Restricted mechanism/purview sizes and
sampled cuts approximate the exhaustive search, and the author has to declare
them. **Do not pick them yourself**; use the measurements to show what is
feasible:

- `--iim-max-nodes` is the strongest lever (roughly 30x per node).
- `--iim-max-mechanism-size` / `--iim-max-purview-size`: at 10 nodes a limit of
  3 reduces the work per TPM from 8.1e8 to 1.6e5 evaluations (about 5000x).
- `--iim-n-parts P`: the work is proportional to 1 + P sampled cuts (at 10
  nodes, 64 instead of 511 cuts is about 8x less).
- K surrogate and B bootstrap runs multiply everything by (1 + K + B).
- More shards per run (`--hunter-phase1-shards-per-run`,
  `--hunter-cut-shards-per-run`) do not reduce the total work. They only
  shorten the longest shard, which must stay under the 24 h walltime.

### What to report back to the author

Before any production submission, send:

1. `cost_calibration.json` of the calibration run (and of the section-10 smoke
   test), its `timing/` folder, the smoke job logs `pbs/logs/impact_smoke.o*`
   and `hardware_selftest.json`.
2. The `hunter_cost` table for the author's candidate configurations at the
   measured rate, and, for each configuration you built, the `IIM cost
   estimate` log lines or `iim_cost_estimate` from `campaign_manifest.json`.
3. The hardware target and kernel measured (`context.psi_kernels`,
   `context.hardware_backends`), workers per shard and shards per node, and
   anything that failed (walltime hit, out of memory).
4. The project's remaining node-hour budget and the queue limits you saw.

Build and submit the production campaign only with the configuration the author
confirms.

## 11. Full submission

**[to be verified on Hunter]**

```bash
cd "$WS/impact-synergy-pipeline"
source "$WS/hunter_pbs_setup.sh"
bash "$WS/outputs/ds003171/cache/hunter_iim_campaign/pbs/00_submit_all.sh"
```

`00_submit_all.sh` **[checked: content]** changes into `pbs/logs/` (PBS writes
the `<jobname>.o<jobid>` files into the submission directory) and runs:

```bash
jid_p1=$(qsub "${script_dir}/01_phase1_shards.pbs")
jid_cut=$(qsub "${script_dir}/02_cut_shards.pbs")
jid_red=$(qsub -W depend=afterok:${jid_p1}:${jid_cut} "${script_dir}/03_reduce_finalize.pbs")
```

It prints the three job IDs (arrays look like `123456[].hunter-pbs01`) and
writes them to `pbs/submitted_jobs.txt`. Keep that file.

- PBS dependencies work on whole arrays only. `afterok` on an array is expected
  to wait until every subjob has exited 0 (to be verified on Hunter; open
  question 2).
- A stage that needs only one node is written as a plain job instead of an
  array, because PBS arrays need at least 2 subjobs.
- Job names: `impact_p1`, `impact_cut`, `impact_red` (and `impact_smoke`).

## 12. Monitoring

**[to be verified on Hunter]** (kb `Batch_System_PBSPro_(Hunter)`)

```bash
qstat -u $USER                 # your jobs; arrays show as <id>[]
qstat -a                       # all jobs, with elapsed times
qstat -t <id>[]                # the subjobs of an array
qstat -T <id>                  # estimated start time
qstat -f <id>                  # full record: exit status, resources, comment
qdel <id>                      # cancel (arrays: qdel '<id>[]')
batchstat; pbsnodes            # queue and node overview
```

- PBS copies job output into `pbs/logs/` only after the job ends.
- Progress while jobs run **[checked]**: every task writes
  `timing/<stage>/<task>.json` when it finishes, and the status stage counts
  completed shards:

  ```bash
  python3 run_pipeline.py --execution-mode hunter --hunter-stage status \
      --hunter-campaign-dir "$WS/outputs/ds003171/cache/hunter_iim_campaign" \
      --dataset-id ds003171 --data-origin real --out-dir "$WS/outputs/ds003171"
  ```

  It logs, and writes to `status.json`, the number of complete and missing
  phase-1 and cut shards with the missing indices (abbreviated output of a
  small check campaign before any shard had run):

  ```text
  INFO:pipeline:Hunter campaign status: {"phase1-shard": {"complete": 0, "missing": 12, "missing_indices": [0, 1, ...]}, "cut-shard": {...}}
  ```

- A per-task timing file **[checked]** looks like this (cut shard, CPU run):

  ```json
  {"stage": "cut-shard", "task_index": 0, "wall_seconds": 7.51, "cpu_seconds": 40.42,
   "children_cpu_seconds": 40.35, "peak_rss_mb": 361.6, "n_cuts": 3, "resumed_cuts": 0,
   "hardware_backend": "cpu->cpu", "psi_kernel": "numba", "host": "...",
   "scheduler_env": {"PBS_ARRAY_INDEX": "0", "PMI_LOCAL_RANK": "0"}}
  ```

  On Hunter, expect `hardware_backend` `hunter-apu->hunter-apu runtime=rocm ...`
  and `psi_kernel` `xp`.

## 13. Resuming and re-running

The campaign is idempotent **[checked]**:

- **Resubmit after a failure** by running `00_submit_all.sh` again. A shard is
  skipped when its result exists with the same identity (problem digest, shard
  range and code version). The log then says
  `phase1 shard .../shard_0000.json already complete; skipping.` Cut shards also
  checkpoint after every cut, so an interrupted cut shard resumes.
- Each resubmitted array subjob is still allocated a whole node, even if it only
  skips work. Check `status` first; if only a few shards are missing, you can
  still resubmit the whole chain (PBS cannot depend on single subjobs).
- **Force a recomputation** with `export IMPACT_HUNTER_FORCE=1` **in the setup
  file** (jobs do not see your login environment), then resubmit. Remove it
  afterwards.
- **Do not update the checkout while a campaign runs.** Reducers refuse shards
  that were computed by different code versions ("Shards of ... were computed
  by different code versions ... Resubmit the campaign ...").
- **Rebuilding** into an existing campaign directory removes the reduced
  results of the previous build. Finalize refuses final results from another
  build ("... does not belong to the current campaign build; run the reduce-all
  stage again").
- The last job (`03_reduce_finalize.pbs`) can be resubmitted alone once all
  shards are complete: `cd pbs/logs && qsub ../03_reduce_finalize.pbs`.

## 14. Collecting results

After `03_reduce_finalize.pbs` has finished **[checked: file names, from a CPU
run of every stage on a tiny campaign]**:

```text
<out-dir>/
  IMPaCT_Empirical_Validation_<dataset>.docx   Word report
  stats/statistics_summary.json                all statistics (paired tests, Holm, definedness)
  stats/component_tests.csv, theta_tests_S.csv, motion_covariates.csv (fMRI)
  pipe_figures/                                figures
  cache/step2_df.csv                           per run x theta: metrics, CI, MPC verdict and evidence
  cache/step2_df_mean.csv                      per subject x session means
  cache/step2_theta_stats.csv                  exploratory S by theta
  cache/hunter_iim_results.csv                 one row per real run: IIM value, calibration and bootstrap-SE fields
  cache/provenance_manifest.json               code version, runtime versions, parameters, hardware
  cache/hunter_iim_campaign/
      iim_results.csv / iim_results.json       same IIM table as above
      timing_summary.json                      wall/CPU seconds and peak RSS per stage
      timing/<stage>/<task>.json               per-task timing (shards: also Ψ evaluations done)
      cost_calibration.json                    measured seconds per Ψ evaluation (section 10a)
      status.json, campaign_manifest.json, pbs/campaign_plan.json
      pbs/submitted_jobs.txt                   job IDs (written by 00_submit_all.sh)
      pbs/logs/*.o*                            PBS job output
      hardware_selftest.json                   (smoke campaigns; written by the smoke job)
```

Copy results off-site before the workspace expires. From your machine
**[to be verified on Hunter]**:

```bash
rsync -a --exclude 'runs/' --exclude 'preprocessed/' \
    hunter:<WS path>/outputs/ds003171/ ./impact_ds003171_results/
```

The `runs/` folders hold the per-run TPMs and shard payloads. They can be large
and are only needed to re-reduce; keep them (and `preprocessed/`) in the
workspace until the author confirms the results.

## 15. Resources and accounting

**[to be verified on Hunter]** (kb `Batch_System_PBSPro_(Hunter)`, HLRS
application pages)

- **Whole-node charging.** Every `mi300a` job gets a whole node (4 MI300A APUs,
  96 cores, 512 GB HBM3) and is charged for all of it in node-hours. The
  generated scripts therefore pack **4 shards per node**, one per APU, with
  PALS: `mpiexec -n 4 --ppn 4 --cpu-bind list:0-23:24-47:48-71:72-95 --gpu-bind
  list:0:1:2:3`. The shard of a rank is `4 * PBS_ARRAY_INDEX + PMI_LOCAL_RANK`.
  Other packings: `--hunter-shards-per-node` / `IMPACT_HUNTER_SHARDS_PER_NODE`
  at build time; GPU binding for anything other than 4 ranks per node is not
  verified (`IMPACT_HUNTER_PBS_GPU_BIND`).
- **GPU use is mandatory** on Hunter. Both shard stages request `hunter-apu`.
  The reduce/finalize job is short and mostly CPU work on one node. It can be
  sent to a CPU queue with `IMPACT_HUNTER_PBS_CPU_QUEUE` and
  `IMPACT_HUNTER_PBS_CPU_NODE_TYPE` (for example `pre` and `genoa3tb64c`), but
  academic users cannot use the genoa nodes of the normal queues.
- **Walltime** is at most 24 h per job for academic users (the build refuses
  more). The `test` queue allows 25 min. Long shards can be shortened by more
  shards per run (`--hunter-phase1-shards-per-run`, `--hunter-cut-shards-per-run`).
- **Queue limits** (queue `single`, all single-node jobs): 24 h per job, 480 h
  in total per user, 40 queued and 20 running jobs per user (60/30 per group).
  Whether each array subjob counts against these limits, and the maximum array
  size, are not documented (open question 2). The build refuses arrays above
  `IMPACT_HUNTER_MAX_ARRAY_SIZE` (default 10000, the PBS default).
- **Budget check:** `pbs/campaign_plan.json` gives `subjobs`, `walltime` and
  `node_hours_upper_bound` per stage, and `iim_cost_estimate` the expected
  node-hours (order of magnitude; measured rate with
  `--hunter-seconds-per-psi-eval`, section 10a). A test project has 100
  node-hours in total. The build refuses campaigns above
  `IMPACT_HUNTER_MAX_PSI_EVALS` (default 1e11 Ψ evaluations) unless
  `--hunter-allow-large` is given.
- **Job farming:** the GCS fact sheet says job farming "is not an approbate way
  to use a large number of nodes in parallel". The campaign is a set of
  single-node array jobs. Ask HLRS whether this is acceptable (open question 3).
- **MPC-Bench** (`scripts/run_bench.py --n-shards N --pbs-template bench.pbs`)
  writes a PBS array script too, but the bench is CPU-bound (one process pool
  per node). Do not run it on Hunter unless HLRS accepts CPU-only work there
  (open question 3) and the author asks for it.
- `$TMPDIR` is a RAM disk (it uses node memory). The host (numba) kernel keeps
  node-local SQLite caches there, or in `/localscratch/$PBS_JOBID` on the 20
  localscratch nodes (`IMPACT_HUNTER_PBS_LOCALSCRATCH=1`). The device kernel does
  not use them.

### 15.1 MPC-Bench v2 confirmatory run

The MPC-Bench v2 confirmatory run (preregistration
[`MPC_BENCH_PREREGISTRATION_V2.md`](preregistration/MPC_BENCH_PREREGISTRATION_V2.md),
section 11) is **not planned on Hunter**. It is CPU-only (no IIM campaign, no
GPU kernel) and is planned on the author's workstation, whose environment the v2
environment lock records (darwin-arm64, Python 3.10.19, numpy 2.2.6, scipy
1.15.2, numba 0.61.2, pandas 2.3.3; `scripts/v2/env_lock.py --check`). Its size:
11193 tasks (6933 bench tasks and 4260 family-B tasks) plus the oracle checks
on 40 seeds, about 131 CPU-h by the development cost model and about 167 CPU-h
with the family-C1 IIM scorings priced at their measured reference-block rate,
15-19 h wall time at 12 workers.

A run on Hunter would be a deviation from that plan, to be agreed with the
author beforehand and reported. It would need:

- HLRS's agreement to CPU-only work on Hunter (open question 3);
- an environment that matches the lock: the default stack (numpy 1.24.4, scipy
  1.10.1, Python 3.11, section 5.1) does not, and conda is not documented for
  Hunter (open question 4);
- a clean clone of the tag `mpcbench-freeze-v2` (section 4; the tag must have
  been pushed by the author). The v2 confirmatory guard refuses any checkout
  whose `src/` and `scripts/` trees differ from the tag, or with untracked files
  under `src/` or `scripts/`, so keep the venv and the outputs outside the
  clone;
- the stored v1 outputs (`outputs/paper1_mpcbench`, the folder holding `c2/`;
  not versioned) copied to the workspace and named by
  `MPCBENCH_V1_OUTPUTS`, which the regression gate and the audit's IA-1 read;
  without them the gate fails with exit status 2;
- the pre-run checks of preregistration v2, section 11, in the clone:
  `python scripts/v2/env_lock.py --check`, `python scripts/v2/regression_gate.py
  --quick`, `git diff --quiet mpcbench-freeze-v2 -- protocols/v2` (the guard
  compares only `src/` and `scripts/`) and `python scripts/v2/build_protocols_v2.py
  check`; the plan can be previewed with `python scripts/run_bench_v2.py plan
  --split confirmatory` and `python scripts/v2/iim_validation_v2.py --split
  confirmatory --list`;
- jobs within the 24-h walltime limit. `scripts/v2/mpcbench_confirmatory_v2.sh`
  runs its steps in a fixed order, resumes each step (completed tasks are
  skipped, failed tasks run again; a record line cut off by a killed job is
  dropped before the step continues) and takes a subset of steps with `STEPS`
  (an unknown step name stops it before anything runs), so the plan can be
  split into several single-node jobs that each run some steps with `WORKERS`
  set to the cores used and `OUT` on the workspace. By the workstation cost
  model (131-167 CPU-h) one 96-core node at `WORKERS=96` would need about 2-3 h
  of wall time; Hunter's per-core speed and its contention at 96 workers were
  not measured, so plan with a margin. Per-step CPU hours for packing are in
  section 5 of
  [`v2/operating_characteristics.md`](preregistration/v2/operating_characteristics.md#5-confirmatory-cost).
  A job skeleton:

  ```bash
  #!/bin/bash
  #PBS -N mpcbench-v2
  #PBS -l select=1:node_type=mi300a
  #PBS -l walltime=24:00:00
  cd "$PBS_O_WORKDIR"                       # the clean clone of the tag
  WS=$(ws_find impact)                      # the workspace (section 3)
  source "$WS/venvs/mpcbench-v2/bin/activate"   # an environment that matches the v2 lock
  export MPCBENCH_V1_OUTPUTS="$WS/paper1_mpcbench"
  OUT="$WS/v2_confirmatory" WORKERS=96 PY=python STEPS="anchor_replication A_witnesses" \
      bash scripts/v2/mpcbench_confirmatory_v2.sh
  ```

  A mi300a node has 96 CPU cores (section 15). The clone, the venv and the outputs follow the rules above.

When the script exits with 1 it names the steps with task errors; they are run
once more with `STEPS` set to those steps and the same `OUT`, and that rerun is
recorded. The registry builder, the integrity audit and the evaluator then run
from the tag on the outputs, in that order (preregistration v2, section 12).

## 16. Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| `module: command not found` in a job log | the job did not run under bash, or no setup file was sourced | Check the first line (`#!/bin/bash`) and the `source` line of the `.pbs` file. Rebuild with `IMPACT_HUNTER_SETUP_FILE` exported. |
| `ERROR: workspace 'impact' not found (see ws_list)` | workspace expired, or another name | `ws_list`; set `IMPACT_WS_NAME` in the setup file. |
| `ERROR: venv '...' missing; run scripts/hunter/install_hunter_env.sh first` | venv not created or elsewhere | Section 5.2; or set `IMPACT_VENV`. |
| `Invalid Python runtime for run_pipeline.py ... Expected conda env: 'impact-synergy-clean'` | setup file not sourced in this shell | `source "$WS/hunter_pbs_setup.sh"` (it exports `IMPACT_SKIP_ENV_CHECK=1`). |
| `Missing IMPACT_HUNTER_SETUP_FILE: <path>` at build | the exported path does not exist | Fix the path; the build checks it. |
| Warning at build: `Hardware target 'hunter-apu' is not available on this build host` | login node has no APU | Expected. The jobs still request `hunter-apu`. |
| `Hardware target unavailable: ... requires CuPy` in a job | CuPy missing in the venv, or venv not active | Section 6.3; run the self-test on a compute node. |
| Self-test exit 1, `iim_psi_xp_parity FAIL` | device kernel disagrees with the host kernel | Do not run the campaign. Send the JSON (section 17). |
| Self-test: only `eigh FAIL` | `eigh` not reliable on ROCm | Not fatal; the pipeline uses the CPU `eigh`. Report it. |
| Build: `IMPACT_HUNTER_PHASE1_TIME='30:00:00' exceeds the PBS walltime ... limit of 24:00:00` | walltime too long (25 min on `test`) | Shorter walltime, more shards per run. |
| Build: `... needs N array subjobs, above the configured maximum array size ...` | too many shards | Fewer shards per run, or raise `IMPACT_HUNTER_MAX_ARRAY_SIZE` after asking HLRS (open question 2). |
| Build: `HunterCostError: Hunter IIM campaign refused: an estimated ... Psi evaluations exceed the ceiling ...` | IIM configuration too large (e.g. no `--iim-max-nodes`: 10 nodes) | Section 10a: measure, report, and build the configuration the author confirms. Only then, if still needed, `--hunter-allow-large` or a higher `--hunter-max-psi-evals`. |
| Build warning: `Estimated longest shard exceeds the walltime in cut-shard` | too few shards for the work per run | More shards per run (`--hunter-cut-shards-per-run`), or check the rate with the calibration smoke test (section 10a). |
| `qsub` rejects `-l ws13=True` | project on ws12, or resource not available | `export IMPACT_HUNTER_PBS_WORKSPACE_RESOURCE=ws12=True` (or `none`) and rebuild. |
| `Packed shard execution needs the node-local rank from the launcher` | a shard stage was started without PALS `mpiexec` | Use the generated scripts; do not call shard stages by hand. |
| `--hunter-shards-per-node 2 does not match the campaign packing (4 shards per node)` | shard stage called with another packing | Use the packing the campaign was built with. |
| Reducer: `Shards of ... were computed by different code versions` | checkout changed during the campaign | Restore the commit, or resubmit with `IMPACT_HUNTER_FORCE=1` in the setup file. |
| Finalize: `... does not belong to the current campaign build; run the reduce-all stage again` | campaign rebuilt after reduce | Resubmit `03_reduce_finalize.pbs`. |
| `Missing preprocessing outputs at '<out-dir>/preprocessed'` | time series not staged at `--out-dir` | Section 8. |
| Many `No events.tsv found for sub-X session=Y` warnings; RAM/SRPI undefined | BIDS events not staged, or wrong `--bids-root` | Stage the BIDS tree with `events.tsv`; rebuild. |
| Device out-of-memory in a shard | device kernel batches too large | Lower `IMPACT_IIM_XP_MAX_ELEMENTS` in the setup file (default 8388608). |
| Jobs stay queued | queue limits, `test` allows 1 job per user | `qstat -T <id>`; `qstat -f <id>` (comment field). |
| "no further jobs will be executed" / jobs held | workspace or group quota exceeded | `ws_quota`; clean up or ask HLRS. |
| A job tries to download something / network errors | compute nodes have no internet | Stage all data in advance (section 8). |
| Repeated 2FA prompts | round-robin login alias | Pin a login node in `~/.ssh/config` (section 2). |
| Build or install killed on the login node | 2 h CPU-time limit | Run it in an interactive job (`qsub -I ...`). |
| GPUs listed as 4-7 in `rocm-smi` | known HLRS issue | Check the self-test's device lines; report it (open question 9). |

## 17. What to send back to the author

Send the following as one archive (tar it on the login node, copy it with
`rsync`/`scp` from your machine):

1. `hardware_selftest*.json` (every self-test you ran) and the printed output.
2. The environment: `module list 2>&1`, `python3 -m pip freeze`,
   `python3 -c "import cupy; cupy.show_config()"` (on a compute node),
   `echo $HLRS_SOFTWARE_STACK_RELEASE_VERSION $HLRS_SOFTWARE_STACK_CPE_VERSION`.
3. The campaign's bookkeeping: `pbs/campaign_plan.json`, `pbs/submitted_jobs.txt`,
   `status.json`, `timing_summary.json`, `cost_calibration.json`, the `timing/`
   folder and all `pbs/logs/*.o*` files. Also `qstat -f` of failed jobs, if any.
   Before the production campaign: the sizing report of section 10a.
4. The results: `<out-dir>/cache/step2_df.csv`, `step2_df_mean.csv`,
   `step2_theta_stats.csv`, `hunter_iim_results.csv`, `provenance_manifest.json`,
   the `stats/` folder and the Word report.
5. Your notes: anything marked "to be verified on Hunter" that turned out to be
   different (module names, `ROCM_PATH`, quotas, array limits, queue behaviour),
   and HLRS's answers to the open questions below.

```bash
tar czf "$WS/impact_ds003171_handback.tar.gz" \
    -C "$WS/outputs" ds003171/stats ds003171/IMPaCT_Empirical_Validation_ds003171.docx \
    ds003171/cache/step2_df.csv ds003171/cache/step2_df_mean.csv ds003171/cache/step2_theta_stats.csv \
    ds003171/cache/hunter_iim_results.csv ds003171/cache/provenance_manifest.json \
    ds003171/cache/hunter_iim_campaign/pbs ds003171/cache/hunter_iim_campaign/timing \
    ds003171/cache/hunter_iim_campaign/timing_summary.json ds003171/cache/hunter_iim_campaign/status.json \
    ds003171/cache/hunter_iim_campaign/cost_calibration.json
```

Do not send raw data or the `runs/` folders unless the author asks.

## 18. Open questions for HLRS staff

These could not be answered from the HLRS documentation (read 2026-09-27).
Please ask via `rt-platform-hunter@hlrs.de` before the production campaign and
pass the answers to the author.

1. Is there no Slurm compatibility layer (`sbatch`/`srun` wrappers) on Hunter,
   and is PBS Pro the only supported scheduler? Which PBS Pro version runs? The
   KB links the 2024.1 guides.
2. Job arrays: what is the maximum array size (`max_array_size`)? Do subjobs
   count individually against the `single` queue limits (40 queued / 20 running
   per user, 60 / 30 per group)? Does `-W depend=afterok:<array>[]` behave as
   documented in the PBS 2024.1 User Guide (all subjobs must exit 0)? Is job
   history enabled for dependencies on finished jobs?
3. Policy: is a workflow of many single-node array jobs acceptable (the GCS fact
   sheet says job farming is "not an approbate way to use a large number of
   nodes"), or should tasks be packed into multi-node jobs? What share of
   CPU-only work (no GPU use) is tolerated on mi300a nodes, given that "using
   these GPU cores is mandatory"? Would Vulcan access be advisable for CPU-only
   phases?
4. Is conda, miniforge or micromamba permitted or supported on Hunter (installed
   in a workspace)? Is there a micromamba module as on Vulcan? Does conda work
   through the SSH SOCKS5 tunnel? Or should we strictly use cray-python plus
   `venv --system-site-packages`?
5. Is there an HLRS-provided CuPy for ROCm, for example under
   `/opt/hlrs/stack/ai-frameworks/modulefiles`, or a recommended recipe? Is
   building CuPy 13.6.0 from source on a login node acceptable given the 2 h CPU
   limit, or should it be built in an interactive mi300a job?
6. Inside `HLRS/APU/2026.1`: what is the exact `rocm` module name and version
   (`rocm/6.4.1`?), is it loaded by default with `HLRS/APU`, and what is
   `$ROCM_PATH`? Is `/opt/rocm` present or symlinked? Are hipblas, hipsparse,
   rocsparse, rocrand, hiprand, rocthrust, rocsolver, rocfft, hipfft, hipcub,
   rocprim, rccl and roctracer all included (CuPy build requirements)?
7. When will `HLRS/2026.2` (CPE 26.03, ROCm 7.0.2, cray-python 3.12.12) become
   the default? May production jobs use `HLRS/APU/testing-2026.2` now, for
   example with the `cupy-rocm-7-0` wheel?
8. HPE's CPE 25.09 notes say `cray-libsci_acc` "may generate wrong numerical
   results on AMD GPUs with ROCm 6.4.0 and 6.4.1" (rocBLAS). Does this affect
   rocBLAS/hipBLAS used by other software such as CuPy on Hunter? Has HLRS
   applied a workaround?
9. The Known-issues entry (2025-12-06) says GPUs may appear with IDs 4-7 in
   `rocm-smi`. Does this affect `HIP_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES`
   numbering or the PALS `--gpu-bind` lists for Python/CuPy processes?
10. The `Secure_Shell_ssh` page carries a 2025-07-28 notice that "reverse
    connections to the user network are no longer permitted", but
    `SSH_Tunnel_with_Proxy` (May 2026) still prescribes `ssh -R` SOCKS tunnels.
    Is the reverse SOCKS tunnel still allowed for pip/git on login nodes, and for
    interactive compute nodes via ProxyJump?
11. What is the recommended transfer path for about 10-100+ GB of OpenNeuro BIDS
    data into ws13: `scp` to a login node (the "10 minutes before the first job"
    page warns that scp via frontends fails due to CPU limits), UFTP
    (`gridftp-fr1:9000`, SSH key via ticket) or GridFTP? Are datalad/git-annex
    available or usable through the tunnel?
12. What are our project's ws13 capacity and file-count quotas (`ws_quota`)? Is a
    Python venv with tens of thousands of files, plus per-shard outputs,
    acceptable on Lustre, or should we use an Apptainer SIF or a conda-pack
    tarball instead?
13. Is Lmod's `module load` safe in job scripts that run `set -euo pipefail`
    (nounset)? Or should modules be loaded before `set -u` (as the generated
    scripts currently do)?
14. Does cray-python on Hunter include pandas? The KB says yes; the HPE CPE 25.09
    release notes list only numpy, scipy, mpi4py and dask. Is shadowing the
    LibSci-linked numpy/scipy in a `--system-site-packages` venv, when newer
    packages force upgrades, acceptable or discouraged?
15. Would the HLRS colleague run the jobs under their own account and our project
    group, or under ours? We need to know whether to use `-W group_list=<group>`,
    `ws_share` for read access, and who registers IPs and SSH keys.
16. Can the 20 mi300a localscratch nodes (`node_type_storage=localscratch`) be
    used to stage datasets per job, and is there any limit on requesting them?

## 19. Reference: files, variables and stages

### 19.1 Stages of `run_pipeline.py --execution-mode hunter`

| `--hunter-stage` | Runs where | What it does |
|---|---|---|
| `build-campaign` (default) | login node | estimates the IIM work (Ψ evaluations, runtime) and refuses it above the ceiling; prepares every run's IIM problem and writes the PBS scripts |
| `phase1-shard` | array job | Ψ contributions of one mechanism chunk (intact TPM) |
| `cut-shard` | array job | Ψ of a slice of system cuts (checkpoint after every cut) |
| `phase1-reduce`, `cut-reduce` | (by hand) | reduce one run (`--hunter-run-index`) |
| `phase1-reduce-all`, `cut-reduce-all` | (by hand) | reduce one stage for all runs |
| `reduce-all` | reduce job | both reductions for all runs; writes `iim_results.json/.csv` and `cost_calibration.json`; IIM null calibration |
| `finalize-pipeline` | reduce job | step 2 (RAM/PDI/NAS/SRPI, IIM from the campaign, legacy CI, MPC evidence) and steps 3-9 |
| `status` | anywhere | counts complete/missing shards; writes `status.json` and `cost_calibration.json` |

### 19.2 Environment variables

The full table of build-time and job-time variables (scheduler, queue, group,
walltimes, packing, bindings, caches, kernels) is in
[`scripts/hunter/README.md`](../scripts/hunter/README.md#configuration). The ones
you are most likely to need:

| Variable | Read at | Default | Purpose |
|---|---|---|---|
| `IMPACT_HUNTER_SETUP_FILE` | build | none | file sourced by every job (section 5.3) |
| `IMPACT_HUNTER_PYTHON` | build | `python3` | interpreter in the job scripts (exported by the setup file) |
| `IMPACT_HUNTER_PBS_GROUP_LIST` | build | primary group | `-W group_list=` for a non-default project |
| `IMPACT_HUNTER_PBS_QUEUE` | build | routing queue | e.g. `test` (all walltimes then default to 25 min) |
| `IMPACT_HUNTER_PHASE1_TIME`, `IMPACT_HUNTER_CUT_TIME`, `IMPACT_HUNTER_REDUCE_TIME` | build | 24h / 24h / 4h | walltimes (max 24 h) |
| `IMPACT_HUNTER_PBS_WORKSPACE_RESOURCE` | build | `ws13=True` | `none` disables it |
| `IMPACT_HUNTER_MAX_PSI_EVALS` | build | 1e11 | ceiling on the estimated Ψ evaluations (`--hunter-max-psi-evals`); `--hunter-allow-large` overrides (section 10a) |
| `IMPACT_HUNTER_SECONDS_PER_PSI_EVAL` | build | unset (reference rate) | measured shard wall-seconds per Ψ evaluation for the runtime estimate (`--hunter-seconds-per-psi-eval`) |
| `IMPACT_HUNTER_FORCE` | job | unset | `1` recomputes completed shards |
| `IMPACT_IIM_XP_MAX_ELEMENTS` | job | 8388608 | largest temporary array of the device kernel (float64 elements) |
| `IMPACT_EIGH_BACKEND` | job | `auto` | `cpu`/`device` forces the eigh choice |
| `IMPACT_IIM_PSI_KERNEL` | job | `auto` | `numba`/`xp` forces the Ψ kernel (debugging only) |
| `IMPACT_CODE_VERSION` | job | unset | version suffix for checkouts without `.git` |

### 19.3 Sources

HLRS knowledge base pages read on 2026-09-27: `Hunter_(HPE)` (2026-08-31),
`Batch_System_PBSPro_(Hunter)` (2026-07-01), `HLRS_Software_Stacks_for_Hunter`
(2026-08-07), `Python_Virtual_Environments_and_Packages` (2026-05-19),
`Storage_(Hunter)` (2026-04-15), `SSH_Tunnel_with_Proxy` (2026-05-22),
`Secure_Shell_ssh` (2025-08-26), `Known_issues` (2026-06-18),
`Using_container_on_hunter` (2026-08-20), `AI_Agents_on_hww_clusters`
(2026-09-01); CuPy 13.6/14.x documentation and release notes; HPE CPE 25.09 and
26.03 release notes; AMD HPCTrainingDock CuPy build script.
