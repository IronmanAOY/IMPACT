# Running the IMPaCT IIM campaign on HLRS Hunter

HLRS Hunter runs **Altair PBS Pro**, not Slurm. `--execution-mode hunter` therefore
generates PBS scripts by default (`--hunter-scheduler pbs`). A generic Slurm backend
remains available with `--hunter-scheduler slurm` (or `IMPACT_HUNTER_SCHEDULER=slurm`).

Items marked **UNVERIFIED** have not been confirmed on Hunter. Each one is configurable
and should be checked with HLRS (`rt-platform-hunter@hlrs.de`) before a production run.

## One-time setup (login node, run by a person)

1. Clone the repository into a workspace. HOME must not be used for job I/O:
   `ws_allocate impact 60`, then `ws_find impact`.
2. Print the install steps with `bash scripts/hunter/install_hunter_env.sh --cupy source-13.6`.
   Run them with `--run` while the SOCKS tunnel is open. This creates a
   `cray-python` venv with `--system-site-packages` and installs
   `requirements-hunter.txt`. `constraints-hunter.txt` keeps the Cray numpy 1.24.4
   and scipy 1.10.1 of the default stack `HLRS/APU/2026.1` (ROCm 6.4.1, CuPy 13.6.0
   built from source for gfx942). The pipeline runs from the checkout
   (`run_pipeline.py` puts `src/` on `sys.path`); it is not pip-installed, because
   `pyproject.toml` asks for scipy>=1.11.
   - Alternative, testing stack `HLRS/APU/testing-2026.2` (ROCm 7.0.2, cray-python
     3.12.12, numpy 2.3.5, scipy 1.16.3), **UNVERIFIED**: change the constraints to
     numpy==2.3.5 / scipy==1.16.3, use numba==0.62.1 and `--cupy wheel-rocm7`
     (`cupy-rocm-7-0>=14.1`). mne 1.6.1 predates NumPy 2, so this route needs
     mne>=1.7, which the pipeline has not been validated with.
3. Set up the job environment:
   `cp scripts/hunter/hunter_pbs_setup.sh $WS/`, edit the copy if needed,
   `export IMPACT_HUNTER_SETUP_FILE=$WS/hunter_pbs_setup.sh`, then source it.
   Every generated job sources this file before `set -u`.
4. Stage the data with scp, UFTP or GridFTP from a machine with internet access.
   Compute nodes have no internet.

## Smoke test (PBS `test` queue: 25 min, 1 job per user)

```bash
bash scripts/hunter/hunter_smoke_test.sh --bids-root <BIDS> --out-dir <OUT> \
     --subjects "<ID>" --hardware-target hunter-apu --submit
```

This builds a tiny campaign (4 nodes; add `--null-surrogates 2` to exercise the
surrogate calibration). It then submits `pbs/90_smoke_all_in_one.pbs`, a single job
that runs these steps in order:

- `python -m impact_pipeline.hardware_selftest`, which prints the device information,
  the code version and the IIM Ψ kernel selected for the target, and compares CuPy
  with NumPy for matmul, eigh, `ufunc.at`, bincount, SVD, pinv, the IIM TPM kernels
  and `iim_psi_xp_parity`: phase-1 Ψ and the bidirectional/directional cut TPMs and
  their Ψ from the device kernel against the numba/host reference kernel;
- all shards;
- the merged reduce;
- finalize.

The self-test is the first thing to run on a node (also standalone, in an interactive
or `-q test` job): `python3 -m impact_pipeline.hardware_selftest --target hunter-apu
--json selftest.json`. Exit code 0 = all required cases passed, 1 = a comparison
failed (do not run a campaign), 2 = no accelerator visible. A failing `eigh` alone is
not fatal (the pipeline then uses the CPU eigh).

## Production campaign

```bash
python3 run_pipeline.py --execution-mode hunter --hunter-stage build-campaign \
  --hardware-target hunter-apu --dataset-id ds003171 --bids-root <BIDS> --out-dir <OUT> \
  --mpc-metrics RAM PDI NAS IIM SRPI --iim-max-nodes <N> \
  --hunter-iim-null-surrogates <K>      # optional: surrogate-null calibration (below)
bash <OUT>/cache/hunter_iim_campaign/pbs/00_submit_all.sh
```

- You can build on a login node. If no APU is visible, the problem is prepared on the
  CPU, which gives numerically identical TPMs. The jobs still request `hunter-apu`.
- DAG: `01_phase1_shards.pbs` and `02_cut_shards.pbs` run concurrently. Neither depends
  on the other. `03_reduce_finalize.pbs` then runs with
  `qsub -W depend=afterok:<p1>:<cut>` and holds the merged reducers plus finalize.
  PBS dependencies apply to whole arrays only.
- Packing: each `mi300a` node is allocated exclusively and charged in full. It runs
  `shards_per_node` shards (default 4, one per APU) through PALS:
  `mpiexec -n 4 --ppn 4 --cpu-bind list:0-23:24-47:48-71:72-95 --gpu-bind list:0:1:2:3`.
  The shard index is `4 * PBS_ARRAY_INDEX + PMI_LOCAL_RANK`.
- Arrays need at least 2 subjobs. A stage that needs only one node is written as a
  plain job with array index 0.
- Resubmission is safe: rerun `00_submit_all.sh`. Completed shards are skipped. The
  skip check uses the task identity: problem digest, shard range and code version.
  Cut shards also checkpoint after every cut. Set `IMPACT_HUNTER_FORCE=1` to recompute.
  Reducers refuse shards from another build or from mixed code versions, and finalize
  refuses final results from an earlier build of the same campaign directory.
- `qsub` does not forward the login shell's environment to the jobs (no `-V`). Export
  job-time variables (`IMPACT_HUNTER_FORCE`, `IMPACT_IIM_*`, `IMPACT_EIGH_BACKEND`,
  `IMPACT_CODE_VERSION`) in the setup file that every job sources. The `IMPACT_HUNTER_*`
  scheduler settings in the table below are read when the campaign is built.
- Status and timing: `--hunter-stage status` writes `status.json`, which lists missing
  shard indices. Every task writes `timing/<stage>/<task>.json` (wall/CPU time, peak
  RSS, host, PBS ids, backend and Ψ kernel). Finalize writes `timing_summary.json`.
- Checkout: the jobs `cd` into the checkout recorded at build time and export it as
  `IMPACT_REPO_ROOT`. It is resolved as `--repo-root` > `IMPACT_REPO_ROOT` > the
  package's own checkout > the directory of `run_pipeline.py`. A non-editable
  (site-packages) install therefore works when `--repo-root`/`IMPACT_REPO_ROOT` is set.
- Code version (shard identities, provenance): package version + commit, e.g.
  `1.1.0+g<sha>[.dirty]`; without repository metadata `1.1.0+$IMPACT_CODE_VERSION`.

## IIM on the APU (Ψ kernel)

On `--hardware-target hunter-apu` (or `gpu`) the phase-1 and cut shards compute Ψ with
the array-module kernel `impact_pipeline.iim_xp` on the device (CuPy); both PBS stages
request the accelerator target. On `cpu` they use the numba/host kernel with worker
processes and node-local SQLite kernel caches. Every shard runs on the
`--hardware-target` of its own job (the optional Slurm backend keeps phase-1 shards on
its CPU partition with `--hardware-target cpu`). The two kernels agree to ~1e-15
(tests: 1e-10). Notes:

- With the device kernel one process per APU does the work; `--hunter-workers-per-task`
  and the SQLite kernel caches are not used.
- `IMPACT_IIM_PSI_KERNEL=numba|xp` overrides the choice (debugging);
  `IMPACT_IIM_XP_MAX_ELEMENTS` bounds the largest temporary array of the device kernel
  (default 2^23 float64 = 64 MB; peak device memory is a small multiple of it) and
  `IMPACT_IIM_XP_CACHE_ELEMENTS` the cached conditional rows / purview marginalisers
  (default 2^27).
- Shard payloads, `final_result.json` and `iim_results.csv` record the kernel used
  (`psi_kernel`).

## Surrogate-null calibration of IIM

`--hunter-iim-null-surrogates K` (build-campaign) adds K surrogate runs per real run:
independent circular shifts of every node but node 0 (minimum shift max(lag+1, 10 % of
T)), drawn from one `RandomState(seed)` stream exactly as
`compute_IIM(null_surrogates=K)` draws them (seed 0 = compute_IIM's default). Each
surrogate run is prepared with the real run's bins, lag, cut sample and mechanism /
purview sizes and is sharded like any other run (cost: (K+1) times the IIM campaign).
The surrogate series are stored as `runs/<run>__nullNNN/surrogate_ts.npy`; seeds,
method, minimum shift and the surrogate run keys are recorded in the real run's
`meta.json` (`iim_null`). The reducer computes, with the same function as the local
path, `Delta_Psi`, `IIM_null_mean`, `IIM_null_sd`, `IIM_z`, `IIM_null_p`,
`IIM_excess` and `canonical_calibrated` (the calibrated canonical IIM in bits, which
then is `value`); `value` is NaN with `IIM_null_undefined_reason` when no surrogate can
be built. The Hunter results equal `compute_IIM(..., null_surrogates=K)` to 1e-9
(`tests/test_hunter_calibration.py`). Use K >= 19 for inference with `IIM_null_p`.

`reduce-all` writes `iim_results.json/.csv` (one row per real run: value, raw,
calibration fields, `tpm_estimator`, `iim_algorithm_version`, `cut_mode`, `psi_kernel`,
selected nodes, node-selection rule, bins requested/used, `budget_adjustments`,
observed states, code version); finalize copies the table to
`<out>/cache/hunter_iim_results.csv` next to the step-2 outputs. Each run's `meta.json`
records the same estimator, algorithm-version, node-selection and state-budget
fields.

## Configuration (environment variables; CLI flags take precedence)

| Variable | Default | Notes |
|---|---|---|
| `IMPACT_HUNTER_SCHEDULER` | `pbs` | or `slurm` (`--hunter-scheduler`) |
| `IMPACT_HUNTER_SETUP_FILE` | none | sourced before `set -u`. Legacy name `IMPACT_HUNTER_SLURM_SETUP_FILE` still works |
| `IMPACT_HUNTER_SETUP` | none | inline setup lines. Legacy name `IMPACT_HUNTER_SLURM_SETUP` still works |
| `IMPACT_HUNTER_PYTHON` | `python3` | interpreter (venv). `IMPACT_HUNTER_CONDA_ENV` optional (`conda run --no-capture-output`) |
| `IMPACT_HUNTER_PBS_GROUP_LIST` | primary group | `-W group_list=`. Also accepts `IMPACT_HUNTER_ACCOUNT` / `IMPACT_HUNTER_SLURM_ACCOUNT` |
| `IMPACT_HUNTER_PBS_QUEUE` | route | e.g. `test` (then all walltimes default to 25 min) |
| `IMPACT_HUNTER_PBS_NODE_TYPE` | `mi300a` | |
| `IMPACT_HUNTER_PBS_WORKSPACE_RESOURCE` | `ws13=True` | `none` disables it |
| `IMPACT_HUNTER_PBS_LOCALSCRATCH` | off | adds `node_type_storage=localscratch` (20 nodes) |
| `IMPACT_HUNTER_PBS_CPU_QUEUE`, `IMPACT_HUNTER_PBS_CPU_NODE_TYPE` | unset | send reduce/finalize to a CPU/pre queue (e.g. `pre` + `genoa3tb64c`), using `--hardware-target cpu`. Unset means `mi300a`, because academic users cannot use genoa nodes |
| `IMPACT_HUNTER_PHASE1_TIME`, `IMPACT_HUNTER_CUT_TIME`, `IMPACT_HUNTER_REDUCE_TIME` | 24h / 24h / 4h | validated against 24 h (`IMPACT_HUNTER_PBS_MAX_WALLTIME`) |
| `IMPACT_HUNTER_SHARDS_PER_NODE` | 4 | `--hunter-shards-per-node` |
| `IMPACT_HUNTER_PHASE1_SHARDS_PER_RUN`, `IMPACT_HUNTER_CUT_SHARDS_PER_RUN` | 16 / 256 | `--hunter-phase1-shards-per-run`, `--hunter-cut-shards-per-run` |
| `IMPACT_HUNTER_WORKERS_PER_TASK` | cores per rank minus 2 (22) | `--hunter-workers-per-task` |
| `IMPACT_HUNTER_MAX_ARRAY_SIZE` | 10000 (PBS default) | **UNVERIFIED** for Hunter. Exceeding it is a build-time error |
| `IMPACT_HUNTER_PBS_CPU_BIND`, `IMPACT_HUNTER_PBS_GPU_BIND` | derived | GPU binding for packings other than 4 ranks per node is **UNVERIFIED** |
| `IMPACT_IIM_CACHE_DIR` | `/localscratch/$PBS_JOBID`, else `$TMPDIR` | node-local SQLite kernel caches, removed after each task. `IMPACT_IIM_KERNEL_CACHE=0` disables them |
| `IMPACT_EIGH_BACKEND` | `auto` | the device eigh runs only after a NumPy parity check, otherwise the CPU is used. `cpu` and `device` force a choice |
| `IMPACT_IIM_PSI_KERNEL` | `auto` | `auto` = device (xp/CuPy) kernel on accelerator targets, numba on `cpu`; `numba`/`xp` force one |
| `IMPACT_IIM_XP_MAX_ELEMENTS`, `IMPACT_IIM_XP_CACHE_ELEMENTS` | 2^23 / 2^27 | batch and cache bounds of the device Ψ kernel (float64 elements) |
| `IMPACT_REPO_ROOT` | package checkout | checkout the jobs run (`--repo-root`); exported by every job |
| `IMPACT_CODE_VERSION` | unset | code version suffix when the checkout has no repository metadata |

## Known limitations

- The device Ψ kernel is validated against the host kernel on NumPy and with a
  NumPy-backed stand-in for ROCm CuPy; its numerical parity and speed on a real
  MI300A (ROCm 6.4.1 / CuPy 13.6.0, where HPE warns about rocBLAS results) are
  **UNVERIFIED** until `hardware_selftest` has been run there. Its work grows like the
  host kernel's (all mechanism/purview bipartitions); up to 6-7 nodes (2 bins) per run
  keep campaigns tractable.
- These points are **UNVERIFIED** on Hunter:
  - whether array subjobs count against the `single` queue limits (40 queued and 20
    running jobs per user);
  - the PALS behaviour for non-MPI Python ranks;
  - the name of the `rocm` module;
  - whether Lmod works under `set -u`.
