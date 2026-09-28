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
   and scipy 1.10.1.
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

This builds a tiny campaign (4 nodes). It then submits
`pbs/90_smoke_all_in_one.pbs`, a single job that runs these steps in order:

- `python -m impact_pipeline.hardware_selftest`, which compares CuPy with NumPy for
  matmul, eigh, `ufunc.at`, bincount, SVD, pinv and the IIM TPM kernels;
- all shards;
- the merged reduce;
- finalize.

## Production campaign

```bash
python3 run_pipeline.py --execution-mode hunter --hunter-stage build-campaign \
  --hardware-target hunter-apu --dataset-id ds003171 --bids-root <BIDS> --out-dir <OUT> \
  --mpc-metrics RAM PDI NAS IIM SRPI --iim-max-nodes <N>
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
- Status and timing: `--hunter-stage status` writes `status.json`, which lists missing
  shard indices. Every task writes `timing/<stage>/<task>.json` (wall/CPU time, peak
  RSS, host, PBS ids). Finalize writes `timing_summary.json`.

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

## Known limitations

- The Ψ kernels of phase 1 and of the cut shards run on the CPU. Only the TPM
  construction uses CuPy. Hunter policy requires GPU use, so a GPU Ψ kernel is still
  open work.
- These points are **UNVERIFIED** on Hunter:
  - whether array subjobs count against the `single` queue limits (40 queued and 20
    running jobs per user);
  - the PALS behaviour for non-MPI Python ranks;
  - the name of the `rocm` module;
  - whether Lmod works under `set -u`.
