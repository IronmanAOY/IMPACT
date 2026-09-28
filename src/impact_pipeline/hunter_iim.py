from __future__ import annotations

import concurrent.futures
import contextlib
import csv
import dataclasses
import hashlib
import json
import logging
import math
import os
import shlex
import shutil
import socket
import sqlite3
import sys
import tempfile
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

from impact_pipeline.execution_profiles import (
    ExecutionProfile,
    HunterPBSProfile,
    HunterSlurmProfile,
    normalize_hunter_scheduler,
)
from impact_pipeline import iim_xp
from impact_pipeline.hardware_backend import (
    backend_summary,
    configure_process_for_hardware,
    get_array_module,
    normalize_hardware_target,
)
from impact_pipeline.mpc_metrics import (
    IIM_ALGORITHM_VERSION,
    _IIMDiskKernelCache,
    _iim_build_cut_tpm_for_mode,
    _iim_build_phase1_chunks_adaptive,
    _iim_cut_to_key,
    _iim_null_fields,
    _iim_phase1_chunk_contribution,
    _iim_phase_worker_init_static,
    _iim_phase_worker_run_chunk_for_tpm,
    _resolve_null_seed,
    iim_calibrated_value,
    iim_null_calibration_fields,
    iim_null_min_shift as _resolve_null_min_shift,
    iim_null_surrogate_series,
    prepare_iim_problem,
    resolve_iim_psi_kernel,
)
from impact_pipeline.provenance import (
    REPO_ROOT_ENV,
    collect_code_version,
    resolve_repo_root,
)
from impact_pipeline.synergy_ci import build_ci_run_specs

log = logging.getLogger(__name__)

# Package-relative checkout root (a source checkout or editable install).
# Stages resolve the root with provenance.resolve_repo_root, which also honours
# an explicit --repo-root and IMPACT_REPO_ROOT (needed for non-editable installs,
# where this path points into site-packages).
REPO_ROOT = Path(__file__).resolve().parents[2]
LOCALSCRATCH_ROOT = Path("/localscratch")
PHASE1_STAGE = "phase1-shard"
CUT_STAGE = "cut-shard"
FORCE_ENV = "IMPACT_HUNTER_FORCE"
IIM_CACHE_DIR_ENV = "IMPACT_IIM_CACHE_DIR"
KERNEL_CACHE_ENV = "IMPACT_IIM_KERNEL_CACHE"
KEEP_KERNEL_CACHE_ENV = "IMPACT_IIM_KEEP_KERNEL_CACHE"
# Environment variables that expose the node-local rank of a packed launch.
LOCAL_RANK_ENV_VARS = (
    "PMI_LOCAL_RANK",
    "PALS_LOCAL_RANKID",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "MPI_LOCALRANKID",
    "SLURM_LOCALID",
)
# New generic names -> deprecated Slurm-specific aliases (still accepted).
_ENV_ALIASES = {
    "IMPACT_HUNTER_SETUP_FILE": ("IMPACT_HUNTER_SLURM_SETUP_FILE",),
    "IMPACT_HUNTER_SETUP": ("IMPACT_HUNTER_SLURM_SETUP",),
    "IMPACT_HUNTER_ACCOUNT": ("IMPACT_HUNTER_SLURM_ACCOUNT",),
}
_SLURM_ONLY_ENV = (
    "IMPACT_HUNTER_APU_PARTITION",
    "IMPACT_HUNTER_CPU_PARTITION",
    "IMPACT_HUNTER_SLURM_QOS",
    "IMPACT_HUNTER_CPUS_PER_TASK",
    "IMPACT_HUNTER_MEM_PER_TASK",
    "IMPACT_HUNTER_GPUS_PER_TASK",
    "IMPACT_HUNTER_SLURM_ARRAY_THROTTLE",
)
_TRUE_TOKENS = {"1", "true", "yes", "on"}
_FALSE_TOKENS = {"0", "false", "no", "off", "none", ""}


def _json_dump(path: Path, payload) -> None:
    # Atomic replace: a job killed mid-write never leaves a truncated result
    # that a resubmission could mistake for a completed shard.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _json_load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sanitize_token(text: str) -> str:
    raw = str(text)
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in raw)


def _split_evenly(n_items: int, n_shards: int):
    total = int(max(0, n_items))
    if total == 0:
        return []
    shards = int(max(1, n_shards))
    shards = min(shards, max(1, total))
    out = []
    start = 0
    for shard_idx in range(shards):
        width = total // shards
        if shard_idx < (total % shards):
            width += 1
        stop = start + width
        out.append((start, stop))
        start = stop
    return out


# ---------------------------------------------------------------------------
# Scheduler / runtime settings
# ---------------------------------------------------------------------------


def _env_value(name: str, env=None) -> str | None:
    """Value of an IMPACT_HUNTER_* variable, honouring deprecated aliases."""
    env = os.environ if env is None else env
    raw = env.get(name)
    if raw is not None and str(raw).strip():
        return str(raw).strip()
    for alias in _ENV_ALIASES.get(name, ()):
        raw = env.get(alias)
        if raw is not None and str(raw).strip():
            log.info("%s is deprecated; use %s instead.", alias, name)
            return str(raw).strip()
    return None


def _env_optional_resource(name: str, default: str | None, env=None) -> str | None:
    """Like _env_value, but an explicitly empty/'none' value disables the default."""
    env = os.environ if env is None else env
    if name not in env:
        return default
    raw = str(env.get(name) or "").strip()
    return None if raw.lower() in _FALSE_TOKENS else raw


def _env_flag(name: str, default: bool = False, env=None) -> bool:
    env = os.environ if env is None else env
    raw = env.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in _TRUE_TOKENS


def parse_walltime(text) -> int:
    """Parse [[D-]HH:]MM:SS style walltimes into seconds."""
    raw = str(text).strip()
    days = 0
    if "-" in raw:
        day_txt, raw = raw.split("-", 1)
        days = int(day_txt)
    parts = [int(p) for p in raw.split(":")]
    if not parts or len(parts) > 3 or any(p < 0 for p in parts):
        raise ValueError(f"Invalid walltime {text!r}; expected HH:MM:SS.")
    while len(parts) < 3:
        parts.insert(0, 0)
    hours, minutes, seconds = parts
    return int(days) * 86400 + hours * 3600 + minutes * 60 + seconds


def format_walltime(seconds: int) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def _validate_walltime(label: str, value: str, maximum: str, context: str) -> str:
    secs = parse_walltime(value)
    if secs <= 0:
        raise ValueError(f"{label}={value!r} must be positive.")
    if secs > parse_walltime(maximum):
        raise ValueError(
            f"{label}={value!r} exceeds the {context} limit of {maximum}. "
            "Reduce the walltime (more shards per run shorten each task)."
        )
    return format_walltime(secs)


def default_cpu_bind(cores_per_node: int, shards_per_node: int) -> str | None:
    """PALS --cpu-bind list with one contiguous core block per rank."""
    spn = int(shards_per_node)
    cores = int(cores_per_node)
    if spn <= 1:
        return None
    if cores % spn != 0:
        raise ValueError(
            f"cores_per_node={cores} is not divisible by shards_per_node={spn}; "
            "set IMPACT_HUNTER_PBS_CPU_BIND."
        )
    width = cores // spn
    return "list:" + ":".join(f"{i * width}-{(i + 1) * width - 1}" for i in range(spn))


def default_gpu_bind(gpus_per_node: int, shards_per_node: int) -> str | None:
    """One APU per rank (HLRS example); other packings: IMPACT_HUNTER_PBS_GPU_BIND."""
    spn = int(shards_per_node)
    if spn <= 1 or int(gpus_per_node) != spn:
        return None
    return "list:" + ":".join(str(i) for i in range(spn))


def _python_launcher(env=None) -> list[str]:
    env = os.environ if env is None else env
    explicit = _env_value("IMPACT_HUNTER_PYTHON", env)
    if explicit:
        return shlex.split(explicit)
    conda_env = _env_value("IMPACT_HUNTER_CONDA_ENV", env) or _env_value(
        "IMPACT_CONDA_ENV", env
    )
    if conda_env:
        conda_bin = (
            _env_value("IMPACT_HUNTER_CONDA_BIN", env)
            or _env_value("CONDA_BIN", env)
            or "conda"
        )
        # --no-capture-output streams logs instead of buffering them until exit.
        return [conda_bin, "run", "--no-capture-output", "-n", conda_env, "python"]
    return ["python3"]


def _int_setting(overrides, key, env, env_name, default, minimum=1):
    value = overrides.get(key)
    if value is None:
        value = _env_value(env_name, env)
    if value is None:
        value = default
    if value is None:
        return None
    ivalue = int(value)
    if ivalue < int(minimum):
        raise ValueError(f"{key} must be >= {minimum}; got {ivalue}.")
    return ivalue


def resolve_hunter_settings(
    profile: ExecutionProfile,
    *,
    scheduler: str | None = None,
    overrides: dict | None = None,
    env=None,
):
    """
    Resolve the effective shard/worker counts and scheduler settings.

    Precedence: explicit overrides (CLI) > IMPACT_HUNTER_* environment > profile.
    Returns ``(effective_profile, settings)``; both are stored in the campaign
    manifest so every stage runs with the configuration it was built with.
    """
    env = os.environ if env is None else env
    overrides = {k: v for k, v in dict(overrides or {}).items() if v is not None}
    sched = normalize_hunter_scheduler(
        overrides.get("scheduler")
        or scheduler
        or _env_value("IMPACT_HUNTER_SCHEDULER", env)
        or profile.hunter_scheduler
    )
    phase1_shards = _int_setting(
        overrides,
        "phase1_shards_per_run",
        env,
        "IMPACT_HUNTER_PHASE1_SHARDS_PER_RUN",
        profile.hunter_phase1_shards_per_run,
    )
    cut_shards = _int_setting(
        overrides,
        "cut_shards_per_run",
        env,
        "IMPACT_HUNTER_CUT_SHARDS_PER_RUN",
        profile.hunter_cut_shards_per_run,
    )
    chunk_size = _int_setting(
        overrides,
        "chunk_size",
        env,
        "IMPACT_HUNTER_CHUNK_SIZE",
        profile.hunter_phase1_chunk_size,
    )

    settings: dict = {"scheduler": sched}
    if sched == "pbs":
        pbs = profile.hunter_pbs or HunterPBSProfile()
        spn = _int_setting(
            overrides,
            "shards_per_node",
            env,
            "IMPACT_HUNTER_SHARDS_PER_NODE",
            pbs.shards_per_node,
        )
        cores = _int_setting(
            overrides,
            "cores_per_node",
            env,
            "IMPACT_HUNTER_CORES_PER_NODE",
            pbs.cores_per_node,
        )
        gpus = _int_setting(
            overrides,
            "gpus_per_node",
            env,
            "IMPACT_HUNTER_GPUS_PER_NODE",
            pbs.gpus_per_node,
            minimum=0,
        )
        if spn > cores:
            raise ValueError(f"shards_per_node={spn} exceeds cores_per_node={cores}.")
        default_workers = max(1, cores // spn - 2)
        queue = (
            overrides.get("queue")
            or _env_value("IMPACT_HUNTER_PBS_QUEUE", env)
            or pbs.queue
        )
        max_walltime = (
            _env_value("IMPACT_HUNTER_PBS_MAX_WALLTIME", env) or pbs.max_walltime
        )
        time_limit, limit_context = max_walltime, "PBS walltime (academic users: 24 h)"
        if queue == "test":
            time_limit, limit_context = pbs.test_queue_max_walltime, "'test' queue"
        default_time = pbs.test_queue_max_walltime if queue == "test" else None
        times = {}
        for key, env_name, prof_default in (
            ("phase1_time", "IMPACT_HUNTER_PHASE1_TIME", pbs.phase1_time),
            ("cut_time", "IMPACT_HUNTER_CUT_TIME", pbs.cut_time),
            ("reduce_time", "IMPACT_HUNTER_REDUCE_TIME", pbs.reduce_time),
        ):
            raw = (
                overrides.get(key)
                or _env_value(env_name, env)
                or default_time
                or prof_default
            )
            times[key] = _validate_walltime(env_name, raw, time_limit, limit_context)
        for slurm_only in _SLURM_ONLY_ENV:
            if _env_value(slurm_only, env):
                log.warning(
                    "%s has no PBS equivalent on Hunter (whole nodes are "
                    "allocated) and is ignored.",
                    slurm_only,
                )
        settings.update(
            {
                "node_type": _env_value("IMPACT_HUNTER_PBS_NODE_TYPE", env)
                or pbs.node_type,
                "queue": queue,
                "group_list": (
                    overrides.get("group_list")
                    or _env_value("IMPACT_HUNTER_PBS_GROUP_LIST", env)
                    or _env_value("IMPACT_HUNTER_ACCOUNT", env)
                    or pbs.group_list
                ),
                "workspace_resource": _env_optional_resource(
                    "IMPACT_HUNTER_PBS_WORKSPACE_RESOURCE", pbs.workspace_resource, env
                ),
                "localscratch": _env_flag(
                    "IMPACT_HUNTER_PBS_LOCALSCRATCH", pbs.localscratch, env
                ),
                "cpu_queue": _env_value("IMPACT_HUNTER_PBS_CPU_QUEUE", env)
                or pbs.cpu_queue,
                "cpu_node_type": _env_value("IMPACT_HUNTER_PBS_CPU_NODE_TYPE", env)
                or pbs.cpu_node_type,
                "smoke_queue": _env_optional_resource(
                    "IMPACT_HUNTER_PBS_SMOKE_QUEUE", "test", env
                ),
                "max_walltime": max_walltime,
                "shards_per_node": int(spn),
                "cores_per_node": int(cores),
                "gpus_per_node": int(gpus),
                "cpu_bind": _env_value("IMPACT_HUNTER_PBS_CPU_BIND", env)
                or pbs.cpu_bind
                or default_cpu_bind(cores, spn),
                "gpu_bind": _env_value("IMPACT_HUNTER_PBS_GPU_BIND", env)
                or pbs.gpu_bind
                or default_gpu_bind(gpus, spn),
                "launcher": _env_value("IMPACT_HUNTER_PBS_LAUNCHER", env)
                or pbs.launcher,
                "max_array_size": _int_setting(
                    overrides,
                    "max_array_size",
                    env,
                    "IMPACT_HUNTER_MAX_ARRAY_SIZE",
                    pbs.max_array_size,
                ),
                **times,
            }
        )
    else:
        slurm = dataclasses.asdict(profile.hunter_slurm or HunterSlurmProfile())
        env_overrides = {
            "IMPACT_HUNTER_APU_PARTITION": "apu_partition",
            "IMPACT_HUNTER_CPU_PARTITION": "cpu_partition",
            "IMPACT_HUNTER_ACCOUNT": "account",
            "IMPACT_HUNTER_SLURM_QOS": "qos",
            "IMPACT_HUNTER_PHASE1_TIME": "phase1_time",
            "IMPACT_HUNTER_CUT_TIME": "cut_time",
            "IMPACT_HUNTER_REDUCE_TIME": "reduce_time",
            "IMPACT_HUNTER_CPUS_PER_TASK": "cpus_per_task",
            "IMPACT_HUNTER_MEM_PER_TASK": "mem_per_task",
            "IMPACT_HUNTER_GPUS_PER_TASK": "gpus_per_task",
            "IMPACT_HUNTER_SLURM_ARRAY_THROTTLE": "array_throttle",
            "IMPACT_HUNTER_MAX_ARRAY_SIZE": "max_array_size",
        }
        for env_name, key in env_overrides.items():
            value = _env_value(env_name, env)
            if value is not None:
                slurm[key] = value
        for key in ("phase1_time", "cut_time", "reduce_time"):
            # Passed to --time verbatim: Slurm reads a bare number as minutes and
            # accepts D-HH[:MM[:SS]], which the PBS walltime parser does not.
            slurm[key] = str(slurm[key]).strip()
        slurm["cpus_per_task"] = int(slurm["cpus_per_task"])
        slurm["gpus_per_task"] = int(slurm.get("gpus_per_task") or 0)
        slurm["max_array_size"] = int(
            overrides.get("max_array_size") or slurm["max_array_size"]
        )
        if slurm.get("array_throttle") not in (None, ""):
            slurm["array_throttle"] = int(slurm["array_throttle"])
        default_workers = int(slurm["cpus_per_task"])
        settings.update(slurm)
        settings["shards_per_node"] = 1

    workers = _int_setting(
        overrides,
        "workers_per_task",
        env,
        "IMPACT_HUNTER_WORKERS_PER_TASK",
        profile.hunter_phase1_workers_per_task,
    )
    if workers is None:
        workers = default_workers

    setup_file = _env_value("IMPACT_HUNTER_SETUP_FILE", env)
    if setup_file:
        setup_path = Path(setup_file).expanduser().resolve()
        if not setup_path.exists():
            raise FileNotFoundError(f"Missing IMPACT_HUNTER_SETUP_FILE: {setup_path}")
        setup_file = str(setup_path)
    settings.update(
        {
            "python_launcher": _python_launcher(env),
            "setup_file": setup_file,
            "inline_setup": _env_value("IMPACT_HUNTER_SETUP", env),
            "repo_root": str(resolve_repo_root(overrides.get("repo_root"), env=env)),
            "shard_omp_threads": _int_setting(
                {}, "shard_omp_threads", env, "IMPACT_HUNTER_SHARD_OMP_THREADS", 1
            ),
        }
    )
    effective = dataclasses.replace(
        profile,
        hunter_phase1_shards_per_run=int(phase1_shards),
        hunter_cut_shards_per_run=int(cut_shards),
        hunter_phase1_workers_per_task=int(workers),
        hunter_phase1_chunk_size=int(chunk_size),
        hunter_scheduler=sched,
    )
    return effective, settings


# ---------------------------------------------------------------------------
# Packing (several shards per exclusively allocated node)
# ---------------------------------------------------------------------------


def packed_array_size(n_tasks: int, shards_per_node: int) -> int:
    """Number of array subjobs (nodes) needed for ``n_tasks`` packed shards."""
    n = int(max(0, n_tasks))
    spn = int(max(1, shards_per_node))
    return (n + spn - 1) // spn


def resolve_local_rank(env=None) -> int | None:
    env = os.environ if env is None else env
    for name in LOCAL_RANK_ENV_VARS:
        raw = env.get(name)
        if raw is not None and str(raw).strip() != "":
            return int(str(raw).strip())
    return None


def resolve_packed_task_index(array_index: int, shards_per_node: int, env=None) -> int:
    """
    Shard index handled by this process: ``shards_per_node * array_index +
    local_rank`` where the local rank comes from the launcher (PALS exports
    PMI_LOCAL_RANK).
    """
    spn = int(shards_per_node)
    if spn < 1:
        raise ValueError("shards_per_node must be >= 1")
    if int(array_index) < 0:
        raise ValueError("array index must be >= 0")
    local = resolve_local_rank(env)
    if local is None:
        if spn > 1:
            raise RuntimeError(
                "Packed shard execution needs the node-local rank from the launcher "
                f"({', '.join(LOCAL_RANK_ENV_VARS)}); "
                f"launch with PALS mpiexec -n {spn} --ppn {spn}."
            )
        local = 0
    if not 0 <= int(local) < spn:
        raise RuntimeError(
            f"Local rank {local} is outside 0..{spn - 1} for shards_per_node={spn}."
        )
    return int(array_index) * spn + int(local)


# ---------------------------------------------------------------------------
# Node-local kernel caches, timing and identities
# ---------------------------------------------------------------------------


def resolve_iim_cache_dir(env=None) -> Path:
    """
    Directory for per-task SQLite kernel caches. Never the shared Lustre
    campaign directory (SQLite WAL is unsafe on parallel file systems):
    IMPACT_IIM_CACHE_DIR, else /localscratch/$PBS_JOBID on localscratch nodes,
    else $TMPDIR (a RAM disk on Hunter), else the system temp dir.
    """
    env = os.environ if env is None else env
    explicit = str(env.get(IIM_CACHE_DIR_ENV) or "").strip()
    if explicit:
        return Path(explicit).expanduser()
    job_id = str(env.get("PBS_JOBID") or "").strip()
    if job_id:
        local = LOCALSCRATCH_ROOT / job_id
        if local.is_dir():
            return local
    tmp = str(env.get("TMPDIR") or "").strip()
    if tmp:
        return Path(tmp)
    return Path(tempfile.gettempdir())


@contextlib.contextmanager
def _kernel_cache_scope(campaign_dir: Path, label: str):
    if not _env_flag(KERNEL_CACHE_ENV, True):
        yield None
        return
    campaign_tag = hashlib.sha1(str(campaign_dir).encode("utf-8")).hexdigest()[:10]
    scope = (
        resolve_iim_cache_dir()
        / "impact_iim_kernels"
        / f"{campaign_tag}_{_sanitize_token(label)}_{os.getpid()}"
    )
    scope.mkdir(parents=True, exist_ok=True)
    try:
        yield scope
    finally:
        # Kernel caches are only valid within one Psi evaluation of one task;
        # nothing reuses them, so they are always removed.
        if not _env_flag(KEEP_KERNEL_CACHE_ENV, False):
            shutil.rmtree(scope, ignore_errors=True)


def _peak_rss_mb() -> float | None:
    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX
        return None
    usage = max(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    )
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return float(usage) / divisor


def _children_cpu_seconds() -> float | None:
    """CPU time of terminated, reaped child processes (the Psi worker pools)."""
    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX
        return None
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(usage.ru_utime + usage.ru_stime)


def _timing_start() -> dict:
    return {
        "started_unix": float(time.time()),
        "_t0": time.perf_counter(),
        "_cpu0": time.process_time(),
        "_child_cpu0": _children_cpu_seconds(),
    }


def _timing_finish(start: dict, **extra) -> dict:
    env_keys = (
        "PBS_JOBID",
        "PBS_ARRAY_INDEX",
        "PMI_RANK",
        "PMI_LOCAL_RANK",
        "SLURM_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
    )
    process_cpu = float(time.process_time() - start["_cpu0"])
    child_cpu0, child_cpu1 = start.get("_child_cpu0"), _children_cpu_seconds()
    # Worker pools do the Psi work in child processes: their CPU time is only
    # visible through RUSAGE_CHILDREN (pools are joined before this point).
    children_cpu = (
        None
        if child_cpu0 is None or child_cpu1 is None
        else max(0.0, float(child_cpu1 - child_cpu0))
    )
    out = {
        "started_unix": start["started_unix"],
        "finished_unix": float(time.time()),
        "wall_seconds": float(time.perf_counter() - start["_t0"]),
        "process_cpu_seconds": process_cpu,
        "children_cpu_seconds": children_cpu,
        "cpu_seconds": process_cpu + (children_cpu or 0.0),
        "peak_rss_mb": _peak_rss_mb(),
        "host": socket.gethostname(),
        "pid": int(os.getpid()),
        "scheduler_env": {k: os.environ[k] for k in env_keys if k in os.environ},
    }
    out.update(extra)
    return out


def _write_timing(campaign_dir: Path, stage: str, name: str, timing: dict) -> None:
    _json_dump(
        Path(campaign_dir)
        / "timing"
        / _sanitize_token(stage)
        / f"{_sanitize_token(name)}.json",
        timing,
    )


def _code_root() -> Path:
    """Checkout used for the code version (IMPACT_REPO_ROOT is exported by the jobs)."""
    root = resolve_repo_root(required=False)
    return REPO_ROOT if root is None else root


def _runtime_code_version() -> str:
    """Package version + git SHA (or IMPACT_CODE_VERSION) of the running code."""
    return str(collect_code_version(_code_root())["code_version"])


def _problem_digest(prep) -> str:
    h = hashlib.sha1()
    for key in ("curr_obs", "tpm_full", "states_full"):
        arr = np.ascontiguousarray(prep[key])
        h.update(key.encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(str(arr.shape).encode("utf-8"))
        h.update(arr.tobytes())
    h.update(json.dumps([list(x) for x in prep["mechanisms_all"]]).encode("utf-8"))
    h.update(json.dumps([list(x) for x in prep["purviews_all"]]).encode("utf-8"))
    h.update(
        json.dumps([[list(a), list(b)] for a, b in prep["cuts_eval"]]).encode("utf-8")
    )
    return h.hexdigest()


def _task_identity(meta: dict, task: dict, stage: str, code_version: str) -> dict:
    return {
        "stage": str(stage),
        "run_key": str(task["run_key"]),
        "task_index": int(task["task_index"]),
        "start": int(task["start"]),
        "stop": int(task["stop"]),
        "problem_digest": meta.get("problem_digest"),
        "code_version": str(code_version),
    }


def _stored_completion(path: Path, identity: dict):
    """The stored payload if it is a completed result for ``identity``, else None."""
    if not path.exists():
        return None
    try:
        rec = _json_load(path)
    except Exception:
        return None
    if rec.get("status") != "complete" or rec.get("identity") != identity:
        return None
    return rec


def _completed_payload(path: Path, identity: dict):
    """Return the stored payload when it is a completed result for ``identity``."""
    if _env_flag(FORCE_ENV, False):
        return None
    return _stored_completion(path, identity)


# Results derived from shards by the reducers; stale copies from an earlier
# build of the same campaign directory must never be reused.
_DERIVED_RESULT_FILES = ("phase1_result.json", "final_result.json")


def _check_single_code_version(versions, run_key: str) -> str | None:
    """All shards of a run must come from the same code version."""
    distinct = sorted({str(v) for v in versions if v is not None})
    if len(distinct) > 1:
        raise RuntimeError(
            f"Shards of {run_key} were computed by different code versions "
            f"({', '.join(distinct)}). Resubmit the campaign so every shard is "
            "recomputed with the current code (or set IMPACT_HUNTER_FORCE=1)."
        )
    return distinct[0] if distinct else None


def _mk_readonly_array_spec(label, arr, use_shared_memory, tmp_dir, owner_shms, owner_files):
    arr_c = np.ascontiguousarray(arr)
    if bool(use_shared_memory):
        try:
            shm = shared_memory.SharedMemory(create=True, size=int(arr_c.nbytes))
            shm_arr = np.ndarray(arr_c.shape, dtype=arr_c.dtype, buffer=shm.buf)
            shm_arr[...] = arr_c
            owner_shms.append(shm)
            return {
                "mode": "shared_memory",
                "name": str(shm.name),
                "shape": list(arr_c.shape),
                "dtype": str(arr_c.dtype),
            }
        except Exception:
            pass
    path = Path(tmp_dir) / f"{label}.npy"
    np.save(path, arr_c, allow_pickle=False)
    owner_files.append(path)
    return {"mode": "memmap", "path": str(path)}


def _cleanup_specs(owner_shms, owner_files, tmp_dir):
    for shm in owner_shms:
        try:
            shm.close()
        except Exception:
            pass
        try:
            shm.unlink()
        except Exception:
            pass
    for path in owner_files:
        try:
            Path(path).unlink()
        except Exception:
            pass
    if tmp_dir and os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _compute_psi_for_problem(
    *,
    tpm,
    curr_obs,
    states_full,
    base,
    mechanisms,
    purviews,
    cut_mask_a=None,
    phase1_parallel_workers=None,
    phase1_chunk_size=8,
    phase1_shared_memory=True,
    kernel_cache_path=None,
    kernel_cache_memory_entries=300_000,
    kernel_cache_flush_batch=5_000,
    static_cache=None,
    obs_state_cache=None,
):
    mechanisms = tuple(tuple(m) for m in mechanisms)
    purviews = tuple(tuple(z) for z in purviews)
    if not mechanisms or not purviews:
        return 0.0

    workers_eff = 1
    if phase1_parallel_workers is not None:
        workers_eff = max(1, int(phase1_parallel_workers))
    chunk_size_eff = max(1, int(phase1_chunk_size))
    chunks = _iim_build_phase1_chunks_adaptive(
        mechanisms,
        max_chunk_size=chunk_size_eff,
        workers=workers_eff,
    )

    cache = None
    cache_spec = None
    if kernel_cache_path:
        cache_spec = {
            "enabled": True,
            "path": str(kernel_cache_path),
            "memory_entries": int(kernel_cache_memory_entries),
            "flush_batch": int(kernel_cache_flush_batch),
        }
    if workers_eff <= 1 or len(chunks) <= 1:
        if cache_spec is not None:
            cache = _IIMDiskKernelCache(
                str(kernel_cache_path),
                signature=None,
                memory_entries=int(kernel_cache_memory_entries),
                flush_batch=int(kernel_cache_flush_batch),
            )
        try:
            psi = 0.0
            cache_enabled = cache is not None
            for chunk in chunks:
                psi = float(
                    math.fsum(
                        (
                            psi,
                            float(
                                _iim_phase1_chunk_contribution(
                                    chunk,
                                    purviews,
                                    int(base),
                                    tpm,
                                    curr_obs,
                                    states_full,
                                    static_cache=static_cache,
                                    obs_state_cache=obs_state_cache,
                                    kernel_cache=cache,
                                    cut_mask_a=cut_mask_a,
                                    use_induced_partition_cache=bool(cache_enabled),
                                    kernel_cache_lookup_only=False,
                                )
                            ),
                        )
                    )
                )
            return float(psi)
        finally:
            if cache is not None:
                cache.close()

    if cache_spec is not None:
        # All workers open this one SQLite file. Creating it (WAL switch and
        # schema) concurrently fails with "database is locked" (no busy wait
        # for the journal-mode change), so the parent creates it first.
        _IIMDiskKernelCache(
            str(kernel_cache_path),
            signature=None,
            memory_entries=int(kernel_cache_memory_entries),
            flush_batch=int(kernel_cache_flush_batch),
        ).close()

    owner_shms = []
    owner_files = []
    tmp_dir = tempfile.mkdtemp(prefix="hunter_iim_psi_")
    try:
        spec_curr = _mk_readonly_array_spec(
            "curr_obs",
            curr_obs,
            phase1_shared_memory,
            tmp_dir,
            owner_shms,
            owner_files,
        )
        spec_states = _mk_readonly_array_spec(
            "states_full",
            states_full,
            phase1_shared_memory,
            tmp_dir,
            owner_shms,
            owner_files,
        )
        spec_tpm = _mk_readonly_array_spec(
            "tpm",
            tpm,
            phase1_shared_memory,
            tmp_dir,
            owner_shms,
            owner_files,
        )

        def _run_pool(spec):
            # Every chunk is (re)computed into a fresh list: a retry never
            # double counts chunks of an aborted attempt.
            psi_terms = []
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=int(workers_eff),
                initializer=_iim_phase_worker_init_static,
                initargs=(spec_curr, spec_states, int(base), purviews),
            ) as ex:
                futures = [
                    ex.submit(
                        _iim_phase_worker_run_chunk_for_tpm,
                        spec_tpm,
                        chunk,
                        spec,
                        (None if cut_mask_a is None else int(cut_mask_a)),
                        bool(spec is not None),
                        False,
                    )
                    for chunk in chunks
                ]
                for fut in concurrent.futures.as_completed(futures):
                    psi_chunk, _chunk_len = fut.result()
                    psi_terms.append(float(psi_chunk))
            return float(math.fsum(psi_terms)) if psi_terms else 0.0

        try:
            return _run_pool(cache_spec)
        except sqlite3.Error as exc:  # locked, read-only, corrupt cache file
            if cache_spec is None:
                raise
            # The cache only memoises kernel values; without it the result is
            # identical, so keep the parallel run instead of failing the shard.
            log.warning(
                "IIM kernel cache %s failed in a worker (%s); recomputing this "
                "Psi in parallel without the cache.",
                kernel_cache_path,
                exc,
            )
            return _run_pool(None)
    finally:
        _cleanup_specs(owner_shms, owner_files, tmp_dir)


def _run_artifact_dir(campaign_dir: Path, run_key: str) -> Path:
    return campaign_dir / "runs" / str(run_key)


def _load_problem(run_dir: Path):
    meta = _json_load(run_dir / "meta.json")
    if not bool(meta.get("defined", False)):
        return meta, None
    problem = {
        "meta": meta,
        "curr_obs": np.load(run_dir / "curr_obs.npy"),
        "tpm_full": np.load(run_dir / "tpm_full.npy"),
        "states_full": np.load(run_dir / "states_full.npy"),
        "mechanisms_all": [tuple(x) for x in _json_load(run_dir / "mechanisms.json")],
        "purviews_all": [tuple(x) for x in _json_load(run_dir / "purviews.json")],
        "cuts_eval": [
            (tuple(item[0]), tuple(item[1]))
            for item in _json_load(run_dir / "cuts.json")
        ],
    }
    return meta, problem


def _prep_record(prep) -> dict:
    """Estimator, subsystem and budget provenance of a prepared IIM problem."""
    return {
        "iim_algorithm_version": IIM_ALGORITHM_VERSION,
        "tpm_estimator": prep.get("tpm_estimator"),
        "tpm_alpha": prep.get("tpm_alpha"),
        "max_state_space": prep.get("max_state_space"),
        "cut_mode": prep.get("cut_mode", "bidirectional"),
        "bearer_nodes": prep.get("bearer_nodes"),
        "selected_nodes": (
            None
            if prep.get("selected_nodes") is None
            else [int(x) for x in prep["selected_nodes"]]
        ),
        "node_selection_rule": prep.get("node_selection_rule"),
        "node_selection_degenerate": bool(prep.get("node_selection_degenerate", False)),
        "bins_requested": prep.get("bins_requested"),
        "state_budget_policy": prep.get("state_budget_policy"),
        "budget_adjustments": list(prep.get("budget_adjustments") or []),
        "n_states": prep.get("n_states"),
        "n_states_observed": prep.get("n_states_observed"),
        "n_transitions": prep.get("n_transitions"),
    }


def _undefined_prep_record(
    prep, *, tpm_estimator, cut_mode, state_budget_policy, bearer_nodes
) -> dict:
    """
    Provenance of an undefined problem. prepare_iim_problem can stop before it
    resolves the estimator settings, so the requested ones are recorded (as in
    compute_IIM's undefined payload); fields the preparation never reached are
    left out rather than filled with defaults.
    """
    rec = {
        key: val
        for key, val in _prep_record(prep).items()
        if key == "iim_algorithm_version" or key in prep
    }
    bearer = prep.get("bearer_nodes", bearer_nodes)
    rec.update(
        {
            "tpm_estimator": str(tpm_estimator),
            "cut_mode": str(cut_mode),
            "bearer_nodes": (
                None if bearer is None else sorted(int(x) for x in bearer)
            ),
            "state_budget_policy": str(
                prep.get("state_budget_policy") or state_budget_policy
            ),
            "budget_adjustments": list(prep.get("budget_adjustments") or []),
        }
    )
    return rec


def _write_undefined_run(run_dir: Path, meta: dict, prep, profile) -> None:
    _json_dump(run_dir / "meta.json", meta)
    final_payload = {
        "value": None,
        "raw": None,
        "canonical": None,
        "clipped": None,
        "iim_plus": None,
        "defined": False,
        "undefined_reason": str(prep.get("undefined_reason")),
        "n_nodes_used": prep.get("n_nodes_used"),
        "bins_used": prep.get("bins_used"),
        "n_cuts_evaluated": 0,
        "mip_cut": None,
        "phase1_parallel_workers": profile.hunter_phase1_workers_per_task,
        "phase1_chunk_size": int(profile.hunter_phase1_chunk_size),
        "phase1_shared_memory": bool(profile.hunter_shared_memory),
        "iim_algorithm_version": IIM_ALGORITHM_VERSION,
        "budget_adjustments": list(prep.get("budget_adjustments") or []),
    }
    _json_dump(run_dir / "final_result.json", final_payload)


def _register_defined_run(run_dir: Path, meta: dict, prep, profile, iim_bins) -> dict:
    """Store a prepared problem and its shard plan; returns the updated meta."""
    np.save(run_dir / "curr_obs.npy", prep["curr_obs"], allow_pickle=False)
    np.save(run_dir / "tpm_full.npy", prep["tpm_full"], allow_pickle=False)
    np.save(run_dir / "states_full.npy", prep["states_full"], allow_pickle=False)
    _json_dump(run_dir / "mechanisms.json", [list(x) for x in prep["mechanisms_all"]])
    _json_dump(run_dir / "purviews.json", [list(x) for x in prep["purviews_all"]])
    _json_dump(
        run_dir / "cuts.json",
        [[list(A), list(B)] for A, B in prep["cuts_eval"]],
    )
    phase1_ranges = _split_evenly(
        len(prep["mechanisms_all"]),
        int(profile.hunter_phase1_shards_per_run),
    )
    cut_ranges = _split_evenly(
        len(prep["cuts_eval"]),
        int(profile.hunter_cut_shards_per_run),
    )
    bins_used = int(prep["bins_used"])
    bins_reason = None
    if bins_used != int(iim_bins):
        bins_reason = (
            f"requested bins={int(iim_bins)} exceed max_state_space="
            f"{int(prep.get('max_state_space', 0))} (bins are reduced before nodes)"
        )
    meta.update(
        {
            "defined": True,
            "n_regions_input": int(prep["n_regions_input"]),
            "n_time_input": int(prep["n_time_input"]),
            "n_nodes_used": int(prep["n_nodes_used"]),
            "bins_used": bins_used,
            "bins_reduction_reason": bins_reason,
            "lag_trs": int(prep["lag_trs"]),
            "max_mechanism_size_used": int(prep["max_mechanism_size_used"]),
            "max_purview_size_used": int(prep["max_purview_size_used"]),
            "n_cuts_evaluated": int(len(prep["cuts_eval"])),
            "n_mechanisms": int(len(prep["mechanisms_all"])),
            "n_purviews": int(len(prep["purviews_all"])),
            "problem_digest": _problem_digest(prep),
            "phase1_shards": [
                {"task_index": int(i), "start": int(a), "stop": int(b)}
                for i, (a, b) in enumerate(phase1_ranges)
            ],
            "cut_shards": [
                {"task_index": int(i), "start": int(a), "stop": int(b)}
                for i, (a, b) in enumerate(cut_ranges)
            ],
            **_prep_record(prep),
        }
    )
    _json_dump(run_dir / "meta.json", meta)
    return meta


def prepare_hunter_campaign(
    *,
    data_dir,
    atlas,
    sessions,
    condition,
    stimulus_onsets,
    subjects,
    campaign_dir,
    execution_profile: ExecutionProfile,
    iim_bins,
    iim_lag_trs,
    iim_n_parts,
    iim_max_timepoints,
    iim_max_nodes,
    iim_max_mechanism_size,
    iim_max_purview_size,
    step2_context,
    hardware_target="cpu",
    scheduler=None,
    settings_overrides=None,
    build_hardware_backend=None,
    iim_tpm_estimator="node_shrinkage",
    iim_node_selection="variance",
    iim_state_budget_policy="reduce_bins_first",
    iim_cut_mode="bidirectional",
    iim_bearer_nodes=None,
    iim_null_surrogates=0,
    iim_null_method="circular_shift",
    iim_null_seed=None,
    iim_null_min_shift=None,
    iim_psi_kernel="auto",
    repo_root=None,
):
    """
    Build a Hunter IIM campaign: one prepared IIM problem per unique run plus
    phase-1 (mechanism) and cut shards, and the scheduler scripts.

    ``hardware_target`` is the target the compute jobs will request;
    ``build_hardware_backend`` (optional) is the backend used for the
    preparation step itself, so a campaign can be built on a login node
    without an accelerator. Scheduler, shard and worker settings are resolved
    by :func:`resolve_hunter_settings` and frozen in the manifest.

    Estimator settings (``iim_tpm_estimator``, ``iim_node_selection``,
    ``iim_state_budget_policy``, ``iim_cut_mode``, ``iim_bearer_nodes``) are
    passed to ``prepare_iim_problem`` exactly as compute_IIM would, and each
    run's meta.json records the estimator, algorithm version, node selection
    and every state-budget adjustment.

    Surrogate calibration: with ``iim_null_surrogates`` = K > 0 every real run
    gets K extra campaign runs holding its null surrogates, drawn exactly as
    compute_IIM draws them (``iim_null_surrogate_series``: method
    ``iim_null_method``, seed ``iim_null_seed`` (default 0, compute_IIM's
    default ``rng``), minimum shift ``iim_null_min_shift``), prepared with the
    same bins, lag, cut sample and mechanism/purview sizes. The reducer then
    computes IIM_null_mean/IIM_null_sd/IIM_z and the calibrated canonical value
    with the same function as compute_IIM(null_surrogates=K).
    """
    campaign_dir = Path(campaign_dir).resolve()
    requested_target = normalize_hardware_target(hardware_target)
    if build_hardware_backend is None:
        hardware_backend = configure_process_for_hardware(requested_target)
    else:
        hardware_backend = configure_process_for_hardware(build_hardware_backend)
    if int(iim_null_surrogates) < 0:
        raise ValueError("iim_null_surrogates must be >= 0")
    resolve_iim_psi_kernel(iim_psi_kernel, "cpu")  # validates the name at build time
    overrides = dict(settings_overrides or {})
    if repo_root is not None:
        overrides["repo_root"] = str(repo_root)
    effective_profile, scheduler_settings = resolve_hunter_settings(
        execution_profile,
        scheduler=scheduler,
        overrides=overrides,
    )
    code_root = Path(scheduler_settings["repo_root"])
    code_version = collect_code_version(code_root)
    kernel_code_version = _runtime_code_version()
    run_specs = build_ci_run_specs(
        str(data_dir),
        atlas,
        tuple(sessions),
        str(condition),
        stimulus_onsets=stimulus_onsets,
        subjects=subjects,
    )
    unique_specs = []
    seen = set()
    for spec in run_specs:
        ts_path = str(Path(spec["ts_path"]).resolve())
        if ts_path in seen:
            continue
        seen.add(ts_path)
        unique_specs.append(
            {
                "subject": str(spec["subject"]),
                "session": str(spec["session"]),
                "ts_path": ts_path,
                "ts_path_input": str(spec["ts_path"]),
            }
        )

    iim_settings = {
        "tpm_estimator": str(iim_tpm_estimator),
        "node_selection": str(iim_node_selection),
        "state_budget_policy": str(iim_state_budget_policy),
        "cut_mode": str(iim_cut_mode),
        "bearer_nodes": (
            None
            if iim_bearer_nodes is None
            else [int(x) for x in np.asarray(iim_bearer_nodes).reshape(-1)]
        ),
        "psi_kernel": str(iim_psi_kernel),
        "null_surrogates": int(iim_null_surrogates),
        "null_method": str(iim_null_method),
        "null_seed": _resolve_null_seed(iim_null_seed, 0),
        "null_min_shift": (
            None if iim_null_min_shift is None else int(iim_null_min_shift)
        ),
    }
    runs = []
    phase1_tasks = []
    cut_tasks = []

    def _add_tasks(meta):
        for shard in meta["phase1_shards"]:
            phase1_tasks.append({"run_key": str(meta["run_key"]), **shard})
        for shard in meta["cut_shards"]:
            cut_tasks.append({"run_key": str(meta["run_key"]), **shard})

    for spec_index, spec in enumerate(unique_specs):
        ts_path = Path(spec["ts_path"])
        run_index = len(runs)
        run_digest = hashlib.sha1(str(ts_path).encode("utf-8")).hexdigest()[:12]
        run_key = (
            f"run-{spec_index:04d}_{_sanitize_token(spec['subject'])}_"
            f"{_sanitize_token(spec['session'])}_{run_digest}"
        )
        run_dir = _run_artifact_dir(campaign_dir, run_key)
        run_dir.mkdir(parents=True, exist_ok=True)
        # A rebuild into an existing campaign directory invalidates the reduced
        # results (shard files are kept: their identity decides reuse).
        for name in _DERIVED_RESULT_FILES:
            with contextlib.suppress(FileNotFoundError):
                (run_dir / name).unlink()

        ts_time_region = np.load(ts_path)
        ts_iim = np.asarray(ts_time_region.T, dtype=float)
        if iim_max_timepoints is not None and int(iim_max_timepoints) > 0 and ts_iim.shape[1] > int(iim_max_timepoints):
            step = int(np.ceil(ts_iim.shape[1] / float(int(iim_max_timepoints))))
            ts_iim = ts_iim[:, ::step]

        prep = prepare_iim_problem(
            ts_iim,
            bins=int(iim_bins),
            lag_trs=int(iim_lag_trs),
            n_parts=iim_n_parts,
            rng=0,
            partition_mode="all",
            max_nodes=iim_max_nodes,
            max_mechanism_size=iim_max_mechanism_size,
            max_purview_size=iim_max_purview_size,
            hardware_backend=hardware_backend,
            tpm_estimator=str(iim_tpm_estimator),
            node_selection=str(iim_node_selection),
            state_budget_policy=str(iim_state_budget_policy),
            bearer_nodes=iim_bearer_nodes,
            cut_mode=str(iim_cut_mode),
            log_label=run_key,
        )

        meta = {
            "run_index": int(run_index),
            "run_key": str(run_key),
            "subject": str(spec["subject"]),
            "session": str(spec["session"]),
            "ts_path": str(ts_path),
            "ts_path_input": str(spec["ts_path_input"]),
            "dataset_id": step2_context.get("dataset_id"),
            "data_origin": step2_context.get("data_origin"),
            "dataset_role": step2_context.get("dataset_role"),
            "provenance_label": step2_context.get("provenance_label"),
            "defined": bool(prep.get("defined", False)),
            "undefined_reason": prep.get("undefined_reason"),
            "created_unix": float(time.time()),
            "iim_bins": int(iim_bins),
            "iim_lag_trs": int(iim_lag_trs),
            "iim_n_parts": (None if iim_n_parts is None else int(iim_n_parts)),
            "iim_max_timepoints": (
                None if iim_max_timepoints is None else int(iim_max_timepoints)
            ),
            "iim_max_nodes": (None if iim_max_nodes is None else int(iim_max_nodes)),
            "iim_max_mechanism_size": (
                None if iim_max_mechanism_size is None else int(iim_max_mechanism_size)
            ),
            "iim_max_purview_size": (
                None if iim_max_purview_size is None else int(iim_max_purview_size)
            ),
            "iim_settings": dict(iim_settings),
            "is_null_surrogate": False,
            "iim_null": None,
        }
        if not bool(prep.get("defined", False)):
            meta.update(
                _undefined_prep_record(
                    prep,
                    tpm_estimator=iim_tpm_estimator,
                    cut_mode=iim_cut_mode,
                    state_budget_policy=iim_state_budget_policy,
                    bearer_nodes=iim_settings["bearer_nodes"],
                )
            )
            _write_undefined_run(run_dir, meta, prep, effective_profile)
            runs.append(meta)
            continue

        meta = _register_defined_run(run_dir, meta, prep, effective_profile, iim_bins)
        runs.append(meta)
        _add_tasks(meta)

        n_null = int(iim_null_surrogates)
        if n_null <= 0:
            continue
        # Null surrogates: extra runs, drawn and prepared exactly as in
        # compute_IIM(null_surrogates=K) for this run.
        null_seed = int(iim_settings["null_seed"])
        null_min_shift = _resolve_null_min_shift(
            iim_null_min_shift, iim_lag_trs, prep["ts_selected"].shape[1]
        )
        null_info = {
            "n_surrogates": n_null,
            "method": str(iim_null_method),
            "seed": null_seed,
            "min_shift": int(null_min_shift),
            "run_keys": [],
            "unavailable_reason": None,
        }
        try:
            surrogates = iim_null_surrogate_series(
                prep["ts_selected"],
                n_null,
                method=str(iim_null_method),
                seed=null_seed,
                min_shift=null_min_shift,
            )
        except ValueError as exc:
            surrogates = []
            null_info["unavailable_reason"] = f"surrogates_unavailable: {exc}"
        for k, surr in enumerate(surrogates):
            null_key = f"{run_key}__null{k:03d}"
            null_dir = _run_artifact_dir(campaign_dir, null_key)
            null_dir.mkdir(parents=True, exist_ok=True)
            for name in _DERIVED_RESULT_FILES:
                with contextlib.suppress(FileNotFoundError):
                    (null_dir / name).unlink()
            np.save(null_dir / "surrogate_ts.npy", surr, allow_pickle=False)
            null_prep = prepare_iim_problem(
                surr,
                bins=int(prep["bins_used"]),
                lag_trs=int(iim_lag_trs),
                n_parts=iim_n_parts,
                rng=0,
                partition_mode="all",
                max_mechanism_size=int(prep["max_mechanism_size_used"]),
                max_purview_size=int(prep["max_purview_size_used"]),
                tpm_alpha=float(prep["tpm_alpha"]),
                max_state_space=int(prep["max_state_space"]),
                hardware_backend=hardware_backend,
                tpm_estimator=str(iim_tpm_estimator),
                node_selection="index",
                state_budget_policy="error",
                cut_mode=str(iim_cut_mode),
                log_label=null_key,
            )
            null_meta = {
                "run_index": int(len(runs)),
                "run_key": null_key,
                "subject": str(spec["subject"]),
                "session": str(spec["session"]),
                "ts_path": None,
                "ts_path_input": None,
                "surrogate_ts_path": str(null_dir / "surrogate_ts.npy"),
                "dataset_id": step2_context.get("dataset_id"),
                "data_origin": step2_context.get("data_origin"),
                "dataset_role": step2_context.get("dataset_role"),
                "provenance_label": step2_context.get("provenance_label"),
                "defined": bool(null_prep.get("defined", False)),
                "undefined_reason": null_prep.get("undefined_reason"),
                "created_unix": float(time.time()),
                "iim_bins": int(prep["bins_used"]),
                "iim_lag_trs": int(iim_lag_trs),
                "iim_n_parts": (None if iim_n_parts is None else int(iim_n_parts)),
                "iim_max_timepoints": meta["iim_max_timepoints"],
                "iim_max_nodes": None,
                "iim_max_mechanism_size": int(prep["max_mechanism_size_used"]),
                "iim_max_purview_size": int(prep["max_purview_size_used"]),
                "iim_settings": dict(iim_settings),
                "is_null_surrogate": True,
                "null_of": str(run_key),
                "null_index": int(k),
                "null_method": str(iim_null_method),
                "null_seed": null_seed,
                "null_min_shift": int(null_min_shift),
                "iim_null": None,
            }
            if bool(null_prep.get("defined", False)):
                null_meta = _register_defined_run(
                    null_dir, null_meta, null_prep, effective_profile, prep["bins_used"]
                )
                _add_tasks(null_meta)
            else:
                null_meta.update(
                    _undefined_prep_record(
                        null_prep,
                        tpm_estimator=iim_tpm_estimator,
                        cut_mode=iim_cut_mode,
                        state_budget_policy="error",
                        bearer_nodes=None,
                    )
                )
                _write_undefined_run(null_dir, null_meta, null_prep, effective_profile)
            runs.append(null_meta)
            null_info["run_keys"].append(null_key)
        meta["iim_null"] = null_info
        _json_dump(run_dir / "meta.json", meta)

    manifest = {
        "created_unix": float(time.time()),
        "campaign_dir": str(campaign_dir),
        "data_dir": str(Path(data_dir).resolve()),
        "dataset_id": step2_context.get("dataset_id"),
        "data_origin": step2_context.get("data_origin"),
        "dataset_role": step2_context.get("dataset_role"),
        "provenance_label": step2_context.get("provenance_label"),
        "atlas": str(atlas),
        "sessions": list(sessions),
        "condition": str(condition),
        "execution_profile": dataclasses.asdict(effective_profile),
        "hardware_backend": hardware_backend.to_dict(),
        "hardware_target": requested_target,
        "build_hardware_backend": backend_summary(hardware_backend),
        "scheduler": scheduler_settings,
        # Cut Psi does not depend on the phase-1 result, so the PBS DAG runs
        # phase-1 and cut shards concurrently; the legacy Slurm chain keeps
        # the phase-1 reduce before the cut shards.
        "cut_requires_phase1": scheduler_settings["scheduler"] != "pbs",
        "code_version": code_version,
        "kernel_code_version": kernel_code_version,
        "iim_algorithm_version": IIM_ALGORITHM_VERSION,
        "iim_settings": iim_settings,
        "iim_psi_kernel": str(iim_psi_kernel),
        "runs": runs,
        "phase1_tasks": phase1_tasks,
        "cut_tasks": cut_tasks,
        "step2_context": step2_context,
    }
    _json_dump(campaign_dir / "campaign_manifest.json", manifest)
    write_hunter_scheduler_scripts(campaign_dir, manifest)
    return manifest


# ---------------------------------------------------------------------------
# Scheduler scripts
# ---------------------------------------------------------------------------


def _stage_command(manifest, campaign_dir: Path, hardware_target: str) -> list[str]:
    settings = manifest.get("scheduler") or {}
    ctx = manifest.get("step2_context") or {}
    repo_root = Path(settings.get("repo_root") or REPO_ROOT)
    cmd = [
        *(settings.get("python_launcher") or _python_launcher()),
        str(repo_root / "run_pipeline.py"),
        "--execution-mode",
        "hunter",
        "--hunter-campaign-dir",
        str(campaign_dir),
        "--hardware-target",
        str(hardware_target),
    ]
    if ctx.get("dataset_id"):
        cmd.extend(["--dataset-id", str(ctx["dataset_id"])])
    if ctx.get("data_origin"):
        cmd.extend(["--data-origin", str(ctx["data_origin"])])
    if ctx.get("out_dir"):
        cmd.extend(["--out-dir", str(ctx["out_dir"])])
    return cmd


def _quote_cmd(parts) -> str:
    return " ".join(shlex.quote(str(x)) for x in parts)


def _runtime_preamble(
    settings: dict, *, uses_conda: bool, shard_threads: int | None
) -> str:
    lines = [
        "set -eo pipefail",
        "# Site setup is sourced before `set -u` (Lmod under nounset is unverified).",
    ]
    if settings.get("setup_file"):
        lines.append(f"source {shlex.quote(str(settings['setup_file']))}")
    if settings.get("inline_setup"):
        lines.append(str(settings["inline_setup"]).rstrip())
    lines.append("set -u")
    repo_root = shlex.quote(str(settings.get("repo_root") or REPO_ROOT))
    lines.append(f"cd {repo_root}")
    # The checkout the jobs run (code version and paths also for non-editable installs).
    lines.append(f"export {REPO_ROOT_ENV}={repo_root}")
    if not uses_conda:
        lines.append("# The setup above pins the interpreter; skip the conda check.")
        lines.append("export IMPACT_SKIP_ENV_CHECK=1")
    if shard_threads is not None:
        lines.append(
            "# One BLAS/OpenMP thread per worker process (workers are processes)."
        )
        for var in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            lines.append(f"export {var}={int(shard_threads)}")
    return "\n".join(lines) + "\n"


def _uses_conda(settings: dict) -> bool:
    launcher = [str(x) for x in (settings.get("python_launcher") or [])]
    return len(launcher) >= 2 and launcher[1] == "run" and "-n" in launcher


def _write_script(scripts_dir: Path, name: str, text: str) -> Path:
    path = scripts_dir / name
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


def _check_array_size(
    stage: str, n_subjobs: int, max_array_size: int, hint: str
) -> None:
    if int(n_subjobs) > int(max_array_size):
        raise ValueError(
            f"Hunter stage '{stage}' needs {int(n_subjobs)} array subjobs, above "
            f"the configured maximum array size {int(max_array_size)} "
            f"(IMPACT_HUNTER_MAX_ARRAY_SIZE). {hint}"
        )


def _pbs_select(
    node_type: str | None, *, mpiprocs: int | None = None, localscratch: bool = False
) -> str:
    parts = ["1"]
    if node_type:
        parts.append(f"node_type={node_type}")
    if localscratch:
        parts.append("node_type_storage=localscratch")
    if mpiprocs and int(mpiprocs) > 1:
        parts.append(f"mpiprocs={int(mpiprocs)}")
    return ":".join(parts)


def _pbs_header(
    name, *, select, walltime, settings, queue, array_size=1, comment=None
) -> list[str]:
    lines = [
        "#!/bin/bash",
        f"#PBS -N {name}",
        f"#PBS -l select={select}",
        f"#PBS -l walltime={walltime}",
    ]
    if settings.get("workspace_resource"):
        lines.append(f"#PBS -l {settings['workspace_resource']}")
    if queue:
        lines.append(f"#PBS -q {queue}")
    if settings.get("group_list"):
        lines.append(f"#PBS -W group_list={settings['group_list']}")
    lines.append("#PBS -j oe")
    if int(array_size) >= 2:
        # PBS arrays need >= 2 subjobs and must be rerunnable on Hunter.
        lines.append(f"#PBS -J 0-{int(array_size) - 1}")
        lines.append("#PBS -r y")
    if comment:
        lines.append(f"# {comment}")
    return lines


def _pbs_shard_launch(
    cmd: list[str], stage: str, settings: dict, index_expr: str
) -> str:
    spn = int(settings.get("shards_per_node", 1))
    if spn <= 1:
        return (
            f"{_quote_cmd(cmd)} --hunter-stage {stage} --hunter-task-index {index_expr}"
        )
    launch = [
        str(settings.get("launcher") or "mpiexec"),
        "-n",
        str(spn),
        "--ppn",
        str(spn),
    ]
    if settings.get("cpu_bind"):
        launch.extend(["--cpu-bind", str(settings["cpu_bind"])])
    if settings.get("gpu_bind"):
        launch.extend(["--gpu-bind", str(settings["gpu_bind"])])
    return (
        f"{_quote_cmd(launch)} {_quote_cmd(cmd)} --hunter-stage {stage} "
        f"--hunter-array-index {index_expr} --hunter-shards-per-node {spn}"
    )


def _write_hunter_pbs_scripts(campaign_dir: Path, manifest) -> dict:
    settings = dict(manifest.get("scheduler") or {})
    target = str(
        manifest.get("hardware_target")
        or (manifest.get("step2_context") or {}).get("hardware_target")
        or "cpu"
    )
    spn = int(settings.get("shards_per_node", 1))
    uses_conda = _uses_conda(settings)
    scripts_dir = campaign_dir / "pbs"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "logs").mkdir(exist_ok=True)

    phase1_tasks = manifest.get("phase1_tasks", [])
    cut_tasks = manifest.get("cut_tasks", [])
    n_p1 = packed_array_size(len(phase1_tasks), spn)
    n_cut = packed_array_size(len(cut_tasks), spn)
    hint = "Increase IMPACT_HUNTER_SHARDS_PER_NODE packing or lower the shards per run."
    _check_array_size("phase1-shard", n_p1, settings["max_array_size"], hint)
    _check_array_size("cut-shard", n_cut, settings["max_array_size"], hint)

    shard_select = _pbs_select(
        settings.get("node_type"),
        mpiprocs=spn,
        localscratch=bool(settings.get("localscratch")),
    )
    shard_preamble = _runtime_preamble(
        settings,
        uses_conda=uses_conda,
        shard_threads=settings.get("shard_omp_threads", 1),
    )
    # Phase-1 and cut shards both run on the requested target: on an
    # accelerator (hunter-apu) the Psi kernels run on the APU via CuPy (HLRS:
    # using the GPU cores is mandatory on Hunter).
    cmd_p1 = _stage_command(manifest, campaign_dir, target)
    cmd_cut = _stage_command(manifest, campaign_dir, target)
    written = {}

    def _index_line(n_array):
        return (
            'array_index="${PBS_ARRAY_INDEX:-0}"' if n_array >= 2 else "array_index=0"
        )

    if phase1_tasks:
        text = (
            "\n".join(
                _pbs_header(
                    "impact_p1",
                    select=shard_select,
                    walltime=settings["phase1_time"],
                    settings=settings,
                    queue=settings.get("queue"),
                    array_size=n_p1,
                    comment=(
                        f"IMPaCT IIM phase-1 shards: {len(phase1_tasks)} tasks, "
                        f"{spn} per node, {n_p1} node(s)."
                    ),
                )
            )
            + "\n"
            + shard_preamble
            + _index_line(n_p1)
            + "\n"
            + _pbs_shard_launch(cmd_p1, PHASE1_STAGE, settings, '"${array_index}"')
            + "\n"
        )
        written["phase1"] = _write_script(scripts_dir, "01_phase1_shards.pbs", text)
    if cut_tasks:
        text = (
            "\n".join(
                _pbs_header(
                    "impact_cut",
                    select=shard_select,
                    walltime=settings["cut_time"],
                    settings=settings,
                    queue=settings.get("queue"),
                    array_size=n_cut,
                    comment=(
                        f"IMPaCT IIM cut shards: {len(cut_tasks)} tasks, "
                        f"{spn} per node, {n_cut} node(s)."
                    ),
                )
            )
            + "\n"
            + shard_preamble
            + _index_line(n_cut)
            + "\n"
            + _pbs_shard_launch(cmd_cut, CUT_STAGE, settings, '"${array_index}"')
            + "\n"
        )
        written["cut"] = _write_script(scripts_dir, "02_cut_shards.pbs", text)

    # Reduce + finalize: one job for all runs. Pure-CPU work goes to a CPU/pre
    # queue only when one is configured (academic users cannot use genoa).
    use_cpu_queue = bool(settings.get("cpu_queue") or settings.get("cpu_node_type"))
    red_node_type = (
        settings.get("cpu_node_type") if use_cpu_queue else settings.get("node_type")
    )
    red_queue = settings.get("cpu_queue") if use_cpu_queue else settings.get("queue")
    red_target = "cpu" if use_cpu_queue else target
    cmd_red = _stage_command(manifest, campaign_dir, red_target)
    red_settings = dict(settings)
    if use_cpu_queue:
        red_settings["localscratch"] = False
    text = (
        "\n".join(
            _pbs_header(
                "impact_red",
                select=_pbs_select(red_node_type),
                walltime=settings["reduce_time"],
                settings=red_settings,
                queue=red_queue,
                comment="IMPaCT IIM reduce (all runs) + finalize pipeline in one job.",
            )
        )
        + "\n"
        + _runtime_preamble(settings, uses_conda=uses_conda, shard_threads=None)
        + (
            f"{_quote_cmd(cmd_red)} --hunter-stage reduce-all\n"
            f"{_quote_cmd(cmd_red)} --hunter-stage finalize-pipeline\n"
        )
    )
    written["reduce_finalize"] = _write_script(
        scripts_dir, "03_reduce_finalize.pbs", text
    )

    # Single-job smoke test (the 'test' queue allows one job per user, 25 min).
    smoke_lines = _pbs_header(
        "impact_smoke",
        select=shard_select,
        walltime=parse_and_cap_smoke_walltime(settings),
        settings=settings,
        queue=settings.get("smoke_queue"),
        comment="All stages in one job, for a tiny campaign (e.g. --iim-max-nodes 4).",
    )
    repo_src = Path(settings.get("repo_root") or REPO_ROOT) / "src"
    selftest_cmd = [
        *(settings.get("python_launcher") or ["python3"]),
        "-m",
        "impact_pipeline.hardware_selftest",
        "--target",
        target,
        "--json",
        str(campaign_dir / "hardware_selftest.json"),
    ]
    pythonpath = f"{shlex.quote(str(repo_src))}${{PYTHONPATH:+:${{PYTHONPATH}}}}"
    smoke_body = [
        shard_preamble.rstrip("\n"),
        f"export PYTHONPATH={pythonpath}",
        _quote_cmd(selftest_cmd),
    ]
    if phase1_tasks:
        smoke_body.append(
            f"for ((array_index=0; array_index<{n_p1}; array_index++)); do "
            + _pbs_shard_launch(cmd_p1, PHASE1_STAGE, settings, '"${array_index}"')
            + "; done"
        )
    if cut_tasks:
        smoke_body.append(
            f"for ((array_index=0; array_index<{n_cut}; array_index++)); do "
            + _pbs_shard_launch(cmd_cut, CUT_STAGE, settings, '"${array_index}"')
            + "; done"
        )
    cmd_smoke = _quote_cmd(_stage_command(manifest, campaign_dir, target))
    smoke_body.append(f"{cmd_smoke} --hunter-stage reduce-all")
    smoke_body.append(f"{cmd_smoke} --hunter-stage finalize-pipeline")
    written["smoke"] = _write_script(
        scripts_dir,
        "90_smoke_all_in_one.pbs",
        "\n".join(smoke_lines) + "\n" + "\n".join(smoke_body) + "\n",
    )

    submit = [
        "#!/bin/bash",
        "# Submit the IMPaCT Hunter IIM campaign (PBS Pro).",
        "# Run interactively on a Hunter login node. Re-running after a failure is",
        "# safe: completed shards are skipped. qsub does not forward this shell's",
        "# environment (no -V): job-time settings such as IMPACT_HUNTER_FORCE=1",
        "# (recompute) belong in the setup file sourced by every job.",
        "set -euo pipefail",
        'script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'campaign_dir="$(cd "${script_dir}/.." && pwd)"',
        'mkdir -p "${script_dir}/logs"',
        "# PBS writes <jobname>.o<jobid> files into the submission directory.",
        'cd "${script_dir}/logs"',
    ]
    deps = []
    if phase1_tasks:
        submit.append('jid_p1=$(qsub "${script_dir}/01_phase1_shards.pbs")')
        submit.append('echo "phase1_shards=${jid_p1}"')
        deps.append("${jid_p1}")
    if cut_tasks:
        submit.append('jid_cut=$(qsub "${script_dir}/02_cut_shards.pbs")')
        submit.append('echo "cut_shards=${jid_cut}"')
        deps.append("${jid_cut}")
    if deps:
        # Dependencies apply to whole arrays (PBS has no subjob dependencies).
        depend = "afterok:" + ":".join(deps)
        red_script = '"${script_dir}/03_reduce_finalize.pbs"'
        submit.append(f"jid_red=$(qsub -W depend={depend} {red_script})")
    else:
        submit.append('jid_red=$(qsub "${script_dir}/03_reduce_finalize.pbs")')
    submit.append('echo "reduce_finalize=${jid_red}"')
    submit.append(
        '{ echo "phase1_shards=${jid_p1:-}"; echo "cut_shards=${jid_cut:-}"; '
        'echo "reduce_finalize=${jid_red}"; } > "${script_dir}/submitted_jobs.txt"'
    )
    written["submit"] = _write_script(
        scripts_dir, "00_submit_all.sh", "\n".join(submit) + "\n"
    )

    plan = {
        "scheduler": "pbs",
        "shards_per_node": spn,
        "stages": {
            "phase1-shard": {
                "tasks": len(phase1_tasks),
                "subjobs": n_p1,
                "array": n_p1 >= 2,
                "walltime": settings["phase1_time"],
                "node_hours_upper_bound": n_p1
                * parse_walltime(settings["phase1_time"])
                / 3600.0,
            },
            "cut-shard": {
                "tasks": len(cut_tasks),
                "subjobs": n_cut,
                "array": n_cut >= 2,
                "walltime": settings["cut_time"],
                "node_hours_upper_bound": n_cut
                * parse_walltime(settings["cut_time"])
                / 3600.0,
            },
            "reduce-finalize": {
                "tasks": 1,
                "subjobs": 1,
                "array": False,
                "walltime": settings["reduce_time"],
                "node_type": red_node_type,
                "queue": red_queue,
                "hardware_target": red_target,
                "node_hours_upper_bound": parse_walltime(settings["reduce_time"])
                / 3600.0,
            },
        },
        "notes": [
            "Whole mi300a nodes are allocated and charged; shards are packed per "
            "node with PALS mpiexec.",
            "HLRS 'single' queue: 40 queued / 20 running jobs per user; whether "
            "array subjobs count individually is unverified (ask HLRS).",
            "Array maximum size on Hunter is unverified; the guard uses "
            "IMPACT_HUNTER_MAX_ARRAY_SIZE.",
        ],
    }
    _json_dump(scripts_dir / "campaign_plan.json", plan)
    return {
        "scripts_dir": str(scripts_dir),
        "plan": plan,
        "written": {k: str(v) for k, v in written.items()},
    }


def parse_and_cap_smoke_walltime(settings: dict) -> str:
    cap = "00:25:00"
    raw = _env_value("IMPACT_HUNTER_SMOKE_TIME") or cap
    if settings.get("smoke_queue") == "test":
        return _validate_walltime("IMPACT_HUNTER_SMOKE_TIME", raw, cap, "'test' queue")
    return _validate_walltime(
        "IMPACT_HUNTER_SMOKE_TIME",
        raw,
        settings.get("max_walltime") or "24:00:00",
        "PBS walltime",
    )


def _write_hunter_slurm_scripts(campaign_dir: Path, manifest):
    settings = dict(manifest.get("scheduler") or {})
    if settings.get("scheduler") != "slurm":
        # Manifests built before the PBS backend existed: resolve Slurm settings now.
        profile = manifest.get("execution_profile") or {}
        _eff, settings = resolve_hunter_settings(
            ExecutionProfile(
                name=str(profile.get("name", "hunter")),
                description=str(profile.get("description", "")),
                distributed_iim=True,
                hunter_slurm=HunterSlurmProfile(),
            ),
            scheduler="slurm",
        )
    target = str(
        manifest.get("hardware_target")
        or (manifest.get("step2_context") or {}).get("hardware_target")
        or (manifest.get("hardware_backend") or {}).get("requested")
        or "cpu"
    )
    use_accelerator = target in {"gpu", "hunter-apu"}
    uses_conda = _uses_conda(settings)
    account_line = ""
    if settings.get("account"):
        account_line += f"#SBATCH --account={settings['account']}\n"
    if settings.get("qos"):
        account_line += f"#SBATCH --qos={settings['qos']}\n"

    scripts_dir = campaign_dir / "slurm"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    log_dir = scripts_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    phase1_tasks = manifest.get("phase1_tasks", [])
    cut_tasks = manifest.get("cut_tasks", [])
    runs = manifest.get("runs", [])
    hint = "Lower IMPACT_HUNTER_*_SHARDS_PER_RUN or raise MaxArraySize."
    for stage, n in (
        ("phase1-shard", len(phase1_tasks)),
        ("cut-shard", len(cut_tasks)),
        ("reduce", len(runs)),
    ):
        _check_array_size(stage, n, settings["max_array_size"], hint)
    throttle = settings.get("array_throttle")
    throttle_txt = f"%{int(throttle)}" if throttle not in (None, "") else ""

    def _array_line(n):
        return f"#SBATCH --array=0-{int(n) - 1}{throttle_txt}\n"

    def _out_line(array):
        pattern = "%x_%A_%a.out" if array else "%x_%j.out"
        return f"#SBATCH --output={log_dir / pattern}\n"

    def _gpu_line():
        gpus = int(settings.get("gpus_per_task", 0) or 0)
        return f"#SBATCH --gpus-per-task={max(1, gpus)}\n"

    cpu_partition = str(settings.get("cpu_partition", "cpu"))
    apu_partition = str(settings.get("apu_partition", "apu"))
    shard_preamble = _runtime_preamble(
        settings,
        uses_conda=uses_conda,
        shard_threads=settings.get("shard_omp_threads", 1),
    )
    plain_preamble = _runtime_preamble(
        settings, uses_conda=uses_conda, shard_threads=None
    )
    cmd_cpu = _quote_cmd(_stage_command(manifest, campaign_dir, "cpu"))
    cmd_target = _quote_cmd(_stage_command(manifest, campaign_dir, target))
    task_idx = "--hunter-task-index $SLURM_ARRAY_TASK_ID"
    run_idx = "--hunter-run-index $SLURM_ARRAY_TASK_ID"
    cut_partition = apu_partition if use_accelerator else cpu_partition

    if phase1_tasks:
        # Phase-1 Psi is CPU-only: CPU partition, no GPU request.
        _write_script(
            scripts_dir,
            "01_phase1_shards.sbatch",
            (
                "#!/bin/bash\n"
                "#SBATCH --job-name=impact_iim_p1\n"
                f"#SBATCH --partition={cpu_partition}\n"
                f"#SBATCH --time={settings['phase1_time']}\n"
                f"#SBATCH --cpus-per-task={int(settings['cpus_per_task'])}\n"
                f"#SBATCH --mem={settings.get('mem_per_task', '0')}\n"
                f"{_array_line(len(phase1_tasks))}"
                f"{_out_line(True)}"
                f"{account_line}"
                f"{shard_preamble}"
                f"{cmd_cpu} --hunter-stage phase1-shard {task_idx}\n"
            ),
        )
    if runs:
        _write_script(
            scripts_dir,
            "02_phase1_reduce.sbatch",
            (
                "#!/bin/bash\n"
                "#SBATCH --job-name=impact_iim_p1r\n"
                f"#SBATCH --partition={cpu_partition}\n"
                f"#SBATCH --time={settings['reduce_time']}\n"
                "#SBATCH --cpus-per-task=1\n"
                f"#SBATCH --mem={settings.get('mem_per_task', '0')}\n"
                f"{_array_line(len(runs))}"
                f"{_out_line(True)}"
                f"{account_line}"
                f"{plain_preamble}"
                f"{cmd_cpu} --hunter-stage phase1-reduce {run_idx}\n"
            ),
        )
    if cut_tasks:
        _write_script(
            scripts_dir,
            "03_cut_shards.sbatch",
            (
                "#!/bin/bash\n"
                "#SBATCH --job-name=impact_iim_cut\n"
                f"#SBATCH --partition={cut_partition}\n"
                f"#SBATCH --time={settings['cut_time']}\n"
                f"#SBATCH --cpus-per-task={int(settings['cpus_per_task'])}\n"
                f"#SBATCH --mem={settings.get('mem_per_task', '0')}\n"
                f"{_gpu_line() if use_accelerator else ''}"
                f"{_array_line(len(cut_tasks))}"
                f"{_out_line(True)}"
                f"{account_line}"
                f"{shard_preamble}"
                f"{cmd_target} --hunter-stage cut-shard {task_idx}\n"
            ),
        )
    if runs:
        _write_script(
            scripts_dir,
            "04_cut_reduce.sbatch",
            (
                "#!/bin/bash\n"
                "#SBATCH --job-name=impact_iim_red\n"
                f"#SBATCH --partition={cpu_partition}\n"
                f"#SBATCH --time={settings['reduce_time']}\n"
                "#SBATCH --cpus-per-task=1\n"
                f"#SBATCH --mem={settings.get('mem_per_task', '0')}\n"
                f"{_array_line(len(runs))}"
                f"{_out_line(True)}"
                f"{account_line}"
                f"{plain_preamble}"
                f"{cmd_cpu} --hunter-stage cut-reduce {run_idx}\n"
            ),
        )
        _write_script(
            scripts_dir,
            "05_finalize_pipeline.sbatch",
            (
                "#!/bin/bash\n"
                "#SBATCH --job-name=impact_finalize\n"
                f"#SBATCH --partition={cpu_partition}\n"
                f"#SBATCH --time={settings['reduce_time']}\n"
                "#SBATCH --cpus-per-task=4\n"
                f"#SBATCH --mem={settings.get('mem_per_task', '0')}\n"
                f"{_out_line(False)}"
                f"{account_line}"
                f"{plain_preamble}"
                f"{cmd_cpu} --hunter-stage finalize-pipeline\n"
            ),
        )
        submit_lines = [
            "#!/bin/bash",
            "set -euo pipefail",
            'script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
            'campaign_dir="$(cd "${script_dir}/.." && pwd)"',
            'cd "${campaign_dir}"',
        ]
        if phase1_tasks:
            submit_lines.extend(
                [
                    'jid_p1=$(sbatch --parsable slurm/01_phase1_shards.sbatch)',
                    'jid_p1r=$(sbatch --parsable --dependency=afterok:${jid_p1} slurm/02_phase1_reduce.sbatch)',
                ]
            )
        else:
            submit_lines.append('jid_p1r=$(sbatch --parsable slurm/02_phase1_reduce.sbatch)')
        if cut_tasks:
            submit_lines.extend(
                [
                    'jid_cut=$(sbatch --parsable --dependency=afterok:${jid_p1r} slurm/03_cut_shards.sbatch)',
                    'jid_red=$(sbatch --parsable --dependency=afterok:${jid_cut} slurm/04_cut_reduce.sbatch)',
                ]
            )
        else:
            submit_lines.append('jid_red=$(sbatch --parsable --dependency=afterok:${jid_p1r} slurm/04_cut_reduce.sbatch)')
        submit_lines.extend(
            [
                'jid_fin=$(sbatch --parsable --dependency=afterok:${jid_red} slurm/05_finalize_pipeline.sbatch)',
                'echo "phase1_reduce=${jid_p1r}"',
                'echo "cut_reduce=${jid_red}"',
                'echo "finalize=${jid_fin}"',
            ]
        )
        _write_script(scripts_dir, "00_submit_all.sh", "\n".join(submit_lines) + "\n")
    return {"scripts_dir": str(scripts_dir)}


def write_hunter_scheduler_scripts(campaign_dir, manifest) -> dict:
    campaign_dir = Path(campaign_dir).resolve()
    sched = (manifest.get("scheduler") or {}).get("scheduler") or "slurm"
    if sched == "pbs":
        out = _write_hunter_pbs_scripts(campaign_dir, manifest)
    else:
        out = _write_hunter_slurm_scripts(campaign_dir, manifest)
    return out


# ---------------------------------------------------------------------------
# Stage execution
# ---------------------------------------------------------------------------


def _load_manifest(campaign_dir: Path):
    return _json_load(campaign_dir / "campaign_manifest.json")


def _task_label(stage: str, task: dict) -> str:
    return f"{task['run_key']}_{stage}_{int(task['task_index']):04d}"


def _shard_hardware_backend(manifest, hardware_target=None):
    """
    Backend of a shard job (strict): the job's own ``--hardware-target`` when
    the caller passes it, else the campaign's requested target. The scheduler
    scripts set the target per stage (PBS: the campaign target for both shard
    stages; Slurm: phase-1 shards on the CPU partition with 'cpu'), so the
    job's target must win over the campaign's.
    """
    if hardware_target is None:
        hardware_target = (
            (manifest.get("step2_context") or {}).get("hardware_target")
            or manifest.get("hardware_target")
            or (manifest.get("hardware_backend") or {}).get("requested")
            or "cpu"
        )
    return configure_process_for_hardware(hardware_target)


def _shard_psi_kernel(manifest, backend) -> str:
    """
    Psi kernel of a shard: the array-module kernel (CuPy) on accelerator
    targets so the Psi work runs on the GPU/APU, the numba kernel on the CPU
    (overridable by the campaign's iim_psi_kernel or IMPACT_IIM_PSI_KERNEL).
    """
    return resolve_iim_psi_kernel(manifest.get("iim_psi_kernel") or "auto", backend)


def run_phase1_shard(campaign_dir, task_index, *, hardware_target=None):
    campaign_dir = Path(campaign_dir).resolve()
    manifest = _load_manifest(campaign_dir)
    hardware_backend = _shard_hardware_backend(manifest, hardware_target)
    psi_kernel = _shard_psi_kernel(manifest, hardware_backend)
    task = manifest["phase1_tasks"][int(task_index)]
    run_dir = _run_artifact_dir(campaign_dir, task["run_key"])
    meta, problem = _load_problem(run_dir)
    out_path = run_dir / "phase1_shards" / f"shard_{int(task['task_index']):04d}.json"
    if not bool(meta.get("defined", False)):
        payload = {"status": "skipped", "defined": False, "psi_partial": None}
        _json_dump(out_path, payload)
        return payload

    identity = _task_identity(meta, task, PHASE1_STAGE, _runtime_code_version())
    existing = _completed_payload(out_path, identity)
    if existing is not None:
        log.info("phase1 shard %s already complete; skipping.", out_path)
        return existing

    start = _timing_start()
    shard_mechanisms = problem["mechanisms_all"][int(task["start"]):int(task["stop"])]
    profile = manifest["execution_profile"]
    if psi_kernel == "xp":
        # Psi on the device (CuPy on hunter-apu); no process pool or SQLite cache.
        xp = get_array_module(hardware_backend)
        psi_partial = iim_xp.psi_contribution(
            shard_mechanisms,
            problem["purviews_all"],
            int(meta["bins_used"]),
            xp.asarray(problem["tpm_full"], dtype=xp.float64),
            problem["curr_obs"],
            problem["states_full"],
            xp=xp,
        )
    else:
        with _kernel_cache_scope(campaign_dir, _task_label("p1", task)) as cache_dir:
            kernel_cache_path = (
                None if cache_dir is None else cache_dir / "kernel.sqlite3"
            )
            psi_partial = _compute_psi_for_problem(
                tpm=problem["tpm_full"],
                curr_obs=problem["curr_obs"],
                states_full=problem["states_full"],
                base=int(meta["bins_used"]),
                mechanisms=shard_mechanisms,
                purviews=problem["purviews_all"],
                cut_mask_a=None,
                phase1_parallel_workers=profile["hunter_phase1_workers_per_task"],
                phase1_chunk_size=profile["hunter_phase1_chunk_size"],
                phase1_shared_memory=profile["hunter_shared_memory"],
                kernel_cache_path=(
                    None if kernel_cache_path is None else str(kernel_cache_path)
                ),
            )
    timing = _timing_finish(
        start,
        stage=PHASE1_STAGE,
        task_index=int(task_index),
        n_mechanisms=len(shard_mechanisms),
        hardware_backend=backend_summary(hardware_backend),
        psi_kernel=psi_kernel,
    )
    payload = {
        "status": "complete",
        "defined": True,
        "run_key": str(task["run_key"]),
        "task_index": int(task["task_index"]),
        "start": int(task["start"]),
        "stop": int(task["stop"]),
        "n_mechanisms": int(len(shard_mechanisms)),
        "psi_partial": float(psi_partial),
        "psi_kernel": psi_kernel,
        "identity": identity,
        "timing": timing,
    }
    _json_dump(out_path, payload)
    _write_timing(campaign_dir, PHASE1_STAGE, _task_label("p1", task), timing)
    return payload


def _shard_record(path: Path, meta: dict, shard: dict, stage: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {stage} result: {path}")
    rec = _json_load(path)
    if rec.get("status") != "complete":
        raise RuntimeError(f"{stage} did not complete: {path}")
    identity = rec.get("identity")
    if identity is not None:
        if identity.get("problem_digest") != meta.get("problem_digest") or (
            int(identity.get("start", -1)),
            int(identity.get("stop", -1)),
        ) != (int(shard["start"]), int(shard["stop"])):
            raise RuntimeError(
                f"{stage} result {path} belongs to a different campaign build; "
                "rerun the shard."
            )
    return rec


def run_phase1_reduce(campaign_dir, run_index):
    campaign_dir = Path(campaign_dir).resolve()
    manifest = _load_manifest(campaign_dir)
    run_meta = manifest["runs"][int(run_index)]
    run_dir = _run_artifact_dir(campaign_dir, run_meta["run_key"])
    if not bool(run_meta.get("defined", False)):
        payload = {
            "status": "skipped",
            "defined": False,
            "undefined_reason": run_meta.get("undefined_reason"),
        }
        _json_dump(run_dir / "phase1_result.json", payload)
        return payload

    partials = []
    versions = []
    kernels = set()
    for shard in run_meta["phase1_shards"]:
        shard_path = run_dir / "phase1_shards" / f"shard_{int(shard['task_index']):04d}.json"
        rec = _shard_record(shard_path, run_meta, shard, "Phase1 shard")
        partials.append(float(rec["psi_partial"]))
        versions.append((rec.get("identity") or {}).get("code_version"))
        if rec.get("psi_kernel"):
            kernels.add(str(rec["psi_kernel"]))
    code_version = _check_single_code_version(versions, run_meta["run_key"])

    psi_full = float(math.fsum(partials)) if partials else 0.0
    payload = {
        "status": "complete",
        "defined": True,
        "psi_full": float(psi_full),
        "n_shards": int(len(partials)),
        "problem_digest": run_meta.get("problem_digest"),
        "code_version": code_version,
        "psi_kernels": sorted(kernels),
    }
    _json_dump(run_dir / "phase1_result.json", payload)
    return payload


def run_cut_shard(
    campaign_dir, task_index, *, require_phase1=None, hardware_target=None
):
    campaign_dir = Path(campaign_dir).resolve()
    manifest = _load_manifest(campaign_dir)
    hardware_backend = _shard_hardware_backend(manifest, hardware_target)
    psi_kernel = _shard_psi_kernel(manifest, hardware_backend)
    task = manifest["cut_tasks"][int(task_index)]
    run_dir = _run_artifact_dir(campaign_dir, task["run_key"])
    meta, problem = _load_problem(run_dir)
    out_path = run_dir / "cut_shards" / f"shard_{int(task['task_index']):04d}.json"
    if not bool(meta.get("defined", False)):
        payload = {"status": "skipped", "defined": False, "best_cut": None, "best_psi": None}
        _json_dump(out_path, payload)
        return payload

    if require_phase1 is None:
        require_phase1 = bool(manifest.get("cut_requires_phase1", True))
    phase1_path = run_dir / "phase1_result.json"
    if require_phase1 and not phase1_path.exists():
        raise FileNotFoundError(f"Missing phase1 reduction result: {phase1_path}")

    identity = _task_identity(meta, task, CUT_STAGE, _runtime_code_version())
    existing = _completed_payload(out_path, identity)
    if existing is not None:
        log.info("cut shard %s already complete; skipping.", out_path)
        return existing

    # Per-cut checkpoint so a walltime kill loses at most one cut.
    partial_path = out_path.with_name(out_path.stem + ".partial.json")
    cut_scores = {}
    if partial_path.exists() and not _env_flag(FORCE_ENV, False):
        try:
            partial = _json_load(partial_path)
            if partial.get("identity") == identity:
                cut_scores = {
                    str(k): float(v)
                    for k, v in (partial.get("cut_scores") or {}).items()
                }
        except Exception:
            cut_scores = {}
    resumed_cuts = len(cut_scores)

    start = _timing_start()
    shard_cuts = problem["cuts_eval"][int(task["start"]):int(task["stop"])]
    static_cache = {}
    obs_state_cache = {}
    profile = manifest["execution_profile"]
    cut_mode = str(meta.get("cut_mode") or "bidirectional")
    base = int(meta["bins_used"])
    n_nodes = int(problem["states_full"].shape[1])
    xp = None
    xp_workspace = None
    tpm_full_xp = None
    if psi_kernel == "xp":
        # Cut TPMs and their Psi on the device (CuPy on hunter-apu); the
        # purview marginalisers are shared by all cuts of the shard.
        xp = get_array_module(hardware_backend)
        xp_workspace = iim_xp.IIMXpWorkspace(xp, problem["states_full"], base)
        tpm_full_xp = xp.asarray(problem["tpm_full"], dtype=xp.float64)
    with _kernel_cache_scope(campaign_dir, _task_label("cut", task)) as cache_dir:
        for local_idx, (A, B) in enumerate(shard_cuts):
            cut_key = _iim_cut_to_key(A, B)
            if cut_key in cut_scores:
                continue
            if psi_kernel == "xp":
                psi_cut = iim_xp.psi_contribution(
                    problem["mechanisms_all"],
                    problem["purviews_all"],
                    base,
                    iim_xp.cut_tpm(
                        tpm_full_xp, n_nodes, base, A, B, cut_mode=cut_mode, xp=xp
                    ),
                    problem["curr_obs"],
                    problem["states_full"],
                    xp=xp,
                    workspace=xp_workspace,
                )
                cut_scores[cut_key] = float(psi_cut)
                _json_dump(
                    partial_path, {"identity": identity, "cut_scores": cut_scores}
                )
                continue
            cut_mask_a = 0
            for nn in A:
                cut_mask_a |= 1 << int(nn)
            tpm_cut = _iim_build_cut_tpm_for_mode(
                problem["tpm_full"],
                problem["states_full"],
                base,
                A,
                B,
                cut_mode=cut_mode,
                hardware_backend=hardware_backend,
            )
            # A fresh cache per cut: within one cut the induced-partition keys
            # only memoise, so this is valid for both cut modes.
            kernel_cache_path = (
                None
                if cache_dir is None
                else cache_dir / f"kernel_{int(local_idx):04d}.sqlite3"
            )
            psi_cut = _compute_psi_for_problem(
                tpm=tpm_cut,
                curr_obs=problem["curr_obs"],
                states_full=problem["states_full"],
                base=int(meta["bins_used"]),
                mechanisms=problem["mechanisms_all"],
                purviews=problem["purviews_all"],
                cut_mask_a=int(cut_mask_a),
                phase1_parallel_workers=profile["hunter_phase1_workers_per_task"],
                phase1_chunk_size=profile["hunter_phase1_chunk_size"],
                phase1_shared_memory=profile["hunter_shared_memory"],
                kernel_cache_path=(
                    None if kernel_cache_path is None else str(kernel_cache_path)
                ),
                static_cache=static_cache,
                obs_state_cache=obs_state_cache,
            )
            if kernel_cache_path is not None:
                for suffix in ("", "-wal", "-shm"):
                    with contextlib.suppress(OSError):
                        Path(str(kernel_cache_path) + suffix).unlink()
            cut_scores[cut_key] = float(psi_cut)
            _json_dump(partial_path, {"identity": identity, "cut_scores": cut_scores})

    best_cut = None
    best_psi = -np.inf
    # Deterministic MIP choice: first cut (campaign order) with the max preserved Psi.
    for A, B in shard_cuts:
        psi_cut = cut_scores[_iim_cut_to_key(A, B)]
        if float(psi_cut) > float(best_psi):
            best_psi = float(psi_cut)
            best_cut = [list(A), list(B)]

    timing = _timing_finish(
        start,
        stage=CUT_STAGE,
        task_index=int(task_index),
        n_cuts=len(shard_cuts),
        resumed_cuts=int(resumed_cuts),
        hardware_backend=backend_summary(hardware_backend),
        psi_kernel=psi_kernel,
    )
    payload = {
        "status": "complete",
        "defined": True,
        "run_key": str(task["run_key"]),
        "task_index": int(task["task_index"]),
        "start": int(task["start"]),
        "stop": int(task["stop"]),
        "cut_scores": cut_scores,
        "best_cut": best_cut,
        "best_psi": (None if not np.isfinite(best_psi) else float(best_psi)),
        "psi_kernel": psi_kernel,
        "identity": identity,
        "timing": timing,
    }
    _json_dump(out_path, payload)
    with contextlib.suppress(OSError):
        partial_path.unlink()
    _write_timing(campaign_dir, CUT_STAGE, _task_label("cut", task), timing)
    return payload


def _run_provenance_fields(run_meta) -> dict:
    """Estimator/subsystem provenance of a run (as in compute_IIM details)."""
    out = {
        key: run_meta.get(key)
        for key in (
            "iim_algorithm_version",
            "tpm_estimator",
            "tpm_alpha",
            "cut_mode",
            "bearer_nodes",
            "selected_nodes",
            "node_selection_rule",
            "node_selection_degenerate",
            "bins_requested",
            "state_budget_policy",
            "budget_adjustments",
            "n_states",
            "n_states_observed",
            "n_transitions",
        )
        if key in run_meta
    }
    out.setdefault("iim_algorithm_version", IIM_ALGORITHM_VERSION)
    if "lag_trs" in run_meta or "iim_lag_trs" in run_meta:
        out["lag_trs"] = run_meta.get("lag_trs", run_meta.get("iim_lag_trs"))
    out["is_null_surrogate"] = bool(run_meta.get("is_null_surrogate", False))
    if out["is_null_surrogate"]:
        out["null_of"] = run_meta.get("null_of")
        out["null_index"] = run_meta.get("null_index")
    return out


def _run_null_plan(run_meta) -> dict:
    """
    Null-calibration plan of a run: number of surrogates, method, seed and
    minimum shift exactly as compute_IIM resolves them (surrogate runs are not
    calibrated themselves).
    """
    settings = dict(run_meta.get("iim_settings") or {})
    null_info = run_meta.get("iim_null") or {}
    n_null = 0
    if not bool(run_meta.get("is_null_surrogate", False)):
        n_null = int(
            null_info.get("n_surrogates", settings.get("null_surrogates", 0)) or 0
        )
    min_shift = null_info.get("min_shift")
    if min_shift is None and run_meta.get("n_time_input") is not None:
        min_shift = _resolve_null_min_shift(
            settings.get("null_min_shift"),
            int(run_meta.get("iim_lag_trs", 1)),
            int(run_meta["n_time_input"]),
        )
    return {
        "n_surrogates": n_null,
        "method": str(
            null_info.get("method", settings.get("null_method", "circular_shift"))
        ),
        "seed": _resolve_null_seed(null_info.get("seed", settings.get("null_seed")), 0),
        "min_shift": min_shift,
        "requested_min_shift": settings.get("null_min_shift"),
        "run_keys": list(null_info.get("run_keys") or []),
        "unavailable_reason": null_info.get("unavailable_reason"),
    }


def _undefined_null_fields(plan) -> dict:
    """Null schema of an undefined result (compute_IIM._undefined_payload)."""
    return _iim_null_fields(
        None,
        psi_full=np.nan,
        delta_psi=np.nan,
        meta={
            "IIM_null_method": plan["method"],
            "IIM_null_seed": int(plan["seed"]),
            "IIM_null_min_shift": (
                None
                if plan["requested_min_shift"] is None
                else int(plan["requested_min_shift"])
            ),
            "IIM_null_failed": 0,
            "IIM_null_undefined_reason": (
                "observed_iim_undefined" if plan["n_surrogates"] > 0 else None
            ),
        },
    )


def run_cut_reduce(campaign_dir, run_index, clamp=True, scale=1.0):
    """
    Final IIM of one run from its phase-1 result and cut shards. For a real run
    with surrogate runs (``iim_null``) the surrogate runs are reduced first and
    the null-calibration fields (IIM_null_mean/sd, IIM_z, IIM_excess,
    canonical_calibrated) are computed with the same function as
    compute_IIM(null_surrogates=K); ``value`` is then the calibrated value.
    """
    campaign_dir = Path(campaign_dir).resolve()
    manifest = _load_manifest(campaign_dir)
    run_meta = manifest["runs"][int(run_index)]
    run_dir = _run_artifact_dir(campaign_dir, run_meta["run_key"])
    plan = _run_null_plan(run_meta)
    provenance = _run_provenance_fields(run_meta)
    if not bool(run_meta.get("defined", False)):
        final_path = run_dir / "final_result.json"
        if final_path.exists():
            payload = _json_load(final_path)
        else:
            payload = {
                "value": None,
                "raw": None,
                "canonical": None,
                "clipped": None,
                "iim_plus": None,
                "defined": False,
                "undefined_reason": run_meta.get("undefined_reason"),
            }
        for key, val in {**provenance, **_undefined_null_fields(plan)}.items():
            payload.setdefault(key, val)
        _json_dump(final_path, payload)
        return payload

    phase1 = _json_load(run_dir / "phase1_result.json")
    if phase1.get("problem_digest", run_meta.get("problem_digest")) != run_meta.get(
        "problem_digest"
    ):
        raise RuntimeError(
            f"{run_dir / 'phase1_result.json'} belongs to a different campaign build; "
            "rerun the phase-1 reduction."
        )
    psi_full = float(phase1["psi_full"])
    build_fields = {
        "problem_digest": run_meta.get("problem_digest"),
        "code_version": phase1.get("code_version"),
    }
    if not np.isfinite(psi_full) or psi_full <= 0:
        payload = {
            "value": None,
            "raw": None,
            "canonical": None,
            "clipped": None,
            "iim_plus": None,
            "defined": False,
            "undefined_reason": "nonpositive_psi_full",
            "Psi_full": float(psi_full),
            "Psi_mip_preserved": None,
            "n_nodes_used": int(run_meta["n_nodes_used"]),
            "bins_used": int(run_meta["bins_used"]),
            "n_cuts_evaluated": int(run_meta["n_cuts_evaluated"]),
            **provenance,
            **_undefined_null_fields(plan),
            **build_fields,
        }
        _json_dump(run_dir / "final_result.json", payload)
        return payload

    best_psi = -np.inf
    best_cut = None
    n_scored = 0
    versions = [phase1.get("code_version")]
    kernels = set()
    for shard in run_meta["cut_shards"]:
        shard_path = run_dir / "cut_shards" / f"shard_{int(shard['task_index']):04d}.json"
        rec = _shard_record(shard_path, run_meta, shard, "Cut shard")
        versions.append((rec.get("identity") or {}).get("code_version"))
        n_scored += len(rec.get("cut_scores") or {})
        if rec.get("psi_kernel"):
            kernels.add(str(rec["psi_kernel"]))
        psi = rec.get("best_psi")
        if psi is None:
            continue
        psi = float(psi)
        if psi > best_psi:
            best_psi = psi
            best_cut = rec.get("best_cut")
    if n_scored != int(run_meta["n_cuts_evaluated"]):
        raise RuntimeError(
            f"Cut shards for {run_meta['run_key']} scored {n_scored} cuts, "
            f"expected {int(run_meta['n_cuts_evaluated'])}."
        )
    # Psi_full (phase 1) and the cut Psi values must come from the same code.
    build_fields["code_version"] = _check_single_code_version(
        versions, run_meta["run_key"]
    )
    kernels.update(str(k) for k in (phase1.get("psi_kernels") or []))

    if not np.isfinite(best_psi):
        payload = {
            "value": None,
            "raw": None,
            "canonical": None,
            "clipped": None,
            "iim_plus": None,
            "defined": False,
            "undefined_reason": "mip_not_found",
            "Psi_full": float(psi_full),
            "Psi_mip_preserved": None,
            "n_nodes_used": int(run_meta["n_nodes_used"]),
            "bins_used": int(run_meta["bins_used"]),
            "n_cuts_evaluated": int(run_meta["n_cuts_evaluated"]),
            **provenance,
            **_undefined_null_fields(plan),
            **build_fields,
        }
        _json_dump(run_dir / "final_result.json", payload)
        return payload

    raw = float((float(psi_full) - float(best_psi)) / (float(psi_full) + 1e-12))
    canonical = float(np.clip(raw, 0.0, 1.0))
    delta_psi = float(psi_full - best_psi)
    null_results = []
    if plan["n_surrogates"] > 0:
        index_by_key = {
            str(r["run_key"]): i for i, r in enumerate(manifest.get("runs", []))
        }
        for key in plan["run_keys"]:
            if key not in index_by_key:
                raise RuntimeError(
                    f"Surrogate run {key} of {run_meta['run_key']} is missing from "
                    "the campaign manifest; rebuild the campaign."
                )
            null_results.append(
                run_cut_reduce(
                    campaign_dir, index_by_key[key], clamp=clamp, scale=scale
                )
            )
    null_fields, _stats = iim_null_calibration_fields(
        delta_psi,
        psi_full,
        null_results,
        n_requested=plan["n_surrogates"],
        method=plan["method"],
        seed=plan["seed"],
        min_shift=plan["min_shift"],
        unavailable_reason=plan["unavailable_reason"],
    )
    value = iim_calibrated_value(
        null_fields, canonical, raw, plan["n_surrogates"], clamp=clamp, scale=scale
    )
    payload = {
        "value": value,
        "raw": raw,
        "canonical": canonical,
        "clipped": canonical,
        "iim_plus": canonical,
        "scale": float(scale),
        "I_full": float(psi_full),
        "min_partition_sum": float(best_psi),
        "Psi_full": float(psi_full),
        "Psi_mip_preserved": float(best_psi),
        "n_nodes_used": int(run_meta["n_nodes_used"]),
        "bins_used": int(run_meta["bins_used"]),
        "bins_reduction_reason": run_meta.get("bins_reduction_reason"),
        "max_nodes_requested": run_meta.get("iim_max_nodes"),
        "max_mechanism_size_used": int(run_meta["max_mechanism_size_used"]),
        "max_purview_size_used": int(run_meta["max_purview_size_used"]),
        "n_parts_requested": run_meta.get("iim_n_parts"),
        "n_cuts_evaluated": int(run_meta["n_cuts_evaluated"]),
        "mip_cut": best_cut,
        "checkpoint_path": None,
        "checkpoint_resumed": False,
        "checkpoint_reused_cuts": 0,
        "checkpoint_used_psi_full": False,
        "phase1_resumed_partial": False,
        "phase1_psi_partial": float(psi_full),
        "phase1_mechanisms_done": int(run_meta["n_mechanisms"]),
        "phase1_total_mechanisms": int(run_meta["n_mechanisms"]),
        "phase1_eta_seconds": None,
        "phase1_parallel_workers": manifest["execution_profile"][
            "hunter_phase1_workers_per_task"
        ],
        "phase1_chunk_size": int(
            manifest["execution_profile"]["hunter_phase1_chunk_size"]
        ),
        "phase1_shared_memory": bool(
            manifest["execution_profile"]["hunter_shared_memory"]
        ),
        "induced_partition_cache_enabled": False,
        "induced_partition_cache_path": None,
        "induced_partition_cache_stats": None,
        "psi_kernel": (",".join(sorted(kernels)) if kernels else None),
        **provenance,
        **null_fields,
        "defined": True,
        "undefined_reason": None,
        **build_fields,
    }
    _json_dump(run_dir / "final_result.json", payload)
    return payload


def _reduce_all(campaign_dir, fn, label):
    campaign_dir = Path(campaign_dir).resolve()
    manifest = _load_manifest(campaign_dir)
    results, errors = [], []
    for run_index, run_meta in enumerate(manifest.get("runs", [])):
        try:
            results.append(fn(campaign_dir, run_index))
        except (FileNotFoundError, RuntimeError) as exc:
            errors.append(f"{run_meta.get('run_key')}: {exc}")
    if errors:
        raise RuntimeError(
            f"{label} failed for {len(errors)} run(s):\n" + "\n".join(errors)
        )
    return results


def run_phase1_reduce_all(campaign_dir):
    """Phase-1 reduction for every run in one process (merged reducer)."""
    return _reduce_all(campaign_dir, run_phase1_reduce, "phase1-reduce")


def run_cut_reduce_all(campaign_dir):
    """Cut reduction (final IIM) for every run in one process (merged reducer)."""
    return _reduce_all(campaign_dir, run_cut_reduce, "cut-reduce")


def run_reduce_all(campaign_dir):
    start = _timing_start()
    campaign_dir = Path(campaign_dir).resolve()
    p1 = run_phase1_reduce_all(campaign_dir)
    final = run_cut_reduce_all(campaign_dir)
    write_iim_results_table(campaign_dir)
    timing = _timing_finish(start, stage="reduce-all", n_runs=len(final))
    _write_timing(campaign_dir, "reduce-all", "reduce_all", timing)
    return {"phase1": p1, "final": final, "timing": timing}


# Columns of the per-run IIM results table (step-2 IIM output of a campaign).
IIM_RESULTS_COLUMNS = (
    "run_key",
    "subject",
    "session",
    "ts_path",
    "defined",
    "undefined_reason",
    "value",
    "raw",
    "canonical",
    "Psi_full",
    "Psi_mip_preserved",
    "Delta_Psi",
    "IIM_null_n",
    "IIM_null_mean",
    "IIM_null_sd",
    "IIM_z",
    "IIM_null_p",
    "IIM_excess",
    "canonical_calibrated",
    "IIM_null_method",
    "IIM_null_seed",
    "IIM_null_min_shift",
    "IIM_null_failed",
    "IIM_null_undefined_reason",
    "iim_algorithm_version",
    "tpm_estimator",
    "cut_mode",
    "psi_kernel",
    "bearer_nodes",
    "selected_nodes",
    "node_selection_rule",
    "node_selection_degenerate",
    "bins_requested",
    "bins_used",
    "state_budget_policy",
    "budget_adjustments",
    "n_states_observed",
    "n_transitions",
    "code_version",
)


def write_iim_results_table(campaign_dir, out_path=None) -> list[dict]:
    """
    Per-run IIM results with estimator/calibration provenance (tpm_estimator,
    algorithm version, node selection, budget adjustments, null fields) as
    ``iim_results.json`` / ``iim_results.csv`` in the campaign directory, and
    optionally as CSV at ``out_path`` (e.g. next to the step-2 outputs).
    Surrogate runs are not listed; they enter through the null fields.
    """
    campaign_dir = Path(campaign_dir).resolve()
    manifest = _load_manifest(campaign_dir)
    rows = []
    for run_meta in manifest.get("runs", []):
        if bool(run_meta.get("is_null_surrogate", False)):
            continue
        run_dir = _run_artifact_dir(campaign_dir, run_meta["run_key"])
        final_path = run_dir / "final_result.json"
        result = _json_load(final_path) if final_path.exists() else {}
        rec = {**run_meta, **result}
        rows.append({col: rec.get(col) for col in IIM_RESULTS_COLUMNS})
    _json_dump(campaign_dir / "iim_results.json", rows)

    def _cell(value):
        if isinstance(value, (list, tuple, dict)):
            return json.dumps(value)
        return "" if value is None else value

    targets = [campaign_dir / "iim_results.csv"]
    if out_path is not None:
        targets.append(Path(out_path))
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(IIM_RESULTS_COLUMNS))
            writer.writeheader()
            for row in rows:
                writer.writerow({k: _cell(v) for k, v in row.items()})
    return rows


def run_packed_shard(
    campaign_dir,
    stage: str,
    array_index: int,
    shards_per_node=None,
    env=None,
    *,
    hardware_target=None,
):
    """
    Run the shard of a packed launch (``shards_per_node`` ranks per node).
    Ranks beyond the last task of a partially filled node exit without work.
    ``shards_per_node`` defaults to the packing the campaign was built with; a
    different explicit value would skip or duplicate tasks and is rejected.
    ``hardware_target`` is the job's own target (see ``_shard_hardware_backend``).
    """
    campaign_dir = Path(campaign_dir).resolve()
    manifest = _load_manifest(campaign_dir)
    tasks_key = {PHASE1_STAGE: "phase1_tasks", CUT_STAGE: "cut_tasks"}.get(stage)
    if tasks_key is None:
        raise ValueError(
            f"Packed execution supports {PHASE1_STAGE} and {CUT_STAGE}, not '{stage}'."
        )
    built_spn = (manifest.get("scheduler") or {}).get("shards_per_node")
    if shards_per_node is None:
        shards_per_node = int(built_spn or 1)
    elif built_spn is not None and int(shards_per_node) != int(built_spn):
        raise ValueError(
            f"--hunter-shards-per-node {int(shards_per_node)} does not match the "
            f"campaign packing ({int(built_spn)} shards per node); the array index "
            "would map to the wrong shards."
        )
    task_index = resolve_packed_task_index(array_index, shards_per_node, env=env)
    n_tasks = len(manifest.get(tasks_key, []))
    if task_index >= n_tasks:
        log.info(
            "%s: packed slot %d has no task (n_tasks=%d); nothing to do.",
            stage,
            task_index,
            n_tasks,
        )
        return None
    if stage == PHASE1_STAGE:
        return run_phase1_shard(
            campaign_dir, task_index, hardware_target=hardware_target
        )
    return run_cut_shard(campaign_dir, task_index, hardware_target=hardware_target)


def campaign_status(campaign_dir) -> dict:
    """Completed/missing shards and results of a campaign (written to status.json)."""
    campaign_dir = Path(campaign_dir).resolve()
    manifest = _load_manifest(campaign_dir)
    runs_by_key = {str(r["run_key"]): r for r in manifest.get("runs", [])}
    # 'complete' means a resubmission would skip the shard (same identity check).
    code_version = _runtime_code_version()
    status = {
        "phase1-shard": {"complete": [], "missing": []},
        "cut-shard": {"complete": [], "missing": []},
    }
    for stage, key, sub in (
        (PHASE1_STAGE, "phase1_tasks", "phase1_shards"),
        (CUT_STAGE, "cut_tasks", "cut_shards"),
    ):
        for i, task in enumerate(manifest.get(key, [])):
            path = (
                _run_artifact_dir(campaign_dir, task["run_key"])
                / sub
                / f"shard_{int(task['task_index']):04d}.json"
            )
            identity = _task_identity(
                runs_by_key.get(str(task["run_key"]), {}), task, stage, code_version
            )
            ok = _stored_completion(path, identity) is not None
            status[stage]["complete" if ok else "missing"].append(int(i))

    def _final_current(run_meta) -> bool:
        run_dir = _run_artifact_dir(campaign_dir, run_meta["run_key"])
        path = run_dir / "final_result.json"
        if not path.exists():
            return False
        try:
            rec = _json_load(path)
        except Exception:
            return False
        return rec.get("problem_digest") == run_meta.get("problem_digest")

    finals = [_final_current(r) for r in manifest.get("runs", [])]
    summary = {
        "campaign_dir": str(campaign_dir),
        "n_runs": len(finals),
        "final_results": int(sum(finals)),
        "stages": {
            stage: {
                "complete": len(v["complete"]),
                "missing": len(v["missing"]),
                "missing_indices": v["missing"],
            }
            for stage, v in status.items()
        },
        "timing": summarize_campaign_timing(campaign_dir, write=False),
    }
    _json_dump(campaign_dir / "status.json", summary)
    return summary


def summarize_campaign_timing(campaign_dir, write=True) -> dict:
    """Aggregate per-task timing records (wall time, CPU time, peak RSS) per stage."""
    campaign_dir = Path(campaign_dir).resolve()
    out = {}
    root = campaign_dir / "timing"
    if root.exists():
        for stage_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            walls, rss, cpu, cpu_all = [], [], [], []
            for path in sorted(stage_dir.glob("*.json")):
                try:
                    rec = _json_load(path)
                except Exception:
                    continue
                walls.append(float(rec.get("wall_seconds") or 0.0))
                cpu.append(float(rec.get("process_cpu_seconds") or 0.0))
                # parent + worker processes (older records: parent only)
                cpu_all.append(
                    float(rec.get("cpu_seconds", rec.get("process_cpu_seconds")) or 0.0)
                )
                if rec.get("peak_rss_mb") is not None:
                    rss.append(float(rec["peak_rss_mb"]))
            if walls:
                out[stage_dir.name] = {
                    "n_tasks": len(walls),
                    "wall_seconds_total": float(sum(walls)),
                    "wall_seconds_mean": float(np.mean(walls)),
                    "wall_seconds_max": float(max(walls)),
                    "process_cpu_seconds_total": float(sum(cpu)),
                    "cpu_seconds_total": float(sum(cpu_all)),
                    "peak_rss_mb_max": (None if not rss else float(max(rss))),
                }
    if write:
        _json_dump(campaign_dir / "timing_summary.json", out)
    return out


def collect_iim_results_by_path(campaign_dir):
    campaign_dir = Path(campaign_dir).resolve()
    manifest = _load_manifest(campaign_dir)
    out = {}
    for run_meta in manifest.get("runs", []):
        is_null = bool(run_meta.get("is_null_surrogate", False))
        if is_null or not run_meta.get("ts_path"):
            # Surrogate runs only feed the calibration of their real run.
            continue
        run_dir = _run_artifact_dir(campaign_dir, run_meta["run_key"])
        final_path = run_dir / "final_result.json"
        if not final_path.exists():
            raise FileNotFoundError(f"Missing final IIM result: {final_path}")
        result = _json_load(final_path)
        # Undefined runs have no digest; for defined runs the result must come
        # from this build of the campaign (not a stale earlier reduction).
        if result.get("problem_digest") != run_meta.get("problem_digest"):
            raise RuntimeError(
                f"Final IIM result {final_path} does not belong to the current "
                "campaign build; run the reduce-all stage again."
            )
        out[str(run_meta["ts_path"])] = result
        # Also key by the path as discovered (may differ from the resolved path
        # when the output directory is reached through a symlink).
        if run_meta.get("ts_path_input"):
            out.setdefault(str(run_meta["ts_path_input"]), result)
    return out
