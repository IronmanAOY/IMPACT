"""
MPC-Bench runner: simulate systems, score them with the public estimators and
write JSONL/CSV results with provenance.

Designs: ``factorial`` (2^5 cells x seeds), ``sweep`` (dose-response, 10
levels per knob x seeds), ``witnesses`` (``witnesses.yaml`` x seeds) and
``timing`` (simulation runtimes per generator). Tasks run in a process pool;
with ``--n-shards`` / ``--shard-index`` a run is split deterministically
(``--pbs-template`` writes a PBS Pro array script for HLRS Hunter).

Seed policy: development seeds are 0-999; confirmatory seeds start at 10000
and family C (held out) and confirmatory seeds run only with
``--confirmatory``, which requires a clean git tree (no modified tracked
files, no untracked files under ``src/`` or ``scripts/``), optionally a
freeze tag that is an ancestor of HEAD, and records the commit.

The estimators are imported lazily; ``impact_pipeline.evidence`` (verdicts)
and optional estimator modes are used when available and recorded.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import logging
import math
import multiprocessing
import os
import shlex
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

from impact_pipeline.bench import BENCH_VERSION
from impact_pipeline.bench.factorial import (
    FAMILY_GENERATOR,
    BenchTask,
    factorial_tasks,
)
from impact_pipeline.bench.generators import (
    GENERATOR_VERSION,
    PRINCIPLES,
    config_from_dict,
    knobs_from_dict,
    make_system,
    summarise_oracle,
)
from impact_pipeline.bench.sweeps import SWEEP_KNOBS, sweep_tasks

log = logging.getLogger("impact_pipeline.bench")

RESULT_SCHEMA = "mpc-bench-result/1"
DEV_SEED_MAX = 999
CONFIRMATORY_SEED_START = 10000
REPO_ROOT = Path(__file__).resolve().parents[3]
DESIGNS = ("factorial", "sweep", "witnesses", "timing")
RESULTS_JSONL = "results.jsonl"
RESULTS_CSV = "results.csv"
MANIFEST = "run_manifest.json"
LZC_SEGMENT_SAMPLES = 1000


# ---------------------------------------------------------------------------
# Seed policy and confirmatory guard
# ---------------------------------------------------------------------------


def seed_set(seed: int) -> str:
    seed = int(seed)
    if 0 <= seed <= DEV_SEED_MAX:
        return "dev"
    if seed >= CONFIRMATORY_SEED_START:
        return "confirmatory"
    return "reserved"


def check_seed_policy(tasks: Sequence[BenchTask], confirmatory: bool) -> None:
    """Refuse tasks that break the development / confirmatory separation."""
    for t in tasks:
        s = seed_set(t.seed)
        if s == "reserved":
            raise ValueError(
                f"{t.task_id}: seed {t.seed} is reserved (use 0-{DEV_SEED_MAX} "
                f"or >= {CONFIRMATORY_SEED_START})"
            )
        if confirmatory and s != "confirmatory":
            raise ValueError(
                f"{t.task_id}: confirmatory runs use seeds >= "
                f"{CONFIRMATORY_SEED_START}"
            )
        if not confirmatory and s == "confirmatory":
            raise ValueError(
                f"{t.task_id}: seeds >= {CONFIRMATORY_SEED_START} are "
                "confirmatory; pass --confirmatory after the code freeze"
            )
        if not confirmatory and t.family == "C":
            raise ValueError(
                f"{t.task_id}: family C is held out; it runs only with "
                "--confirmatory after the code freeze"
            )


def _git(repo_root: Path, *args) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except Exception:
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def confirmatory_guard(repo_root=REPO_ROOT, freeze_tag: Optional[str] = None) -> dict:
    """
    Code identity for a confirmatory run. Raises ``RuntimeError`` unless the
    checkout is a git work tree with no modified tracked files and no
    untracked files under ``src/`` or ``scripts/``, and ``freeze_tag`` (the
    code-freeze tag; required) exists, is an ancestor of (or equal to) HEAD
    and has the same ``src/`` and ``scripts/`` trees as HEAD (later commits
    may only touch other paths, e.g. documentation or results).
    """
    from impact_pipeline.provenance import collect_code_version

    root = Path(repo_root)
    info = collect_code_version(root)
    if not info.get("git_sha"):
        raise RuntimeError(
            "--confirmatory needs a git checkout (commit could not be read)"
        )
    if info.get("git_dirty") is not False:
        raise RuntimeError(
            "--confirmatory needs a clean git tree (tracked files modified)"
        )
    untracked = _git(
        root, "status", "--porcelain", "--untracked-files=all", "--", "src", "scripts"
    )
    if untracked is None or untracked.strip():
        raise RuntimeError(
            "--confirmatory needs a clean git tree (untracked or modified "
            "files under src/ or scripts/)"
        )
    info["tags_at_head"] = (_git(root, "tag", "--points-at", "HEAD") or "").split()
    if not freeze_tag:
        raise RuntimeError(
            "--confirmatory needs the code-freeze tag (--freeze-tag): "
            "confirmatory seeds and family C run only on frozen code"
        )
    tag_sha = _git(root, "rev-list", "-n", "1", str(freeze_tag))
    if not tag_sha:
        raise RuntimeError(f"freeze tag {freeze_tag!r} not found")

    def _git_ok(*args) -> bool:
        return (
            subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                check=False,
                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            ).returncode
            == 0
        )

    if not _git_ok("merge-base", "--is-ancestor", tag_sha, "HEAD"):
        raise RuntimeError(f"HEAD does not descend from freeze tag {freeze_tag!r}")
    if not _git_ok("diff", "--quiet", tag_sha, "HEAD", "--", "src", "scripts"):
        raise RuntimeError(
            f"src/ or scripts/ changed since freeze tag {freeze_tag!r}: "
            "confirmatory runs must use the frozen code"
        )
    info["freeze_tag"] = str(freeze_tag)
    info["freeze_tag_sha"] = tag_sha
    info["confirmatory"] = True
    return info


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


def witness_tasks(
    seeds: Iterable[int],
    family: str = "A",
    witness_ids=None,
    config: Optional[dict] = None,
    catalogue: Optional[dict] = None,
) -> List[BenchTask]:
    """Witness tasks; the patchwork is scored with system and per-principle bearers."""
    from impact_pipeline.bench.witnesses import load_witnesses

    cat = catalogue if catalogue is not None else load_witnesses()
    family = str(family).upper()
    wanted = None if witness_ids is None else set(witness_ids)
    tasks = []
    seeds = [int(s) for s in seeds]
    for w in cat["witnesses"]:
        if wanted is not None and w["id"] not in wanted:
            continue
        if family not in w["families"]:
            continue
        modes = (
            ("system", "principle") if w["generator"] == "patchwork" else ("system",)
        )
        for mode in modes:
            for seed in seeds:
                suffix = "" if mode == "system" else "-principle"
                tasks.append(
                    BenchTask(
                        task_id=f"witness-{family}-{w['id']}{suffix}-s{seed:05d}",
                        design="witnesses",
                        family=family,
                        generator=w["generator"],
                        cell_id=w["id"],
                        knobs=dict(w.get("knobs") or {}),
                        seed=seed,
                        config=dict(config or {}),
                        witness_id=w["id"],
                        bearer_mode=mode,
                    )
                )
    return tasks


def build_system(task: BenchTask):
    """Realise the system of a task (knobs are applied on top of the nominal)."""
    cfg = config_from_dict(task.config)
    kn = knobs_from_dict(task.knobs)
    gen = task.generator
    kw = {}
    if gen == "family_a" and task.family == "C":
        gen = FAMILY_GENERATOR["C"]
    if gen == "patchwork":
        kw["dynamics"] = "rate" if task.family == "A" else "stuart_landau"
    return make_system(gen, kn, cfg, task.seed, template_family=task.family, **kw)


def _null_seed(seed: int) -> int:
    return int(seed) * 1000 + 17


def _sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj


def _markers(system) -> dict:
    from impact_pipeline.bench.lz import lzc
    from impact_pipeline.bench.phiid_gaussian import phi_r_marker

    out = {}
    t0 = time.perf_counter()
    out["LZc"] = lzc(system.ts, segment_samples=LZC_SEGMENT_SAMPLES)
    out["LZc_seconds"] = round(time.perf_counter() - t0, 4)
    t0 = time.perf_counter()
    try:
        out["PhiR_bits"] = phi_r_marker(system.ts, system.meta.get("iim_macro_nodes"))
    except Exception as exc:  # pragma: no cover - numerical corner cases
        out["PhiR_bits"] = float("nan")
        out["PhiR_reason"] = str(exc)
    out["PhiR_seconds"] = round(time.perf_counter() - t0, 4)
    return out


def run_task(
    task: BenchTask,
    metrics: Sequence[str] = PRINCIPLES,
    null_surrogates: int = 0,
    params: Optional[dict] = None,
    provenance: Optional[dict] = None,
    markers: bool = True,
    with_verdict: bool = True,
) -> dict:
    """Simulate and score one task; errors are recorded, not raised."""
    from impact_pipeline.bench.export import evidence_verdict, run_in_memory

    rec = {
        "schema": RESULT_SCHEMA,
        "bench_version": BENCH_VERSION,
        "generator_version": GENERATOR_VERSION,
        **task.to_dict(),
        "seed_set": seed_set(task.seed),
        "null_surrogates": int(null_surrogates),
        "null_seed": _null_seed(task.seed),
        "metrics": list(metrics),
        "provenance": provenance or {},
        "status": "ok",
    }
    t_all = time.perf_counter()
    try:
        t0 = time.perf_counter()
        system = build_system(task)
        rec["timing"] = {"simulate_s": round(time.perf_counter() - t0, 4)}
        rec["system"] = {
            "family": system.meta.get("family"),
            "n_nodes": system.n_nodes,
            "n_time": system.n_time,
            "dt": system.dt,
        }
        rec["intended_bits"] = list(system.oracle.get("intended_bits", []))
        rec["oracle_summary"] = summarise_oracle(system)
        res = run_in_memory(
            system,
            metrics=metrics,
            params=params,
            null_surrogates=null_surrogates,
            null_seed=_null_seed(task.seed),
            bearer_mode=task.bearer_mode,
        )
        rec["components"] = res["components"]
        rec["estimator_modes"] = res["estimator_modes"]
        rec["timing"]["estimators_s"] = {
            p: c["seconds"] for p, c in res["components"].items()
        }
        if markers:
            rec["markers"] = _markers(system)
        if with_verdict:
            protocol = (
                f"{BENCH_VERSION}:{GENERATOR_VERSION}:nulls{int(null_surrogates)}"
            )
            rec["verdict"] = evidence_verdict(res, system.meta, protocol_id=protocol)
    except Exception as exc:
        rec["status"] = "error"
        rec["error"] = f"{type(exc).__name__}: {exc}"
    rec.setdefault("timing", {})["total_s"] = round(time.perf_counter() - t_all, 4)
    return _sanitize(rec)


def flatten_record(rec: dict) -> dict:
    """One CSV row per record (components as <P>_estimate, <P>_z, ...)."""
    row = {
        k: rec.get(k)
        for k in (
            "task_id",
            "design",
            "family",
            "generator",
            "cell_id",
            "witness_id",
            "sweep_knob",
            "sweep_level",
            "bearer_mode",
            "seed",
            "seed_set",
            "null_surrogates",
            "status",
            "error",
        )
    }
    bits = rec.get("intended_bits") or []
    row["intended_bits"] = ("b" + "".join(str(b) for b in bits)) if bits else None
    for k, v in (rec.get("knobs") or {}).items():
        row[f"knob_{k}"] = v
    for p, c in (rec.get("components") or {}).items():
        est, nm, nsd = c.get("estimate"), c.get("null_mean"), c.get("null_sd")
        row[f"{p}_estimate"] = est
        row[f"{p}_value"] = c.get("value")
        row[f"{p}_null_mean"] = nm
        row[f"{p}_null_sd"] = nsd
        row[f"{p}_n_null"] = c.get("n_null")
        row[f"{p}_null_family"] = c.get("null_family")
        row[f"{p}_null_impl"] = c.get("null_impl")
        z = None
        if est is not None and nm is not None and nsd not in (None, 0):
            z = (est - nm) / nsd
        row[f"{p}_z"] = z
        row[f"{p}_defined"] = c.get("defined")
        row[f"{p}_reason"] = c.get("reason")
        row[f"{p}_seconds"] = c.get("seconds")
    for k, v in (rec.get("markers") or {}).items():
        row[f"marker_{k}"] = v
    for k, v in (rec.get("oracle_summary") or {}).items():
        if not isinstance(v, (list, dict)):
            row[f"oracle_{k}"] = v
    verdict = rec.get("verdict") or {}
    row["verdict"] = verdict.get("verdict")
    row["verdict_reasons"] = ";".join(verdict.get("reasons") or [])
    timing = rec.get("timing") or {}
    row["simulate_s"] = timing.get("simulate_s")
    row["total_s"] = timing.get("total_s")
    prov = rec.get("provenance") or {}
    code = prov.get("code_version") or {}
    row["git_sha"] = code.get("git_sha")
    row["git_dirty"] = code.get("git_dirty")
    return row


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _worker_init():
    """Pool worker: silence estimator logging and warnings (worker-local)."""
    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore")


@contextlib.contextmanager
def _quiet_in_process():
    """Serial runs: silence estimator logging/warnings, then restore both."""
    previous = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        logging.disable(previous)


@contextlib.contextmanager
def _single_threaded_children():
    """One BLAS/OpenMP thread per pool worker (inherited at spawn); the parent
    environment is restored afterwards. Explicit user settings are kept."""
    saved = {v: os.environ.get(v) for v in _THREAD_VARS}
    for v in _THREAD_VARS:
        os.environ.setdefault(v, "1")
    try:
        yield
    finally:
        for v, val in saved.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val


def load_done(jsonl_path: Path) -> set:
    done = set()
    p = Path(jsonl_path)
    if not p.exists():
        return done
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") == "ok":
                done.add(rec.get("task_id"))
    return done


def shard(
    tasks: Sequence[BenchTask], shard_index: int = 0, n_shards: int = 1
) -> List[BenchTask]:
    n_shards = int(n_shards)
    shard_index = int(shard_index)
    if n_shards < 1 or not 0 <= shard_index < n_shards:
        raise ValueError("need 0 <= shard_index < n_shards")
    ordered = sorted(tasks, key=lambda t: t.task_id)
    return ordered[shard_index::n_shards]


def collect_provenance(
    args: Optional[dict] = None, code_version: Optional[dict] = None
) -> dict:
    from impact_pipeline.bench.export import BENCH_ESTIMATOR_PARAMS
    from impact_pipeline.provenance import (
        collect_code_version,
        collect_runtime_versions,
    )

    return {
        "code_version": code_version or collect_code_version(REPO_ROOT),
        "runtime": collect_runtime_versions(),
        "bench_version": BENCH_VERSION,
        "generator_version": GENERATOR_VERSION,
        "estimator_params": BENCH_ESTIMATOR_PARAMS,
        "args": args or {},
        "created_unix": time.time(),
    }


def run_tasks(
    tasks: Sequence[BenchTask],
    out_dir,
    n_workers: int = 1,
    metrics: Sequence[str] = PRINCIPLES,
    null_surrogates: int = 0,
    params: Optional[dict] = None,
    resume: bool = True,
    provenance: Optional[dict] = None,
    markers: bool = True,
    confirmatory: bool = False,
) -> List[dict]:
    """
    Run tasks (skipping task ids already completed in ``out_dir``) and append
    records to ``results.jsonl``; ``results.csv`` and ``run_manifest.json`` are
    rewritten at the end. Returns the records produced by this call.
    ``confirmatory=True`` requires provenance whose ``code_version`` came from
    :func:`confirmatory_guard` (frozen, clean code).
    """
    check_seed_policy(tasks, confirmatory)
    prov = provenance if provenance is not None else collect_provenance()
    if confirmatory and not (prov.get("code_version") or {}).get("confirmatory"):
        raise RuntimeError(
            "confirmatory runs need provenance from confirmatory_guard "
            "(clean checkout of the code-freeze tag)"
        )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / RESULTS_JSONL
    done = load_done(jsonl) if resume else set()
    todo = [t for t in tasks if t.task_id not in done]
    small_prov = {
        "code_version": prov.get("code_version"),
        "bench_version": BENCH_VERSION,
        "confirmatory": bool(confirmatory),
    }
    records = []
    t_start = time.time()
    with open(jsonl, "a", encoding="utf-8") as fh:

        def _emit(rec):
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
            fh.flush()
            records.append(rec)
            log.info(
                "MPC-Bench %s: %s (%.2f s)",
                rec["task_id"],
                rec["status"],
                (rec.get("timing") or {}).get("total_s", float("nan")),
            )

        if int(n_workers) <= 1:
            with _quiet_in_process():
                for t in todo:
                    _emit(
                        run_task(
                            t, metrics, null_surrogates, params, small_prov, markers
                        )
                    )
        else:
            # Spawned (not forked) workers start a fresh interpreter, so the
            # one-thread BLAS/OpenMP settings take effect on every platform
            # (a forked child inherits the parent's initialised BLAS pool).
            with (
                _single_threaded_children(),
                concurrent.futures.ProcessPoolExecutor(
                    max_workers=int(n_workers),
                    initializer=_worker_init,
                    mp_context=multiprocessing.get_context("spawn"),
                ) as ex,
            ):
                futs = [
                    ex.submit(
                        run_task,
                        t,
                        metrics,
                        null_surrogates,
                        params,
                        small_prov,
                        markers,
                    )
                    for t in todo
                ]
                for fut in concurrent.futures.as_completed(futs):
                    _emit(fut.result())
    manifest = dict(prov)
    manifest.update(
        {
            "n_tasks": len(tasks),
            "n_skipped_done": len(tasks) - len(todo),
            "n_run": len(records),
            "n_errors": sum(r["status"] != "ok" for r in records),
            "wall_s": round(time.time() - t_start, 3),
            "task_ids": [t.task_id for t in tasks],
            "confirmatory": bool(confirmatory),
        }
    )
    (out / MANIFEST).write_text(
        json.dumps(_sanitize(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    merge_results(out)
    return records


def merge_results(out_dir) -> "object":
    """Merge every ``results.jsonl`` below ``out_dir`` (shards included) into
    ``out_dir/results.csv`` (latest record per task id). Returns the DataFrame."""
    import pandas as pd

    out = Path(out_dir)
    latest = {}
    for path in sorted(out.rglob(RESULTS_JSONL)):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    latest[rec["task_id"]] = rec
    rows = [flatten_record(latest[k]) for k in sorted(latest)]
    df = pd.DataFrame(rows)
    df.to_csv(out / RESULTS_CSV, index=False)
    return df


def pbs_array_script(
    cli_args: Sequence[str],
    n_shards: int,
    walltime: str = "24:00:00",
    queue: Optional[str] = None,
    group_list: Optional[str] = None,
    workers: int = 96,
    job_name: str = "mpc_bench",
    venv: str = "${BENCH_VENV:?set BENCH_VENV}",
) -> str:
    """
    PBS Pro array script for HLRS Hunter (one whole mi300a node per subjob;
    arrays need >= 2 subjobs, so ``n_shards == 1`` gives a plain job). Output
    must go to a workspace (``BENCH_OUT_DIR``), not HOME.
    """
    from impact_pipeline.execution_profiles import HunterPBSProfile

    prof = HunterPBSProfile()
    n_shards = int(n_shards)
    if n_shards < 1:
        raise ValueError("n_shards must be >= 1")
    lines = [
        "#!/bin/bash",
        f"#PBS -N {job_name}",
        f"#PBS -l select=1:node_type={prof.node_type}",
        f"#PBS -l walltime={walltime}",
    ]
    if prof.workspace_resource:
        lines.append(f"#PBS -l {prof.workspace_resource}")
    if queue:
        lines.append(f"#PBS -q {queue}")
    if group_list:
        lines.append(f"#PBS -W group_list={group_list}")
    lines.append("#PBS -j oe")
    if n_shards >= 2:
        lines += [f"#PBS -J 0-{n_shards - 1}", "#PBS -r y"]
    index = '"${PBS_ARRAY_INDEX:-0}"' if n_shards >= 2 else "0"
    args = " ".join(shlex.quote(a) for a in cli_args)
    lines += [
        "# MPC-Bench shard runner (CPU-bound; one process pool per node).",
        "set -euo pipefail",
        'cd "$PBS_O_WORKDIR"',
        "module load cray-python",
        f"source {venv}/bin/activate",
        'OUT_DIR="${BENCH_OUT_DIR:?set BENCH_OUT_DIR to a workspace path (ws_find)}"',
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
        f"python3 scripts/run_bench.py {args} "
        f'--out "$OUT_DIR/shard-$(printf %04d {index})" '
        f"--workers {int(workers)} --n-shards {n_shards} --shard-index {index}",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runtime measurement
# ---------------------------------------------------------------------------


def timing_report(
    seeds: Sequence[int] = (0, 1, 2),
    config: Optional[dict] = None,
    binary_steps: int = 10000,
) -> dict:
    """Wall-clock simulation time per generator (mean, sd, n) on this machine."""
    from impact_pipeline.bench.generators import (
        BINARY_NETWORK_KINDS,
        family_b_network,
        sample_binary_trajectory,
        simulate_family_a,
        simulate_family_c,
    )
    from impact_pipeline.bench.patchwork import simulate_patchwork

    cfg = config_from_dict(config)
    rows = {}

    def _time(name, fn):
        vals = []
        shape = None
        for s in seeds:
            t0 = time.perf_counter()
            out = fn(int(s))
            vals.append(time.perf_counter() - t0)
            shape = getattr(out, "shape", None) or getattr(
                getattr(out, "ts", None), "shape", None
            )
        rows[name] = {
            "mean_s": float(np.mean(vals)),
            "sd_s": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n": len(vals),
            "shape": list(shape) if shape is not None else None,
        }

    _time("family_A", lambda s: simulate_family_a(None, cfg, s))
    _time("family_C", lambda s: simulate_family_c(None, cfg, s))
    _time("patchwork", lambda s: simulate_patchwork(None, cfg, s))
    for kind in BINARY_NETWORK_KINDS:
        net = family_b_network(kind, n=4)
        _time(
            f"family_B_{kind}_n4_T{binary_steps}",
            lambda s, net=net: sample_binary_trajectory(net, binary_steps, seed=s),
        )
    import platform

    return {
        "machine": platform.platform(),
        "processor": platform.processor(),
        "config": cfg.to_dict(),
        "seeds": [int(s) for s in seeds],
        "timings": rows,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_seeds(text: str) -> List[int]:
    """'0-19' or '0,3,7' or '10000-10019'."""
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    if not out:
        raise ValueError(f"no seeds in {text!r}")
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="run_bench.py", description=__doc__.split("\n\n")[0]
    )
    ap.add_argument("design", choices=DESIGNS)
    ap.add_argument("--family", default="A", choices=("A", "C"))
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument(
        "--out", default=None, help="output directory (required except timing)"
    )
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--metrics", default=",".join(PRINCIPLES))
    ap.add_argument("--null-surrogates", type=int, default=0)
    ap.add_argument(
        "--config", default=None, help="JSON object (or path) of AgentConfig overrides"
    )
    ap.add_argument(
        "--params",
        default=None,
        help="JSON object (or path) of estimator parameter overrides",
    )
    ap.add_argument("--cells", default=None, help="factorial: comma-separated cell ids")
    ap.add_argument(
        "--knobs", default=",".join(SWEEP_KNOBS), help="sweep: knobs to sweep"
    )
    ap.add_argument("--levels", type=int, default=10, help="sweep: levels per knob")
    ap.add_argument("--witnesses", default=None, help="witnesses: comma-separated ids")
    ap.add_argument(
        "--no-markers", action="store_true", help="skip LZc / Phi_R markers"
    )
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--confirmatory", action="store_true")
    ap.add_argument("--freeze-tag", default=None)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument(
        "--pbs-template",
        default=None,
        help="write a PBS Pro array script for --n-shards shards and exit",
    )
    ap.add_argument("--pbs-walltime", default="24:00:00")
    ap.add_argument("--pbs-queue", default=None)
    ap.add_argument("--pbs-group", default=None)
    ap.add_argument("--list", action="store_true", help="print the task ids and exit")
    return ap


def _json_arg(text):
    if text is None:
        return None
    p = Path(text)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(text)


def make_tasks(args) -> List[BenchTask]:
    seeds = parse_seeds(args.seeds)
    config = _json_arg(args.config)
    if args.design == "factorial":
        cells = None if not args.cells else [c.strip() for c in args.cells.split(",")]
        return factorial_tasks(seeds, family=args.family, config=config, cells=cells)
    if args.design == "sweep":
        knobs = [k.strip() for k in args.knobs.split(",") if k.strip()]
        return sweep_tasks(
            seeds, knobs=knobs, family=args.family, n_levels=args.levels, config=config
        )
    if args.design == "witnesses":
        ids = (
            None
            if not args.witnesses
            else [w.strip() for w in args.witnesses.split(",")]
        )
        return witness_tasks(seeds, family=args.family, witness_ids=ids, config=config)
    raise ValueError(f"design {args.design!r} has no tasks")


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    try:
        return _main(args, argv)
    except (ValueError, RuntimeError) as exc:
        print(f"run_bench: error: {exc}", file=sys.stderr)
        return 2


def _main(args, argv: List[str]) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.design == "timing":
        rep = timing_report(parse_seeds(args.seeds), _json_arg(args.config))
        text = json.dumps(_sanitize(rep), indent=2, sort_keys=True)
        if args.out:
            Path(args.out).mkdir(parents=True, exist_ok=True)
            (Path(args.out) / "timing.json").write_text(text, encoding="utf-8")
        print(text)
        return 0
    if args.pbs_template:
        keep = []
        skip_next = False
        drop = {
            "--pbs-template",
            "--out",
            "--workers",
            "--n-shards",
            "--shard-index",
            "--pbs-walltime",
            "--pbs-queue",
            "--pbs-group",
        }
        for a in argv:
            if skip_next:
                skip_next = False
                continue
            key = a.split("=", 1)[0]
            if key in drop:
                skip_next = "=" not in a
                continue
            keep.append(a)
        script = pbs_array_script(
            keep,
            args.n_shards,
            walltime=args.pbs_walltime,
            queue=args.pbs_queue,
            group_list=args.pbs_group,
        )
        Path(args.pbs_template).write_text(script, encoding="utf-8")
        print(f"PBS array script written to {args.pbs_template}")
        return 0
    tasks = make_tasks(args)
    check_seed_policy(tasks, args.confirmatory)
    tasks = shard(tasks, args.shard_index, args.n_shards)
    if args.list:
        for t in tasks:
            print(t.task_id)
        return 0
    if not args.out:
        raise SystemExit("--out is required")
    code = confirmatory_guard(REPO_ROOT, args.freeze_tag) if args.confirmatory else None
    prov = collect_provenance(vars(args), code_version=code)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    recs = run_tasks(
        tasks,
        args.out,
        n_workers=args.workers,
        metrics=metrics,
        null_surrogates=args.null_surrogates,
        params=_json_arg(args.params),
        resume=not args.no_resume,
        provenance=prov,
        markers=not args.no_markers,
        confirmatory=args.confirmatory,
    )
    n_err = sum(r["status"] != "ok" for r in recs)
    print(
        f"MPC-Bench: {len(recs)} task(s) run, {n_err} error(s); results in {args.out}"
    )
    return 1 if n_err else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
