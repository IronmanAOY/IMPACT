#!/usr/bin/env python
"""
IIM validation on MPC-Bench family B: binary networks with exact TPMs.

For every network kind (``bench.generators.BINARY_NETWORK_KINDS``: independent,
ring, all_to_all, feedforward_star, xor_loop, hidden_driver; ``--n`` units)
and cut mode (``bidirectional`` / ``directional``), the exact IIM of the
generating TPM (``compute_IIM_from_tpm``, state weights = the stationary
distribution, the default) is compared with the sampled estimator
``compute_IIM`` (binary data, ``bins=2``, ``lag_trs=1``, the default
``node_shrinkage`` TPM estimator) on trajectories of ``--T`` steps drawn from
the network's own dynamics, with ``--null-surrogates`` circular-shift
surrogates (``Delta_Psi`` in bits, its null mean / SD and the calibrated
margin ``iim_z = (Delta_Psi - null_mean) / null_sd``). An optional coupling
sweep (``--sweep-kind``, ``--couplings``, at the largest ``T``) gives the
dose-response of the calibrated excess. The hidden-driver network's exact
TPM is interventional (driver marginalised) and not conditionally
independent, so its exact directional value is undefined (recorded).

Outputs (``--out``): ``iim_validation.jsonl`` (one record per run, appended:
the run is resumable and skips completed task ids), ``iim_validation.csv``
(the Figure-6 table: ``system``, ``n_time``, ``seed``, ``cut_mode``,
``delta_psi_est``, ``delta_psi_exact``, ``null_mean``, ``null_sd``,
``iim_z``, ``coupling``), ``iim_validation_exact.csv`` and
``iim_validation.json`` (parameters, provenance).

Seed policy (spec v2, V2-6): development seeds 0-999; confirmatory seeds
>= 10000 only with ``--confirmatory`` and the code-freeze tag
(``--freeze-tag``, checked by ``bench.run_bench.confirmatory_guard``).

Example::

    python scripts/iim_validation.py --out outputs/iim_validation \\
        --T 1000,3000,10000 --seeds 0-4 --null-surrogates 19 --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

VALIDATION_VERSION = "iim-validation/1.0.0"
CUT_MODES = ("bidirectional", "directional")
RESULTS_JSONL = "iim_validation.jsonl"
DEFAULT_COUPLINGS = (0.0, 0.15, 0.3, 0.45, 0.6, 0.9)


def _seed(*keys) -> int:
    text = "|".join(str(k) for k in keys)
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def exact_iim(kind, n, cut_mode, coupling=None):
    """Exact IIM of a family-B network (NaN with a reason when undefined)."""
    from impact_pipeline import mpc_metrics as mm
    from impact_pipeline.bench.generators import family_b_network

    kw = {} if coupling is None else {"coupling": float(coupling)}
    net = family_b_network(kind, n=int(n), **kw)
    try:
        d = mm.compute_IIM_from_tpm(net.tpm, cut_mode=cut_mode, return_details=True)
    except ValueError as exc:
        return {"delta_psi_exact": float("nan"), "exact_reason": str(exc)}
    return {
        "delta_psi_exact": float(d["Delta_Psi"]),
        "psi_full_exact": float(d["Psi_full"]),
        "exact_reason": None,
    }


def tasks_for(kinds, n, n_times, seeds, cut_modes, sweep_kind=None, couplings=()):
    tasks = []
    for kind in kinds:
        for t in n_times:
            for cm in cut_modes:
                for s in seeds:
                    tasks.append(
                        {
                            "task_id": f"{kind}-n{n}-T{t}-{cm}-s{s:05d}",
                            "kind": kind,
                            "n": int(n),
                            "n_time": int(t),
                            "cut_mode": cm,
                            "seed": int(s),
                            "coupling": None,
                        }
                    )
    if sweep_kind:
        t = int(max(n_times))
        for cp in couplings:
            for cm in cut_modes:
                for s in seeds:
                    tasks.append(
                        {
                            "task_id": (
                                f"sweep-{sweep_kind}-n{n}-T{t}-c{cp:g}-"
                                f"{cm}-s{s:05d}"
                            ),
                            "kind": sweep_kind,
                            "n": int(n),
                            "n_time": t,
                            "cut_mode": cm,
                            "seed": int(s),
                            "coupling": float(cp),
                        }
                    )
    return tasks


def run_task(task, null_surrogates, exact_cache=None) -> dict:
    """Sample one trajectory and compute the calibrated sampled IIM
    (estimator logging and warnings are silenced for the call only)."""
    previous = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _run_task(task, null_surrogates)
    finally:
        logging.disable(previous)


def _run_task(task, null_surrogates) -> dict:
    from impact_pipeline import mpc_metrics as mm
    from impact_pipeline.bench.generators import (
        family_b_network,
        sample_binary_trajectory,
    )

    kw = {} if task["coupling"] is None else {"coupling": task["coupling"]}
    net = family_b_network(task["kind"], n=task["n"], **kw)
    traj_seed = _seed(
        "iim_validation",
        task["kind"],
        task["n"],
        task["n_time"],
        task["coupling"],
        task["seed"],
    )
    ts = sample_binary_trajectory(net, task["n_time"], seed=traj_seed).T.astype(float)
    t0 = time.perf_counter()
    rec = dict(task)
    try:
        d = mm.compute_IIM(
            ts,
            bins=2,
            lag_trs=1,
            return_details=True,
            null_surrogates=int(null_surrogates),
            null_seed=_seed("iim_validation_null", traj_seed),
            cut_mode=task["cut_mode"],
            progress_log_every_cuts=10**9,
        )
        est = float(d.get("Delta_Psi", np.nan))
        nm = float(d.get("Delta_Psi_null_mean", np.nan))
        nsd = float(d.get("Delta_Psi_null_sd", np.nan))
        rec.update(
            {
                "delta_psi_est": est,
                "null_mean": nm,
                "null_sd": nsd,
                "n_null": int(d.get("IIM_null_n", 0) or 0),
                "iim_z": (est - nm) / nsd if nsd > 0 else float("nan"),
                "psi_full_est": float(d.get("Psi_full", np.nan)),
                "status": "ok",
                "error": None,
            }
        )
    except Exception as exc:  # noqa: BLE001 - recorded, not raised
        rec.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    rec["traj_seed"] = traj_seed
    rec["seconds"] = round(time.perf_counter() - t0, 3)
    ex = exact_iim(task["kind"], task["n"], task["cut_mode"], task["coupling"])
    rec.update(ex)
    return rec


def _done(path):
    out = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("status") == "ok":
                    out.add(r["task_id"])
    return out


def _jsonable(v):
    if isinstance(v, (np.floating, float)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    return v


def run(
    out_dir,
    *,
    kinds=None,
    n=4,
    n_times=(1000, 3000, 10000),
    seeds=(0,),
    cut_modes=CUT_MODES,
    null_surrogates=19,
    sweep_kind="ring",
    couplings=DEFAULT_COUPLINGS,
    workers=1,
    confirmatory=False,
    freeze_tag=None,
) -> dict:
    from impact_pipeline.bench.generators import BINARY_NETWORK_KINDS
    from impact_pipeline.bench.run_bench import confirmatory_guard, seed_set
    from impact_pipeline.provenance import collect_code_version

    kinds = tuple(BINARY_NETWORK_KINDS if kinds is None else kinds)
    bad = sorted(set(kinds) - set(BINARY_NETWORK_KINDS))
    if bad:
        raise ValueError(f"unknown network kinds {bad}")
    bad = sorted(set(cut_modes) - set(CUT_MODES))
    if bad:
        raise ValueError(f"unknown cut modes {bad}")
    want = "confirmatory" if confirmatory else "dev"
    wrong = [s for s in seeds if seed_set(s) != want]
    if wrong:
        raise ValueError(
            f"{'confirmatory' if confirmatory else 'development'} runs use "
            f"{want} seeds (without --confirmatory: 0-999; with it: >= 10000); "
            f"got {wrong[:5]}"
        )
    code = (
        confirmatory_guard(REPO_ROOT, freeze_tag)
        if confirmatory
        else collect_code_version(REPO_ROOT)
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / RESULTS_JSONL
    tasks = tasks_for(kinds, n, n_times, seeds, cut_modes, sweep_kind, couplings)
    done = _done(jsonl)
    todo = [t for t in tasks if t["task_id"] not in done]
    t0 = time.time()
    with open(jsonl, "a", encoding="utf-8") as fh:

        def _emit(rec):
            rec["code_git_sha"] = code.get("git_sha")
            rec["freeze_tag"] = code.get("freeze_tag")
            fh.write(
                json.dumps({k: _jsonable(v) for k, v in rec.items()}, sort_keys=True)
                + "\n"
            )
            fh.flush()

        if int(workers) > 1:
            with ProcessPoolExecutor(max_workers=int(workers)) as ex:
                futs = [ex.submit(run_task, t, null_surrogates) for t in todo]
                for f in as_completed(futs):
                    _emit(f.result())
        else:
            for t in todo:
                _emit(run_task(t, null_surrogates))
    latest = {}
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            latest[r["task_id"]] = r
    df = pd.DataFrame([latest[k] for k in sorted(latest)])
    df = df.rename(columns={"kind": "system"})
    df.to_csv(out / "iim_validation.csv", index=False)
    exact_rows = []
    for kind in kinds:
        for cm in cut_modes:
            exact_rows.append(
                {"system": kind, "n": int(n), "cut_mode": cm, **exact_iim(kind, n, cm)}
            )
    pd.DataFrame(exact_rows).to_csv(out / "iim_validation_exact.csv", index=False)
    summary = {
        "version": VALIDATION_VERSION,
        "design": {
            "kinds": list(kinds),
            "n": int(n),
            "n_times": list(n_times),
            "seeds": [int(s) for s in seeds],
            "cut_modes": list(cut_modes),
            "null_surrogates": int(null_surrogates),
            "bins": 2,
            "lag_trs": 1,
            "tpm_estimator": "node_shrinkage",
            "sweep_kind": sweep_kind,
            "couplings": list(couplings),
        },
        "confirmatory": bool(confirmatory),
        "n_tasks": len(tasks),
        "n_run": len(todo),
        "seconds": round(time.time() - t0, 2),
        "code_version": code,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "iim_validation.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return {"summary": summary, "results": df}


def _ints(text):
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
    return tuple(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--kinds", default=None, help="comma-separated network kinds")
    ap.add_argument("--n", type=int, default=4, help="units per network")
    ap.add_argument("--T", default="1000,3000,10000", help="run lengths (steps)")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--cut-modes", default=",".join(CUT_MODES))
    ap.add_argument("--null-surrogates", type=int, default=19)
    ap.add_argument(
        "--sweep-kind",
        default="ring",
        help="network kind of the coupling sweep ('' = none)",
    )
    ap.add_argument(
        "--couplings", default=",".join(f"{c:g}" for c in DEFAULT_COUPLINGS)
    )
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--confirmatory", action="store_true")
    ap.add_argument("--freeze-tag", default=None)
    args = ap.parse_args(argv)
    kinds = (
        None
        if not args.kinds
        else tuple(k.strip() for k in args.kinds.split(",") if k.strip())
    )
    res = run(
        args.out,
        kinds=kinds,
        n=args.n,
        n_times=_ints(args.T),
        seeds=_ints(args.seeds),
        cut_modes=tuple(c.strip() for c in args.cut_modes.split(",") if c.strip()),
        null_surrogates=args.null_surrogates,
        sweep_kind=args.sweep_kind or None,
        couplings=tuple(float(c) for c in args.couplings.split(",") if c.strip()),
        workers=args.workers,
        confirmatory=args.confirmatory,
        freeze_tag=args.freeze_tag,
    )
    df = res["results"]
    ok = df[df["status"] == "ok"]
    print(
        ok.groupby(["system", "cut_mode", "n_time"])[
            ["delta_psi_exact", "delta_psi_est", "iim_z"]
        ]
        .median()
        .to_string()
    )
    return 0 if (df["status"] == "ok").all() else 1


if __name__ == "__main__":
    sys.exit(main())
