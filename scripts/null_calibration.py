#!/usr/bin/env python
"""
Null calibration of the MPC component estimators: how often does each
estimator report PRESENT on data without the mechanism it measures?

For every null family, run length ``T`` (samples) and node count, ``R``
replicate null systems are generated and every applicable estimator is run
through the MPC-Bench in-memory runner (``bench.export.run_in_memory``: the
same estimator settings, bearer views and component nulls as the bench),
with ``K`` null surrogates and a delete-a-group jackknife SE over ``G``
groups (``--se-groups``). Each component is classified by the evidence layer
v2 (``impact_pipeline.evidence.component_assessment``) under a declared
protocol (``--protocol``, default ``protocols/mpc_bench_v1.json``): the
construct scale ``c = (m - nu) / (rho - nu)`` with the protocol's external
reference anchor (the bench positive control), its cutoffs ``(z, delta)`` and
``alpha``, the jackknife SE with ``G - 1`` degrees of freedom (Student-t
bounds) and the protocol's declared null families (a different family is
UNDEFINED, ``NULL_FAMILY_MISMATCH``). A component without a sampling SE is
UNDEFINED (``NO_SAMPLING_SE``) and one without an anchor UNDEFINED
(``INVALID_ANCHORS``), so a false PRESENT can only come from the calibrated
rule. ``--status-rule legacy_v1`` instead applies the superseded v1 null-SD
rule (margin ``(m - nu) / sigma_null`` against ``z_present``, TOST for
ABSENT) as a documented diagnostic comparator; its statuses carry
``status_impl = legacy_v1`` and never count for the registry's H0.

Null families (data generated here, seeded per replicate):

- ``ar1``: independent AR(1) per node (``--ar-coef``);
- ``pink``: independent 1/f noise per node (spectral exponent ``--pink-beta``);
- ``surrogate_iid``: per-node (independent-phase) Fourier surrogates of a
  coupled VAR(1): every node keeps its spectrum, cross-node dependence is
  destroyed;
- ``surrogate_linear``: multivariate (shared-phase) Fourier surrogates of a
  coupled non-linear (tanh) network: a linear Gaussian process with the same
  auto- and cross-spectra. It is a null only for PDI (non-linear repertoire)
  and for the event-locked RAM and SRPI, not for NAS or IIM.

Events: a synthetic task schedule (goal cue, stimulus, response with a random
choice, feedback with a random +-1 reward; then self-caused events with
stimulus-identical, phase-matched yoked replays) that is independent of the
data, so RAM and SRPI are also null. Declared meta: workspace = the first
``max(2, nodes // 5)`` nodes, IIM on ``--iim-macro-nodes`` macro nodes (means
of equal node groups outside the workspace).

Rates per (family, T, nodes, principle): false-PRESENT, ABSENT and UNDEFINED
rates with Wilson intervals, the mean and SD of ``c``, the rate of null-SD
margins above ``z_{1-alpha}`` and the false-PRESENT rate expected for an
exchangeable Gaussian null under the v1 rule with ``K`` surrogates
(``P(sqrt(1 + 1/K) t_{K-1} > z_present)``; v1 diagnostics), and the entry
check of spec V2-5 (a): rate at most ``alpha + 0.02`` (one-sided, the
registry default; a calibrated v2 rule has a false-PRESENT rate far below
``alpha``) and within ``alpha +- 0.02``. Verdict-level: the rate of
MPC_CONSISTENT (and EXCLUDED) verdicts on the null systems over the
principles computed (Kleene AND of the statuses).

Outputs (``--out``): ``null_calibration_replicates.csv``,
``null_calibration_rates.csv``, ``null_calibration_verdicts.csv`` and
``null_calibration.json`` (provenance, protocol and its hash, parameters).

Example::

    python scripts/null_calibration.py --out outputs/null_calibration \
        --protocol protocols/mpc_bench_v1.json \
        --kinds ar1,pink,surrogate_iid,surrogate_linear --T 1200,2400 \
        --nodes 8,16 --replicates 100 --null-surrogates 19 --se-groups 10 \
        --workers 8
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import logging
import math
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t as t_dist

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from impact_pipeline.necessity import normalize_status  # noqa: E402

CALIBRATION_VERSION = "null-calibration/2.0.0"
DEFAULT_PROTOCOL = REPO_ROOT / "protocols" / "mpc_bench_v1.json"
STATUS_RULES = ("v2", "legacy_v1")
PRINCIPLES = ("RAM", "PDI", "NAS", "IIM", "SRPI")
NULL_KINDS = {
    "ar1": PRINCIPLES,
    "pink": PRINCIPLES,
    "surrogate_iid": PRINCIPLES,
    "surrogate_linear": ("RAM", "PDI", "SRPI"),
}
BAND = 0.02


def _seed(base, *keys) -> int:
    text = "|".join([str(int(base))] + [str(k) for k in keys])
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


# --------------------------------------------------------------------------
# null data and design
# --------------------------------------------------------------------------
def _standardize(x):
    x = x - x.mean(axis=1, keepdims=True)
    return x / (x.std(axis=1, keepdims=True) + 1e-12)


def _coupled(n_nodes, n_time, rng, nonlinear=False, radius=0.9, burn=200):
    a = rng.standard_normal((n_nodes, n_nodes))
    a *= float(radius) / max(np.max(np.abs(np.linalg.eigvals(a))), 1e-12)
    x = np.zeros((n_nodes, n_time + burn))
    e = rng.standard_normal(x.shape)
    for k in range(1, x.shape[1]):
        drive = a @ x[:, k - 1]
        x[:, k] = (np.tanh(1.5 * drive) if nonlinear else drive) + e[:, k]
    return x[:, burn:]


def null_timeseries(kind, n_nodes, n_time, rng, *, ar_coef=0.5, pink_beta=1.0):
    """``(n_nodes, n_time)`` z-scored null data of family ``kind``."""
    from impact_pipeline import nulls

    n_nodes, n_time = int(n_nodes), int(n_time)
    if kind == "ar1":
        if not -1.0 < float(ar_coef) < 1.0:
            raise ValueError("ar_coef must be in (-1, 1)")
        e = rng.standard_normal((n_nodes, n_time + 200))
        x = np.zeros_like(e)
        for k in range(1, x.shape[1]):
            x[:, k] = float(ar_coef) * x[:, k - 1] + e[:, k]
        return _standardize(x[:, 200:])
    if kind == "pink":
        f = np.fft.rfftfreq(n_time)
        amp = np.zeros_like(f)
        amp[1:] = f[1:] ** (-float(pink_beta) / 2.0)
        spec = (rng.standard_normal((n_nodes, f.size))
                + 1j * rng.standard_normal((n_nodes, f.size))) * amp[None, :]
        return _standardize(np.fft.irfft(spec, n=n_time, axis=1))
    if kind == "surrogate_iid":
        base = _coupled(n_nodes, n_time, rng)
        return _standardize(nulls.phase_randomize(base, rng, multivariate=False))
    if kind == "surrogate_linear":
        base = _coupled(n_nodes, n_time, rng, nonlinear=True)
        return _standardize(nulls.phase_randomize(base, rng, multivariate=True))
    raise ValueError(
        f"unknown null kind {kind!r}; expected one of {sorted(NULL_KINDS)}"
    )


def null_events(n_time, dt, rng, *, task_fraction=0.5, rhythm_hz=0.25,
                n_phase_bins=4):
    """
    Synthetic, data-independent events table in the MPC-Bench layout: bandit
    trials (goal cue, stimulus +0.6 s, response +1.2 s, feedback +1.8 s,
    random choice and +-1 reward) over the first ``task_fraction`` of the
    run, then self-caused events each followed one rhythm period later by a
    stimulus-identical, phase-matched ``other_caused`` replay.
    """
    from impact_pipeline.bench.generators import EVENT_COLUMNS

    dur = float(n_time) * float(dt)
    period = 1.0 / float(rhythm_hz)
    phi0 = float(rng.uniform(0, 2 * np.pi))

    def _pbin(t):
        ph = (2 * np.pi * rhythm_hz * t + phi0) % (2 * np.pi)
        return int(ph // (2 * np.pi / n_phase_bins))

    rows = []

    def _row(t, d, tt, value, channel, trial=None, choice=None, reward=None,
             yoked=None, eid=None):
        rows.append({"onset": round(t, 6), "duration": d, "trial_type": tt,
                     "value": value, "choice": choice, "reward": reward,
                     "yoked_to": yoked, "phase_bin": _pbin(t),
                     "impact_channel": channel, "trial": trial, "event_id": eid})

    t, i = 1.0, 0
    while t + 2.5 < task_fraction * dur:
        c = int(rng.integers(0, 2))
        r = float(rng.choice([-1.0, 1.0]))
        ch = "behavioural_feedback"
        _row(t, 0.3, "goal_cue", int(rng.integers(0, 2)), ch, i, eid=f"goal{i:04d}")
        _row(t + 0.6, 0.3, "stimulus", int(rng.integers(0, 2)), ch, i,
             eid=f"stim{i:04d}")
        _row(t + 1.2, 0.15, "response", c, ch, i, choice=c, eid=f"resp{i:04d}")
        _row(t + 1.8, 0.3, "feedback", r, ch, i, choice=c, reward=r,
             eid=f"fb{i:04d}")
        t += 3.0 + float(rng.uniform(0, 1))
        i += 1
    j = 0
    t = max(t, task_fraction * dur)
    while t + period + 1.0 < dur:
        sid = int(rng.integers(0, 4))
        _row(t, 0.2, "self_caused", sid, "agency", eid=f"self{j:04d}")
        _row(t + period, 0.2, "other_caused", sid, "agency", yoked=f"self{j:04d}",
             eid=f"other{j:04d}")
        # next self-caused event after this replay (no overlap with it)
        t += period + 1.5 + float(rng.uniform(0, 1))
        j += 1
    df = pd.DataFrame(rows, columns=list(EVENT_COLUMNS))
    df = df.sort_values(["onset", "event_id"], kind="mergesort").reset_index(drop=True)
    for col in ("trial", "choice"):
        df[col] = df[col].astype("Int64")
    return df


def null_system(kind, n_nodes, n_time, seed, *, dt=0.05, iim_macro_nodes=4,
                ar_coef=0.5, pink_beta=1.0):
    """A ``BenchSystem`` holding null data, a null event design and the meta."""
    from impact_pipeline.bench.generators import BenchSystem

    rng = np.random.default_rng(int(seed))
    ts = null_timeseries(kind, n_nodes, n_time, rng, ar_coef=ar_coef,
                         pink_beta=pink_beta)
    events = null_events(n_time, dt, rng)
    n_ws = max(2, int(n_nodes) // 5)
    rest = list(range(n_ws, int(n_nodes)))
    k = max(2, min(int(iim_macro_nodes), len(rest)))
    groups = np.array_split(np.asarray(rest), k)
    meta = {
        "family": "null_calibration", "null_kind": kind, "seed": int(seed),
        "dt": float(dt), "bearer_nodes": list(range(int(n_nodes))),
        "workspace_nodes": list(range(n_ws)),
        "iim_macro_nodes": {f"m{i}": [int(v) for v in g] for i, g in enumerate(groups)},
        "iim_grain": f"null_macro_{k}", "iim_bins": 2, "iim_lag_samples": 2,
    }
    return BenchSystem(ts=ts, events=events, meta=meta, oracle={"null_kind": kind})


# --------------------------------------------------------------------------
# evidence layer v2 (protocol) and the legacy v1 comparator
# --------------------------------------------------------------------------
def resolve_calibration_protocol(protocol):
    """
    The protocol of the classification (``evidence.Protocol``, dict or JSON
    path; None: ``protocols/mpc_bench_v1.json`` of the checkout). It must
    declare an ``external`` reference: null systems have no cohort.
    """
    from impact_pipeline import evidence as ev

    if protocol is None and DEFAULT_PROTOCOL.is_file():
        protocol = str(DEFAULT_PROTOCOL)
    proto = ev.resolve_protocol(protocol)
    if proto is None:
        raise ValueError(
            "null calibration needs a protocol with an external reference "
            f"anchor (--protocol; {DEFAULT_PROTOCOL} not found)"
        )
    if proto.reference.get("kind") != "external":
        raise ValueError(
            "null calibration needs an external reference anchor; the "
            f"protocol's reference is {proto.reference.get('kind')!r}"
        )
    return proto


def local_status(estimate, null_mean, null_sd, se=0.0, z_present=1.645,
                 delta_equiv=1.0, alpha=0.05):
    """
    The superseded v1 status rule (null-SD units, no reference anchor), kept
    as a documented diagnostic comparator (``--status-rule legacy_v1``):
    PRESENT if ``(m - nu) - z_{1-alpha} se > z_present sigma_null``, ABSENT
    if ``|m - nu| + z_{1-alpha} se <= delta_equiv sigma_null`` (TOST).
    """
    vals = [estimate, null_mean, null_sd, se]
    if not all(np.isfinite(v) for v in vals) or null_sd <= 0 or se < 0:
        return "UNDEFINED"
    za = float(norm.ppf(1.0 - alpha))
    excess = estimate - null_mean
    if excess - za * se > z_present * null_sd:
        return "PRESENT"
    if abs(excess) + za * se <= delta_equiv * null_sd:
        return "ABSENT"
    return "UNDEFINED"


def _se_df(comp):
    if comp.get("se_df") is not None:
        return float(comp["se_df"])
    n = comp.get("se_n")
    if n is not None and np.isfinite(float(n)) and int(n) >= 2:
        return float(int(n) - 1)
    return None


def classify(principle, comp, protocol, *, status_rule="v2"):
    """
    ``(status, reason, impl, assessment)`` of one component record.

    ``v2``: ``evidence.component_assessment`` on the construct scale with the
    protocol's reference, cutoffs and alpha, the component's sampling SE
    (``se``, ``se_df`` or ``se_n - 1`` degrees of freedom) and null family
    check (``impl = 'evidence'``). ``legacy_v1``: :func:`local_status`
    (``impl = 'legacy_v1'``, no assessment).
    """
    from impact_pipeline import evidence as ev

    est, nm, nsd = (float(comp.get(k, np.nan)) for k in ("estimate", "null_mean",
                                                         "null_sd"))
    if status_rule == "legacy_v1":
        if not bool(comp.get("defined", True)):
            return "UNDEFINED", comp.get("reason") or "NOT_DEFINED", "legacy_v1", None
        se = comp.get("se")
        se = 0.0 if se is None or not np.isfinite(float(se)) else float(se)
        return local_status(est, nm, nsd, se), None, "legacy_v1", None
    if status_rule != "v2":
        raise ValueError(f"status_rule must be one of {STATUS_RULES}")
    channel = str(comp.get("channel") or "default")
    ref = protocol.reference
    key = next((k for k in (f"{principle}:{channel}", principle)
                if k in ref.get("values", {})), None)
    anchor = {}
    if key is not None:
        anchor = {"reference": ref["values"][key],
                  "reference_se": ref.get("se", {}).get(key),
                  "reference_scale": ref.get("scale", "excess")}
    se = comp.get("se")
    item = ev.ComponentEvidence(
        principle=principle, estimate=est, null_mean=nm, null_sd=nsd,
        se=0.0 if se is None else float(se), se_df=_se_df(comp),
        channel=channel, defined=bool(comp.get("defined", True)),
        reason=comp.get("reason"), null_family=comp.get("null_family"),
        n_null=int(comp.get("n_null", 0) or 0), **anchor,
    )
    declared = protocol.null_families.get(principle)
    if declared is not None and item.null_family is not None and (
            str(item.null_family) != declared):
        return ("UNDEFINED",
                f"{ev.REASON_NULL_FAMILY_MISMATCH}:{declared}/{item.null_family}",
                "evidence", None)
    a = ev.component_assessment(item, cutoff=protocol.cutoff_for(principle),
                                alpha=protocol.alpha)
    return a.status.value, a.reason, "evidence", a


def kleene_verdict(statuses):
    """v2 verdict name of the strong-Kleene AND of component statuses."""
    s = [normalize_status(x) for x in statuses]
    if any(x == "ABSENT" for x in s):
        return "EXCLUDED"
    if s and all(x == "PRESENT" for x in s):
        return "MPC_CONSISTENT"
    return "UNDETERMINED"


# --------------------------------------------------------------------------
# replicate runner
# --------------------------------------------------------------------------
AGENCY_FIX_REASON = "missing_agency_events"


def srpi_agency_component(system, null_surrogates, seed, params=None) -> dict:
    """
    SRPI in agency mode with the parsed ``agency_events`` passed explicitly
    and the estimator's own yoked-cluster label-permutation null
    (``SRPI_null_*``). Used when the bench runner does not pass agency events
    (it then reports ``agency_contract_violation:missing_agency_events``).
    """
    from impact_pipeline import mpc_metrics as mm
    from impact_pipeline.bench.export import BENCH_ESTIMATOR_PARAMS
    from impact_pipeline.event_parsing import events_table_to_bundle

    kw = dict(BENCH_ESTIMATOR_PARAMS["SRPI"])
    kw.update((params or {}).get("SRPI") or {})
    b = events_table_to_bundle(system.events)
    k = int(null_surrogates) if int(null_surrogates) > 0 else 200
    t0 = time.perf_counter()
    d = mm.compute_SRPI(
        system.ts, tr=system.dt, self_onsets=b["self_onsets"],
        nonself_onsets=b["nonself_onsets"], return_details=True, mode="agency",
        agency_events=b["agency_events"], agency_null_permutations=k,
        agency_random_state=int(seed), **kw,
    )
    est = float(d.get("raw", np.nan)) if isinstance(d, dict) else np.nan
    defined = bool(np.isfinite(est))
    return {
        "estimate": est if defined else np.nan,
        "value": float(d.get("value", np.nan)),
        "statistic": "raw",
        "null_mean": float(d.get("SRPI_null_mean", np.nan)),
        "null_sd": float(d.get("SRPI_null_sd", np.nan)),
        "n_null": int(d.get("SRPI_null_n", 0) or 0),
        "null_family": d.get("SRPI_null_method"),
        "defined": defined,
        "reason": None if defined else (d.get("undefined_reason") or "undefined"),
        "bearer_id": "system",
        "seconds": round(time.perf_counter() - t0, 4),
    }


@contextlib.contextmanager
def _quiet():
    """Silence estimator logging and warnings, then restore both (serial runs
    share the caller's process)."""
    previous = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        logging.disable(previous)


def run_replicate(task: dict) -> list:
    """Run the estimators on one null system; one record per principle."""
    from impact_pipeline.bench.export import run_in_memory

    kind, n_time, n_nodes, rep = (task["kind"], task["n_time"], task["n_nodes"],
                                  task["replicate"])
    seed = _seed(task["seed"], kind, n_time, n_nodes, rep)
    system = null_system(kind, n_nodes, n_time, seed, dt=task["dt"],
                         iim_macro_nodes=task["iim_macro_nodes"],
                         ar_coef=task["ar_coef"], pink_beta=task["pink_beta"])
    metrics = [p for p in task["metrics"] if p in NULL_KINDS[kind]]
    t0 = time.perf_counter()
    runner = {p: "bench.run_in_memory" for p in metrics}
    with _quiet():
        res = run_in_memory(system, metrics=metrics, params=task.get("params"),
                            null_surrogates=task["null_surrogates"],
                            null_seed=_seed(seed, "null"),
                            se_groups=task["se_groups"])
        srpi = res["components"].get("SRPI")
        if (srpi is not None and task.get("srpi_agency_fix", True)
                and AGENCY_FIX_REASON in str(srpi.get("reason") or "")):
            res["components"]["SRPI"] = srpi_agency_component(
                system, task["null_surrogates"], _seed(seed, "srpi"),
                task.get("params"))
            runner["SRPI"] = "null_calibration.srpi_agency_component"
    seconds = time.perf_counter() - t0
    from impact_pipeline import evidence as ev

    proto = ev.Protocol.from_dict(task["protocol"])
    rows = []
    nan = float("nan")
    for p in metrics:
        comp = res["components"].get(p, {})
        st, reason, impl, a = classify(p, comp, proto,
                                       status_rule=task["status_rule"])
        est, nm, nsd = (float(comp.get(k, np.nan)) for k in ("estimate", "null_mean",
                                                             "null_sd"))
        with np.errstate(invalid="ignore", divide="ignore"):
            margin = (est - nm) / nsd if nsd > 0 else np.nan
        rows.append({
            "null_kind": kind, "n_time": int(n_time), "n_nodes": int(n_nodes),
            "replicate": int(rep), "seed": int(seed), "principle": p,
            "estimate": est, "null_mean": nm, "null_sd": nsd,
            "n_null": int(comp.get("n_null", 0) or 0),
            "null_family": comp.get("null_family"), "statistic": comp.get("statistic"),
            "defined": bool(comp.get("defined", False)),
            "estimator_reason": comp.get("reason"), "margin": margin,
            "se": float(comp.get("se") if comp.get("se") is not None else nan),
            "se_n": int(comp.get("se_n", 0) or 0),
            "c": nan if a is None else a.c,
            "c_se": nan if a is None else a.se,
            "c_df": nan if a is None else a.df,
            "c_lower": nan if a is None else a.lower,
            "c_upper": nan if a is None else a.upper,
            "status": st, "status_reason": reason, "status_impl": impl,
            "runner": runner[p], "seconds_system": round(seconds, 3),
        })
    return rows


def wilson(k, n, alpha=0.05):
    if n <= 0:
        return (np.nan, np.nan)
    z = float(norm.ppf(1 - alpha / 2))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def expected_exchangeable_rate(k, z_present=1.645):
    """v1 false-PRESENT rate for an exchangeable Gaussian null with ``k`` draws."""
    k = int(k)
    if k < 2:
        return np.nan
    return float(t_dist.sf(z_present / math.sqrt(1.0 + 1.0 / k), k - 1))


def summarise(rep: pd.DataFrame, alpha=0.05, z_present=1.645) -> tuple:
    keys = ["null_kind", "n_time", "n_nodes", "principle"]
    rows = []
    for key, sub in rep.groupby(keys, sort=True):
        n = int(len(sub))
        k_p = int((sub["status"] == "PRESENT").sum())
        k_a = int((sub["status"] == "ABSENT").sum())
        k_u = n - k_p - k_a
        lo, hi = wilson(k_p, n)
        rate = k_p / n if n else np.nan
        margins = sub["margin"].to_numpy(dtype=float)
        fin = np.isfinite(margins)
        cs = sub["c"].to_numpy(dtype=float) if "c" in sub else np.full(n, np.nan)
        cfin = np.isfinite(cs)
        # null size of the replicates that have a null (undefined components
        # record n_null = 0 and must not shrink the expected-rate K)
        with_null = sub["n_null"][sub["n_null"] > 0]
        k_null = int(with_null.median()) if len(with_null) else 0
        rows.append({
            **dict(zip(keys, key)), "n": n, "n_defined": int(sub["defined"].sum()),
            "n_present": k_p, "false_present_rate": rate,
            "false_present_lo": lo, "false_present_hi": hi,
            "n_absent": k_a, "absent_rate": k_a / n if n else np.nan,
            "n_undefined": k_u, "undefined_rate": k_u / n if n else np.nan,
            "margin_exceed_rate": float(np.mean(margins[fin] > z_present))
            if fin.any() else np.nan,
            "margin_mean": float(np.mean(margins[fin])) if fin.any() else np.nan,
            "margin_sd": (float(np.std(margins[fin], ddof=1))
                          if fin.sum() > 1 else np.nan),
            "c_mean": float(np.mean(cs[cfin])) if cfin.any() else np.nan,
            "c_sd": float(np.std(cs[cfin], ddof=1)) if cfin.sum() > 1 else np.nan,
            "median_n_null": k_null,
            "expected_rate_exchangeable_v1": expected_exchangeable_rate(k_null,
                                                                        z_present),
            "within_alpha_band": bool(abs(rate - alpha) <= BAND) if n else False,
            "at_most_alpha_plus_band": bool(rate <= alpha + BAND) if n else False,
            "ci_overlaps_band": bool(lo <= alpha + BAND and hi >= alpha - BAND)
            if n else False,
            "status_impl": ";".join(sorted(sub["status_impl"].astype(str).unique())),
            "runner": ";".join(sorted(sub["runner"].astype(str).unique())),
            "mean_seconds_system": float(sub["seconds_system"].mean()),
        })
    rates = pd.DataFrame(rows)
    vrows = []
    for key, sub in rep.groupby(["null_kind", "n_time", "n_nodes", "replicate"]):
        vrows.append({**dict(zip(["null_kind", "n_time", "n_nodes", "replicate"], key)),
                      "necessity_set": ",".join(sorted(sub["principle"])),
                      "verdict": kleene_verdict(sub["status"])})
    vdf = pd.DataFrame(vrows)
    vsum = []
    for key, sub in vdf.groupby(["null_kind", "n_time", "n_nodes", "necessity_set"]):
        n = len(sub)
        k_c = int((sub["verdict"] == "MPC_CONSISTENT").sum())
        lo, hi = wilson(k_c, n)
        vsum.append({**dict(zip(["null_kind", "n_time", "n_nodes", "necessity_set"],
                                key)),
                     "n": n, "consistent_rate": k_c / n, "consistent_lo": lo,
                     "consistent_hi": hi,
                     "excluded_rate": float((sub["verdict"] == "EXCLUDED").mean()),
                     "undetermined_rate": float(
                         (sub["verdict"] == "UNDETERMINED").mean())})
    return rates, pd.DataFrame(vsum)


def seed_policy(seed, confirmatory=False, freeze_tag=None):
    """
    Development / confirmatory split of the replicate seed base (spec v2,
    V2-6): development calibrations use a base in 0-999; a confirmatory
    calibration uses a base >= 10000 and runs only on the frozen code
    (``bench.run_bench.confirmatory_guard`` with the freeze tag). Returns the
    code identity (the guard's record for a confirmatory run, else None).
    """
    seed = int(seed)
    if confirmatory:
        if seed < 10000:
            raise ValueError("a confirmatory calibration uses --seed >= 10000")
        from impact_pipeline.bench.run_bench import confirmatory_guard

        return confirmatory_guard(REPO_ROOT, freeze_tag)
    if not 0 <= seed <= 999:
        raise ValueError(
            "a development calibration uses --seed 0-999 (>= 10000 is "
            "confirmatory: pass --confirmatory after the code freeze)"
        )
    return None


def run(out_dir, *, kinds=("ar1", "pink", "surrogate_iid", "surrogate_linear"),
        n_times=(1200,), n_nodes=(8,), replicates=20, null_surrogates=19,
        metrics=PRINCIPLES, seed=0, dt=0.05, iim_macro_nodes=4, ar_coef=0.5,
        pink_beta=1.0, params=None, protocol=None, se_groups=10,
        status_rule="v2", workers=1, srpi_agency_fix=True, confirmatory=False,
        freeze_tag=None) -> dict:
    """Run the calibration; the protocol's alpha sets the rate criteria."""
    if status_rule not in STATUS_RULES:
        raise ValueError(f"status_rule must be one of {STATUS_RULES}")
    guard = seed_policy(seed, confirmatory, freeze_tag)
    if int(se_groups) == 1 or int(se_groups) < 0:
        raise ValueError("se_groups must be 0 (off) or >= 2")
    proto = resolve_calibration_protocol(protocol)
    alpha = float(proto.alpha)
    # the protocol's declared estimator options (e.g. the IIM cut mode) apply
    from impact_pipeline.bench.export import protocol_params

    params = protocol_params(proto, params)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bad = sorted(set(kinds) - set(NULL_KINDS))
    if bad:
        raise ValueError(f"unknown null kinds {bad}")
    bad = sorted(set(metrics) - set(PRINCIPLES))
    if bad:
        raise ValueError(f"unknown principles {bad}")
    tasks = [
        {"kind": k, "n_time": int(t), "n_nodes": int(n), "replicate": r,
         "seed": int(seed), "dt": float(dt), "iim_macro_nodes": int(iim_macro_nodes),
         "ar_coef": float(ar_coef), "pink_beta": float(pink_beta),
         "null_surrogates": int(null_surrogates), "metrics": tuple(metrics),
         "params": params, "protocol": proto.to_dict(),
         "se_groups": int(se_groups), "status_rule": status_rule,
         "srpi_agency_fix": bool(srpi_agency_fix)}
        for k, t, n, r in itertools.product(kinds, n_times, n_nodes,
                                            range(int(replicates)))
    ]
    t0 = time.time()
    if int(workers) > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            results = list(ex.map(run_replicate, tasks, chunksize=1))
    else:
        results = [run_replicate(t) for t in tasks]
    rep = pd.DataFrame([row for rows in results for row in rows])
    rep.to_csv(out / "null_calibration_replicates.csv", index=False)
    rates, verdicts = summarise(rep, alpha=alpha)
    rates.to_csv(out / "null_calibration_rates.csv", index=False)
    verdicts.to_csv(out / "null_calibration_verdicts.csv", index=False)
    try:
        from impact_pipeline.bench import export as bench_export
        from impact_pipeline.provenance import collect_code_version

        prov = guard if guard is not None else collect_code_version(REPO_ROOT)
        est_params = {"defaults": bench_export.BENCH_ESTIMATOR_PARAMS,
                      "optional_modes": bench_export.OPTIONAL_MODES,
                      "overrides": params}
    except Exception as exc:  # noqa: BLE001 - provenance must not fail a run
        prov, est_params = {"code_version": "unknown", "error": str(exc)}, params
    prov["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary = {
        "version": CALIBRATION_VERSION,
        "evidence_api": "v2",
        "status_rule": status_rule,
        "protocol": proto.to_dict(),
        "protocol_hash": proto.hash,
        "design": {"kinds": list(kinds), "n_times": list(n_times),
                   "n_nodes": list(n_nodes), "replicates": int(replicates),
                   "null_surrogates": int(null_surrogates), "metrics": list(metrics),
                   "dt": dt, "iim_macro_nodes": iim_macro_nodes, "ar_coef": ar_coef,
                   "pink_beta": pink_beta, "se_groups": int(se_groups),
                   "alpha": alpha, "band": BAND,
                   "srpi_agency_fix": bool(srpi_agency_fix),
                   "applicability": {k: list(v) for k, v in NULL_KINDS.items()}},
        "estimator_params": est_params,
        "seed": int(seed),
        "split": "confirmatory" if confirmatory else "development",
        "workers": int(workers),
        "seconds": round(time.time() - t0, 2),
        "provenance": prov,
    }
    with open(out / "null_calibration.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return {"summary": summary, "replicates": rep, "rates": rates,
            "verdicts": verdicts}


def _load_json_arg(text):
    if text is None:
        return None
    p = Path(text)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(text)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--kinds", default="ar1,pink,surrogate_iid,surrogate_linear")
    ap.add_argument("--T", default="1200,2400", help="run lengths in samples")
    ap.add_argument("--nodes", default="8,16")
    ap.add_argument("--replicates", type=int, default=20)
    ap.add_argument("--null-surrogates", type=int, default=19)
    ap.add_argument("--metrics", default=",".join(PRINCIPLES))
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--iim-macro-nodes", type=int, default=4)
    ap.add_argument("--ar-coef", type=float, default=0.5)
    ap.add_argument("--pink-beta", type=float, default=1.0)
    ap.add_argument("--params", default=None, help="estimator overrides (JSON/path)")
    ap.add_argument("--protocol", default=None,
                    help="evidence.Protocol JSON with an external reference anchor "
                         f"(default {DEFAULT_PROTOCOL.relative_to(REPO_ROOT)})")
    ap.add_argument("--se-groups", type=int, default=10,
                    help="jackknife groups of the sampling SE (0 = none: every "
                         "component UNDEFINED with NO_SAMPLING_SE)")
    ap.add_argument("--status-rule", choices=STATUS_RULES, default="v2",
                    help="legacy_v1: the superseded null-SD rule (diagnostic only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--confirmatory", action="store_true",
                    help="confirmatory calibration (--seed >= 10000, frozen code)")
    ap.add_argument("--freeze-tag", default=None,
                    help="code-freeze tag (required with --confirmatory)")
    ap.add_argument("--no-srpi-agency-fix", action="store_true",
                    help="keep the bench runner's SRPI result even when it lacks "
                         "agency events")
    args = ap.parse_args(argv)

    def _list(text, cast=str):
        return tuple(cast(v.strip()) for v in str(text).split(",") if v.strip())

    res = run(
        args.out, kinds=_list(args.kinds), n_times=_list(args.T, int),
        n_nodes=_list(args.nodes, int), replicates=args.replicates,
        null_surrogates=args.null_surrogates, metrics=_list(args.metrics),
        seed=args.seed, dt=args.dt, iim_macro_nodes=args.iim_macro_nodes,
        ar_coef=args.ar_coef, pink_beta=args.pink_beta,
        params=_load_json_arg(args.params), protocol=args.protocol,
        se_groups=args.se_groups, status_rule=args.status_rule,
        workers=args.workers,
        srpi_agency_fix=not args.no_srpi_agency_fix,
        confirmatory=args.confirmatory, freeze_tag=args.freeze_tag,
    )
    cols = ["null_kind", "n_time", "n_nodes", "principle", "n", "false_present_rate",
            "undefined_rate", "expected_rate_exchangeable_v1"]
    print(res["rates"][cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
