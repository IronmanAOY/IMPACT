"""
Cost model and size guard of the Hunter IIM campaign.

The exhaustive IIM search grows roughly 30-fold per added subsystem node, so a
campaign that looks harmless on the command line can need years of compute.
This module counts the work of a campaign before anything is prepared or
submitted, turns the count into an order-of-magnitude runtime and refuses
configurations above a ceiling.

Unit of work ("Psi evaluation"): one (mechanism, purview, mechanism
bipartition, purview bipartition) term of the integration mass Psi under one
TPM, where the TPMs of a run are the intact TPM and one TPM per evaluated
system cut. So, per run::

    psi_evaluations = (sum_M bip(|M|)) * (sum_Z bip(|Z|)) * (1 + n_cuts_evaluated)

with ``bip(k) = 2**(k-1) - 1`` unordered bipartitions of a k-node set (0 for
singletons, whose Psi term is zero), summed over the enumerated mechanisms M
and purviews Z. A campaign multiplies this by its real runs and by
``1 + K_null + K_boot`` (every surrogate and bootstrap replicate run is a full
IIM run). Cause/effect directions, both pairings of a bipartition pair and the
observed mechanism states are not counted separately; their cost is absorbed in
the time per evaluation.

Nothing here computes or changes an estimator: the counts mirror the
enumerations of ``mpc_metrics.prepare_iim_problem`` (tests check them against
it).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from math import comb

COST_SCHEMA = "impact-hunter-iim-cost/1"
CALIBRATION_SCHEMA = "impact-hunter-iim-calibration/1"

MAX_PSI_EVALS_ENV = "IMPACT_HUNTER_MAX_PSI_EVALS"
SECONDS_PER_PSI_EVAL_ENV = "IMPACT_HUNTER_SECONDS_PER_PSI_EVAL"

# Default ceiling: 1e11 Psi evaluations, about 5.6e7 core-seconds (15,600
# core-hours, ~160 mi300a node-hours) at the reference rate below. Larger
# campaigns need an explicit decision (--hunter-allow-large or a higher
# --hunter-max-psi-evals / IMPACT_HUNTER_MAX_PSI_EVALS).
DEFAULT_MAX_PSI_EVALS = 10**11

# Reference rate: one CPU core, numba host kernel, exhaustive IIM (all
# mechanism/purview sizes and all cuts), T = 1000 time points. Measured on a
# workstation: 1.7 s at 4 nodes (5,000 evaluations), 46 s at 5 nodes (129,600)
# and about 27 min at 6 nodes (2,899,232), i.e. 3.4e-4, 3.5e-4 and 5.6e-4 s
# per evaluation. The largest (6-node) value is the conservative reference.
REFERENCE_SECONDS_PER_PSI_EVAL = 5.6e-4
REFERENCE_MEASUREMENTS = (
    {"n_nodes": 4, "wall_seconds": 1.7, "psi_evaluations": 5_000},
    {"n_nodes": 5, "wall_seconds": 46.0, "psi_evaluations": 129_600},
    {"n_nodes": 6, "wall_seconds": 1620.0, "psi_evaluations": 2_899_232},
)
REFERENCE_NOTE = (
    "one CPU core, numba host kernel, exhaustive IIM, T = 1000; measured "
    "1.7 s (4 nodes), 46 s (5 nodes), ~27 min (6 nodes)"
)

# A calibration with fewer evaluations than this is dominated by the fixed
# cost of a shard (imports, JIT compilation, worker start-up) and overstates
# the time per evaluation.
MIN_CALIBRATION_PSI_EVALS = 1_000_000

UNIT_DEFINITION = (
    "one (mechanism, purview, mechanism bipartition, purview bipartition) term "
    "of Psi under one TPM (the intact TPM or one system-cut TPM)"
)


class HunterCostError(ValueError):
    """A campaign configuration exceeds the Psi-evaluation ceiling."""


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


def bipartitions_of_size(k: int) -> int:
    """Unordered bipartitions of a k-node set (``_iim_enumerate_bipartitions``)."""
    k = int(k)
    return 0 if k < 2 else 2 ** (k - 1) - 1


def _size_cap(n_nodes: int, max_size) -> int:
    n = int(n_nodes)
    return n if max_size is None else max(0, min(int(max_size), n))


def n_subsets(n_nodes: int, max_size=None) -> int:
    """Subsets of 1..max_size nodes (``_iim_enumerate_subsets``)."""
    n = int(n_nodes)
    return sum(comb(n, k) for k in range(1, _size_cap(n, max_size) + 1))


def subset_bipartition_sum(n_nodes: int, max_size=None) -> int:
    """Sum of ``bip(|S|)`` over the subsets of 1..max_size nodes."""
    n = int(n_nodes)
    return sum(
        comb(n, k) * bipartitions_of_size(k)
        for k in range(2, _size_cap(n, max_size) + 1)
    )


def subset_range_bipartition_sum(n_nodes: int, max_size, start: int, stop: int) -> int:
    """
    Sum of ``bip(|S|)`` over the subsets with enumeration index in
    [start, stop). Subsets are enumerated by size, smallest first, as
    ``_iim_enumerate_subsets`` does, so a mechanism shard is a slice of them.
    """
    n = int(n_nodes)
    total = 0
    offset = 0
    for k in range(1, _size_cap(n, max_size) + 1):
        block = comb(n, k)
        lo, hi = max(int(start), offset), min(int(stop), offset + block)
        if hi > lo:
            total += (hi - lo) * bipartitions_of_size(k)
        offset += block
    return total


def n_system_cuts(
    n_nodes: int, partition_mode: str = "all", cut_mode: str = "bidirectional"
) -> int:
    """System cuts of an n-node subsystem (``_iim_all_system_cuts``)."""
    n = int(n_nodes)
    if n < 2:
        return 0
    if str(partition_mode) == "balanced":
        cuts = comb(n, n // 2) // 2 if n % 2 == 0 else comb(n, (n - 1) // 2)
    else:
        cuts = 2 ** (n - 1) - 1
    return 2 * cuts if str(cut_mode) == "directional" else cuts


def planned_iim_structure(
    n_regions: int,
    *,
    bins: int,
    max_nodes=None,
    max_state_space: int = 1500,
    state_budget_policy: str = "reduce_bins_first",
    n_bearer=None,
) -> dict:
    """
    Subsystem size and bins ``prepare_iim_problem`` will use for a run with
    ``n_regions`` regions (its node cap and state-budget loop), without
    touching the data. ``defined`` is False when preparation would return an
    undefined problem for a structural reason; data-dependent reasons (too few
    time points) are not detected, so counts based on this are upper bounds.
    """
    n_regions = int(n_regions)
    if n_regions < 2:
        return {"defined": False, "reason": "insufficient_shape"}
    if n_bearer is not None and int(n_bearer) < 2:
        return {"defined": False, "reason": "insufficient_bearer_nodes"}
    n_candidates = n_regions if n_bearer is None else int(n_bearer)
    n_sel = n_candidates if max_nodes is None else min(int(max_nodes), n_candidates)
    eff_bins = int(bins)

    def _over_budget():
        return eff_bins ** n_sel > int(max_state_space)

    if n_sel >= 2 and _over_budget() and state_budget_policy == "error":
        return {"defined": False, "reason": "state_space_too_large"}
    while n_sel >= 2 and _over_budget():
        if eff_bins > 2 and (state_budget_policy == "reduce_bins_first" or n_sel <= 2):
            eff_bins -= 1
        else:
            n_sel -= 1
    if n_sel < 2:
        return {"defined": False, "reason": "state_space_too_large"}
    return {"defined": True, "n_nodes": int(n_sel), "bins": int(eff_bins)}


def iim_run_cost(
    n_nodes: int,
    *,
    max_mechanism_size=None,
    max_purview_size=None,
    n_parts=None,
    cut_mode: str = "bidirectional",
    partition_mode: str = "all",
) -> dict:
    """Closed-form Psi-evaluation count of one IIM run on an n-node subsystem."""
    n = int(n_nodes)
    mech_bip = subset_bipartition_sum(n, max_mechanism_size)
    purv_bip = subset_bipartition_sum(n, max_purview_size)
    n_cuts = n_system_cuts(n, partition_mode, cut_mode)
    n_cuts_eval = n_cuts if n_parts is None else min(int(n_parts), n_cuts)
    per_tpm = mech_bip * purv_bip
    return {
        "n_nodes": n,
        "max_mechanism_size_used": _size_cap(n, max_mechanism_size),
        "max_purview_size_used": _size_cap(n, max_purview_size),
        "n_mechanisms": n_subsets(n, max_mechanism_size),
        "n_purviews": n_subsets(n, max_purview_size),
        "mechanism_bipartitions": mech_bip,
        "purview_bipartitions": purv_bip,
        "psi_evaluations_per_tpm": per_tpm,
        "n_cuts_total": n_cuts,
        "n_cuts_evaluated": n_cuts_eval,
        "tpm_evaluations": 1 + n_cuts_eval,
        "psi_evaluations": per_tpm * (1 + n_cuts_eval),
    }


def enumerated_run_cost(mechanisms, purviews, n_cuts_evaluated: int) -> dict:
    """Psi-evaluation count of a prepared problem, from its enumerations."""
    mech_bip = sum(bipartitions_of_size(len(m)) for m in mechanisms)
    purv_bip = sum(bipartitions_of_size(len(z)) for z in purviews)
    per_tpm = mech_bip * purv_bip
    n_cuts = int(n_cuts_evaluated)
    return {
        "n_mechanisms": len(mechanisms),
        "n_purviews": len(purviews),
        "mechanism_bipartitions": mech_bip,
        "purview_bipartitions": purv_bip,
        "psi_evaluations_per_tpm": per_tpm,
        "n_cuts_evaluated": n_cuts,
        "tpm_evaluations": 1 + n_cuts,
        "psi_evaluations": per_tpm * (1 + n_cuts),
    }


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def parse_count(text) -> int:
    """Positive integer count; accepts scientific notation such as ``1e12``."""
    try:
        value = float(str(text).strip())
    except ValueError as exc:
        raise ValueError(f"expected a positive count, got {text!r}") from exc
    if not math.isfinite(value) or value < 1:
        raise ValueError(f"expected a positive count, got {text!r}")
    return int(value)


def parse_rate(text) -> float:
    try:
        value = float(str(text).strip())
    except ValueError as exc:
        raise ValueError(f"expected seconds per evaluation > 0, got {text!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"expected seconds per evaluation > 0, got {text!r}")
    return value


def resolve_psi_eval_ceiling(value=None, env=None) -> tuple[int, str]:
    """Ceiling and its source: argument > IMPACT_HUNTER_MAX_PSI_EVALS > default."""
    env = os.environ if env is None else env
    if value is not None:
        return parse_count(value), "argument (--hunter-max-psi-evals)"
    raw = str(env.get(MAX_PSI_EVALS_ENV) or "").strip()
    if raw:
        return parse_count(raw), f"environment ({MAX_PSI_EVALS_ENV})"
    return int(DEFAULT_MAX_PSI_EVALS), "default"


def resolve_seconds_per_psi_eval(value=None, env=None) -> tuple[float | None, str]:
    """
    Measured shard wall-seconds per evaluation: argument >
    IMPACT_HUNTER_SECONDS_PER_PSI_EVAL > None (use the reference rate).
    """
    env = os.environ if env is None else env
    if value is not None:
        return parse_rate(value), "measured (--hunter-seconds-per-psi-eval)"
    raw = str(env.get(SECONDS_PER_PSI_EVAL_ENV) or "").strip()
    if raw:
        return parse_rate(raw), f"measured ({SECONDS_PER_PSI_EVAL_ENV})"
    return None, "reference"


# ---------------------------------------------------------------------------
# Estimate, guard and report
# ---------------------------------------------------------------------------


def runtime_estimate(
    *,
    psi_evaluations: int,
    max_phase1_shard_psi_evaluations: int,
    max_cut_shard_psi_evaluations: int,
    workers_per_task: int,
    shards_per_node: int,
    seconds_per_psi_evaluation=None,
    rate_source: str = "reference",
    phase1_walltime_seconds=None,
    cut_walltime_seconds=None,
) -> dict:
    """
    Order-of-magnitude runtime. With a measured rate (shard wall-seconds per
    evaluation, from ``cost_calibration.json`` of a calibration run with the
    same kernel and packing) the shard seconds are ``evaluations x rate``.
    Without one, the reference single-core rate is divided by the workers per
    shard (ideal speed-up of the host kernel; the device kernel on the APU is
    not covered by the reference and must be measured).
    """
    workers = max(1, int(workers_per_task or 1))
    spn = max(1, int(shards_per_node or 1))
    evals = int(psi_evaluations)
    if seconds_per_psi_evaluation is None:
        shard_rate = REFERENCE_SECONDS_PER_PSI_EVAL / workers
        basis = (
            f"reference rate {REFERENCE_SECONDS_PER_PSI_EVAL:.2g} s per evaluation "
            f"({REFERENCE_NOTE}), divided by {workers} worker(s) per shard; not "
            "measured on Hunter"
        )
        rate_source = "reference"
    else:
        shard_rate = float(seconds_per_psi_evaluation)
        basis = f"measured {shard_rate:.3g} shard wall-seconds per evaluation"
    shard_seconds = evals * shard_rate
    out = {
        "order_of_magnitude_only": True,
        "rate_source": str(rate_source),
        "basis": basis,
        "shard_seconds_per_psi_evaluation": shard_rate,
        "reference_seconds_per_psi_evaluation_one_core": REFERENCE_SECONDS_PER_PSI_EVAL,
        "single_core_hours_at_reference_rate": evals
        * REFERENCE_SECONDS_PER_PSI_EVAL
        / 3600.0,
        "shard_hours": shard_seconds / 3600.0,
        "node_hours": shard_seconds / spn / 3600.0,
        "workers_per_task": workers,
        "shards_per_node": spn,
        "longest_phase1_shard_seconds": int(max_phase1_shard_psi_evaluations)
        * shard_rate,
        "longest_cut_shard_seconds": int(max_cut_shard_psi_evaluations) * shard_rate,
        "phase1_walltime_seconds": phase1_walltime_seconds,
        "cut_walltime_seconds": cut_walltime_seconds,
        "stages_exceeding_walltime": [],
    }
    for stage, key, limit in (
        ("phase1-shard", "longest_phase1_shard_seconds", phase1_walltime_seconds),
        ("cut-shard", "longest_cut_shard_seconds", cut_walltime_seconds),
    ):
        if limit is not None and out[key] > float(limit):
            out["stages_exceeding_walltime"].append(stage)
    return out


def apply_ceiling(
    estimate: dict, *, ceiling: int, source: str, allow_large: bool
) -> dict:
    """Attach the guard decision to ``estimate`` (does not raise)."""
    exceeds = int(estimate["psi_evaluations"]) > int(ceiling)
    estimate["guard"] = {
        "max_psi_evaluations": int(ceiling),
        "source": str(source),
        "allow_large": bool(allow_large),
        "exceeds_ceiling": bool(exceeds),
        "refused": bool(exceeds and not allow_large),
    }
    return estimate


def check_ceiling(estimate: dict) -> None:
    """Raise HunterCostError when the guard refuses ``estimate``."""
    guard = estimate.get("guard") or {}
    if not guard.get("refused"):
        return
    rt = estimate.get("runtime") or {}
    raise HunterCostError(
        f"Hunter IIM campaign refused: an estimated "
        f"{estimate['psi_evaluations']:.3g} Psi evaluations exceed the ceiling of "
        f"{guard['max_psi_evaluations']:.3g} ({guard['source']}). Order of "
        f"magnitude: {rt.get('single_core_hours_at_reference_rate', float('nan')):.3g} "
        f"single-core hours, about {rt.get('node_hours', float('nan')):.3g} node-hours "
        f"({rt.get('basis', '')}). Exhaustive IIM grows roughly 30x per added "
        "node. Size the configuration first (docs/HLRS_HUNTER_RUNBOOK.md, section "
        "10a): lower --iim-max-nodes, or, if the author has declared it, "
        "--iim-max-mechanism-size / --iim-max-purview-size / --iim-n-parts, or "
        "fewer surrogate/bootstrap runs. To build anyway, raise the ceiling "
        f"(--hunter-max-psi-evals or {MAX_PSI_EVALS_ENV}) or pass "
        "--hunter-allow-large."
    )


def _fmt_count(x) -> str:
    x = int(x)
    return f"{x:,}" if x < 10**7 else f"{x:.3g}"


def _fmt_seconds(s) -> str:
    """Duration with two significant digits in a readable unit."""
    s = float(s)
    for limit, div, unit in (
        (120, 1, "s"),
        (7200, 60, "min"),
        (172800, 3600, "h"),
        (2 * 365 * 86400, 86400, "days"),
        (math.inf, 365 * 86400, "years"),
    ):
        if s < limit:
            v = s / div
            if v < 10:
                return f"{v:.2g} {unit}"
            digits = int(math.floor(math.log10(v))) - 1
            return f"{round(v, -digits):,.0f} {unit}"
    return f"{s:.2g} s"  # pragma: no cover (NaN)


def format_cost_summary(estimate: dict) -> list[str]:
    """Human-readable lines of a campaign cost estimate (for the build log)."""
    runs = estimate.get("runs") or {}
    lines = [
        (
            f"IIM cost estimate ({estimate.get('basis')}): "
            f"{runs.get('real', 0)} real run(s) x (1 + {runs.get('null_per_run', 0)} "
            f"surrogate + {runs.get('bootstrap_per_run', 0)} bootstrap) = "
            f"{runs.get('total_iim_runs', 0)} IIM runs; "
            f"{_fmt_count(estimate['psi_evaluations'])} Psi evaluations in "
            f"{_fmt_count(estimate.get('tpm_evaluations', 0))} TPM evaluations "
            f"(unit: {UNIT_DEFINITION})."
        )
    ]
    for group in estimate.get("structures") or []:
        if not group.get("defined", True):
            lines.append(
                f"  {group['n_real_runs']} real run(s) undefined "
                f"({group.get('reason')}): no Psi evaluations."
            )
            continue
        lines.append(
            f"  {group['n_real_runs']} real run(s) with {group['n_nodes']} nodes, "
            f"{group['bins']} bins: {group['n_mechanisms']} mechanisms "
            f"({group['mechanism_bipartitions']} bipartitions) x {group['n_purviews']} "
            f"purviews ({group['purview_bipartitions']}) = "
            f"{_fmt_count(group['psi_evaluations_per_tpm'])} per TPM x "
            f"{group['tpm_evaluations']} TPMs (intact + {group['n_cuts_evaluated']} of "
            f"{group['n_cuts_total']} cuts) = {_fmt_count(group['psi_evaluations'])} "
            "per IIM run."
        )
    rt = estimate.get("runtime") or {}
    if rt:
        lines.append(
            f"  Runtime (order of magnitude only; {rt['basis']}): "
            f"{rt['single_core_hours_at_reference_rate']:.3g} single-core hours at the "
            f"reference rate; about {rt['node_hours']:.3g} node-hours with "
            f"{rt['shards_per_node']} shard(s) per node; longest phase-1 shard "
            f"{_fmt_seconds(rt['longest_phase1_shard_seconds'])}, longest cut shard "
            f"{_fmt_seconds(rt['longest_cut_shard_seconds'])}."
        )
    guard = estimate.get("guard") or {}
    if guard:
        verdict = "within the ceiling"
        if guard.get("exceeds_ceiling"):
            verdict = (
                "ABOVE the ceiling, built because --hunter-allow-large was given"
                if guard.get("allow_large")
                else "ABOVE the ceiling: refused"
            )
        lines.append(
            f"  Ceiling: {guard['max_psi_evaluations']:.3g} Psi evaluations "
            f"({guard['source']}): {verdict}."
        )
    return lines


# ---------------------------------------------------------------------------
# Calibration from measured shard timings
# ---------------------------------------------------------------------------


def _record_psi_evaluations(record) -> int:
    """Psi evaluations of a timing record; 0 for older or malformed records."""
    if not isinstance(record, dict):
        return 0
    try:
        return max(0, int(record.get("psi_evaluations") or 0))
    except (TypeError, ValueError):
        return 0


def calibration_from_records(records) -> dict:
    """
    Measured rate from shard timing records (``wall_seconds``,
    ``cpu_seconds``, ``psi_evaluations`` per record, as written by the shard
    stages). Returns totals and ``shard_wall_seconds_per_psi_evaluation``,
    the value to pass as ``--hunter-seconds-per-psi-eval``. Records without
    evaluations (older builds, skipped shards, malformed files) are ignored.
    """
    rows = [r for r in records if _record_psi_evaluations(r) > 0]
    evals = sum(int(r["psi_evaluations"]) for r in rows)
    wall = sum(float(r.get("wall_seconds") or 0.0) for r in rows)
    cpu = sum(
        float(r.get("cpu_seconds", r.get("process_cpu_seconds")) or 0.0) for r in rows
    )
    per_shard = sorted(
        float(r.get("wall_seconds") or 0.0) / int(r["psi_evaluations"]) for r in rows
    )
    out = {
        "n_shards": len(rows),
        "psi_evaluations": int(evals),
        "wall_seconds": wall,
        "cpu_seconds": cpu,
        "shard_wall_seconds_per_psi_evaluation": (wall / evals) if evals else None,
        "cpu_seconds_per_psi_evaluation": (cpu / evals) if evals else None,
        "per_shard_wall_seconds_per_psi_evaluation": (
            None
            if not per_shard
            else {
                "min": per_shard[0],
                "median": per_shard[len(per_shard) // 2],
                "max": per_shard[-1],
            }
        ),
        "overhead_dominated": bool(evals < MIN_CALIBRATION_PSI_EVALS),
    }
    return out


# ---------------------------------------------------------------------------
# Command line: size a configuration without building a campaign
# ---------------------------------------------------------------------------


def _parse_nodes(text):
    """argparse type: subsystem sizes '4-10' or '5,6,7' (each >= 2)."""
    text = str(text).strip()
    try:
        if "-" in text:
            lo, hi = (int(x) for x in text.split("-", 1))
            nodes = list(range(lo, hi + 1))
        else:
            nodes = [int(x) for x in text.split(",") if x.strip()]
    except ValueError:
        nodes = []
    if not nodes or min(nodes) < 2:
        raise argparse.ArgumentTypeError(
            f"expected node counts >= 2 such as 4-10 or 5,6,7, got {text!r}"
        )
    return nodes


def _opt_size(text):
    """argparse type: 'all' (None, exhaustive) or an integer >= 1."""
    if str(text).strip().lower() in {"all", "none", ""}:
        return None
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        raise argparse.ArgumentTypeError(
            f"expected 'all' or an integer >= 1, got {text!r}"
        )
    return value


def _count_arg(minimum):
    def _parse(text):
        try:
            value = int(text)
        except ValueError:
            value = minimum - 1
        if value < minimum:
            raise argparse.ArgumentTypeError(
                f"expected an integer >= {minimum}, got {text!r}"
            )
        return value

    return _parse


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m impact_pipeline.hunter_cost",
        description=(
            "Psi-evaluation count and order-of-magnitude runtime of a Hunter IIM "
            "campaign, per subsystem size (no data needed)."
        ),
    )
    ap.add_argument(
        "--nodes", type=_parse_nodes, default="4-10", help="e.g. 4-10 or 5,6,7"
    )
    ap.add_argument("--max-mechanism-size", type=_opt_size, default="all")
    ap.add_argument("--max-purview-size", type=_opt_size, default="all")
    ap.add_argument(
        "--n-parts",
        type=_opt_size,
        default="all",
        help="sampled cuts (all = exhaustive)",
    )
    ap.add_argument("--cut-mode", default="bidirectional",
                    choices=("bidirectional", "directional"))
    ap.add_argument("--runs", type=_count_arg(1), default=1, help="real runs")
    ap.add_argument(
        "--null", type=_count_arg(0), default=0, help="K surrogate runs per run"
    )
    ap.add_argument(
        "--boot", type=_count_arg(0), default=0, help="B bootstrap runs per run"
    )
    ap.add_argument("--workers-per-task", type=_count_arg(1), default=22)
    ap.add_argument("--shards-per-node", type=_count_arg(1), default=4)
    ap.add_argument(
        "--seconds-per-psi-eval",
        type=parse_rate,
        default=None,
        help="measured shard wall-seconds per evaluation (cost_calibration.json)",
    )
    ap.add_argument("--json", action="store_true", help="print JSON rows")
    args = ap.parse_args(argv)

    factor = int(args.runs) * (1 + int(args.null) + int(args.boot))
    rows = []
    for n in args.nodes:
        cost = iim_run_cost(
            n,
            max_mechanism_size=args.max_mechanism_size,
            max_purview_size=args.max_purview_size,
            n_parts=args.n_parts,
            cut_mode=args.cut_mode,
        )
        total = cost["psi_evaluations"] * factor
        rt = runtime_estimate(
            psi_evaluations=total,
            max_phase1_shard_psi_evaluations=0,
            max_cut_shard_psi_evaluations=0,
            workers_per_task=args.workers_per_task,
            shards_per_node=args.shards_per_node,
            seconds_per_psi_evaluation=args.seconds_per_psi_eval,
            rate_source=(
                "reference" if args.seconds_per_psi_eval is None else "measured"
            ),
        )
        rows.append(
            {
                **cost,
                "iim_runs": factor,
                "campaign_psi_evaluations": total,
                "single_core_seconds_per_run_at_reference_rate": cost[
                    "psi_evaluations"
                ]
                * REFERENCE_SECONDS_PER_PSI_EVAL,
                "node_hours": rt["node_hours"],
                "exceeds_default_ceiling": total > DEFAULT_MAX_PSI_EVALS,
            }
        )
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    print(
        f"{'nodes':>5} {'per TPM':>13} {'TPMs':>6} {'per run':>11} "
        f"{'1-core/run':>11} {'campaign':>11} {'node-h':>9}  ceiling"
    )
    for r in rows:
        print(
            f"{r['n_nodes']:>5} {r['psi_evaluations_per_tpm']:>13,} "
            f"{r['tpm_evaluations']:>6} {r['psi_evaluations']:>11.3g} "
            f"{_fmt_seconds(r['single_core_seconds_per_run_at_reference_rate']):>11} "
            f"{r['campaign_psi_evaluations']:>11.3g} {r['node_hours']:>9.3g}  "
            f"{'ABOVE' if r['exceeds_default_ceiling'] else 'ok'}"
        )
    print(
        f"({factor} IIM runs; rate: "
        + (
            f"measured {args.seconds_per_psi_eval:.3g} s per evaluation per shard"
            if args.seconds_per_psi_eval is not None
            else f"reference {REFERENCE_SECONDS_PER_PSI_EVAL:.2g} s per evaluation on "
            f"one core / {args.workers_per_task} workers"
        )
        + f"; default ceiling {DEFAULT_MAX_PSI_EVALS:.0e}; order of magnitude only)"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
