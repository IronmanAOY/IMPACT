"""
Declarative evaluator of the MPC-Bench v2 hypotheses.

The hypotheses, their parts, decision rules, thresholds and data selections
are declared in ``protocols/v2/hypotheses_v2.json`` (schema
``mpc-bench-hypotheses/2``); this module interprets that file on the bench
records (result schema ``mpc-bench-result/3``) and the auxiliary inputs
(manipulation checks, registry v3 admission entries, the frozen family
protocols). Nothing in a decision is coded per hypothesis here: a part names
one rule of :data:`RULES`, its parameters and a data selection, so the file
is the single statement of every decision, and the preregistration quotes it.

Outcomes
--------
``SUPPORTED``, ``FALSIFIED``, ``INDETERMINATE`` (the rule does not decide),
``NOT_EVALUABLE`` (missing input, anchor or manipulation check) and
``NOT_TESTABLE_BY_DESIGN`` (declared before the freeze from attainable
precision: the gate of a part says that its development rate ``pi0`` or
``pi1`` is below 0.9, so the part is replaced by its weaker claim). A part
is *decisive* (its outcome enters the hypothesis) or *reported*. A part
*removed* before the freeze is not evaluated; it is listed in the tally with
its development outcome as the predicted outcome. A hypothesis combines its
decisive parts as v1 did: FALSIFIED if any evaluable decisive part is
FALSIFIED, SUPPORTED if every evaluable one is SUPPORTED, else INDETERMINATE;
NOT_EVALUABLE when no decisive part is evaluable (NOT_TESTABLE_BY_DESIGN
when every decisive part is gated). NOT_EVALUABLE and
NOT_TESTABLE_BY_DESIGN parts are listed, never dropped.

Decision rules (design section 4.9)
-----------------------------------
Rates are counted over clusters: every re-scoring of one simulation (the R
and H declarations, cut modes, views, estimator forms) is one cluster, and a
cluster contributes the share of its rows with the event, so a cell has the
effective count ``k = sum of cluster shares`` out of ``n = clusters`` (one
row per cluster gives the plain count). This treats a cluster as a single
trial, which is conservative for every Clopper-Pearson (CP) bound below.

* ``h0_cell`` (rate <= bound): FALSIFIED if some cell's one-sided CP lower
  bound at ``alpha / m`` exceeds the bound (``m`` cells); SUPPORTED if every
  pool's (by default: principle's) pooled upper bound at ``alpha / P``
  (``P`` pools) is below the bound; else INDETERMINATE.
* ``h0_cell_exceeds``: the mirror of ``h0_cell`` for a predicted failure
  (SUPPORTED iff the H0 cell rule would be FALSIFIED, FALSIFIED iff it would
  be SUPPORTED).
* ``three_zone`` ("in >= x of seeds"): SUPPORTED iff the point rate >= x;
  FALSIFIED iff the CP upper bound at ``alpha / m`` < x; else INDETERMINATE.
* ``three_zone_at_most`` ("in <= x"): SUPPORTED iff the point rate <= x;
  FALSIFIED iff the CP lower bound at ``alpha / m`` > x.
* ``reversed_three_zone`` (predicted failure "< x"): SUPPORTED iff the CP
  upper bound at ``alpha`` < x; FALSIFIED iff the point rate >= x.
* ``count_at_most`` ("<= 2/20"): SUPPORTED iff the point rate <= max_rate;
  FALSIFIED iff the CP lower bound at ``alpha`` exceeds ``falsify_above``.
* ``demonstration``: SUPPORTED iff the CP upper bound at ``level`` is below
  the bound; FALSIFIED when it is not and the planned run count is reached
  (or the events already make the demonstration impossible at the planned
  count: curtailment, which only stops towards non-admission).
* ``any_event_clusters`` (rare-event claims): FALSIFIED if some class's CP
  lower bound at ``alpha / m`` exceeds the bound; SUPPORTED if the "any
  event" seed-cluster rate (a cluster is an event if any of its rows is) has
  CP upper bound at ``alpha`` below the bound; else INDETERMINATE.
* ``rate_lower_bound`` ("more often than x"): SUPPORTED iff the CP lower
  bound at ``alpha`` exceeds x; FALSIFIED iff the CP upper bound at
  ``alpha`` is below x (credibly not more often than x, so a correct
  estimator at the boundary is falsified with probability <= alpha).
* ``usable_share`` (prerequisite M): usable iff passes >= ceil(share x n).
* ``false_exclusion`` (HR2): FALSIFIED if some cell's share of seeds with an
  event has CP lower bound (``alpha / m``) above ``seed_falsify`` or its
  cluster-bootstrap lower bound of the per-row rate exceeds ``row_bound``;
  SUPPORTED if every cell's per-row point rate is <= ``row_bound`` and its
  event-seed share has CP upper bound (``alpha / m``) below ``seed_bound``.
* paired and value tests: ``median_threshold``; ``hodges_lehmann`` (sample
  median against a threshold and the one-sided Hodges-Lehmann lower bound
  from the exact signed-rank distribution); ``wilcoxon`` (one-sided signed
  rank, exact without ties up to 50 pairs, else normal with tie and
  continuity correction); ``sign_test`` (exact binomial);
  ``spearman_ols`` (Spearman's rho with the t approximation and the OLS slope
  on the dose rescaled to [0, 1]); ``jonckheere_terpstra`` (ordered
  alternative; exact null distribution without ties up to 200 values, else
  normal with the tie-corrected variance); ``spearman_monotone``;
  ``stepwise_sign``; ``newcombe_includes_zero`` (Newcombe's hybrid score
  interval of a difference of two proportions).
* calibration: ``kappa_null`` (interval calibration at the null),
  ``kappa_twins`` (SE calibration against white-box twins: pooled
  within-network SD over the RMS of the SE, chi-square interval on
  ``sum (n_twin - 1)`` degrees of freedom, and the ``q_A`` tails),
  ``concordance_twins``.
* admission and anchors: ``admission_matches``, ``anchor_replicates``,
  ``fmd_concordance``.
* ``describe``: a reported summary without a decision.

A test-type part ("at one-sided p < 0.05") is SUPPORTED when the stated
criterion holds and FALSIFIED otherwise (v1 convention).

Data selections
---------------
A part's ``data`` names a source (``components``: one row per task x
scoring x component of the ``/3`` records; ``verdicts``: one row per
scoring; or any auxiliary source such as ``manipulation`` and ``registry``),
a filter (``where``), optional pairing (``pair``: rows a and b joined on
key fields, ``delta = value(a) - value(b)``), unions of members with their
own filters and events, cells, clusters, pools, the event predicate and the
value, dose and group fields. Field names resolve through the file's alias
table (``@name``), a dotted path into the row (``details.te_in.z``,
``config.sweep_level``) or a derived field (``reason_code``, ``excess``,
``own_bit``, ``bit.<P>``, ...). Predicates are three-valued: a comparison
with a missing value is unknown, and a row whose event is unknown is left
out of the counts (and counted as missing), never counted as a non-event.
The calibration rules read the SE and df of a row from the selection's
``se`` and ``df`` fields when it names them (NAS is calibrated per
direction, because its component ``c`` is the smaller of the two
directions), else from ``se_c`` and ``df_c``.

Prerequisite M is applied per row: a part's ``requires`` names switch or
realisation checks (``usable``; check ids may be templates on the row) and,
with ``oracle``, the checks of the row's system in the file's
``oracle_checks`` (both systems of a paired row). A row whose check is not
usable in its family is dropped; a cell that loses every row is listed as
``NOT_EVALUABLE`` (reason ORACLE), and the part is NOT_EVALUABLE when no
row is left.

Also here: the one map between the names of the v2 protocol files and the
names of the v2 estimators (:data:`PROTOCOL_ESTIMATOR_NAMES`), and the one
convention of protocol keys (a record's ``protocol_id`` is the key, the
frozen file is ``mpc_bench_v2_<key>.json``; :func:`load_protocol_files`),
used by the evaluator, the integrity audit and the runner.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from scipy import stats as _st

from impact_pipeline.v2 import PRINCIPLES
from impact_pipeline.v2 import reasons as R
from impact_pipeline.v2 import seeds as S

SPEC_SCHEMA = "mpc-bench-hypotheses/2"
SPEC_PATH = (
    Path(__file__).resolve().parents[3] / "protocols" / "v2" / "hypotheses_v2.json"
)
REPORT_VERSION = "mpc-bench-hypotheses-evaluator/2.0.0"

SUPPORTED, FALSIFIED, INDETERMINATE = "SUPPORTED", "FALSIFIED", "INDETERMINATE"
NOT_EVALUABLE, NOT_TESTABLE_BY_DESIGN = "NOT_EVALUABLE", "NOT_TESTABLE_BY_DESIGN"
REPORTED, REMOVED, NOT_RUN = "REPORTED", "REMOVED", "NOT_RUN"
OUTCOMES = (SUPPORTED, FALSIFIED, INDETERMINATE, NOT_EVALUABLE, NOT_TESTABLE_BY_DESIGN)
TALLY_KEYS = OUTCOMES + (REMOVED,)
EVALUABLE = (SUPPORTED, FALSIFIED, INDETERMINATE)
DECISIVE, REPORTED_ROLE = "decisive", "reported"
ROLES = (DECISIVE, REPORTED_ROLE)
ACTIVE, REMOVED_STATUS = "active", "removed"
PART_STATUSES = (ACTIVE, REMOVED_STATUS)
LABELS = ("C", "R", "HO")
DEVELOPMENT_FLAG = "DEVELOPMENT - NOT A RESULT"
SIDE_CONSERVATIVE, SIDE_ANTI = "conservative", "anti_conservative"
_TOL = 1e-12


class SpecError(ValueError):
    """A hypotheses file that does not follow ``mpc-bench-hypotheses/2``."""


# ==========================================================================
# protocol names <-> estimator names (one place)
# ==========================================================================
# The v2 template protocol names RAM-PE's untested facets by their short
# names (F, G, speed) and the PDI content bearer by the option
# ``pdi_bearer``; the estimators (ram_v3, pdi_v3) use long names. The map is
# here once; every consumer (runner, evaluator, audit) translates through
# :func:`ram_facets_for_estimator` and :func:`pdi_params_for_estimator`.
RAM_FACET_NAMES = {"G": "goal_alignment", "F": "feedback_magnitude", "speed": "speed"}
PDI_OPTION_NAMES = {"pdi_bearer": "bearer"}
# protocol-only keys of the PDI block (not estimator parameters)
PDI_PROTOCOL_ONLY_KEYS = ("mode",)
PROTOCOL_ESTIMATOR_NAMES = {
    "RAM": {"facets_not_applicable": dict(RAM_FACET_NAMES)},
    "PDI": dict(PDI_OPTION_NAMES),
}


def ram_facets_for_estimator(protocol_facets: Mapping) -> dict:
    """The protocol's ``estimators.RAM.facets_not_applicable`` block
    (``{"F": reason, "G": reason, "speed": reason}``) in the estimator's
    names (``ram_v3.FACETS_NOT_APPLICABLE`` keys). Unknown facet names are
    refused."""
    out = {}
    for k, v in dict(protocol_facets or {}).items():
        if k not in RAM_FACET_NAMES:
            raise ValueError(
                f"unknown RAM facet {k!r}; one of {sorted(RAM_FACET_NAMES)}"
            )
        out[RAM_FACET_NAMES[k]] = v
    return out


def ram_facets_for_protocol(estimator_facets: Mapping) -> dict:
    """The inverse of :func:`ram_facets_for_estimator`."""
    inv = {v: k for k, v in RAM_FACET_NAMES.items()}
    out = {}
    for k, v in dict(estimator_facets or {}).items():
        if k not in inv:
            raise ValueError(f"unknown RAM facet {k!r}; one of {sorted(inv)}")
        out[inv[k]] = v
    return out


def pdi_params_for_estimator(protocol_block: Mapping) -> dict:
    """The keyword arguments of ``pdi_v3.PDIParams`` from the protocol's
    ``estimators.PDI`` block: ``pdi_bearer`` becomes ``bearer``; the
    protocol-only key ``mode`` is dropped; every other key is passed
    unchanged (``PDIParams.from_mapping`` refuses unknown ones)."""
    out = {}
    for k, v in dict(protocol_block or {}).items():
        if k in PDI_PROTOCOL_ONLY_KEYS:
            continue
        out[PDI_OPTION_NAMES.get(k, k)] = v
    return out


# ==========================================================================
# protocol keys (one place)
# ==========================================================================
# A family protocol is identified by its key, ``<family protocol>`` or
# ``<family protocol>+<form>`` (``A-R``, ``A-H+nas_secondary``,
# ``hopf-eeg64``): the v2 runner writes the key as ``protocol_id`` into every
# scoring and component, its frozen file is ``mpc_bench_v2_<key>.json``, and
# a protocol named after the convention is named ``mpc-bench-v2-<key>`` (a
# development draft ``mpc-bench-v2-<key>-draft``). The runner, the evaluator
# and the integrity audit read protocol files only through
# :func:`load_protocol_files`, so a record's ``protocol_id`` is the key its
# protocol is looked up by.
PROTOCOL_FILE_PREFIX = "mpc_bench_v2_"
PROTOCOL_FILE_SUFFIX = ".json"
PROTOCOL_FILE_GLOB = f"{PROTOCOL_FILE_PREFIX}*{PROTOCOL_FILE_SUFFIX}"
_PROTOCOL_NAME_RE = re.compile(r"^mpc-bench-v2-(?P<key>.+?)(?P<draft>-draft)?$")


def protocol_file_name(key: str) -> str:
    """The file name of a protocol key (``mpc_bench_v2_A-R.json``)."""
    key = str(key)
    if not key or "/" in key or key.strip() != key:
        raise ValueError(f"not a protocol key: {key!r}")
    return f"{PROTOCOL_FILE_PREFIX}{key}{PROTOCOL_FILE_SUFFIX}"


def protocol_key_of_file(path) -> str:
    """The protocol key of a protocol file named by the convention;
    ``ValueError`` for any other file name."""
    name = Path(path).name
    pre, suf = PROTOCOL_FILE_PREFIX, PROTOCOL_FILE_SUFFIX
    if not (name.startswith(pre) and name.endswith(suf)
            and len(name) > len(pre) + len(suf)):
        raise ValueError(f"{name}: a family protocol file is named "
                         f"{PROTOCOL_FILE_PREFIX}<key>{PROTOCOL_FILE_SUFFIX}")
    return name[len(PROTOCOL_FILE_PREFIX): -len(PROTOCOL_FILE_SUFFIX)]


def protocol_key_of_name(name) -> Optional[str]:
    """The key in a protocol name that follows the convention
    (``mpc-bench-v2-<key>`` or ``mpc-bench-v2-<key>-draft``), else None."""
    m = _PROTOCOL_NAME_RE.match(str(name or ""))
    return None if m is None else m.group("key")


def protocol_key(proto, path=None) -> str:
    """The key of a protocol: from its file name when it has a file, else
    from its name; a name that follows the convention must agree with the
    file. ``ValueError`` when neither gives a key."""
    name = proto.get("name") if isinstance(proto, Mapping) else getattr(
        proto, "name", None)
    by_name = protocol_key_of_name(name)
    if path is not None:
        key = protocol_key_of_file(path)
        if by_name is not None and by_name != key:
            raise ValueError(f"{Path(path).name}: the protocol is named {name!r}, "
                             f"the file names the key {key!r}")
        return key
    if by_name is None:
        raise ValueError(f"protocol {name!r} does not carry a key "
                         "(mpc-bench-v2-<key>)")
    return by_name


def load_protocol_files(items) -> "OrderedDict[str, Any]":
    """``{key: ProtocolV3}`` of protocol files and directories (a directory
    contributes its ``mpc_bench_v2_*.json`` files, so registries and anchor
    tables next to them are not read as protocols). A file outside the
    naming convention and a key given twice are refused."""
    from impact_pipeline import evidence_v2 as E

    paths = []
    for it in items or ():
        p = Path(it)
        if p.is_dir():
            paths.extend(sorted(p.glob(PROTOCOL_FILE_GLOB)))
        elif p.is_file():
            paths.append(p)
        else:
            raise FileNotFoundError(p)
    out: "OrderedDict[str, Any]" = OrderedDict()
    for p in paths:
        protocol_key_of_file(p)  # a file outside the convention is refused first
        pr = E.load_protocol(p)
        key = protocol_key(pr, p)
        if key in out:
            raise ValueError(f"protocol key {key!r} given twice")
        out[key] = pr
    return out


# ==========================================================================
# statistics
# ==========================================================================
def _finite(v) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def cp_upper(k, n, level=0.05) -> float:
    """One-sided ``1 - level`` Clopper-Pearson upper bound of ``k / n``
    (``k`` may be an effective, non-integer count); NaN for ``n = 0``."""
    k, n = float(k), float(n)
    if n <= 0:
        return float("nan")
    if k < -_TOL or k > n + _TOL:
        raise ValueError("need 0 <= k <= n")
    if k >= n - _TOL:
        return 1.0
    return float(_st.beta.ppf(1.0 - float(level), max(k, 0.0) + 1.0, n - k))


def cp_lower(k, n, level=0.05) -> float:
    """One-sided ``1 - level`` Clopper-Pearson lower bound; NaN for n = 0."""
    k, n = float(k), float(n)
    if n <= 0:
        return float("nan")
    if k < -_TOL or k > n + _TOL:
        raise ValueError("need 0 <= k <= n")
    if k <= _TOL:
        return 0.0
    return float(_st.beta.ppf(float(level), k, n - min(k, n) + 1.0))


def n_needed_zero_events(bound, level=0.05) -> int:
    """The smallest n whose CP upper bound with 0 events is below ``bound``
    (``1 - level**(1/n) < bound``)."""
    b, a = float(bound), float(level)
    if not (0 < b < 1 and 0 < a < 1):
        raise ValueError("bound and level must lie in (0, 1)")
    n = max(1, int(math.floor(math.log(a) / math.log(1.0 - b))) - 1)
    while cp_upper(0, n, a) >= b:
        n += 1
    while n > 1 and cp_upper(0, n - 1, a) < b:
        n -= 1
    return n


@dataclass(frozen=True)
class Counts:
    """Event counts of a set of rows: rows (``k_rows`` / ``n_rows``),
    clusters (``n_clusters``), the effective count ``k_eff`` (sum of the
    clusters' event shares) and ``k_any`` (clusters with an event)."""

    k_rows: int
    n_rows: int
    n_clusters: int
    k_eff: float
    k_any: int
    k_by_cluster: Tuple[float, ...] = ()
    n_by_cluster: Tuple[float, ...] = ()
    n_missing: int = 0

    @property
    def rate(self) -> float:
        return self.k_eff / self.n_clusters if self.n_clusters else float("nan")

    @property
    def row_rate(self) -> float:
        return self.k_rows / self.n_rows if self.n_rows else float("nan")

    def to_dict(self) -> dict:
        return {
            "k": self.k_rows,
            "n": self.n_rows,
            "clusters": self.n_clusters,
            "k_eff": self.k_eff,
            "k_any": self.k_any,
            "rate": _jsonf(self.rate),
            "n_missing": self.n_missing,
        }


def count_events(events: Sequence, clusters: Optional[Sequence] = None) -> Counts:
    """Counts from per-row events (True, False or None = unknown, left out)
    and cluster keys (default: every row its own cluster)."""
    ev = list(events)
    cl = list(range(len(ev))) if clusters is None else list(clusters)
    if len(cl) != len(ev):
        raise ValueError("events and clusters differ in length")
    by: "OrderedDict[Any, List[bool]]" = OrderedDict()
    missing = 0
    for e, c in zip(ev, cl):
        if e is None:
            missing += 1
            continue
        by.setdefault(c, []).append(bool(e))
    k_rows = sum(sum(v) for v in by.values())
    n_rows = sum(len(v) for v in by.values())
    kc = tuple(float(sum(v)) for v in by.values())
    nc = tuple(float(len(v)) for v in by.values())
    k_eff = float(sum(k / n for k, n in zip(kc, nc)))
    return Counts(
        int(k_rows),
        int(n_rows),
        len(by),
        k_eff,
        int(sum(1 for k in kc if k > 0)),
        kc,
        nc,
        missing,
    )


def wilson_interval(k, n, conf=0.95) -> Tuple[float, float]:
    """Wilson score interval of ``k / n`` (two-sided ``conf``)."""
    k, n = float(k), float(n)
    if n <= 0:
        return float("nan"), float("nan")
    z = float(_st.norm.ppf(1.0 - (1.0 - conf) / 2.0))
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe_difference(k1, n1, k2, n2, conf=0.95) -> Tuple[float, float, float]:
    """Newcombe's hybrid score interval (method 10, Newcombe 1998) of the
    difference ``k1/n1 - k2/n2``: ``(difference, lower, upper)``."""
    p1, p2 = float(k1) / float(n1), float(k2) / float(n2)
    l1, u1 = wilson_interval(k1, n1, conf)
    l2, u2 = wilson_interval(k2, n2, conf)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def signed_rank_pmf(n: int) -> np.ndarray:
    """Null distribution of the Wilcoxon signed-rank statistic ``T+`` for
    ``n`` untied non-zero differences: ``pmf[t] = P(T+ = t)``."""
    n = int(n)
    p = np.ones(1)
    for i in range(1, n + 1):
        q = np.zeros(p.size + i)
        q[: p.size] += 0.5 * p
        q[i:] += 0.5 * p
        p = q
    return p


def _clean(x) -> np.ndarray:
    a = np.asarray([v for v in (_finite(t) for t in x) if v is not None], dtype=float)
    return a


def wilcoxon_signed_rank(d, alternative="greater", exact_max_n=50) -> dict:
    """One-sided Wilcoxon signed-rank test of paired differences (zeros
    dropped, Wilcoxon's convention): ``T+`` and its p-value, exact without
    ties up to ``exact_max_n`` pairs, else the normal approximation with
    the tie correction and a continuity correction of 0.5."""
    x = _clean(d)
    x = x[x != 0]
    n = int(x.size)
    if n == 0:
        return {"n": 0, "t_plus": float("nan"), "p": float("nan"), "method": None}
    a = np.abs(x)
    ranks = _st.rankdata(a)
    t_plus = float(ranks[x > 0].sum())
    ties = np.unique(a).size < n
    if not ties and n <= exact_max_n:
        pmf = signed_rank_pmf(n)
        t = int(round(t_plus))
        p = (
            float(pmf[t:].sum())
            if alternative == "greater"
            else float(pmf[: t + 1].sum())
        )
        method = "exact"
    else:
        _, counts = np.unique(a, return_counts=True)
        mean = n * (n + 1) / 4.0
        var = (
            n * (n + 1) * (2 * n + 1) / 24.0
            - float(((counts**3) - counts).sum()) / 48.0
        )
        sd = math.sqrt(var)
        if alternative == "greater":
            z = (t_plus - mean - 0.5) / sd
            p = float(_st.norm.sf(z))
        else:
            z = (t_plus - mean + 0.5) / sd
            p = float(_st.norm.cdf(z))
        method = "normal"
    return {"n": n, "t_plus": t_plus, "p": min(1.0, p), "method": method}


def sign_test(d, alternative="greater") -> dict:
    """One-sided exact sign test of paired differences (zeros dropped)."""
    x = _clean(d)
    pos, neg = int((x > 0).sum()), int((x < 0).sum())
    n = pos + neg
    if n == 0:
        return {"n": 0, "positive": pos, "p": float("nan")}
    if alternative == "greater":
        p = float(_st.binom.sf(pos - 1, n, 0.5))
    else:
        p = float(_st.binom.cdf(pos, n, 0.5))
    return {"n": n, "positive": pos, "negative": neg, "p": p}


def _signed_rank_critical(n: int, alpha: float, exact_max_n: int = 50) -> Optional[int]:
    """The smallest t with ``P(T+ >= t) <= alpha`` (None when no t qualifies)."""
    m = n * (n + 1) // 2
    if n <= exact_max_n:
        pmf = signed_rank_pmf(n)
        tail = np.cumsum(pmf[::-1])[::-1]  # tail[t] = P(T+ >= t)
        ok = np.flatnonzero(tail <= alpha + 1e-15)
        return int(ok[0]) if ok.size else None
    mean = m / 2.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    t = int(math.ceil(mean + float(_st.norm.ppf(1 - alpha)) * sd + 0.5))
    return t if t <= m else None


def hodges_lehmann(d, alpha=0.05) -> dict:
    """The Hodges-Lehmann estimate of a paired location shift (median of the
    Walsh averages) and its one-sided ``1 - alpha`` lower bound, obtained by
    inverting the signed-rank test: ``W_(M - t_alpha + 1)`` with ``M =
    n (n + 1) / 2`` Walsh averages and ``t_alpha`` the smallest t with
    ``P(T+ >= t) <= alpha`` (``-inf`` when no such t exists)."""
    x = np.sort(_clean(d))
    n = int(x.size)
    if n == 0:
        return {
            "n": 0,
            "estimate": float("nan"),
            "lower": float("nan"),
            "median": float("nan"),
        }
    i, j = np.triu_indices(n)
    walsh = np.sort((x[i] + x[j]) / 2.0)
    m = walsh.size
    t_a = _signed_rank_critical(n, float(alpha))
    lower = float("-inf") if t_a is None else float(walsh[m - t_a])
    return {
        "n": n,
        "estimate": float(np.median(walsh)),
        "lower": lower,
        "median": float(np.median(x)),
        "t_alpha": t_a,
    }


def spearman(x, y, alternative="greater") -> dict:
    """Spearman's rho (average ranks) with the one-sided p-value of the t
    approximation (``t = rho sqrt((n - 2) / (1 - rho^2))``, ``n - 2`` df)."""
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[ok], ya[ok]
    n = int(xa.size)
    if n < 3 or np.ptp(xa) == 0 or np.ptp(ya) == 0:
        return {"n": n, "rho": float("nan"), "p": float("nan")}
    rx, ry = _st.rankdata(xa), _st.rankdata(ya)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    if abs(rho) >= 1.0 - 1e-15:
        p = 0.0 if (rho > 0) == (alternative == "greater") else 1.0
    else:
        t = rho * math.sqrt((n - 2) / (1.0 - rho * rho))
        p = (
            float(_st.t.sf(t, n - 2))
            if alternative == "greater"
            else float(_st.t.cdf(t, n - 2))
        )
    return {"n": n, "rho": rho, "p": p}


def ols_slope_unit(x, y) -> float:
    """OLS slope of y on x rescaled to [0, 1] (NaN without spread)."""
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[ok], ya[ok]
    if xa.size < 2 or np.ptp(xa) == 0:
        return float("nan")
    u = (xa - xa.min()) / np.ptp(xa)
    uc = u - u.mean()
    return float((uc * (ya - ya.mean())).sum() / (uc * uc).sum())


def mann_whitney_pmf(a: int, b: int) -> np.ndarray:
    """Null distribution of the Mann-Whitney count between samples of sizes
    ``a`` and ``b`` without ties (the normalised q-binomial coefficient)."""
    a, b = int(a), int(b)
    prev = [np.ones(1) for _ in range(b + 1)]
    for i in range(1, a + 1):
        cur = [np.ones(1)]
        for j in range(1, b + 1):
            out = np.zeros(i * j + 1)
            x = prev[j]
            out[j : j + x.size] += (i / (i + j)) * x
            y = cur[j - 1]
            out[: y.size] += (j / (i + j)) * y
            cur.append(out)
        prev = cur
    return prev[b]


def jonckheere_terpstra(groups: Sequence[Sequence[float]], exact_max_n=200) -> dict:
    """Jonckheere-Terpstra test against the ordered alternative "increasing
    over the groups in the given order": ``JT = sum_{i<j} #{x in g_i, y in
    g_j: x < y} + 0.5 #{x = y}``; one-sided p = P(JT >= observed). Exact
    without ties up to ``exact_max_n`` values (the null distribution is the
    convolution of Mann-Whitney distributions of each group against the
    union of the groups before it), else normal with the tie-corrected
    variance (Hollander and Wolfe)."""
    gs = [_clean(g) for g in groups]
    sizes = [g.size for g in gs]
    if len(gs) < 2 or any(s == 0 for s in sizes):
        return {"jt": float("nan"), "p": float("nan"), "sizes": sizes, "method": None}
    jt = 0.0
    for i in range(len(gs)):
        for j in range(i + 1, len(gs)):
            diff = gs[j][None, :] - gs[i][:, None]
            jt += float((diff > 0).sum()) + 0.5 * float((diff == 0).sum())
    allv = np.concatenate(gs)
    n = allv.size
    _, tcounts = np.unique(allv, return_counts=True)
    ties = bool((tcounts > 1).any())
    if not ties and n <= exact_max_n:
        pmf = np.ones(1)
        before = sizes[0]
        for s in sizes[1:]:
            pmf = np.convolve(pmf, mann_whitney_pmf(before, s))
            before += s
        t = int(round(jt))
        p = float(pmf[t:].sum())
        method = "exact"
    else:
        ns = np.asarray(sizes, dtype=float)
        tt = tcounts.astype(float)
        mean = (n * n - float((ns**2).sum())) / 4.0
        var = (
            n * (n - 1) * (2 * n + 5)
            - float((ns * (ns - 1) * (2 * ns + 5)).sum())
            - float((tt * (tt - 1) * (2 * tt + 5)).sum())
        ) / 72.0
        # the two correction terms vanish with fewer than three (two) values
        if n > 2:
            var += (
                float((ns * (ns - 1) * (ns - 2)).sum())
                * float((tt * (tt - 1) * (tt - 2)).sum())
                / (36.0 * n * (n - 1) * (n - 2))
            )
        if n > 1:
            var += (
                float((ns * (ns - 1)).sum())
                * float((tt * (tt - 1)).sum())
                / (8.0 * n * (n - 1))
            )
        p = (
            float(_st.norm.sf((jt - mean) / math.sqrt(var)))
            if var > 0
            else float("nan")
        )
        method = "normal"
    return {"jt": jt, "p": min(1.0, p), "sizes": sizes, "method": method}


def kappa_interval(kappa, df, level=0.90) -> Tuple[float, float]:
    """The ``level`` chi-square interval of an SD ratio with ``df`` degrees
    of freedom: ``kappa sqrt(df / chi2(1 - a/2, df))`` to ``kappa sqrt(df /
    chi2(a/2, df))``."""
    a = (1.0 - float(level)) / 2.0
    df = float(df)
    if not (df > 0 and math.isfinite(kappa)):
        return float("nan"), float("nan")
    return (
        kappa * math.sqrt(df / _st.chi2.ppf(1 - a, df)),
        kappa * math.sqrt(df / _st.chi2.ppf(a, df)),
    )


def pooled_within_sd(groups: Sequence[Sequence[float]]) -> Tuple[float, int]:
    """Pooled within-group SD (sum of squared deviations from each group's
    mean over ``sum (n_g - 1)``) and its degrees of freedom."""
    ss, df = 0.0, 0
    for g in groups:
        x = _clean(g)
        if x.size >= 2:
            ss += float(((x - x.mean()) ** 2).sum())
            df += int(x.size - 1)
    return (math.sqrt(ss / df) if df > 0 else float("nan")), df


def rms(x) -> float:
    a = _clean(x)
    return float(math.sqrt((a * a).mean())) if a.size else float("nan")


def cluster_bootstrap_rate(
    k_by_cluster, n_by_cluster, *, b=2000, seed=0, levels=(0.05, 0.95)
) -> Tuple[float, ...]:
    """Percentile cluster-bootstrap bounds of the per-row rate ``sum k /
    sum n`` (clusters resampled with replacement; inverted-CDF quantiles,
    so the bounds are values the statistic can take)."""
    k = np.asarray(k_by_cluster, dtype=float)
    n = np.asarray(n_by_cluster, dtype=float)
    if k.size == 0:
        return tuple(float("nan") for _ in levels)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k.size, size=(int(b), k.size))
    rates = k[idx].sum(axis=1) / n[idx].sum(axis=1)
    return tuple(float(np.quantile(rates, q, method="inverted_cdf")) for q in levels)


_jsonf = _finite  # a finite float for the report, None otherwise


# ==========================================================================
# rule results and cell combination
# ==========================================================================
@dataclass
class RuleResult:
    outcome: str
    reason: str
    cells: List[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def combine_outcomes(outcomes: Iterable[str], *, empty=NOT_EVALUABLE) -> str:
    """FALSIFIED if any evaluable outcome is FALSIFIED; SUPPORTED if all
    evaluable outcomes are SUPPORTED; else INDETERMINATE. Without an
    evaluable outcome: NOT_TESTABLE_BY_DESIGN if every outcome is, else
    ``empty`` (NOT_EVALUABLE)."""
    outs = list(outcomes)
    ev = [o for o in outs if o in EVALUABLE]
    if not ev:
        if outs and all(o == NOT_TESTABLE_BY_DESIGN for o in outs):
            return NOT_TESTABLE_BY_DESIGN
        return empty
    if FALSIFIED in ev:
        return FALSIFIED
    if all(o == SUPPORTED for o in ev):
        return SUPPORTED
    return INDETERMINATE


def _cells_result(cells: List[dict], reason: str, **stats) -> RuleResult:
    return RuleResult(
        combine_outcomes(c["outcome"] for c in cells), reason, cells, stats
    )


# ==========================================================================
# prepared data
# ==========================================================================
@dataclass
class Prepared:
    """Rows of a part after selection: each carries ``_cell``, ``_cluster``,
    ``_pool``, ``_event`` (True/False/None), ``_value``, ``_dose``,
    ``_group`` and ``member``."""

    rows: List[dict]
    notes: List[str] = field(default_factory=list)
    fields: Optional["Fields"] = None

    def get(self, row, path):
        return (self.fields or _PLAIN_FIELDS).get(row, path)

    def cells(self) -> "OrderedDict[str, List[dict]]":
        out: "OrderedDict[str, List[dict]]" = OrderedDict()
        for r in self.rows:
            out.setdefault(r["_cell"], []).append(r)
        return out

    def by(self, key: str, rows=None) -> "OrderedDict[Any, List[dict]]":
        out: "OrderedDict[Any, List[dict]]" = OrderedDict()
        for r in (self.rows if rows is None else rows):
            out.setdefault(r.get(key), []).append(r)
        return out


def _counts(rows) -> Counts:
    return count_events([r["_event"] for r in rows], [r["_cluster"] for r in rows])


def _values(rows, key="_value") -> np.ndarray:
    return np.asarray(
        [v for v in (_finite(r.get(key)) for r in rows) if v is not None], dtype=float
    )


def _p(params, key, default=None):
    v = params.get(key, default)
    return default if v is None else v


def _cmp(op, a, b) -> bool:
    if op in ("ge", ">="):
        return a >= b - _TOL
    if op in ("gt", ">"):
        return a > b
    if op in ("le", "<="):
        return a <= b + _TOL
    if op in ("lt", "<"):
        return a < b
    raise SpecError(f"unknown comparison {op!r}")


# ==========================================================================
# rules
# ==========================================================================
def rule_h0_cell(prep: Prepared, p: Mapping) -> RuleResult:
    bound, alpha = float(p["bound"]), float(_p(p, "alpha", 0.05))
    cells = prep.cells()
    rows_cells = []
    for label, rows in cells.items():
        c = _counts(rows)
        if c.n_clusters:
            rows_cells.append((label, c))
    if not rows_cells:
        return RuleResult(NOT_EVALUABLE, "no rows with a defined event")
    m = int(_p(p, "m", len(rows_cells)))
    out_cells, falsified = [], False
    for label, c in rows_cells:
        lo = cp_lower(c.k_eff, c.n_clusters, alpha / m)
        f = lo > bound
        falsified |= f
        out_cells.append(
            {
                "cell": label,
                **c.to_dict(),
                "lower": lo,
                "outcome": FALSIFIED if f else SUPPORTED,
            }
        )
    pools = OrderedDict()
    for r in prep.rows:
        if r["_event"] is not None:
            pools.setdefault(r["_pool"], []).append(r)
    P = int(_p(p, "P", len(pools)))
    pooled = {}
    all_below = True
    for key, rows in pools.items():
        c = _counts(rows)
        up = cp_upper(c.k_eff, c.n_clusters, alpha / P)
        pooled[str(key)] = {**c.to_dict(), "upper": up}
        all_below &= up < bound
    if falsified:
        outcome, reason = (
            FALSIFIED,
            f"a cell's CP lower bound at {alpha}/{m} exceeds {bound}",
        )
    elif all_below:
        outcome, reason = (
            SUPPORTED,
            f"every pooled CP upper bound at {alpha}/{P} is below {bound}",
        )
    else:
        outcome, reason = INDETERMINATE, "a pooled upper bound is not below the bound"
    return RuleResult(
        outcome, reason, out_cells, {"bound": bound, "m": m, "P": P, "pooled": pooled}
    )


def rule_h0_cell_exceeds(prep: Prepared, p: Mapping) -> RuleResult:
    r = rule_h0_cell(prep, p)
    mirror = {FALSIFIED: SUPPORTED, SUPPORTED: FALSIFIED}
    if r.outcome in mirror:
        r.reason = "predicted exceedance: " + r.reason
        r.outcome = mirror[r.outcome]
    return r


def _zone_cells(prep: Prepared, p: Mapping, decide: Callable) -> RuleResult:
    cells = prep.cells()
    if not cells:
        return RuleResult(NOT_EVALUABLE, "no rows")
    m = int(_p(p, "m", len(cells)))
    alpha = float(_p(p, "alpha", 0.05))
    min_n = int(_p(p, "min_n", 1))
    out = []
    for label, rows in cells.items():
        c = _counts(rows)
        if c.n_clusters < min_n:
            out.append(
                {
                    "cell": label,
                    **c.to_dict(),
                    "outcome": NOT_EVALUABLE,
                    "reason": f"fewer than {min_n} clusters",
                }
            )
            continue
        out.append({"cell": label, **c.to_dict(), **decide(c, alpha, m)})
    return _cells_result(out, "per cell")


def rule_three_zone(prep, p):
    x = float(p["x"])

    def decide(c, alpha, m):
        up = cp_upper(c.k_eff, c.n_clusters, alpha / m)
        if c.rate >= x - _TOL:
            o = SUPPORTED
        elif up < x:
            o = FALSIFIED
        else:
            o = INDETERMINATE
        return {"upper": up, "x": x, "outcome": o}

    r = _zone_cells(prep, p, decide)
    r.reason = (
        f"three-zone at {x}: SUPPORTED iff rate >= {x}, FALSIFIED iff CP upper < {x}"
    )
    return r


def rule_three_zone_at_most(prep, p):
    x = float(p["x"])

    def decide(c, alpha, m):
        lo = cp_lower(c.k_eff, c.n_clusters, alpha / m)
        if c.rate <= x + _TOL:
            o = SUPPORTED
        elif lo > x:
            o = FALSIFIED
        else:
            o = INDETERMINATE
        return {"lower": lo, "x": x, "outcome": o}

    r = _zone_cells(prep, p, decide)
    r.reason = (
        f"three-zone at most {x}: SUPPORTED iff rate <= {x}, "
        f"FALSIFIED iff CP lower > {x}"
    )
    return r


def rule_reversed_three_zone(prep, p):
    x = float(p["x"])
    p = {**p, "m": _p(p, "m", 1)}

    def decide(c, alpha, m):
        up = cp_upper(c.k_eff, c.n_clusters, alpha / m)
        if up < x:
            o = SUPPORTED
        elif c.rate >= x - _TOL:
            o = FALSIFIED
        else:
            o = INDETERMINATE
        return {"upper": up, "x": x, "outcome": o}

    r = _zone_cells(prep, p, decide)
    r.reason = (
        f"reversed three-zone at {x}: SUPPORTED iff CP upper < {x}, "
        f"FALSIFIED iff rate >= {x}"
    )
    return r


def rule_count_at_most(prep, p):
    max_rate = float(p["max_rate"])
    fa = float(_p(p, "falsify_above", 0.10))
    p = {**p, "m": _p(p, "m", 1)}

    def decide(c, alpha, m):
        lo = cp_lower(c.k_eff, c.n_clusters, alpha / m)
        if c.rate <= max_rate + _TOL:
            o = SUPPORTED
        elif lo > fa:
            o = FALSIFIED
        else:
            o = INDETERMINATE
        return {"lower": lo, "outcome": o}

    r = _zone_cells(prep, p, decide)
    r.reason = f"SUPPORTED iff rate <= {max_rate}; FALSIFIED iff CP lower > {fa}"
    return r


def rule_rate_lower_bound(prep, p):
    x = float(p["x"])
    p = {**p, "m": _p(p, "m", 1)}

    def decide(c, alpha, m):
        lo = cp_lower(c.k_eff, c.n_clusters, alpha / m)
        up = cp_upper(c.k_eff, c.n_clusters, alpha / m)
        if lo > x:
            o = SUPPORTED
        elif up < x:
            o = FALSIFIED
        else:
            o = INDETERMINATE
        return {"lower": lo, "upper": up, "outcome": o}

    r = _zone_cells(prep, p, decide)
    r.reason = f"SUPPORTED iff CP lower > {x}; FALSIFIED iff CP upper < {x}"
    return r


def rule_demonstration(prep, p):
    bound = float(p["bound"])
    level = float(_p(p, "level", 0.05))
    n_planned = p.get("n_planned")
    cells = prep.cells()
    if not cells:
        return RuleResult(NOT_EVALUABLE, "no rows")
    out = []
    for label, rows in cells.items():
        c = _counts(rows)
        if not c.n_clusters:
            out.append({"cell": label, **c.to_dict(), "outcome": NOT_EVALUABLE})
            continue
        up = cp_upper(c.k_eff, c.n_clusters, level)
        if up < bound:
            o, why = SUPPORTED, "demonstrated"
        elif n_planned is not None and c.n_clusters < int(n_planned):
            best = cp_upper(c.k_eff, int(n_planned), level)
            if best >= bound:
                o, why = FALSIFIED, "curtailed: the events rule out the demonstration"
            else:
                o, why = INDETERMINATE, "incomplete: planned runs not reached"
        else:
            o, why = FALSIFIED, "not demonstrated"
        out.append(
            {"cell": label, **c.to_dict(), "upper": up, "outcome": o, "why": why}
        )
    return _cells_result(out, f"CP upper bound at {level} below {bound}")


def rule_any_event_clusters(prep, p):
    bound, alpha = float(p["bound"]), float(_p(p, "alpha", 0.05))
    cells = [(lbl, _counts(rows)) for lbl, rows in prep.cells().items()]
    cells = [(lbl, c) for lbl, c in cells if c.n_clusters]
    if not cells:
        return RuleResult(NOT_EVALUABLE, "no rows")
    m = int(_p(p, "m", len(cells)))
    out, falsified = [], False
    for lbl, c in cells:
        lo = cp_lower(c.k_eff, c.n_clusters, alpha / m)
        falsified |= lo > bound
        out.append(
            {
                "cell": lbl,
                **c.to_dict(),
                "lower": lo,
                "outcome": FALSIFIED if lo > bound else SUPPORTED,
            }
        )
    pooled = _counts([r for r in prep.rows if r["_event"] is not None])
    up = cp_upper(pooled.k_any, pooled.n_clusters, alpha)
    if falsified:
        o, why = FALSIFIED, f"a class's CP lower bound at {alpha}/{m} exceeds {bound}"
    elif up < bound:
        o, why = SUPPORTED, f"any-event cluster CP upper bound {up:.4f} below {bound}"
    else:
        o, why = INDETERMINATE, "the any-event upper bound is not below the bound"
    return RuleResult(
        o,
        why,
        out,
        {
            "clusters": pooled.n_clusters,
            "event_clusters": pooled.k_any,
            "upper": up,
            "m": m,
        },
    )


def rule_usable_share(prep, p):
    share = float(_p(p, "share", 0.9))
    band = float(_p(p, "near_band", 0.05))
    out = []
    for label, rows in prep.cells().items():
        c = _counts(rows)
        if not c.n_rows:
            out.append({"cell": label, **c.to_dict(), "outcome": NOT_EVALUABLE})
            continue
        req = int(math.ceil(share * c.n_rows - 1e-12))
        usable = c.k_rows >= req
        out.append(
            {
                "cell": label,
                "family": rows[0].get("family"),
                "check": rows[0].get("check_id"),
                **c.to_dict(),
                "required": req,
                "usable": usable,
                "near_threshold": abs(c.row_rate - share) <= band,
                "outcome": SUPPORTED if usable else FALSIFIED,
            }
        )
    if not out:
        return RuleResult(NOT_EVALUABLE, "no rows")
    return _cells_result(out, f"usable iff passes >= ceil({share} x seeds)")


def rule_false_exclusion(prep, p):
    row_bound = float(_p(p, "row_bound", 0.02))
    seed_bound = float(_p(p, "seed_bound", 0.15))
    seed_falsify = float(_p(p, "seed_falsify", 0.05))
    alpha = float(_p(p, "alpha", 0.05))
    b = int(_p(p, "bootstrap_b", 2000))
    seed = int(_p(p, "bootstrap_seed", 20261005))
    cells = [(lbl, _counts(rows)) for lbl, rows in prep.cells().items()]
    cells = [(lbl, c) for lbl, c in cells if c.n_clusters]
    if not cells:
        return RuleResult(NOT_EVALUABLE, "no mechanism-on rows")
    m = int(_p(p, "m", len(cells)))
    out = []
    for lbl, c in cells:
        lvl = alpha / m
        s_lo = cp_lower(c.k_any, c.n_clusters, lvl)
        s_up = cp_upper(c.k_any, c.n_clusters, lvl)
        (b_lo,) = cluster_bootstrap_rate(
            c.k_by_cluster, c.n_by_cluster, b=b, seed=seed, levels=(lvl,)
        )
        point = c.row_rate
        if s_lo > seed_falsify or b_lo > row_bound:
            o = FALSIFIED
        elif point <= row_bound + _TOL and s_up < seed_bound:
            o = SUPPORTED
        else:
            o = INDETERMINATE
        out.append(
            {
                "cell": lbl,
                **c.to_dict(),
                "row_rate": point,
                "event_seed_share": c.k_any / c.n_clusters,
                "seed_lower": s_lo,
                "seed_upper": s_up,
                "bootstrap_lower": b_lo,
                "outcome": o,
            }
        )
    return _cells_result(out, "false exclusion per cell (seed clusters)", m=m)


def _value_cells(prep, p, decide, *, key="_value", min_n=1) -> RuleResult:
    out = []
    for label, rows in prep.cells().items():
        v = _values(rows, key)
        if v.size < int(_p(p, "min_n", min_n)):
            out.append(
                {
                    "cell": label,
                    "n": int(v.size),
                    "outcome": NOT_EVALUABLE,
                    "reason": "too few finite values",
                }
            )
            continue
        out.append({"cell": label, "n": int(v.size), **decide(v, rows)})
    if not out:
        return RuleResult(NOT_EVALUABLE, "no rows")
    return _cells_result(out, "per cell")


def rule_median_threshold(prep, p):
    op, x = p["op"], float(p["x"])

    def decide(v, rows):
        med = float(np.median(v))
        return {"median": med, "outcome": SUPPORTED if _cmp(op, med, x) else FALSIFIED}

    r = _value_cells(prep, p, decide)
    r.reason = f"median {op} {x}"
    return r


def rule_hodges_lehmann(prep, p):
    alpha = float(_p(p, "alpha", 0.05))
    m_op, m_x = p.get("median_op"), p.get("median_x")
    lower_gt = float(p["lower_gt"])

    def decide(v, rows):
        hl = hodges_lehmann(v, alpha)
        ok = hl["lower"] > lower_gt
        if m_op is not None:
            ok &= _cmp(m_op, hl["median"], float(m_x))
        return {
            **{k: _jsonf(v_) if k != "t_alpha" else v_ for k, v_ in hl.items()},
            "outcome": SUPPORTED if ok else FALSIFIED,
        }

    r = _value_cells(prep, p, decide)
    r.reason = (
        f"median {m_op} {m_x} and " if m_op is not None else ""
    ) + f"one-sided {1 - alpha:.2f} Hodges-Lehmann lower bound > {lower_gt}"
    return r


def rule_wilcoxon(prep, p):
    alpha = float(_p(p, "alpha", 0.05))
    alt = _p(p, "alternative", "greater")

    def decide(v, rows):
        w = wilcoxon_signed_rank(v, alt)
        return {**w, "outcome": SUPPORTED if w["p"] < alpha else FALSIFIED}

    r = _value_cells(prep, p, decide)
    r.reason = f"one-sided Wilcoxon signed-rank p < {alpha}"
    return r


def rule_sign_test(prep, p):
    alpha = float(_p(p, "alpha", 0.05))
    alt = _p(p, "alternative", "greater")

    def decide(v, rows):
        s = sign_test(v, alt)
        return {**s, "outcome": SUPPORTED if (s["p"] < alpha) else FALSIFIED}

    r = _value_cells(prep, p, decide)
    r.reason = f"one-sided sign test p < {alpha}"
    return r


def _dose_value(rows):
    pairs = [(_finite(r.get("_dose")), _finite(r.get("_value"))) for r in rows]
    pairs = [(d, v) for d, v in pairs if d is not None and v is not None]
    return (
        np.asarray([d for d, _ in pairs], dtype=float),
        np.asarray([v for _, v in pairs], dtype=float),
    )


def rule_spearman_ols(prep, p):
    alpha = float(_p(p, "alpha", 0.05))
    alt = _p(p, "alternative", "greater")
    need_slope = bool(_p(p, "require_positive_slope", True))
    out = []
    for label, rows in prep.cells().items():
        d, v = _dose_value(rows)
        s = spearman(d, v, alt)
        if not math.isfinite(s["rho"]):
            out.append(
                {
                    "cell": label,
                    "n": int(d.size),
                    "outcome": NOT_EVALUABLE,
                    "reason": "no spread in dose or value",
                }
            )
            continue
        slope = ols_slope_unit(d, v)
        ok = s["p"] < alpha and ((s["rho"] > 0) if alt == "greater" else (s["rho"] < 0))
        if need_slope:
            ok &= (slope > 0) if alt == "greater" else (slope < 0)
        out.append(
            {
                "cell": label,
                **s,
                "slope_unit_dose": slope,
                "outcome": SUPPORTED if ok else FALSIFIED,
            }
        )
    if not out:
        return RuleResult(NOT_EVALUABLE, "no rows")
    return _cells_result(
        out,
        f"Spearman one-sided p < {alpha}"
        + (" with a positive OLS slope" if need_slope else ""),
    )


def _ordered_groups(rows, order):
    by = OrderedDict((str(g), []) for g in order)
    for r in rows:
        g = r.get("_group")
        if g is None:
            continue
        key = _group_key(g)
        if key in by:
            v = _finite(r.get("_value"))
            if v is not None:
                by[key].append(v)
    return by


def _group_key(g) -> str:
    """A stable text key of a value: integral numbers without a decimal
    point (``1.0`` and ``1`` are one level), other numbers by ``repr``."""
    if g is None or isinstance(g, (str, bool)):
        return str(g)
    f = _finite(g)
    if f is None:
        return str(g)
    if f == int(f) and abs(f) < 1e15:
        return str(int(f))
    return repr(f)


def rule_jonckheere_terpstra(prep, p):
    alpha = float(_p(p, "alpha", 0.05))
    order = [str(o) for o in p["order"]]
    med_rule = p.get("medians")
    out = []
    for label, rows in prep.cells().items():
        groups = _ordered_groups(rows, order)
        if any(len(v) == 0 for v in groups.values()):
            out.append(
                {
                    "cell": label,
                    "outcome": NOT_EVALUABLE,
                    "reason": "a group has no finite values",
                    "sizes": {k: len(v) for k, v in groups.items()},
                }
            )
            continue
        jt = jonckheere_terpstra(list(groups.values()))
        meds = [float(np.median(v)) for v in groups.values()]
        ok = jt["p"] < alpha
        if med_rule == "last_gt_first":
            ok &= meds[-1] > meds[0]
        elif med_rule == "strictly_increasing":
            ok &= all(b > a for a, b in zip(meds, meds[1:]))
        out.append(
            {
                "cell": label,
                **jt,
                "medians": dict(zip(groups, meds)),
                "outcome": SUPPORTED if ok else FALSIFIED,
            }
        )
    if not out:
        return RuleResult(NOT_EVALUABLE, "no rows")
    return _cells_result(
        out,
        f"Jonckheere-Terpstra one-sided p < {alpha}"
        + (f" and medians {med_rule}" if med_rule else ""),
    )


def rule_spearman_monotone(prep, p):
    alpha = float(_p(p, "alpha", 0.05))
    tol = float(_p(p, "tie_tolerance", 0.10))
    out = []
    for label, rows in prep.cells().items():
        d, v = _dose_value(rows)
        s = spearman(d, v, "greater")
        levels = OrderedDict()
        for r in rows:
            dv, vv = _finite(r.get("_dose")), _finite(r.get("_value"))
            if dv is None or vv is None:
                continue
            levels.setdefault(_group_key(r.get("_group", dv)), ([], []))
            levels[_group_key(r.get("_group", dv))][0].append(dv)
            levels[_group_key(r.get("_group", dv))][1].append(vv)
        lv = sorted(
            (
                (float(np.median(e)), float(np.median(c)), k)
                for k, (e, c) in levels.items()
            )
        )
        mono = all(b[1] >= a[1] - tol for a, b in zip(lv, lv[1:]))
        if not math.isfinite(s["rho"]):
            out.append({"cell": label, "outcome": NOT_EVALUABLE, "reason": "no spread"})
            continue
        ok = s["p"] < alpha and s["rho"] > 0 and mono
        out.append(
            {
                "cell": label,
                **s,
                "monotone_medians": mono,
                "level_medians": [
                    {"level": k, "exact": e, "median": c} for e, c, k in lv
                ],
                "outcome": SUPPORTED if ok else FALSIFIED,
            }
        )
    if not out:
        return RuleResult(NOT_EVALUABLE, "no rows")
    return _cells_result(
        out,
        f"Spearman(sampled, exact) one-sided p < {alpha} and level "
        f"medians non-decreasing in the exact value (tolerance {tol})",
    )


def _level_of(value, levels) -> Optional[str]:
    """The declared level a value equals (numeric tolerance), as its key."""
    for lv in levels:
        if _eq(value, lv) or (isinstance(lv, str) and str(value) == lv):
            return _group_key(lv)
    return None


def rule_stepwise_sign(prep, p):
    alpha = float(_p(p, "alpha", 0.025))
    gate = float(_p(p, "gate_share", 0.5))
    declared = [lv for st in p["steps"] for lv in st]
    steps = [[_group_key(a), _group_key(b)] for a, b in p["steps"]]
    pair_on = list(_p(p, "pair_on", ["seed"]))
    out = []
    for label, rows in prep.cells().items():
        by_level = OrderedDict()
        for r in rows:
            key = _level_of(r.get("_group"), declared)
            if key is not None:
                by_level.setdefault(key, []).append(r)
        step_out = []
        for a, b in steps:
            ra, rb = by_level.get(a, []), by_level.get(b, [])
            if not ra or not rb:
                step_out.append(
                    {
                        "step": f"{a}->{b}",
                        "outcome": NOT_EVALUABLE,
                        "reason": "a level has no runs",
                    }
                )
                continue
            defined_b = [bool(r.get("_defined")) for r in rb]
            share = float(np.mean(defined_b))
            if share < gate:
                step_out.append(
                    {
                        "step": f"{a}->{b}",
                        "defined_share": share,
                        "outcome": SUPPORTED,
                        "why": "correctly undefined",
                    }
                )
                continue
            ex_a = _values(ra, "_exact")
            ex_b = _values(rb, "_exact")
            if not (ex_a.size and ex_b.size):
                step_out.append(
                    {
                        "step": f"{a}->{b}",
                        "outcome": NOT_EVALUABLE,
                        "reason": "no exact values",
                    }
                )
                continue
            sign = np.sign(float(np.median(ex_b)) - float(np.median(ex_a)))
            if sign == 0:
                step_out.append(
                    {
                        "step": f"{a}->{b}",
                        "outcome": NOT_EVALUABLE,
                        "reason": "exact change is zero",
                    }
                )
                continue
            ia = {tuple(r.get(k) for k in pair_on): r for r in ra if r.get("_defined")}
            diffs = []
            for r in rb:
                key = tuple(r.get(k) for k in pair_on)
                if r.get("_defined") and key in ia:
                    va, vb = _finite(ia[key].get("_value")), _finite(r.get("_value"))
                    if va is not None and vb is not None:
                        diffs.append(vb - va)
            st = sign_test(diffs, "greater" if sign > 0 else "less")
            med = float(np.median(diffs)) if diffs else float("nan")
            # the median paired change has the sign of the exact change, and
            # the one-sided sign test in that direction is significant
            ok = (
                math.isfinite(st["p"])
                and st["p"] < alpha
                and math.isfinite(med)
                and np.sign(med) == sign
            )
            step_out.append(
                {
                    "step": f"{a}->{b}",
                    "defined_share": share,
                    "exact_sign": int(sign),
                    "median_change": med if diffs else None,
                    **st,
                    "outcome": SUPPORTED if ok else FALSIFIED,
                }
            )
        out.append(
            {
                "cell": label,
                "steps": step_out,
                "outcome": combine_outcomes(s["outcome"] for s in step_out),
            }
        )
    if not out:
        return RuleResult(NOT_EVALUABLE, "no rows")
    return _cells_result(
        out, "each step: correct sign (sign test) or correctly undefined"
    )


def _se_of(row):
    """The SE of a calibration row: the data selection's ``se`` field (for
    example one NAS direction's SE) when it names one, else ``se_c``."""
    return row["_se"] if "_se" in row else row.get("se_c")


def _df_of(row):
    """The degrees of freedom that go with :func:`_se_of`."""
    return row["_df"] if "_df" in row else row.get("df_c")


def _q(alpha, df):
    from impact_pipeline import evidence_v2 as E

    d = _finite(df)
    return E.quantile(alpha, math.inf if d is None or d <= 0 else d)


def rule_kappa_null(prep, p):
    point = tuple(_p(p, "point", (0.8, 1.25)))
    inside = tuple(_p(p, "interval", (0.67, 1.5)))
    level = float(_p(p, "level", 0.90))
    tail_max = float(_p(p, "tail_max", 0.075))
    tail_falsify = float(_p(p, "tail_falsify", 0.10))
    q_alpha = float(_p(p, "q_alpha", 0.05))
    cells = prep.cells()
    if not cells:
        return RuleResult(NOT_EVALUABLE, "no rows")
    P = int(_p(p, "P", len(cells)))
    t_level = float(_p(p, "tail_level", 0.05 / (2 * P)))
    out = []
    for label, rows in cells.items():
        good = [
            r
            for r in rows
            if _finite(r.get("_value")) is not None and _finite(_se_of(r)) is not None
        ]
        if len(good) < 3:
            out.append({"cell": label, "n": len(good), "outcome": NOT_EVALUABLE})
            continue
        c = np.asarray([float(r["_value"]) for r in good])
        se = np.asarray([float(_se_of(r)) for r in good])
        kap = float(np.std(c, ddof=1) / rms(se))
        df = len(good) - 1
        lo, hi = kappa_interval(kap, df, level)
        q = np.asarray([_q(q_alpha, _df_of(r)) for r in good])
        up_ev = list(c - q * se > 0)
        lo_ev = list(c + q * se < 0)
        clusters = [r["_cluster"] for r in good]
        cu, cl = count_events(up_ev, clusters), count_events(lo_ev, clusters)
        tails = {}
        tail_fail, tail_ok = False, True
        for name, cc in (("upper", cu), ("lower", cl)):
            lb = cp_lower(cc.k_eff, cc.n_clusters, t_level)
            tails[name] = {**cc.to_dict(), "lower_bound": lb}
            tail_ok &= cc.rate <= tail_max + _TOL
            tail_fail |= lb > tail_falsify
        a_ok = point[0] <= kap <= point[1] and lo >= inside[0] and hi <= inside[1]
        out_of = hi < point[0] or lo > point[1]
        if out_of or tail_fail:
            o = FALSIFIED
        elif a_ok and tail_ok:
            o = SUPPORTED
        else:
            o = INDETERMINATE
        side = SIDE_CONSERVATIVE if kap < 1.0 else SIDE_ANTI
        out.append(
            {
                "cell": label,
                "n": len(good),
                "kappa": kap,
                "df": df,
                "interval": [lo, hi],
                "tails": tails,
                "side": side,
                "outcome": o,
            }
        )
    return _cells_result(
        out,
        "kappa0 in [0.8, 1.25] with its 90 % interval inside "
        "[0.67, 1.5], and both tails <= 0.075",
        P=P,
    )


def rule_kappa_twins(prep, p):
    point = tuple(_p(p, "point", (0.8, 1.25)))
    inside = tuple(_p(p, "interval", (0.67, 1.5)))
    level = float(_p(p, "level", 0.90))
    tail_max = float(_p(p, "tail_max", 0.02))
    tail_falsify = float(_p(p, "tail_falsify", 0.03))
    alpha_a = float(_p(p, "alpha_absent", 0.01))
    members = dict(_p(p, "members", {"NAS": 2}))
    min_defined = float(_p(p, "min_defined", 0.8))
    cells = prep.cells()
    if not cells:
        return RuleResult(NOT_EVALUABLE, "no rows")
    m = int(_p(p, "m", 2 * len(cells)))
    out = []
    for label, rows in cells.items():
        nets = OrderedDict()
        for r in rows:
            nets.setdefault(r["_network"], []).append(r)
        defined = [
            (_finite(r.get("_value")) is not None and _finite(_se_of(r)) is not None)
            for r in rows
        ]
        share = float(np.mean(defined)) if defined else 0.0
        if share < min_defined:
            out.append(
                {
                    "cell": label,
                    "defined_share": share,
                    "outcome": NOT_EVALUABLE,
                    "reason": f"c defined in fewer than {min_defined:.0%} of sessions",
                }
            )
            continue
        groups, ses, up_ev, lo_ev, clusters = [], [], [], [], []
        for key, rr in nets.items():
            good = [
                r
                for r in rr
                if _finite(r.get("_value")) is not None
                and _finite(_se_of(r)) is not None
            ]
            if len(good) < 2:
                continue
            c = np.asarray([float(r["_value"]) for r in good])
            se = np.asarray([float(_se_of(r)) for r in good])
            groups.append(c)
            ses.extend(se.tolist())
            mean = float(c.mean())
            k = int(members.get(good[0].get("principle"), 1))
            q = np.asarray([_q(alpha_a / k, _df_of(r)) for r in good])
            up_ev.extend(list(c + q * se < mean))
            lo_ev.extend(list(c - q * se > mean))
            clusters.extend([key] * len(good))
        sd, df = pooled_within_sd(groups)
        if df < 1:
            out.append(
                {"cell": label, "outcome": NOT_EVALUABLE, "reason": "no twin set"}
            )
            continue
        kap = sd / rms(ses)
        lo, hi = kappa_interval(kap, df, level)
        tails, tail_fail, tail_ok = {}, False, True
        for name, ev in (("below", up_ev), ("above", lo_ev)):
            cc = count_events(ev)  # sessions are the units of the tail rate
            lb = cp_lower(cc.k_rows, cc.n_rows, 0.05 / m)
            tails[name] = {**cc.to_dict(), "lower_bound": lb}
            tail_ok &= cc.row_rate <= tail_max + _TOL
            tail_fail |= lb > tail_falsify
        a_ok = point[0] <= kap <= point[1] and lo >= inside[0] and hi <= inside[1]
        out_of = hi < point[0] or lo > point[1]
        if out_of or tail_fail:
            o = FALSIFIED
        elif a_ok and tail_ok:
            o = SUPPORTED
        else:
            o = INDETERMINATE
        anti = (lo > point[1]) or tail_fail
        side = SIDE_ANTI if anti else (SIDE_CONSERVATIVE if hi < point[0] else None)
        first = rows[0]
        out.append(
            {
                "cell": label,
                "networks": len(groups),
                "df": df,
                "kappa": kap,
                "interval": [lo, hi],
                "tails": tails,
                "defined_share": share,
                "side": side,
                "estimator_version": first.get("estimator_version"),
                "principle": first.get("principle"),
                "se_method": first.get("se_method"),
                "outcome": o,
            }
        )
    return _cells_result(
        out,
        "kappa in [0.8, 1.25] with its 90 % interval inside "
        "[0.67, 1.5]; q_A tails <= 0.02",
        m=m,
    )


def rule_concordance_twins(prep, p):
    share_max = float(_p(p, "share_max", 0.05))
    alpha = float(_p(p, "alpha", 0.05))
    nets = OrderedDict()
    for r in prep.rows:
        nets.setdefault(r["_network"], []).append(r)
    k = n = 0
    used = 0
    for key, rr in nets.items():
        conc = [
            r
            for r in rr
            if r.get("se_method") == "concordant"
            and _finite(r.get("_value")) is not None
        ]
        if not conc:
            continue
        used += 1
        ref = float(conc[0]["_value"])
        for r in rr:
            if r is conc[0]:
                continue
            v = _finite(r.get("_value"))
            if v is None:
                continue
            n += 1
            k += int(abs(v - ref) > 1e-9)
    if n == 0:
        return RuleResult(NOT_EVALUABLE, "no concordant twin set")
    share = k / n
    lo = cp_lower(k, n, alpha)
    o = (
        SUPPORTED
        if share <= share_max + _TOL
        else (FALSIFIED if lo > share_max else INDETERMINATE)
    )
    return RuleResult(
        o,
        f"share of twin sessions differing from the concordant count "
        f"<= {share_max}",
        [
            {
                "cell": "all",
                "networks": used,
                "k": k,
                "n": n,
                "share": share,
                "lower": lo,
                "outcome": o,
            }
        ],
    )


def rule_newcombe_includes_zero(prep, p):
    conf = float(_p(p, "conf", 0.95))
    a_lbl, b_lbl = str(p["a"]), str(p["b"])
    out = []
    for label, rows in prep.cells().items():
        ra = [r for r in rows if r.get("member") == a_lbl]
        rb = [r for r in rows if r.get("member") == b_lbl]
        ca, cb = _counts(ra), _counts(rb)
        if not (ca.n_clusters and cb.n_clusters):
            out.append({"cell": label, "outcome": NOT_EVALUABLE})
            continue
        d, lo, hi = newcombe_difference(
            ca.k_eff, ca.n_clusters, cb.k_eff, cb.n_clusters, conf
        )
        o = SUPPORTED if lo <= 0.0 <= hi else FALSIFIED
        out.append(
            {
                "cell": label,
                "a": ca.to_dict(),
                "b": cb.to_dict(),
                "difference": d,
                "interval": [lo, hi],
                "outcome": o,
            }
        )
    if not out:
        return RuleResult(NOT_EVALUABLE, "no rows")
    return _cells_result(
        out, f"Newcombe {conf:.0%} interval of the rate difference includes 0"
    )


def rule_admission_matches(prep, p):
    preds = dict(p["predictions"])
    anchor_cell = p.get("anchor_cell")
    anchor_field = p.get("anchor_field", "anchor_valid")
    cells = prep.cells()
    if anchor_cell is not None:
        rows = cells.get(anchor_cell) or []
        if rows and rows[0].get(anchor_field) is False:
            return RuleResult(NOT_EVALUABLE, f"the anchor of {anchor_cell} is invalid")
    out = []
    for cell, want in preds.items():
        rows = cells.get(cell)
        if not rows:
            out.append(
                {"cell": cell, "outcome": NOT_EVALUABLE, "reason": "no registry entry"}
            )
            continue
        row = rows[0]
        got = {k: prep.get(row, k) for k in want}
        ok = all(str(got[k]) == str(v) for k, v in want.items())
        out.append(
            {
                "cell": cell,
                "predicted": want,
                "observed": got,
                "failing_criteria": row.get("failing_criteria"),
                "outcome": SUPPORTED if ok else FALSIFIED,
            }
        )
    return _cells_result(out, "observed admission equals the prediction")


def rule_anchor_replicates(prep, p):
    from impact_pipeline.v2 import testability as T

    block = int(_p(p, "block_size", 20))
    share = float(_p(p, "min_finite_share", 0.9))
    alpha = float(_p(p, "alpha", 0.05))
    ratio = float(_p(p, "ratio", 0.5))
    pc_member, lesion_member = p.get("pc", "pc"), p.get("lesion", "lesion")
    dirs = dict(_p(p, "directions", {}))
    validity_only = bool(_p(p, "validity_only", False))
    # draft predictions (used only where no frozen protocol states the
    # development status): [{protocol, principle, status}]
    fallback = {
        (str(e["protocol"]), str(e["principle"])): e["status"]
        for e in (_p(p, "predictions", []) or [])
    }
    out = []
    for label, rows in prep.cells().items():
        principle = rows[0].get("principle")
        proto = rows[0].get("_protocol")
        pred = rows[0].get("_predicted_anchor")
        source = "protocol" if pred is not None else None
        if pred is None and (str(proto), str(principle)) in fallback:
            pred, source = fallback[(str(proto), str(principle))], "draft"
        fields = dirs.get(principle) or ["_value"]
        if not validity_only and not any(
            r.get("member") == lesion_member for r in rows
        ):
            # specificity needs the own-lesion runs: without them the status
            # cannot be judged (a missing input, not a non-specific anchor)
            out.append(
                {
                    "cell": label,
                    "protocol": proto,
                    "principle": principle,
                    "outcome": NOT_EVALUABLE,
                    "reason": "no own-lesion replication runs",
                }
            )
            continue
        entries = {}
        for fld in fields:
            pc = {
                r.get("seed"): _finite(
                    r.get(fld) if fld == "_value" else prep.get(r, fld)
                )
                for r in rows
                if r.get("member") == pc_member
            }
            les = {
                r.get("seed"): _finite(
                    r.get(fld) if fld == "_value" else prep.get(r, fld)
                )
                for r in rows
                if r.get("member") == lesion_member
            }
            seeds = sorted(set(pc) | set(les), key=lambda s: (s is None, s))
            size = max(block, len(seeds))
            x = np.full(size, np.nan)
            for i, s in enumerate(seeds):
                v = pc.get(s)
                x[i] = np.nan if v is None else v
            val = T.anchor_validity(
                x,
                block_size=size,
                min_finite=int(math.ceil(share * size - 1e-12)),
                alpha=alpha,
            )
            pcv = np.asarray([np.nan if pc.get(s) is None else pc[s] for s in seeds])
            lev = np.asarray([np.nan if les.get(s) is None else les[s] for s in seeds])
            spec = (
                T.anchor_specificity(pcv, lev, ratio=ratio, alpha=alpha)
                if les
                else None
            )
            entries[fld] = T.anchor_entry(val, spec)
        entry = (
            T.combine_anchor_entries(entries)
            if len(entries) > 1
            else next(iter(entries.values()))
        )
        observed = entry["status"]
        if validity_only:
            observed = "valid" if entry["valid"] else "invalid"
            if pred is not None:
                pred = "invalid" if pred in ("invalid", T.ANCHOR_INVALID) else "valid"
        if pred is None:
            out.append(
                {
                    "cell": label,
                    "observed": observed,
                    "outcome": NOT_EVALUABLE,
                    "reason": "no predicted anchor status",
                }
            )
            continue
        ok = observed == pred
        out.append(
            {
                "cell": label,
                "protocol": proto,
                "principle": principle,
                "predicted": pred,
                "prediction_source": source,
                "observed": observed,
                "flag": None if ok else R.ANCHOR_NOT_REPLICATED,
                "outcome": SUPPORTED if ok else FALSIFIED,
            }
        )
    if not out:
        return RuleResult(NOT_EVALUABLE, "no replication rows")
    return _cells_result(out, "every (protocol, principle) anchor status replicates")


def rule_fmd_concordance(prep, p):
    ratio = float(_p(p, "ratio", 0.5))
    levels = list(p["levels"])
    lo_l, hi_l = (_group_key(v) for v in levels)
    zone = _p(p, "zone", "three_zone")
    src_member, view_member = p.get("source", "source"), p.get("view", "view")
    src = {}
    for r in prep.rows:
        lv = _level_of(r.get("_dose"), levels)
        if r.get("member") == src_member and lv is not None:
            src.setdefault(r.get("seed"), {})[lv] = _finite(r.get("_value"))
    ev_rows = []
    by_cell = OrderedDict()
    for r in prep.rows:
        lv = _level_of(r.get("_dose"), levels)
        if r.get("member") == view_member and lv is not None:
            by_cell.setdefault(r["_cell"], {}).setdefault(r.get("seed"), {})[lv] = (
                _finite(r.get("_value"))
            )
    for cell, seeds in by_cell.items():
        for seed, vals in seeds.items():
            s = src.get(seed, {})
            dv = (
                None
                if vals.get(hi_l) is None or vals.get(lo_l) is None
                else vals[hi_l] - vals[lo_l]
            )
            ds = (
                None
                if s.get(hi_l) is None or s.get(lo_l) is None
                else s[hi_l] - s[lo_l]
            )
            if dv is None or ds is None or ds == 0:
                ev = None
            else:
                ev = bool(np.sign(dv) == np.sign(ds) and abs(dv) >= ratio * abs(ds))
            ev_rows.append(
                {"_cell": cell, "_cluster": seed, "_event": ev, "_pool": cell}
            )
    sub = Prepared(ev_rows)
    fn = {"three_zone": rule_three_zone, "reversed": rule_reversed_three_zone}[zone]
    r = fn(sub, p)
    r.reason = (
        f"FMd: forward contrast has the source's sign and >= {ratio} of its "
        f"size; {r.reason}"
    )
    return r


def rule_describe(prep, p):
    out = []
    for label, rows in prep.cells().items():
        c = _counts(rows)
        v = _values(rows)
        out.append(
            {
                "cell": label,
                **c.to_dict(),
                "median": float(np.median(v)) if v.size else None,
                "mean": float(np.mean(v)) if v.size else None,
                "n_values": int(v.size),
                "outcome": REPORTED,
            }
        )
    return RuleResult(REPORTED, "reported", out)


RULES: Dict[str, Callable[[Prepared, Mapping], RuleResult]] = {
    "h0_cell": rule_h0_cell,
    "h0_cell_exceeds": rule_h0_cell_exceeds,
    "three_zone": rule_three_zone,
    "three_zone_at_most": rule_three_zone_at_most,
    "reversed_three_zone": rule_reversed_three_zone,
    "count_at_most": rule_count_at_most,
    "rate_lower_bound": rule_rate_lower_bound,
    "demonstration": rule_demonstration,
    "any_event_clusters": rule_any_event_clusters,
    "usable_share": rule_usable_share,
    "false_exclusion": rule_false_exclusion,
    "median_threshold": rule_median_threshold,
    "hodges_lehmann": rule_hodges_lehmann,
    "wilcoxon": rule_wilcoxon,
    "sign_test": rule_sign_test,
    "spearman_ols": rule_spearman_ols,
    "jonckheere_terpstra": rule_jonckheere_terpstra,
    "spearman_monotone": rule_spearman_monotone,
    "stepwise_sign": rule_stepwise_sign,
    "kappa_null": rule_kappa_null,
    "kappa_twins": rule_kappa_twins,
    "concordance_twins": rule_concordance_twins,
    "newcombe_includes_zero": rule_newcombe_includes_zero,
    "admission_matches": rule_admission_matches,
    "anchor_replicates": rule_anchor_replicates,
    "fmd_concordance": rule_fmd_concordance,
    "describe": rule_describe,
}
# parameters each rule requires
RULE_REQUIRED = {
    "h0_cell": ("bound",),
    "h0_cell_exceeds": ("bound",),
    "three_zone": ("x",),
    "three_zone_at_most": ("x",),
    "reversed_three_zone": ("x",),
    "count_at_most": ("max_rate",),
    "rate_lower_bound": ("x",),
    "demonstration": ("bound",),
    "any_event_clusters": ("bound",),
    "median_threshold": ("op", "x"),
    "hodges_lehmann": ("lower_gt",),
    "jonckheere_terpstra": ("order",),
    "stepwise_sign": ("steps",),
    "newcombe_includes_zero": ("a", "b"),
    "admission_matches": ("predictions",),
    "fmd_concordance": ("levels", "x"),
}
# rules whose rows need an event / a value / a dose / a group
_NEEDS_EVENT = {
    "h0_cell",
    "h0_cell_exceeds",
    "three_zone",
    "three_zone_at_most",
    "reversed_three_zone",
    "count_at_most",
    "rate_lower_bound",
    "demonstration",
    "any_event_clusters",
    "usable_share",
    "false_exclusion",
    "newcombe_includes_zero",
}


# ==========================================================================
# rows from records and fields
# ==========================================================================
_BIT_RE = re.compile(r"^b[01]{5}$")
DEFINEDNESS_REASONS = frozenset(
    {
        R.SAMPLING_UNRESOLVED,
        R.NOT_DEFINED,
        R.NON_FINITE_ESTIMATE,
        R.ESTIMATOR_ERROR,
        R.INSUFFICIENT_TIMEPOINTS,
        R.OBSERVATION_MIXED_NOT_ADMITTED,
        R.NOT_APPLICABLE_OBSERVATION_MODEL,
        R.MISSING,
        R.INSUFFICIENT_OCCUPANCY,
        R.MACRO_RANK_DEFICIENT,
        R.INSUFFICIENT_UPDATES,
        R.NOT_IMPLEMENTED,
    }
)


def _as_dict(rec) -> dict:
    if hasattr(rec, "to_dict") and not isinstance(rec, Mapping):
        return rec.to_dict()
    return dict(rec)


def component_rows(records: Iterable) -> List[dict]:
    """One flat row per (task, scoring, component) of ``/3`` records
    (``TaskRecord`` objects or their dicts): task fields, scoring fields
    and every component field; ``config``, ``details``, ``identifiability``
    and ``simulation`` stay nested (dotted paths reach them)."""
    out = []
    for rec in records:
        d = _as_dict(rec)
        base = {
            k: d.get(k)
            for k in (
                "task_id",
                "design",
                "family",
                "system",
                "seed",
                "replicate",
                "split",
                "generator_version",
            )
        }
        base["task_status"] = d.get("status")
        base["config"] = d.get("config") or {}
        base["simulation"] = d.get("simulation") or {}
        base["provenance"] = d.get("provenance") or {}
        for s in d.get("scorings") or ():
            sd = _as_dict(s)
            sbase = {
                k: sd.get(k)
                for k in (
                    "scoring_id",
                    "declaration_id",
                    "observation_stage",
                    "view",
                    "estimator_form",
                    "protocol_id",
                    "protocol_hash",
                )
            }
            sbase["scoring_details"] = sd.get("details") or {}
            for p, comp in (sd.get("components") or {}).items():
                cd = _as_dict(comp)
                row = {**base, **sbase, **cd}
                row["principle"] = cd.get("principle", p)
                row["flags"] = list(cd.get("flags") or ())
                row["details"] = cd.get("details") or {}
                out.append(row)
    return out


def verdict_rows(records: Iterable) -> List[dict]:
    """One row per scoring with a verdict: task and scoring fields, the
    verdict block (``verdict``; its value also as ``verdict_value``) and the
    component statuses and estimator versions of the scoring."""
    out = []
    for rec in records:
        d = _as_dict(rec)
        base = {
            k: d.get(k)
            for k in (
                "task_id",
                "design",
                "family",
                "system",
                "seed",
                "replicate",
                "split",
            )
        }
        base["config"] = d.get("config") or {}
        for s in d.get("scorings") or ():
            sd = _as_dict(s)
            v = sd.get("verdict")
            if not v:
                continue
            comps = {p: _as_dict(c) for p, c in (sd.get("components") or {}).items()}
            out.append(
                {
                    **base,
                    **{
                        k: sd.get(k)
                        for k in (
                            "scoring_id",
                            "declaration_id",
                            "observation_stage",
                            "view",
                            "estimator_form",
                            "protocol_id",
                            "protocol_hash",
                        )
                    },
                    "verdict": dict(v),
                    "verdict_value": v.get("verdict"),
                    "component_status": {p: c.get("status") for p, c in comps.items()},
                    "component_method": {
                        p: [c.get("estimator_version"), c.get("se_method")]
                        for p, c in comps.items()
                    },
                }
            )
    return out


def _blank(v) -> bool:
    return (
        v is None
        or (isinstance(v, float) and math.isnan(v))
        or (isinstance(v, str) and not v.strip())
    )


def check_id(row: Mapping) -> str:
    """The id of a prerequisite-M check: the switch name for a frozen 1.0.0
    switch check (``eta``, ``K``, ``g_b``, ``ff_only``, ``c_int``, ``e``),
    ``<system>[:<variant>]/<check>`` for a realisation check of a new
    system (``bench.manipulation_v2`` rows)."""
    if not _blank(row.get("switch")):
        return str(row["switch"])
    sysid, check = row.get("system_id"), row.get("check")
    if _blank(sysid) or _blank(check):
        raise ValueError("a manipulation row names a switch or a system and a check")
    var = row.get("variant")
    return f"{sysid}{'' if _blank(var) else ':' + str(var)}/{check}"


def manipulation_rows(
    switches: Iterable[Mapping] = (), realisation: Iterable[Mapping] = ()
) -> List[dict]:
    """Rows of the ``manipulation`` source from the per-seed tables of the
    prerequisite M (``bench.manipulation_v2.prerequisite_m``: ``switches``
    with family, switch, seed, passed; ``realisation`` with system_id,
    variant, family, check, seed, passed): ``kind`` (switch or new_system),
    ``check_id``, ``family``, ``seed`` and ``passed``."""
    out = []
    for kind, rows in (("switch", switches), ("new_system", realisation)):
        for r in rows:
            d = {k: (None if _blank(v) else v) for k, v in dict(r).items()}
            d["kind"] = kind
            d["check_id"] = check_id(d)
            passed = d.get("passed")
            d["passed"] = (
                (str(passed).strip().lower() in ("true", "1"))
                if isinstance(passed, str)
                else bool(passed)
            )
            if d.get("seed") is not None:
                d["seed"] = int(d["seed"])
            out.append(d)
    return out


def registry_rows(entries) -> List[dict]:
    """Rows of the ``registry`` source from registry v3 admission entries
    (a list, or a mapping with ``entries``): one flat row per entry with
    ``principle``, ``view``, ``admitted_for_present``,
    ``admitted_for_absent``, ``anchor_valid`` (FM0) and ``failing_criteria``;
    keys of a nested ``regime`` block are lifted when the entry does not
    have them, and ``view`` defaults to the regime's view or observation
    stage."""
    if isinstance(entries, Mapping):
        entries = entries.get("entries") or []
    out = []
    for e in entries:
        d = dict(e)
        regime = d.get("regime")
        if isinstance(regime, Mapping):
            for k, v in regime.items():
                d.setdefault(k, v)
        if d.get("view") is None:
            d["view"] = d.get("observation_stage")
        out.append(d)
    return out


def _walk(obj, parts):
    cur = obj
    for p in parts:
        if isinstance(cur, Mapping):
            if p not in cur:
                return None
            cur = cur[p]
        elif isinstance(cur, (list, tuple)) and p.isdigit():
            i = int(p)
            if i >= len(cur):
                return None
            cur = cur[i]
        else:
            return None
    return cur


def _cell_id(row: Mapping, fields: "Fields" = None):
    """The factorial cell id of a row (``b`` and five bits): the file's
    ``cell_id`` alias when it declares one, else ``config.cell_id``."""
    if fields is not None and "cell_id" in fields.aliases:
        return fields.get(row, "@cell_id")
    return _walk(row, ["config", "cell_id"])


def _builtin_derived(row: Mapping, name: str, fields: "Fields" = None):
    if name == "reason_code":
        reason = row.get("reason")
        if reason is None:
            # a decided component has no reason: "" (a value, not a missing one)
            return "" if row.get("status") is not None else None
        return R.parse_reason(reason)[0]
    if name == "excess":
        e, m = _finite(row.get("estimate")), _finite(row.get("null_mean"))
        return None if e is None or m is None else e - m
    if name == "abs_c":
        c = _finite(row.get("c"))
        return None if c is None else abs(c)
    if name == "abs_delta":
        c = _finite(row.get("delta"))
        return None if c is None else abs(c)
    if name == "concordant":
        return row.get("se_method") == "concordant"
    if name == "defined":
        code = _builtin_derived(row, "reason_code")
        return (
            _finite(row.get("estimate")) is not None and code not in DEFINEDNESS_REASONS
        )
    if name == "own_bit" or name.startswith("bit."):
        cid = _cell_id(row, fields)
        if not isinstance(cid, str) or not _BIT_RE.match(cid):
            return None
        p = row.get("principle") if name == "own_bit" else name[4:]
        if p not in PRINCIPLES:
            return None
        return int(cid[1 + PRINCIPLES.index(p)])
    if name == "network":
        return "|".join(str(row.get(k)) for k in ("family", "system", "seed"))
    return _MISSING


_MISSING = object()


class Fields:
    """Field resolution against a hypotheses file: aliases (``@name``),
    declared derived predicates, built-in derived fields and dotted paths."""

    def __init__(self, aliases: Mapping = None, derived: Mapping = None):
        self.aliases = dict(aliases or {})
        self.derived = dict(derived or {})

    def get(self, row: Mapping, path: str, _depth=0):
        if _depth > 8:
            raise SpecError(f"alias cycle at {path!r}")
        if path.startswith("@"):
            name = path[1:]
            if name in self.derived:
                return evaluate_predicate(self.derived[name], row, self)
            if name not in self.aliases:
                raise SpecError(f"unknown field alias {path!r}")
            return self.get(row, self.aliases[name], _depth + 1)
        if path in row:
            return row[path]
        v = _builtin_derived(row, path, self)
        if v is not _MISSING:
            return v
        return _walk(row, path.split("."))


_PLAIN_FIELDS = Fields()


def _get(row, path, fields: Fields = None):
    return (fields or _PLAIN_FIELDS).get(row, path)


# --------------------------------------------------------------------------
# predicates (three-valued)
# --------------------------------------------------------------------------
_OPS = (
    "eq",
    "ne",
    "in",
    "not_in",
    "lt",
    "le",
    "gt",
    "ge",
    "abs_lt",
    "abs_le",
    "abs_gt",
    "abs_ge",
    "min",
    "max",
    "exists",
    "contains",
    "prefix",
)


def _cond(value, cond):
    """True/False/None (None: the value is missing and the condition needs it)."""
    if isinstance(cond, Mapping):
        res = True
        for op, arg in cond.items():
            if op not in _OPS:
                raise SpecError(f"unknown predicate operator {op!r}")
            r = _op(value, op, arg)
            if r is False:
                return False
            if r is None:
                res = None
        return res
    if isinstance(cond, list):
        return _op(value, "in", cond)
    return _op(value, "eq", cond)


NUMERIC_EQ_TOL = 1e-6


def _eq(a, b) -> bool:
    """Equality; numbers within a relative (absolute below 1) tolerance of
    1e-6, because dose levels are stored rounded (v1 sweeps round to 6
    decimals; G_nom = 8/7 is written 1.142857)."""
    fa, fb = _finite(a), _finite(b)
    if (
        fa is not None
        and fb is not None
        and not isinstance(a, str)
        and not isinstance(b, str)
    ):
        return abs(fa - fb) <= NUMERIC_EQ_TOL * max(1.0, abs(fa), abs(fb))
    return a == b


def _op(value, op, arg):
    if op == "exists":
        present = value is not None and not (
            isinstance(value, float) and math.isnan(value)
        )
        return present == bool(arg)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if op == "eq":
        return _eq(value, arg)
    if op == "ne":
        return not _eq(value, arg)
    if op == "in":
        return any(_eq(value, a) for a in arg)
    if op == "not_in":
        return not any(_eq(value, a) for a in arg)
    if op == "contains":
        return arg in value if isinstance(value, (list, tuple, set, str)) else False
    if op == "prefix":
        return isinstance(value, str) and value.startswith(str(arg))
    v = _finite(value)
    if v is None:
        return None
    a = float(arg)
    return {
        "lt": v < a,
        "le": v <= a + _TOL,
        "gt": v > a,
        "ge": v >= a - _TOL,
        "abs_lt": abs(v) < a,
        "abs_le": abs(v) <= a + _TOL,
        "abs_gt": abs(v) > a,
        "abs_ge": abs(v) >= a - _TOL,
        "min": v >= a - _TOL,
        "max": v <= a + _TOL,
    }[op]


def evaluate_predicate(pred, row: Mapping, fields: Fields = None):
    """Evaluate a predicate on a row: a mapping of field -> condition (all
    must hold) with the combinators ``all``, ``any`` and ``not``; True,
    False or None (unknown)."""
    if pred is None:
        return True
    if isinstance(pred, bool):
        return pred
    if not isinstance(pred, Mapping):
        raise SpecError(f"a predicate is an object, got {pred!r}")
    fields = fields or _PLAIN_FIELDS
    res = True
    for key, cond in pred.items():
        if key == "all":
            vals = [evaluate_predicate(p, row, fields) for p in cond]
            r = False if False in vals else (None if None in vals else True)
        elif key == "any":
            vals = [evaluate_predicate(p, row, fields) for p in cond]
            r = True if True in vals else (None if None in vals else False)
        elif key == "not":
            v = evaluate_predicate(cond, row, fields)
            r = None if v is None else (not v)
        else:
            r = _cond(fields.get(row, key), cond)
        if r is False:
            return False
        if r is None:
            res = None
    return res


_TEMPLATE_RE = re.compile(r"\{([^{}]+)\}")


def render_template(template: str, row: Mapping, fields: Fields = None) -> str:
    def sub(m):
        v = (fields or _PLAIN_FIELDS).get(row, m.group(1).strip())
        return _group_key(v) if v is not None else "None"

    return _TEMPLATE_RE.sub(sub, str(template))


# ==========================================================================
# context
# ==========================================================================
def _protocol_info(proto) -> dict:
    """``{name, hash, necessity_set, anchors, precision, concordance_route}``
    of a ``ProtocolV3`` or of a protocol mapping (only the blocks the
    evaluator reads)."""
    if isinstance(proto, Mapping):
        d = dict(proto)
        anchors = d.get("anchors")
        nset = d.get("necessity_set")
        if nset is None and isinstance(anchors, Mapping):
            nset = anchors.get("necessity_set")
        return {
            "name": d.get("name"),
            "hash": d.get("hash"),
            "necessity_set": list(nset or ()),
            "anchors": anchors,
            "precision": d.get("precision"),
            "concordance_route": list(d.get("concordance_route") or ()),
        }
    return {
        "name": getattr(proto, "name", None),
        "hash": getattr(proto, "hash", None),
        "necessity_set": list(getattr(proto, "necessity_set", ()) or ()),
        "anchors": copy.deepcopy(_plain(getattr(proto, "anchors", None))),
        "precision": copy.deepcopy(_plain(getattr(proto, "precision", None))),
        "concordance_route": [
            _plain(c) for c in (getattr(proto, "concordance_route", ()) or ())
        ],
    }


def _plain(obj):
    if isinstance(obj, Mapping):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


@dataclass
class Context:
    """Everything a part may read: row sources, the frozen family protocols
    (by protocol id), the evaluation split, excluded task ids (integrity
    audit), the CD-8 mechanism-on table and the usability map of
    prerequisite M."""

    sources: Dict[str, List[dict]] = field(default_factory=dict)
    protocols: Dict[str, dict] = field(default_factory=dict)
    split: str = S.DEVELOPMENT
    excluded_task_ids: frozenset = frozenset()
    mechanism_on: Optional[Sequence[Mapping]] = None
    seed_blocks: Optional[Mapping] = None
    usability: Dict[Tuple[str, str], bool] = field(default_factory=dict)
    uncalibrated: frozenset = frozenset()

    def rows(self, name: str, designs: Optional[Sequence] = None) -> List[dict]:
        """Rows of a source (excluded tasks dropped); with ``designs``, only
        the rows of those designs (through a cached index)."""
        rows = self.sources.get(name)
        if rows is None:
            return []
        if designs is not None:
            index = self.__dict__.setdefault("_design_index", {})
            key = (name, id(rows))
            if key not in index:
                by = {}
                for r in rows:
                    by.setdefault(r.get("design"), []).append(r)
                index[key] = by
            rows = [r for d in designs for r in index[key].get(d, ())]
        if not self.excluded_task_ids:
            return rows
        return [r for r in rows if r.get("task_id") not in self.excluded_task_ids]


def build_context(
    records: Iterable = (),
    *,
    sources: Mapping = None,
    protocols=None,
    split: Optional[str] = None,
    excluded_task_ids=(),
    mechanism_on=None,
    seed_blocks=None,
) -> Context:
    """A context from ``/3`` records (rows of ``components`` and
    ``verdicts``) and auxiliary sources (mappings name -> list of rows).
    ``protocols``: ``{protocol key: ProtocolV3 or mapping}`` or a list of
    them (keyed by the key in their name, :func:`protocol_key`); rows find
    their protocol by their ``protocol_id``, the key the runner writes. The
    split is the records' split (one split
    per evaluation; mixing is refused) unless given; the seeds of the
    auxiliary sources (for example the manipulation checks) must belong to
    the same split, so a confirmatory evaluation never reads development
    checks (``SeedPolicyError`` otherwise)."""
    recs = [_as_dict(r) for r in records]
    extra = {k: [dict(r) for r in v] for k, v in dict(sources or {}).items()}
    seeds = [r.get("seed") for r in recs if r.get("seed") is not None]
    aux = [
        r.get("seed")
        for rows in extra.values()
        for r in rows
        if r.get("seed") is not None and not _blank(r.get("seed"))
    ]
    if split is None:
        split = S.check_seeds(seeds + aux) if seeds or aux else S.DEVELOPMENT
    elif seeds or aux:
        S.check_seeds(seeds + aux, split)
    srcs = {"components": component_rows(recs), "verdicts": verdict_rows(recs)}
    srcs.update(extra)
    protos = {}
    if protocols:
        items = (
            protocols.items()
            if isinstance(protocols, Mapping)
            else ((None, p) for p in protocols)
        )
        for key, pr in items:
            info = _protocol_info(pr)
            protos[str(key if key is not None else protocol_key(pr))] = info
    return Context(
        srcs,
        protos,
        split,
        frozenset(excluded_task_ids or ()),
        None if mechanism_on is None else list(mechanism_on),
        None if seed_blocks is None else dict(seed_blocks),
    )


# ==========================================================================
# the specification
# ==========================================================================
_SPEC_KEYS = {
    "schema",
    "version",
    "status",
    "design",
    "conventions",
    "vocabulary",
    "fields",
    "derived",
    "seed_blocks",
    "witness_targets",
    "own_lesions",
    "declared_dependencies",
    "mechanism_on",
    "oracle_checks",
    "pending_calibration",
    "hypotheses",
    "integrity_audit",
    "descriptive",
    "notes",
}
_HYP_KEYS = {
    "id",
    "title",
    "tier",
    "labels",
    "admitted",
    "role",
    "prediction",
    "parts",
    "verdict_level_absent",
    "successor_of",
    "expectation",
    "why_v1_differed",
    "statement",
    "notes",
    "combine",
}
_PART_KEYS = {
    "id",
    "text",
    "label",
    "role",
    "status",
    "rule",
    "params",
    "data",
    "requires",
    "gate",
    "gating_outcome",
    "gating_outcome_by_cell",
    "prediction",
    "cell_predictions",
    "replaced_by",
    "removed",
    "choose",
    "notes",
    "expectation",
}
_DATA_KEYS = {
    "source",
    "where",
    "union",
    "label",
    "pair",
    "event",
    "value",
    "dose",
    "group",
    "cells",
    "cell_label",
    "cluster",
    "pool_by",
    "principle_filter",
    "target_filter",
    "exclude",
    "mechanism_on",
    "min_rows",
    "network",
    "defined",
    "exact",
    "se",
    "df",
    "protocol_predictions",
}
_MEMBER_KEYS = _DATA_KEYS - {"union", "cluster", "pool_by", "min_rows"}
TIER_A_IDS = tuple(f"HCv2-{i}" for i in range(25))
IA_IDS = tuple(f"IA-{i}" for i in range(1, 11))


def spec_sha256(spec: Mapping) -> str:
    """SHA-256 of the canonical JSON of a hypotheses file (sorted keys)."""
    text = json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_spec(path=None) -> dict:
    p = Path(path) if path is not None else SPEC_PATH
    return validate_spec(json.loads(p.read_text(encoding="utf-8")))


def _vocab(spec) -> dict:
    return dict(spec.get("vocabulary") or {})


def resolve(value, spec):
    """Replace ``"@vocabulary.key"`` strings (recursively) by their value;
    field aliases (``@name`` where ``name`` is an alias) are left for the
    field resolver."""
    vocab = _vocab(spec)
    if isinstance(value, str) and value.startswith("@") and value[1:] in vocab:
        return copy.deepcopy(vocab[value[1:]])
    if isinstance(value, list):
        out = []
        for v in value:
            r = resolve(v, spec)
            if (
                isinstance(v, str)
                and v.startswith("@")
                and v[1:] in vocab
                and isinstance(r, list)
            ):
                out.extend(r)
            else:
                out.append(r)
        return out
    if isinstance(value, Mapping):
        return {
            (
                resolve(k, spec)
                if isinstance(k, str) and k.startswith("@") and k[1:] in vocab
                else k
            ): resolve(v, spec)
            for k, v in value.items()
        }
    return value


def _check_predicate(pred, where, fields):
    if pred is None or isinstance(pred, bool):
        return
    if not isinstance(pred, Mapping):
        raise SpecError(f"{where}: a predicate is an object")
    for k, cond in pred.items():
        if k in ("all", "any"):
            if not isinstance(cond, list):
                raise SpecError(f"{where}: {k} takes a list")
            for c in cond:
                _check_predicate(c, where, fields)
        elif k == "not":
            _check_predicate(cond, where, fields)
        else:
            if (
                k.startswith("@")
                and k[1:] not in fields.aliases
                and k[1:] not in fields.derived
            ):
                raise SpecError(f"{where}: unknown field alias {k!r}")
            if isinstance(cond, Mapping):
                bad = set(cond) - set(_OPS)
                if bad:
                    raise SpecError(f"{where}: unknown operators {sorted(bad)}")


def _check_data(data, where, spec, fields, *, member=False):
    if not isinstance(data, Mapping):
        raise SpecError(f"{where}: data must be an object")
    allowed = _MEMBER_KEYS if member else _DATA_KEYS
    bad = set(data) - allowed
    if bad:
        raise SpecError(f"{where}: unknown data keys {sorted(bad)}")
    for k in ("where", "event", "defined"):
        if k in data:
            w = resolve(data[k], spec)
            if isinstance(w, Mapping) and "seed_block" in w:
                w = dict(w)
                sb = w.pop("seed_block")
                if sb not in (spec.get("seed_blocks") or {}):
                    raise SpecError(f"{where}: unknown seed block {sb!r}")
            _check_predicate(w, where, fields)
    for k in ("value", "dose", "group", "exact", "se", "df"):
        v = data.get(k)
        if isinstance(v, str) and v.startswith("@") and v[1:] not in fields.aliases:
            raise SpecError(f"{where}: unknown field alias {v!r}")
    if "pair" in data:
        pr = data["pair"]
        if not isinstance(pr, Mapping) or not {"on", "a", "b"} <= set(pr):
            raise SpecError(f"{where}: pair needs on, a and b")
        _check_predicate(resolve(pr["a"], spec), where, fields)
        _check_predicate(resolve(pr["b"], spec), where, fields)
    for m in data.get("union") or ():
        _check_data(m, where, spec, fields, member=True)
    for k in ("principle_filter", "target_filter"):
        if k in data and data[k] not in ("valid_anchor", "in_n_anch"):
            raise SpecError(f"{where}: {k} must be valid_anchor or in_n_anch")
    for e in data.get("exclude") or ():
        if e not in ("target", "non_target", "declared_dependencies"):
            raise SpecError(f"{where}: unknown exclusion {e!r}")
    if "mechanism_on" in data and data["mechanism_on"] not in ("own", "all_n_anch"):
        raise SpecError(f"{where}: mechanism_on must be own or all_n_anch")


def _check_refs(obj, where, vocab, fields) -> None:
    """Every ``@name`` (a value, a key or a ``{@name}`` in a template) names a
    vocabulary entry, a field alias or a derived field, so a misspelt name
    can never select nothing in silence."""

    def ok(name):
        return name in vocab or name in fields.aliases or name in fields.derived

    if isinstance(obj, str):
        if obj.startswith("@") and not ok(obj[1:]):
            raise SpecError(f"{where}: unknown reference {obj!r}")
        for ref in _TEMPLATE_RE.findall(obj):
            ref = ref.strip()
            if ref.startswith("@") and not ok(ref[1:]):
                raise SpecError(f"{where}: unknown reference {{{ref}}}")
    elif isinstance(obj, Mapping):
        for k, v in obj.items():
            _check_refs(k, where, vocab, fields)
            _check_refs(v, where, vocab, fields)
    elif isinstance(obj, list):
        for v in obj:
            _check_refs(v, where, vocab, fields)


def validate_spec(spec: Mapping) -> dict:
    """Check a hypotheses file: schema, keys, unique hypothesis and part
    ids, known rules with their required parameters, data selections with
    known keys, operators, aliases and seed blocks (development blocks in
    0-999, confirmatory blocks at or above 20000), every Tier-A hypothesis
    HCv2-0 to HCv2-24 and every integrity check IA-1 to IA-10 listed.
    Returns the spec; raises :class:`SpecError`."""
    if not isinstance(spec, Mapping):
        raise SpecError("the hypotheses file is an object")
    if spec.get("schema") != SPEC_SCHEMA:
        raise SpecError(f"schema must be {SPEC_SCHEMA!r}")
    bad = set(spec) - _SPEC_KEYS
    if bad:
        raise SpecError(f"unknown top-level keys {sorted(bad)}")
    fields = Fields(spec.get("fields"), spec.get("derived"))
    vocab = _vocab(spec)
    clash = (set(fields.aliases) | set(fields.derived)) & set(vocab)
    if clash or set(fields.aliases) & set(fields.derived):
        raise SpecError(
            f"names used twice (vocabulary, alias, derived): {sorted(clash)}"
        )
    for name, pred in (spec.get("derived") or {}).items():
        _check_predicate(pred, f"derived {name}", fields)
        _check_refs(pred, f"derived {name}", vocab, fields)
    _check_refs(
        (spec.get("mechanism_on") or {}).get("entries") or [],
        "mechanism_on",
        vocab,
        fields,
    )
    oracle = spec.get("oracle_checks")
    if oracle is not None:
        if not isinstance(oracle, Mapping) or not isinstance(
            oracle.get("systems"), Mapping
        ):
            raise SpecError("oracle_checks needs a 'systems' map")
        for name, checks in oracle["systems"].items():
            if not isinstance(checks, list) or not all(
                isinstance(c, str) and c for c in checks
            ):
                raise SpecError(f"oracle_checks of {name}: a list of check ids")
        _check_refs(oracle["systems"], "oracle_checks", vocab, fields)
    for name, blk in (spec.get("seed_blocks") or {}).items():
        dev, conf = blk.get("development"), blk.get("confirmatory")
        if dev is not None:
            if len(dev) != 2 or dev[0] > dev[1]:
                raise SpecError(f"seed block {name}: development is [first, last]")
            try:
                S.assert_development(dev)
            except S.SeedPolicyError as exc:
                raise SpecError(f"seed block {name}: {exc}") from None
        if conf is not None:
            if len(conf) != 2 or conf[0] > conf[1]:
                raise SpecError(f"seed block {name}: confirmatory is [first, last]")
            try:
                S.assert_confirmatory(conf)
            except S.SeedPolicyError as exc:
                raise SpecError(f"seed block {name}: {exc}") from None
    for w, p in (spec.get("witness_targets") or {}).items():
        if p not in PRINCIPLES:
            raise SpecError(f"witness target of {w} must be a principle")
    hyps = spec.get("hypotheses")
    if not isinstance(hyps, list) or not hyps:
        raise SpecError("hypotheses must be a non-empty list")
    hids, pids = set(), set()
    for h in hyps:
        bad = set(h) - _HYP_KEYS
        if bad:
            raise SpecError(f"{h.get('id')}: unknown keys {sorted(bad)}")
        hid = h.get("id")
        if not hid or hid in hids:
            raise SpecError(f"duplicate or missing hypothesis id {hid!r}")
        hids.add(hid)
        if h.get("tier") not in ("A", "B"):
            raise SpecError(f"{hid}: tier must be A or B")
        parts = h.get("parts") or []
        if h.get("tier") == "A" and not parts:
            raise SpecError(f"{hid}: a Tier-A hypothesis has parts")
        for part in parts:
            pid = part.get("id")
            where = f"{hid} part {pid!r}"
            if not pid or pid in pids:
                raise SpecError(f"{where}: duplicate or missing part id")
            pids.add(pid)
            bad = set(part) - _PART_KEYS
            if bad:
                raise SpecError(f"{where}: unknown keys {sorted(bad)}")
            if part.get("role", DECISIVE) not in ROLES:
                raise SpecError(f"{where}: role must be one of {ROLES}")
            st = part.get("status", ACTIVE)
            if st not in PART_STATUSES:
                raise SpecError(f"{where}: status must be one of {PART_STATUSES}")
            if st == REMOVED_STATUS:
                rem = part.get("removed") or {}
                if rem.get("development_outcome") not in OUTCOMES:
                    raise SpecError(
                        f"{where}: a removed part states its development " "outcome"
                    )
                continue
            for key in (
                "data",
                "params",
                "gate",
                "cell_predictions",
                "choose",
                "requires",
            ):
                if key in part:
                    _check_refs(part[key], where, vocab, fields)
            req = part.get("requires")
            if req is not None:
                if not isinstance(req, Mapping) or set(req) - {"usable", "oracle"}:
                    raise SpecError(f"{where}: requires takes 'usable' and 'oracle'")
                usable = req.get("usable", [])
                if not isinstance(usable, list) or not all(
                    isinstance(c, str) and c for c in usable
                ):
                    raise SpecError(f"{where}: requires.usable is a list of check ids")
                if not isinstance(req.get("oracle", False), bool):
                    raise SpecError(f"{where}: requires.oracle is true or false")
                if req.get("oracle") and oracle is None:
                    raise SpecError(f"{where}: requires.oracle needs oracle_checks")
            rule = part.get("rule")
            if rule not in RULES:
                raise SpecError(f"{where}: unknown rule {rule!r}")
            params = part.get("params") or {}
            missing = [k for k in RULE_REQUIRED.get(rule, ()) if k not in params]
            if missing:
                raise SpecError(f"{where}: rule {rule} needs {missing}")
            if "data" not in part:
                raise SpecError(f"{where}: no data selection")
            _check_data(part["data"], where, spec, fields)
            if rule in _NEEDS_EVENT and not _has_event(part["data"]):
                raise SpecError(f"{where}: rule {rule} needs an event predicate")
            for k in ("prediction",):
                if k in part and part[k] not in OUTCOMES:
                    raise SpecError(f"{where}: prediction must be an outcome")
            go = part.get("gating_outcome")
            if go is not None and go not in ("decisive", NOT_TESTABLE_BY_DESIGN):
                raise SpecError(
                    f"{where}: gating_outcome must be decisive or "
                    f"{NOT_TESTABLE_BY_DESIGN}"
                )
            if "choose" in part:
                ch = part["choose"]
                if not {"if", "then", "else"} <= set(ch):
                    raise SpecError(f"{where}: choose needs if, then and else")
                for br in ("then", "else"):
                    sub = ch[br]
                    if "rule" in sub and sub["rule"] not in RULES:
                        raise SpecError(f"{where}: unknown rule in choose.{br}")
                    if "data" in sub:
                        _check_data(sub["data"], f"{where} choose.{br}", spec, fields)
    missing_a = [i for i in TIER_A_IDS if i not in hids]
    if missing_a:
        raise SpecError(f"missing Tier-A hypotheses {missing_a}")
    ia = [e.get("id") for e in spec.get("integrity_audit") or ()]
    if sorted(ia) != sorted(IA_IDS):
        raise SpecError(f"the integrity audit must list {list(IA_IDS)}")
    for part_id in (p.get("replaced_by") for h in hyps for p in h.get("parts") or ()):
        if part_id is not None and part_id not in pids:
            raise SpecError(f"replaced_by names an unknown part {part_id!r}")
    return dict(spec)


def _has_event(data) -> bool:
    if "event" in data:
        return True
    members = data.get("union") or ()
    return bool(members) and all("event" in m for m in members)


# ==========================================================================
# data preparation
# ==========================================================================
def _seed_range(spec, ctx: Context, name) -> Tuple[int, int]:
    blocks = dict(spec.get("seed_blocks") or {})
    if ctx.seed_blocks:
        blocks.update(ctx.seed_blocks)
    blk = blocks.get(name)
    if blk is None:
        raise SpecError(f"unknown seed block {name!r}")
    rng = blk.get(ctx.split)
    if rng is None:
        raise SpecError(f"seed block {name!r} has no {ctx.split} range")
    return int(rng[0]), int(rng[1])


def _where(pred, spec, ctx):
    """A predicate with ``seed_block`` replaced by a seed range."""
    if pred is None:
        return None
    pred = dict(resolve(pred, spec))
    sb = pred.pop("seed_block", None)
    if sb is not None:
        lo, hi = _seed_range(spec, ctx, sb)
        rng = {"seed": {"min": lo, "max": hi}}
        if "seed" in pred:
            pred = {**pred, "all": list(pred.get("all") or []) + [rng]}
        else:
            pred["seed"] = rng["seed"]
    return pred


def _proto_for(row, ctx: Context) -> Optional[dict]:
    pid = row.get("protocol_id")
    return None if pid is None else ctx.protocols.get(str(pid))


def _anchor_status(proto: dict, principle) -> Optional[str]:
    anchors = proto.get("anchors") if proto else None
    if not isinstance(anchors, Mapping):
        return None
    entry = (anchors.get("principles") or {}).get(principle)
    return None if not isinstance(entry, Mapping) else entry.get("status")


def _n_anch(proto: dict) -> Optional[set]:
    if not proto:
        return None
    anchors = proto.get("anchors")
    if isinstance(anchors, Mapping) and anchors.get("necessity_set") is not None:
        return set(anchors["necessity_set"])
    ns = proto.get("necessity_set")
    return None if ns is None else set(ns)


def _principle_ok(row, ctx, mode, principle=None) -> Optional[bool]:
    proto = _proto_for(row, ctx)
    if proto is None:
        return None
    p = principle or row.get("principle")
    if mode == "in_n_anch":
        n = _n_anch(proto)
        return None if n is None else (p in n)
    st = _anchor_status(proto, p)
    return None if st is None else st in ("valid_specific", "valid_nonspecific")


def _mechanism_entries(spec, ctx) -> Tuple[Optional[list], Optional[str]]:
    if ctx.mechanism_on is not None:
        return resolve(list(ctx.mechanism_on), spec), "context"
    block = spec.get("mechanism_on") or {}
    status = block.get("status")
    if ctx.split == S.CONFIRMATORY and status != "final":
        return None, f"mechanism-on labels are {status!r}, not final (CD-8)"
    return resolve(list(block.get("entries") or ()), spec), status


def _mechanism_on(row, principle, entries, fields) -> bool:
    """Whether ``principle``'s mechanism is on in ``row`` by the CD-8 table:
    the first matching entry decides (unmatched: not on). An entry's
    ``where`` sees the row with ``principle`` set to the principle asked
    about, so ``own_bit`` refers to it (also on verdict rows)."""
    row = {**row, "principle": principle}
    for e in entries:
        if e.get("principle") not in (None, principle):
            continue
        if e.get("family") not in (None, row.get("family")):
            continue
        if e.get("declaration") not in (None, row.get("declaration_id")):
            continue
        w = e.get("where")
        if w is not None and evaluate_predicate(w, row, fields) is not True:
            continue
        return bool(e.get("on", True))
    return False


def _pair_rows(rows, pair, spec, ctx, fields, notes):
    on = [str(k) for k in pair["on"]]
    a_pred, b_pred = _where(pair["a"], spec, ctx), _where(pair["b"], spec, ctx)
    vfield = pair.get("value", "c")
    side = {"a": OrderedDict(), "b": OrderedDict()}
    for r in rows:
        for nm, pred in (("a", a_pred), ("b", b_pred)):
            if evaluate_predicate(pred, r, fields) is True:
                key = tuple(_group_key(fields.get(r, k)) for k in on)
                side[nm].setdefault(key, []).append(r)
    out, conflicts = [], 0
    for key, ras in side["a"].items():
        rbs = side["b"].get(key)
        if not rbs:
            continue
        va = {_finite(fields.get(r, vfield)) for r in ras}
        vb = {_finite(fields.get(r, vfield)) for r in rbs}
        if len(va) > 1 or len(vb) > 1:
            conflicts += 1
            continue
        a_v, b_v = next(iter(va)), next(iter(vb))
        row = dict(ras[0])
        row["value_a"], row["value_b"] = a_v, b_v
        row["delta"] = None if a_v is None or b_v is None else a_v - b_v
        row["b_task_id"] = rbs[0].get("task_id")
        row["b_system"] = rbs[0].get("system")
        row["_b_row"] = rbs[0]
        row["_pair_key"] = "|".join(key)
        out.append(row)
    if conflicts:
        notes.append(f"{conflicts} pair keys with conflicting duplicate rows dropped")
    return out


def _designs_of(preds) -> Optional[list]:
    """The designs a filter admits when one of its top-level conditions is
    an equality or membership on ``design`` (an index can then serve it)."""
    for pr in preds:
        if isinstance(pr, Mapping) and "design" in pr:
            v = pr["design"]
            if isinstance(v, str):
                return [v]
            if isinstance(v, list) and all(isinstance(x, str) for x in v):
                return list(v)
    return None


def prepare(
    data: Mapping, spec: Mapping, ctx: Context, fields: Fields = None, part_id: str = ""
) -> Prepared:
    """Select and annotate the rows of a data selection (see the module
    docstring)."""
    fields = fields or Fields(spec.get("fields"), spec.get("derived"))
    data = dict(data)
    members = data.pop("union", None) or [{}]
    notes: List[str] = []
    rows_all: List[dict] = []
    targets = dict(spec.get("witness_targets") or {})
    deps = (
        {tuple(d) for d in spec.get("declared_dependencies", {}).get("pairs", ())}
        if isinstance(spec.get("declared_dependencies"), Mapping)
        else {tuple(d) for d in (spec.get("declared_dependencies") or ())}
    )
    mech_entries = None
    for idx, member in enumerate(members):
        eff = {**{k: v for k, v in data.items() if k in _MEMBER_KEYS}, **member}
        label = str(member.get("label", data.get("label", idx)))
        src = eff.get("source", "components")
        preds = [_where(data.get("where"), spec, ctx)]
        if "where" in member:
            preds.append(_where(member.get("where"), spec, ctx))
        rows = ctx.rows(src, _designs_of(preds))
        rows = [
            dict(r)
            for r in rows
            if all(evaluate_predicate(pr, r, fields) is True for pr in preds)
        ]
        if eff.get("principle_filter"):
            kept, unknown = [], 0
            for r in rows:
                ok = _principle_ok(r, ctx, eff["principle_filter"])
                if ok is None:
                    unknown += 1
                elif ok:
                    kept.append(r)
            if unknown:
                notes.append(
                    f"{label}: {unknown} rows without protocol anchor information "
                    "dropped"
                )
            rows = kept
        if "pair" in eff:
            # pair rows are copies of the a rows (system = the a system), so the
            # witness-based filters below see the witness of a paired contrast
            rows = _pair_rows(rows, eff["pair"], spec, ctx, fields, notes)
        if eff.get("target_filter"):
            kept, unknown = [], 0
            for r in rows:
                t = targets.get(r.get("system"))
                if t is None:
                    continue
                ok = _principle_ok(r, ctx, eff["target_filter"], t)
                if ok is None:
                    unknown += 1
                elif ok:
                    r["target"] = t
                    kept.append(r)
            if unknown:
                notes.append(
                    f"{label}: {unknown} rows without protocol anchor information "
                    "dropped"
                )
            rows = kept
        for ex in eff.get("exclude") or ():
            if ex == "target":
                rows = [
                    r
                    for r in rows
                    if r.get("principle") != targets.get(r.get("system"))
                ]
            elif ex == "non_target":
                rows = [
                    r
                    for r in rows
                    if r.get("principle") == targets.get(r.get("system"))
                ]
            elif ex == "declared_dependencies":
                rows = [
                    r for r in rows if (r.get("system"), r.get("principle")) not in deps
                ]
        if eff.get("mechanism_on"):
            if mech_entries is None:
                mech_entries, why = _mechanism_entries(spec, ctx)
                if mech_entries is None:
                    raise _NotEvaluable(why)
            if eff["mechanism_on"] == "own":
                rows = [
                    r
                    for r in rows
                    if _mechanism_on(r, r.get("principle"), mech_entries, fields)
                ]
            else:
                kept = []
                for r in rows:
                    n = _n_anch(_proto_for(r, ctx))
                    if n is None:
                        continue
                    if all(_mechanism_on(r, p, mech_entries, fields) for p in n):
                        kept.append(r)
                rows = kept
        ev_pred = resolve(eff.get("event"), spec) if "event" in eff else None
        def_pred = resolve(eff.get("defined"), spec) if "defined" in eff else None
        for r in rows:
            r["member"] = label
            if "cell_label" in eff:
                r["_cell"] = render_template(eff["cell_label"], r, fields)
            elif eff.get("cells"):
                r["_cell"] = "|".join(
                    f"{k.lstrip('@')}={_group_key(fields.get(r, k))}"
                    for k in eff["cells"]
                )
            else:
                r["_cell"] = "all"
            r["_event"] = (
                evaluate_predicate(ev_pred, r, fields) if ev_pred is not None else None
            )
            r["_value"] = (
                fields.get(r, eff["value"])
                if eff.get("value")
                else (r.get("delta") if "pair" in eff else r.get("c"))
            )
            r["_dose"] = fields.get(r, eff["dose"]) if eff.get("dose") else None
            r["_group"] = (
                label
                if eff.get("group") == "member"
                else fields.get(r, eff["group"]) if eff.get("group") else None
            )
            r["_exact"] = fields.get(r, eff["exact"]) if eff.get("exact") else None
            if eff.get("se"):
                r["_se"] = fields.get(r, eff["se"])
            if eff.get("df"):
                r["_df"] = fields.get(r, eff["df"])
            r["_defined"] = (
                evaluate_predicate(def_pred, r, fields) is True
                if def_pred is not None
                else bool(_builtin_derived(r, "defined"))
            )
            if eff.get("network"):
                r["_network"] = "|".join(
                    _group_key(fields.get(r, k)) for k in eff["network"]
                )
        rows_all.extend(rows)
    cl = data.get("cluster")
    pool = data.get("pool_by")
    for r in rows_all:
        if cl:
            r["_cluster"] = "|".join(_group_key(fields.get(r, k)) for k in cl)
        elif "_pair_key" in r:
            r["_cluster"] = r["_pair_key"]
        else:
            r["_cluster"] = r.get("task_id") or id(r)
        if pool:
            r["_pool"] = "|".join(_group_key(fields.get(r, k)) for k in pool)
        else:
            r["_pool"] = r.get("principle", "all")
        r.setdefault("_network", r.get("_cluster"))
    return Prepared(rows_all, notes, fields)


class _NotEvaluable(Exception):
    """A part input that is not available (raised during preparation)."""


# ==========================================================================
# evaluation
# ==========================================================================
def _precision_rows(ctx: Context) -> List[dict]:
    out = []
    for pid, info in ctx.protocols.items():
        block = info.get("precision")
        if isinstance(block, Mapping):
            for row in block.get("rows") or ():
                out.append({**row, "_protocol": pid})
    return out


def _resolve_gate(
    gate: Mapping, ctx: Context, spec, row=None, fields=None
) -> Optional[dict]:
    """The precision row of a gate (kind, family, principle, witness; values
    may be templates on a cell row); None when no protocol declares it."""
    g = resolve(dict(gate), spec)
    if row is not None:
        g = {
            k: render_template(v, row, fields) if isinstance(v, str) else v
            for k, v in g.items()
        }
    for pr in _precision_rows(ctx):
        if all(
            str(pr.get(k)) == str(g.get(k))
            for k in ("kind", "family", "principle", "witness")
            if k in g
        ):
            return pr
    return None


def _concordance_admitted(cond: Mapping, ctx: Context) -> Optional[bool]:
    seen = False
    for info in ctx.protocols.values():
        route = info.get("concordance_route")
        if route is None:
            continue
        seen = True
        for cell in route:
            if all(
                str(cell.get(k)) == str(v)
                for k, v in cond.items()
                if k not in ("direction",)
            ):
                if cell.get(cond.get("direction", "absent")):
                    return True
    return False if seen and ctx.protocols else None


def _row_checks(row, req: Mapping, spec: Mapping, fields) -> List[str]:
    """The prerequisite-M checks a row needs: the part's ``usable`` checks
    and, with ``oracle``, the checks of the row's system (and of the b system
    of a paired row) in the file's ``oracle_checks``; check ids may be
    templates on the row (``{system}:{@variant}/stated_time_constants``)."""
    out = [render_template(c, row, fields) for c in req.get("usable") or ()]
    if req.get("oracle"):
        systems = (spec.get("oracle_checks") or {}).get("systems") or {}
        for r in (row, row.get("_b_row")):
            if r is not None:
                out += [
                    render_template(c, r, fields)
                    for c in systems.get(r.get("system"), ())
                ]
    return out


def _requires(
    part, ctx, prep: Prepared, fields, spec=None
) -> Tuple[Optional[str], list]:
    """Prerequisite M: drop the rows that need a manipulation or realisation
    check that is not usable in their family (reason ORACLE). Returns the
    blocking reason when this check dropped every row that was left (a
    selection that was already empty is reported by the caller as having
    no input rows) and the cells that lost every row (listed as
    NOT_EVALUABLE cells, never silently dropped)."""
    req = part.get("requires") or {}
    if not (req.get("usable") or req.get("oracle")):
        return None, []
    if not ctx.usability:
        prep.notes.append("no manipulation-check input: prerequisite M not evaluated")
        return None, []
    kept = []
    failing: "OrderedDict[str, set]" = OrderedDict()
    n_kept: Dict[str, int] = {}
    for r in prep.rows:
        fam = str(r.get("family"))
        bad = sorted(
            {
                c
                for c in _row_checks(r, req, spec or {}, fields)
                if not ctx.usability.get((fam, str(c)), False)
            }
        )
        cell = r.get("_cell")
        n_kept.setdefault(cell, 0)
        if bad:
            failing.setdefault(cell, set()).update(f"{fam}:{c}" for c in bad)
        else:
            kept.append(r)
            n_kept[cell] += 1
    if failing:
        dropped = len(prep.rows) - len(kept)
        prep.notes.append(
            f"{dropped} rows dropped: a required manipulation or realisation "
            "check is not usable in their family (ORACLE)"
        )
    oracle_cells = [
        {
            "cell": cell,
            "outcome": NOT_EVALUABLE,
            "reason": "ORACLE: a required manipulation check is not usable",
            "checks": sorted(chk),
        }
        for cell, chk in failing.items()
        if n_kept.get(cell, 0) == 0
    ]
    prep.rows = kept
    if not kept and failing:
        return "ORACLE: a required manipulation check is not usable", oracle_cells
    return None, oracle_cells


def _apply_choose(part, ctx, spec) -> Tuple[dict, Optional[str]]:
    ch = part.get("choose")
    if not ch:
        return part, None
    cond = dict(ch["if"])
    if "concordance_admitted" in cond:
        ok = _concordance_admitted(resolve(cond["concordance_admitted"], spec), ctx)
        if ok is None:
            return (
                part,
                "concordance admission unknown (no protocol with a route block)",
            )
    else:
        raise SpecError(f"{part['id']}: unknown choose condition {sorted(cond)}")
    branch = ch["then"] if ok else ch["else"]
    out = {**part, **{k: v for k, v in branch.items() if k != "text"}}
    out["chosen"] = "then" if ok else "else"
    if "text" in branch:
        out["chosen_text"] = branch["text"]
    return out, None


EVALUATOR_ERROR = "EVALUATOR_ERROR"


def evaluate_part(
    part: Mapping, spec: Mapping, ctx: Context, fields: Fields = None
) -> dict:
    """Evaluate one part: removed, gated (NOT_TESTABLE_BY_DESIGN), not
    evaluable, or the outcome of its rule on its data. An unexpected error
    of the evaluator inside the part (not a malformed file, which is refused
    before) makes the part NOT_EVALUABLE with reason ``EVALUATOR_ERROR`` and
    ``evaluator_error`` set, so that one defect neither stops the other
    parts nor passes unnoticed (:func:`evaluate` lists them)."""
    try:
        return _evaluate_part(part, spec, ctx, fields)
    except SpecError:
        raise
    except Exception as exc:  # noqa: BLE001 - recorded and listed, never hidden
        return {
            "id": part.get("id"),
            "text": part.get("text"),
            "label": part.get("label"),
            "role": part.get("role", DECISIVE),
            "rule": part.get("rule"),
            "prediction": part.get("prediction", SUPPORTED),
            "status": ACTIVE,
            "outcome": NOT_EVALUABLE,
            "reason": f"{EVALUATOR_ERROR}: {type(exc).__name__}: {exc}",
            "evaluator_error": True,
            "notes": [],
            "replaced_by": part.get("replaced_by"),
        }


def _evaluate_part(
    part: Mapping, spec: Mapping, ctx: Context, fields: Fields = None
) -> dict:
    fields = fields or Fields(spec.get("fields"), spec.get("derived"))
    base = {
        "id": part["id"],
        "text": part.get("text"),
        "label": part.get("label"),
        "role": part.get("role", DECISIVE),
        "rule": part.get("rule"),
        "prediction": part.get("prediction", SUPPORTED),
        "notes": [],
        "replaced_by": part.get("replaced_by"),
    }
    if part.get("status", ACTIVE) == REMOVED_STATUS:
        rem = part.get("removed") or {}
        return {
            **base,
            "status": REMOVED_STATUS,
            "outcome": REMOVED,
            "predicted_outcome": rem.get("development_outcome"),
            "reason": rem.get("reason"),
        }
    part, why = _apply_choose(dict(part), ctx, spec)
    if why:
        return {**base, "status": ACTIVE, "outcome": NOT_EVALUABLE, "reason": why}
    if "chosen" in part:
        base["chosen"] = part["chosen"]
        base["rule"] = part.get("rule")
        if part.get("chosen_text"):
            base["chosen_text"] = part["chosen_text"]
    params = resolve(dict(part.get("params") or {}), spec)
    # part-level gate
    gate = part.get("gate")
    gating = part.get("gating_outcome")
    gate_info = None
    if gate and not gate.get("per_cell") and gating is None:
        gate_info = _resolve_gate(gate, ctx, spec)
        if gate_info is None:
            return {
                **base,
                "status": ACTIVE,
                "outcome": NOT_EVALUABLE,
                "reason": "testability gate not declared in the frozen protocols",
            }
        gating = "decisive" if gate_info.get("decisive") else NOT_TESTABLE_BY_DESIGN
    if gating == NOT_TESTABLE_BY_DESIGN:
        # declared before the freeze (in the file, or by the protocol's gate)
        return {
            **base,
            "status": ACTIVE,
            "outcome": NOT_TESTABLE_BY_DESIGN,
            "reason": "development rate below the gate (pi < 0.9)",
            "gate": _gate_summary(gate_info),
            "replacement": (gate_info or {}).get("replacement"),
        }
    try:
        prep = prepare(part["data"], spec, ctx, fields, part["id"])
    except _NotEvaluable as exc:
        return {**base, "status": ACTIVE, "outcome": NOT_EVALUABLE, "reason": str(exc)}
    blocked, oracle_cells = _requires(part, ctx, prep, fields, spec)
    if blocked:
        return {
            **base,
            "status": ACTIVE,
            "outcome": NOT_EVALUABLE,
            "reason": blocked,
            "cells": oracle_cells,
            "notes": prep.notes,
        }
    if part.get("rule") == "anchor_replicates":
        _annotate_anchor_predictions(prep, ctx, part, spec)
    min_rows = int((part.get("data") or {}).get("min_rows", 1))
    if len(prep.rows) < min_rows:
        return {
            **base,
            "status": ACTIVE,
            "outcome": NOT_EVALUABLE,
            "reason": (
                "no input rows" if not prep.rows else f"fewer than {min_rows} rows"
            ),
            "notes": prep.notes,
        }
    # per-cell gate
    cell_gates = {}
    if gate and gate.get("per_cell"):
        keep = []
        decided = OrderedDict()
        for r in prep.rows:
            if r["_cell"] in decided:
                if decided[r["_cell"]] == "decisive":
                    keep.append(r)
                continue
            fixed = (part.get("gating_outcome_by_cell") or {}).get(r["_cell"])
            info = None
            if fixed is None:
                info = _resolve_gate(
                    {k: v for k, v in gate.items() if k != "per_cell"},
                    ctx,
                    spec,
                    r,
                    fields,
                )
                fixed = (
                    None
                    if info is None
                    else "decisive" if info.get("decisive") else NOT_TESTABLE_BY_DESIGN
                )
            decided[r["_cell"]] = fixed
            cell_gates[r["_cell"]] = {
                **(_gate_summary(info) or {}),
                "outcome": fixed or NOT_EVALUABLE,
            }
            if fixed == "decisive":
                keep.append(r)
        prep.rows = keep
    rule = RULES[part["rule"]]
    if prep.rows:
        res = rule(prep, params)
    elif cell_gates and all(
        g["outcome"] == NOT_TESTABLE_BY_DESIGN for g in cell_gates.values()
    ):
        res = RuleResult(
            NOT_TESTABLE_BY_DESIGN, "every cell is below the gate (pi < 0.9)"
        )
    else:
        res = RuleResult(NOT_EVALUABLE, "no decisive cell with a declared gate")
    cells = list(res.cells)
    for cell, g in cell_gates.items():
        if g["outcome"] != "decisive":
            cells.append({"cell": cell, "outcome": g["outcome"], "gate": g})
    cells.extend(oracle_cells)
    outcome = res.outcome
    preds = resolve(list(part.get("cell_predictions") or ()), spec)
    if preds:
        first = {}
        for r in prep.rows:
            first.setdefault(r["_cell"], r)
        for c in cells:
            row = first.get(c.get("cell"))
            for cp in preds:
                if (
                    row is not None
                    and evaluate_predicate(cp.get("match"), row, fields) is True
                ):
                    c["prediction"] = cp["prediction"]
                    break
    out = {
        **base,
        "status": ACTIVE,
        "outcome": outcome,
        "reason": res.reason,
        "cells": cells,
        "stats": res.stats,
        "notes": prep.notes,
        "params": params,
    }
    if gate_info is not None:
        out["gate"] = _gate_summary(gate_info)
    out["matches_prediction"] = (
        (outcome == out["prediction"]) if outcome in EVALUABLE else None
    )
    return out


def _gate_summary(info):
    """The precision row behind a gating decision (its own ``outcome`` is
    reported as ``gate_outcome``)."""
    if not info:
        return None
    out = {
        k: info.get(k)
        for k in (
            "family",
            "principle",
            "witness",
            "kind",
            "pi",
            "decisive",
            "replacement",
            "predicted_absent_not_reachable_rate",
            "_protocol",
        )
    }
    out["gate_outcome"] = info.get("outcome")
    return out


def _annotate_anchor_predictions(prep, ctx, part, spec):
    for r in prep.rows:
        proto = _proto_for(r, ctx)
        r["_protocol"] = r.get("protocol_id")
        r["_predicted_anchor"] = (
            _anchor_status(proto, r.get("principle")) if proto else None
        )


def evaluate_hypothesis(
    h: Mapping, spec: Mapping, ctx: Context, fields: Fields = None
) -> dict:
    fields = fields or Fields(spec.get("fields"), spec.get("derived"))
    base = {
        "id": h["id"],
        "title": h.get("title"),
        "tier": h.get("tier"),
        "labels": h.get("labels"),
        "prediction": h.get("prediction", SUPPORTED),
    }
    if h.get("tier") == "B" and not h.get("admitted", False):
        return {
            **base,
            "outcome": NOT_RUN,
            "admitted": False,
            "parts": [],
            "reason": "Tier-B item not admitted: the v1 outcome stands",
        }
    parts = [evaluate_part(p, spec, ctx, fields) for p in h.get("parts") or ()]
    decisive = [p for p in parts if p["role"] == DECISIVE and p["outcome"] != REMOVED]
    outcome = combine_outcomes(p["outcome"] for p in decisive)
    out = {
        **base,
        "outcome": outcome,
        "parts": parts,
        "decisive_outcomes": {p["id"]: p["outcome"] for p in decisive},
    }
    out["matches_prediction"] = (
        (outcome == out["prediction"]) if outcome in EVALUABLE else None
    )
    return out


def _uncalibrated_from(h4: Mapping) -> frozenset:
    out = set()
    for part in h4.get("parts") or ():
        for c in part.get("cells") or ():
            if c.get("outcome") == FALSIFIED and c.get("side") == SIDE_ANTI:
                if c.get("estimator_version") and c.get("se_method"):
                    out.add((c["estimator_version"], c["se_method"]))
    return frozenset(out)


def apply_reversion(ctx: Context, uncalibrated) -> Context:
    """A context in which every ABSENT component of an uncalibrated
    (estimator version, SE method) is ``UNDEFINED(SE_NOT_CALIBRATED)`` and
    every EXCLUDED verdict that rested only on such components is
    UNDETERMINED (the reversion rule of the SE calibration)."""
    unc = {tuple(u) for u in uncalibrated}
    if not unc:
        return ctx
    new = copy.copy(ctx)
    new.__dict__.pop("_design_index", None)
    srcs = dict(ctx.sources)
    comps = []
    for r in ctx.sources.get("components", []):
        if (
            r.get("status") == R.ABSENT
            and (r.get("estimator_version"), r.get("se_method")) in unc
        ):
            r = {
                **r,
                "status": R.UNDEFINED,
                "reason": R.SE_NOT_CALIBRATED,
                "flags": [
                    f for f in r.get("flags") or () if f != R.ABSENT_BY_CONCORDANCE
                ],
            }
        comps.append(r)
    srcs["components"] = comps
    verds = []
    for v in ctx.sources.get("verdicts", []):
        if v.get("verdict_value") == "EXCLUDED":
            proto = _proto_for(v, ctx)
            n = _n_anch(proto) or set(v.get("component_status") or {})
            remaining = [
                p
                for p in n
                if (v.get("component_status") or {}).get(p) == R.ABSENT
                and tuple(v.get("component_method", {}).get(p) or ()) not in unc
            ]
            if not remaining:
                v = {
                    **v,
                    "verdict_value": "UNDETERMINED",
                    "verdict": {
                        **(v.get("verdict") or {}),
                        "verdict": "UNDETERMINED",
                        "reverted": True,
                    },
                }
        verds.append(v)
    srcs["verdicts"] = verds
    new.sources = srcs
    new.uncalibrated = frozenset(unc)
    return new


def _usability_from(h0: Mapping) -> Dict[Tuple[str, str], bool]:
    out = {}
    for part in h0.get("parts") or ():
        for c in part.get("cells") or ():
            fam, chk = c.get("family"), c.get("check")
            if fam is not None and chk is not None and "usable" in c:
                out[(str(fam), str(chk))] = bool(c["usable"])
    return out


def evaluate(spec: Mapping, ctx: Context) -> dict:
    """Evaluate every hypothesis of a (validated) hypotheses file. Order:
    the prerequisite M first (its usability map gates the parts that need a
    switch), then the SE calibration against twins (its anti-conservative
    SE methods trigger the reversion rule for the hypotheses flagged
    ``verdict_level_absent``, which are reported both ways), then the rest.
    Results are listed in the file's order, with the tally."""
    spec = validate_spec(spec)
    fields = Fields(spec.get("fields"), spec.get("derived"))
    hyps = list(spec["hypotheses"])
    # the usability map of the prerequisite M is filled below; the caller's
    # context is left as it was
    ctx = copy.copy(ctx)
    ctx.usability = dict(ctx.usability)
    results: Dict[str, dict] = {}
    pre = [h for h in hyps if h.get("role") == "prerequisite"]
    for h in pre:
        results[h["id"]] = evaluate_hypothesis(h, spec, ctx, fields)
        ctx.usability.update(_usability_from(results[h["id"]]))
    se = [h for h in hyps if h.get("role") == "se_calibration"]
    for h in se:
        results[h["id"]] = evaluate_hypothesis(h, spec, ctx, fields)
    unc = (
        frozenset().union(*[_uncalibrated_from(results[h["id"]]) for h in se])
        if se
        else (frozenset())
    )
    rctx = apply_reversion(ctx, unc) if unc else ctx
    for h in hyps:
        if h["id"] in results:
            continue
        if unc and h.get("verdict_level_absent"):
            res = evaluate_hypothesis(h, spec, rctx, fields)
            res_raw = evaluate_hypothesis(h, spec, ctx, fields)
            res["outcome_as_recorded"] = res_raw["outcome"]
            res["reversion"] = sorted(list(u) for u in unc)
            results[h["id"]] = res
        else:
            results[h["id"]] = evaluate_hypothesis(h, spec, ctx, fields)
    ordered = [results[h["id"]] for h in hyps]
    return {
        "version": REPORT_VERSION,
        "spec_version": spec.get("version"),
        "spec_sha256": spec_sha256(spec),
        "split": ctx.split,
        "status": DEVELOPMENT_FLAG if ctx.split == S.DEVELOPMENT else "confirmatory",
        "excluded_task_ids": sorted(ctx.excluded_task_ids),
        "uncalibrated_se_methods": sorted(list(u) for u in unc),
        "usability": {f"{k[0]}:{k[1]}": v for k, v in sorted(ctx.usability.items())},
        "outcomes": {r["id"]: r["outcome"] for r in ordered},
        "hypotheses": ordered,
        "tally": tally(ordered),
        "evaluator_errors": [
            {"hypothesis": r["id"], "part": p["id"], "reason": p["reason"]}
            for r in ordered
            for p in r.get("parts") or ()
            if p.get("evaluator_error")
        ],
    }


def tally(results: Sequence[Mapping]) -> dict:
    """Counts of decisive parts per outcome (removed parts counted as
    REMOVED and listed with their development outcome as the predicted
    outcome), hypothesis outcomes, and the Tier-B items not run."""
    parts = {k: 0 for k in TALLY_KEYS}
    removed = []
    reported = 0
    for h in results:
        for p in h.get("parts") or ():
            if p["role"] != DECISIVE:
                reported += 1
                continue
            parts[p["outcome"]] = parts.get(p["outcome"], 0) + 1
            if p["outcome"] == REMOVED:
                removed.append(
                    {
                        "hypothesis": h["id"],
                        "part": p["id"],
                        "predicted_outcome": p.get("predicted_outcome"),
                        "reason": p.get("reason"),
                    }
                )
    hyp = {}
    for h in results:
        hyp[h["outcome"]] = hyp.get(h["outcome"], 0) + 1
    return {
        "decisive_parts": parts,
        "reported_parts": reported,
        "removed_parts": removed,
        "hypotheses": hyp,
        "not_run": [h["id"] for h in results if h["outcome"] == NOT_RUN],
    }


def parts_table(report: Mapping) -> List[dict]:
    """One flat row per part (hypothesis, part, role, label, rule, outcome,
    prediction, reason) for the tables of the results summary."""
    rows = []
    for h in report.get("hypotheses") or ():
        for p in h.get("parts") or ():
            rows.append(
                {
                    "hypothesis": h["id"],
                    "hypothesis_outcome": h["outcome"],
                    "part": p["id"],
                    "role": p["role"],
                    "label": p.get("label"),
                    "rule": p.get("rule"),
                    "outcome": p["outcome"],
                    "prediction": p.get("prediction"),
                    "reason": p.get("reason"),
                }
            )
    return rows


__all__ = [
    "Context",
    "Counts",
    "DEVELOPMENT_FLAG",
    "FALSIFIED",
    "Fields",
    "INDETERMINATE",
    "NOT_EVALUABLE",
    "NOT_RUN",
    "NOT_TESTABLE_BY_DESIGN",
    "OUTCOMES",
    "PDI_OPTION_NAMES",
    "PROTOCOL_ESTIMATOR_NAMES",
    "Prepared",
    "RAM_FACET_NAMES",
    "REMOVED",
    "REPORTED",
    "RULES",
    "RuleResult",
    "SPEC_PATH",
    "SPEC_SCHEMA",
    "SUPPORTED",
    "SpecError",
    "apply_reversion",
    "build_context",
    "cluster_bootstrap_rate",
    "combine_outcomes",
    "component_rows",
    "count_events",
    "cp_lower",
    "cp_upper",
    "evaluate",
    "evaluate_hypothesis",
    "evaluate_part",
    "evaluate_predicate",
    "hodges_lehmann",
    "jonckheere_terpstra",
    "kappa_interval",
    "load_spec",
    "mann_whitney_pmf",
    "n_needed_zero_events",
    "newcombe_difference",
    "ols_slope_unit",
    "parts_table",
    "pdi_params_for_estimator",
    "PROTOCOL_FILE_GLOB",
    "PROTOCOL_FILE_PREFIX",
    "load_protocol_files",
    "protocol_file_name",
    "protocol_key",
    "protocol_key_of_file",
    "protocol_key_of_name",
    "pooled_within_sd",
    "prepare",
    "ram_facets_for_estimator",
    "ram_facets_for_protocol",
    "render_template",
    "resolve",
    "sign_test",
    "signed_rank_pmf",
    "spearman",
    "spec_sha256",
    "tally",
    "validate_spec",
    "verdict_rows",
    "wilcoxon_signed_rank",
    "wilson_interval",
]
