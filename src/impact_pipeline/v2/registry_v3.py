"""
Applicability registry v3 (``impact-mpc-registry/3``) and the forward-model
admission procedure of MPC-Bench v2.

Real data can never validate an estimator, so an entry for human EEG or fMRI
(substrates ``eeg_like_forward`` / ``bold_like_forward``) comes only from a
forward-model arm with known sources. One procedure serves every principle.

Entry
-----
One entry per (principle, estimator@version, substrate, view regime) with two
flags, one per status direction:

* ``admitted_for_present``: ``yes`` iff FM0, FMa, FMb1, FMd and, for NAS,
  FMb2 are all demonstrated (an intersection-union of the criteria);
  otherwise ``no``;
* ``admitted_for_absent``: ``yes`` iff FM0 and FMabs are demonstrated, or
  ``vacuous`` when, in addition, the ABSENT rate at the null ``pi0`` is
  below 0.1 (ABSENT admissible but practically unreachable); otherwise
  ``no``;
* both ``not_observable`` when the observation does not resolve the
  construct: every run of the view is UNDEFINED with an observation-model
  reason (:data:`NOT_OBSERVABLE_REASONS`, e.g. NAS ``SAMPLING_UNRESOLVED``
  on BOLD).

The evidence layer (:func:`impact_pipeline.evidence_v2.registry_admission`)
licenses each status direction separately: a PRESENT on a regime not
admitted for PRESENT becomes ``UNDEFINED(ESTIMATOR_NOT_VALIDATED:
not_admitted)`` (``:not_observable`` for a not-observable entry), and
likewise for ABSENT; ``vacuous`` admits. A query that matches no entry is not
admitted in either direction. When several entries match, each direction
takes the most restrictive value (``not_observable`` > ``no`` > ``vacuous`` >
``yes``), as the v1 registry lets a not-validated entry win.

Criteria (preregistration v2, section 4.5)
------------------------------------------
Levels in :data:`DEFAULT_ADMISSION_CRITERIA`.

* **FM0** anchor: the view's anchor is valid on its reference block.
* **FMa** specificity: on every declared null class, false PRESENT with a
  one-sided Clopper-Pearson upper bound < 0.07 at level ``0.05 / m``
  (``m`` = the number of null classes unless the arm declares it). With 0
  events this needs 42 (m = 1), 51 (m = 2) or 61 (m = 4) seeds per class.
* **FMb1** dose: Spearman ``rho(c, dose) > 0`` at one-sided ``p < 0.05``
  over the dose sweep (seeds common to every dose level).
* **FMb2** (NAS only): hub lesion minus its size-matched random lesion,
  paired ``Delta c < 0`` in >= 80 % of seeds (three-zone rule).
* **FMd** source concordance: the forward contrast between the dose
  extremes has the sign of the source contrast on the same seeds and at
  least half its size in >= 80 % of seeds (three-zone rule; trivially met
  by the source view itself).
* **FMabs** exclusion safety: false ABSENT among mechanism-on runs with a
  one-sided CP upper bound < 0.02 (>= 149 runs with 0 events), evaluated in
  seed order with curtailment.

"Demonstrated" means the CP bound (or the three-zone rule's SUPPORTED zone,
or the one-sided test) clears its threshold. Runs whose estimator raised
(``ESTIMATOR_ERROR``) are not observations and leave every denominator.

Curtailed sampling. On-runs are evaluated in seed order and the arm stops as
soon as an event makes the demonstration impossible at the planned number of
runs (:class:`Curtailment`). Curtailment only ever stops towards
non-admission: whenever the planned runs are all evaluated, the curtailed
decision equals the full-sample decision, so it cannot raise the admission
error.

Regime keys and matching
------------------------
The twelve regime keys of :data:`REGIME_KEYS` describe a view. The forward
model's own parameters (:data:`PROVENANCE_KEYS`: lead-field family, seed and
conduction width) are recorded in ``regime_keys`` but are not constraints, so
a recording can match an entry by what it can declare; every other key
present in a view becomes a constraint of the entry's ``regime``: its value
when it is constant over the admission runs, the observed ``{min, max}`` when
a number varies (for example the run length of family-A agents), the list of
values otherwise (:func:`regime_constraints`). Matching follows the v1
registry (a scalar matches by equality, a list by membership, ``{min, max}``
as an interval) and also knows the named bounds of
:data:`REGIME_BOUNDS_V3` (``n_sensors_min``, ``fs_min``, ``tr_max``,
``duration_min_s`` ...). A query that omits a constrained key does not
match. There is no extrapolation beyond the tested regime: the paper-2
protocol declares its recording regime (CD-12) and matches it.

Schema ``impact-mpc-registry/2`` registries (the v1 registry) load as
entries whose two flags are both ``yes`` (validated) or both ``no``.

Every entry carries the scope note: admitted on the bench forward-model
class, not on human heads.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from impact_pipeline import evidence as v1
from impact_pipeline.v2 import PRINCIPLES
from impact_pipeline.v2 import reasons as R

REGISTRY_SCHEMA_V3 = "impact-mpc-registry/3"
REGISTRY_SCHEMA_V2 = v1.REGISTRY_SCHEMA

YES, NO, VACUOUS, NOT_OBSERVABLE = "yes", "no", "vacuous", "not_observable"
ADMISSION_VALUES = (YES, NO, VACUOUS, NOT_OBSERVABLE)
ADMITTING = (YES, VACUOUS)
# most restrictive value wins when several entries match
RESTRICTIVENESS = MappingProxyType({YES: 0, VACUOUS: 1, NO: 2, NOT_OBSERVABLE: 3})

CRITERIA = ("FM0", "FMa", "FMb1", "FMb2", "FMd", "FMabs")
PRESENT_CRITERIA = ("FM0", "FMa", "FMb1", "FMd")
ABSENT_CRITERIA = ("FM0", "FMabs")
LESION_PRINCIPLES = ("NAS",)

DEFAULT_ADMISSION_CRITERIA = MappingProxyType({
    "fma_bound": 0.07,
    "fma_level": 0.05,
    "fmabs_bound": 0.02,
    "fmabs_level": 0.05,
    "pi0_vacuous_below": 0.1,
    "dose_alpha": 0.05,
    "three_zone_rate": 0.8,
    "three_zone_level": 0.05,
    "fmd_size_ratio": 0.5,
})

# Reasons that say the observation does not resolve the construct.
NOT_OBSERVABLE_REASONS = (R.SAMPLING_UNRESOLVED, R.NOT_APPLICABLE_OBSERVATION_MODEL)

REGIME_KEYS = (
    "observation_stage",
    "leadfield_family",
    "leadfield_seed",
    "conduction_width",
    "n_sensors",
    "reference",
    "sensor_filter",
    "fs",
    "tr",
    "duration_s",
    "inputs_declared",
    "snr",
)
PROVENANCE_KEYS = ("leadfield_family", "leadfield_seed", "conduction_width")
CONSTRAINT_KEYS = tuple(k for k in REGIME_KEYS if k not in PROVENANCE_KEYS)
# Named bounds: key -> (query key, relation); the v1 bounds plus the v3 ones.
REGIME_BOUNDS_V3 = MappingProxyType({
    **v1.REGIME_BOUNDS,
    "n_sensors_min": ("n_sensors", "min"),
    "n_sensors_max": ("n_sensors", "max"),
    "fs_min": ("fs", "min"),
    "fs_max": ("fs", "max"),
    "tr_min": ("tr", "min"),
    "tr_max": ("tr", "max"),
    "duration_min_s": ("duration_s", "min"),
    "duration_max_s": ("duration_s", "max"),
})
_V1_SPECIAL_KEYS = ("fs_or_tr", "bins")

SCOPE_NOTE = (
    "Admitted on the MPC-Bench forward-model class (whole-brain Hopf model or "
    "family-A agents with known sources; spherical Gaussian lead field; the "
    "stated regime), not on human heads."
)

ZONE_SUPPORTED, ZONE_FALSIFIED = "SUPPORTED", "FALSIFIED"
ZONE_INDETERMINATE, ZONE_NOT_EVALUABLE = "INDETERMINATE", "NOT_EVALUABLE"


class RegistryV3Error(ValueError):
    """A malformed registry v3 entry or admission input."""


# --------------------------------------------------------------------------
# demonstration statistics
# --------------------------------------------------------------------------
def _check_kn(k, n):
    k, n = int(k), int(n)
    if n < 0 or not 0 <= k <= max(n, 0):
        raise RegistryV3Error(f"need 0 <= k <= n, got k={k}, n={n}")
    return k, n


def _check_level(level):
    level = float(level)
    if not 0.0 < level < 1.0:
        raise RegistryV3Error(f"level must lie in (0, 1), got {level}")
    return level


def cp_upper(k, n, level=0.05) -> float:
    """One-sided Clopper-Pearson upper bound at level ``level`` for ``k``
    events in ``n`` runs (1 for ``n = 0``)."""
    from scipy.stats import beta

    k, n = _check_kn(k, n)
    level = _check_level(level)
    if n == 0 or k >= n:
        return 1.0
    return float(beta.ppf(1.0 - level, k + 1, n - k))


def cp_lower(k, n, level=0.05) -> float:
    """One-sided Clopper-Pearson lower bound (0 for ``k = 0``)."""
    from scipy.stats import beta

    k, n = _check_kn(k, n)
    level = _check_level(level)
    if n == 0 or k == 0:
        return 0.0
    return float(beta.ppf(level, k, n - k + 1))


def demonstration(k, n, bound, level=0.05) -> dict:
    """Rate demonstration: ``demonstrated`` iff the one-sided CP upper bound
    of ``k / n`` at ``level`` is below ``bound``."""
    k, n = _check_kn(k, n)
    upper = cp_upper(k, n, level)
    return {"events": k, "n": n, "rate": (k / n) if n else None, "upper": upper,
            "bound": float(bound), "level": float(level),
            "demonstrated": bool(n > 0 and upper < float(bound))}


def max_demonstrable_events(n, bound, level=0.05) -> int:
    """The largest event count ``k`` with ``cp_upper(k, n) < bound`` (-1 if
    not even 0 events can demonstrate the bound with ``n`` runs)."""
    n = int(n)
    if n <= 0:
        return -1
    k = -1
    while k + 1 <= n and cp_upper(k + 1, n, level) < float(bound):
        k += 1
    return k


def min_runs_for_demonstration(bound, level=0.05, events=0, n_max=100000) -> int:
    """The smallest ``n`` with ``cp_upper(events, n) < bound`` (42 / 51 / 61
    for 0.07 at levels 0.05 / 0.025 / 0.0125; 149 for 0.02 at 0.05)."""
    events = int(events)
    n = max(events + 1, 1)
    while n <= n_max:
        if cp_upper(events, n, level) < float(bound):
            return n
        n += 1
    raise RegistryV3Error("no run count up to n_max demonstrates the bound")


class Curtailment:
    """
    Sequential demonstration of a rate bound with curtailment: runs are fed
    in seed order (:meth:`update` with ``event`` True for a false ABSENT);
    the sequence stops as soon as the events exceed
    :func:`max_demonstrable_events` at the planned ``n_planned`` runs (the
    demonstration has become impossible). Stopping only ever leads to
    non-admission.
    """

    def __init__(self, n_planned, bound, level=0.05):
        self.n_planned = int(n_planned)
        self.bound = float(bound)
        self.level = _check_level(level)
        self.k_max = max_demonstrable_events(self.n_planned, self.bound, self.level)
        self.n = 0
        self.events = 0
        self.stop_index = None
        self.stopped = self.k_max < 0

    def update(self, event) -> bool:
        """Feed one run; returns True when the sequence has stopped (the
        run that stops it is counted)."""
        if self.stopped:
            return True
        self.n += 1
        self.events += int(bool(event))
        if self.events > self.k_max:
            self.stopped = True
            self.stop_index = self.n - 1
        return self.stopped

    def result(self) -> dict:
        """The decision so far: not demonstrated once stopped, else the CP
        demonstration over the runs evaluated."""
        upper = cp_upper(self.events, self.n, self.level) if self.n else 1.0
        demonstrated = (not self.stopped and self.n > 0 and upper < self.bound)
        return {"events": self.events, "n": self.n, "n_planned": self.n_planned,
                "k_max": self.k_max, "stopped": bool(self.stopped),
                "stop_index": self.stop_index, "upper": upper, "bound": self.bound,
                "level": self.level, "rate": (self.events / self.n) if self.n else None,
                "demonstrated": bool(demonstrated)}


def curtailed_demonstration(events_in_order: Iterable, n_planned, bound,
                            level=0.05) -> dict:
    """Run a :class:`Curtailment` over a sequence of event flags (seed
    order); the result lists how many runs were evaluated before it
    stopped."""
    cur = Curtailment(n_planned, bound, level)
    for ev in events_in_order:
        if cur.update(ev):
            break
    out = cur.result()
    out["n_available"] = None  # filled by callers that know it
    return out


def three_zone(successes, n, rate=0.8, level=0.05, m=1) -> dict:
    """Three-zone rule for "in >= rate of seeds": SUPPORTED iff the point
    rate is >= ``rate``; FALSIFIED iff the CP upper bound at ``level / m``
    is < ``rate``; else INDETERMINATE (NOT_EVALUABLE without pairs).
    ``pass`` iff SUPPORTED."""
    s, n = _check_kn(successes, n)
    if n == 0:
        return {"successes": 0, "n": 0, "point": None, "upper": None,
                "rate": float(rate), "zone": ZONE_NOT_EVALUABLE, "pass": False}
    point = s / n
    upper = cp_upper(s, n, float(level) / int(m))
    if point >= float(rate):
        zone = ZONE_SUPPORTED
    elif upper < float(rate):
        zone = ZONE_FALSIFIED
    else:
        zone = ZONE_INDETERMINATE
    return {"successes": s, "n": n, "point": point, "upper": upper,
            "rate": float(rate), "zone": zone, "pass": zone == ZONE_SUPPORTED}


def spearman_dose(c: Sequence[float], dose: Sequence[float], alpha=0.05) -> dict:
    """FMb1: Spearman ``rho(c, dose)`` with the one-sided p-value for
    ``rho > 0``; ``pass`` iff ``rho > 0`` and ``p < alpha``. Pairs with a
    non-finite ``c`` are dropped."""
    from scipy.stats import spearmanr

    c = np.asarray(c, dtype=float)
    d = np.asarray(dose, dtype=float)
    if c.shape != d.shape:
        raise RegistryV3Error("c and dose must be paired")
    ok = np.isfinite(c) & np.isfinite(d)
    c, d = c[ok], d[ok]
    out = {"n": int(c.size), "n_levels": int(np.unique(d).size), "rho": None,
           "p": None, "alpha": float(alpha), "pass": False}
    if c.size < 3 or np.unique(d).size < 2 or np.unique(c).size < 2:
        return out
    res = spearmanr(d, c, alternative="greater")
    rho, p = float(res.statistic), float(res.pvalue)
    out.update(rho=rho if math.isfinite(rho) else None,
               p=p if math.isfinite(p) else None)
    out["pass"] = bool(math.isfinite(rho) and math.isfinite(p) and rho > 0
                       and p < float(alpha))
    return out


def lesion_contrast(hub_c: Sequence[float], random_c: Sequence[float], rate=0.8,
                    level=0.05) -> dict:
    """FMb2: paired ``c(hub lesion) - c(matched random lesion) < 0`` in
    >= ``rate`` of seeds (three-zone; non-finite pairs dropped)."""
    h = np.asarray(hub_c, dtype=float)
    r = np.asarray(random_c, dtype=float)
    if h.shape != r.shape:
        raise RegistryV3Error("hub and random lesion values must be paired")
    ok = np.isfinite(h) & np.isfinite(r)
    d = h[ok] - r[ok]
    out = three_zone(int(np.sum(d < 0)), int(d.size), rate, level)
    out["median_delta_c"] = float(np.median(d)) if d.size else None
    return out


def source_concordance(fwd_high, fwd_low, src_high, src_low, ratio=0.5, rate=0.8,
                       level=0.05) -> dict:
    """FMd: per seed, the forward contrast ``d_f = c_f(high) - c_f(low)``
    has the sign of the source contrast ``d_s`` and ``|d_f| >= ratio
    |d_s|``; three-zone over the seeds with all four values finite."""
    a = [np.asarray(v, dtype=float) for v in (fwd_high, fwd_low, src_high, src_low)]
    if len({x.shape for x in a}) != 1:
        raise RegistryV3Error("forward and source contrasts must be paired by seed")
    ok = np.all([np.isfinite(x) for x in a], axis=0)
    d_f = a[0][ok] - a[1][ok]
    d_s = a[2][ok] - a[3][ok]
    hit = (np.sign(d_f) == np.sign(d_s)) & (np.abs(d_f) >= float(ratio) * np.abs(d_s))
    out = three_zone(int(np.sum(hit)), int(hit.size), rate, level)
    out["median_forward_contrast"] = float(np.median(d_f)) if d_f.size else None
    out["median_source_contrast"] = float(np.median(d_s)) if d_s.size else None
    out["ratio"] = float(ratio)
    return out


# --------------------------------------------------------------------------
# admission runs and designs
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AdmissionRun:
    """One scored run of a view: the principle's status, reason and ``c``
    under the view's family protocol (no registry gating), its condition
    label, seed and dose."""

    principle: str
    view: str
    condition: str
    seed: int
    status: str
    reason: Optional[str] = None
    c: Optional[float] = None
    dose: Optional[float] = None

    def __post_init__(self):
        if self.principle not in PRINCIPLES:
            raise RegistryV3Error(f"unknown principle {self.principle!r}")
        if self.status not in R.STATUSES:
            raise RegistryV3Error(f"status must be one of {R.STATUSES}")
        object.__setattr__(self, "seed", int(self.seed))

    @property
    def is_error(self) -> bool:
        return R.is_estimator_error(self.reason)

    @property
    def c_value(self) -> float:
        v = v1._as_float(self.c)
        return v if math.isfinite(v) else float("nan")


@dataclass(frozen=True)
class AdmissionDesign:
    """
    What each criterion reads for one principle on one forward arm:
    ``null_conditions`` (FMa and ``pi0``), ``on_conditions`` (FMabs, in this
    order within a seed), ``dose_conditions`` (``{condition: dose}``, FMb1),
    ``concordance`` (``(high, low)`` conditions of FMd), ``lesion``
    (``(hub, matched random)`` of FMb2; NAS only), ``fma_m`` (the FMa
    Bonferroni divisor; default the number of null classes),
    ``n_planned_on`` (planned on-runs of the curtailment; default the runs
    available) and ``source_view`` (the view whose contrast FMd follows).
    """

    principle: str
    arm: str
    views: Tuple[str, ...]
    null_conditions: Tuple[str, ...]
    on_conditions: Tuple[str, ...]
    dose_conditions: Mapping[str, float]
    concordance: Tuple[str, str]
    lesion: Optional[Tuple[str, str]] = None
    fma_m: Optional[int] = None
    n_planned_on: Optional[int] = None
    source_view: str = "source"

    def __post_init__(self):
        if self.principle not in PRINCIPLES:
            raise RegistryV3Error(f"unknown principle {self.principle!r}")
        if (self.lesion is not None) != (self.principle in LESION_PRINCIPLES):
            raise RegistryV3Error("FMb2 (lesion pairs) applies to NAS only, and NAS "
                                  "needs it")
        if not self.null_conditions or not self.on_conditions:
            raise RegistryV3Error("an arm needs null and on conditions")
        if len(self.dose_conditions) < 2:
            raise RegistryV3Error("FMb1 needs at least two dose levels")
        if self.fma_m is not None and int(self.fma_m) < 1:
            raise RegistryV3Error("fma_m must be >= 1")
        object.__setattr__(self, "views", tuple(self.views))
        object.__setattr__(self, "null_conditions", tuple(self.null_conditions))
        object.__setattr__(self, "on_conditions", tuple(self.on_conditions))
        object.__setattr__(self, "dose_conditions", MappingProxyType(
            {str(k): float(v) for k, v in dict(self.dose_conditions).items()}))
        object.__setattr__(self, "concordance", tuple(self.concordance))
        if self.lesion is not None:
            object.__setattr__(self, "lesion", tuple(self.lesion))


def _valid_runs(runs):
    ok = [r for r in runs if not r.is_error]
    return ok, len(runs) - len(ok)


def _by_condition(runs) -> Dict[str, Dict[int, AdmissionRun]]:
    out: Dict[str, Dict[int, AdmissionRun]] = {}
    for r in runs:
        seeds = out.setdefault(r.condition, {})
        if r.seed in seeds:
            raise RegistryV3Error(
                f"two runs of {r.principle} on view {r.view}, condition "
                f"{r.condition}, seed {r.seed}")
        seeds[r.seed] = r
    return out


def _anchor_criterion(anchor) -> dict:
    if anchor is None:
        return {"pass": False, "valid": None, "why": "no anchor for the view"}
    if isinstance(anchor, (bool, np.bool_)):
        return {"pass": bool(anchor), "valid": bool(anchor)}
    if not isinstance(anchor, Mapping) or "valid" not in anchor:
        raise RegistryV3Error("an anchor is a bool or a mapping with 'valid'")
    out = {k: v for k, v in dict(anchor).items() if k != "pass"}
    out["valid"] = bool(anchor["valid"])
    out["pass"] = bool(anchor["valid"])
    return out


def is_not_observable(runs: Sequence[AdmissionRun]) -> bool:
    """True iff there is at least one run and every (non-error) run is
    UNDEFINED with an observation-model reason."""
    ok, _ = _valid_runs(list(runs))
    return bool(ok) and all(
        r.status == R.UNDEFINED and r.reason is not None
        and R.parse_reason(r.reason)[0] in NOT_OBSERVABLE_REASONS
        for r in ok)


@dataclass(frozen=True)
class AdmissionResult:
    """The admission of one (principle, view): the two flags, every
    criterion's record, the failing criteria and the counts."""

    principle: str
    view: str
    admitted_for_present: str
    admitted_for_absent: str
    criteria: Mapping
    failing: Tuple[str, ...]
    pi0: Optional[float]
    n_runs: int
    n_errors: int

    def to_dict(self) -> dict:
        return {"principle": self.principle, "view": self.view,
                "admitted_for_present": self.admitted_for_present,
                "admitted_for_absent": self.admitted_for_absent,
                "criteria": _jsonable(self.criteria), "failing": list(self.failing),
                "pi0": self.pi0, "n_runs": self.n_runs, "n_errors": self.n_errors}


def evaluate_view(design: AdmissionDesign, view: str, runs: Sequence[AdmissionRun], *,
                  stage: str, anchor=None,
                  source_runs: Optional[Sequence[AdmissionRun]] = None,
                  criteria: Optional[Mapping] = None) -> AdmissionResult:
    """
    Apply FM0, FMa, FMb1, FMb2 (NAS), FMd and FMabs to the runs of one
    (principle, view) of an arm and derive the two admission flags (module
    docstring). ``stage`` is the view's observation stage (FMd is trivially
    met on ``source``); ``source_runs`` are the same principle's runs on the
    arm's source view (FMd); ``anchor`` is the view's anchor record (a bool
    or a mapping with ``valid``).
    """
    crit = dict(DEFAULT_ADMISSION_CRITERIA)
    crit.update(dict(criteria or {}))
    p = design.principle
    runs = [r for r in runs if r.principle == p and r.view == view]
    valid, n_err = _valid_runs(runs)
    by_c = _by_condition(valid)
    rec: Dict[str, dict] = {}

    rec["FM0"] = _anchor_criterion(anchor)

    # FMa: every declared null class demonstrated at level / m
    m = int(design.fma_m) if design.fma_m is not None else len(design.null_conditions)
    level_a = float(crit["fma_level"]) / m
    classes = {}
    for cond in design.null_conditions:
        rs = list(by_c.get(cond, {}).values())
        k = sum(r.status == R.PRESENT for r in rs)
        classes[cond] = demonstration(k, len(rs), crit["fma_bound"], level_a)
    rec["FMa"] = {"m": m, "level": level_a, "bound": float(crit["fma_bound"]),
                  "classes": classes,
                  "pass": bool(classes) and all(v["demonstrated"]
                                                for v in classes.values())}

    # FMb1: Spearman over the seeds common to every dose level
    dose_conds = list(design.dose_conditions)
    common = None
    for cond in dose_conds:
        s = set(by_c.get(cond, {}))
        common = s if common is None else common & s
    common = sorted(common or ())
    cs, ds = [], []
    for cond in dose_conds:
        for seed in common:
            cs.append(by_c[cond][seed].c_value)
            ds.append(design.dose_conditions[cond])
    fmb1 = spearman_dose(cs, ds, crit["dose_alpha"])
    fmb1["seeds"] = len(common)
    rec["FMb1"] = fmb1

    # FMb2 (NAS): hub vs size-matched random lesion
    if design.lesion is not None:
        hub, rnd = design.lesion
        seeds = sorted(set(by_c.get(hub, {})) & set(by_c.get(rnd, {})))
        rec["FMb2"] = lesion_contrast(
            [by_c[hub][s].c_value for s in seeds],
            [by_c[rnd][s].c_value for s in seeds],
            crit["three_zone_rate"], crit["three_zone_level"])

    # FMd: forward contrast follows the source contrast
    hi, lo = design.concordance
    if stage == "source":
        rec["FMd"] = {"pass": True, "trivial": True,
                      "why": "the source view is its own source contrast"}
    else:
        src = [r for r in (source_runs or []) if r.principle == p]
        src_valid, _ = _valid_runs(src)
        sb = _by_condition(src_valid)
        seeds = sorted(set(by_c.get(hi, {})) & set(by_c.get(lo, {}))
                       & set(sb.get(hi, {})) & set(sb.get(lo, {})))
        rec["FMd"] = source_concordance(
            [by_c[hi][s].c_value for s in seeds], [by_c[lo][s].c_value for s in seeds],
            [sb[hi][s].c_value for s in seeds], [sb[lo][s].c_value for s in seeds],
            crit["fmd_size_ratio"], crit["three_zone_rate"], crit["three_zone_level"])

    # FMabs: false ABSENT among on-runs, seed order, curtailed
    order = {c: i for i, c in enumerate(design.on_conditions)}
    on_runs = sorted((r for c in design.on_conditions
                      for r in by_c.get(c, {}).values()),
                     key=lambda r: (r.seed, order[r.condition]))
    n_planned = (int(design.n_planned_on) if design.n_planned_on is not None
                 else len(on_runs))
    fmabs = curtailed_demonstration((r.status == R.ABSENT for r in on_runs),
                                    max(n_planned, len(on_runs)),
                                    crit["fmabs_bound"], crit["fmabs_level"])
    fmabs["n_available"] = len(on_runs)
    fmabs["pass"] = bool(fmabs["demonstrated"])
    rec["FMabs"] = fmabs

    # pi0: ABSENT rate at the null
    null_runs = [r for c in design.null_conditions for r in by_c.get(c, {}).values()]
    pi0 = (sum(r.status == R.ABSENT for r in null_runs) / len(null_runs)
           if null_runs else None)
    rec["FMabs"]["pi0"] = pi0

    if is_not_observable(runs):
        present = absent = NOT_OBSERVABLE
    else:
        need = PRESENT_CRITERIA + (("FMb2",) if design.lesion is not None else ())
        present = YES if all(rec[k]["pass"] for k in need) else NO
        if all(rec[k]["pass"] for k in ABSENT_CRITERIA):
            absent = VACUOUS if (pi0 is not None and pi0 < float(
                crit["pi0_vacuous_below"])) else YES
        else:
            absent = NO
    failing = tuple(k for k in CRITERIA if k in rec and not rec[k]["pass"])
    return AdmissionResult(p, view, present, absent, rec, failing, pi0, len(runs),
                           n_err)


# --------------------------------------------------------------------------
# regime constraints and matching
# --------------------------------------------------------------------------
def _num(v):
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(
        v, (bool, np.bool_))


def regime_constraints(regimes: Sequence[Mapping]) -> Tuple[dict, dict]:
    """
    ``(constraints, regime_keys)`` of an entry from the regime keys of its
    admission runs. ``regime_keys`` summarises every key (value, observed
    ``{min, max}`` of a varying number, or sorted list of varying values);
    ``constraints`` keeps the non-provenance keys that are not None.
    """
    regimes = [dict(r) for r in regimes]
    if not regimes:
        raise RegistryV3Error("no regimes to summarise")
    unknown = sorted({k for r in regimes for k in r} - set(REGIME_KEYS))
    if unknown:
        raise RegistryV3Error(f"unknown regime keys {unknown}")
    summary = {}
    for key in REGIME_KEYS:
        vals = [r.get(key) for r in regimes]
        if all(v is None for v in vals):
            summary[key] = None
            continue
        if any(v is None for v in vals):
            raise RegistryV3Error(f"regime key {key!r} is missing on some runs")
        uniq = []
        for v in vals:
            if not any(_same_value(v, u) for u in uniq):
                uniq.append(v)
        if len(uniq) == 1:
            summary[key] = uniq[0]
        elif all(_num(v) for v in uniq):
            summary[key] = {"min": float(min(uniq)), "max": float(max(uniq))}
        else:
            summary[key] = sorted(str(v) for v in uniq)
    constraints = {k: v for k, v in summary.items()
                   if v is not None and k not in PROVENANCE_KEYS}
    return constraints, summary


def _same_value(a, b) -> bool:
    if _num(a) and _num(b):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=0.0)
    return str(a) == str(b)


def regime_failure(constraints: Mapping, regime: Mapping) -> Optional[str]:
    """The first constraint key that ``regime`` does not meet (None when all
    are met): named bounds of :data:`REGIME_BOUNDS_V3`, the v1 special
    ``fs_or_tr``, else equality / membership / ``{min, max}`` as in v1. A
    query that omits a constrained key fails on that key."""
    regime = dict(regime or {})
    for key, spec in dict(constraints or {}).items():
        if key in REGIME_BOUNDS_V3:
            qkey, rel = REGIME_BOUNDS_V3[key]
            x = v1._as_float(regime.get(qkey))
            ok = math.isfinite(x) and (x >= float(spec) if rel == "min"
                                       else x <= float(spec))
        elif key == "fs_or_tr":
            qkey = "fs_or_tr" if "fs_or_tr" in regime else "tr"
            ok = qkey in regime and v1._regime_value_matches(spec, regime[qkey])
        else:
            ok = (key in regime and regime[key] is not None
                  and v1._regime_value_matches(spec, regime[key]))
        if not ok:
            return str(key)
    return None


def _check_constraints(regime, legacy=False) -> dict:
    if not isinstance(regime, Mapping):
        raise RegistryV3Error("entry 'regime' must be an object")
    allowed = set(REGIME_KEYS) | set(REGIME_BOUNDS_V3) | set(_V1_SPECIAL_KEYS)
    unknown = sorted(set(regime) - allowed)
    if unknown and not legacy:  # a v1 entry may constrain any key it names
        raise RegistryV3Error(f"entry regime has unknown keys {unknown}")
    for key in REGIME_BOUNDS_V3:
        if key in regime:
            v = v1._as_float(regime[key])
            if not (math.isfinite(v) and v >= 0):
                raise RegistryV3Error(f"regime {key} must be a finite number >= 0")
    return dict(regime)


# Entry fields match as in the v1 registry: ``*``, a list of alternatives or
# a case-insensitive glob.
_match_field = v1._match_field


# --------------------------------------------------------------------------
# entries and the registry
# --------------------------------------------------------------------------
def _jsonable(obj):
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj


def _canonical(payload) -> str:
    return json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


@dataclass(frozen=True)
class AdmissionEntry:
    """One registry v3 entry (module docstring)."""

    principle: str
    estimator: object
    version: object
    substrate: object
    admitted_for_present: str
    admitted_for_absent: str
    grain: object = "*"
    view: Optional[str] = None
    arm: Optional[str] = None
    regime: Mapping = field(default_factory=dict)
    regime_keys: Mapping = field(default_factory=dict)
    criteria: Mapping = field(default_factory=dict)
    failing: Tuple[str, ...] = ()
    evidence: Mapping = field(default_factory=dict)
    scope_note: str = SCOPE_NOTE
    note: Optional[str] = None
    legacy: bool = False

    @property
    def entry_id(self) -> str:
        return (f"{self.principle}:{self.estimator}@{self.version}:{self.substrate}:"
                f"{self.view or '*'}")

    def to_dict(self) -> dict:
        return {
            "principle": self.principle,
            "estimator": self.estimator,
            "version": self.version,
            "substrate": self.substrate,
            "grain": self.grain,
            "arm": self.arm,
            "view": self.view,
            "regime": _jsonable(dict(self.regime)),
            "regime_keys": _jsonable(dict(self.regime_keys)),
            "admitted_for_present": self.admitted_for_present,
            "admitted_for_absent": self.admitted_for_absent,
            "criteria": _jsonable(dict(self.criteria)),
            "failing": list(self.failing),
            "evidence": _jsonable(dict(self.evidence)),
            "scope_note": self.scope_note,
            "note": self.note,
            "legacy": bool(self.legacy),
        }


_ENTRY_KEYS = ("principle", "estimator", "version", "substrate", "grain", "arm", "view",
               "regime", "regime_keys", "admitted_for_present", "admitted_for_absent",
               "criteria", "failing", "evidence", "scope_note", "note", "legacy")


def _pi0_of(criteria: Mapping):
    fmabs = criteria.get("FMabs") or {}
    v = fmabs.get("pi0")
    return None if v is None else float(v)


def check_entry(raw: Mapping, criteria: Optional[Mapping] = None) -> AdmissionEntry:
    """
    Validate one ``impact-mpc-registry/3`` entry and build it. Every entry:
    a known principle, values of :data:`ADMISSION_VALUES`, known regime
    constraint keys, ``failing`` equal to the criteria whose ``pass`` is
    false. An admitting value (``yes``, ``vacuous``) also needs a pinned
    version, an explicit forward or bench substrate (not ``eeg`` / ``fmri``
    / ``bold``), ``evidence.run_id`` and the criteria behind it: FM0, FMa,
    FMb1, FMd (and FMb2 for NAS) passed for PRESENT; FM0 and FMabs passed
    for ABSENT, with ``vacuous`` iff ``pi0 < pi0_vacuous_below``.
    """
    crit = dict(DEFAULT_ADMISSION_CRITERIA)
    crit.update(dict(criteria or {}))
    if not isinstance(raw, Mapping):
        raise RegistryV3Error("registry entries must be JSON objects")
    unknown = sorted(set(raw) - set(_ENTRY_KEYS))
    if unknown:
        raise RegistryV3Error(f"registry entry has unknown keys {unknown}")
    for key in ("principle", "estimator", "version", "substrate",
                "admitted_for_present", "admitted_for_absent"):
        if raw.get(key) is None:
            raise RegistryV3Error(f"registry entry is missing {key!r}")
    p = str(raw["principle"]).upper()
    if p not in PRINCIPLES:
        raise RegistryV3Error(f"unknown principle {p!r}")
    present, absent = str(raw["admitted_for_present"]), str(raw["admitted_for_absent"])
    for v in (present, absent):
        if v not in ADMISSION_VALUES:
            raise RegistryV3Error(f"admission values must be one of {ADMISSION_VALUES}")
    if present == VACUOUS:
        raise RegistryV3Error("'vacuous' qualifies the ABSENT direction only")
    if (present == NOT_OBSERVABLE) != (absent == NOT_OBSERVABLE):
        raise RegistryV3Error("not_observable applies to both directions or neither")
    legacy = bool(raw.get("legacy", False))
    regime = _check_constraints(raw.get("regime") or {}, legacy)
    regime_keys = dict(raw.get("regime_keys") or {})
    bad = sorted(set(regime_keys) - set(REGIME_KEYS))
    if bad:
        raise RegistryV3Error(f"regime_keys has unknown keys {bad}")
    rec = dict(raw.get("criteria") or {})
    for name, c in rec.items():
        if name not in CRITERIA:
            raise RegistryV3Error(f"unknown criterion {name!r}")
        if not isinstance(c, Mapping) or not isinstance(c.get("pass"), bool):
            raise RegistryV3Error(f"criterion {name} needs a boolean 'pass'")
    failing = tuple(raw.get("failing") or ())
    want = tuple(k for k in CRITERIA if k in rec and not rec[k]["pass"])
    if failing != want:
        raise RegistryV3Error(f"'failing' must list the failed criteria {list(want)}")
    evidence = dict(raw.get("evidence") or {})
    version, substrate = raw["version"], raw["substrate"]
    if not legacy and (present in ADMITTING or absent in ADMITTING):
        problems = []
        if (isinstance(version, (list, tuple))
                or any(ch in str(version) for ch in "*?[")):
            problems.append("a pinned 'version' is required")
        subs = substrate if isinstance(substrate, (list, tuple)) else [substrate]
        if any(str(s) == "*" for s in subs):
            problems.append("an explicit 'substrate' is required")
        if any(str(s).lower() in v1.EMPIRICAL_SUBSTRATES for s in subs):
            problems.append("empirical substrates are admitted only through their "
                            "forward-modelled counterparts")
        run_id = evidence.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            problems.append("evidence.run_id is required")
        if present == YES:
            need = PRESENT_CRITERIA + (("FMb2",) if p in LESION_PRINCIPLES else ())
            miss = [k for k in need if not (rec.get(k) or {}).get("pass")]
            if miss:
                problems.append(f"PRESENT admission without passed {miss}")
        if absent in ADMITTING:
            miss = [k for k in ABSENT_CRITERIA if not (rec.get(k) or {}).get("pass")]
            if miss:
                problems.append(f"ABSENT admission without passed {miss}")
            pi0 = _pi0_of(rec)
            vac = pi0 is not None and pi0 < float(crit["pi0_vacuous_below"])
            if (absent == VACUOUS) != vac:
                problems.append(f"absent {absent!r} contradicts pi0 = {pi0}")
        if problems:
            raise RegistryV3Error(
                f"registry entry for {p} {raw['estimator']!r}@{version} on "
                f"{substrate!r}: {'; '.join(problems)}")
    return AdmissionEntry(
        principle=p, estimator=raw["estimator"], version=version, substrate=substrate,
        admitted_for_present=present, admitted_for_absent=absent,
        grain=raw.get("grain", "*") if raw.get("grain") is not None else "*",
        view=raw.get("view"), arm=raw.get("arm"), regime=MappingProxyType(regime),
        regime_keys=MappingProxyType(regime_keys), criteria=MappingProxyType(rec),
        failing=failing, evidence=MappingProxyType(evidence),
        scope_note=str(raw.get("scope_note") or SCOPE_NOTE), note=raw.get("note"),
        legacy=legacy)


def entry_from_v2(e: "v1.RegistryEntry") -> AdmissionEntry:
    """A v1 (``impact-mpc-registry/2``) entry as a v3 entry: both flags
    ``yes`` when validated, both ``no`` otherwise; its regime constraints
    keep their v1 meaning."""
    value = YES if e.validated else NO
    return AdmissionEntry(
        principle=e.principle, estimator=e.estimator, version=e.version,
        substrate=e.substrate, admitted_for_present=value, admitted_for_absent=value,
        grain=e.grain, regime=MappingProxyType(dict(e.regime)),
        evidence=MappingProxyType(dict(e.evidence)),
        criteria=MappingProxyType({}), failing=(),
        scope_note="v1 registry entry (impact-mpc-registry/2), one decision for both "
                   "status directions",
        note=e.note, legacy=True)


class RegistryV3:
    """
    The applicability registry v3. :meth:`admission` answers the evidence
    layer v2 (``{"present": .., "absent": ..}`` per status direction plus
    the matched entries); :meth:`is_validated` gives the v1-style single
    decision (both directions admitted). JSON layout::

        {"schema": "impact-mpc-registry/3", "version": "...",
         "criteria": {...DEFAULT_ADMISSION_CRITERIA...},
         "scope_note": "...",
         "entries": [{"principle": "NAS",
                      "estimator": "compute_NAS:*", "version": "nas-v3-2026.10",
                      "substrate": "eeg_like_forward", "arm": "hopf",
                      "view": "eeg64", "regime": {...}, "regime_keys": {...},
                      "admitted_for_present": "no",
                      "admitted_for_absent": "vacuous",
                      "criteria": {"FM0": {"pass": true, ...}, ...},
                      "failing": ["FMa"],
                      "evidence": {"run_id": "...", ...}}]}
    """

    def __init__(self, entries=(), *, version=None, source=None, criteria=None,
                 provenance=None):
        crit = dict(DEFAULT_ADMISSION_CRITERIA)
        unknown = sorted(set(criteria or {}) - set(crit))
        if unknown:
            raise RegistryV3Error(f"registry criteria has unknown keys {unknown}")
        crit.update(dict(criteria or {}))
        self.criteria = MappingProxyType(crit)
        self.entries = tuple(
            e if isinstance(e, AdmissionEntry) else check_entry(e, crit)
            for e in entries)
        self.version = version
        self.source = source
        self.provenance = dict(provenance or {})

    # -- lookup ------------------------------------------------------------
    def matching(self, principle, estimator, substrate=None, grain=None,
                 **regime) -> Tuple[List[AdmissionEntry], Optional[str]]:
        """The entries that match, and when none does the reason
        (``ESTIMATOR_UNDECLARED``, ``NO_ENTRY`` or ``REGIME_MISMATCH:<key>``)."""
        if estimator is None:
            return [], "ESTIMATOR_UNDECLARED"
        name, version = v1.split_estimator(estimator)
        p = str(principle).upper()
        subs = [substrate]
        if substrate is not None:
            subs += list(v1.EMPIRICAL_SUBSTRATES.get(str(substrate).lower(), ()))
        cands = [
            e for e in self.entries
            if e.principle == p
            and _match_field(e.estimator, name)
            and (str(e.version) == "*" or _match_field(e.version, version))
            and any(_match_field(e.substrate, s) for s in subs)
            and _match_field(e.grain, grain)
        ]
        hits, failures = [], []
        for e in cands:
            key = (v1._regime_failure(e.regime, regime) if e.legacy
                   else regime_failure(e.regime, regime))
            if key is None:
                hits.append(e)
            else:
                failures.append(key)
        if hits:
            return hits, None
        if failures:
            return [], f"REGIME_MISMATCH:{failures[0]}"
        return [], "NO_ENTRY"

    def admission(self, principle, estimator, substrate=None, grain=None,
                  **regime) -> dict:
        """
        ``{"present": value, "absent": value, "entries": [ids], "why": ..}``
        for a component (values of :data:`ADMISSION_VALUES`). No matching
        entry: ``no`` for both directions; several: the most restrictive
        value per direction.
        """
        hits, why = self.matching(principle, estimator, substrate, grain, **regime)
        if not hits:
            return {"present": NO, "absent": NO, "entries": [], "why": why}
        present = max((e.admitted_for_present for e in hits), key=RESTRICTIVENESS.get)
        absent = max((e.admitted_for_absent for e in hits), key=RESTRICTIVENESS.get)
        return {"present": present, "absent": absent,
                "entries": [e.entry_id for e in hits], "why": None}

    def is_validated(self, principle, estimator, substrate=None, grain=None, **regime):
        """v1-style single decision: ``(True, None)`` iff both directions are
        admitted, else ``(False, reason)``."""
        out = self.admission(principle, estimator, substrate, grain, **regime)
        if out["present"] in ADMITTING and out["absent"] in ADMITTING:
            return True, None
        if out["why"]:
            return False, out["why"]
        return False, f"NOT_ADMITTED:present={out['present']},absent={out['absent']}"

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "schema": REGISTRY_SCHEMA_V3,
            "version": self.version,
            "criteria": dict(self.criteria),
            "scope_note": SCOPE_NOTE,
            "provenance": _jsonable(self.provenance),
            "entries": [e.to_dict() for e in self.entries],
        }

    def hash(self) -> str:
        """SHA-256 of the canonical JSON of :meth:`to_dict`."""
        return hashlib.sha256(_canonical(self.to_dict()).encode("utf-8")).hexdigest()

    def to_json(self, path=None, indent=2) -> str:
        text = json.dumps(_jsonable(self.to_dict()), indent=indent, sort_keys=True,
                          allow_nan=False) + "\n"
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    @classmethod
    def from_dict(cls, payload, source=None) -> "RegistryV3":
        """A ``/3`` payload, or a ``/2`` payload (or v1 registry object)
        converted entry by entry (:func:`entry_from_v2`)."""
        if isinstance(payload, v1.ApplicabilityRegistry):
            return cls([entry_from_v2(e) for e in payload.entries],
                       version=payload.version, source=source or payload.source)
        if not isinstance(payload, Mapping):
            raise RegistryV3Error("a registry is a JSON object")
        schema = payload.get("schema")
        if schema == REGISTRY_SCHEMA_V2:
            reg = v1.ApplicabilityRegistry.from_dict(payload, source=source)
            return cls([entry_from_v2(e) for e in reg.entries], version=reg.version,
                       source=source)
        if schema != REGISTRY_SCHEMA_V3:
            raise RegistryV3Error(f"unsupported registry schema {schema!r}")
        unknown = sorted(set(payload) - {"schema", "version", "criteria", "scope_note",
                                         "provenance", "entries", "source"})
        if unknown:
            raise RegistryV3Error(f"registry has unknown keys {unknown}")
        return cls(payload.get("entries") or [], version=payload.get("version"),
                   source=source, criteria=payload.get("criteria"),
                   provenance=payload.get("provenance"))

    @classmethod
    def from_json(cls, path) -> "RegistryV3":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f), source=str(path))


def resolve_registry_v3(registry) -> Optional[RegistryV3]:
    """``None`` | :class:`RegistryV3` | v1 registry | JSON path | dict."""
    if registry is None or isinstance(registry, RegistryV3):
        return registry
    if isinstance(registry, v1.ApplicabilityRegistry):
        return RegistryV3.from_dict(registry)
    if isinstance(registry, (str, os.PathLike)):
        return RegistryV3.from_json(os.fspath(registry))
    if isinstance(registry, Mapping):
        return RegistryV3.from_dict(registry)
    raise TypeError("registry must be None, a RegistryV3, a v1 registry, a JSON path "
                    "or a registry dict")


def build_entry(result: AdmissionResult, *, estimator: str, version: str,
                substrate: str, regimes: Sequence[Mapping], run_id: str,
                arm: Optional[str] = None, grain="*",
                evidence: Optional[Mapping] = None,
                note: Optional[str] = None,
                criteria: Optional[Mapping] = None) -> AdmissionEntry:
    """The registry entry of an :class:`AdmissionResult`: the regime
    constraints from the runs' regime keys (:func:`regime_constraints`) and
    the evidence (``run_id``, counts, ``pi0``), validated by
    :func:`check_entry`."""
    constraints, summary = regime_constraints(regimes)
    ev = {"run_id": str(run_id), "n_runs": result.n_runs, "n_errors": result.n_errors,
          "pi0": result.pi0}
    ev.update(dict(evidence or {}))
    raw = {
        "principle": result.principle, "estimator": estimator, "version": version,
        "substrate": substrate, "grain": grain, "arm": arm, "view": result.view,
        "regime": constraints, "regime_keys": summary,
        "admitted_for_present": result.admitted_for_present,
        "admitted_for_absent": result.admitted_for_absent,
        "criteria": _jsonable(dict(result.criteria)), "failing": list(result.failing),
        "evidence": _jsonable(ev), "scope_note": SCOPE_NOTE, "note": note,
    }
    return check_entry(raw, criteria)


__all__ = [
    "ABSENT_CRITERIA",
    "ADMISSION_VALUES",
    "ADMITTING",
    "AdmissionDesign",
    "AdmissionEntry",
    "AdmissionResult",
    "AdmissionRun",
    "CONSTRAINT_KEYS",
    "CRITERIA",
    "Curtailment",
    "DEFAULT_ADMISSION_CRITERIA",
    "LESION_PRINCIPLES",
    "NOT_OBSERVABLE_REASONS",
    "PRESENT_CRITERIA",
    "PROVENANCE_KEYS",
    "REGIME_BOUNDS_V3",
    "REGIME_KEYS",
    "REGISTRY_SCHEMA_V2",
    "REGISTRY_SCHEMA_V3",
    "RegistryV3",
    "RegistryV3Error",
    "SCOPE_NOTE",
    "build_entry",
    "check_entry",
    "cp_lower",
    "cp_upper",
    "curtailed_demonstration",
    "demonstration",
    "entry_from_v2",
    "evaluate_view",
    "is_not_observable",
    "lesion_contrast",
    "max_demonstrable_events",
    "min_runs_for_demonstration",
    "regime_constraints",
    "regime_failure",
    "resolve_registry_v3",
    "source_concordance",
    "spearman_dose",
    "three_zone",
]
