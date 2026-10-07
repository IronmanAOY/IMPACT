"""
RAM-PE v3 (``ram-v3-2026.10``): the plasticity channel of RAM, measured as a
signed, cross-validated readout of prediction-error-driven pattern change.

**Construct (RAM-PE).** Across consecutive trials of a two-option choice task,
the change of the bearer's stimulus-evoked response pattern carries the
intervening outcome's prediction error signed by the chosen option, beyond
what the unaligned trial sequence explains. It is the plasticity channel of
RAM only. Goal alignment, feedback magnitude and response speed have no
referent in the bench generator and are declared not applicable
(:data:`FACETS_NOT_APPLICABLE`); the v1 product composite and its speed term
are gone, and the evoked magnitude and FIR latency are descriptors without a
null. RAM-PE does not distinguish contingent from non-contingent feedback (an
agent that updates on scrambled feedback is RAM-PE-on), and the persistence
of the update beyond one trial is not tested here: an outcome-locked
transient echo can produce the observable. The statistic is a correlation
(first order in the effect), so the absence margin keeps its nominal meaning.
The name RAM-PE (:data:`CONSTRUCT`) is used in every record.

**Algorithm** (all lengths declared in :class:`RAMParams`):

1. Nodes of the bearer are z-scored over the run. The response pattern of
   stimulus ``k`` is the window mean ``r_k`` over ``[s_k + L, s_k + L + W)``
   (``W = 0.4 s``, ``L = 0``); stimuli whose window leaves the run are
   dropped. The window must end before the logged response in at least 95 %
   of trials (otherwise ``NOT_DEFINED:window_includes_response``).
2. The frozen maximum-likelihood Rescorla-Wagner fit of v1
   (``mpc_metrics._fit_rescorla_wagner``, unchanged) on the chronological
   choice/outcome log gives the prediction error ``delta_j`` of every logged
   trial. The target of stimulus ``k`` is ``y_k = s_k delta_j`` for the one
   outcome ``j`` logged in ``[sigma_k, sigma_{k+1})``, with ``s_k = +1`` for
   the first option label in sorted order and ``-1`` otherwise (the statistic
   is invariant to this convention).
3. Update pairs at lag ``m``: ``d_k = r_{k+m} - r_k`` with target ``y_k``
   (Tier A uses ``m = 1``; the functions take any lag so that a persistence
   readout can reuse them). Fewer than 30 updates, or fewer than two options
   chosen, is ``INSUFFICIENT_UPDATES``.
4. Readout ``m``: ridge regression ``d -> y`` with 5 contiguous outer folds;
   the training rows of a fold exclude the fold and every row within
   ``purge = m`` trials of it (neighbouring updates share a response
   pattern); features are standardised on the training rows; ``lambda`` in
   ``{0.1, 1, 10, 100, 1000}`` is chosen by an inner purged 4-fold contiguous
   CV on the training rows (largest inner out-of-fold Pearson r, ties to the
   smallest ``lambda``). ``m`` is the Pearson r of the pooled out-of-fold
   predictions with ``y``.
5. Null (family ``trial_circular_shift``, exact and deterministic): every
   cyclic shift ``h`` in ``{5, ..., n - 5}`` of ``y`` against the pattern
   changes, each scored by the identical CV. ``nu`` and ``s`` are the mean and
   SD (ddof 1) over all shifts; ``n_null = n - 9``.
6. SE: ``shift_null_sd`` (default): ``se = s`` with ``se_df = n_null - 1``;
   fallback ``jackknife_trials_10``: delete-a-group jackknife over 10
   contiguous blocks of update rows, ``se_df = 9``. The jackknife is always
   computed and reported. There is no below-null guard (the status rule's
   ``NULL_MODEL_VIOLATED`` replaces it).

The v1 family-C carrier recording is declared not applicable before any run
(``NOT_APPLICABLE_OBSERVATION_MODEL``): the pre-response window (0.6 s) is
shorter than one carrier period (0.87 s), and the plasticity signal lives in
the amplitude envelope, which is not recoverable from Re z. The readout is
then reported descriptively only.

**Output.** :func:`compute_ram_v3` returns a dict with ``estimate`` (``m``),
``value`` (excess ``m - nu``), the null moments and family, ``se``,
``se_df``, ``se_method``, ``defined``, ``reason`` (a reason of the central
vocabulary :mod:`~impact_pipeline.v2.reasons` when undefined) and
``details``; :func:`component_fields` maps it onto the estimator fields of a
``mpc-bench-result/3`` component. The status is decided by the status rule,
never here.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from impact_pipeline import mpc_metrics as mm
from impact_pipeline.v2 import numerics as NUM
from impact_pipeline.v2 import reasons as R

PRINCIPLE = "RAM"
CONSTRUCT = "RAM-PE"
ESTIMATOR_VERSION = "ram-v3-2026.10"
MODE = "pe_readout"
ESTIMATOR_ID = f"compute_RAM:{MODE}@{ESTIMATOR_VERSION}"
NULL_FAMILY = "trial_circular_shift"
SE_METHOD_SHIFT_NULL = "shift_null_sd"
SE_METHOD_JACKKNIFE = "jackknife_trials_10"
SE_METHODS = (SE_METHOD_SHIFT_NULL, SE_METHOD_JACKKNIFE)
JACKKNIFE_GROUPS = 10
LAMBDAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
DEFAULT_CHANNEL = "behavioural_feedback"
RESPONSE_TRIAL_TYPE = "response"

# Facets of the v1 RAM composite that have no referent in the bench
# generator; the protocol declares them NOT_APPLICABLE with these reasons.
FACETS_NOT_APPLICABLE = {
    "goal_alignment": "no_goal_contingency_referent",
    "feedback_magnitude": "no_graded_feedback_referent",
    "speed": "speed_not_in_construct",
}

# Observation models on which RAM-PE is declared not applicable before any
# run (UNDEFINED(NOT_APPLICABLE_OBSERVATION_MODEL), never ABSENT), and the
# aliases that name them.
NOT_APPLICABLE_OBSERVATION_MODELS = {
    "C1": (
        "v1 family-C carrier recording (Re z at 20 Hz of 1.15-1.25 Hz "
        "carriers): the pre-response window (0.6 s) is shorter than one "
        "carrier period (0.87 s) and the plasticity signal lives in the "
        "amplitude envelope, which Re z does not carry"
    ),
}
OBSERVATION_MODEL_ALIASES = {"C": "C1"}

_SD_FLOOR = 1e-12


# --------------------------------------------------------------------------
# declarations
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RAMParams:
    """Declared estimator settings (bench defaults; hash-covered in the
    protocol through :meth:`to_dict`)."""

    window_sec: float = 0.4
    lag_sec: float = 0.0
    n_folds: int = 5
    inner_folds: int = 4
    lambdas: Tuple[float, ...] = LAMBDAS
    h_min: int = 5
    min_updates: int = 30
    max_response_overlap: float = 0.05
    channel: Optional[str] = DEFAULT_CHANNEL
    se_method: str = SE_METHOD_SHIFT_NULL
    placebo: bool = True
    magnitude_window_sec: float = 0.3
    fir_window_sec: float = 0.8

    def __post_init__(self):
        set_ = object.__setattr__
        set_(self, "lambdas", tuple(float(v) for v in self.lambdas))
        if not (math.isfinite(self.window_sec) and self.window_sec > 0):
            raise ValueError("window_sec must be > 0")
        if not (math.isfinite(self.lag_sec) and self.lag_sec >= 0):
            raise ValueError("lag_sec must be >= 0")
        if int(self.n_folds) < 2 or int(self.inner_folds) < 2:
            raise ValueError("n_folds and inner_folds must be >= 2")
        if not self.lambdas or any(not (math.isfinite(v) and v > 0)
                                   for v in self.lambdas):
            raise ValueError("lambdas must be positive and finite")
        if list(self.lambdas) != sorted(self.lambdas):
            raise ValueError("lambdas must be ascending (ties go to the smallest)")
        if int(self.h_min) < 1:
            raise ValueError("h_min must be >= 1")
        if int(self.min_updates) < 2 * int(self.h_min) or int(self.min_updates) < int(
                self.n_folds):
            raise ValueError("min_updates must allow every fold and shift")
        if not 0.0 <= float(self.max_response_overlap) < 1.0:
            raise ValueError("max_response_overlap must lie in [0, 1)")
        if self.se_method not in SE_METHODS:
            raise ValueError(f"se_method must be one of {SE_METHODS}")
        for name in ("n_folds", "inner_folds", "h_min", "min_updates"):
            set_(self, name, int(getattr(self, name)))

    def to_dict(self) -> dict:
        out = dataclasses.asdict(self)
        out["lambdas"] = list(self.lambdas)
        return out

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping]) -> "RAMParams":
        """Parameters from a protocol block; unknown keys are refused."""
        if payload is None:
            return cls()
        if isinstance(payload, cls):
            return payload
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(payload) - names)
        if unknown:
            raise ValueError(f"unknown RAM-PE parameters {unknown}")
        return cls(**dict(payload))


def normalize_observation_model(observation_model) -> Optional[str]:
    """The canonical name of an observation model (``'C'`` is the v1
    family-C carrier recording ``'C1'``); None stays None."""
    if observation_model is None:
        return None
    name = str(observation_model).strip()
    return OBSERVATION_MODEL_ALIASES.get(name, name)


def not_applicable(observation_model) -> Optional[str]:
    """The declared reason why RAM-PE is not applicable to an observation
    model, or None when it is applicable."""
    return NOT_APPLICABLE_OBSERVATION_MODELS.get(
        normalize_observation_model(observation_model))


# --------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class RAMEvents:
    """The logged task of one run: stimulus and response onsets (s) and the
    choice/outcome log (one row per trial, onset = outcome time)."""

    stimulus_onsets: np.ndarray
    response_onsets: np.ndarray
    choice_onsets: np.ndarray
    choices: Tuple[object, ...]
    rewards: np.ndarray
    goal_onsets: np.ndarray = dataclasses.field(
        default_factory=lambda: np.empty(0))
    feedback_onsets: np.ndarray = dataclasses.field(
        default_factory=lambda: np.empty(0))
    source: str = "mapping"


def _floats(values) -> np.ndarray:
    if values is None:
        return np.empty(0, dtype=float)
    return np.asarray(list(values), dtype=float).reshape(-1)


def _response_onsets_from_table(df, channel) -> np.ndarray:
    import pandas as pd

    cols = {str(c).strip().lower(): c for c in df.columns}
    if "onset" not in cols or "trial_type" not in cols:
        return np.empty(0, dtype=float)
    tt = df[cols["trial_type"]].astype(str).str.strip().str.lower()
    rows = tt == RESPONSE_TRIAL_TYPE
    if channel is not None and "impact_channel" in cols:
        ch = df[cols["impact_channel"]].astype(str).str.strip().str.lower()
        rows &= ch == str(channel)
    on = pd.to_numeric(df.loc[rows, cols["onset"]], errors="coerce")
    return on.to_numpy(dtype=float)


def ram_events(events, channel: Optional[str] = DEFAULT_CHANNEL):
    """
    The RAM-PE inputs of an events table (BIDS-like ``DataFrame``), of a
    mapping or of a :class:`RAMEvents`. Returns ``(RAMEvents, reason)`` with
    ``reason`` None or the estimator-level reason why the inputs are not
    usable.

    A table is parsed by the pipeline's parser
    (``event_parsing.events_table_to_bundle``); with a ``channel`` declared,
    only that ``impact_channel``'s events are used, and a table without
    channel labels is ``untyped_events``. Responses are the rows whose
    ``trial_type`` is ``response``. A mapping is taken as the channel's events
    already (keys ``onsets`` or ``stimulus_onsets``, ``response_onsets``,
    ``choice_onsets``, ``choices``, ``rewards``, optional ``goal_onsets`` and
    ``feedback_onsets``).
    """
    if isinstance(events, RAMEvents):
        return events, None
    if events is None:
        return None, "missing_events"
    if hasattr(events, "columns") and hasattr(events, "to_dict"):
        from impact_pipeline.event_parsing import events_table_to_bundle

        bundle = events_table_to_bundle(events)
        if channel is not None:
            per = bundle.get("channel_bundles")
            if not isinstance(per, dict) or not per:
                return None, "untyped_events"
            bundle = per.get(str(channel)) or {}
        return RAMEvents(
            stimulus_onsets=_floats(bundle.get("onsets")),
            response_onsets=_response_onsets_from_table(events, channel),
            choice_onsets=_floats(bundle.get("choice_onsets")),
            choices=tuple(bundle.get("choices") or ()),
            rewards=mm._coerce_numeric_feedback(bundle.get("rewards")),
            goal_onsets=_floats(bundle.get("goal_onsets")),
            feedback_onsets=_floats(bundle.get("feedback_onsets")),
            source="events_table",
        ), None
    if isinstance(events, Mapping):
        stim = events.get("stimulus_onsets", events.get("onsets"))
        return RAMEvents(
            stimulus_onsets=_floats(stim),
            response_onsets=_floats(events.get("response_onsets")),
            choice_onsets=_floats(events.get("choice_onsets")),
            choices=tuple(events.get("choices") or ()),
            rewards=mm._coerce_numeric_feedback(events.get("rewards")),
            goal_onsets=_floats(events.get("goal_onsets")),
            feedback_onsets=_floats(events.get("feedback_onsets")),
            source="mapping",
        ), None
    raise TypeError("events must be a DataFrame, a mapping or RAMEvents")


# --------------------------------------------------------------------------
# patterns, targets and update pairs
# --------------------------------------------------------------------------
def zscore_nodes(x) -> np.ndarray:
    """Nodes z-scored over the run (constant nodes become 0)."""
    x = np.asarray(x, dtype=float)
    sd = x.std(axis=1, keepdims=True)
    z = (x - x.mean(axis=1, keepdims=True)) / np.where(sd > _SD_FLOOR, sd, 1.0)
    return np.where(sd > _SD_FLOOR, z, 0.0)


def window_samples(params: RAMParams, dt: float) -> Tuple[int, int]:
    """``(W_s, L_s)``: window length (>= 1) and lag (>= 0) in samples."""
    w = max(1, int(round(float(params.window_sec) / float(dt))))
    lag = max(0, int(round(float(params.lag_sec) / float(dt))))
    return w, lag


def response_patterns(z, stim_idx, w_s: int, l_s: int) -> np.ndarray:
    """``r_k``: the mean of ``z`` over ``[s_k + L_s, s_k + L_s + W_s)`` per
    stimulus (rows), NaN where the window leaves the run."""
    z = np.asarray(z, dtype=float)
    n_nodes, n_time = z.shape
    idx = np.asarray(stim_idx, dtype=np.int64).reshape(-1)
    out = np.full((idx.size, n_nodes), np.nan)
    ok = (idx + l_s >= 0) & (idx + l_s + w_s <= n_time)
    if np.any(ok):
        offs = np.arange(l_s, l_s + w_s, dtype=np.int64)
        out[ok] = z[:, idx[ok][:, None] + offs[None, :]].mean(axis=2).T
    return out


def window_response_overlap(stim_idx, stim_onsets, response_onsets, dt,
                            w_s: int, l_s: int, n_time: int) -> dict:
    """
    How often the response window reaches the logged response: for every
    stimulus whose window lies in the run and that has a response logged in
    ``[sigma_k, sigma_{k+1})`` (the first one), the window overlaps the
    response iff ``s_k + L_s + W_s > rint(rho_k / dt)`` (the window is
    half-open, so a window ending at the response sample is clear). Returns
    ``{n_checked, n_overlap, fraction}`` (fraction NaN without checks).
    """
    idx = np.asarray(stim_idx, dtype=np.int64)
    on = np.asarray(stim_onsets, dtype=float)
    resp = _floats(response_onsets)
    resp = np.sort(resp[np.isfinite(resp)])
    n_checked = n_overlap = 0
    for k in range(idx.size):
        if idx[k] + l_s + w_s > int(n_time) or idx[k] + l_s < 0:
            continue
        hi = on[k + 1] if k + 1 < on.size else np.inf
        j = int(np.searchsorted(resp, on[k], side="left"))
        if j >= resp.size or not resp[j] < hi:
            continue
        n_checked += 1
        if idx[k] + l_s + w_s > int(np.rint(resp[j] / float(dt))):
            n_overlap += 1
    frac = n_overlap / n_checked if n_checked else float("nan")
    return {"n_checked": n_checked, "n_overlap": n_overlap, "fraction": frac}


def option_signs(choices, options: Sequence[str]) -> np.ndarray:
    """``+1`` for the first option label in sorted order, ``-1`` otherwise."""
    first = sorted(str(o) for o in options)[0]
    return np.asarray([1.0 if str(c) == first else -1.0 for c in choices])


def stimulus_targets(stim_onsets, outcome_onsets, signed_pe) -> Tuple[np.ndarray, dict]:
    """
    The target ``y_k`` of every stimulus: the signed prediction error of the
    one outcome logged in ``[sigma_k, sigma_{k+1})`` (the last stimulus runs
    to the end), NaN when no outcome or several outcomes fall there. Returns
    ``(targets, counts)``.
    """
    on = np.asarray(stim_onsets, dtype=float)
    tau = np.asarray(outcome_onsets, dtype=float)
    val = np.asarray(signed_pe, dtype=float)
    pos = np.searchsorted(on, tau, side="right") - 1
    inside = pos >= 0
    n_per = np.bincount(pos[inside], minlength=on.size) if on.size else np.zeros(0, int)
    out = np.full(on.size, np.nan)
    one = np.flatnonzero(n_per == 1)
    if one.size:
        first = {int(p): j for j, p in enumerate(pos) if p >= 0 and n_per[p] == 1}
        for k in one:
            out[k] = val[first[int(k)]]
    counts = {
        "n_outcomes_before_first_stimulus": int(np.sum(~inside)),
        "n_stimuli_without_outcome": int(np.sum(n_per == 0)),
        "n_stimuli_with_several_outcomes": int(np.sum(n_per > 1)),
    }
    return out, counts


def update_pairs(patterns, targets, lag: int = 1):
    """
    Update rows at lag ``m``: ``d_k = r_{k+m} - r_k`` with target ``y_k`` for
    every ``k`` with both patterns and the target defined. Returns
    ``(D, y, keys)``; ``keys`` are the stimulus indices ``k`` (the purge
    distance is measured on them).
    """
    r = np.asarray(patterns, dtype=float)
    y = np.asarray(targets, dtype=float)
    m = int(lag)
    if m < 1:
        raise ValueError("lag must be >= 1")
    n = r.shape[0]
    k = np.arange(max(0, n - m))
    ok = (np.all(np.isfinite(r[k]), axis=1) & np.all(np.isfinite(r[k + m]), axis=1)
          & np.isfinite(y[k]))
    k = k[ok]
    return r[k + m] - r[k], y[k], k


def placebo_pairs(patterns, targets):
    """Precedence placebo rows: the change ``r_k - r_{k-1}`` that precedes
    outcome ``k``, with target ``y_k``. Returns ``(D, y, keys)``."""
    r = np.asarray(patterns, dtype=float)
    y = np.asarray(targets, dtype=float)
    k = np.arange(1, r.shape[0])
    ok = (np.all(np.isfinite(r[k]), axis=1) & np.all(np.isfinite(r[k - 1]), axis=1)
          & np.isfinite(y[k]))
    k = k[ok]
    return r[k] - r[k - 1], y[k], k


# --------------------------------------------------------------------------
# cross-validated ridge readout
# --------------------------------------------------------------------------
def contiguous_folds(n: int, n_folds: int):
    """``n_folds`` contiguous blocks of row positions (``np.array_split``)."""
    return np.array_split(np.arange(int(n)), int(n_folds))


def purged_training_rows(keys, test_rows, purge: int) -> np.ndarray:
    """
    Training positions for a test fold: every row whose key lies more than
    ``purge`` from every key of the test fold (so the fold and the ``purge``
    rows next to each of its boundaries are excluded; rows across a gap in
    the keys, for example a deleted jackknife block, are not neighbours).
    """
    keys = np.asarray(keys, dtype=np.int64).reshape(-1)
    test_rows = np.asarray(test_rows, dtype=np.int64).reshape(-1)
    if test_rows.size == 0:
        return np.arange(keys.size)
    tk = np.sort(keys[test_rows])
    pos = np.searchsorted(tk, keys)
    left = tk[np.clip(pos - 1, 0, tk.size - 1)]
    right = tk[np.clip(pos, 0, tk.size - 1)]
    dist = np.minimum(np.abs(keys - left), np.abs(keys - right))
    return np.flatnonzero(dist > int(purge))


def _ridge_hat(x, train, test, lambdas):
    """Hat matrices ``H_lambda`` (test x train) of the standardised ridge:
    ``pred = H (y_train - mean) + mean``."""
    xtr = x[train]
    mu = xtr.mean(axis=0)
    sd = xtr.std(axis=0)
    sd = np.where(sd > _SD_FLOOR, sd, 1.0)
    ztr = (xtr - mu) / sd
    zte = (x[test] - mu) / sd
    evals, evecs = np.linalg.eigh(ztr.T @ ztr)
    evals = np.clip(evals, 0.0, None)
    left = zte @ evecs
    right = evecs.T @ ztr.T
    return [(left / (evals + lam)) @ right for lam in lambdas]


def _pearson_columns(a, b) -> np.ndarray:
    """Pearson r of matching columns (NaN for a constant column)."""
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    num = np.sum(a * b, axis=0)
    den = np.sqrt(np.sum(a * a, axis=0) * np.sum(b * b, axis=0))
    out = np.full(num.shape, np.nan)
    ok = np.isfinite(den) & (den > 0)
    out[ok] = num[ok] / den[ok]
    return out


def out_of_fold_predictions(d, y, keys=None, *, n_folds=5, inner_folds=4,
                            purge=1, lambdas=LAMBDAS):
    """
    Pooled out-of-fold predictions of the purged ridge readout ``d -> y``
    for one or several target columns ``y`` (``n`` or ``n x s``). Returns
    ``(pred, lambda_index)``: predictions shaped like ``y`` and the chosen
    ``lambda`` index per (outer fold, column).
    """
    d = np.asarray(d, dtype=float)
    yy = np.asarray(y, dtype=float)
    single = yy.ndim == 1
    y2 = yy.reshape(yy.shape[0], -1)
    n, s = y2.shape
    if d.shape[0] != n:
        raise ValueError("d and y must have the same number of rows")
    keys = np.arange(n) if keys is None else np.asarray(keys, dtype=np.int64)
    lams = tuple(float(v) for v in lambdas)
    pred = np.full((n, s), np.nan)
    chosen = np.zeros((int(n_folds), s), dtype=np.int64)
    for f, test in enumerate(contiguous_folds(n, n_folds)):
        train = purged_training_rows(keys, test, purge)
        if train.size < 2 or test.size == 0:
            continue
        dtr, ytr, ktr = d[train], y2[train], keys[train]
        inner = np.full((train.size, len(lams), s), np.nan)
        for g in contiguous_folds(train.size, inner_folds):
            itr = purged_training_rows(ktr, g, purge)
            if itr.size < 2 or g.size == 0:
                continue
            yi = ytr[itr]
            ym = yi.mean(axis=0, keepdims=True)
            for li, h in enumerate(_ridge_hat(dtr, itr, g, lams)):
                inner[g, li, :] = h @ (yi - ym) + ym
        rows = np.all(np.isfinite(inner[:, 0, :]), axis=1)
        if np.sum(rows) > 3:
            r_in = np.stack([_pearson_columns(inner[rows, li, :], ytr[rows])
                             for li in range(len(lams))])
        else:
            r_in = np.full((len(lams), s), np.nan)
        r_in = np.where(np.isfinite(r_in), r_in, -np.inf)
        best = np.argmax(r_in, axis=0)  # first maximum: the smallest lambda
        chosen[f] = best
        ym = ytr.mean(axis=0, keepdims=True)
        outs = np.stack([h @ (ytr - ym) + ym
                         for h in _ridge_hat(d, train, test, lams)])
        pred[test] = np.take_along_axis(outs, best[None, None, :], axis=0)[0]
    return (pred[:, 0] if single else pred), chosen


def readout_statistic(d, y, keys=None, *, n_folds=5, inner_folds=4, purge=1,
                      lambdas=LAMBDAS):
    """``m``: Pearson r of the pooled out-of-fold predictions with ``y``
    (one value per target column)."""
    yy = np.asarray(y, dtype=float)
    pred, _ = out_of_fold_predictions(d, yy, keys, n_folds=n_folds,
                                      inner_folds=inner_folds, purge=purge,
                                      lambdas=lambdas)
    p2 = pred.reshape(pred.shape[0], -1)
    y2 = yy.reshape(yy.shape[0], -1)
    ok = np.all(np.isfinite(p2), axis=1)
    m = _pearson_columns(p2[ok], y2[ok]) if np.sum(ok) > 2 else np.full(
        y2.shape[1], np.nan)
    return float(m[0]) if yy.ndim == 1 else m


def null_shifts(n: int, h_min: int) -> np.ndarray:
    """Every cyclic shift ``h`` in ``{h_min, ..., n - h_min}``."""
    return np.arange(int(h_min), int(n) - int(h_min) + 1, dtype=np.int64)


def readout_with_null(d, y, keys=None, *, h_min=5, n_folds=5, inner_folds=4,
                      purge=1, lambdas=LAMBDAS):
    """
    The readout and its exact shift null in one pass: ``(m, null, shifts)``
    with ``null[i]`` the readout of ``y`` cyclically shifted by
    ``shifts[i]`` rows against the unchanged pattern changes.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    shifts = null_shifts(y.size, h_min)
    cols = [y] + [np.roll(y, int(h)) for h in shifts]
    m = readout_statistic(d, np.column_stack(cols), keys, n_folds=n_folds,
                          inner_folds=inner_folds, purge=purge, lambdas=lambdas)
    return float(m[0]), np.asarray(m[1:], dtype=float), shifts


def jackknife_readout(d, y, keys=None, *, groups=JACKKNIFE_GROUPS, n_folds=5,
                      inner_folds=4, purge=1, lambdas=LAMBDAS):
    """
    Delete-a-group jackknife over ``groups`` contiguous blocks of update
    rows (the remaining rows keep their keys, so no purge spans the deleted
    block): ``(se, replicates)`` with ``se = sqrt((G - 1) / G * sum (m_g -
    mean)^2)`` over the finite replicates (NaN with fewer than two).
    """
    d = np.asarray(d, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    keys = np.arange(y.size) if keys is None else np.asarray(keys, dtype=np.int64)
    reps = []
    for block in np.array_split(np.arange(y.size), int(groups)):
        keep = np.setdiff1d(np.arange(y.size), block)
        if keep.size < max(int(n_folds), 3):
            reps.append(float("nan"))
            continue
        reps.append(readout_statistic(d[keep], y[keep], keys[keep], n_folds=n_folds,
                                      inner_folds=inner_folds, purge=purge,
                                      lambdas=lambdas))
    reps = np.asarray(reps, dtype=float)
    fin = reps[np.isfinite(reps)]
    if fin.size < 2:
        return float("nan"), reps
    g = fin.size
    return float(np.sqrt((g - 1) / g * np.sum((fin - fin.mean()) ** 2))), reps


# --------------------------------------------------------------------------
# descriptors (no null, no status)
# --------------------------------------------------------------------------
def _boxcar(onsets, dt, n_time, width):
    out = np.zeros(int(n_time))
    for on in np.asarray(onsets, dtype=float):
        if not np.isfinite(on):
            continue
        i = int(np.rint(on / float(dt)))
        if 0 <= i < n_time:
            out[i:min(int(n_time), i + width)] += 1.0
    return out / float(width)


def evoked_descriptors(x, dt, ev: RAMEvents, stim_idx, params: RAMParams) -> dict:
    """Evoked magnitude (mean |beta| of a stimulus boxcar GLM with goal and
    feedback nuisance regressors and an intercept) and FIR latency (v1
    ``_fir_latency_seconds``); descriptors without a null."""
    x = np.asarray(x, dtype=float)
    n_time = x.shape[1]
    width = max(1, int(round(float(params.magnitude_window_sec) / float(dt))))
    cols = [_boxcar(ev.stimulus_onsets, dt, n_time, width)]
    for on in (ev.goal_onsets, ev.feedback_onsets):
        col = _boxcar(on, dt, n_time, width)
        if np.any(col != 0.0):
            cols.append(col)
    design = np.column_stack(cols + [np.ones(n_time)])
    out = {"evoked_magnitude": float("nan"), "evoked_magnitude_reason": None,
           "fir_latency_sec": float("nan"), "fir_latency_reason": None}
    if NUM.matrix_rank(design) < design.shape[1]:
        out["evoked_magnitude_reason"] = "magnitude_design_rank_deficient"
    else:
        beta, *_ = NUM.lstsq(design, x.T, rcond=None)
        out["evoked_magnitude"] = float(np.mean(np.abs(beta[0])))
    lat, reason = mm._fir_latency_seconds(x, stim_idx, float(dt),
                                          float(params.fir_window_sec))
    out["fir_latency_sec"] = float(lat)
    out["fir_latency_reason"] = reason
    return out


def absent_reachable(se, reference, se_df, delta=0.10,
                     alpha_absent=0.01) -> Optional[bool]:
    """``t_q se / rho < delta`` (``t_q`` the one-sided ``1 - alpha_A``
    Student quantile with ``se_df`` df): whether an equivalence test could
    pass at this precision. None when an input is missing."""
    from scipy import stats

    vals = [se, reference, se_df]
    if any(v is None or not np.isfinite(float(v)) for v in vals):
        return None
    if float(reference) <= 0 or float(se_df) <= 0:
        return None
    q = float(stats.t.ppf(1.0 - float(alpha_absent), float(se_df)))
    return bool(q * float(se) / float(reference) < float(delta))


# --------------------------------------------------------------------------
# estimator
# --------------------------------------------------------------------------
def _result(details, *, reason=None, estimate=float("nan"), null_mean=float("nan"),
            null_sd=float("nan"), n_null=0, se=float("nan"), se_df=None,
            se_method=None) -> dict:
    if reason is not None:
        R.lookup(reason)
    defined = reason is None
    value = estimate - null_mean if defined else float("nan")
    return {
        "principle": PRINCIPLE,
        "construct": CONSTRUCT,
        "estimator_version": ESTIMATOR_VERSION,
        "estimator_id": ESTIMATOR_ID,
        "defined": defined,
        "reason": reason,
        "estimate": float(estimate),
        "value": float(value),
        "null_mean": float(null_mean),
        "null_sd": float(null_sd),
        "n_null": int(n_null),
        "null_family": NULL_FAMILY if n_null else None,
        "se": float(se),
        "se_df": None if se_df is None else float(se_df),
        "se_method": se_method,
        "exact": False,
        "details": details,
    }


def _not_defined(detail) -> str:
    return R.format_reason(R.NOT_DEFINED, R.clean_detail(detail))


def compute_ram_v3(
    ts,
    dt: float,
    events,
    *,
    bearer_nodes: Optional[Sequence[int]] = None,
    observation_model=None,
    params=None,
    reference: Optional[float] = None,
    describe_not_applicable: bool = True,
) -> dict:
    """
    RAM-PE v3 of one run (module docstring).

    ``ts`` is the node x time recording at interval ``dt`` (s); ``events``
    an events table, a mapping or :class:`RAMEvents` (:func:`ram_events`);
    ``bearer_nodes`` restricts the rows; ``observation_model`` names the
    recording (``'C1'``/``'C'`` is declared not applicable); ``params`` the
    declared settings (:class:`RAMParams` or a mapping); ``reference`` the
    anchor (reference excess), used only for the ``absent_reachable``
    descriptor. Undefined results carry a reason of the central vocabulary:
    ``NOT_APPLICABLE_OBSERVATION_MODEL``, ``INSUFFICIENT_UPDATES`` (fewer than
    30 updates, fewer than two options chosen) or ``NOT_DEFINED:<reason>``
    (``untyped_events``, ``non_finite_timeseries``,
    ``missing_stimulus_events``, ``missing_choice_reward_log``,
    ``choice_reward_log_misaligned``, ``n_options_not_implemented``,
    ``prediction_error_model_unidentifiable``, ``missing_response_log``,
    ``window_includes_response``, ``degenerate_readout``,
    ``insufficient_null_shifts``).
    """
    p = RAMParams.from_mapping(params)
    if not (dt is not None and np.isfinite(float(dt)) and float(dt) > 0):
        raise ValueError("dt must be a positive number")
    dt = float(dt)
    x = np.asarray(ts, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"ts must be 2D (nodes x time), got shape {x.shape}")
    if bearer_nodes is not None:
        x = x[np.asarray(list(bearer_nodes), dtype=np.int64)]
    model = normalize_observation_model(observation_model)
    details = {
        "construct": CONSTRUCT,
        "estimator_id": ESTIMATOR_ID,
        "mode": MODE,
        "params": p.to_dict(),
        "facets_not_applicable": dict(FACETS_NOT_APPLICABLE),
        "persistence": "not_tested",
        "observation_model": model,
        "bearer_nodes": (None if bearer_nodes is None
                         else [int(i) for i in bearer_nodes]),
        "n_bearer_nodes": int(x.shape[0]),
        "channel": p.channel,
    }
    na = not_applicable(model)
    if na is not None:
        details["not_applicable"] = na
        if describe_not_applicable:
            try:
                inner = compute_ram_v3(ts, dt, events, bearer_nodes=bearer_nodes,
                                       observation_model=None, params=p,
                                       describe_not_applicable=False)
                details["descriptive"] = {
                    k: inner[k] for k in ("defined", "reason", "estimate", "value",
                                          "null_mean", "null_sd", "n_null")}
            except Exception as exc:  # noqa: BLE001 - descriptive only
                details["descriptive"] = {"error": type(exc).__name__}
        return _result(details, reason=R.NOT_APPLICABLE_OBSERVATION_MODEL)

    ev, ev_reason = ram_events(events, p.channel)
    if ev_reason is not None:
        return _result(details, reason=_not_defined(ev_reason))
    details["events_source"] = ev.source
    if not np.all(np.isfinite(x)):
        return _result(details, reason=_not_defined("non_finite_timeseries"))
    n_time = x.shape[1]
    stim_on, stim_idx = mm._sanitize_onset_seconds(ev.stimulus_onsets, tr=dt,
                                                   n_tp=n_time)
    details["n_stimuli"] = int(stim_idx.size)
    if stim_idx.size == 0:
        return _result(details, reason=_not_defined("missing_stimulus_events"))

    # choice / outcome log (chronological), as the v1 prediction-error mode
    on, ch, rw = ev.choice_onsets, list(ev.choices), np.asarray(ev.rewards, float)
    if on.size == 0 and not ch and rw.size == 0:
        return _result(details, reason=_not_defined("missing_choice_reward_log"))
    if not (on.size == len(ch) == rw.size):
        return _result(details, reason=_not_defined("choice_reward_log_misaligned"))
    keep = np.isfinite(on) & np.isfinite(rw)
    keep &= np.asarray([not mm._is_missing_label(c) for c in ch], dtype=bool)
    order = np.argsort(on[keep], kind="mergesort")
    on, rw = on[keep][order], rw[keep][order]
    ch = [c for c, k in zip(ch, keep) if k]
    ch = [ch[i] for i in order]
    options = sorted({str(c) for c in ch})
    details["options"] = options
    details["n_logged_trials"] = int(on.size)
    if len(options) < 2 or on.size < 3:
        return _result(details, reason=R.INSUFFICIENT_UPDATES)
    if len(options) > 2:
        return _result(details, reason=_not_defined("n_options_not_implemented"))
    fit = mm._fit_rescorla_wagner(ch, rw)
    if fit is None:
        return _result(details,
                       reason=_not_defined("prediction_error_model_unidentifiable"))
    pe = np.asarray(fit.pop("prediction_errors"), dtype=float)
    details["prediction_error_model"] = dict(fit, model="rescorla_wagner_softmax")
    details["option_sign"] = {"plus": options[0], "minus": options[1]}
    targets, counts = stimulus_targets(stim_on, on, option_signs(ch, options) * pe)
    details.update(counts)

    # patterns and the window-before-response check
    w_s, l_s = window_samples(p, dt)
    details["window_samples"] = int(w_s)
    details["lag_samples"] = int(l_s)
    z = zscore_nodes(x)
    patterns = response_patterns(z, stim_idx, w_s, l_s)
    details["n_stimuli_window_outside_run"] = int(
        np.sum(~np.all(np.isfinite(patterns), axis=1)))
    overlap = window_response_overlap(stim_idx, stim_on, ev.response_onsets, dt,
                                      w_s, l_s, n_time)
    details["window_response_overlap"] = overlap
    if overlap["n_checked"] == 0:
        return _result(details, reason=_not_defined("missing_response_log"))
    if overlap["fraction"] > p.max_response_overlap:
        return _result(details, reason=_not_defined("window_includes_response"))

    d1, y1, k1 = update_pairs(patterns, targets, lag=1)
    details["n_updates"] = int(y1.size)
    if y1.size < p.min_updates:
        return _result(details, reason=R.INSUFFICIENT_UPDATES)
    cv = {"n_folds": p.n_folds, "inner_folds": p.inner_folds, "purge": 1,
          "lambdas": p.lambdas}
    m1, null, shifts = readout_with_null(d1, y1, k1, h_min=p.h_min, **cv)
    null = null[np.isfinite(null)]
    if not np.isfinite(m1):
        return _result(details, reason=_not_defined("degenerate_readout"))
    if null.size < 2:
        return _result(details, reason=_not_defined("insufficient_null_shifts"))
    nu = float(np.mean(null))
    s1 = float(np.std(null, ddof=1))
    _, lam_idx = out_of_fold_predictions(d1, y1, k1, **cv)
    details.update({
        "null_shifts": [int(shifts[0]), int(shifts[-1])],
        "null_enumerated": True,
        "p_shift": float((1.0 + np.sum(null >= m1)) / (1.0 + null.size)),
        "z_shift": float((m1 - nu) / s1) if s1 > 0 else float("nan"),
        "lambda_per_fold": [float(p.lambdas[i]) for i in lam_idx[:, 0]],
        "se_shift_null": s1,
    })
    se_jk, reps = jackknife_readout(d1, y1, k1, groups=JACKKNIFE_GROUPS, **cv)
    details["se_jackknife"] = se_jk
    details["jackknife_replicates"] = [float(v) for v in reps]
    if p.se_method == SE_METHOD_SHIFT_NULL:
        se, se_df = s1, float(null.size - 1)
    else:
        n_ok = int(np.sum(np.isfinite(reps)))
        se, se_df = se_jk, (float(n_ok - 1) if n_ok >= 2 else None)

    if p.placebo:
        dp, yp, kp = placebo_pairs(patterns, targets)
        if yp.size >= p.min_updates:
            mp, nullp, _ = readout_with_null(dp, yp, kp, h_min=p.h_min, **cv)
            nullp = nullp[np.isfinite(nullp)]
            details["placebo"] = {
                "n_rows": int(yp.size),
                "estimate": mp,
                "null_mean": float(np.mean(nullp)) if nullp.size else float("nan"),
                "excess": (mp - float(np.mean(nullp))) if nullp.size else float("nan"),
                "p_shift": (float((1.0 + np.sum(nullp >= mp)) / (1.0 + nullp.size))
                            if nullp.size else float("nan")),
            }
        else:
            details["placebo"] = {"n_rows": int(yp.size),
                                  "reason": "insufficient_updates"}
    details["descriptors"] = evoked_descriptors(x, dt, ev, stim_idx, p)
    if reference is not None:
        details["absent_reachable"] = absent_reachable(se, reference, se_df)
    return _result(details, estimate=m1, null_mean=nu, null_sd=s1,
                   n_null=int(null.size), se=se, se_df=se_df, se_method=p.se_method)


def component_fields(result: Mapping) -> dict:
    """The estimator fields of a ``mpc-bench-result/3`` component
    (``estimate``, null moments and family, ``se``, ``se_df``,
    ``se_method``, ``details``); the runner adds status, reason, ``c`` and
    the protocol fields. An undefined result carries no SE."""
    defined = bool(result.get("defined"))
    details = dict(result.get("details") or {})
    details["defined"] = defined
    details["estimator_reason"] = result.get("reason")
    return {
        "estimate": result.get("estimate") if defined else None,
        "null_mean": result.get("null_mean") if defined else None,
        "null_sd": result.get("null_sd") if defined else None,
        "n_null": int(result.get("n_null") or 0) if defined else 0,
        "null_family": result.get("null_family") if defined else None,
        "se": result.get("se") if defined else None,
        "se_df": result.get("se_df") if defined else None,
        "se_method": result.get("se_method") if defined else None,
        "details": details,
    }


__all__ = [
    "CONSTRUCT",
    "ESTIMATOR_ID",
    "ESTIMATOR_VERSION",
    "FACETS_NOT_APPLICABLE",
    "JACKKNIFE_GROUPS",
    "LAMBDAS",
    "NOT_APPLICABLE_OBSERVATION_MODELS",
    "NULL_FAMILY",
    "PRINCIPLE",
    "RAMEvents",
    "RAMParams",
    "SE_METHODS",
    "SE_METHOD_JACKKNIFE",
    "SE_METHOD_SHIFT_NULL",
    "absent_reachable",
    "component_fields",
    "compute_ram_v3",
    "contiguous_folds",
    "evoked_descriptors",
    "jackknife_readout",
    "normalize_observation_model",
    "not_applicable",
    "null_shifts",
    "option_signs",
    "out_of_fold_predictions",
    "placebo_pairs",
    "purged_training_rows",
    "ram_events",
    "readout_statistic",
    "readout_with_null",
    "response_patterns",
    "stimulus_targets",
    "update_pairs",
    "window_response_overlap",
    "window_samples",
    "zscore_nodes",
]
