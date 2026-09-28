"""
Generic surrogate generators and component null distributions.

Every MPC component is evidence only relative to a declared null family: the
same estimator, with the same configuration, applied to surrogate data in which
the structure the component claims to measure has been destroyed while nuisance
structure is kept. This module provides the surrogate generators and one
generic driver, :func:`component_null`, that turns any estimator into a null
distribution ``(null_mean, null_sd, samples, n_failed)``.

Time-series surrogates (``ts`` is ``n_nodes x n_time``):

- ``circular_shift``: every node but the first is circularly shifted by its
  own random lag of at least ``min_shift`` samples (default 10% of the run).
  Each node's marginal and amplitude spectrum are preserved exactly;
  cross-node alignment (cross-correlation, coupling) is destroyed.
- ``phase``: multivariate Fourier phase randomisation with the *same* random
  phase offsets for all nodes. Amplitude spectra and all cross-spectra (hence
  the circular auto- and cross-correlation functions) are preserved exactly;
  non-linear / non-Gaussian structure is destroyed; marginals become Gaussian.
- ``phase_independent``: univariate phase randomisation (independent phases
  per node). Spectra are preserved, cross-node structure is destroyed.
- ``iaaft`` / ``iaaft_independent``: iterative amplitude-adjusted Fourier
  transform surrogates (Schreiber & Schmitz 1996), initialised with shared
  (``iaaft``) or independent phases. A *few* iterations are run
  (``n_iter``, default 10) and the last step is the rank remap, so each node's
  marginal is preserved exactly and its amplitude spectrum approximately; with
  shared initial phases the cross-correlation is approximately preserved.

Event surrogates (``events`` is a BIDS events ``DataFrame``, an RAM/SRPI event
bundle ``dict`` as produced by :mod:`impact_pipeline.event_parsing`, or a plain
sequence of onsets in seconds):

- ``label_permutation``: labels are exchanged between events, preserving the
  label counts. For DataFrames the ``label_col`` values are permuted, within
  strata of ``stratify_col`` (default ``phase_bin``) when that column exists,
  and optionally only among rows whose label is in ``among``. For bundles the
  onsets of each exchangeable group (default: ``self_onsets`` and
  ``nonself_onsets``) are pooled and re-dealt with the original counts
  (stratified by ``<key>_phase_bin`` lists when present), and paired values
  (default: ``feedback_values`` of ``feedback_onsets``) are permuted across
  their events.
- ``onset_jitter``: onsets are displaced in time. ``common=True`` shifts the
  whole event train rigidly (one shift for all events, wrapped into
  ``[0, t_max)``), which keeps the relative timing of goal, stimulus and
  feedback events and destroys only their alignment with the recording;
  ``common=False`` jitters each onset independently by ``U(-max_jitter,
  max_jitter)`` (``max_jitter=None``: uniform re-placement in ``[0, t_max)``).
  Paired values stay attached to their events; row order is kept.

All generators take a ``numpy.random.Generator`` (or a seed) and are
deterministic given it.
"""
from __future__ import annotations

import hashlib
import logging
import math
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SURROGATE_KINDS = (
    "circular_shift",
    "phase",
    "phase_independent",
    "iaaft",
    "iaaft_independent",
    "label_permutation",
    "onset_jitter",
)
TS_SURROGATE_KINDS = SURROGATE_KINDS[:5]
EVENT_SURROGATE_KINDS = SURROGATE_KINDS[5:]
_KIND_ALIASES = {
    "circular": "circular_shift",
    "phase_randomize": "phase",
    "multivariate_phase": "phase",
    "phase_randomize_independent": "phase_independent",
    "univariate_phase": "phase_independent",
    "label_perm": "label_permutation",
    "jitter": "onset_jitter",
}
IAAFT_DEFAULT_ITERATIONS = 10
BUNDLE_LABEL_GROUPS = (("self_onsets", "nonself_onsets"),)
BUNDLE_VALUE_PAIRS = (("feedback_onsets", "feedback_values"),)
# Errors an estimator may legitimately raise on a degenerate surrogate; they
# are counted as failed surrogates. Anything else propagates.
_ESTIMATOR_ERRORS = (ValueError, ArithmeticError, np.linalg.LinAlgError)


class ComponentNull(NamedTuple):
    """Null distribution of one component: unpacks as a 4-tuple."""

    null_mean: float
    null_sd: float
    samples: np.ndarray
    n_failed: int


def derive_seed(seed, *keys) -> int:
    """
    Deterministic 32-bit child seed from a base seed and labels (e.g. the
    component and the run file), stable across processes and platforms.
    """
    text = "|".join([repr(int(seed))] + [str(k) for k in keys])
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def normalize_kind(kind) -> str:
    """Canonical surrogate kind name (accepts the documented aliases)."""
    key = str(kind).strip().lower()
    key = _KIND_ALIASES.get(key, key)
    if key not in SURROGATE_KINDS:
        raise ValueError(
            f"unknown surrogate kind {kind!r}; expected one of {SURROGATE_KINDS}"
        )
    return key


def _as_rng(rng):
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(0 if rng is None else rng)


def _as_ts(ts, finite=False):
    x = np.asarray(ts, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"ts must be 2D (n_nodes x n_time), got shape {x.shape}")
    if finite and not np.all(np.isfinite(x)):
        raise ValueError("Fourier surrogates require a finite time series")
    return x


# --------------------------------------------------------------------------
# time-series surrogates
# --------------------------------------------------------------------------
def circular_shift(ts, rng=None, *, min_shift=None) -> np.ndarray:
    """
    Independent circular shift per node: node 0 is the reference and every
    other node is rolled by its own lag in ``[min_shift, n_time - min_shift]``
    (default ``min_shift`` = 10% of the run, at least 1 sample), so each node
    is misaligned with node 0 by at least ``min_shift`` (as in
    ``mpc_metrics._surrogate_timeseries``). Two non-reference nodes can
    receive similar lags; their relative alignment is random, not excluded.
    """
    x = _as_ts(ts)
    rng = _as_rng(rng)
    n, t = x.shape
    lo = int(math.ceil(0.1 * t)) if min_shift is None else int(min_shift)
    lo = max(1, lo)
    hi = t - lo
    if hi < lo:
        raise ValueError(
            f"run too short (n_time={t}) for circular shifts with min_shift={lo}"
        )
    shifts = rng.integers(lo, hi + 1, size=max(0, n - 1))
    out = x.copy()
    for i in range(1, n):
        out[i] = np.roll(x[i], int(shifts[i - 1]))
    return out


def _random_phases(rng, rows, n_freq, t):
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(rows, n_freq))
    phases[:, 0] = 0.0  # keep the mean
    if t % 2 == 0 and n_freq > 1:
        phases[:, -1] = 0.0  # Nyquist bin must stay real
    return phases


def phase_randomize(ts, rng=None, *, multivariate=True) -> np.ndarray:
    """
    Fourier phase-randomised surrogate (no amplitude adjustment). With
    ``multivariate=True`` the same phase offsets are applied to every node, so
    amplitude spectra and cross-spectra are preserved exactly.
    """
    x = _as_ts(ts, finite=True)
    rng = _as_rng(rng)
    n, t = x.shape
    spec = np.fft.rfft(x, axis=1)
    phases = _random_phases(rng, 1 if multivariate else n, spec.shape[1], t)
    return np.fft.irfft(spec * np.exp(1j * phases), n=t, axis=1)


def iaaft(
    ts, rng=None, *, n_iter=IAAFT_DEFAULT_ITERATIONS, multivariate=True
) -> np.ndarray:
    """
    Iterative amplitude-adjusted Fourier transform surrogate.

    Initialised with a (shared-phase when ``multivariate``) phase-randomised
    surrogate, then ``n_iter`` rounds of (rank remap to the original marginal,
    re-impose the original amplitude spectrum keeping the current phases),
    ending with a rank remap. Marginals are exact, spectra approximate; few
    iterations are used on purpose (cost; the spectral error is already small
    after ~10 rounds for smooth spectra).
    """
    x = _as_ts(ts, finite=True)
    if int(n_iter) < 0:
        raise ValueError("n_iter must be >= 0")
    rng = _as_rng(rng)
    n, t = x.shape
    target_amp = np.abs(np.fft.rfft(x, axis=1))
    sorted_vals = np.sort(x, axis=1)
    surr = phase_randomize(x, rng, multivariate=multivariate)
    rows = np.arange(n)[:, None]
    out = np.empty_like(x)
    for it in range(int(n_iter) + 1):
        ranks = np.argsort(np.argsort(surr, axis=1, kind="mergesort"), axis=1)
        out = sorted_vals[rows, ranks]
        if it == int(n_iter):
            break
        cur = np.fft.rfft(out, axis=1)
        surr = np.fft.irfft(target_amp * np.exp(1j * np.angle(cur)), n=t, axis=1)
    return out


# --------------------------------------------------------------------------
# event surrogates
# --------------------------------------------------------------------------
def _permute_within_strata(n_items, strata, rng):
    """Index permutation that only exchanges items within equal strata."""
    perm = np.arange(n_items)
    if strata is None:
        return rng.permutation(n_items)
    keys = pd.Series(strata).astype(object).where(pd.notna(strata), "__nan__")
    for _, idx in keys.groupby(keys, sort=False).groups.items():
        idx = np.asarray(list(idx), dtype=int)
        perm[idx] = idx[rng.permutation(idx.size)]
    return perm


def permute_labels(
    events,
    rng=None,
    *,
    label_col="trial_type",
    stratify_col="phase_bin",
    among=None,
    groups=BUNDLE_LABEL_GROUPS,
    value_pairs=BUNDLE_VALUE_PAIRS,
):
    """
    Label-permutation surrogate of an events table or RAM/SRPI event bundle
    (see the module docstring). Label counts (and within-stratum counts) are
    preserved; onsets are unchanged.
    """
    rng = _as_rng(rng)
    if isinstance(events, pd.DataFrame):
        if label_col not in events.columns:
            raise ValueError(f"events have no label column {label_col!r}")
        out = events.copy()
        rows = np.arange(len(out))
        if among is not None:
            allowed = set(among)
            rows = rows[out[label_col].isin(allowed).to_numpy()]
        strata = None
        if stratify_col is not None and stratify_col in out.columns:
            strata = out[stratify_col].to_numpy()[rows]
        perm = _permute_within_strata(rows.size, strata, rng)
        col = out.columns.get_loc(label_col)
        labels = out.iloc[rows, col].to_numpy(copy=True)
        out.iloc[rows, col] = labels[perm]
        return out
    if isinstance(events, dict):
        out = dict(events)
        for group in groups:
            present = [k for k in group if k in out and out[k] is not None]
            if len(present) < 2:
                continue
            pooled, counts, bins = [], [], []
            has_bins = all(f"{k}_phase_bin" in out for k in present)
            for k in present:
                vals = list(np.asarray(out[k], dtype=float).reshape(-1))
                pooled.extend(vals)
                counts.append(len(vals))
                if has_bins:
                    kb = list(out[f"{k}_phase_bin"])
                    if len(kb) != len(vals):
                        raise ValueError(f"{k}_phase_bin must align with {k}")
                    bins.extend(kb)
            # Re-deal the pooled events to the labels (counts preserved, and
            # within phase bins when bins are given).
            perm = _permute_within_strata(len(pooled), bins if has_bins else None, rng)
            pooled_arr = np.asarray(pooled, dtype=float)[perm]
            bins_arr = np.asarray(bins, dtype=object)[perm] if has_bins else None
            start = 0
            for k, c in zip(present, counts):
                out[k] = pooled_arr[start:start + c].tolist()
                if has_bins:
                    out[f"{k}_phase_bin"] = bins_arr[start:start + c].tolist()
                start += c
        for onset_key, value_key in value_pairs:
            vals = out.get(value_key)
            if vals is None:
                continue
            vals = list(vals)
            if len(vals) != len(list(out.get(onset_key) or [])):
                raise ValueError(f"{value_key} must align with {onset_key}")
            order = rng.permutation(len(vals))
            out[value_key] = [vals[i] for i in order]
        return out
    raise ValueError(
        "label permutation needs labelled events (a DataFrame or an event "
        "bundle dict); a plain onset list carries no labels"
    )


def _shift_onsets(onsets, offsets, t_max):
    x = np.asarray(onsets, dtype=float) + offsets
    if t_max is not None:
        return np.mod(x, float(t_max))
    return np.maximum(x, 0.0)


def jitter_onsets(
    events,
    rng=None,
    *,
    max_jitter=None,
    t_max=None,
    common=False,
    min_shift=0.0,
    onset_col="onset",
):
    """
    Onset-jitter surrogate (see the module docstring). ``t_max`` (seconds) is
    the recording length; shifted onsets wrap into ``[0, t_max)``. Without
    ``t_max`` onsets are only floored at 0, and ``max_jitter`` is required.
    ``min_shift`` (seconds) is the minimum magnitude of the common shift and
    is only valid with ``common=True``.
    """
    rng = _as_rng(rng)
    if t_max is not None and not (np.isfinite(t_max) and float(t_max) > 0):
        raise ValueError("t_max must be a positive number of seconds")
    if max_jitter is None and t_max is None:
        raise ValueError("onset jitter needs max_jitter or t_max")
    if max_jitter is not None and not float(max_jitter) > 0:
        raise ValueError("max_jitter must be > 0")
    if min_shift < 0:
        raise ValueError("min_shift must be >= 0")
    if min_shift > 0 and not common:
        # Independent jitter / re-placement has no minimum displacement; a
        # silently ignored min_shift would misdeclare the null family.
        raise ValueError("min_shift applies to the common (rigid) shift only")
    common_shift = None
    if common:
        if max_jitter is None:
            lo, hi = float(min_shift), float(t_max) - float(min_shift)
            if hi < lo:
                raise ValueError("t_max too short for the requested min_shift")
            common_shift = float(rng.uniform(lo, hi))
        else:
            if float(max_jitter) < float(min_shift):
                raise ValueError("max_jitter must be >= min_shift")
            mag = float(rng.uniform(float(min_shift), float(max_jitter)))
            common_shift = mag if rng.random() < 0.5 else -mag

    def _offsets(n):
        if common_shift is not None:
            return np.full(n, common_shift)
        if max_jitter is None:
            return None  # uniform re-placement
        return rng.uniform(-float(max_jitter), float(max_jitter), size=n)

    def _jitter(onsets):
        arr = np.asarray(onsets, dtype=float).reshape(-1)
        off = _offsets(arr.size)
        if off is None:
            return rng.uniform(0.0, float(t_max), size=arr.size)
        return _shift_onsets(arr, off, t_max)

    if isinstance(events, pd.DataFrame):
        if onset_col not in events.columns:
            raise ValueError(f"events have no onset column {onset_col!r}")
        out = events.copy()
        onsets = pd.to_numeric(out[onset_col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(onsets)
        new = onsets.copy()
        new[ok] = _jitter(onsets[ok])
        out[onset_col] = new
        return out
    if isinstance(events, dict):
        out = dict(events)
        for key, val in events.items():
            if str(key).endswith("onsets") and val is not None:
                out[key] = _jitter(val).tolist()
        return out
    return _jitter(events).tolist()


# --------------------------------------------------------------------------
# dispatch and component nulls
# --------------------------------------------------------------------------
def make_surrogate(kind, ts, events=None, rng=None, **kwargs):
    """
    One surrogate draw ``(ts_s, events_s)`` of kind ``kind``: time-series
    kinds return a surrogate ``ts`` and the events unchanged, event kinds the
    reverse. ``kwargs`` are passed to the generator.
    """
    kind = normalize_kind(kind)
    rng = _as_rng(rng)
    if kind == "circular_shift":
        return circular_shift(ts, rng, **kwargs), events
    if kind == "phase":
        return phase_randomize(ts, rng, multivariate=True, **kwargs), events
    if kind == "phase_independent":
        return phase_randomize(ts, rng, multivariate=False, **kwargs), events
    if kind == "iaaft":
        return iaaft(ts, rng, multivariate=True, **kwargs), events
    if kind == "iaaft_independent":
        return iaaft(ts, rng, multivariate=False, **kwargs), events
    if events is None:
        raise ValueError(f"surrogate kind {kind!r} needs events")
    if kind == "label_permutation":
        return ts, permute_labels(events, rng, **kwargs)
    return ts, jitter_onsets(events, rng, **kwargs)


def _estimator_value(out):
    if isinstance(out, dict):
        out = out.get("value", np.nan)
    try:
        return float(out)
    except (TypeError, ValueError):
        return float("nan")


def component_null(
    estimator_fn: Callable,
    ts,
    events=None,
    kind="circular_shift",
    n=100,
    seed=0,
    *,
    surrogate=None,
    **surrogate_kwargs,
) -> ComponentNull:
    """
    Null distribution of a component estimator under a declared surrogate
    family.

    ``estimator_fn(ts, events)`` must return the component value (a float, or a
    details dict with ``"value"``); it is called on ``n`` surrogate draws of
    kind ``kind`` (``surrogate`` is an alias) generated from
    ``numpy.random.default_rng(seed)``. Draws on which the estimator is
    undefined (non-finite value) or raises ``ValueError``/``ArithmeticError``/
    ``LinAlgError`` count as failed; a surrogate that cannot be generated (e.g.
    a run too short for the minimum shift) fails the whole null.

    Returns ``ComponentNull(null_mean, null_sd, samples, n_failed)``:
    ``samples`` holds the finite null values in draw order, ``null_sd`` uses
    ``ddof=1`` and is NaN with fewer than two finite samples.
    """
    if surrogate is not None:
        kind = surrogate
    kind = normalize_kind(kind)
    n = int(n)
    if n < 0:
        raise ValueError("n must be >= 0")
    rng = np.random.default_rng(seed)
    vals = []
    n_failed = 0
    for _ in range(n):
        try:
            ts_s, ev_s = make_surrogate(kind, ts, events, rng, **surrogate_kwargs)
        except ValueError as exc:
            log.warning("component null (%s): surrogates unavailable: %s", kind, exc)
            return ComponentNull(float("nan"), float("nan"), np.empty(0), n)
        try:
            val = _estimator_value(estimator_fn(ts_s, ev_s))
        except _ESTIMATOR_ERRORS:
            val = float("nan")
        if np.isfinite(val):
            vals.append(val)
        else:
            n_failed += 1
    samples = np.asarray(vals, dtype=float)
    mean = float(np.mean(samples)) if samples.size else float("nan")
    sd = float(np.std(samples, ddof=1)) if samples.size > 1 else float("nan")
    return ComponentNull(mean, sd, samples, int(n_failed))
