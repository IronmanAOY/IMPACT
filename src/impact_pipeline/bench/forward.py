"""
Forward models from simulated sources to EEG-like and BOLD-like observations,
and the observation views of the forward-model arms.

Real data can never validate an estimator, so applicability-registry entries
for ``eeg_like_forward`` and ``bold_like_forward`` come only from
forward-model arms with known sources: one simulated source system is
observed through several views, on which the registry admission procedure
decides whether an estimator is licensed for human EEG or fMRI.

EEG-like (:func:`eeg_forward`)
    ``sensors = L @ sources + EMG + sensor noise``, then average reference and
    a zero-phase band-pass. ``L`` is a random-but-fixed smooth lead field:
    sources and ``n_sensors`` electrodes sit on the unit sphere (electrodes
    quasi-uniformly on the upper cap; whole-brain regions at pseudo-positions
    set by hemisphere and lobe plus a fixed jitter; other systems at fixed
    random positions) and each gain is a Gaussian kernel of the chord
    distance (width ``conduction_width``: volume conduction) times a fixed
    random dipole sign. The lead field depends only on ``leadfield_seed`` and
    the source layout, never on the simulation seed. EMG-like noise is
    broadband (white noise high-passed at ``emg_highpass_hz``), strongest at
    low (temporal/neck) electrodes. A ``montage`` records a subset of the
    electrodes of the same head and noise.
BOLD-like (:func:`bold_forward`)
    neural drive (the source amplitude envelope by default) convolved with the
    SPM canonical double-gamma HRF (peak 5 s, undershoot 15 s, ratio 1/6),
    sampled every ``tr`` seconds, z-scored per region and summed with AR(1)
    physiological noise (coefficient ``ar_coef`` per TR, SD ``noise_sd``
    relative to the unit-variance BOLD signal).

The outputs are :class:`~impact_pipeline.bench.generators.BenchSystem` objects
with ``meta['substrate']`` = ``eeg_like_forward`` / ``bold_like_forward`` and
the source's declared meta; the source signals and the lead field stay in the
oracle.

Views (one simulation, several views; :func:`observe`)
------------------------------------------------------
``source``
    the source system itself (observation ``direct``; the oracle control).
``eeg64``
    64 electrodes, average reference, the regime's sensor band.
``eeg64_noref``
    the same recording without a reference (IIM comparator).
``eeglow``
    the low-density montage: ``n_low`` of the 64 electrode positions,
    chosen by farthest-point sampling (:func:`low_density_subset`), with
    the average re-computed over the recorded electrodes (``n_low``
    defaults to :data:`N_LOW_DEFAULT`, 32).
``mne_template``
    a minimum-norm source estimate of the ``eeg64`` recording with a
    *template* lead field (:class:`TemplateSpec`: the development lead-field
    seed, electrode positions jittered N(0, 0.05), conduction width 0.4),
    ``K = Lc' (Lc Lc' + lambda I)^-1`` with the average-referenced template
    ``Lc`` and ``lambda = tr(Lc Lc') / (n SNR^2)``, SNR 3 (Hamalainen and
    Ilmoniemi 1994, doi:10.1007/BF02512476), written in numpy.
``bold``
    the BOLD-like model (canonical HRF, TR 2 s, AR(1) 0.6, noise 0.5).

All EEG views of one task share one parent recording (lead field, EMG and
sensor noise drawn once for the 64-electrode montage), so the low-density
view is a subset of the same head and noise and the views differ only by
montage, reference and the analyst's inverse.

Regimes
-------
:data:`DEVELOPMENT_REGIME` is the regime of every development run (lead-field
seed 20260928, conduction width 0.5); :data:`HELD_OUT_REGIME` is the
held-out admission regime of confirmatory runs (lead-field seed 20261001,
width 0.6). The lead-field seeds are regime parameters, not task seeds.

Regime keys (registry ``impact-mpc-registry/1``)
------------------------------------------------
Every view carries ``meta['forward_view']['regime']`` with the twelve keys of
:data:`REGIME_KEYS` (None where a key does not apply to the view):
``observation_stage``, ``leadfield_family``, ``leadfield_seed``,
``conduction_width``, ``n_sensors``, ``reference``, ``sensor_filter``,
``fs``, ``tr``, ``duration_s``, ``inputs_declared`` and ``snr``.

Template draws
--------------
The electrode jitter of the template lead field comes from the reserved
stream ``SeedSequence([leadfield_seed, 44])`` (:data:`TEMPLATE_STREAM_KEY`;
keys 41-43 belong to the declared-input and staggered-driver streams). It
depends on the template's lead-field seed only, so the template is one fixed
head model, never a function of the task seed.

This module must not import ``impact_pipeline.mpc_metrics``.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from impact_pipeline import numerics as NUM
from impact_pipeline.bench.generators import BenchSystem, _streams
from impact_pipeline.registry import REGIME_KEYS

FORWARD_VERSION = "mpc-bench-forward/1.0.0"
SPM_HRF = {"a1": 6.0, "a2": 16.0, "b": 1.0, "ratio": 1.0 / 6.0, "length_sec": 32.0}
LEADFIELD_FAMILY = "spherical_gauss"
DEVELOPMENT_LEADFIELD_SEED = 20260928
HELD_OUT_LEADFIELD_SEED = 20261001
N_PARENT_SENSORS = 64
# electrodes of the low-density montage
N_LOW_DEFAULT = 32
# Reserved stream key of the template electrode jitter (41-43: label errors,
# cue jitter, staggered driver).
TEMPLATE_STREAM_KEY = 44
# Size of the rank-safe IIM electrode clusters (macro IIM default): the
# k = min(CLUSTER_MAX, floor(n_sensors / CLUSTER_DIVISOR)) electrodes nearest
# each quadrant centroid, so the four clusters hold at most half the montage
# and never partition it.
CLUSTER_MAX = 8
CLUSTER_DIVISOR = 8

OBSERVATION_STAGES = ("source", "sensor", "source_estimate", "bold")
# identifiability record ``observation`` of each stage (records vocabulary)
OBSERVATION_OF_STAGE = {
    "source": "direct",
    "sensor": "sensor_mixing",
    "source_estimate": "source_estimate",
    "bold": "hemodynamic",
}
# registry substrate of a forward stage (the source stage keeps its own)
SUBSTRATE_OF_STAGE = {
    "sensor": "eeg_like_forward",
    "source_estimate": "eeg_like_forward",
    "bold": "bold_like_forward",
}
# REGIME_KEYS (the twelve registry regime keys) come from the registry.
INPUTS_DECLARED = ("complete", "partial", "none")
REFERENCES = ("average", "none")
MONTAGES = ("full", "low")

_LOBE_POSITION = {
    # (anterior-posterior y, superior-inferior z) on the unit sphere
    "frontal": (0.7, 0.55),
    "cingulate": (0.1, 0.85),
    "insula": (0.2, 0.2),
    "parietal": (-0.5, 0.7),
    "temporal": (-0.1, -0.1),
    "occipital": (-0.9, 0.25),
    "subcortical": (0.0, 0.1),
    "brainstem": (-0.2, -0.5),
    "other": (0.0, 0.0),
}


class ForwardError(ValueError):
    """An invalid view, regime or montage."""


# ---------------------------------------------------------------------------
# Geometry and lead field
# ---------------------------------------------------------------------------


def sensor_positions(n_sensors: int = 64) -> np.ndarray:
    """Quasi-uniform electrode positions on the upper spherical cap (z >= -0.2),
    Fibonacci lattice; deterministic."""
    n = int(n_sensors)
    if n < 4:
        raise ValueError("n_sensors must be >= 4")
    k = np.arange(n) + 0.5
    z = 1.0 - 1.2 * k / n  # from +1 to -0.2
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    phi = k * math.pi * (3.0 - math.sqrt(5.0))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def source_positions(meta: dict, n_sources: int, rng: np.random.Generator):
    """Pseudo-positions of the sources on the sphere (see module docstring)."""
    hemi = meta.get("hemisphere")
    lob = meta.get("lobe")
    jitter = rng.normal(0.0, 0.08, size=(n_sources, 3))
    if hemi is not None and lob is not None and len(hemi) == n_sources:
        pos = np.zeros((n_sources, 3))
        for i in range(n_sources):
            y, z = _LOBE_POSITION.get(lob[i], (0.0, 0.0))
            x = {"L": -0.6, "R": 0.6}.get(hemi[i], 0.0)
            pos[i] = (x, y, z)
        return _unit(0.85 * _unit(pos + 1e-9) + jitter)
    v = rng.normal(size=(n_sources, 3))
    v[:, 2] = np.abs(v[:, 2])
    return _unit(_unit(v) + jitter)


def lead_field(
    src_pos: np.ndarray,
    sens_pos: np.ndarray,
    conduction_width: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Smooth mixing matrix (sensors x sources); columns scaled to max |L| 1."""
    if not conduction_width > 0:
        raise ValueError("conduction_width must be > 0")
    d2 = np.sum((sens_pos[:, None, :] - src_pos[None, :, :]) ** 2, axis=-1)
    L = np.exp(-d2 / (2.0 * float(conduction_width) ** 2))
    if rng is not None:
        L = L * rng.choice([-1.0, 1.0], size=(1, src_pos.shape[0]))
    return L / np.max(np.abs(L), axis=0, keepdims=True)


def _butter(x, fs, lo=None, hi=None, order=4):
    from scipy.signal import butter, sosfiltfilt

    nyq = 0.5 * float(fs)
    if lo and hi:
        sos = butter(order, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    elif lo:
        sos = butter(order, lo / nyq, btype="highpass", output="sos")
    elif hi:
        sos = butter(order, hi / nyq, btype="lowpass", output="sos")
    else:
        return x
    return sosfiltfilt(sos, x, axis=-1)


# ---------------------------------------------------------------------------
# EEG-like
# ---------------------------------------------------------------------------


def _check_eeg_args(source: BenchSystem, band, reference) -> float:
    if reference not in REFERENCES:
        raise ValueError("reference must be 'average' or 'none'")
    fs = 1.0 / float(source.dt)
    if band and band[1] and band[1] >= 0.5 * fs:
        raise ValueError("band upper edge must be below the Nyquist frequency")
    return fs


def _parent_recording(source: BenchSystem, *, n_sensors, seed, leadfield_seed,
                      conduction_width, emg_sd, emg_highpass_hz,
                      sensor_noise_sd) -> dict:
    """The unreferenced, unfiltered recording of the full montage
    (``ts = L s + EMG + noise``), with its lead field, electrode and source
    positions, EMG and source signals."""
    fs = 1.0 / float(source.dt)
    n_src, T = source.ts.shape
    lf_rng = np.random.default_rng(int(leadfield_seed) + 7919 * n_src)
    src_pos = source_positions(source.meta, n_src, lf_rng)
    sens = sensor_positions(n_sensors)
    L = lead_field(src_pos, sens, conduction_width, lf_rng)
    clean = L @ np.asarray(source.ts, dtype=float)
    scale = float(clean.std()) or 1.0
    noise_rng = _streams(seed, 8)[7]
    emg_w = 0.5 + (1.0 - sens[:, 2]) / 1.2  # stronger at low electrodes
    emg = noise_rng.standard_normal((n_sensors, T))
    if emg_highpass_hz and emg_highpass_hz < 0.5 * fs:
        emg = _butter(emg, fs, lo=float(emg_highpass_hz))
    emg = emg / (emg.std(axis=1, keepdims=True) + 1e-12)
    emg = float(emg_sd) * scale * emg_w[:, None] * emg
    white = float(sensor_noise_sd) * scale * noise_rng.standard_normal((n_sensors, T))
    return {
        "ts": clean + emg + white,
        "lead_field": L,
        "source_ts": np.asarray(source.ts, dtype=np.float32),
        "sensor_positions": sens,
        "source_positions": src_pos,
        "emg": emg.astype(np.float32),
    }


def _sensor_system(parent: dict, source: BenchSystem, idx, *, reference,
                   band, fs, params: dict, montage_info: Optional[dict]) -> BenchSystem:
    """One sensor recording from the parent: montage rows, reference, band;
    meta and oracle of the recorded electrodes."""
    if idx is None:
        x = parent["ts"]
        L = parent["lead_field"]
        sens = parent["sensor_positions"]
        emg = parent["emg"]
    else:
        idx = np.asarray(idx, dtype=int)
        x = parent["ts"][idx]
        L = parent["lead_field"][idx]
        sens = parent["sensor_positions"][idx]
        emg = parent["emg"][idx]
    if reference == "average":
        x = x - x.mean(axis=0, keepdims=True)
    if band:
        x = _butter(x, fs, lo=band[0], hi=band[1])
    n_sensors = int(x.shape[0])
    ws_src = [int(i) for i in (source.meta.get("workspace_nodes") or [])]
    ws = sorted({int(np.argmax(np.abs(L[:, j]))) for j in ws_src}) or None
    quad = quadrant_clusters(sens)
    lf_hash = hashlib.sha256(np.ascontiguousarray(L).tobytes()).hexdigest()[:16]
    meta = dict(source.meta)
    forward_block = {
        "model": "eeg_like",
        "n_sensors": n_sensors,
        "conduction_width": float(params["conduction_width"]),
        "leadfield_seed": int(params["leadfield_seed"]),
        "leadfield_sha256_16": lf_hash,
        "emg_sd": float(params["emg_sd"]),
        "emg_highpass_hz": float(params["emg_highpass_hz"]),
        "sensor_noise_sd": float(params["sensor_noise_sd"]),
        "band_hz": list(band) if band else None,
        "reference": reference,
    }
    if montage_info is not None:
        forward_block["montage"] = dict(montage_info)
    meta.update(
        {
            "substrate": "eeg_like_forward",
            "forward_version": FORWARD_VERSION,
            "source_family": source.meta.get("family"),
            "family": f"{source.meta.get('family')}_eeg",
            "n_nodes": n_sensors,
            "n_time": int(x.shape[1]),
            "workspace_nodes": ws,
            "bearer_nodes": list(range(n_sensors)),
            "iim_macro_nodes": quad,
            "iim_grain": "electrode_quadrants",
            "modules": quad,
            "module_order": list(quad),
            "forward": forward_block,
        }
    )
    for key in ("hemisphere", "lobe", "region_labels"):
        meta.pop(key, None)
    oracle = dict(source.oracle)
    oracle.update(
        {
            "lead_field": L,
            "source_ts": parent["source_ts"],
            "sensor_positions": sens,
            "source_positions": parent["source_positions"],
            "emg": emg,
        }
    )
    return BenchSystem(ts=x, events=source.events.copy(), meta=meta, oracle=oracle)


def _montage_indices(montage, n_sensors: int):
    if montage is None:
        return None, None
    idx = [int(i) for i in montage]
    if (len(set(idx)) != len(idx) or not idx
            or any(i < 0 or i >= n_sensors for i in idx)):
        raise ForwardError("montage must list distinct electrode indices of the "
                           "parent montage")
    if len(idx) < 4:
        raise ForwardError("a montage needs at least four electrodes")
    idx = sorted(idx)
    return idx, {"parent_n_sensors": int(n_sensors), "indices": idx}


def eeg_forward(
    source: BenchSystem,
    n_sensors: int = 64,
    seed: int = 0,
    leadfield_seed: int = DEVELOPMENT_LEADFIELD_SEED,
    conduction_width: float = 0.5,
    emg_sd: float = 0.3,
    emg_highpass_hz: float = 20.0,
    sensor_noise_sd: float = 0.02,
    band: tuple = (1.0, 40.0),
    reference: str = "average",
    montage: Optional[Sequence[int]] = None,
) -> BenchSystem:
    """
    EEG-like observation of ``source`` (see the module docstring). ``emg_sd``
    and ``sensor_noise_sd`` are relative to the SD of the noise-free sensor
    signal. ``montage`` lists the recorded electrodes among the
    ``n_sensors`` positions (None: all): the recording of the full montage
    is drawn once (lead field, EMG and sensor noise), and the montage rows
    are selected before the reference (the average over the recorded
    electrodes) and the band-pass. Declared meta: ``workspace_nodes`` are
    the electrodes with the largest lead-field gain from the source's
    declared workspace sources (one per source), ``iim_macro_nodes`` the
    four electrode quadrants.
    """
    fs = _check_eeg_args(source, band, reference)
    params = dict(n_sensors=n_sensors, seed=seed, leadfield_seed=leadfield_seed,
                  conduction_width=conduction_width, emg_sd=emg_sd,
                  emg_highpass_hz=emg_highpass_hz, sensor_noise_sd=sensor_noise_sd)
    parent = _parent_recording(source, **params)
    idx, info = _montage_indices(montage, int(n_sensors))
    return _sensor_system(parent, source, idx, reference=reference, band=band, fs=fs,
                          params=params, montage_info=info)


# ---------------------------------------------------------------------------
# BOLD-like
# ---------------------------------------------------------------------------


def canonical_hrf(dt: float, length_sec: float = None, **params) -> np.ndarray:
    """
    SPM canonical double-gamma HRF sampled every ``dt`` seconds from 0 to
    ``length_sec`` (default 32 s), normalised to unit sum:
    ``g(t; a1, b) - ratio * g(t; a2, b)`` with gamma densities of shape
    ``a1 = 6`` (peak at ``(a1 - 1) b = 5 s``) and ``a2 = 16`` and ratio 1/6.
    """
    from scipy.stats import gamma

    p = dict(SPM_HRF)
    p.update(params)
    length = float(length_sec if length_sec is not None else p["length_sec"])
    t = np.arange(0.0, length + 1e-12, float(dt))
    h = gamma.pdf(t, p["a1"], scale=p["b"]) - p["ratio"] * gamma.pdf(
        t, p["a2"], scale=p["b"]
    )
    return h / h.sum()


def neural_drive(source: BenchSystem, kind: str = "envelope") -> np.ndarray:
    """Neural signal convolved with the HRF: ``'envelope'`` (amplitude
    ``|z|`` from the oracle, else the Hilbert envelope of ``ts``) or
    ``'signal'`` (the source ``ts`` itself)."""
    if kind == "signal":
        return np.asarray(source.ts, dtype=float)
    if kind != "envelope":
        raise ValueError("kind must be 'envelope' or 'signal'")
    env = source.oracle.get("envelope")
    if env is not None and np.shape(env) == source.ts.shape:
        return np.asarray(env, dtype=float)
    from scipy.signal import hilbert

    return np.abs(hilbert(np.asarray(source.ts, dtype=float), axis=1))


def bold_forward(
    source: BenchSystem,
    tr: float = 2.0,
    seed: int = 0,
    drive: str = "envelope",
    ar_coef: float = 0.6,
    noise_sd: float = 0.5,
    hrf_params: Optional[dict] = None,
) -> BenchSystem:
    """
    BOLD-like observation of ``source`` (module docstring): HRF convolution
    at the source rate, point sampling at ``t_k = k * tr``, z-scoring per
    region and additive AR(1) noise. ``tr`` must be a multiple of the source
    sampling interval.
    """
    dt = float(source.dt)
    step = int(round(float(tr) / dt))
    if step < 1 or not math.isclose(step * dt, float(tr), rel_tol=1e-6):
        raise ValueError("tr must be a positive multiple of the source interval")
    if not -1.0 < float(ar_coef) < 1.0:
        raise ValueError("ar_coef must be in (-1, 1)")
    from scipy.signal import fftconvolve

    x = neural_drive(source, drive)
    x = x - x.mean(axis=1, keepdims=True)
    h = canonical_hrf(dt, **(hrf_params or {}))
    conv = fftconvolve(x, h[None, :], mode="full", axes=1)[:, : x.shape[1]]
    bold = conv[:, ::step]
    sd = bold.std(axis=1, keepdims=True)
    bold = (bold - bold.mean(axis=1, keepdims=True)) / np.where(sd > 0, sd, 1.0)
    n, K = bold.shape
    rng = _streams(seed, 8)[7]
    xi = rng.standard_normal((n, K))
    e = np.zeros((n, K))
    if K:
        e[:, 0] = xi[:, 0]
    for k in range(1, K):
        e[:, k] = ar_coef * e[:, k - 1] + math.sqrt(1.0 - ar_coef**2) * xi[:, k]
    noise = float(noise_sd) * e
    meta = dict(source.meta)
    meta.update(
        {
            "substrate": "bold_like_forward",
            "forward_version": FORWARD_VERSION,
            "source_family": source.meta.get("family"),
            "family": f"{source.meta.get('family')}_bold",
            "dt": float(tr),
            "sampling_frequency": float(1.0 / tr),
            "RepetitionTime": float(tr),
            "n_time": int(K),
            "iim_lag_samples": 1,
            "forward": {
                "model": "bold_like",
                "tr": float(tr),
                "drive": drive,
                "hrf": {**SPM_HRF, **(hrf_params or {})},
                "sampling": "point samples at t_k = k * tr",
                "ar_coef": float(ar_coef),
                "noise_sd": float(noise_sd),
            },
        }
    )
    oracle = dict(source.oracle)
    oracle.update(
        {
            "bold_clean": bold,
            "physiological_noise": noise,
            "hrf": h,
            "source_ts": np.asarray(source.ts, dtype=np.float32),
        }
    )
    return BenchSystem(
        ts=bold + noise, events=source.events.copy(), meta=meta, oracle=oracle
    )


# ---------------------------------------------------------------------------
# Regimes and views
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemplateSpec:
    """The analyst's template head of the ``mne_template`` view: the lead
    field of the development lead-field seed (its source positions and
    dipole signs) seen from electrode positions jittered by
    ``N(0, electrode_jitter_sd)`` (then projected back onto the sphere) with
    conduction width ``conduction_width``; minimum-norm SNR ``snr``."""

    leadfield_seed: int = DEVELOPMENT_LEADFIELD_SEED
    electrode_jitter_sd: float = 0.05
    conduction_width: float = 0.4
    snr: float = 3.0

    def __post_init__(self):
        if not (self.electrode_jitter_sd >= 0 and self.conduction_width > 0
                and self.snr > 0):
            raise ForwardError("template: jitter >= 0, width > 0 and snr > 0")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ForwardRegime:
    """
    The parameters of one forward regime: lead field (seed, conduction
    width), montage (``n_sensors`` and the low-density ``n_low``), sensor
    noise and band of the EEG-like model, the template inverse, and the
    BOLD-like model. ``band = None`` records without a sensor band-pass (the
    20 Hz family-A agents, where the 1-40 Hz band is above Nyquist).
    ``held_out`` marks the held-out admission regime.
    """

    name: str
    leadfield_seed: int
    conduction_width: float
    held_out: bool = False
    n_sensors: int = N_PARENT_SENSORS
    n_low: int = N_LOW_DEFAULT
    band: Optional[Tuple[float, float]] = (1.0, 40.0)
    emg_sd: float = 0.3
    emg_highpass_hz: float = 20.0
    sensor_noise_sd: float = 0.02
    template: TemplateSpec = field(default_factory=TemplateSpec)
    tr: float = 2.0
    bold_ar_coef: float = 0.6
    bold_noise_sd: float = 0.5

    def __post_init__(self):
        if not str(self.name).strip():
            raise ForwardError("a regime needs a name")
        if not self.conduction_width > 0:
            raise ForwardError("conduction_width must be > 0")
        if int(self.n_sensors) != self.n_sensors or self.n_sensors < 4:
            raise ForwardError("n_sensors must be an integer >= 4")
        if int(self.n_low) != self.n_low or not 4 <= self.n_low <= self.n_sensors:
            raise ForwardError("n_low must be an integer in [4, n_sensors]")
        if self.band is not None:
            lo, hi = (float(v) for v in self.band)
            if not 0 < lo < hi:
                raise ForwardError("band must be (lo, hi) with 0 < lo < hi")
            object.__setattr__(self, "band", (lo, hi))
        if not self.tr > 0:
            raise ForwardError("tr must be > 0")
        object.__setattr__(self, "n_sensors", int(self.n_sensors))
        object.__setattr__(self, "n_low", int(self.n_low))

    def replace(self, **kw) -> "ForwardRegime":
        return replace(self, **kw)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["band"] = list(self.band) if self.band is not None else None
        return out


DEVELOPMENT_REGIME = ForwardRegime("development", DEVELOPMENT_LEADFIELD_SEED, 0.5)
HELD_OUT_REGIME = ForwardRegime("held_out", HELD_OUT_LEADFIELD_SEED, 0.6, held_out=True)


@dataclass(frozen=True)
class ViewSpec:
    """One observation view: its stage, montage and reference."""

    name: str
    stage: str
    montage: Optional[str] = None
    reference: Optional[str] = None

    def __post_init__(self):
        if self.stage not in OBSERVATION_STAGES:
            raise ForwardError(f"stage must be one of {OBSERVATION_STAGES}")
        eeg = self.stage in ("sensor", "source_estimate")
        if eeg != (self.montage is not None) or eeg != (self.reference is not None):
            raise ForwardError(
                f"view {self.name}: montage and reference belong to EEG stages only")
        if eeg and (self.montage not in MONTAGES or self.reference not in REFERENCES):
            raise ForwardError(f"view {self.name}: montage in {MONTAGES}, "
                               f"reference in {REFERENCES}")
        if self.stage == "source_estimate" and self.reference != "average":
            raise ForwardError(
                "the template inverse works on average-referenced data")


VIEWS: Mapping[str, ViewSpec] = {
    v.name: v
    for v in (
        ViewSpec("source", "source"),
        ViewSpec("eeg64", "sensor", "full", "average"),
        ViewSpec("eeg64_noref", "sensor", "full", "none"),
        ViewSpec("eeglow", "sensor", "low", "average"),
        ViewSpec("mne_template", "source_estimate", "full", "average"),
        ViewSpec("bold", "bold"),
    )
}


def view_spec(name: str) -> ViewSpec:
    if name not in VIEWS:
        raise ForwardError(f"unknown view {name!r}; one of {sorted(VIEWS)}")
    return VIEWS[name]


# ---------------------------------------------------------------------------
# Montage, reference and rank
# ---------------------------------------------------------------------------


def low_density_subset(n_low: int,
                       n_sensors: int = N_PARENT_SENSORS) -> Tuple[int, ...]:
    """
    The low-density montage: ``n_low`` of the ``n_sensors`` electrode
    positions of :func:`sensor_positions`, chosen by greedy farthest-point
    sampling from the vertex electrode (index 0; ties broken by the lower
    index), returned sorted. Deterministic and quasi-uniform, as a
    low-density cap is a subset of a dense one.
    """
    n_low, n_sensors = int(n_low), int(n_sensors)
    if not 1 <= n_low <= n_sensors:
        raise ForwardError("need 1 <= n_low <= n_sensors")
    pos = sensor_positions(n_sensors)
    chosen = [0]
    d = np.linalg.norm(pos - pos[0], axis=1)
    while len(chosen) < n_low:
        d_masked = d.copy()
        d_masked[chosen] = -1.0
        nxt = int(np.argmax(d_masked))  # first index among ties
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(pos - pos[nxt], axis=1))
    return tuple(sorted(chosen))


_QUADRANTS = (("L_ant", -1, 1), ("R_ant", 1, 1), ("L_post", -1, -1), ("R_post", 1, -1))


def quadrant_clusters(sens_pos: np.ndarray) -> Dict[str, list]:
    """The four electrode quadrants (signs of x and y; the declared grain of
    :func:`eeg_forward`), indices local to ``sens_pos``. Under the average
    reference their means are linearly dependent when they partition the
    montage (rank 3)."""
    out = {}
    for name, sx, sy in _QUADRANTS:
        idx = [
            int(i)
            for i in range(sens_pos.shape[0])
            if np.sign(sens_pos[i, 0]) == sx and np.sign(sens_pos[i, 1]) == sy
        ]
        if idx:
            out[name] = idx
    return out


def cluster_size(n_sensors: int) -> int:
    """The macro IIM default cluster size ``k = min(8, floor(n_sensors / 8))``
    (8 on 64 electrodes, 4 on 32)."""
    return min(CLUSTER_MAX, int(n_sensors) // CLUSTER_DIVISOR)


def rank_safe_clusters(sens_pos: np.ndarray) -> Dict[str, list]:
    """
    Rank-safe IIM electrode clusters, the macro IIM default: in every quadrant
    the ``k`` electrodes nearest the quadrant centroid,
    ``k = min(8, floor(n_sensors / 8))`` (:func:`cluster_size`; capped by
    the smallest quadrant of an irregular montage). The four clusters are
    disjoint and hold at most half the montage, so they never cover it and
    their means carry no linear constraint under the average reference.
    """
    sens_pos = np.asarray(sens_pos, dtype=float)
    quad = quadrant_clusters(sens_pos)
    if len(quad) != 4:
        raise ForwardError("the montage does not cover four quadrants")
    per = min(cluster_size(sens_pos.shape[0]), min(len(v) for v in quad.values()))
    if per < 1:
        raise ForwardError("too few electrodes per quadrant for a cluster")
    out = {}
    for name, idx in quad.items():
        cen = sens_pos[idx].mean(axis=0)
        dist = np.linalg.norm(sens_pos[idx] - cen, axis=1)
        order = np.argsort(dist, kind="stable")[:per]
        out[name] = sorted(int(idx[j]) for j in order)
    return out


def average_reference_operator(n: int) -> np.ndarray:
    """The centring matrix ``I - 11'/n`` (rank ``n - 1``)."""
    n = int(n)
    if n < 2:
        raise ForwardError("need at least two channels")
    return np.eye(n) - np.full((n, n), 1.0 / n)


def average_reference(x: np.ndarray) -> np.ndarray:
    """Re-reference channels (rows) to their average at every sample (or
    every column of a lead field)."""
    x = np.asarray(x, dtype=float)
    return x - x.mean(axis=0, keepdims=True)


def montage_rank(x: np.ndarray, rtol: float = 1e-10) -> int:
    """Numerical rank of a channel x time (or channel x source) matrix:
    singular values above ``rtol`` times the largest."""
    s = NUM.svd(np.asarray(x, dtype=float), compute_uv=False)
    if s.size == 0 or s[0] == 0:
        return 0
    return int(np.sum(s > float(rtol) * s[0]))


def cluster_means(x: np.ndarray, clusters: Mapping[str, Sequence[int]]) -> np.ndarray:
    """Cluster x time matrix of the cluster-mean signals."""
    x = np.asarray(x, dtype=float)
    return np.stack([x[np.asarray(v, dtype=int)].mean(axis=0)
                     for v in clusters.values()])


# ---------------------------------------------------------------------------
# Template lead field and minimum-norm inverse
# ---------------------------------------------------------------------------


def template_lead_field(source_meta: Mapping, n_sources: int, sens_pos: np.ndarray,
                        template: TemplateSpec = TemplateSpec()
                        ) -> Tuple[np.ndarray, dict]:
    """
    The analyst's template lead field (sensors x sources): source positions
    and dipole signs of the template's lead-field seed (drawn exactly as
    :func:`eeg_forward` draws a lead field of that seed), electrode
    positions jittered by ``N(0, electrode_jitter_sd)`` from
    ``SeedSequence([leadfield_seed, 44])`` and projected back onto the unit
    sphere, and the template's conduction width. Returns ``(L, info)``.
    """
    n_sources = int(n_sources)
    sens_pos = np.asarray(sens_pos, dtype=float)
    lf_rng = np.random.default_rng(int(template.leadfield_seed) + 7919 * n_sources)
    src_pos = source_positions(dict(source_meta), n_sources, lf_rng)
    jitter_rng = np.random.default_rng(
        np.random.SeedSequence([int(template.leadfield_seed), TEMPLATE_STREAM_KEY]))
    sens_t = sens_pos + jitter_rng.normal(0.0, float(template.electrode_jitter_sd),
                                          size=sens_pos.shape)
    sens_t = _unit(sens_t)
    L = lead_field(src_pos, sens_t, float(template.conduction_width), lf_rng)
    info = {
        "leadfield_seed": int(template.leadfield_seed),
        "electrode_jitter_sd": float(template.electrode_jitter_sd),
        "conduction_width": float(template.conduction_width),
        "jitter_stream": [int(template.leadfield_seed), TEMPLATE_STREAM_KEY],
        "leadfield_sha256_16": hashlib.sha256(
            np.ascontiguousarray(L).tobytes()).hexdigest()[:16],
    }
    return L, info


def mne_lambda(L: np.ndarray, snr: float = 3.0) -> float:
    """``lambda = tr(L L') / (n_sensors SNR^2)``."""
    L = np.asarray(L, dtype=float)
    if not snr > 0:
        raise ForwardError("snr must be > 0")
    return float(np.trace(L @ L.T) / (L.shape[0] * float(snr) ** 2))


def mne_operator(L: np.ndarray, snr: float = 3.0,
                 reference: str = "average") -> Tuple[np.ndarray, float]:
    """
    Minimum-norm inverse operator (sources x sensors)
    ``K = Lc' (Lc Lc' + lambda I)^-1`` with ``Lc`` the lead field under the
    data's reference (``average``: each column re-referenced to the
    electrode mean) and ``lambda = tr(Lc Lc') / (n SNR^2)``. Returns
    ``(K, lambda)``. ``K Lc`` is the Tikhonov resolution matrix
    ``V diag(s^2 / (s^2 + lambda)) V'`` of the SVD ``Lc = U S V'``.
    """
    if reference not in REFERENCES:
        raise ForwardError(f"reference must be one of {REFERENCES}")
    L = np.asarray(L, dtype=float)
    Lc = average_reference(L) if reference == "average" else L
    n = Lc.shape[0]
    lam = mne_lambda(Lc, snr)
    gram = Lc @ Lc.T
    K = np.linalg.solve(gram + lam * np.eye(n), Lc).T
    return K, lam


def _source_estimate_system(eeg: BenchSystem, source: BenchSystem,
                            template: TemplateSpec) -> BenchSystem:
    n_src = int(source.ts.shape[0])
    L_t, info = template_lead_field(source.meta, n_src, eeg.oracle["sensor_positions"],
                                    template)
    K, lam = mne_operator(L_t, template.snr, eeg.meta["forward"]["reference"])
    est = K @ np.asarray(eeg.ts, dtype=float)
    meta = dict(source.meta)
    meta.update(
        {
            "substrate": "eeg_like_forward",
            "forward_version": FORWARD_VERSION,
            "source_family": source.meta.get("family"),
            "family": f"{source.meta.get('family')}_mne",
            "n_nodes": n_src,
            "n_time": int(est.shape[1]),
            "forward": dict(eeg.meta["forward"]),
            "inverse": {
                "method": "minimum_norm",
                "lambda": lam,
                "snr": float(template.snr),
                "lambda_rule": "tr(Lc Lc') / (n_sensors SNR^2)",
                "template": info,
            },
        }
    )
    oracle = dict(source.oracle)
    oracle.update(
        {
            "lead_field": eeg.oracle["lead_field"],
            "lead_field_template": L_t,
            "inverse_operator": K,
            "source_ts": eeg.oracle["source_ts"],
            "sensor_positions": eeg.oracle["sensor_positions"],
            "source_positions": eeg.oracle["source_positions"],
        }
    )
    return BenchSystem(ts=est, events=source.events.copy(), meta=meta, oracle=oracle)


# ---------------------------------------------------------------------------
# Views and regime keys
# ---------------------------------------------------------------------------


def sensor_filter_label(band) -> str:
    """``bandpass_<lo>_<hi>`` (Hz) or ``none``."""
    if not band:
        return "none"
    return f"bandpass_{float(band[0]):g}_{float(band[1]):g}"


def view_regime(spec: ViewSpec, system: BenchSystem, regime: ForwardRegime, *,
                inputs_declared: str) -> dict:
    """The twelve registry regime keys of a view (None where not applicable)."""
    if inputs_declared not in INPUTS_DECLARED:
        raise ForwardError(f"inputs_declared must be one of {INPUTS_DECLARED}")
    dt = float(system.dt)
    out = {k: None for k in REGIME_KEYS}
    out.update(
        observation_stage=spec.stage,
        fs=round(1.0 / dt, 9),
        duration_s=round(int(system.n_time) * dt, 9),
        inputs_declared=inputs_declared,
    )
    if spec.stage in ("sensor", "source_estimate"):
        f = system.meta["forward"]
        out.update(
            leadfield_family=LEADFIELD_FAMILY,
            leadfield_seed=int(f["leadfield_seed"]),
            conduction_width=float(f["conduction_width"]),
            n_sensors=int(f["n_sensors"]),
            reference=f["reference"],
            sensor_filter=sensor_filter_label(f["band_hz"]),
        )
    if spec.stage == "source_estimate":
        out["snr"] = float(regime.template.snr)
    if spec.stage == "bold":
        out["tr"] = float(system.meta["forward"]["tr"])
    return out


def _attach(system: BenchSystem, spec: ViewSpec, regime: ForwardRegime, *,
            inputs_declared: str, extra: Optional[dict] = None) -> BenchSystem:
    block = {
        "version": FORWARD_VERSION,
        "view": spec.name,
        "observation_stage": spec.stage,
        "observation": OBSERVATION_OF_STAGE[spec.stage],
        "regime_name": regime.name,
        "held_out_regime": bool(regime.held_out),
        "regime": view_regime(spec, system, regime, inputs_declared=inputs_declared),
    }
    if extra:
        block.update(extra)
    system.meta["forward_view"] = block
    return system


def observe(
    source: BenchSystem,
    views: Iterable[str],
    regime: ForwardRegime,
    seed: int,
    *,
    bold_source: Optional[BenchSystem] = None,
    bold_drive: str = "envelope",
    inputs_declared: str = "none",
) -> Dict[str, BenchSystem]:
    """
    The requested views of one simulated ``source`` at ``regime`` (the
    module docstring). EEG views share one parent recording drawn with the
    task ``seed``; ``bold`` observes ``bold_source`` (default ``source``;
    the Hopf arm uses a longer run) with drive ``bold_drive``. Every view
    carries ``meta['forward_view']`` (view, stage, identifiability
    observation, regime name, the registry regime keys); sensor views
    declare the rank-safe electrode clusters as ``iim_macro_nodes``
    (grain ``electrode_clusters_rank_safe``) and keep the electrode
    quadrants in ``meta['forward_view']['iim_quadrants']`` for the quadrant
    comparator.
    """
    names = list(dict.fromkeys(views))
    specs = [view_spec(n) for n in names]
    declared = inputs_declared
    out: Dict[str, BenchSystem] = {}
    need_eeg = any(s.stage in ("sensor", "source_estimate") for s in specs)
    parent = None
    params = None
    fs = None
    if need_eeg:
        fs = _check_eeg_args(source, regime.band, "average")
        params = dict(n_sensors=regime.n_sensors, seed=int(seed),
                      leadfield_seed=regime.leadfield_seed,
                      conduction_width=regime.conduction_width, emg_sd=regime.emg_sd,
                      emg_highpass_hz=regime.emg_highpass_hz,
                      sensor_noise_sd=regime.sensor_noise_sd)
        parent = _parent_recording(source, **params)
    low_idx = None
    eeg_cache: Dict[Tuple[str, str], BenchSystem] = {}

    def sensor(montage: str, reference: str) -> BenchSystem:
        nonlocal low_idx
        key = (montage, reference)
        if key not in eeg_cache:
            if montage == "low":
                if low_idx is None:
                    low_idx = list(low_density_subset(regime.n_low, regime.n_sensors))
                idx, info = _montage_indices(low_idx, regime.n_sensors)
                info["rule"] = "farthest_point_from_vertex"
            else:
                idx, info = None, None
            eeg_cache[key] = _sensor_system(parent, source, idx, reference=reference,
                                            band=regime.band, fs=fs, params=params,
                                            montage_info=info)
        return eeg_cache[key]

    for spec in specs:
        if spec.stage == "source":
            sysm = BenchSystem(ts=source.ts, events=source.events.copy(),
                               meta=copy.deepcopy(source.meta), oracle=source.oracle,
                               rest_ts=source.rest_ts)
            out[spec.name] = _attach(sysm, spec, regime, inputs_declared=declared)
        elif spec.stage == "sensor":
            base = sensor(spec.montage, spec.reference)
            sysm = BenchSystem(ts=base.ts, events=base.events.copy(),
                               meta=copy.deepcopy(base.meta), oracle=base.oracle)
            sens = base.oracle["sensor_positions"]
            clusters = rank_safe_clusters(sens)
            quad = dict(sysm.meta["iim_macro_nodes"])
            sysm.meta["iim_macro_nodes"] = clusters
            sysm.meta["iim_grain"] = "electrode_clusters_rank_safe"
            out[spec.name] = _attach(sysm, spec, regime, inputs_declared=declared,
                                     extra={"iim_quadrants": quad,
                                            "iim_clusters": clusters})
        elif spec.stage == "source_estimate":
            base = sensor(spec.montage, spec.reference)
            sysm = _source_estimate_system(base, source, regime.template)
            out[spec.name] = _attach(sysm, spec, regime, inputs_declared=declared)
        else:
            bsrc = bold_source if bold_source is not None else source
            sysm = bold_forward(bsrc, tr=regime.tr, seed=int(seed), drive=bold_drive,
                                ar_coef=regime.bold_ar_coef,
                                noise_sd=regime.bold_noise_sd)
            out[spec.name] = _attach(sysm, spec, regime, inputs_declared=declared,
                                     extra={"drive": bold_drive})
    return out


def regime_keys(system: BenchSystem) -> dict:
    """The registry regime keys of a view built by :func:`observe`."""
    block = system.meta.get("forward_view")
    if not isinstance(block, Mapping) or "regime" not in block:
        raise ForwardError("not a forward view (no meta['forward_view'])")
    return dict(block["regime"])


def substrate_of(system: BenchSystem) -> str:
    """The registry substrate of a view: ``eeg_like_forward`` for sensor and
    source-estimate views, ``bold_like_forward`` for BOLD, the source's own
    substrate for the source view."""
    block = system.meta.get("forward_view") or {}
    stage = block.get("observation_stage")
    if stage in SUBSTRATE_OF_STAGE:
        return SUBSTRATE_OF_STAGE[stage]
    return str(system.meta.get("substrate"))


def scoring_details(system: BenchSystem) -> dict:
    """What a scoring record of a view keeps in ``details`` for the registry
    builder: the view, its stage and identifiability observation, the
    registry substrate, the regime name and the regime keys."""
    block = system.meta.get("forward_view")
    if not isinstance(block, Mapping):
        raise ForwardError("not a forward view (no meta['forward_view'])")
    return {
        "view": block["view"],
        "observation_stage": block["observation_stage"],
        "observation": block["observation"],
        "substrate": substrate_of(system),
        "regime_name": block["regime_name"],
        "held_out_regime": block["held_out_regime"],
        "regime": dict(block["regime"]),
    }


__all__ = [
    "CLUSTER_DIVISOR",
    "CLUSTER_MAX",
    "DEVELOPMENT_LEADFIELD_SEED",
    "DEVELOPMENT_REGIME",
    "FORWARD_VERSION",
    "ForwardError",
    "ForwardRegime",
    "HELD_OUT_LEADFIELD_SEED",
    "HELD_OUT_REGIME",
    "LEADFIELD_FAMILY",
    "N_LOW_DEFAULT",
    "N_PARENT_SENSORS",
    "OBSERVATION_OF_STAGE",
    "REGIME_KEYS",
    "SPM_HRF",
    "SUBSTRATE_OF_STAGE",
    "TEMPLATE_STREAM_KEY",
    "TemplateSpec",
    "VIEWS",
    "ViewSpec",
    "average_reference",
    "average_reference_operator",
    "bold_forward",
    "canonical_hrf",
    "cluster_means",
    "cluster_size",
    "eeg_forward",
    "lead_field",
    "low_density_subset",
    "mne_lambda",
    "mne_operator",
    "montage_rank",
    "neural_drive",
    "observe",
    "quadrant_clusters",
    "rank_safe_clusters",
    "regime_keys",
    "scoring_details",
    "sensor_filter_label",
    "sensor_positions",
    "source_positions",
    "substrate_of",
    "template_lead_field",
    "view_regime",
]
