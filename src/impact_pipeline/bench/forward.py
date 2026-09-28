"""
Forward models from simulated sources to EEG-like and BOLD-like observations
(Paper-1 spec v2, V2-5/V2-6): human EEG/fMRI registry entries require
validation on forward-modelled synthetic data.

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
    low (temporal/neck) electrodes.
BOLD-like (:func:`bold_forward`)
    neural drive (the source amplitude envelope by default) convolved with the
    SPM canonical double-gamma HRF (peak 5 s, undershoot 15 s, ratio 1/6),
    sampled every ``tr`` seconds, z-scored per region and summed with AR(1)
    physiological noise (coefficient ``ar_coef`` per TR, SD ``noise_sd``
    relative to the unit-variance BOLD signal).

The outputs are :class:`~impact_pipeline.bench.generators.BenchSystem` objects
with ``meta['substrate']`` = ``eeg_like_forward`` / ``bold_like_forward`` and
the source's declared meta; the source signals and the lead field stay in the
oracle. This module must not import ``impact_pipeline.mpc_metrics``.
"""

from __future__ import annotations

import hashlib
import math
from typing import Optional

import numpy as np

from impact_pipeline.bench.generators import BenchSystem, _streams

FORWARD_VERSION = "mpc-bench-forward/1.0.0"
SPM_HRF = {"a1": 6.0, "a2": 16.0, "b": 1.0, "ratio": 1.0 / 6.0, "length_sec": 32.0}

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


def eeg_forward(
    source: BenchSystem,
    n_sensors: int = 64,
    seed: int = 0,
    leadfield_seed: int = 20260928,
    conduction_width: float = 0.5,
    emg_sd: float = 0.3,
    emg_highpass_hz: float = 20.0,
    sensor_noise_sd: float = 0.02,
    band: tuple = (1.0, 40.0),
    reference: str = "average",
) -> BenchSystem:
    """
    EEG-like observation of ``source`` (see the module docstring). ``emg_sd``
    and ``sensor_noise_sd`` are relative to the SD of the noise-free sensor
    signal. Declared meta: ``workspace_nodes`` are the electrodes with the
    largest lead-field gain from the source's declared workspace sources (one
    per source), ``iim_macro_nodes`` four electrode quadrants.
    """
    if reference not in ("average", "none"):
        raise ValueError("reference must be 'average' or 'none'")
    fs = 1.0 / float(source.dt)
    if band and band[1] and band[1] >= 0.5 * fs:
        raise ValueError("band upper edge must be below the Nyquist frequency")
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
    x = clean + emg + white
    if reference == "average":
        x = x - x.mean(axis=0, keepdims=True)
    if band:
        x = _butter(x, fs, lo=band[0], hi=band[1])
    ws_src = [int(i) for i in (source.meta.get("workspace_nodes") or [])]
    ws = sorted({int(np.argmax(np.abs(L[:, j]))) for j in ws_src}) or None
    quad = {}
    for name, sx, sy in (
        ("L_ant", -1, 1),
        ("R_ant", 1, 1),
        ("L_post", -1, -1),
        ("R_post", 1, -1),
    ):
        idx = [
            int(i)
            for i in range(n_sensors)
            if np.sign(sens[i, 0]) == sx and np.sign(sens[i, 1]) == sy
        ]
        if idx:
            quad[name] = idx
    lf_hash = hashlib.sha256(np.ascontiguousarray(L).tobytes()).hexdigest()[:16]
    meta = dict(source.meta)
    meta.update(
        {
            "substrate": "eeg_like_forward",
            "forward_version": FORWARD_VERSION,
            "source_family": source.meta.get("family"),
            "family": f"{source.meta.get('family')}_eeg",
            "n_nodes": int(n_sensors),
            "n_time": int(T),
            "workspace_nodes": ws,
            "bearer_nodes": list(range(int(n_sensors))),
            "iim_macro_nodes": quad,
            "iim_grain": "electrode_quadrants",
            "modules": quad,
            "module_order": list(quad),
            "forward": {
                "model": "eeg_like",
                "n_sensors": int(n_sensors),
                "conduction_width": float(conduction_width),
                "leadfield_seed": int(leadfield_seed),
                "leadfield_sha256_16": lf_hash,
                "emg_sd": float(emg_sd),
                "emg_highpass_hz": float(emg_highpass_hz),
                "sensor_noise_sd": float(sensor_noise_sd),
                "band_hz": list(band) if band else None,
                "reference": reference,
            },
        }
    )
    for key in ("hemisphere", "lobe", "region_labels"):
        meta.pop(key, None)
    oracle = dict(source.oracle)
    oracle.update(
        {
            "lead_field": L,
            "source_ts": np.asarray(source.ts, dtype=np.float32),
            "sensor_positions": sens,
            "source_positions": src_pos,
            "emg": emg.astype(np.float32),
        }
    )
    return BenchSystem(ts=x, events=source.events.copy(), meta=meta, oracle=oracle)


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
