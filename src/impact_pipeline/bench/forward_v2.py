"""
Forward-model layer of MPC-Bench v2 (``mpc-bench-forward/2.0.0``): the
observation views on which the registry v3 admission procedure decides
whether an estimator is licensed for human EEG or fMRI.

Real data can never validate an estimator, so registry entries for
``eeg_like_forward`` and ``bold_like_forward`` come only from forward-model
arms with known sources. This module turns one simulated source system into
the views of those arms; the frozen v1 forward models
(:mod:`impact_pipeline.bench.forward`) remain the engine and are not edited.

Views (one simulation, several views)
-------------------------------------
``source``
    the source system itself (observation ``direct``; the oracle control).
``eeg64``
    64 electrodes, average reference, the regime's sensor band.
``eeg64_noref``
    the same recording without a reference (IIM comparator).
``eeglow``
    the low-density montage: ``n_low`` of the 64 electrode positions,
    chosen by farthest-point sampling (:func:`low_density_subset`), with
    the average re-computed over the recorded electrodes. ``n_low`` is the
    smallest channel count of the paper-2 EEG datasets, read before the
    freeze from their BIDS metadata (32 when unavailable,
    :data:`N_LOW_DEFAULT`).
``mne_template``
    a minimum-norm source estimate of the ``eeg64`` recording with a
    *template* lead field (:class:`TemplateSpec`: the development lead-field
    seed, electrode positions jittered N(0, 0.05), conduction width 0.4),
    ``K = Lc' (Lc Lc' + lambda I)^-1`` with the average-referenced template
    ``Lc`` and ``lambda = tr(Lc Lc') / (n SNR^2)``, SNR 3 (Hamalainen and
    Ilmoniemi 1994, doi:10.1007/BF02512476), written in numpy.
``bold``
    the v1 BOLD-like model (canonical HRF, TR 2 s, AR(1) 0.6, noise 0.5).

All EEG views of one task share one parent recording (lead field, EMG and
sensor noise drawn once for the 64-electrode montage, as v1 draws them), so
the low-density view is a subset of the same head and noise and the views
differ only by montage, reference and the analyst's inverse.

Regimes
-------
:data:`DEVELOPMENT_REGIME` is the regime of every development run (lead-field
seed 20260928, conduction width 0.5); :data:`HELD_OUT_REGIME` is the
confirmatory admission regime (lead-field seed 20261001, width 0.6), held
out: before the v2 freeze only its reference-block anchors and discarded
smoke tests may be computed. The lead-field seeds are regime parameters, not
task seeds (``protocols/v2/seed_map_v2.json``).

Regime keys (registry ``impact-mpc-registry/3``)
------------------------------------------------
Every view carries ``meta['forward_v2']['regime']`` with the twelve keys of
:data:`REGIME_KEYS` (None where a key does not apply to the view):
``observation_stage``, ``leadfield_family``, ``leadfield_seed``,
``conduction_width``, ``n_sensors``, ``reference``, ``sensor_filter``,
``fs``, ``tr``, ``duration_s``, ``inputs_declared`` and ``snr``.

Bit-identity with v1
--------------------
:func:`eeg_forward_v2` and :func:`bold_forward_v2` take the v1 signatures
(plus ``montage`` for the EEG model). With the v1 parameters and the full
montage they return the v1 systems bit for bit (series, events, meta and
oracle); the v2 additions live only in the views of :func:`observe`.

New random draws
----------------
The only new draw is the electrode jitter of the template lead field, from
the reserved stream ``SeedSequence([leadfield_seed, 44])``
(:data:`TEMPLATE_STREAM_KEY`; keys 41-43 belong to the declared-input and
staggered-driver streams). It depends on the template's lead-field seed only,
so the template is one fixed head model, never a function of the task seed.

This module must not import ``impact_pipeline.mpc_metrics``.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from impact_pipeline.bench import forward as fwd
from impact_pipeline.bench.generators import BenchSystem
from impact_pipeline.v2 import numerics as NUM
from impact_pipeline.v2.registry_v3 import REGIME_KEYS

FORWARD_V2_VERSION = "mpc-bench-forward/2.0.0"
LEADFIELD_FAMILY = "spherical_gauss"
DEVELOPMENT_LEADFIELD_SEED = 20260928
HELD_OUT_LEADFIELD_SEED = 20261001
N_PARENT_SENSORS = 64
# CD-12 replaces this by the smallest channel count of the paper-2 EEG
# datasets (from their BIDS metadata) before the freeze.
N_LOW_DEFAULT = 32
# Reserved stream key of the template electrode jitter (41-43: label errors,
# cue jitter, staggered driver).
TEMPLATE_STREAM_KEY = 44
# Size of the rank-safe IIM electrode clusters (IIM v5 default): the
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
# REGIME_KEYS (the twelve registry regime keys) come from registry_v3.
INPUTS_DECLARED = ("complete", "partial", "none")
REFERENCES = ("average", "none")
MONTAGES = ("full", "low")

# The v1 defaults of eeg_forward / bold_forward (checked against the v1
# signatures by the tests).
V1_EEG_DEFAULTS = {
    "n_sensors": 64,
    "seed": 0,
    "leadfield_seed": DEVELOPMENT_LEADFIELD_SEED,
    "conduction_width": 0.5,
    "emg_sd": 0.3,
    "emg_highpass_hz": 20.0,
    "sensor_noise_sd": 0.02,
    "band": (1.0, 40.0),
    "reference": "average",
}
V1_BOLD_DEFAULTS = {
    "tr": 2.0,
    "seed": 0,
    "drive": "envelope",
    "ar_coef": 0.6,
    "noise_sd": 0.5,
    "hrf_params": None,
}


class ForwardV2Error(ValueError):
    """An invalid view, regime or montage."""


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
            raise ForwardV2Error("template: jitter >= 0, width > 0 and snr > 0")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ForwardRegime:
    """
    The parameters of one forward regime: lead field (seed, conduction
    width), montage (``n_sensors`` and the low-density ``n_low``), sensor
    noise and band of the EEG-like model, the template inverse, and the
    BOLD-like model. ``band = None`` records without a sensor band-pass (the
    20 Hz family-A agents, where the v1 1-40 Hz band is above Nyquist).
    ``held_out`` marks the confirmatory admission regime.
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
            raise ForwardV2Error("a regime needs a name")
        if not self.conduction_width > 0:
            raise ForwardV2Error("conduction_width must be > 0")
        if int(self.n_sensors) != self.n_sensors or self.n_sensors < 4:
            raise ForwardV2Error("n_sensors must be an integer >= 4")
        if int(self.n_low) != self.n_low or not 4 <= self.n_low <= self.n_sensors:
            raise ForwardV2Error("n_low must be an integer in [4, n_sensors]")
        if self.band is not None:
            lo, hi = (float(v) for v in self.band)
            if not 0 < lo < hi:
                raise ForwardV2Error("band must be (lo, hi) with 0 < lo < hi")
            object.__setattr__(self, "band", (lo, hi))
        if not self.tr > 0:
            raise ForwardV2Error("tr must be > 0")
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
REGIMES = {r.name: r for r in (DEVELOPMENT_REGIME, HELD_OUT_REGIME)}


@dataclass(frozen=True)
class ViewSpec:
    """One observation view: its stage, montage and reference."""

    name: str
    stage: str
    montage: Optional[str] = None
    reference: Optional[str] = None

    def __post_init__(self):
        if self.stage not in OBSERVATION_STAGES:
            raise ForwardV2Error(f"stage must be one of {OBSERVATION_STAGES}")
        eeg = self.stage in ("sensor", "source_estimate")
        if eeg != (self.montage is not None) or eeg != (self.reference is not None):
            raise ForwardV2Error(
                f"view {self.name}: montage and reference belong to EEG stages only")
        if eeg and (self.montage not in MONTAGES or self.reference not in REFERENCES):
            raise ForwardV2Error(f"view {self.name}: montage in {MONTAGES}, "
                                 f"reference in {REFERENCES}")
        if self.stage == "source_estimate" and self.reference != "average":
            raise ForwardV2Error(
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
        raise ForwardV2Error(f"unknown view {name!r}; one of {sorted(VIEWS)}")
    return VIEWS[name]


# ---------------------------------------------------------------------------
# Montage, reference and rank
# ---------------------------------------------------------------------------


def low_density_subset(n_low: int,
                       n_sensors: int = N_PARENT_SENSORS) -> Tuple[int, ...]:
    """
    The low-density montage: ``n_low`` of the ``n_sensors`` electrode
    positions of :func:`impact_pipeline.bench.forward.sensor_positions`,
    chosen by greedy farthest-point sampling from the vertex electrode
    (index 0; ties broken by the lower index), returned sorted. Deterministic
    and quasi-uniform, as a low-density cap is a subset of a dense one.
    """
    n_low, n_sensors = int(n_low), int(n_sensors)
    if not 1 <= n_low <= n_sensors:
        raise ForwardV2Error("need 1 <= n_low <= n_sensors")
    pos = fwd.sensor_positions(n_sensors)
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
    """The v1 electrode quadrants (signs of x and y; v1 ``eeg_forward``),
    indices local to ``sens_pos``. Under the average reference their means
    are linearly dependent when they partition the montage (rank 3)."""
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
    """The IIM v5 default cluster size ``k = min(8, floor(n_sensors / 8))``
    (8 on 64 electrodes, 4 on 32)."""
    return min(CLUSTER_MAX, int(n_sensors) // CLUSTER_DIVISOR)


def rank_safe_clusters(sens_pos: np.ndarray) -> Dict[str, list]:
    """
    Rank-safe IIM electrode clusters, the IIM v5 default: in every quadrant
    the ``k`` electrodes nearest the quadrant centroid,
    ``k = min(8, floor(n_sensors / 8))`` (:func:`cluster_size`; capped by
    the smallest quadrant of an irregular montage). The four clusters are
    disjoint and hold at most half the montage, so they never cover it and
    their means carry no linear constraint under the average reference.
    """
    sens_pos = np.asarray(sens_pos, dtype=float)
    quad = quadrant_clusters(sens_pos)
    if len(quad) != 4:
        raise ForwardV2Error("the montage does not cover four quadrants")
    per = min(cluster_size(sens_pos.shape[0]), min(len(v) for v in quad.values()))
    if per < 1:
        raise ForwardV2Error("too few electrodes per quadrant for a cluster")
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
        raise ForwardV2Error("need at least two channels")
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
# EEG-like model (v1 engine, staged)
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
                      sensor_noise_sd) -> BenchSystem:
    """The unreferenced, unfiltered recording of the full montage: the v1
    model with ``reference='none'`` and ``band=None`` (``ts`` is then
    ``L s + EMG + noise`` with the v1 draws)."""
    return fwd.eeg_forward(
        source,
        n_sensors=n_sensors,
        seed=seed,
        leadfield_seed=leadfield_seed,
        conduction_width=conduction_width,
        emg_sd=emg_sd,
        emg_highpass_hz=emg_highpass_hz,
        sensor_noise_sd=sensor_noise_sd,
        band=None,
        reference="none",
    )


def _sensor_system(parent: BenchSystem, source: BenchSystem, idx, *, reference,
                   band, fs, params: dict, montage_info: Optional[dict]) -> BenchSystem:
    """One sensor recording from the parent: montage rows, reference, band;
    meta and oracle by the v1 rules on the selected electrodes."""
    if idx is None:
        x = parent.ts
        L = parent.oracle["lead_field"]
        sens = parent.oracle["sensor_positions"]
        emg = parent.oracle["emg"]
    else:
        idx = np.asarray(idx, dtype=int)
        x = parent.ts[idx]
        L = parent.oracle["lead_field"][idx]
        sens = parent.oracle["sensor_positions"][idx]
        emg = parent.oracle["emg"][idx]
    if reference == "average":
        x = x - x.mean(axis=0, keepdims=True)
    if band:
        x = fwd._butter(x, fs, lo=band[0], hi=band[1])
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
            "forward_version": fwd.FORWARD_VERSION,
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
            "source_ts": parent.oracle["source_ts"],
            "sensor_positions": sens,
            "source_positions": parent.oracle["source_positions"],
            "emg": emg,
        }
    )
    return BenchSystem(ts=x, events=source.events.copy(), meta=meta, oracle=oracle)


def eeg_forward_v2(
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
    The EEG-like observation of ``source`` with the v1 parameters and
    signature, plus ``montage``: indices of the recorded electrodes among
    the ``n_sensors`` positions (None: all). The recording of the full
    montage is drawn once (v1 lead field, EMG and sensor noise); the montage
    rows are selected before the reference (the average over the recorded
    electrodes) and the band-pass. With ``montage=None`` the result equals
    :func:`impact_pipeline.bench.forward.eeg_forward` bit for bit.
    """
    fs = _check_eeg_args(source, band, reference)
    params = dict(n_sensors=n_sensors, seed=seed, leadfield_seed=leadfield_seed,
                  conduction_width=conduction_width, emg_sd=emg_sd,
                  emg_highpass_hz=emg_highpass_hz, sensor_noise_sd=sensor_noise_sd)
    parent = _parent_recording(source, **params)
    idx, info = _montage_indices(montage, int(n_sensors))
    return _sensor_system(parent, source, idx, reference=reference, band=band, fs=fs,
                          params=params, montage_info=info)


def _montage_indices(montage, n_sensors: int):
    if montage is None:
        return None, None
    idx = [int(i) for i in montage]
    if (len(set(idx)) != len(idx) or not idx
            or any(i < 0 or i >= n_sensors for i in idx)):
        raise ForwardV2Error("montage must list distinct electrode indices of the "
                             "parent montage")
    if len(idx) < 4:
        raise ForwardV2Error("a montage needs at least four electrodes")
    idx = sorted(idx)
    return idx, {"parent_n_sensors": int(n_sensors), "indices": idx}


def bold_forward_v2(
    source: BenchSystem,
    tr: float = 2.0,
    seed: int = 0,
    drive: str = "envelope",
    ar_coef: float = 0.6,
    noise_sd: float = 0.5,
    hrf_params: Optional[dict] = None,
) -> BenchSystem:
    """The BOLD-like observation of ``source``: the v1 model
    (:func:`impact_pipeline.bench.forward.bold_forward`) unchanged."""
    return fwd.bold_forward(source, tr=tr, seed=seed, drive=drive, ar_coef=ar_coef,
                            noise_sd=noise_sd, hrf_params=hrf_params)


# ---------------------------------------------------------------------------
# Template lead field and minimum-norm inverse
# ---------------------------------------------------------------------------


def template_lead_field(source_meta: Mapping, n_sources: int, sens_pos: np.ndarray,
                        template: TemplateSpec = TemplateSpec()
                        ) -> Tuple[np.ndarray, dict]:
    """
    The analyst's template lead field (sensors x sources): source positions
    and dipole signs of the template's lead-field seed (drawn exactly as v1
    draws a lead field of that seed), electrode positions jittered by
    ``N(0, electrode_jitter_sd)`` from ``SeedSequence([leadfield_seed, 44])``
    and projected back onto the unit sphere, and the template's conduction
    width. Returns ``(L, info)``.
    """
    n_sources = int(n_sources)
    sens_pos = np.asarray(sens_pos, dtype=float)
    lf_rng = np.random.default_rng(int(template.leadfield_seed) + 7919 * n_sources)
    src_pos = fwd.source_positions(dict(source_meta), n_sources, lf_rng)
    jitter_rng = np.random.default_rng(
        np.random.SeedSequence([int(template.leadfield_seed), TEMPLATE_STREAM_KEY]))
    sens_t = sens_pos + jitter_rng.normal(0.0, float(template.electrode_jitter_sd),
                                          size=sens_pos.shape)
    sens_t = fwd._unit(sens_t)
    L = fwd.lead_field(src_pos, sens_t, float(template.conduction_width), lf_rng)
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
        raise ForwardV2Error("snr must be > 0")
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
        raise ForwardV2Error(f"reference must be one of {REFERENCES}")
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
            "forward_version": FORWARD_V2_VERSION,
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
        raise ForwardV2Error(f"inputs_declared must be one of {INPUTS_DECLARED}")
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
        "version": FORWARD_V2_VERSION,
        "view": spec.name,
        "observation_stage": spec.stage,
        "observation": OBSERVATION_OF_STAGE[spec.stage],
        "regime_name": regime.name,
        "held_out_regime": bool(regime.held_out),
        "regime": view_regime(spec, system, regime, inputs_declared=inputs_declared),
    }
    if extra:
        block.update(extra)
    system.meta["forward_v2"] = block
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
    carries ``meta['forward_v2']`` (view, stage, identifiability
    observation, regime name, the registry regime keys); sensor views
    declare the rank-safe electrode clusters as ``iim_macro_nodes``
    (grain ``electrode_clusters_rank_safe``) and keep the v1 quadrants in
    ``meta['forward_v2']['iim_quadrants_v1']`` for the v1 comparator.
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
                                     extra={"iim_quadrants_v1": quad,
                                            "iim_clusters": clusters})
        elif spec.stage == "source_estimate":
            base = sensor(spec.montage, spec.reference)
            sysm = _source_estimate_system(base, source, regime.template)
            out[spec.name] = _attach(sysm, spec, regime, inputs_declared=declared)
        else:
            bsrc = bold_source if bold_source is not None else source
            sysm = bold_forward_v2(bsrc, tr=regime.tr, seed=int(seed), drive=bold_drive,
                                   ar_coef=regime.bold_ar_coef,
                                   noise_sd=regime.bold_noise_sd)
            out[spec.name] = _attach(sysm, spec, regime, inputs_declared=declared,
                                     extra={"drive": bold_drive})
    return out


def regime_keys(system: BenchSystem) -> dict:
    """The registry regime keys of a view built by :func:`observe`."""
    block = system.meta.get("forward_v2")
    if not isinstance(block, Mapping) or "regime" not in block:
        raise ForwardV2Error("not a forward_v2 view (no meta['forward_v2'])")
    return dict(block["regime"])


def substrate_of(system: BenchSystem) -> str:
    """The registry substrate of a view: ``eeg_like_forward`` for sensor and
    source-estimate views, ``bold_like_forward`` for BOLD, the source's own
    substrate for the source view."""
    block = system.meta.get("forward_v2") or {}
    stage = block.get("observation_stage")
    if stage in SUBSTRATE_OF_STAGE:
        return SUBSTRATE_OF_STAGE[stage]
    return str(system.meta.get("substrate"))


def scoring_details(system: BenchSystem) -> dict:
    """What a scoring record of a view keeps in ``details`` for the registry
    builder: the view, its stage and identifiability observation, the
    registry substrate, the regime name and the regime keys."""
    block = system.meta.get("forward_v2")
    if not isinstance(block, Mapping):
        raise ForwardV2Error("not a forward_v2 view (no meta['forward_v2'])")
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
    "FORWARD_V2_VERSION",
    "ForwardRegime",
    "ForwardV2Error",
    "HELD_OUT_LEADFIELD_SEED",
    "HELD_OUT_REGIME",
    "LEADFIELD_FAMILY",
    "N_LOW_DEFAULT",
    "N_PARENT_SENSORS",
    "OBSERVATION_OF_STAGE",
    "REGIMES",
    "REGIME_KEYS",
    "SUBSTRATE_OF_STAGE",
    "TEMPLATE_STREAM_KEY",
    "TemplateSpec",
    "VIEWS",
    "ViewSpec",
    "average_reference",
    "average_reference_operator",
    "bold_forward_v2",
    "cluster_means",
    "cluster_size",
    "eeg_forward_v2",
    "low_density_subset",
    "mne_lambda",
    "mne_operator",
    "montage_rank",
    "observe",
    "quadrant_clusters",
    "rank_safe_clusters",
    "regime_keys",
    "scoring_details",
    "sensor_filter_label",
    "substrate_of",
    "template_lead_field",
    "view_regime",
]
