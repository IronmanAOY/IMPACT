# -*- coding: utf-8 -*-
"""
The forward-model layer of MPC-Bench and its design builders:

* the EEG-like and BOLD-like forward models (known answers: average
  reference, smooth lead field, band-pass, canonical HRF, AR(1) noise), and
  the views at the development regime are their recordings;
* the regimes: development (lead-field seed 20260928, width 0.5) and the
  held-out admission regime (20261001, width 0.6), the low-density
  ``n_low`` montage as a subset of the same recording;
* the numpy minimum-norm template inverse: shape, lambda, noise-free
  recovery and the template head of the development lead-field seed;
* the rank of the average-referenced montage (and of the quadrant and
  rank-safe cluster means);
* the forward arms: plan sizes, seeds and regimes per purpose, the held-out
  regime policy, truncation of the Hopf run, realised views, scorings and
  admission designs.
"""
import inspect
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from impact_pipeline import reasons as REASONS
from impact_pipeline import registry as RV
from impact_pipeline.bench import forward as FW
from impact_pipeline.bench import generators as g
from impact_pipeline.bench import seeds as S
from impact_pipeline.bench import whole_brain as wb
from impact_pipeline.bench.designs import forward as D
from impact_pipeline.estimators import declared_inputs as DI

REPO_ROOT = Path(__file__).resolve().parents[1]
SHORT_WB = wb.WholeBrainConfig(duration_sec=4.0, transient_sec=0.5)
SMALL_A = g.AgentConfig(n_trials=12, n_reafference_pairs=4)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _same(a, b, where=""):
    if isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(a, b, check_exact=True)
        except AssertionError as exc:
            return [f"{where}: {exc}"]
        return []
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.dtype != b.dtype or a.shape != b.shape:
            return [f"{where}: {a.dtype}{a.shape} != {b.dtype}{b.shape}"]
        return [] if np.array_equal(a, b, equal_nan=a.dtype.kind in "fc") else [
            f"{where}: arrays differ"]
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f"{where}: keys {sorted(set(a) ^ set(b))}"]
        out = []
        for k in a:
            out += _same(a[k], b[k], f"{where}.{k}")
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            return [f"{where}: sequences differ"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += _same(x, y, f"{where}[{i}]")
        return out
    if (isinstance(a, float) and isinstance(b, float)
            and math.isnan(a) and math.isnan(b)):
        return []
    if type(a) is not type(b) or a != b:
        return [f"{where}: {a!r} != {b!r}"]
    return []


def assert_same_system(a, b):
    diffs = (_same(a.ts, b.ts, "ts") + _same(a.events, b.events, "events")
             + _same(a.meta, b.meta, "meta") + _same(a.oracle, b.oracle, "oracle")
             + _same(a.rest_ts, b.rest_ts, "rest_ts"))
    assert not diffs, diffs[:10]


@pytest.fixture(scope="module")
def hopf():
    return wb.simulate_whole_brain(SHORT_WB, seed=3)


@pytest.fixture(scope="module")
def agent():
    return g.simulate_family_a(None, SMALL_A, 5)


@pytest.fixture(scope="module")
def held_out_views(hopf):
    return FW.observe(hopf, list(FW.VIEWS), FW.HELD_OUT_REGIME, seed=3)


@pytest.fixture(scope="module")
def source():
    cfg = wb.WholeBrainConfig(duration_sec=20.0)
    return wb.simulate_whole_brain(cfg, seed=1, connectome=wb.load_connectome())


# --------------------------------------------------------------------------
# forward models
# --------------------------------------------------------------------------
def test_eeg_forward_model(source):
    e = FW.eeg_forward(source, n_sensors=64, seed=2)
    assert (
        e.ts.shape == (64, source.n_time) and e.meta["substrate"] == "eeg_like_forward"
    )
    # Average reference: sensors sum to zero at every sample.
    assert np.max(np.abs(e.ts.mean(axis=0))) < 1e-10
    # The lead field is smooth (volume conduction) and fixed across seeds.
    L = e.oracle["lead_field"]
    sens = e.oracle["sensor_positions"]
    d = np.linalg.norm(sens[:, None] - sens[None, :], axis=-1)
    rc = np.corrcoef(np.abs(L))
    near = rc[(d > 0) & (d < 0.3)].mean()
    far = rc[d > 1.5].mean()
    assert near > far + 0.3
    e2 = FW.eeg_forward(source, n_sensors=64, seed=9)
    assert np.array_equal(e2.oracle["lead_field"], L)
    assert not np.array_equal(e2.ts, e.ts)
    # Band-pass 1-40 Hz: little power outside the band, alpha kept.
    f = np.fft.rfftfreq(e.n_time, e.dt)
    P = (np.abs(np.fft.rfft(e.ts, axis=1)) ** 2).mean(axis=0)
    alpha = P[(f > 8) & (f < 12)].mean()
    assert P[(f > 60) & (f < 100)].mean() < 1e-3 * alpha
    assert P[f < 0.3].mean() < 1e-2 * alpha
    # Volume conduction makes independent-looking sources correlate at sensors.
    cs = np.abs(np.corrcoef(source.ts)[np.triu_indices(76, 1)]).mean()
    ce = np.abs(np.corrcoef(e.ts)[np.triu_indices(64, 1)]).mean()
    assert ce > cs
    assert set(e.meta["iim_macro_nodes"]) == {"L_ant", "R_ant", "L_post", "R_post"}
    assert e.meta["workspace_nodes"] and "lobe" not in e.meta
    with pytest.raises(ValueError):
        FW.eeg_forward(source, band=(1.0, 200.0))


def test_canonical_hrf_known_answers():
    h = FW.canonical_hrf(0.01)
    t = np.arange(h.size) * 0.01
    assert h.sum() == pytest.approx(1.0)
    assert t[np.argmax(h)] == pytest.approx(5.0, abs=0.02)  # (a1 - 1) * b
    assert 14.0 < t[np.argmin(h)] < 16.5 and h.min() < 0  # undershoot
    assert t[-1] == pytest.approx(32.0)


def test_bold_forward_impulse_response_and_noise(source):
    # Impulse through the forward model (no noise): the HRF sampled at TR.
    imp = source.oracle["envelope"].astype(float) * 0.0
    imp[:, 250] = 1.0  # t = 1 s
    s = type(source)(
        ts=imp, events=source.events, meta=dict(source.meta), oracle={"envelope": imp}
    )
    b = FW.bold_forward(s, tr=2.0, noise_sd=0.0)
    h = FW.canonical_hrf(source.dt)
    clean = b.oracle["bold_clean"][0]
    x = imp[0] - imp[0].mean()
    ref = np.convolve(x, h)[: x.size][::500]
    ref = (ref - ref.mean()) / ref.std()
    assert np.allclose(clean, ref, atol=1e-8)
    # Impulse at 1 s, HRF peak 5 s later: the largest TR sample is t = 6 s.
    assert int(np.argmax(clean)) == 3
    assert b.dt == 2.0 and b.meta["RepetitionTime"] == 2.0
    assert b.meta["substrate"] == "bold_like_forward"
    # AR(1) physiological noise with the declared coefficient.
    long = type(source)(
        ts=np.zeros((4, 250 * 2000)),
        events=source.events,
        meta=dict(source.meta),
        oracle={"envelope": np.random.default_rng(0).random((4, 250 * 2000))},
    )
    nb = FW.bold_forward(long, tr=2.0, ar_coef=0.6, noise_sd=1.0, seed=3)
    e = nb.oracle["physiological_noise"]
    ac = np.mean([np.corrcoef(r[1:], r[:-1])[0, 1] for r in e])
    assert ac == pytest.approx(0.6, abs=0.05)
    with pytest.raises(ValueError):
        FW.bold_forward(source, tr=0.003)
    with pytest.raises(ValueError):
        FW.bold_forward(source, ar_coef=1.0)


def test_montage_is_the_last_eeg_forward_parameter():
    assert list(inspect.signature(FW.eeg_forward).parameters)[-1] == "montage"


def test_eeg_forward_on_a_family_a_agent(agent):
    e = FW.eeg_forward(agent, band=None, seed=7)  # 20 Hz: 1-40 Hz is above Nyquist
    assert e.ts.shape == (64, agent.n_time)
    with pytest.raises(ValueError, match="Nyquist"):
        FW.eeg_forward(agent)
    with pytest.raises(ValueError, match="reference"):
        FW.eeg_forward(agent, band=None, reference="mastoid")


def test_views_at_the_development_regime_are_the_forward_recordings(hopf):
    reg = FW.DEVELOPMENT_REGIME  # the default lead field, width and noise
    views = FW.observe(hopf, ["source", "eeg64", "eeg64_noref", "bold"], reg, seed=3)
    avg = FW.eeg_forward(hopf, seed=3)
    none = FW.eeg_forward(hopf, seed=3, reference="none")
    assert np.array_equal(views["eeg64"].ts, avg.ts)
    assert np.array_equal(views["eeg64_noref"].ts, none.ts)
    assert np.array_equal(views["bold"].ts, FW.bold_forward(hopf, seed=3).ts)
    assert views["source"].ts is hopf.ts
    # the recording's meta keys keep their values except the IIM grain of the
    # sensor views
    for k, v in avg.meta.items():
        if k in ("iim_macro_nodes", "iim_grain"):
            continue
        assert not _same(views["eeg64"].meta[k], v, k), k
    assert views["eeg64"].meta["forward_view"]["iim_quadrants"] == avg.meta[
        "iim_macro_nodes"]
    assert "forward_view" not in hopf.meta  # the source system is not touched


# --------------------------------------------------------------------------
# regimes
# --------------------------------------------------------------------------
def test_regimes():
    dev, ho = FW.DEVELOPMENT_REGIME, FW.HELD_OUT_REGIME
    assert (dev.leadfield_seed, dev.conduction_width, dev.held_out) == (
        20260928, 0.5, False)
    assert (ho.leadfield_seed, ho.conduction_width, ho.held_out) == (
        20261001, 0.6, True)
    for r in (dev, ho):
        assert (r.n_sensors, r.band, r.emg_sd, r.sensor_noise_sd, r.tr) == (
            64, (1.0, 40.0), 0.3, 0.02, 2.0)
        assert (r.bold_ar_coef, r.bold_noise_sd, r.template.snr) == (0.6, 0.5, 3.0)
        assert r.template == FW.TemplateSpec(20260928, 0.05, 0.4, 3.0)
    assert FW.N_LOW_DEFAULT == 32
    with pytest.raises(FW.ForwardError):
        ho.replace(n_low=80)
    with pytest.raises(FW.ForwardError):
        ho.replace(band=(40.0, 1.0))


def test_held_out_views_use_the_held_out_lead_field(hopf, held_out_views):
    want = FW.eeg_forward(hopf, seed=3, leadfield_seed=20261001, conduction_width=0.6)
    v = held_out_views["eeg64"]
    assert np.array_equal(v.ts, want.ts)
    assert np.array_equal(v.oracle["lead_field"], want.oracle["lead_field"])
    dev_lf = FW.eeg_forward(hopf, seed=3).oracle["lead_field"]
    assert not np.allclose(v.oracle["lead_field"], dev_lf)
    reg = FW.regime_keys(v)
    assert reg == {
        "observation_stage": "sensor", "leadfield_family": "spherical_gauss",
        "leadfield_seed": 20261001, "conduction_width": 0.6, "n_sensors": 64,
        "reference": "average", "sensor_filter": "bandpass_1_40", "fs": 250.0,
        "tr": None, "duration_s": 4.0, "inputs_declared": "none", "snr": None}
    assert tuple(reg) == RV.REGIME_KEYS == FW.REGIME_KEYS
    assert v.meta["forward_view"]["held_out_regime"] is True
    assert v.meta["forward_view"]["observation"] == "sensor_mixing"


def test_regime_keys_of_every_view(held_out_views):
    want = {"source": ("source", None, None, None, 250.0, None),
            "eeg64": ("sensor", 64, "average", "bandpass_1_40", 250.0, None),
            "eeg64_noref": ("sensor", 64, "none", "bandpass_1_40", 250.0, None),
            "eeglow": ("sensor", 32, "average", "bandpass_1_40", 250.0, None),
            "mne_template": ("source_estimate", 64, "average", "bandpass_1_40", 250.0,
                             3.0),
            "bold": ("bold", None, None, None, 0.5, None)}
    for name, (stage, n, ref, filt, fs, snr) in want.items():
        r = FW.regime_keys(held_out_views[name])
        assert (r["observation_stage"], r["n_sensors"], r["reference"],
                r["sensor_filter"], r["fs"], r["snr"]) == (stage, n, ref, filt, fs, snr)
        assert r["inputs_declared"] == "none"
        det = FW.scoring_details(held_out_views[name])
        assert det["regime"] == r and det["view"] == name
    assert FW.regime_keys(held_out_views["bold"])["tr"] == 2.0
    subs = {k: FW.substrate_of(v) for k, v in held_out_views.items()}
    assert subs == {"source": "stuart_landau", "eeg64": "eeg_like_forward",
                    "eeg64_noref": "eeg_like_forward", "eeglow": "eeg_like_forward",
                    "mne_template": "eeg_like_forward", "bold": "bold_like_forward"}
    for plain in (FW.eeg_forward(g.BenchSystem(
            ts=np.zeros((3, 200)), events=pd.DataFrame(), meta={"dt": 0.004},
            oracle={}), band=None),):
        with pytest.raises(FW.ForwardError):
            FW.regime_keys(plain)
        with pytest.raises(FW.ForwardError):
            FW.scoring_details(plain)


def test_low_density_subset_is_a_deterministic_spread_subset():
    sub = FW.low_density_subset(32)
    assert sub == FW.low_density_subset(32)
    assert len(sub) == 32 and len(set(sub)) == 32 and sub == tuple(sorted(sub))
    assert 0 in sub and all(0 <= i < 64 for i in sub)
    pos = FW.sensor_positions(64)

    def spread(idx):  # (smallest gap, largest distance of an electrode to the subset)
        p = pos[list(idx)]
        gap = min(np.linalg.norm(p[i] - p[j]) for i in range(len(p)) for j in range(i))
        cover = max(np.min(np.linalg.norm(p - q, axis=1)) for q in pos)
        return gap, cover

    # spread over the cap: wider gaps and a closer cover than every other electrode
    gap, cover = spread(sub)
    gap_alt, cover_alt = spread(range(0, 64, 2))
    assert gap > gap_alt and cover < cover_alt
    assert set(FW.low_density_subset(16)) <= set(FW.low_density_subset(32))
    assert FW.low_density_subset(64) == tuple(range(64))
    with pytest.raises(FW.ForwardError):
        FW.low_density_subset(65)


def test_low_density_view_is_a_subset_of_the_same_recording(hopf, held_out_views):
    low = held_out_views["eeglow"]
    full = held_out_views["eeg64"]
    idx = list(FW.low_density_subset(32))
    assert low.meta["forward"]["montage"] == {
        "parent_n_sensors": 64, "indices": idx, "rule": "farthest_point_from_vertex"}
    assert np.array_equal(low.oracle["lead_field"], full.oracle["lead_field"][idx])
    assert np.array_equal(low.oracle["sensor_positions"],
                          full.oracle["sensor_positions"][idx])
    # the same head and noise: unreferenced parent rows, re-referenced over
    # the recorded electrodes, then band-passed
    raw = FW.eeg_forward(hopf, seed=3, leadfield_seed=20261001, conduction_width=0.6,
                         band=None, reference="none").ts[idx]
    want = FW._butter(raw - raw.mean(axis=0, keepdims=True), 250.0, lo=1.0, hi=40.0)
    assert np.array_equal(low.ts, want)
    assert np.array_equal(FW.eeg_forward(hopf, seed=3, leadfield_seed=20261001,
                                         conduction_width=0.6, montage=idx).ts, want)
    assert low.meta["n_nodes"] == 32 and low.meta["bearer_nodes"] == list(range(32))
    ws = low.meta["workspace_nodes"]
    assert ws and all(0 <= i < 32 for i in ws)
    with pytest.raises(FW.ForwardError):
        FW.eeg_forward(hopf, montage=[0, 0, 1, 2])
    with pytest.raises(FW.ForwardError):
        FW.eeg_forward(hopf, montage=[0, 1, 70, 2])


def test_n_low_flows_from_the_builder_to_the_view():
    (task,) = [t for t in D.build_tasks(D.DRY_RUN, arms=[D.ARM_A_EEG], n_low=24)
               if t.condition == "N_ar1"][:1]
    assert task.n_low == 24
    real = D.realise(task)
    assert real.views["eeglow"].ts.shape[0] == 24
    assert FW.regime_keys(real.views["eeglow"])["n_sensors"] == 24
    cfg = task.to_config()["forward"]
    assert cfg["n_low"] == 24 and cfg["regime_parameters"]["n_low"] == 24


# --------------------------------------------------------------------------
# minimum-norm template inverse
# --------------------------------------------------------------------------
def _toy_lead_field(n_sens=64, n_src=6, seed=0):
    rng = np.random.default_rng(seed)
    src = FW._unit(rng.normal(size=(n_src, 3)) + np.array([0, 0, 0.5]))
    return FW.lead_field(src, FW.sensor_positions(n_sens), 0.5, rng)


def test_mne_operator_shape_and_lambda():
    L = _toy_lead_field(64, 76)
    K, lam = FW.mne_operator(L, snr=3.0)
    assert K.shape == (76, 64)
    Lc = L - L.mean(axis=0, keepdims=True)
    assert lam == pytest.approx(np.trace(Lc @ Lc.T) / (64 * 9.0), rel=1e-12)
    assert FW.mne_lambda(Lc, 3.0) == lam
    want = Lc.T @ np.linalg.inv(Lc @ Lc.T + lam * np.eye(64))
    np.testing.assert_allclose(K, want, rtol=1e-9, atol=1e-12)
    K_none, lam_none = FW.mne_operator(L, snr=2.0, reference="none")
    assert lam_none == pytest.approx(np.trace(L @ L.T) / (64 * 4.0))
    # average-referenced data: the operator ignores the common mode
    np.testing.assert_allclose(K @ np.ones(64), 0.0, atol=1e-10)
    with pytest.raises(FW.ForwardError):
        FW.mne_operator(L, snr=0.0)


def test_mne_operator_is_the_tikhonov_resolution_and_recovers_noise_free_sources():
    L = _toy_lead_field(64, 6, seed=2)
    Lc = L - L.mean(axis=0, keepdims=True)
    K, lam = FW.mne_operator(L, snr=3.0)
    U, s, Vt = np.linalg.svd(Lc, full_matrices=False)
    res = Vt.T @ np.diag(s ** 2 / (s ** 2 + lam)) @ Vt
    np.testing.assert_allclose(K @ Lc, res, atol=1e-10)
    # noise-free recovery: with full column rank and vanishing regularisation
    # the estimate reproduces the sources, also through the average reference
    rng = np.random.default_rng(9)
    srcs = rng.normal(size=(6, 500))
    x = FW.average_reference(L @ srcs)
    K_hi, lam_hi = FW.mne_operator(L, snr=1e6)
    np.testing.assert_allclose(K_hi @ x, srcs, atol=1e-6)
    # and the shrinkage at SNR 3 is exactly the filter factors
    np.testing.assert_allclose(K @ x, res @ srcs, atol=1e-9)


def test_template_lead_field_is_the_development_head(hopf):
    n = hopf.ts.shape[0]
    sens = FW.sensor_positions(64)
    exact = FW.TemplateSpec(electrode_jitter_sd=0.0, conduction_width=0.5)
    L0, info0 = FW.template_lead_field(hopf.meta, n, sens, exact)
    dev = FW.eeg_forward(hopf, seed=3).oracle["lead_field"]
    # same source positions, signs and width (electrodes re-projected only)
    np.testing.assert_allclose(L0, dev, rtol=1e-12, atol=1e-14)
    L, info = FW.template_lead_field(hopf.meta, n, sens)
    assert info["jitter_stream"] == [20260928, FW.TEMPLATE_STREAM_KEY]
    assert info["conduction_width"] == 0.4 and info["electrode_jitter_sd"] == 0.05
    assert not np.allclose(L, dev)
    L_again, _ = FW.template_lead_field(hopf.meta, n, sens)
    assert np.array_equal(L, L_again)  # one fixed template, whatever the task seed
    assert FW.TEMPLATE_STREAM_KEY not in S.STREAM_KEYS.values()


def test_mne_template_view(hopf, held_out_views):
    est = held_out_views["mne_template"]
    eeg = held_out_views["eeg64"]
    L_t, _ = FW.template_lead_field(hopf.meta, hopf.ts.shape[0],
                                    eeg.oracle["sensor_positions"])
    K, lam = FW.mne_operator(L_t, 3.0)
    assert np.array_equal(est.ts, K @ eeg.ts)
    assert est.ts.shape == hopf.ts.shape
    inv = est.meta["inverse"]
    assert inv["lambda"] == lam and inv["snr"] == 3.0
    assert inv["template"]["leadfield_seed"] == 20260928
    # source-space meta (anatomical hubs, modules) and the forward provenance
    assert est.meta["workspace_nodes"] == hopf.meta["workspace_nodes"]
    assert est.meta["modules"] == hopf.meta["modules"]
    assert est.meta["substrate"] == "eeg_like_forward"
    assert est.meta["forward"]["leadfield_seed"] == 20261001
    assert est.meta["forward_view"]["observation"] == "source_estimate"
    assert np.array_equal(est.oracle["inverse_operator"], K)


# --------------------------------------------------------------------------
# rank of the average-referenced montage
# --------------------------------------------------------------------------
def test_average_reference_operator_rank():
    for n in (4, 32, 64):
        H = FW.average_reference_operator(n)
        assert FW.montage_rank(H) == n - 1
        np.testing.assert_allclose(H @ np.ones(n), 0.0, atol=1e-12)
        np.testing.assert_allclose(H @ H, H, atol=1e-12)
    x = np.random.default_rng(0).normal(size=(10, 200))
    np.testing.assert_allclose(FW.average_reference(x),
                               FW.average_reference_operator(10) @ x, atol=1e-12)
    with pytest.raises(FW.ForwardError):
        FW.average_reference_operator(1)


def test_rank_of_the_recorded_montages(held_out_views):
    assert FW.montage_rank(held_out_views["eeg64"].ts) == 63
    assert FW.montage_rank(held_out_views["eeglow"].ts) == 31
    assert FW.montage_rank(held_out_views["eeg64_noref"].ts) == 64
    L = held_out_views["eeg64"].oracle["lead_field"]
    assert FW.montage_rank(FW.average_reference(L)) == 63  # 76 sources, 64 sensors


def test_quadrant_means_are_rank_deficient_and_clusters_are_not(held_out_views):
    eeg = held_out_views["eeg64"]
    quad = eeg.meta["forward_view"]["iim_quadrants"]
    clusters = eeg.meta["iim_macro_nodes"]
    assert eeg.meta["iim_grain"] == "electrode_clusters_rank_safe"
    assert clusters == eeg.meta["forward_view"]["iim_clusters"]
    # quadrants partition the montage: their means sum to zero (rank 3)
    assert sorted(i for v in quad.values() for i in v) == list(range(64))
    assert FW.montage_rank(FW.cluster_means(eeg.ts, quad)) == 3
    noref = held_out_views["eeg64_noref"].ts
    assert FW.montage_rank(FW.cluster_means(noref, quad)) == 4
    # rank-safe clusters: half of each quadrant, disjoint, never the montage
    members = [i for v in clusters.values() for i in v]
    assert len(members) == len(set(members)) == 32
    for name, idx in clusters.items():
        assert set(idx) < set(quad[name]) and len(idx) == 8
    assert FW.montage_rank(FW.cluster_means(eeg.ts, clusters)) == 4
    low = held_out_views["eeglow"]
    lc = low.meta["iim_macro_nodes"]
    assert FW.montage_rank(FW.cluster_means(low.ts, lc)) == 4
    # Macro IIM default size k = min(8, floor(n / 8)): 4 of the 32 electrodes
    assert [len(v) for v in lc.values()] == [4, 4, 4, 4]
    low_quad = low.meta["forward_view"]["iim_quadrants"]
    for name, idx in lc.items():
        assert set(idx) < set(low_quad[name])


@pytest.mark.parametrize("n, k", [(8, 1), (16, 2), (24, 3), (32, 4), (63, 7),
                                  (64, 8), (128, 8)])
def test_cluster_size_is_the_iim_macro_default(n, k):
    assert FW.cluster_size(n) == k == min(8, n // 8)


def test_rank_safe_clusters_on_other_montages():
    for n in (32, 128):
        pos = FW.sensor_positions(n)
        cl = FW.rank_safe_clusters(pos)
        assert [len(v) for v in cl.values()] == [FW.cluster_size(n)] * 4
        quad = FW.quadrant_clusters(pos)
        for name, idx in cl.items():
            # the electrodes nearest the quadrant centroid
            cen = pos[quad[name]].mean(axis=0)
            d_in = np.linalg.norm(pos[idx] - cen, axis=1).max()
            rest = sorted(set(quad[name]) - set(idx))
            assert d_in <= np.linalg.norm(pos[rest] - cen, axis=1).min()
        x = FW.average_reference(np.random.default_rng(n).normal(size=(n, 400)))
        assert FW.montage_rank(FW.cluster_means(x, cl)) == 4
    # an irregular montage caps k at its smallest quadrant
    pos = FW.sensor_positions(64)[list(FW.low_density_subset(16))]
    small = min(len(v) for v in FW.quadrant_clusters(pos).values())
    cl = FW.rank_safe_clusters(pos)
    assert [len(v) for v in cl.values()] == [min(2, small)] * 4
    with pytest.raises(FW.ForwardError):
        FW.rank_safe_clusters(FW.sensor_positions(7))


# --------------------------------------------------------------------------
# the coupling time scale survives every view (NAS sampling rule)
# --------------------------------------------------------------------------
def test_views_keep_the_source_time_scale(held_out_views):
    for name, v in held_out_views.items():
        tau = DI.coupling_timescale(v).tau_c_sec
        assert tau == pytest.approx(0.1)
        resolved = DI.sampling_resolved(tau, v.dt)
        assert resolved is (name != "bold"), name


# --------------------------------------------------------------------------
# forward arms: plan, seeds, regimes
# --------------------------------------------------------------------------
def test_confirmatory_plan_sizes_and_seeds():
    tasks = D.build_tasks(D.CONFIRMATORY)
    by_arm = {a: [t for t in tasks if t.arm == a] for a in D.ARMS}
    assert {a: len(v) for a, v in by_arm.items()} == {
        D.ARM_HOPF: 371, D.ARM_A_EEG: 332, D.ARM_A_BOLD: 271}
    seeds = {(t.arm, t.condition): [] for t in tasks}
    for t in tasks:
        seeds[(t.arm, t.condition)].append(t.seed)
        assert t.regime == "held_out" and t.split == S.CONFIRMATORY
    for (arm, cond), s in seeds.items():
        assert s == list(range(20000, 20000 + len(s)))
    n = {k: len(v) for k, v in seeds.items()}
    gn = D.g_nom_label()
    assert gn == "hopf_G1.14286"
    assert n[(D.ARM_HOPF, "hopf_G0")] == 61 and n[(D.ARM_HOPF, gn)] == 150
    assert n[(D.ARM_HOPF, "hopf_lesion_hub")] == 20
    assert n[(D.ARM_HOPF, "hopf_lesion_random_matched_hub")] == 20
    assert sum(1 for (a, c) in n if a == D.ARM_HOPF and c.startswith("hopf_G")) == 8
    assert n[(D.ARM_A_EEG, "PC_nominal")] == 150
    assert n[(D.ARM_A_EEG, "N_ar1")] == 61
    assert n[(D.ARM_A_EEG, "W_PDI_no_multistability")] == 61
    assert n[(D.ARM_A_BOLD, "PC_nominal")] == 150
    assert (D.ARM_A_BOLD, "N_ar1") not in n
    assert len({t.task_id for t in tasks}) == len(tasks)
    json.dumps([t.to_config() for t in tasks[:3]])  # JSON-able record config


def test_hopf_conditions_are_the_g_sweep_and_g_nom_lesions():
    from impact_pipeline.bench import catalogue as CAT

    conds = D.conditions(D.ARM_HOPF)
    doses = [c.dose for c in conds if c.spec["lesion"] == "none"]
    assert doses == wb.g_sweep_levels()
    lesions = [c for c in conds if c.spec["lesion"] != "none"]
    assert [c.spec["G"] for c in lesions] == [CAT.g_nom()] * 2
    assert [c.spec["lesion"] for c in lesions] == ["hub", "random"]
    assert lesions[1].spec["n_lesion_edges"] > 0
    assert [c.label for c in conds if c.anchor] == [D.g_nom_label()]


def test_development_purposes():
    dry = D.build_tasks(D.DRY_RUN)
    assert {t.regime for t in dry} == {"development"}
    assert all(t.seed in range(384, 400) for t in dry)
    by = {}
    for t in dry:
        by.setdefault((t.arm, t.condition), []).append(t.seed)
    assert len(by[(D.ARM_HOPF, "hopf_G0")]) == 10        # ceil(0.15 x 61)
    assert len(by[(D.ARM_HOPF, D.g_nom_label())]) == 16  # capped by the block
    assert len(by[(D.ARM_A_EEG, "PC_nominal_K1")]) == 3  # ceil(0.15 x 20)
    assert all(t.seed in range(804, 820) for t in D.build_tasks(D.DEV_REGIME))
    for purpose, regime in ((D.REFERENCE, "held_out"),
                            (D.REFERENCE_DEVELOPMENT, "development"),
                            (D.SMOKE, "held_out")):
        ts = D.build_tasks(purpose)
        assert {t.regime for t in ts} == {regime}
        assert {t.condition for t in ts} == {D.g_nom_label(), "PC_nominal"}
        block = range(900, 940) if purpose != D.SMOKE else range(980, 985)
        assert all(t.seed in block for t in ts)
        assert len(ts) == 3 * len(block)


def _task(**kw):
    base = D.build_tasks(D.REFERENCE, arms=[D.ARM_HOPF])[0]
    d = base.to_dict()
    d.update(kw)
    d["views"] = tuple(d["views"])
    return D.ForwardTask(**d)


@pytest.mark.parametrize("kw, msg", [
    ({"purpose": D.DRY_RUN, "seed": 384}, "regime"),          # held-out on a dry run
    ({"condition": "hopf_G0"}, "anchor conditions"),          # held-out non-anchor
    ({"purpose": D.CONFIRMATORY}, "confirmatory"),            # confirmatory on dev seed
    ({"purpose": D.CONFIRMATORY, "seed": 20000, "regime": "development"}, "regime"),
    ({"seed": 950}, "outside"),                               # not the reference block
    ({"seed": 15000}, "unassigned"),
])
def test_regime_policy_refuses(kw, msg):
    with pytest.raises((D.ForwardDesignError, S.SeedPolicyError), match=msg):
        D.check_regime_policy([_task(**kw)])


def test_curtailment_order_of_the_bold_on_runs():
    groups = D.curtailment_order(D.build_tasks(D.CONFIRMATORY, arms=[D.ARM_A_BOLD]))
    (ts,) = groups.values()
    assert list(groups) == [D.CURTAIL_GROUP_A_BOLD]
    assert len(ts) == 150 + 20 + 20
    assert [t.condition for t in ts[:3]] == ["PC_nominal", "PC_nominal_K2",
                                             "PC_nominal_K3"]
    assert [t.seed for t in ts] == sorted(t.seed for t in ts)
    assert {t.condition for t in ts[60:]} == {"PC_nominal"}


def test_only_runs_no_other_criterion_reads_are_curtailable():
    tasks = D.build_tasks(D.CONFIRMATORY)
    cut = [t for t in tasks if D.curtailable(t)]
    # the PC_nominal runs of the BOLD arm beyond the 20 seeds of the K sweep
    assert {(t.arm, t.condition) for t in cut} == {(D.ARM_A_BOLD, "PC_nominal")}
    assert sorted(t.seed for t in cut) == list(range(20020, 20150))
    dry = [t for t in D.build_tasks(D.DRY_RUN) if D.curtailable(t)]
    assert sorted(t.seed for t in dry) == list(range(387, 400))  # K sweep: 3 seeds
    assert not any(D.curtailable(t) for t in D.build_tasks(D.REFERENCE))


def _pdi_bold_status(task, view):
    """PDI on the BOLD arm: specific at the null, monotone in K, concordant
    with the source, one false ABSENT at the fifth on-run."""
    k = {"PC_nominal": 6, "PC_nominal_K1": 1, "PC_nominal_K2": 2,
         "PC_nominal_K3": 3}.get(task.condition)
    if k is None:
        return "UNDEFINED", 0.0
    c = (k - 1) / 5.0 + 0.001 * (task.seed % 7)
    if view == "bold" and task.condition == "PC_nominal_K2" and task.seed == 20001:
        return "ABSENT", c
    return ("PRESENT" if k >= 2 else "UNDEFINED"), c


def test_a_curtailed_bold_arm_keeps_the_present_criteria():
    """The runner contract: stop FMabs at the first false ABSENT and skip only
    the curtailable runs; FMa, FMb1 and FMd still see every planned run, so
    the curtailed PRESENT decision equals the full-sample one."""
    (design,) = [d for d in D.admission_designs() if d.arm == D.ARM_A_BOLD]
    plan = D.build_tasks(D.CONFIRMATORY, arms=[D.ARM_A_BOLD])
    (order,) = D.curtailment_order(plan).values()
    cur = RV.Curtailment(design.n_planned_on, 0.02)
    ran = [t for t in plan if not t.curtail_group]
    for t in order:
        if cur.stopped and D.curtailable(t):
            continue
        ran.append(t)
        if not cur.stopped:
            cur.update(_pdi_bold_status(t, "bold")[0] == "ABSENT")
    assert cur.stopped and cur.stop_index == 4
    assert len(ran) == len(plan) - 130

    def runs(tasks, view):
        out = []
        for t in tasks:
            st, c = _pdi_bold_status(t, view)
            why = REASONS.INCONCLUSIVE if st == "UNDEFINED" else None
            out.append(RV.AdmissionRun("PDI", view, t.condition, t.seed, st, why, c,
                                       t.dose))
        return out

    full, cut = (RV.evaluate_view(design, "bold", runs(ts, "bold"), stage="bold",
                                  anchor=True, source_runs=runs(ts, "source"))
                 for ts in (plan, ran))
    for res in (full, cut):
        assert res.admitted_for_present == "yes" and res.admitted_for_absent == "no"
        assert res.criteria["FMb1"]["seeds"] == 20 and res.criteria["FMd"]["n"] == 20
        assert res.criteria["FMa"]["level"] == pytest.approx(0.025)
    assert cut.criteria["FMabs"]["stopped"] and cut.criteria["FMabs"]["n"] == 5
    assert cut.criteria["FMb1"] == full.criteria["FMb1"]
    assert cut.criteria["FMd"] == full.criteria["FMd"]


# --------------------------------------------------------------------------
# realisation
# --------------------------------------------------------------------------
def test_truncated_hopf_run_equals_the_shorter_run():
    long = wb.simulate_whole_brain(SHORT_WB.replace(duration_sec=6.0), seed=8)
    short = wb.simulate_whole_brain(SHORT_WB.replace(duration_sec=2.0), seed=8)
    assert_same_system(D.truncate_whole_brain(long, 2.0), short)
    with pytest.raises(D.ForwardDesignError):
        D.truncate_whole_brain(long, 7.0)
    with pytest.raises(D.ForwardDesignError):
        D.truncate_whole_brain(g.simulate_family_a(None, SMALL_A, 1), 1.0)


def test_hopf_realisation_uses_one_run_for_every_view(monkeypatch):
    calls = []
    real_sim = wb.simulate_whole_brain

    def short_sim(cfg, seed):
        calls.append((cfg.G, cfg.lesion, cfg.duration_sec))
        # a shorter stand-in run that keeps the 60:600 split of the arm
        return real_sim(cfg.replace(duration_sec=cfg.duration_sec / 100.0,
                                    transient_sec=0.5), seed)

    monkeypatch.setattr(wb, "simulate_whole_brain", short_sim)
    monkeypatch.setattr(D, "HOPF_SOURCE_DURATION_S", 0.6)
    (task,) = [t for t in D.build_tasks(D.DRY_RUN, arms=[D.ARM_HOPF])
               if t.condition == "hopf_lesion_hub"][:1]
    real = D.realise(task)
    assert calls == [(pytest.approx(1.142857), "hub", D.HOPF_BOLD_DURATION_S)]
    assert set(real.views) == set(D.VIEWS_OF_ARM[D.ARM_HOPF])
    assert real.source.n_time == 150  # 0.6 s at 250 Hz
    for name in ("source", "eeg64", "eeglow", "mne_template", "eeg64_noref"):
        assert real.views[name].n_time == 150, name
    assert real.views["bold"].n_time == 3  # the whole 6-s run at TR 2 s
    assert FW.regime_keys(real.views["eeg64"])["leadfield_seed"] == 20260928


def test_family_a_realisation_and_the_slow_context_preset():
    eeg_t = [t for t in D.build_tasks(D.DRY_RUN, arms=[D.ARM_A_EEG])
             if t.condition == "PC_nominal_K2"][0]
    real = D.realise(eeg_t)
    assert set(real.views) == {"source", "eeg64", "eeglow"}
    assert real.source.meta["knobs"]["K"] == 2
    assert FW.regime_keys(real.views["eeg64"])["sensor_filter"] == "none"
    assert FW.regime_keys(real.views["eeg64"])["inputs_declared"] == "complete"
    assert FW.regime_keys(real.views["source"])["fs"] == 20.0
    bold_t = [t for t in D.build_tasks(D.DRY_RUN, arms=[D.ARM_A_BOLD])
              if t.condition == "W_PDI_no_multistability"][0]
    real = D.realise(bold_t)
    cfg = real.source.meta["config"]
    assert (cfg["ctx_dwell"], cfg["trial_sec"], cfg["iti_jitter_sec"]) == (
        [30.0, 60.0], 12.0, 2.0)
    assert real.source.meta["witness_id"] == "W_PDI_no_multistability"
    assert real.views["bold"].meta["forward"]["drive"] == "signal"
    assert real.views["bold"].dt == 2.0


def test_the_family_a_source_is_the_bench_system():
    """A forward family-A run simulates the joint bench's own system: the
    same recording and the same configuration digest as the catalogue
    witness or the sweep agent of that seed, so the duplicate detector
    lists a shared seed as one configuration under two designs."""
    from impact_pipeline.bench import catalogue as CAT
    from impact_pipeline.bench import runner as RB
    from impact_pipeline.bench import run_hygiene as PV

    ref = D.build_tasks(D.REFERENCE_DEVELOPMENT, arms=[D.ARM_A_EEG])[0]
    assert (ref.condition, ref.seed) == ("PC_nominal", 900)
    src = D._family_a_source(ref)
    bench = CAT.build_system("PC_nominal", 900, "A")
    assert PV.array_sha256(src.ts) == PV.array_sha256(bench.ts)
    assert RB.system_config_digest(src) == RB.system_config_digest(bench)
    k2 = [t for t in D.build_tasks(D.DRY_RUN, arms=[D.ARM_A_EEG])
          if t.condition == "PC_nominal_K2"][0]
    agent = RB.build_agent_system(types_task(k2.seed, {"K": 2}))
    src = D._family_a_source(k2)
    assert PV.array_sha256(src.ts) == PV.array_sha256(agent.ts)
    assert RB.system_config_digest(src) == RB.system_config_digest(agent)


def types_task(seed, knobs):
    """A stand-in runner task of the family-A agent at ``knobs``."""
    import types

    from impact_pipeline.bench.generators import NOMINAL_KNOBS

    kn = NOMINAL_KNOBS.replace(**knobs).to_dict()
    return types.SimpleNamespace(seed=seed, replicate=0,
                                 params={"family": "A", "knobs": kn})


# --------------------------------------------------------------------------
# scorings and admission designs
# --------------------------------------------------------------------------
def test_scoring_plan():
    hopf = D.scoring_plan(D.build_tasks(D.DRY_RUN, arms=[D.ARM_HOPF])[0])
    by = {s["scoring_id"]: s for s in hopf}
    assert by["eeg64:primary"]["options"]["NAS"]["observation_gate"] == "admission_run"
    # the sensor pipeline (rank condition, ZCA) on every mixed IIM view, the
    # primary cut only
    assert by["mne_template:primary"]["options"]["IIM"] == {
        "preprocess": "zca", "observation_gate": "admission_run",
        "report_cut_modes": []}
    for v in ("eeg64", "eeg64_noref", "eeglow"):
        assert by[f"{v}:primary"]["options"]["IIM"]["preprocess"] == "zca"
        assert by[f"{v}:primary"]["options"]["IIM"]["macro_nodes"] == (
            "rank_safe_clusters")
    for v in ("source", "bold"):
        assert by[f"{v}:primary"]["options"] == {"IIM": {"report_cut_modes": []}}
    assert by["eeg64_noref:primary"]["principles"] == ["IIM"]
    assert set(by["eeg64_noref:primary"]["options"]) == {"IIM"}
    comps = [s for s in hopf if s["estimator_form"] == D.IIM_QUADRANTS]
    assert {s["view"] for s in comps} == {"eeg64", "eeg64_noref"}
    assert all(s["roles"] == {"IIM": "comparator"} for s in comps)
    assert all(s["declaration_id"] == "none" for s in hopf)
    fa = D.scoring_plan(D.build_tasks(D.DRY_RUN, arms=[D.ARM_A_EEG])[0])
    byf = {s["view"]: s for s in fa}
    # protocol vocabulary, as in the template protocol
    template = json.loads((REPO_ROOT / "protocols/bench/template.json")
                          .read_text())
    assert "pdi_bearer" in template["estimators"]["PDI"]
    assert byf["eeg64"]["options"]["PDI"] == {"pdi_bearer": "full"}
    assert "PDI" not in byf["source"]["options"]
    assert byf["source"]["roles"]["PDI"] == "source_contrast"
    assert byf["eeg64"]["roles"] == {"PDI": "admission", "IIM": "descriptive"}
    assert all(s["declaration_id"] == "R" for s in fa)
    assert D.entry_grain("IIM", "eeg64") == "electrode_clusters_rank_safe"
    assert D.entry_grain("IIM", "mne_template") == "*" == D.entry_grain("NAS", "eeg64")


def test_admission_designs():
    designs = {(d.principle, d.arm): d for d in D.admission_designs()}
    assert set(designs) == {("NAS", D.ARM_HOPF), ("IIM", D.ARM_HOPF),
                            ("PDI", D.ARM_A_EEG), ("PDI", D.ARM_A_BOLD)}
    nas, iim = designs[("NAS", D.ARM_HOPF)], designs[("IIM", D.ARM_HOPF)]
    assert nas.views == ("source", "eeg64", "eeglow", "mne_template", "bold")
    assert iim.views == ("source", "eeg64", "eeg64_noref", "eeglow", "mne_template",
                         "bold")
    assert nas.lesion == ("hopf_lesion_hub", "hopf_lesion_random_matched_hub")
    assert iim.lesion is None and iim.fma_m == 4 and nas.fma_m is None
    assert nas.null_conditions == ("hopf_G0",) and nas.on_conditions == (
        D.g_nom_label(),)
    assert nas.concordance == (D.g_nom_label(), "hopf_G0")
    assert sorted(nas.dose_conditions.values()) == wb.g_sweep_levels()
    assert nas.n_planned_on == 150
    eeg, bold = designs[("PDI", D.ARM_A_EEG)], designs[("PDI", D.ARM_A_BOLD)]
    assert eeg.null_conditions == ("W_PDI_no_multistability", "N_ar1")
    assert bold.null_conditions == ("W_PDI_no_multistability",)
    # FMa at 0.05 / 2 in every PDI view, BOLD included
    assert bold.fma_m == 2 and eeg.fma_m is None and len(eeg.null_conditions) == 2
    assert eeg.on_conditions == ("PC_nominal", "PC_nominal_K2", "PC_nominal_K3")
    assert dict(eeg.dose_conditions) == {"PC_nominal_K1": 1.0, "PC_nominal_K2": 2.0,
                                         "PC_nominal_K3": 3.0, "PC_nominal": 6.0}
    assert eeg.n_planned_on == bold.n_planned_on == 190
    # FMabs needs >= 149 on-runs with 0 events: the plan provides them
    assert min(eeg.n_planned_on, nas.n_planned_on) >= RV.min_runs_for_demonstration(
        0.02, 0.05)
    # FMa: 61 seeds per null class suffice at the declared levels
    for d in designs.values():
        m = d.fma_m or len(d.null_conditions)
        need = RV.min_runs_for_demonstration(0.07, 0.05 / m)
        for c in d.null_conditions:
            assert D.condition(d.arm, c).n_confirmatory >= need
    assert {d.n_planned_on for d in D.admission_designs(D.DRY_RUN)} == {16, 22}


def test_a_run_carries_what_some_criterion_reads():
    hopf = D.build_tasks(D.CONFIRMATORY, arms=[D.ARM_HOPF])
    by_cond = {}
    for t in hopf:
        by_cond.setdefault(t.condition, t)
    # on the admission runs the quadrant comparator only at G = 0
    for cond, t in by_cond.items():
        forms = {(s["view"], s["estimator_form"]) for s in D.scoring_plan(t)}
        comp = {f for f in forms if f[1] == D.IIM_QUADRANTS}
        assert comp == ({("eeg64", D.IIM_QUADRANTS),
                         ("eeg64_noref", D.IIM_QUADRANTS)}
                        if cond == "hopf_G0" else set()), cond
    # only NAS has a lesion criterion (FMb2): the lesion runs score NAS alone
    for cond in ("hopf_lesion_hub", "hopf_lesion_random_matched_hub"):
        plan = D.scoring_plan(by_cond[cond])
        assert {p for s in plan for p in s["principles"]} == {"NAS"}
        assert [s["view"] for s in plan] == ["source", "eeg64", "eeglow",
                                             "mne_template", "bold"]
        assert all(set(s["options"]) <= {"NAS"} for s in plan)
    # the family-A source view (no admission view) only where FMd reads it:
    # PC_nominal and K = 1 on the seeds they share
    for arm in (D.ARM_A_EEG, D.ARM_A_BOLD):
        plan = D.build_tasks(D.CONFIRMATORY, arms=[arm])
        src = sorted((t.condition, t.seed) for t in plan
                     if any(s["view"] == "source" for s in D.scoring_plan(t)))
        want = sorted((c, s) for c in ("PC_nominal", "PC_nominal_K1")
                      for s in range(20000, 20020))
        assert src == want, arm
        # every forward view on every run
        for t in plan:
            views = {s["view"] for s in D.scoring_plan(t)}
            assert views >= set(D.VIEWS_OF_ARM[arm]) - {"source"}
    # the anchor runs keep every view (their anchors are computed there)
    for t in D.build_tasks(D.REFERENCE_DEVELOPMENT, arms=[D.ARM_A_EEG])[:3]:
        assert {s["view"] for s in D.scoring_plan(t)} == {"source", "eeg64", "eeglow"}
    # IIM on the family-A arms is descriptive: value and null, no SE; every
    # forward protocol scores the primary cut only
    for key, spec in D.protocol_options().items():
        iim = spec["estimators"]["IIM"]
        assert iim["report_cut_modes"] == [], key
        if key.startswith("fwdA"):
            assert iim["se_method"] is None, key
        else:
            assert "se_method" not in iim, key


QUADRANTS = {("eeg64", D.IIM_QUADRANTS), ("eeg64_noref", D.IIM_QUADRANTS)}


@pytest.mark.parametrize("purpose, on_g_nom", [
    (D.REFERENCE, True), (D.REPLICATION, True), (D.SMOKE, True),
    # IIM on the sensor views above G = 0 is held out
    (D.REFERENCE_DEVELOPMENT, False),
    (D.CONFIRMATORY, False), (D.DRY_RUN, False), (D.DEV_REGIME, False),
])
def test_the_quadrant_comparator_on_the_anchor_condition(purpose, on_g_nom):
    """The quadrant comparator on the Hopf G_nom anchor condition of the
    held-out-regime anchor runs (own validity-only anchors of its two
    protocols), never at the development regime; at G = 0 on
    every purpose that runs it."""
    tasks = D.build_tasks(purpose, arms=[D.ARM_HOPF])
    gn = [t for t in tasks if t.condition == D.g_nom_label()]
    assert gn
    for t in gn:
        forms = {(s["view"], s["estimator_form"]) for s in D.scoring_plan(t)}
        assert D.reads_comparator(t) is on_g_nom
        assert (forms >= QUADRANTS) if on_g_nom else not (forms & QUADRANTS)
        if on_g_nom:
            quad = [s for s in D.scoring_plan(t)
                    if s["estimator_form"] == D.IIM_QUADRANTS]
            assert all(s["principles"] == ["IIM"] and s["roles"] == {
                "IIM": "comparator"} for s in quad)
            assert all(s["options"]["IIM"]["macro_nodes"] == "quadrants"
                       for s in quad)
    for t in tasks:
        if t.condition == "hopf_G0":
            forms = {(s["view"], s["estimator_form"]) for s in D.scoring_plan(t)}
            assert forms >= QUADRANTS
        elif t.condition != D.g_nom_label():
            assert not D.reads_comparator(t)
    # the family-A arms have no comparator form at all
    for arm in (D.ARM_A_EEG, D.ARM_A_BOLD):
        for t in D.build_tasks(purpose, arms=[arm])[:3]:
            assert not any(s["estimator_form"] == D.IIM_QUADRANTS
                           for s in D.scoring_plan(t))


def test_the_anchor_runners_carry_the_comparator_protocols():
    from impact_pipeline.bench import designs as DV

    rep = DV.get_design(D.ANCHOR_REPLICATION_DESIGN)
    keys = {s.protocol_key for t in rep.tasks(DV.CONFIRMATORY) for s in t.scorings}
    assert {"hopf-eeg64+iim_quadrants",
            "hopf-eeg64_noref+iim_quadrants"} <= keys
    dev = {s.protocol_key for t in rep.tasks(DV.DEVELOPMENT) for s in t.scorings}
    assert not any("iim_quadrants" in k for k in dev)
    # the comparator's options on the anchor runs are those of its protocol
    opts = D.protocol_options()["hopf-eeg64+iim_quadrants"]["estimators"]
    t = rep.tasks(DV.CONFIRMATORY)[0]
    ft = D.forward_task_of(t)
    (q,) = [s for s in D.scoring_plan(ft) if s["scoring_id"] == "eeg64:iim_quadrants"]
    assert q["options"] == opts


def test_the_forward_replication_block_is_extended_per_arm(monkeypatch):
    from impact_pipeline.bench import designs as DV
    from impact_pipeline.bench.designs import anchors as AN

    # the Hopf arm is extended
    assert D.replication_seeds(D.ARM_HOPF) == range(20900, 20940)
    assert all(D.replication_seeds(a) == D.REPLICATION_SEEDS
               for a in D.ARMS if a != D.ARM_HOPF)
    assert D.DOCUMENT_TASKS[D.ANCHOR_REPLICATION_DESIGN] == 40 + 2 * 20
    pending = AN.CALIBRATION_PENDING["forward_replication_extended"]
    monkeypatch.setitem(pending, "value", dict(pending["value"], hopf=True))
    assert D.replication_seeds(D.ARM_HOPF) == range(20900, 20940)
    assert D.replication_seeds(D.ARM_A_EEG) == range(20900, 20920)
    rep = D.build_tasks(D.REPLICATION)
    by_arm = {a: sorted(t.seed for t in rep if t.arm == a) for a in D.ARMS}
    assert by_arm == {D.ARM_HOPF: list(range(20900, 20940)),
                      D.ARM_A_EEG: list(range(20900, 20920)),
                      D.ARM_A_BOLD: list(range(20900, 20920))}
    assert all(t.condition in (D.g_nom_label(), "PC_nominal") for t in rep)
    # the regime policy and the task builder follow the declared block
    t = D.forward_task(D.ARM_HOPF, D.g_nom_label(), 20935, D.REPLICATION)
    assert t.regime == "held_out" and t.split == S.CONFIRMATORY
    with pytest.raises(D.ForwardDesignError, match="not a replication seed"):
        D.forward_task(D.ARM_A_EEG, "PC_nominal", 20935, D.REPLICATION)
    bad = D.ForwardTask(**{**t.to_dict(), "arm": D.ARM_A_EEG,
                           "condition": "PC_nominal",
                           "views": D.VIEWS_OF_ARM[D.ARM_A_EEG]})
    with pytest.raises(D.ForwardDesignError, match="replication block"):
        D.check_regime_policy([bad])
    seeds = {(t.tags["arm"], t.seed) for t in DV.get_design(
        D.ANCHOR_REPLICATION_DESIGN).tasks(DV.CONFIRMATORY)}
    assert len(seeds) == 40 + 20 + 20
    with pytest.raises(D.ForwardDesignError, match="arm"):
        D.replication_seeds("no_such_arm")


def test_runner_tasks_of_the_forward_arms():
    from impact_pipeline.bench import designs as DV

    assert {d.name for d in D.DESIGNS} == {
        "whole_brain", "forward_family_a", "forward_family_a_bold",
        D.ANCHOR_REPLICATION_DESIGN}
    for d in D.DESIGNS:
        assert d.expected_tasks[DV.CONFIRMATORY] == D.DOCUMENT_TASKS[d.name]
        assert len(d.tasks(DV.CONFIRMATORY)) == D.DOCUMENT_TASKS[d.name]
    for arm in D.ARMS:
        name = D.RECORD_DESIGN_OF_ARM[arm]
        dev = DV.get_design(name).tasks(DV.DEVELOPMENT)
        plan = D.build_tasks(D.DRY_RUN, arms=[arm])
        assert [t.task_id for t in dev] == [t.task_id for t in plan]
        for t in dev:
            ft = D.forward_task_of(t)
            assert (t.design, t.family, t.system, t.seed) == (
                name, ft.family, ft.condition, ft.seed)
            assert not t.has_held_out and t.tags["regime"] == "development"
            assert [s.scoring_id for s in t.scorings] == [
                s["scoring_id"] for s in D.scoring_plan(ft)]
            for s in t.scorings:
                assert s.protocol_key == D.protocol_key_of(arm, s.view,
                                                           s.estimator_form)
                assert not s.verdict
        dev_regime = DV.get_design(name).tasks(DV.DEVELOPMENT, purpose=D.DEV_REGIME)
        assert {t.seed for t in dev_regime} <= set(range(804, 820))
        smoke = DV.get_design(name).tasks(DV.DEVELOPMENT, purpose=D.SMOKE)
        assert all(t.smoke and t.has_held_out for t in smoke)
        conf = DV.get_design(name).tasks(DV.CONFIRMATORY)
        assert all(t.has_held_out for t in conf)  # the held-out regime
    rep = DV.get_design(D.ANCHOR_REPLICATION_DESIGN)
    conf = rep.tasks(DV.CONFIRMATORY)
    by_arm = {}
    for t in conf:
        by_arm.setdefault(t.tags["arm"], set()).add(t.seed)
    assert by_arm == {a: set(D.replication_seeds(a)) for a in D.ARMS}
    assert by_arm[D.ARM_HOPF] == set(range(20900, 20940))  # extended
    assert {(t.tags["arm"], t.system) for t in conf} == {
        (D.ARM_HOPF, D.g_nom_label()), (D.ARM_A_EEG, "PC_nominal"),
        (D.ARM_A_BOLD, "PC_nominal")}
    assert {t.tags["regime"] for t in conf} == {"held_out"}
    dev = rep.tasks(DV.DEVELOPMENT)
    assert {t.seed for t in dev} == set(range(900, 940))
    assert {t.tags["regime"] for t in dev} == {"development"} and not any(
        t.has_held_out for t in dev)
    # the held-out regime on the reference block is a held-out condition
    assert all(t.has_held_out for t in rep.tasks(DV.DEVELOPMENT, purpose=D.REFERENCE))
    with pytest.raises(D.ForwardDesignError, match="purpose"):
        rep.tasks(DV.DEVELOPMENT, purpose=D.DRY_RUN)
    with pytest.raises(D.ForwardDesignError, match="purpose"):
        DV.get_design("whole_brain").tasks(DV.DEVELOPMENT, purpose=D.REFERENCE)
    with pytest.raises(D.ForwardDesignError, match="no task"):
        DV.get_design("whole_brain").tasks(DV.DEVELOPMENT, seeds=[320])
    with pytest.raises(D.ForwardDesignError, match="unknown conditions"):
        DV.get_design("whole_brain").tasks(DV.DEVELOPMENT, systems=["PC_nominal"])
    with pytest.raises(D.ForwardDesignError, match="not a dry_run seed"):
        D.forward_task(D.ARM_A_EEG, "PC_nominal_K1", 390, D.DRY_RUN)
    with pytest.raises(D.ForwardDesignError, match="replication block"):
        D.check_regime_policy([D.ForwardTask(**{
            **conf[0].params, "task_id": "x", "seed": 20950, "regime": "held_out",
            "views": D.VIEWS_OF_ARM[D.ARM_HOPF], "dose": None,
            "curtail_group": None})])


def _bold_record(ft, absent=False, error=False):
    """A minimal record of a BOLD-arm run: PDI on the BOLD view."""
    from impact_pipeline.bench import records as REC

    comp = REC.ComponentRecord(
        principle="PDI", status="ABSENT" if absent else "PRESENT", reason=None,
        estimator_version="pdi-declared-2026.10", declaration_id="R",
        observation_stage="bold", protocol_id="fwdA_bold-bold", protocol_hash="a" * 64,
        c=0.0 if absent else 1.0)
    sc = REC.ScoringRecord(scoring_id="bold:primary", declaration_id="R",
                           observation_stage="bold", view="bold",
                           estimator_form="primary", protocol_id="fwdA_bold-bold",
                           protocol_hash="a" * 64, components={"PDI": comp})
    return REC.TaskRecord(
        task_id=ft.task_id, design="forward_family_a_bold", family="A",
        system=ft.condition, seed=ft.seed,
        generator_version="mpc-bench-generators/1.0.0",
        status="error" if error else "ok", error="boom" if error else None,
        scorings=() if error else (sc,),
        simulation=None if error else {"ts_sha256": "b" * 64, "n_nodes": 30,
                                       "n_time": 100, "dt": 2.0})


def test_the_runner_curtailment_follows_the_seed_order():
    from impact_pipeline.bench import designs as DV

    tasks = DV.get_design("forward_family_a_bold").tasks(DV.DEVELOPMENT)
    ctl = D.RUNNER_CURTAILMENT(tasks)
    assert ctl is not None
    order = D.curtailment_order(D.build_tasks(D.DRY_RUN, arms=[D.ARM_A_BOLD]))
    (on,) = order.values()
    gated = [t.task_id for t in on if D.curtailable(t)]
    assert ctl.gated_ids() == gated and len(gated) == 13
    fts = {t.task_id: t for t in on}
    # undecided while a run before it has no record
    assert ctl.decision(gated[0]) is None
    assert ctl.decision(on[0].task_id) is False  # not gated
    # records arrive out of order; the decision waits for the seed order
    later = [t for t in on if t.seed >= 386]
    for t in reversed(later):
        ctl.feed(t.task_id, _bold_record(t))
    assert ctl.decision(gated[0]) is None
    early = [t for t in on if t.seed < 386]
    for t in early:  # a false ABSENT at K = 2, seed 385
        ctl.feed(t.task_id, _bold_record(
            t, absent=(t.condition, t.seed) == ("PC_nominal_K2", 385)))
    assert [ctl.decision(g) for g in gated] == [True] * 13
    summ = ctl.summary()
    (grp,) = summ["groups"]
    assert grp["stopped"] and grp["skipped"] == gated
    assert grp["views"] == [{"principle": "PDI", "view": "bold", "events": 1,
                             "max_events": 0,
                             "stop_after": fts[next(t for t in fts if fts[t].seed == 385
                                                    and fts[t].condition ==
                                                    "PC_nominal_K2")].task_id}]
    # no event: every gated run is kept; an errored run is no event
    ctl = D.RUNNER_CURTAILMENT(tasks)
    for t in on:
        ctl.feed(t.task_id, _bold_record(t, error=(t.seed == 384)))
    assert [ctl.decision(g) for g in gated] == [False] * 13
    # a plan without gated runs needs no controller
    assert D.RUNNER_CURTAILMENT(DV.get_design("whole_brain").tasks(
        DV.DEVELOPMENT)) is None


def test_no_new_dependencies():
    for mod in (FW, D, RV):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert not re.search(r"^\s*(import|from)\s+(mne|sklearn)\b", src, re.M)
        assert "mpc_metrics" not in re.findall(r"import\s+([\w.]+)", src)
