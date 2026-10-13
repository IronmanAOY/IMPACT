"""Known-answer and null tests for the RAM estimator (compute_RAM)."""
import numpy as np
import pytest
from nilearn.glm.first_level import glover_hrf

from impact_pipeline import mpc_metrics as mm

# Few null draws keep the suite fast; the estimator default is 200.
FAST = {"quality_null_samples": 40}


def _fmri_design(seed, n_trials=24, trial_len=16.0, gap=2.0, fb_delay=6.0):
    rng = np.random.RandomState(seed)
    goal = 8.0 + trial_len * np.arange(n_trials) + rng.uniform(0.0, 3.0, n_trials)
    stim = goal + gap
    fb = stim + fb_delay
    return rng, goal, stim, fb


def _bold(tr, n_tp, onsets, patterns):
    t = np.arange(n_tp) * tr
    out = np.zeros((patterns[0].size, n_tp))
    for onset, pat in zip(onsets, patterns):
        out += np.outer(pat, mm.canonical_hrf(t - onset))
    return out


def _simulate_goal_task(seed, coupled, n_regions=40, tr=2.0, noise=1.0, gap=2.0):
    """
    BOLD = canonical-HRF-convolved goal, stimulus and feedback responses +
    white noise. Trial latent ``v`` sets the goal-cue pattern amplitude; the
    stimulus response is modulated by ``v`` (coupled) or by the goal latent of
    another trial (shuffled goal labels).
    """
    rng, goal, stim, fb = _fmri_design(seed, gap=gap)
    n_trials = goal.size
    n_tp = int((fb[-1] + 30.0) / tr)
    v = rng.randn(n_trials)
    v_resp = v if coupled else v[rng.permutation(n_trials)]
    a, b, c, f = (rng.randn(n_regions) for _ in range(4))
    fb_val = rng.uniform(0.2, 2.0, n_trials)
    ts = noise * rng.randn(n_regions, n_tp)
    ts += _bold(tr, n_tp, goal, [a * vi for vi in v])
    ts += _bold(tr, n_tp, stim, [c + b * vi for vi in v_resp])
    ts += _bold(tr, n_tp, fb, [f * fv for fv in fb_val])
    bundle = {
        "onsets": stim.tolist(),
        "goal_onsets": goal.tolist(),
        "feedback_onsets": fb.tolist(),
        "feedback_values": fb_val.tolist(),
    }
    return ts, tr, bundle


def test_canonical_hrf_peak_and_shape_match_nilearn_glover():
    peak_t, _ = mm.canonical_hrf_peak()
    assert 4.5 <= peak_t <= 6.0
    assert mm.canonical_hrf(np.array([peak_t]))[0] == pytest.approx(1.0)
    # Same shape as nilearn's Glover HRF on nilearn's own (oversampled) grid.
    tr, over = 2.0, 50
    ref = glover_hrf(tr, oversampling=over, time_length=32.0)
    grid = np.linspace(0.0, 32.0, ref.size) - tr / over  # nilearn uses loc=dt
    ours = mm.canonical_hrf(grid)
    assert np.corrcoef(ours, ref)[0, 1] > 0.999


@pytest.mark.parametrize("tr", [0.5, 0.8, 2.0])
def test_fmri_magnitude_and_latency_recovered_on_true_time_axis(tr):
    rng = np.random.RandomState(0)
    n_tp = int(420.0 / tr)
    onsets = np.sort(rng.uniform(10.0, 380.0, 25))  # not aligned to the TR grid
    amp = 0.7
    ts = amp * _bold(tr, n_tp, onsets, [np.ones(30)] * onsets.size)
    ts += 0.2 * rng.randn(30, n_tp)
    goal = onsets - 3.0
    fb = onsets + 5.0
    bundle = {
        "onsets": onsets.tolist(),
        "goal_onsets": goal.tolist(),
        "feedback_onsets": fb.tolist(),
        "feedback_values": rng.uniform(0.0, 1.0, onsets.size).tolist(),
    }
    d = mm.compute_RAM(ts, tr=tr, stimulus_onsets=bundle, magnitude_scale=1.0,
                       return_details=True, **FAST)
    # Guards against latency = argmax(50x-oversampled HRF) * tr (~250 s at any TR).
    assert d["latency_seconds"] == pytest.approx(mm.canonical_hrf_peak()[0], abs=1e-9)
    assert 4.5 <= d["latency_seconds"] <= 6.0
    assert d["magnitude_term"] == pytest.approx(amp, rel=0.05)

    fir = mm.compute_RAM(ts, tr=tr, stimulus_onsets=bundle, latency_method="fir",
                         fir_window=12.0, return_details=True, **FAST)
    assert fir["latency_seconds"] == pytest.approx(5.0, abs=max(0.6, 0.5 * tr))


@pytest.mark.parametrize("goal_amp,fb_amp", [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
def test_fmri_magnitude_not_confounded_by_goal_and_feedback_responses(goal_amp, fb_amp):
    # RAM requires goal and feedback events; their hemodynamic responses
    # overlap the stimulus response (goal 2 s before, feedback 6 s after).
    # Unmodelled, they biased M to 1.67 (goal) / 0.63 (feedback) for a true 1.0.
    rng = np.random.RandomState(0)
    tr, n_tp = 2.0, 300
    goal = 10.0 + 20.0 * np.arange(28) + rng.uniform(0.0, 3.0, 28)
    stim, fb = goal + 2.0, goal + 8.0
    ones = [np.ones(10)] * goal.size
    ts = _bold(tr, n_tp, stim, ones)
    ts += goal_amp * _bold(tr, n_tp, goal, ones) + fb_amp * _bold(tr, n_tp, fb, ones)
    ts += 0.1 * rng.randn(10, n_tp)
    bundle = {
        "onsets": stim.tolist(),
        "goal_onsets": goal.tolist(),
        "feedback_onsets": fb.tolist(),
        "feedback_values": rng.rand(goal.size).tolist(),
    }
    d = mm.compute_RAM(ts, tr=tr, stimulus_onsets=bundle, magnitude_scale=1.0,
                       return_details=True, quality_null_samples=5)
    assert d["magnitude_term"] == pytest.approx(1.0, rel=0.05)


def test_magnitude_undefined_when_stimuli_coincide_with_other_events():
    rng = np.random.RandomState(1)
    on = np.arange(4.0, 190.0, 2.5)
    ts = rng.randn(6, 400)
    bundle = {
        "onsets": on.tolist(),
        "goal_onsets": on.tolist(),  # label matching both patterns
        "feedback_onsets": (on + 1.0).tolist(),
        "feedback_values": rng.rand(on.size).tolist(),
    }
    d = mm.compute_RAM(ts, tr=0.5, stimulus_onsets=bundle, response_model="boxcar",
                       latency_method="hrf_peak", response_window_sec=0.5,
                       return_details=True, **FAST)
    assert np.isnan(d["value"])
    assert d["undefined_reason"] == "magnitude_design_rank_deficient"


def test_goal_alignment_null_and_coupled_fmri_with_overlapping_responses():
    # Goal cue 2 s before the stimulus: the BOLD responses overlap heavily.
    g_coupled, g_shuffled = [], []
    for seed in range(6):
        ts, tr, b = _simulate_goal_task(seed, coupled=True)
        g_coupled.append(
            mm.compute_RAM(ts, tr=tr, stimulus_onsets=b, return_details=True, **FAST)[
                "components"]["goal_alignment"]
        )
        ts, tr, b = _simulate_goal_task(seed, coupled=False)
        g_shuffled.append(
            mm.compute_RAM(ts, tr=tr, stimulus_onsets=b, return_details=True, **FAST)[
                "components"]["goal_alignment"]
        )
    assert np.all(np.isfinite(g_coupled)) and np.all(np.isfinite(g_shuffled))
    assert np.mean(g_coupled) > 0.6
    assert np.mean(g_shuffled) < 0.12
    assert np.median(g_shuffled) < 0.05


def test_goal_alignment_does_not_saturate_on_high_dimensional_noise():
    # An in-sample ridge CCA would give G ~ 0.99 on pure noise for every run.
    gs, r_cv = [], []
    for seed in range(12):
        rng = np.random.RandomState(100 + seed)
        ts = rng.randn(400, 300)  # p >> n: 400 regions, 20 events
        on = np.arange(10.0, 290.0, 14.0)
        bundle = {
            "onsets": on.tolist(),
            "goal_onsets": (on - 3.0).tolist(),
            "feedback_onsets": (on + 2.0).tolist(),
            "feedback_values": rng.randn(on.size).tolist(),
        }
        d = mm.compute_RAM(ts, tr=1.0, stimulus_onsets=bundle, response_model="boxcar",
                           latency_method="hrf_peak", return_details=True, **FAST)
        gs.append(d["components"]["goal_alignment"])
        r_cv.append(d["goal_alignment_cv_corr"])
    assert np.all(np.isfinite(gs))
    assert np.median(gs) < 0.05
    assert np.mean(gs) < 0.2
    assert abs(np.mean(r_cv)) < 0.2


def test_goal_alignment_detects_coupling_in_high_dimensions():
    gs = []
    for seed in range(4):
        rng = np.random.RandomState(200 + seed)
        n_regions, n_tp, tr = 200, 600, 1.0
        ts = rng.randn(n_regions, n_tp)
        goal = np.arange(10.0, 580.0, 19.0)
        stim = goal + 6.0
        v = rng.randn(goal.size)
        a, b = rng.randn(n_regions), rng.randn(n_regions)
        for g, s_, vi in zip(goal, stim, v):
            ts[:, int(g):int(g) + 2] += 0.5 * vi * a[:, None]
            ts[:, int(s_):int(s_) + 3] += 0.5 * vi * b[:, None]
        bundle = {
            "onsets": stim.tolist(),
            "goal_onsets": goal.tolist(),
            "feedback_onsets": (stim + 3.0).tolist(),
            "feedback_values": rng.randn(goal.size).tolist(),
        }
        d = mm.compute_RAM(ts, tr=tr, stimulus_onsets=bundle, response_model="boxcar",
                           latency_method="hrf_peak", return_details=True, **FAST)
        gs.append(d["components"]["goal_alignment"])
        assert d["goal_alignment_p_null"] < 0.05
    assert np.mean(gs) > 0.5


def _eeg_task(seed, n_regions=16, tr=0.004, n_trials=20, erp_latency=0.30):
    rng = np.random.RandomState(seed)
    n_tp = int((2.5 * n_trials + 2.0) / tr)
    ts = rng.randn(n_regions, n_tp)
    goal = 1.0 + 2.5 * np.arange(n_trials) + rng.uniform(0.0, 0.1, n_trials)
    stim = goal + 0.8
    fb = stim + 0.9
    tt = np.arange(-200, 200) * tr
    # Post-onset ERP plus a larger anticipatory (pre-onset) deflection.
    erp = 1.5 * np.exp(-0.5 * ((tt - erp_latency) / 0.05) ** 2)
    erp -= 2.0 * np.exp(-0.5 * ((tt + 0.4) / 0.1) ** 2)
    for o in stim:
        i = int(round(o / tr))
        ts[:, i - 200:i + 200] += erp[None, :]
    bundle = {
        "onsets": stim.tolist(),
        "goal_onsets": goal.tolist(),
        "feedback_onsets": fb.tolist(),
        "feedback_values": rng.randn(n_trials).tolist(),
    }
    return ts, tr, bundle


EEG_KW = dict(
    response_model="boxcar",
    response_boxcar_width_sec=0.30,
    latency_method="fir",
    fir_window=0.80,
    goal_pre_window_sec=0.20,
    response_window_sec=0.40,
    goal_objective_window_sec=0.20,
    feedback_window_sec=0.20,
)


def test_eeg_latency_measured_and_speed_not_dominated_by_epsilon():
    ts, tr, b = _eeg_task(0)
    d = mm.compute_RAM(ts, tr=tr, stimulus_onsets=b, return_details=True,
                       quality_null_samples=10, **EEG_KW)
    # Guards against argmax |avg| over +-0.8 s picking the -0.4 s deflection
    # (T clipped to 0 and speed = M / 0.004 s).
    assert d["latency_seconds"] == pytest.approx(0.30, abs=0.03)
    assert d["epsilon"] == pytest.approx(tr)
    assert d["latency_term"] > 50 * d["epsilon"]
    assert d["speed_term"] == pytest.approx(d["magnitude_term"] / d["latency_term"])


def test_fir_latency_undefined_when_peak_at_search_boundary():
    rng = np.random.RandomState(3)
    tr, n_tp = 0.01, 3000
    ts = 0.01 * rng.randn(4, n_tp)
    on = np.arange(2.0, 27.0, 2.0)
    # Response rises monotonically through the whole post-onset window.
    for o in on:
        i = int(round(o / tr))
        ts[:, i:i + 60] += np.linspace(0.0, 3.0, 60)[None, :]
    lat, reason = mm._fir_latency_seconds(ts, np.rint(on / tr).astype(int), tr, 0.5)
    assert np.isnan(lat) and reason == "latency_peak_at_search_boundary"


def test_xcorr_constant_stimulus_regressor_is_undefined_not_crash():
    ts = np.random.RandomState(0).randn(3, 4)
    d = mm.compute_RAM(ts, tr=1.0, stimulus_onsets=[0.0, 1.0, 2.0, 3.0],
                       latency_method="xcorr", response_model="boxcar",
                       require_explicit_feedback=False, require_explicit_goals=False,
                       return_details=True, **FAST)
    assert np.isnan(d["latency_seconds"])
    assert d["latency_undefined_reason"] == "stimulus_regressor_constant"
    assert np.isnan(d["value"])
    with pytest.raises(ValueError):
        mm.compute_RAM(ts, tr=1.0, stimulus_onsets=[1.0], response_model="stick",
                       latency_method="hrf_peak")


def _feedback_fixture():
    rng = np.random.RandomState(2)
    tr, n_tp, n_regions = 0.1, 2000, 8
    ts = 0.05 * rng.randn(n_regions, n_tp)
    fb_on = np.arange(5.0, 195.0, 5.0)
    fb_vals = rng.uniform(0.2, 2.0, fb_on.size)
    for o, v in zip(fb_on, fb_vals):
        i = int(round(o / tr))
        ts[:, i + 1:i + 4] += v
    stim = fb_on - 2.0
    bundle = {
        "onsets": stim.tolist(),
        "goal_onsets": (stim - 1.0).tolist(),
        "feedback_onsets": fb_on.tolist(),
        "feedback_values": fb_vals.tolist(),
    }
    kw = dict(
        response_model="boxcar",
        latency_method="hrf_peak",
        goal_pre_window_sec=0.2,
        response_window_sec=0.4,
        goal_objective_window_sec=0.2,
        feedback_window_sec=0.4,
        return_details=True,
        **FAST,
    )
    return ts, tr, bundle, kw


def test_feedback_values_stay_aligned_when_events_are_dropped_or_merged():
    ts, tr, bundle, kw = _feedback_fixture()
    base = mm.compute_RAM(ts, tr=tr, stimulus_onsets=bundle, **kw)
    assert base["components"]["feedback_integration"] > 0.9

    shifted = dict(bundle)
    # An out-of-range event (dropped) and a duplicate in the same sample
    # (merged) must not shift the value/event pairing of later events.
    fb_on, fb_val = bundle["feedback_onsets"], bundle["feedback_values"]
    shifted["feedback_onsets"] = [-1.0] + fb_on + [fb_on[-1] + 0.01]
    shifted["feedback_values"] = [9.0] + fb_val + [fb_val[-1]]
    d = mm.compute_RAM(ts, tr=tr, stimulus_onsets=shifted, **kw)
    assert d["components"]["feedback_integration"] == pytest.approx(
        base["components"]["feedback_integration"], abs=1e-6
    )

    bad = dict(bundle)
    bad["feedback_values"] = bundle["feedback_values"][:-1]
    d = mm.compute_RAM(ts, tr=tr, stimulus_onsets=bad, **kw)
    assert np.isnan(d["value"])
    assert d["undefined_reason"] == "feedback_values_misaligned"


def test_sanitize_onsets_with_values_merges_and_drops_jointly():
    sec, idx, vals = mm._sanitize_onsets_with_values(
        [3.0, -1.0, 1.0, 1.04, np.nan, 2.0],
        [30.0, 9.0, 10.0, 12.0, 5.0, np.nan],
        tr=0.1,
        n_tp=100,
    )
    assert idx.tolist() == [10, 30]
    assert vals.tolist() == pytest.approx([11.0, 30.0])
    assert sec.tolist() == pytest.approx([1.02, 3.0])


def test_adaptive_update_pairs_feedback_between_consecutive_stimuli():
    # Updates k -> k+1 scale with |feedback| delivered between s_k and s_{k+1}.
    rng = np.random.RandomState(5)
    stim_idx = np.arange(10, 400, 20)
    drive = rng.uniform(0.1, 3.0, stim_idx.size - 1)
    resp = np.zeros((stim_idx.size, 6))
    for k, dk in enumerate(drive):
        step = rng.randn(6)
        resp[k + 1] = resp[k] + dk * step / np.linalg.norm(step)
    fb_idx = stim_idx[:-1] + 7
    u, n, reason = mm._adaptive_update_component(
        resp, stim_idx, fb_idx, drive, n_null=40, rng=np.random.RandomState(0)
    )
    assert reason is None and n == drive.size
    assert u > 0.9
    # Pairing by position after dropping the first feedback event would misalign.
    u_mis, _, _ = mm._adaptive_update_component(
        resp, stim_idx, fb_idx[1:], drive[:-1], n_null=40, rng=np.random.RandomState(0)
    )
    assert u_mis < 0.5


def test_ram_is_undefined_without_goal_or_feedback_structure():
    ts, tr, bundle, kw = _feedback_fixture()
    no_goal = dict(bundle, goal_onsets=[])
    d = mm.compute_RAM(ts, tr=tr, stimulus_onsets=no_goal, **kw)
    assert np.isnan(d["value"]) and d["undefined_reason"] == "missing_goal_events"
    proxy = mm.compute_RAM(
        ts, tr=tr, stimulus_onsets=no_goal, require_explicit_goals=False, **kw
    )
    assert proxy["goal_source"] == "pre_stimulus_state_proxy"
    assert np.isfinite(proxy["value"])

    no_fb = dict(bundle, feedback_onsets=[], feedback_values=None)
    d = mm.compute_RAM(ts, tr=tr, stimulus_onsets=no_fb, **kw)
    assert np.isnan(d["value"]) and d["undefined_reason"] == "missing_explicit_feedback"

    few = {
        "onsets": bundle["onsets"][:2],
        "goal_onsets": bundle["goal_onsets"][:2],
        "feedback_onsets": bundle["feedback_onsets"][:2],
        "feedback_values": [1.0, 2.0],
    }
    d = mm.compute_RAM(ts, tr=tr, stimulus_onsets=few, **kw)
    # Undefined with a reason, not a silent, defined RAM = 0.0.
    assert np.isnan(d["value"])
    assert d["undefined_reason"] == "insufficient_goal_response_pairs"


def test_zero_weight_component_does_not_make_ram_undefined():
    ts, tr, bundle, kw = _feedback_fixture()
    few_goal = dict(bundle, goal_onsets=bundle["goal_onsets"][:1])
    d = mm.compute_RAM(ts, tr=tr, stimulus_onsets=few_goal, **kw)
    assert np.isnan(d["value"])
    d0 = mm.compute_RAM(
        ts, tr=tr, stimulus_onsets=few_goal, quality_weights=(0.0, 1.0, 1.0), **kw
    )
    assert np.isnan(d0["components"]["goal_alignment"])
    assert np.isfinite(d0["value"])


def test_rank_deficient_beta_series_design_makes_ram_undefined():
    rng = np.random.RandomState(0)
    tr, n_tp = 2.0, 150
    # Fast design: three event types every 3 s -> more trial-wise regressors
    # (277) than samples (150). Minimum-norm betas are not measurements; the
    # old behaviour here was a defined RAM = 0.0.
    goal = np.arange(4.0, 280.0, 3.0)
    fast = {
        "onsets": (goal + 1.0).tolist(),
        "goal_onsets": goal.tolist(),
        "feedback_onsets": (goal + 2.0).tolist(),
        "feedback_values": rng.rand(goal.size).tolist(),
    }
    ts = rng.randn(50, n_tp)
    d = mm.compute_RAM(ts, tr=tr, stimulus_onsets=fast, return_details=True, **FAST)
    assert np.isnan(d["value"])
    assert d["undefined_reason"] == "trialwise_design_rank_deficient"

    # Some goal events at the same onsets as their stimuli (e.g. a label
    # matching both patterns): those trial-wise amplitudes cannot be separated.
    on = np.arange(10.0, 280.0, 14.0)
    goal_on = on - 3.0
    goal_on[:2] = on[:2]
    same = {
        "onsets": on.tolist(),
        "goal_onsets": goal_on.tolist(),
        "feedback_onsets": (on + 6.0).tolist(),
        "feedback_values": rng.rand(on.size).tolist(),
    }
    d = mm.compute_RAM(ts, tr=tr, stimulus_onsets=same, return_details=True, **FAST)
    assert np.isnan(d["value"])
    assert d["undefined_reason"] == "trialwise_design_rank_deficient"

    # A separable design on the same data stays defined.
    ok = dict(same, goal_onsets=(on - 3.0).tolist())
    d = mm.compute_RAM(ts, tr=tr, stimulus_onsets=ok, return_details=True, **FAST)
    assert d["undefined_reason"] is None and np.isfinite(d["value"])


def test_non_finite_timeseries_is_undefined_with_reason():
    ts, tr, bundle, kw = _feedback_fixture()
    ts = ts.copy()
    ts[3, 10] = np.nan
    d = mm.compute_RAM(ts, tr=tr, stimulus_onsets=bundle, **kw)
    # A named reason, not NaN (via M) with undefined_reason None.
    assert np.isnan(d["value"])
    assert d["undefined_reason"] == "non_finite_timeseries"
