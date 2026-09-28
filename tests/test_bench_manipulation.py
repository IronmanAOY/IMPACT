"""Preregistered manipulation checks of the MPC-Bench switches on oracle
channels (never on MPC estimators): known-answer tests of the signatures,
families A and C (design 2) pass every check on development seeds, and a
design-1-like family C configuration fails them."""

import numpy as np
import pandas as pd
import pytest

from impact_pipeline.bench import generators as g
from impact_pipeline.bench import manipulation as mp

CHECK_CFG = g.AgentConfig(n_trials=40, n_reafference_pairs=15)


def _var(n, T, A, seed, noise=1.0):
    rng = np.random.default_rng(seed)
    x = np.zeros((n, T))
    for t in range(1, T):
        x[:, t] = A @ x[:, t - 1] + noise * rng.standard_normal(n)
    return x


def test_granger_gain_known_answers():
    # Independent AR(1) processes: no gain in either direction (df-adjusted,
    # so short runs are not biased upwards).
    for seed in range(5):
        x = _var(2, 1500, np.diag([0.8, 0.5]), seed)
        assert mp.granger_gain(x[0], x[1]) < 0.01
    # x1 drives x0 (one direction only).
    A = np.array([[0.5, 0.6], [0.0, 0.5]])
    x = _var(2, 4000, A, 1)
    fwd, bwd = mp.granger_gain(x[0], x[1]), mp.granger_gain(x[1], x[0])
    assert fwd > 0.15 and bwd < 0.01
    # Complex signals: a rotated copy of the source's past is recovered.
    rng = np.random.default_rng(3)
    s = rng.standard_normal(3000) + 1j * rng.standard_normal(3000)
    tgt = np.r_[0, 0.8 * np.exp(1j) * s[:-1]] + 0.3 * (
        rng.standard_normal(3000) + 1j * rng.standard_normal(3000)
    )
    assert mp.granger_gain(tgt, s) > 0.8
    assert np.isnan(mp.granger_gain(np.arange(8.0), np.arange(8.0)))


def test_signature_known_answers():
    s = g.simulate_family_a(None, CHECK_CFG, seed=1)
    # One context pattern carries no context information by definition.
    k1 = g.simulate_family_a(g.NOMINAL_KNOBS.replace(K=1), CHECK_CFG, seed=1)
    assert mp.context_information(k1) == 0.0
    assert 0.0 < mp.context_information(s) < 1.0
    # Identical self and other responses give a contrast of about 0: replace
    # the hidden state by a copy in which each replay window repeats the
    # response of its self-caused event.
    ts = s.ts.copy()
    ev = s.events
    selfs = ev[ev.trial_type == "self_caused"].set_index("event_id")["onset"]
    pre = int(round(mp.PRE_WINDOW_SEC / s.dt))
    post = int(round(mp.RESPONSE_WINDOW_SEC / s.dt))
    for _, row in ev[ev.trial_type == "other_caused"].iterrows():
        i_o = int(round(row["onset"] / s.dt))
        i_s = int(round(selfs.loc[row["yoked_to"]] / s.dt))
        ts[:, i_o - pre : i_o + post] = ts[:, i_s - pre : i_s + post]
    same = g.BenchSystem(ts=ts, events=ev, meta=s.meta, oracle=s.oracle)
    assert mp.self_other_contrast(same) == pytest.approx(0.0, abs=1e-12)
    assert mp.self_other_contrast(s) > 0.3
    # Reversal tracking is the correlation of the hidden choice probability
    # with the rewarded arm.
    o = dict(s.oracle)
    o["p_choice1"] = o["good_arm"].astype(float)
    tracked = g.BenchSystem(ts=s.ts, events=ev, meta=s.meta, oracle=o)
    assert mp.reversal_tracking(tracked) == pytest.approx(1.0)
    assert mp.relative_change(0.0, 0.0) != mp.relative_change(0.0, 0.0)  # NaN
    with pytest.raises(ValueError):
        mp.hidden_state(s, "rotating")


@pytest.mark.parametrize(
    "simulate", [g.simulate_family_a, g.simulate_family_c], ids=["A", "C"]
)
def test_every_switch_passes_its_preregistered_check(simulate):
    rep = mp.manipulation_report(simulate, (0, 1, 2), CHECK_CFG)
    assert set(rep["switch"]) == set(mp.SWITCHES)
    assert len(rep) == 3 * len(mp.SWITCHES)
    failed = rep[~rep["passed"]]
    assert failed.empty, failed.to_string()
    assert (rep["relative_change"] >= rep["threshold"]).all()
    assert (rep["nominal"] >= rep["floor"]).all()
    assert set(rep["check_version"]) == {mp.MANIPULATION_CHECK_VERSION}


def test_design_1_like_family_c_fails_the_checks():
    """Weak coupling against a strong additive-like drive and detuned
    oscillators (the reviewer's diagnosis of design 1) does not pass the NAS
    and IIM checks: the checks can fail."""
    weak = CHECK_CFG.replace(sl_coupling=0.2, sl_input_gain=4.0, sl_freq_hz=(0.5, 1.0))
    rep = mp.manipulation_report(g.simulate_family_c, (0, 1), weak, ("g_b", "c_int"))
    assert not rep["passed"].any()


def test_checks_read_oracle_channels_only(monkeypatch):
    """The signatures never call an MPC estimator."""
    import sys

    monkeypatch.setitem(sys.modules, "impact_pipeline.mpc_metrics", None)
    s = g.simulate_family_c(None, CHECK_CFG, seed=0)
    sig = mp.oracle_signatures(s)
    assert set(sig) == set(mp.SIGNATURES)
    assert all(np.isfinite(v) for v in sig.values())
    assert isinstance(
        mp.manipulation_check(g.simulate_family_a, 0, CHECK_CFG, ("K",)), pd.DataFrame
    )
