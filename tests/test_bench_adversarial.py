"""MPC-Bench adversarial constructions: each realises its documented
mechanism (checked on oracle channels and structure, never on MPC estimator
values), the exact parity TPM, and the catalogue."""

import numpy as np
import pytest

from impact_pipeline.bench import adversarial as ad
from impact_pipeline.bench import generators as g
from impact_pipeline.event_parsing import (
    events_table_to_bundle,
    validate_srpi_agency_contract,
)

CFG = g.AgentConfig(n_trials=40, n_reafference_pairs=15)


def test_catalogue_is_complete_and_documented():
    assert set(ad.ADVERSARIAL_CATALOGUE) == set(ad.ADVERSARIAL_KINDS)
    for kind, e in ad.ADVERSARIAL_CATALOGUE.items():
        assert e["designed_to_fool"] in g.PRINCIPLES, kind
        assert 0 in e["intended_pattern"], kind  # lacks a principle by design
        assert e["mechanism"] and e["claim"] and e["references"], kind
    tab = ad.catalogue_table()
    assert list(tab["kind"]) == list(ad.ADVERSARIAL_KINDS)
    assert set(tab["expected_verdict"]) == {"EXCLUDED"}
    with pytest.raises(ValueError):
        ad.make_adversarial("zombie")


def test_parity_network_exact_tpm_known_answers():
    net = ad.parity_network(4, "mobius", noise=0.0)
    assert net.tpm.shape == (16, 16) and np.allclose(net.tpm.sum(axis=1), 1.0)
    # Moebius ladder on 4 nodes: every unit has 3 inputs; s'_i = XOR of them.
    assert np.array_equal(net.connectivity.sum(axis=0), [3, 3, 3, 3])
    s = np.array([1, 0, 1, 1])
    nbrs = ad.circulant_neighbours(4)
    expected = [int(np.bitwise_xor.reduce(s[nb])) for nb in nbrs]
    assert net.tpm[g.state_index(s), g.state_index(expected)] == pytest.approx(1.0)
    assert net.tpm[0, 0] == pytest.approx(1.0)  # all-off fixed point
    noisy = ad.parity_network(6, noise=0.02)
    assert noisy.n == 6 and np.all(noisy.tpm > 0)
    assert ad.circulant_neighbours(6)[0] == [1, 3, 5]
    with pytest.raises(ValueError):
        ad.parity_network(5, "mobius")
    with pytest.raises(ValueError):
        ad.parity_network(4, noise=0.7)


def test_parity_grid_has_high_exact_integration_but_is_inert():
    mm = pytest.importorskip("impact_pipeline.mpc_metrics")
    s = ad.parity_grid_system(seed=1, n=4, config=CFG)
    assert s.meta["substrate"] == "ising_exact" and s.meta["exact_tpm_state"] == 0
    tpm = np.asarray(s.meta["exact_tpm"])
    w = np.zeros(16)
    w[0] = 1.0
    val = mm.compute_IIM_from_tpm(tpm, state_weights=w)
    ind = g.family_b_network("independent", 4)
    assert val > 0.5 and mm.compute_IIM_from_tpm(ind.tpm) == pytest.approx(0.0)
    # Inert: the hidden state never leaves the fixed point, so the recording
    # is pure measurement noise, and the events do not reach the system.
    assert np.all(s.oracle["hidden_state"] == 0)
    assert s.ts.std() == pytest.approx(0.05, rel=0.05)
    corr = np.corrcoef(s.ts)
    assert np.max(np.abs(corr[np.triu_indices(4, 1)])) < 0.1
    assert len(s.events) > 0 and s.oracle["intended_bits"] == [0, 0, 0, 1, 0]


def test_hypersynchrony_and_common_driver():
    hyp = ad.make_adversarial("hypersynchrony", CFG, seed=2)
    c = np.corrcoef(hyp.ts)
    assert np.mean(c[np.triu_indices_from(c, 1)]) > 0.95
    assert hyp.meta["adversarial"] == "hypersynchrony"
    drv = ad.make_adversarial("common_driver", CFG, seed=2)
    hub = np.asarray(drv.meta["workspace_nodes"])
    lags = drv.oracle["driver_lags_samples"]
    periph = np.setdiff1d(np.arange(drv.n_nodes), hub)
    assert np.all(lags[hub] == 0) and np.all(lags[periph] >= 2)
    assert not np.any(drv.oracle["unit_adjacency"])  # no receive, no return
    # Global co-activation and hub-leads-periphery lagged dependence arise
    # from the common driver alone.
    c = np.corrcoef(drv.ts)
    assert np.mean(c[np.triu_indices_from(c, 1)]) > 0.2
    h = drv.ts[hub].mean(axis=0)
    p = drv.ts[periph].mean(axis=0)
    xc = [np.corrcoef(h[: len(h) - k], p[k:])[0, 1] for k in range(0, 8)]
    assert int(np.argmax(xc)) >= 2
    assert 0.02 < drv.oracle["burst_fraction"] < 0.6


def test_reflex_arc_is_responsive_but_not_adaptive():
    cfg = g.AgentConfig(n_trials=80, n_reafference_pairs=2)
    agree, acc = [], []
    for seed in range(3):
        s = ad.make_adversarial("reflex_arc", cfg, seed)
        stim = s.events.loc[s.events.trial_type == "stimulus", "value"].to_numpy()
        agree.append(np.mean(stim == s.oracle["choices"]))
        acc.append(s.oracle["accuracy"])
        assert np.all(s.oracle["q_by_trial"] == 0)  # no plasticity
        assert s.oracle["module_graph_acyclic"]  # feedforward
        assert s.oracle["agent_knob_bits"] == [0, 0, 0, 0, 0]
    assert np.mean(agree) > 0.75  # the stimulus dictates the response
    assert abs(np.mean(acc) - 0.5) < 0.12  # choices do not track reversals


def test_random_labels_keep_the_agency_contract_but_carry_no_efference():
    s = ad.make_adversarial("random_label_self_other", CFG, seed=4)
    plain = g.simulate_family_a(g.NOMINAL_KNOBS.replace(e=0.0), CFG, seed=4)
    assert np.array_equal(s.ts, plain.ts)  # labels only; dynamics untouched
    assert np.max(s.oracle["efference_tag"]) == 0
    rep = validate_srpi_agency_contract(
        events_table_to_bundle(s.events)["agency_events"]
    )
    assert rep["valid"] and rep["n_pairs"] == CFG.n_reafference_pairs
    swapped = s.oracle["labels_swapped"]
    assert 0 < swapped.sum() < swapped.size
    ev = s.events
    selfs = ev[ev.trial_type == "self_caused"].set_index("event_id")["onset"]
    later_self = [
        selfs.loc[r["yoked_to"]] > r["onset"]
        for _, r in ev[ev.trial_type == "other_caused"].iterrows()
    ]
    assert np.array_equal(np.asarray(later_self), swapped)
    assert s.oracle["intended_bits"] == [1, 1, 1, 1, 0]


def test_scrambled_feedback_removes_the_contingency_not_the_plasticity():
    cfg = g.AgentConfig(n_trials=80, n_reafference_pairs=2)
    acc, corr = [], []
    for seed in range(3):
        s = ad.make_adversarial("scrambled_feedback", cfg, seed)
        fb = s.events[s.events.trial_type == "feedback"]
        correct = fb["choice"].astype(int).to_numpy() == s.oracle["good_arm"]
        corr.append(np.corrcoef(fb["reward"].to_numpy(), correct)[0, 1])
        acc.append(s.oracle["accuracy"])
        assert np.ptp(s.oracle["q_by_trial"]) > 0  # gains still update
        assert s.meta["config"]["feedback_mode"] == "scrambled"
    assert abs(np.mean(corr)) < 0.15
    assert abs(np.mean(acc) - 0.5) < 0.12
    with pytest.raises(ValueError):
        g.AgentConfig(feedback_mode="random")
    with pytest.raises(ValueError):
        g.AgentConfig(reflex_gain=-1.0)
