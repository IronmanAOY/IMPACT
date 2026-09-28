"""MPC-Bench generators: determinism, knob effects on the hidden oracle
channels (never on metric values), event-schema validity, exact family-B
TPMs and their sampler, nulls, patchwork, witness catalogue and designs."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from impact_pipeline.bench import factorial, generators as g, sweeps, witnesses
from impact_pipeline.bench.patchwork import PATCHWORK_MODULES, simulate_patchwork
from impact_pipeline.event_parsing import classify_self_nonself, events_table_to_bundle

SRC = Path(__file__).resolve().parents[1] / "src"
SMALL = g.AgentConfig(n_trials=16, n_reafference_pairs=6)
N_TR, N_PAIRS = SMALL.n_trials, SMALL.n_reafference_pairs


@pytest.fixture(scope="module")
def nominal_a():
    return g.simulate_family_a(None, SMALL, seed=11)


@pytest.fixture(scope="module")
def nominal_c():
    return g.simulate_family_c(None, SMALL, seed=11)


def test_generators_do_not_import_mpc_metrics():
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(SRC)!r})\n"
        "from impact_pipeline.bench import generators, patchwork, witnesses, "
        "factorial, sweeps, lz, phiid_gaussian, rules\n"
        "cfg = generators.AgentConfig(n_trials=4, n_reafference_pairs=2)\n"
        "for gen in generators.GENERATOR_NAMES:\n"
        "    generators.make_system(gen, None, cfg, 0)\n"
        "witnesses.load_witnesses()\n"
        "bad = sorted(m for m in sys.modules if m.startswith('impact_pipeline.') and "
        "m.split('.')[1] in ('mpc_metrics', 'synergy_ci', 'hardware_backend'))\n"
        "print('BAD', bad)\n"
        "sys.exit(1 if bad else 0)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.parametrize("simulate", [g.simulate_family_a, g.simulate_family_c])
def test_determinism_by_seed(simulate):
    a = simulate(None, SMALL, seed=3)
    b = simulate(None, SMALL, seed=3)
    c = simulate(None, SMALL, seed=4)
    assert np.array_equal(a.ts, b.ts)
    pd.testing.assert_frame_equal(a.events, b.events)
    for key, val in a.oracle.items():
        if isinstance(val, np.ndarray):
            assert np.array_equal(val, b.oracle[key]), key
    assert json.dumps(a.meta, sort_keys=True) == json.dumps(b.meta, sort_keys=True)
    assert not np.array_equal(a.ts, c.ts)
    assert not a.events["onset"].equals(c.events["onset"])


def test_knob_changes_keep_structure_schedule_and_noise(nominal_a):
    """Common random numbers: knobs rescale mechanisms, they do not redraw."""
    off = g.simulate_family_a(g.NOMINAL_KNOBS.replace(e=0.0), SMALL, seed=11)
    open_loop = ("goal_cue", "stimulus", "action", "self_caused", "other_caused")
    for tt in open_loop:
        on = nominal_a.events.loc[nominal_a.events.trial_type == tt, ["onset", "value"]]
        of = off.events.loc[off.events.trial_type == tt, ["onset", "value"]]
        assert np.array_equal(on.to_numpy(), of.to_numpy()), tt
    # Unit coupling identical (e only acts on inputs), activity identical until
    # the first self-caused event.
    assert np.array_equal(
        nominal_a.oracle["unit_adjacency"], off.oracle["unit_adjacency"]
    )
    first_self = int(round(nominal_a.meta["reafference_start_sec"] / SMALL.dt))
    assert np.array_equal(nominal_a.ts[:, :first_self], off.ts[:, :first_self])


def test_knob_effects_on_oracle_channels(nominal_a):
    base = g.summarise_oracle(nominal_a)
    assert base["intended_bits"] == [1, 1, 1, 1, 1]
    assert base["q_range"] > 0.3
    assert base["context_states_visited"] >= 4
    assert base["n_ignitions"] >= 1
    assert not base["module_graph_acyclic"]
    assert base["efference_tag_max"] > 0
    assert base["reafference_attenuation_mean"] < 1.0

    def run(**kw):
        return g.simulate_family_a(g.NOMINAL_KNOBS.replace(**kw), SMALL, seed=11)

    s = run(eta=0.0)
    assert np.all(s.oracle["q_by_trial"] == 0.0)
    assert g.summarise_oracle(s)["intended_bits"] == [0, 1, 1, 1, 1]

    s = run(K=1)
    assert s.oracle["context_states_visited"] == 1
    assert s.oracle["intended_bits"] == [1, 0, 1, 1, 1]

    s = run(g_b=0.0)
    hub = np.asarray(s.meta["workspace_nodes"])
    other = np.setdiff1d(np.arange(s.n_nodes), hub)
    adj = s.oracle["unit_adjacency"]  # [source, target]
    assert np.all(adj[np.ix_(hub, other)] == 0) and np.all(adj[np.ix_(other, hub)] == 0)
    assert s.oracle["ignition_onsets_sec"] == []
    assert np.all(s.oracle["ignition_gate"] == 0)

    s = run(ff_only=True)
    adj = s.oracle["unit_adjacency"]
    assert np.all(adj[np.ix_(other, hub)] == 0)  # no receive path
    assert np.any(adj[np.ix_(hub, other)] > 0)  # broadcast kept
    assert s.oracle["intended_bits"] == [1, 1, 0, 1, 1]

    s = run(c_int=0.0)
    assert s.oracle["module_graph_acyclic"]
    assert s.oracle["intended_bits"] == [1, 1, 1, 0, 1]
    assert g.is_acyclic(s.oracle["module_adjacency"])
    # Hub is a relay: receives from upstream only, returns downstream only.
    madj = s.oracle["module_adjacency"]
    order = s.meta["module_order"]
    w = order.index("W")
    assert all(madj[w, j] == 0 for j in range(w))
    assert all(madj[j, w] == 0 for j in range(w + 1, len(order)))

    s = run(e=0.0)
    assert np.max(s.oracle["efference_tag"]) == 0
    assert np.all(s.oracle["reafference_attenuation"] == 1.0)
    assert s.oracle["intended_bits"] == [1, 1, 1, 1, 0]

    off = g.simulate_family_a(g.OFF_KNOBS, SMALL, seed=11)
    assert off.oracle["intended_bits"] == [0, 0, 0, 0, 0]
    assert off.oracle["module_graph_acyclic"]


def test_plasticity_is_behaviourally_effective():
    cfg = g.AgentConfig(n_trials=80, n_reafference_pairs=2)
    acc_on = np.mean(
        [g.simulate_family_a(None, cfg, s).oracle["accuracy"] for s in range(3)]
    )
    acc_off = np.mean(
        [
            g.simulate_family_a(g.NOMINAL_KNOBS.replace(eta=0.0), cfg, s).oracle[
                "accuracy"
            ]
            for s in range(3)
        ]
    )
    assert acc_on > acc_off + 0.1


def _check_events(system):
    ev = system.events
    cfg = system.meta["config"]
    dt = system.dt
    assert tuple(ev.columns) == g.EVENT_COLUMNS
    assert ev["onset"].is_monotonic_increasing
    assert np.allclose(ev["onset"] / dt, np.round(ev["onset"] / dt))
    assert (ev["duration"] > 0).all()
    assert set(ev["trial_type"]) <= set(g.TRIAL_TYPES)
    counts = ev["trial_type"].value_counts()
    for tt in ("goal_cue", "stimulus", "response", "feedback"):
        assert counts[tt] == cfg["n_trials"]
    for tt in ("action", "self_caused", "other_caused"):
        assert counts[tt] == cfg["n_reafference_pairs"]
    assert set(ev["impact_channel"]) == {g.IMPACT_CHANNEL_TASK, g.IMPACT_CHANNEL_AGENCY}
    assert ev["phase_bin"].between(0, cfg["n_phase_bins"] - 1).all()
    assert ev["event_id"].is_unique
    fb = ev[ev.trial_type == "feedback"]
    assert set(fb["reward"]) <= {-1.0, 1.0}
    assert set(fb["choice"].astype(int)) <= {0, 1}
    resp = ev[ev.trial_type == "response"]
    assert np.array_equal(resp["choice"].to_numpy(), fb["choice"].to_numpy())
    # Yoked replays: stimulus-identical, phase-matched, later than the original.
    selfs = ev[ev.trial_type == "self_caused"].set_index("event_id")
    others = ev[ev.trial_type == "other_caused"]
    period = system.meta["config"]["slow_freq_hz"]
    period_samples = int(round(1.0 / (period * dt)))
    for _, row in others.iterrows():
        src = selfs.loc[row["yoked_to"]]
        assert row["value"] == src["value"]
        assert row["duration"] == src["duration"]
        assert row["phase_bin"] == src["phase_bin"]
        lag = int(round((row["onset"] - src["onset"]) / dt))
        assert lag > 0 and lag % period_samples == 0
    assert ev.loc[ev.trial_type != "other_caused", "yoked_to"].isna().all()
    return ev


@pytest.mark.parametrize("which", ["nominal_a", "nominal_c"])
def test_event_schema_validity(which, request):
    system = request.getfixturevalue(which)
    ev = _check_events(system)
    bundle = events_table_to_bundle(ev)
    assert len(bundle["onsets"]) == N_TR
    assert len(bundle["goal_onsets"]) == N_TR
    assert len(bundle["feedback_onsets"]) == N_TR
    assert sorted(set(bundle["feedback_values"])) in ([-1.0, 1.0], [-1.0], [1.0])
    assert len(bundle["self_onsets"]) == N_PAIRS
    assert len(bundle["nonself_onsets"]) == N_PAIRS
    self_mask, non_mask = classify_self_nonself(ev["trial_type"])
    assert set(ev.loc[self_mask, "trial_type"]) == {"self_caused"}
    assert set(ev.loc[non_mask, "trial_type"]) == {"other_caused"}


def test_replays_receive_identical_stimulus_input():
    # Without endogenous drives (context patterns, slow-rhythm drive) the only
    # input to the sensory module in the reafference phase is the stimulus.
    cfg = SMALL.replace(slow_gain=0.0)
    s = g.simulate_family_a(g.NOMINAL_KNOBS.replace(e=0.0, k_gain=0.0), cfg, seed=5)
    ev = s.events
    inp = s.oracle["inputs"]
    sens = np.asarray(s.meta["modules"]["S"])
    selfs = ev[ev.trial_type == "self_caused"].set_index("event_id")
    dur = int(round(0.2 / s.dt))
    for _, row in ev[ev.trial_type == "other_caused"].iterrows():
        i_o = int(round(row["onset"] / s.dt))
        i_s = int(round(selfs.loc[row["yoked_to"], "onset"] / s.dt))
        assert np.any(inp[sens, i_s] != 0)
        assert np.array_equal(inp[sens, i_s : i_s + dur], inp[sens, i_o : i_o + dur])
    # With the efference copy on, self-caused input is attenuated and tagged.
    on = g.simulate_family_a(g.NOMINAL_KNOBS.replace(k_gain=0.0), cfg, seed=5)
    tag = on.oracle["efference_tag"]
    inp_on = on.oracle["inputs"]
    for _, row in on.events[on.events.trial_type == "other_caused"].iterrows():
        i_o = int(round(row["onset"] / on.dt))
        i_s = int(
            round(on.events.set_index("event_id").loc[row["yoked_to"], "onset"] / on.dt)
        )
        assert tag[i_s] > 0 and tag[i_o] == 0
        assert not np.array_equal(inp_on[sens, i_s], inp_on[sens, i_o])
        assert np.array_equal(inp_on[sens, i_o], inp[sens, i_o])


def test_meta_declares_bearer_workspace_and_macro_grain(nominal_a):
    meta = nominal_a.meta
    json.dumps(meta)
    assert (
        meta["n_nodes"] == nominal_a.n_nodes == SMALL.n_modules * SMALL.units_per_module
    )
    assert meta["bearer_nodes"] == list(range(nominal_a.n_nodes))
    assert meta["workspace_nodes"] == meta["modules"]["W"]
    macro = meta["iim_macro_nodes"]
    assert "W" not in macro and len(macro) == SMALL.n_modules - 1
    assert sorted(i for v in macro.values() for i in v) == sorted(
        i for k, v in meta["modules"].items() if k != "W" for i in v
    )
    assert meta["knobs"] == g.NOMINAL_KNOBS.to_dict()
    assert meta["spectral_radius"] == pytest.approx(SMALL.spectral_radius)
    # The oracle is not part of meta.
    assert "q_by_trial" not in meta and "intended_bits" not in meta


@pytest.mark.parametrize("n_modules", [4, 6])
def test_module_layouts(n_modules):
    cfg = SMALL.replace(n_modules=n_modules, units_per_module=4)
    s = g.simulate_family_a(None, cfg, seed=1)
    assert s.n_nodes == 4 * n_modules
    _check_events(s)
    with pytest.raises(ValueError):
        g.AgentConfig(n_modules=7)
    with pytest.raises(ValueError):
        g.AgentConfig(units_per_module=9)


def test_family_c_switch_semantics(nominal_c):
    assert nominal_c.meta["dynamics"] == "stuart_landau"
    base = g.summarise_oracle(nominal_c)
    assert base["q_range"] > 0.3 and base["efference_tag_max"] > 0
    off = g.simulate_family_c(g.OFF_KNOBS, SMALL, seed=11)
    o = g.summarise_oracle(off)
    assert o["q_range"] == 0 and o["context_states_visited"] == 1
    assert o["n_ignitions"] == 0 and o["module_graph_acyclic"]
    assert o["efference_tag_max"] == 0
    assert np.all(np.isfinite(nominal_c.ts)) and np.all(np.isfinite(off.ts))


def _hub_periphery_corr(system):
    """Mean |corr| between the hub mean and each periphery module mean."""
    hub = system.ts[np.asarray(system.meta["workspace_nodes"])].mean(axis=0)
    return float(
        np.mean(
            [
                abs(np.corrcoef(hub, system.ts[np.asarray(v)].mean(axis=0))[0, 1])
                for v in system.meta["iim_macro_nodes"].values()
            ]
        )
    )


def _backward_granger_gain(system, lag=2):
    """Mean fraction of residual variance of an upstream periphery module
    explained by a downstream module's past beyond its own past (lag-2 OLS)."""
    x = np.stack(
        [
            system.ts[np.asarray(v)].mean(axis=0)
            for v in system.meta["iim_macro_nodes"].values()
        ]
    )
    x = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
    gains = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            y, own, other = x[i, lag:], x[i, :-lag], x[j, :-lag]
            r_own = np.linalg.lstsq(own[:, None], y, rcond=None)[1][0]
            r_both = np.linalg.lstsq(np.c_[own, other], y, rcond=None)[1][0]
            gains.append(1.0 - r_both / r_own)
    return float(np.mean(gains))


def _assert_coupling_switches_act(simulate, cfg=SMALL):
    for seed in (0, 1, 2):
        nom = simulate(None, cfg, seed)
        gb0 = simulate(g.NOMINAL_KNOBS.replace(g_b=0.0), cfg, seed)
        c0 = simulate(g.NOMINAL_KNOBS.replace(c_int=0.0), cfg, seed)
        # Workspace off: hub and periphery decouple.
        assert _hub_periphery_corr(gb0) < 0.6 * _hub_periphery_corr(nom), seed
        # Loops off: downstream modules no longer predict upstream ones.
        assert _backward_granger_gain(c0) < 0.5 * _backward_granger_gain(nom), seed


def test_family_a_coupling_switches_change_the_dynamics():
    """Manipulation check on simple observables (not MPC estimators): the
    NAS and IIM switches act on the recorded dynamics, not only on the oracle."""
    _assert_coupling_switches_act(g.simulate_family_a)


def test_family_c_coupling_switches_change_the_dynamics():
    """Family C design 2 (formerly a strict xfail): the same check on the
    recorded oscillations. A lag-2 Granger gain of oscillatory signals needs
    a longer run than the 16-trial SMALL config, hence 40 trials."""
    _assert_coupling_switches_act(
        g.simulate_family_c, g.AgentConfig(n_trials=40, n_reafference_pairs=15)
    )


def test_family_c_ignition_and_plasticity_are_effective():
    """Design 2: ignition is an all-or-none episode on a sizeable fraction of
    the run (design 1: < 1 % of samples) and feedback plasticity improves
    choices, as in family A."""
    cfg = g.AgentConfig(n_trials=80, n_reafference_pairs=2)
    acc_on, acc_off = [], []
    for seed in range(3):
        nom = g.simulate_family_c(None, cfg, seed)
        gate = nom.oracle["ignition_gate"]
        assert 0.15 < float(np.mean(gate > 0.5)) < 0.7
        assert len(nom.oracle["ignition_onsets_sec"]) >= 20
        acc_on.append(nom.oracle["accuracy"])
        off = g.simulate_family_c(g.NOMINAL_KNOBS.replace(eta=0.0), cfg, seed)
        acc_off.append(off.oracle["accuracy"])
        env = nom.oracle["hidden_envelope"]
        assert env.shape == nom.ts.shape and np.iscomplexobj(env)
        f = nom.oracle["oscillator_frequency_hz"]
        assert np.all((f >= cfg.sl_freq_hz[0]) & (f <= cfg.sl_freq_hz[1]))
        # The recorded signal is the real part of the rotating hidden state.
        t = (np.arange(nom.n_time) + 1) * nom.dt
        lab = env * np.exp(2j * np.pi * f[:, None] * t[None, :])
        assert np.allclose(lab.real, nom.ts, atol=1e-5)
    assert np.mean(acc_on) > np.mean(acc_off) + 0.1
    assert nom.meta["family_c_design"] == 2 and nom.meta["substrate"] == "stuart_landau"


def test_rest_run():
    s = g.simulate_family_a(None, SMALL.replace(rest_sec=20.0), seed=2)
    assert s.rest_ts is not None and s.rest_ts.shape == (s.n_nodes, 400)
    assert s.meta["has_rest_run"]
    assert np.all(np.isfinite(s.rest_ts))


# --------------------------------------------------------------------------
# Family B
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", g.BINARY_NETWORK_KINDS)
def test_exact_tpm_rows_sum_to_one(kind):
    net = g.family_b_network(kind, n=4)
    assert net.tpm.shape == (16, 16)
    assert np.all(net.tpm >= 0)
    assert np.allclose(net.tpm.sum(axis=1), 1.0)
    assert np.allclose(
        net.stationary
        @ (net.observational_tpm if net.observational_tpm is not None else net.tpm),
        net.stationary,
    )
    assert all(g.state_index(net.states[i]) == i for i in range(16))
    if net.tpm_sbn is not None:
        assert np.allclose(g.sbn_to_sbs(net.tpm_sbn, net.states), net.tpm)


def test_binary_state_order_matches_iim_kernels():
    mm = pytest.importorskip("impact_pipeline.mpc_metrics")
    states = g.binary_states(3)
    keys = mm._iim_subset_key_matrix(states, (0, 1, 2), 2)
    assert np.array_equal(keys, np.arange(8))


def test_binary_connectivity_and_structure():
    n = 4
    ind = g.family_b_network("independent", n)
    assert np.array_equal(ind.connectivity, np.eye(n, dtype=int))
    # Independent units: the TPM factorises into per-unit self transitions.
    p_self = ind.tpm_sbn
    for i in range(n):
        on = ind.states[:, i] == 1
        assert np.allclose(p_self[on, i], p_self[on, i][0])
        assert np.allclose(p_self[~on, i], p_self[~on, i][0])
    ring = g.family_b_network("ring", n)
    assert ring.connectivity.sum() == 3 * n and np.array_equal(
        ring.connectivity, ring.connectivity.T
    )
    allc = g.family_b_network("all_to_all", n)
    assert allc.connectivity.sum() == n * n
    star = g.family_b_network("feedforward_star", n)
    assert star.connectivity[1:, 0].sum() == 0  # nothing feeds unit 0
    assert star.connectivity[0, 1:].sum() == n - 1  # unit 0 drives all
    zero = star.states[:, 0] == 0
    # Unit 0's next state depends only on its own state.
    assert np.allclose(star.tpm_sbn[zero, 0], star.tpm_sbn[zero, 0][0])
    xor = g.family_b_network("xor_loop", n, noise=0.0)
    s = np.array([1, 0, 1, 1])
    nxt = xor.tpm[g.state_index(s)]
    expected = [s[(i - 1) % n] ^ s[(i + 1) % n] for i in range(n)]
    assert nxt[g.state_index(expected)] == pytest.approx(1.0)
    hid = g.family_b_network("hidden_driver", n, driver_weight=1.0)
    assert np.array_equal(hid.connectivity, np.eye(n, dtype=int))
    assert not np.allclose(hid.tpm, hid.observational_tpm, atol=1e-3)
    flat = g.family_b_network("hidden_driver", n, driver_weight=0.0)
    assert np.allclose(flat.tpm, flat.observational_tpm)
    with pytest.raises(ValueError):
        g.family_b_network("xor_loop", 2)
    with pytest.raises(ValueError):
        g.family_b_network("lattice", 4)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_reducible_chains_get_a_valid_stationary_distribution(n):
    # A noise-free XOR loop is deterministic and reducible (several closed
    # classes); the direct linear solve is singular for n = 3, 5.
    net = g.family_b_network("xor_loop", n, noise=0.0)
    pi = net.stationary
    assert np.all(pi >= 0) and pi.sum() == pytest.approx(1.0)
    assert np.allclose(pi @ net.tpm, pi, atol=1e-10)
    # Cesaro limit from the uniform start: the empirical occupancy of the
    # deterministic map averaged over every start state and a long horizon.
    k = 2**n
    occ = np.zeros(k)
    nxt = net.tpm.argmax(axis=1)
    for s0 in range(k):
        s = s0
        for _ in range(4 * k):
            s = nxt[s]
        for _ in range(840):  # multiple of every cycle length for n <= 5
            occ[s] += 1
            s = nxt[s]
    assert np.allclose(pi, occ / occ.sum(), atol=1e-9)
    traj = g.sample_binary_trajectory(net, 20, seed=1)
    assert traj.shape == (20, n)
    if n == 4:
        # The n = 4 XOR map is nilpotent: every state reaches 0000.
        assert pi[0] == pytest.approx(1.0)


def test_near_deterministic_ising_chains_are_handled():
    for kind in g.BINARY_NETWORK_KINDS:
        net = g.family_b_network(kind, 4, beta=50.0)
        T = net.observational_tpm if net.observational_tpm is not None else net.tpm
        assert np.allclose(T.sum(axis=1), 1.0)
        assert np.allclose(net.stationary @ T, net.stationary, atol=1e-8), kind
    # An irreducible chain keeps the exact linear-solve solution.
    ring = g.family_b_network("ring", 4)
    a = ring.tpm.T - np.eye(16)
    a[-1] = 1.0
    b = np.zeros(16)
    b[-1] = 1.0
    assert np.allclose(ring.stationary, np.linalg.solve(a, b), atol=1e-14)


@pytest.mark.parametrize("kind", g.BINARY_NETWORK_KINDS)
def test_sampler_matches_tpm_empirically(kind):
    # Moderate inverse temperature so that most states are visited often.
    net = g.family_b_network(kind, n=3, beta=0.4)
    traj, driver = g.sample_binary_trajectory(net, 60000, seed=7, return_driver=True)
    assert traj.shape == (60000, 3) and set(np.unique(traj)) <= {0, 1}
    emp, visits = g.empirical_tpm(traj)
    ref = net.observational_tpm if net.observational_tpm is not None else net.tpm
    rows = visits >= 2000
    assert rows.sum() >= 4
    assert np.nanmax(np.abs(emp[rows] - ref[rows])) < 0.03
    if kind == "hidden_driver":
        assert driver is not None and set(np.unique(driver)) == {-1, 1}
        stay = np.mean(driver[1:] == driver[:-1])
        assert stay == pytest.approx(net.params["driver_persistence"], abs=0.02)
    else:
        assert driver is None
    again = g.sample_binary_trajectory(net, 1000, seed=7)
    assert np.array_equal(again, traj[:1000])


# --------------------------------------------------------------------------
# Nulls, hypersynchrony, patchwork
# --------------------------------------------------------------------------


def test_null_and_hypersynchronous_systems(nominal_a):
    for kind in g.NULL_KINDS:
        s = g.null_system(kind, nominal_a, seed=3)
        assert s.ts.shape == nominal_a.ts.shape
        pd.testing.assert_frame_equal(s.events, nominal_a.events)
        assert s.meta["family"] == "null" and s.oracle["intended_bits"] == [0] * 5
        assert np.allclose(s.ts.std(axis=1), nominal_a.ts.std(axis=1))
        x = s.ts - s.ts.mean(axis=1, keepdims=True)
        ac1 = np.mean(np.sum(x[:, 1:] * x[:, :-1], axis=1) / np.sum(x * x, axis=1))
        assert ac1 == pytest.approx(0.9 if kind == "ar1" else 0.0, abs=0.06)
    hyp = g.hypersynchronous_system(nominal_a, seed=3)
    corr = np.corrcoef(hyp.ts)
    assert np.mean(corr[np.triu_indices_from(corr, 1)]) > 0.95
    assert hyp.meta["family"] == "hypersynchronous"
    with pytest.raises(ValueError):
        g.null_system("pink", nominal_a)


def test_patchwork_modules_are_disconnected():
    s = simulate_patchwork(None, SMALL, seed=4)
    meta = s.meta
    bearers = meta["principle_bearers"]
    assert list(bearers) == list(PATCHWORK_MODULES)
    all_nodes = sorted(i for v in bearers.values() for i in v)
    assert all_nodes == list(range(s.n_nodes))
    adj = s.oracle["unit_adjacency"]
    for p in PATCHWORK_MODULES:
        inside = np.asarray(bearers[p])
        outside = np.setdiff1d(np.arange(s.n_nodes), inside)
        assert np.all(adj[np.ix_(inside, outside)] == 0)
        assert np.all(adj[np.ix_(outside, inside)] == 0)
    assert s.oracle["modules_connected"] is False
    assert set(meta["workspace_nodes"]) <= set(bearers["NAS"])
    assert set(
        i for v in meta["principle_macro_nodes"]["IIM"].values() for i in v
    ) <= set(bearers["IIM"])
    # Inputs reach only the module implementing each mechanism.
    inp = s.oracle["inputs"]
    tag_rows = np.flatnonzero(np.any(np.abs(inp) > 0, axis=1))
    assert set(np.asarray(bearers["IIM"])).isdisjoint(tag_rows)
    _check_events(s)
    assert s.oracle["ignition_onsets_sec"]
    off = simulate_patchwork(g.NOMINAL_KNOBS.replace(c_int=0.0), SMALL, seed=4)
    assert off.oracle["iim_submodule_adjacency"][1, 0] == 0
    assert s.oracle["iim_submodule_adjacency"][1, 0] > 0


# --------------------------------------------------------------------------
# Witness catalogue and designs
# --------------------------------------------------------------------------


def test_witness_yaml_and_json_twin_agree():
    yaml = pytest.importorskip("yaml")
    with open(witnesses.WITNESS_YAML, encoding="utf-8") as fh:
        y = yaml.safe_load(fh)
    with open(witnesses.WITNESS_JSON, encoding="utf-8") as fh:
        j = json.load(fh)
    assert y == j
    assert witnesses.validate_catalogue(y) == []


def test_witness_loader_falls_back_to_json(monkeypatch):
    monkeypatch.setitem(sys.modules, "yaml", None)
    cat = witnesses.load_witnesses()
    assert cat["source"].endswith("witnesses.json")
    assert len(cat["witnesses"]) >= 12


def test_witness_catalogue_contents():
    cat = witnesses.load_witnesses()
    ids = {w["id"] for w in cat["witnesses"]}
    classes = {w["class"] for w in cat["witnesses"]}
    assert classes == set(witnesses.WITNESS_CLASSES)
    targets = {w["target"] for w in cat["witnesses"] if w["class"] == "single_deficit"}
    assert targets == set(g.PRINCIPLES)
    assert {
        "PC_nominal",
        "N_independent_noise",
        "N_ar1",
        "O_hypersynchronous",
        "PW_patchwork",
    } <= ids
    for w in cat["witnesses"]:
        assert w["counterexample"]["references"], w["id"]
    bad = json.loads(json.dumps(cat["witnesses"][1]))
    bad["intended_pattern"] = [1, 1, 1, 1, 1]
    errs = witnesses.validate_catalogue(
        {"principles": list(g.PRINCIPLES), "witnesses": [bad]}
    )
    assert errs


def test_witnesses_realise_their_mechanisms():
    cat = witnesses.load_witnesses()
    for w in cat["witnesses"]:
        s = witnesses.build_witness_system(w, seed=2, family="A", config=SMALL)
        assert s.meta["witness_id"] == w["id"]
        if w["generator"] == "family_a":
            assert s.oracle["intended_bits"] == w["intended_pattern"], w["id"]
        else:
            assert s.oracle["witness_intended_pattern"] == w["intended_pattern"]
    pc = witnesses.get_witness("PW_patchwork", cat)
    s = witnesses.build_witness_system(pc, seed=2, family="C", config=SMALL)
    assert s.meta["dynamics"] == "stuart_landau" and s.meta["patchwork"]
    with pytest.raises(KeyError):
        witnesses.get_witness("nope", cat)


def test_factorial_cells_realise_all_bit_patterns():
    cells = factorial.factorial_cells()
    assert len(cells) == 32 and len({c["cell_id"] for c in cells}) == 32
    for c in cells:
        assert list(g.knobs_from_dict(c["knobs"]).bits()) == c["bits"]
        assert c["cell_id"] == "b" + "".join(map(str, c["bits"]))
    tasks = factorial.factorial_tasks(range(3))
    assert len(tasks) == 96 and len({t.task_id for t in tasks}) == 96
    sub = factorial.factorial_tasks([0], cells=["b11111", "b01111"], family="C")
    assert {t.cell_id for t in sub} == {"b11111", "b01111"}
    assert {t.generator for t in sub} == {"family_c"}


def test_sweep_levels_and_tasks():
    principle_of = {k: p for p, k in g.SWITCH_FOR_PRINCIPLE.items()}
    for knob in sweeps.SWEEP_KNOBS:
        lv = sweeps.sweep_levels(knob, 10)
        assert len(lv) == 10 and lv == sorted(lv)
        off = g.NOMINAL_KNOBS.replace(**{knob: lv[0]})
        assert off.bits()[g.PRINCIPLES.index(principle_of[knob])] == 0
        on = g.NOMINAL_KNOBS.replace(**{knob: lv[1]})
        assert on.bits()[g.PRINCIPLES.index(principle_of[knob])] == 1
    assert sweeps.sweep_levels("K") == list(range(1, 11))
    assert sweeps.sweep_levels("eta")[0] == 0.0
    tasks = sweeps.sweep_tasks([0, 1], knobs=("eta", "K"))
    assert len(tasks) == 2 * 10 * 2
    k_task = [t for t in tasks if t.sweep_knob == "K"][0]
    assert k_task.knobs["K"] == 1 and g.knobs_from_dict(k_task.knobs).bits()[1] == 0
    df = pd.DataFrame(
        {
            "sweep_knob": ["eta"] * 10,
            "sweep_level": np.linspace(0, 0.6, 10),
            "y": 2.0 * np.linspace(0, 1, 10) + 1.0,
        }
    )
    slopes = sweeps.dose_response_slopes(df, ["y"])
    assert slopes.loc[0, "slope"] == pytest.approx(2.0)
    with pytest.raises(ValueError):
        sweeps.sweep_levels("gain")


def test_knob_validation():
    with pytest.raises(ValueError):
        g.Knobs(eta=-0.1)
    with pytest.raises(ValueError):
        g.Knobs(K=0)
    for bad in ({"eta": float("nan")}, {"g_b": float("inf")}, {"c_int": np.nan}):
        with pytest.raises(ValueError, match="finite"):
            g.Knobs(**bad)
    # Euler steps longer than the unit time constant are refused.
    with pytest.raises(ValueError, match="tau"):
        g.AgentConfig(dt=0.3, tau=0.1)
    with pytest.raises(ValueError):
        g.AgentConfig(sl_substeps=0)
    with pytest.raises(ValueError):
        g.knobs_from_dict({"gain": 1})
    kn = g.Knobs.from_bits([1, 0, 1, 0, 1])
    assert kn.bits() == (1, 0, 1, 0, 1) and kn.K == 1 and kn.c_int == 0
    assert len(g.all_bit_patterns()) == 32
    with pytest.raises(ValueError):
        g.make_system("family_z")
