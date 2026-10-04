# -*- coding: utf-8 -*-
"""Recording device, input declarations and the common input basis of the
v2 estimators (NAS v3, IIM v5): the export never touches the simulation, the
exogeneity rule, the declarations R, H, P, Q10, Q25, J and none with their
own random streams, the basis columns, and tau_c and the lag sets derived
from generator constants. Development seeds only."""
import hashlib
import json
import math

import numpy as np
import pandas as pd
import pytest

from impact_pipeline.bench import generators as g
from impact_pipeline.v2 import declared_inputs as D
from impact_pipeline.v2 import records as REC
from impact_pipeline.v2.provenance import array_sha256

SMALL = g.AgentConfig(n_trials=16, n_reafference_pairs=6)
SEED = 3  # development seed
SYN_SEED = 7  # development seed of the synthetic recordings


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _digest(obj) -> str:
    """A deep digest of nested meta/oracle content (arrays by bytes)."""
    h = hashlib.sha256()

    def feed(x):
        if isinstance(x, dict):
            h.update(b"{")
            for k in sorted(x, key=str):
                h.update(repr(k).encode())
                feed(x[k])
            h.update(b"}")
        elif isinstance(x, (list, tuple)):
            h.update(b"[")
            for v in x:
                feed(v)
            h.update(b"]")
        elif isinstance(x, np.ndarray):
            h.update(array_sha256(x).encode())
        elif isinstance(x, pd.DataFrame):
            h.update(_frame_digest(x).encode())
        else:
            h.update(f"{type(x).__name__}:{x!r}".encode())

    feed(obj)
    return h.hexdigest()


def _frame_digest(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(repr(list(df.columns)).encode())
    h.update(repr([str(t) for t in df.dtypes]).encode())
    h.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    h.update(df.to_csv().encode())
    return h.hexdigest()


def fingerprint(system) -> dict:
    return {
        "ts": array_sha256(system.ts),
        "events": _frame_digest(system.events),
        "meta": _digest(system.meta),
        "oracle": _digest(system.oracle),
        "rest_ts": None if system.rest_ts is None else array_sha256(system.rest_ts),
    }


def export_everything(system):
    """Every export path of the module on one system."""
    rec = D.record_inputs(system)
    rec.events_table()
    rec.sha256()
    out = []
    for endo in (False, True):
        for did, dec in D.declare_all(rec, include_endogenous=endo).items():
            dec.sha256()
            dec.events_table()
            for est in D.ESTIMATORS:
                try:
                    tau_c = D.coupling_timescale(system).tau_c_sec
                except D.DeclaredInputsError:
                    tau_c = D.TAU_C_DEFAULT_SEC
                b = D.input_basis(dec, tau_c=tau_c, estimator=est)
                b.sha256()
                out.append(b)
    return rec, out


def _canon(x):
    """Comparable form of a v1 estimator result: exact floats (NaN as a
    token), numpy scalars and arrays as Python objects, wall times dropped."""
    if isinstance(x, dict):
        return {k: _canon(v) for k, v in x.items() if not str(k).endswith("seconds")}
    if isinstance(x, (list, tuple)):
        return [_canon(v) for v in x]
    if isinstance(x, np.ndarray):
        return _canon(x.tolist())
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return "nan" if math.isnan(v) else v
    return x


def synthetic_system(n_time=40000, dt=0.05, k=6, seed=SYN_SEED, dwell=(20, 60),
                     events=None, slow=True, extra_oracle=None):
    """A recording with many context switches (consecutive labels differ),
    an optional slow phase and a given events table; nothing is simulated."""
    rng = np.random.default_rng(11)  # fixture construction only
    state = np.zeros(n_time, dtype=np.int64)
    t, cur = 0, int(rng.integers(0, k))
    while t < n_time:
        d = int(rng.integers(dwell[0], dwell[1] + 1))
        state[t:t + d] = cur
        t += d
        if k > 1:
            cur = int(rng.choice([v for v in range(k) if v != cur]))
    oracle = {"context_state": state}
    if slow:
        period = g.AgentConfig().slow_period_samples
        oracle["slow_phase"] = np.mod(2 * np.pi * np.arange(n_time) / period + 0.4,
                                      2 * np.pi)
    oracle.update(extra_oracle or {})
    if events is None:
        events = pd.DataFrame(columns=["onset", "duration", "trial_type", "value"])
    meta = {"dt": dt, "seed": seed, "knobs": {"K": k}, "dynamics": "rate",
            "config": g.AgentConfig(dt=dt).to_dict(), "family": "A"}
    return g.BenchSystem(ts=np.zeros((2, n_time)), events=events, meta=meta,
                         oracle=oracle)


@pytest.fixture(scope="module")
def agent():
    return g.simulate_family_a(None, SMALL, seed=SEED)


@pytest.fixture(scope="module")
def recorded(agent):
    return D.record_inputs(agent)


@pytest.fixture(scope="module")
def long_recording():
    return D.record_inputs(synthetic_system())


# --------------------------------------------------------------------------
# the export never touches the simulation
# --------------------------------------------------------------------------
def _family_c():
    return g.simulate_family_c(None, SMALL, seed=SEED)


def _common_driver():
    from impact_pipeline.bench.adversarial import make_adversarial

    return make_adversarial("common_driver", SMALL, seed=SEED)


def _bold_agent():
    from impact_pipeline.bench.forward import bold_forward

    return bold_forward(g.simulate_family_a(None, SMALL, seed=SEED), tr=2.0, seed=SEED)


def _with_rest():
    return g.simulate_family_a(None, SMALL.replace(rest_sec=10.0), seed=SEED)


@pytest.mark.parametrize("build", [
    lambda: g.simulate_family_a(None, SMALL, seed=SEED), _family_c, _common_driver,
    _bold_agent, _with_rest,
], ids=["family_a", "family_c", "common_driver", "bold_family_a", "with_rest_run"])
def test_export_leaves_the_system_bit_identical(build):
    s = build()
    events_obj, ts_obj = s.events, s.ts
    before = fingerprint(s)
    export_everything(s)
    assert fingerprint(s) == before
    assert s.events is events_obj and s.ts is ts_obj


def test_v1_estimators_return_bit_identical_output_with_and_without_export():
    """Every v1 estimator (the in-memory v1 path of the bench) gives the same
    output, null moments and jackknife SE included, on a system whose inputs
    were exported and on an untouched run of the same seed."""
    from impact_pipeline.bench import export

    plain = g.simulate_family_a(None, SMALL, seed=SEED)
    exported = g.simulate_family_a(None, SMALL, seed=SEED)
    export_everything(exported)
    assert array_sha256(plain.ts) == array_sha256(exported.ts)
    pd.testing.assert_frame_equal(plain.events, exported.events, check_exact=True)
    kw = {"null_surrogates": 2, "se_groups": 2}
    a = export.run_in_memory(plain, **kw)
    b = export.run_in_memory(exported, **kw)
    assert set(a["components"]) == set(g.PRINCIPLES)
    for comp in a["components"].values():
        assert comp["defined"] and comp["n_null"] >= 1
        assert math.isfinite(comp["se"])
    assert _canon(a) == _canon(b)


def test_export_neither_aliases_nor_freezes_system_arrays():
    """The read-only arrays of the record are copies: the system's own arrays
    stay writeable, so a later v1 step that writes into them still works."""
    s = _common_driver()
    a = g.simulate_family_a(None, SMALL, seed=SEED)
    for system in (s, a):
        arrays = {"ts": system.ts}
        arrays.update({k: v for k, v in system.oracle.items()
                       if isinstance(v, np.ndarray)})
        flags = {k: v.flags.writeable for k, v in arrays.items()}
        rec = D.record_inputs(system)
        assert {k: v.flags.writeable for k, v in arrays.items()} == flags
        for ch in rec.channels.values():
            assert not ch.flags.writeable
            assert not any(np.shares_memory(ch, v) for v in arrays.values())


def test_recorded_arrays_are_read_only_copies(agent, recorded):
    with pytest.raises(ValueError):
        recorded.channels["slow_phase_sin"][0] = 1.0
    assert recorded.task_events is not agent.events
    pd.testing.assert_frame_equal(recorded.task_events, agent.events, check_exact=True)
    b = D.system_basis(agent, "R", recorded=recorded)
    with pytest.raises(ValueError):
        b.values[0, 0] = 1.0


# --------------------------------------------------------------------------
# the recording device
# --------------------------------------------------------------------------
def test_context_cues_one_per_switch_with_the_new_label(agent, recorded):
    state = np.asarray(agent.oracle["context_state"])
    dt = agent.dt
    cues = recorded.cues()
    assert list(cues.columns) == list(D.CUE_COLUMNS)
    assert (cues["trial_type"] == D.CONTEXT_CUE).all()
    switches = np.flatnonzero(state[1:] != state[:-1]) + 1
    assert len(cues) == switches.size + 1  # plus the context at the start
    idx = np.rint(cues["onset"].to_numpy() / dt).astype(int)
    assert idx[0] == 0 and np.array_equal(idx[1:], switches)
    assert np.array_equal(cues["value"].to_numpy(), state[idx])
    assert np.all(cues["value"].to_numpy()[1:] != cues["value"].to_numpy()[:-1])
    # durations tile the run: each cue holds until the next one
    ends = np.rint((cues["onset"] + cues["duration"]).to_numpy() / dt).astype(int)
    assert np.array_equal(ends[:-1], idx[1:]) and ends[-1] == agent.n_time
    assert recorded.cue_alphabets[D.CONTEXT_CUE] == tuple(range(g.NOMINAL_KNOBS.K))
    assert recorded.n_time == agent.n_time and recorded.dt == dt
    assert recorded.seed == SEED


def test_slow_phase_channels_on_the_sample_grid(agent, recorded):
    ph = np.asarray(agent.oracle["slow_phase"])
    assert recorded.channel_groups == {D.SLOW_PHASE: D.SLOW_PHASE_CHANNELS}
    assert np.array_equal(recorded.channels["slow_phase_sin"], np.sin(ph))
    assert np.array_equal(recorded.channels["slow_phase_cos"], np.cos(ph))
    assert recorded.channels["slow_phase_sin"].shape == (agent.n_time,)


def test_events_table_merges_task_events_and_cues(agent, recorded):
    table = recorded.events_table()
    assert len(table) == len(agent.events) + len(recorded.cue_events)
    assert table["onset"].is_monotonic_increasing
    assert set(table["trial_type"]) == set(agent.events["trial_type"]) | {D.CONTEXT_CUE}


def test_a_single_context_records_one_cue():
    s = g.simulate_family_a(g.OFF_KNOBS, SMALL, seed=SEED)  # K = 1
    rec = D.record_inputs(s)
    cues = rec.cues()
    assert len(cues) == 1 and cues.loc[0, "onset"] == 0.0
    assert math.isclose(cues.loc[0, "duration"], s.n_time * s.dt)
    assert rec.cue_alphabets[D.CONTEXT_CUE] == (0,)
    b = D.system_basis(s, "R")
    assert not any(c.startswith(D.CONTEXT_CUE) for c in b.channels)
    assert ("context_cue=0", "reference") in b.dropped
    q = D.declare(rec, "Q25")  # no other label exists: nothing is relabelled
    assert q.details["label_error"]["n_relabelled"] == 0


def test_systems_without_inputs_record_only_their_events():
    from impact_pipeline.bench import whole_brain as wb

    nul = g.make_system("null_ar1", None, SMALL, seed=SEED)
    rec = D.record_inputs(nul)
    assert rec.cue_alphabets == {} and rec.channels == {}
    pd.testing.assert_frame_equal(rec.task_events, nul.events, check_exact=True)
    assert D.declare(rec, "R").event_types == D.declare(rec, "H").event_types
    # without cues the held-out declarations have nothing to corrupt: they
    # declare the content of R (no stream is drawn) under 'partial'
    r = D.declare(rec, "R")
    rb = D.input_basis(r, tau_c=0.1)
    for did in ("P", "Q10", "Q25", "J"):
        dec = D.declare(rec, did)
        assert dec.shared_inputs == "partial"
        assert "label_error" not in dec.details and "onset_jitter" not in dec.details
        pd.testing.assert_frame_equal(dec.events, r.events, check_exact=True)
        np.testing.assert_array_equal(D.input_basis(dec, tau_c=0.1).values, rb.values)
    # systems built on a family-A template keep its tau_c
    for system in (nul, _common_driver()):
        tc = D.coupling_timescale(system)
        assert (tc.tau_c_sec, tc.constant) == (0.1, "AgentConfig.tau")
    w = wb.simulate_whole_brain(wb.WholeBrainConfig(duration_sec=2.0), seed=SEED)
    rw = D.record_inputs(w)
    assert len(rw.task_events) == 0 and rw.cue_alphabets == {} and rw.channels == {}
    for did in ("R", "H", "none"):
        b = D.system_basis(w, did)
        assert b.values.shape == (0, w.n_time) and b.n_columns == 0
        assert b.orthonormal_span().shape == (w.n_time - b.first_complete_sample, 0)


def test_common_driver_is_a_recorded_driver():
    s = _common_driver()
    rec = D.record_inputs(s)
    assert rec.channel_groups == {D.COMMON_DRIVER: (D.COMMON_DRIVER,)}
    assert np.array_equal(rec.channels[D.COMMON_DRIVER], s.oracle["driver"])
    for did in ("R", "P", "Q10", "Q25", "J"):
        assert list(D.declare(rec, did).channels) == [D.COMMON_DRIVER]
    for did in ("H", "none"):
        assert D.declare(rec, did).channels == {}


def test_extra_states_and_channels_are_recorded_drivers(agent):
    rng = np.random.default_rng(1)
    lev = np.repeat(rng.integers(0, 3, size=agent.n_time // 40 + 1), 40)[:agent.n_time]
    lev[:40] = -1  # any integer label, including negative ones
    drift = rng.standard_normal(agent.n_time).cumsum()
    rec = D.record_inputs(agent, extra_states={"driver_cue": lev},
                          extra_channels={"drift": drift})
    assert rec.driver_cue_types == ("driver_cue",) and rec.driver_groups == ("drift",)
    for did in ("R", "P", "Q25", "J"):
        dec = D.declare(rec, did)
        assert "driver_cue" in dec.state_types and "drift" in dec.channels
    q = D.declare(rec, "Q25")
    np.testing.assert_array_equal(q.cues("driver_cue")["value"].to_numpy(),
                                  rec.cues("driver_cue")["value"].to_numpy())
    for did in ("H", "none"):
        dec = D.declare(rec, did)
        assert "driver_cue" not in dec.state_types and "drift" not in dec.channels
    b = D.input_basis(D.declare(rec, "R"), tau_c=0.1)
    assert {"driver_cue=0", "driver_cue=1", "driver_cue=2"} <= set(b.channels)
    assert b.reference_levels["driver_cue"] == -1
    # the oracle convention does the same
    s2 = g.BenchSystem(ts=agent.ts, events=agent.events, meta=agent.meta,
                       oracle={**agent.oracle,
                               D.RECORDABLE_STATES_KEY: {"driver_cue": lev},
                               D.RECORDABLE_CHANNELS_KEY: {"drift": drift}})
    assert D.record_inputs(s2).sha256() == rec.sha256()


def _labels(n):
    return np.zeros(n, dtype=int)


@pytest.mark.parametrize("key,name,make,match", [
    ("extra_states", "goal_cue", _labels, "already recorded"),
    ("extra_states", "context_cue", _labels, "already recorded"),
    ("extra_channels", "slow_phase_sin", lambda n: np.zeros(n), "already recorded"),
    ("extra_channels", "slow_phase", lambda n: np.zeros(n), "already recorded"),
    ("extra_states", "slow_phase", _labels, "already recorded"),
    ("extra_states", "slow_phase_cos", _labels, "already recorded"),
    ("extra_states", "bad name", _labels, "identifier"),
    ("extra_states", "x", lambda n: np.full(n, 0.5), "integers"),
    ("extra_states", "x", lambda n: _labels(7), "do not map"),
    ("extra_channels", "x", lambda n: np.full(n, np.nan), "non-finite"),
])
def test_record_inputs_refuses_bad_drivers(agent, key, name, make, match):
    with pytest.raises(D.DeclaredInputsError, match=match):
        D.record_inputs(agent, **{key: {name: make(agent.n_time)}})


def test_inputs_are_placed_on_a_coarser_observation_grid():
    """Forward-modelled BOLD: cues keep their onsets in seconds, continuous
    channels are point-sampled like the BOLD signal."""
    src = g.simulate_family_a(None, SMALL, seed=SEED)
    bold = _bold_agent()
    step = int(round(bold.dt / src.dt))
    rec = D.record_inputs(bold)
    assert rec.n_time == bold.n_time and rec.dt == bold.dt
    assert rec.source["grid_steps"] == {"context_cue": step, "slow_phase_sin": step,
                                        "slow_phase_cos": step}
    np.testing.assert_array_equal(rec.channels["slow_phase_sin"],
                                  np.sin(src.oracle["slow_phase"])[::step])
    pd.testing.assert_frame_equal(rec.cue_events, D.record_inputs(src).cue_events,
                                  check_exact=True)
    b = D.system_basis(bold, "R", estimator="IIM")
    assert b.values.shape[1] == bold.n_time and b.lags == (0, 1)
    broken = g.BenchSystem(ts=bold.ts, events=bold.events,
                           meta={k: v for k, v in bold.meta.items() if k != "config"},
                           oracle=bold.oracle)
    with pytest.raises(D.DeclaredInputsError, match="unknown grid"):
        D.record_inputs(broken)


# --------------------------------------------------------------------------
# exogeneity rule
# --------------------------------------------------------------------------
def test_exogeneity_rule_excludes_endogenous_events_by_default(agent, recorded):
    assert set(agent.events["trial_type"]) == set(g.TRIAL_TYPES)
    for did in ("R", "H", "P", "Q10", "Q25", "J"):
        dec = D.declare(recorded, did)
        assert dec.event_types == ("goal_cue", "stimulus", "other_caused")
        assert set(dec.excluded_event_types) == {"response", "feedback", "action",
                                                 "self_caused"}
        assert not set(dec.events["trial_type"]) & set(D.ENDOGENOUS_EVENT_TYPES)
    b = D.input_basis(D.declare(recorded, "H"), tau_c=0.1)
    assert b.channels == ("goal_cue=0", "goal_cue=1", "stimulus=0", "stimulus=1",
                          "other_caused=0", "other_caused=1")


def test_all_events_sensitivity_declares_every_event_type(recorded):
    dec = D.declare(recorded, "R", include_endogenous=True)
    assert dec.label == "R+all_events" and dec.excluded_event_types == ()
    assert set(dec.event_types) == set(g.TRIAL_TYPES)
    assert dec.event_types[:3] == D.EXOGENOUS_TASK_EVENT_TYPES
    b = D.input_basis(dec, tau_c=0.1)
    assert "feedback=-1" in b.channels and "response=1" in b.channels
    assert b.declaration_id == "R+all_events"


def test_unknown_event_types_are_excluded_unless_all_events():
    assert D.exogenous_event_types(["stimulus", "pupil_blink", "response"]) == (
        ("stimulus",), ("pupil_blink", "response"))
    assert D.exogenous_event_types(["stimulus", "pupil_blink", "response"],
                                   include_endogenous=True) == (
        ("stimulus", "pupil_blink", "response"), ())
    assert set(D.EXOGENOUS_INPUTS) == {"goal_cue", "stimulus", "other_caused",
                                       D.CONTEXT_CUE, D.SLOW_PHASE}


# --------------------------------------------------------------------------
# declarations
# --------------------------------------------------------------------------
EXPECTED = {
    # id: (shared_inputs, context cues, slow phase, held out)
    "R": ("complete", True, True, False),
    "H": ("partial", False, False, False),
    "P": ("partial", True, False, True),
    "Q10": ("partial", True, True, True),
    "Q25": ("partial", True, True, True),
    "J": ("partial", True, True, True),
    "none": ("none", False, False, False),
}


def test_the_declaration_table():
    assert D.DECLARATION_IDS == ("R", "H", "P", "Q10", "Q25", "J", "none")
    assert set(D.DECLARATION_IDS) <= set(REC.KNOWN_DECLARATIONS)
    assert D.HELD_OUT_DECLARATIONS == ("P", "Q10", "Q25", "J")
    q = {d: D.DECLARATIONS[d].label_error_q for d in D.DECLARATION_IDS}
    assert q == {"R": 0, "H": 0, "P": 0, "Q10": 0.10, "Q25": 0.25, "J": 0, "none": 0}
    jit = {d: D.DECLARATIONS[d].onset_jitter_sec for d in D.DECLARATION_IDS}
    assert jit == {"R": 0, "H": 0, "P": 0, "Q10": 0, "Q25": 0, "J": 0.25, "none": 0}
    with pytest.raises(D.DeclaredInputsError, match="unknown declaration"):
        D.declaration_spec("R2")


@pytest.mark.parametrize("did", D.DECLARATION_IDS)
def test_each_declaration_selects_its_inputs(recorded, did):
    shared, ctx, slow, held_out = EXPECTED[did]
    dec = D.declare(recorded, did)
    assert dec.declaration_id == did and dec.shared_inputs == shared
    assert dec.spec.held_out is held_out
    assert (D.CONTEXT_CUE in dec.state_types) is ctx
    assert (set(D.SLOW_PHASE_CHANNELS) <= set(dec.channels)) is slow
    assert bool(len(dec.events)) is (did != "none")
    ident = dec.identifiability()
    assert ident == REC.validate_identifiability(ident)
    assert ident["shared_inputs"] == shared and ident["observation"] == "direct"
    assert ident["hub_privileged"] == "not_tested"
    assert dec.identifiability("sensor_mixing", True)["observation"] == "sensor_mixing"
    # usable as the declaration id of a v2 component record
    comp = REC.ComponentRecord(principle="NAS", status="UNDEFINED",
                               reason="SAMPLING_UNRESOLVED",
                               estimator_version="nas-v3-2026.10",
                               declaration_id=dec.label, observation_stage="source",
                               protocol_id="A-R", protocol_hash="0" * 64,
                               identifiability=ident)
    assert comp.declaration_id == did


def test_held_out_declarations_change_only_their_part(recorded):
    def task_rows(dec):
        ev = dec.events
        return ev[ev["coding"] == D.CODING_EVENT].reset_index(drop=True)

    r = D.declare(recorded, "R")
    for did in ("P", "Q10", "Q25", "J"):
        dec = D.declare(recorded, did)
        pd.testing.assert_frame_equal(task_rows(r), task_rows(dec), check_exact=True)
        if did != "P":
            assert list(dec.channels) == list(r.channels)
    p = D.declare(recorded, "P")
    pd.testing.assert_frame_equal(p.cues(), r.cues(), check_exact=True)
    assert list(p.channels) == []
    # R reproduces the recorded cues exactly
    rc = recorded.cues()
    np.testing.assert_array_equal(r.cues()["onset"].to_numpy(), rc["onset"].to_numpy())
    np.testing.assert_array_equal(r.cues()["value"].to_numpy(dtype=int),
                                  rc["value"].to_numpy())


def test_label_errors_come_from_stream_41(long_recording):
    rec = long_recording
    cues = rec.cues()
    labels = cues["value"].to_numpy()
    n = labels.size
    assert n > 900
    alphabet = rec.cue_alphabets[D.CONTEXT_CUE]
    rng = np.random.default_rng(np.random.SeedSequence([SYN_SEED, 41]))
    u = rng.random(n)
    pick = rng.integers(0, len(alphabet) - 1, size=n)
    hits = {}
    for did, q in (("Q10", 0.10), ("Q25", 0.25)):
        dec = D.declare(rec, did)
        new = dec.cues()["value"].to_numpy(dtype=int)
        np.testing.assert_array_equal(dec.cues()["onset"].to_numpy(),
                                      cues["onset"].to_numpy())
        want_hit = np.flatnonzero(u < q)
        changed = np.flatnonzero(new != labels)
        np.testing.assert_array_equal(changed, want_hit)
        assert dec.details["label_error"]["relabelled"] == [int(j) for j in want_hit]
        assert dec.details["label_error"]["stream"] == [SYN_SEED, 41]
        for j in want_hit:
            others = [a for a in alphabet if a != labels[j]]
            assert new[j] == others[pick[j]] and new[j] != labels[j]
        rate = want_hit.size / n
        assert abs(rate - q) < 4.5 * math.sqrt(q * (1 - q) / n)
        hits[did] = (set(changed.tolist()), new)
    # nested dose: the q = 0.10 errors are a subset of the q = 0.25 errors,
    # with the same wrong labels
    assert hits["Q10"][0] <= hits["Q25"][0]
    for j in hits["Q10"][0]:
        assert hits["Q10"][1][j] == hits["Q25"][1][j]


def test_label_errors_do_not_use_another_stream(long_recording):
    labels = long_recording.cues()["value"].to_numpy()
    dec = D.declare(long_recording, "Q25")
    changed = np.flatnonzero(dec.cues()["value"].to_numpy(dtype=int) != labels)
    n = labels.size
    for key in ([SYN_SEED], [SYN_SEED, 42], [SYN_SEED, 43], [SYN_SEED, 1]):
        rng = np.random.default_rng(np.random.SeedSequence(key))
        assert not np.array_equal(np.flatnonzero(rng.random(n) < 0.25), changed)
    again = D.declare(long_recording, "Q25")
    assert again.sha256() == dec.sha256()
    other_seed = D.declare(long_recording, "Q25", seed=SYN_SEED + 1)
    assert other_seed.sha256() != dec.sha256()


def test_cue_onset_jitter_comes_from_stream_42(long_recording):
    rec = long_recording
    cues = rec.cues()
    n = len(cues)
    rng = np.random.default_rng(np.random.SeedSequence([SYN_SEED, 42]))
    shift = rng.uniform(-0.25, 0.25, size=n)
    t_last = (rec.n_time - 1) * rec.dt
    want = np.clip(np.round(cues["onset"].to_numpy() + shift, 6), 0.0, t_last)
    dec = D.declare(rec, "J")
    got = dec.cues()
    assert dec.details["onset_jitter"]["stream"] == [SYN_SEED, 42]
    assert dec.details["onset_jitter"]["reordered"] is False  # dwell >= 1 s
    np.testing.assert_array_equal(got["onset"].to_numpy(), want)
    np.testing.assert_array_equal(got["value"].to_numpy(dtype=int),
                                  cues["value"].to_numpy())
    # every cue holds until the next jittered cue; the last until the end
    on, du = got["onset"].to_numpy(), got["duration"].to_numpy()
    np.testing.assert_allclose(on[:-1] + du[:-1], on[1:], atol=1e-6)
    assert math.isclose(on[-1] + du[-1], rec.n_time * rec.dt, abs_tol=1e-6)
    d = on[1:] - cues["onset"].to_numpy()[1:]  # the first cue is clipped at 0
    assert np.all(np.abs(d) <= 0.25 + 1e-9)
    assert abs(d.mean()) < 0.03 and abs(d.std() - 0.5 / math.sqrt(12)) < 0.02
    assert dec.details["onset_jitter"]["max_abs_shift_sec"] <= 0.25
    # the label stream is untouched by the jitter and vice versa
    assert D.declare(rec, "R").sha256() == D.declare(rec, "R").sha256()
    np.testing.assert_array_equal(D.declare(rec, "Q25").cues()["onset"].to_numpy(),
                                  cues["onset"].to_numpy())


def test_jittered_cues_are_rendered_as_states(long_recording):
    dec = D.declare(long_recording, "J")
    ch = D.input_channels(dec)
    r = D.input_channels(D.declare(long_recording, "R"))
    assert ch.names == r.names
    ctx = [i for i, k in enumerate(ch.kinds) if k == "state"]
    # each sample carries at most one non-reference label, and the label
    # changes only within 0.25 s (5 samples) of a recorded switch
    assert np.all(ch.values[ctx].sum(axis=0) <= 1)
    diff = np.flatnonzero(np.any(ch.values[ctx] != r.values[ctx], axis=0))
    switches = np.rint(long_recording.cues()["onset"].to_numpy() / 0.05).astype(int)
    nearest = np.min(np.abs(diff[:, None] - switches[None, :]), axis=1)
    assert diff.size > 0 and nearest.max() <= 5


def test_jitter_reorders_close_cues_and_spares_recorded_drivers():
    """Cues closer than the jitter range can change order: each label moves
    with its own cue, the cues still tile the run and code one state per
    sample. Recorded driver tracks are neither jittered nor relabelled."""
    s = synthetic_system(n_time=4000, k=3, dwell=(4, 4), slow=False)  # 0.2 s
    drv = np.repeat(np.arange(100) % 2, 40)
    rec = D.record_inputs(s, extra_states={"driver_cue": drv})
    cues = rec.cues()
    rng = np.random.default_rng(np.random.SeedSequence([SYN_SEED, 42]))
    shift = rng.uniform(-0.25, 0.25, size=len(cues))
    t_last = (rec.n_time - 1) * rec.dt
    new = np.clip(np.round(cues["onset"].to_numpy() + shift, 6), 0.0, t_last)
    order = np.argsort(new, kind="stable")
    dec = D.declare(rec, "J")
    got = dec.cues()
    assert dec.details["onset_jitter"]["reordered"] is True
    on, du = got["onset"].to_numpy(), got["duration"].to_numpy()
    np.testing.assert_array_equal(on, new[order])
    np.testing.assert_array_equal(got["value"].to_numpy(dtype=int),
                                  cues["value"].to_numpy()[order])
    np.testing.assert_allclose(on[:-1] + du[:-1], on[1:], atol=1e-6)
    assert math.isclose(on[-1] + du[-1], rec.n_time * rec.dt, abs_tol=1e-6)
    ch = D.input_channels(dec)
    ctx = [i for i, n in enumerate(ch.names) if n.startswith(D.CONTEXT_CUE)]
    assert ch.values[ctx].sum(axis=0).max() == 1
    r = D.declare(rec, "R")
    for did in ("J", "Q25"):
        pd.testing.assert_frame_equal(D.declare(rec, did).cues("driver_cue"),
                                      r.cues("driver_cue"), check_exact=True)


def test_streams_need_a_seed(agent):
    meta = {k: v for k, v in agent.meta.items() if k != "seed"}
    rec = D.record_inputs(g.BenchSystem(ts=agent.ts, events=agent.events, meta=meta,
                                        oracle=agent.oracle))
    assert rec.seed is None
    D.declare(rec, "R")
    for did in ("Q10", "J"):
        with pytest.raises(D.DeclaredInputsError, match="seed"):
            D.declare(rec, did)
    assert D.declare(rec, "J", seed=SEED).seed == SEED


def test_declaration_protocol_entries_are_stable_and_distinct():
    entries = {d: D.declaration_protocol_entry(d) for d in D.DECLARATION_IDS}
    texts = {d: json.dumps(e, sort_keys=True, allow_nan=False)
             for d, e in entries.items()}
    assert len(set(texts.values())) == len(texts)
    assert texts == {d: json.dumps(D.declaration_protocol_entry(d), sort_keys=True)
                     for d in D.DECLARATION_IDS}
    e = entries["Q10"]
    assert e["shared_inputs_declaration"] == "Q10"
    assert e["streams"] == {"label_error": 41, "cue_jitter": 42}
    assert e["basis"]["null_order"] == "shift_then_project"
    assert e["basis"]["filtered_copies_tau_multipliers"] == [1.0, 3.0, 10.0]
    assert e["version"] == D.DECLARED_INPUTS_VERSION
    assert D.declaration_protocol_entry("R", include_endogenous=True) != entries["R"]


def test_hashes_identify_the_declared_content(recorded):
    r1, r2 = D.declare(recorded, "R"), D.declare(recorded, "R")
    assert r1.sha256() == r2.sha256()
    assert len({D.declare(recorded, d).sha256() for d in D.DECLARATION_IDS}) == 7
    b1 = D.input_basis(r1, tau_c=0.1)
    assert b1.sha256() == D.input_basis(r2, tau_c=0.1).sha256()
    assert b1.sha256() != D.input_basis(r1, tau_c=0.1, estimator="IIM",
                                        lags=(0, 1)).sha256()


# --------------------------------------------------------------------------
# the input basis
# --------------------------------------------------------------------------
def _one_event_system(n_time=400, dt=0.05, onset=1.0, duration=0.3):
    ev = pd.DataFrame({"onset": [onset], "duration": [duration],
                       "trial_type": ["stimulus"], "value": [1.0]})
    return synthetic_system(n_time=n_time, dt=dt, k=1, events=ev, slow=True)


def _recursion(u, dt, tau):
    a = math.exp(-dt / tau)
    y = np.zeros_like(u)
    prev = 0.0
    for t in range(u.size):
        prev = a * prev + (1.0 - a) * u[t]
        y[t] = prev
    return y


def test_basis_columns_are_lags_and_causal_filtered_copies():
    s = _one_event_system()
    dt, n = 0.05, 400
    dec = D.declare(D.record_inputs(s), "H")
    b = D.input_basis(dec, tau_c=0.1)
    assert b.channels == ("stimulus=1",) and b.channel_kinds == ("event",)
    assert b.lags == (0, 1, 2) and b.taus == (0.1, 0.3, 1.0)
    assert b.n_columns == 1 * 4 * 3 and b.first_complete_sample == 2
    assert b.columns == tuple(
        f"stimulus=1{tag}@lag{lag}" for tag in ("", "~0.1s", "~0.3s", "~1s")
        for lag in (0, 1, 2))
    u = np.zeros(n)
    u[20:26] = 1.0  # [1.0 s, 1.3 s) at 20 Hz
    for j, (tau, lag) in enumerate((t, lg) for t in (None, 0.1, 0.3, 1.0)
                                   for lg in (0, 1, 2)):
        base = u if tau is None else _recursion(u, dt, tau)
        want = np.zeros(n)
        want[lag:] = base[:n - lag]
        np.testing.assert_allclose(b.values[j], want, rtol=0, atol=1e-13)
        assert b.column_tau[j] == tau and b.column_lag[j] == lag
        assert not b.values[j, :20 + lag].any()  # causal: nothing before the input
    assert b.column_indices(channel="stimulus=1", tau=None).tolist() == [0, 1, 2]
    assert b.column_indices(tau=1.0, lag=2).tolist() == [11]
    # filtered copies of a step converge to the step height
    step = D.lowpass(np.ones(400), dt, 0.3)
    assert step[0] == pytest.approx(1 - math.exp(-dt / 0.3))
    assert step[-1] == pytest.approx(1)


def test_iim_uses_every_lag_up_to_l_max():
    s = _one_event_system()
    dec = D.declare(D.record_inputs(s), "H")
    b = D.input_basis(dec, tau_c=0.2, estimator="IIM")
    assert b.lags == (0, 1, 2, 3, 4) and b.taus == (0.2, 0.6, 2.0)
    assert b.n_columns == 4 * 5
    nas = D.input_basis(dec, tau_c=0.2, estimator="NAS")
    assert nas.lags == (0, 1, 2, 3, 4)  # geomspace(1, 4, 4) rounds to 1..4
    assert D.input_basis(dec, tau_c=0.3, estimator="NAS").lags == (0, 1, 2, 3, 6)
    with pytest.raises(D.DeclaredInputsError):
        D.input_basis(dec)
    with pytest.raises(D.DeclaredInputsError):
        D.input_basis(dec, lags=(0, 1, 1), taus=(0.1,))
    with pytest.raises(D.DeclaredInputsError):
        D.input_basis(dec, tau_c=0.1, estimator="PDI")


def test_continuous_channels_are_z_scored(agent, recorded):
    b = D.input_basis(D.declare(recorded, "R"), tau_c=0.1)
    for name in D.SLOW_PHASE_CHANNELS:
        (j,) = b.column_indices(channel=name, tau=None, lag=0)
        col = b.values[j]
        assert abs(col.mean()) < 1e-12 and col.std() == pytest.approx(1.0)
        raw = recorded.channels[name]
        np.testing.assert_allclose(col, (raw - raw.mean()) / raw.std(), atol=1e-12)


def test_the_full_basis_of_a_family_a_run(agent, recorded):
    plan = D.lag_plan(agent)
    present = sorted(set(np.asarray(agent.oracle["context_state"]).tolist()))
    want = {"R": 6 + (len(present) - 1) + 2, "H": 6, "P": 6 + len(present) - 1,
            "none": 0}
    for did, n_u0 in want.items():
        b = D.system_basis(agent, did, recorded=recorded)
        assert b.n_channels == n_u0
        assert b.n_columns == n_u0 * 4 * (1 + len(plan.nas_lags))
        assert b.values.shape == (b.n_columns, agent.n_time)
        assert b.shared_inputs == EXPECTED[did][0]


def test_cue_states_are_reference_coded(agent, recorded):
    b = D.input_basis(D.declare(recorded, "R"), tau_c=0.1)
    state = np.asarray(agent.oracle["context_state"])
    present = sorted(set(state.tolist()))
    ref = present[0]
    assert b.reference_levels == {D.CONTEXT_CUE: ref}
    names = [f"context_cue={k}" for k in present[1:]]
    assert [c for c in b.channels if c.startswith("context_cue")] == names
    total = np.zeros(agent.n_time)
    for k, name in zip(present[1:], names):
        (j,) = b.column_indices(channel=name, tau=None, lag=0)
        np.testing.assert_array_equal(b.values[j], (state == k).astype(float))
        total += b.values[j]
    np.testing.assert_array_equal(total + (state == ref), np.ones(agent.n_time))


def test_lagged_filtered_copies_are_collinear_and_the_span_is_exact(agent, recorded):
    """``u~(t) = a u~(t - 1) + (1 - a) u(t)`` makes every lagged filtered copy
    but the last a combination of the raw lags: each event or state channel
    adds ``1 + |L| + 3`` dimensions, the slow phase (a sinusoid) two plus one
    start-up transient per filter. The orthonormal span reproduces least
    squares on the raw columns."""
    h = D.system_basis(agent, "H", recorded=recorded)
    assert h.rank() == h.n_channels * (1 + 2 + 3)
    r = D.system_basis(agent, "R", recorded=recorded)
    n_disc = r.n_channels - 2
    assert r.rank() == n_disc * (1 + 2 + 3) + 2 + 3
    s0 = r.first_complete_sample
    y = np.random.default_rng(0).standard_normal(agent.n_time - s0)
    Q = r.orthonormal_span()
    assert Q.shape == (agent.n_time - s0, r.rank())
    np.testing.assert_allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-10)
    yc = y - y.mean()
    res_span = yc - Q @ (Q.T @ yc)
    A = np.column_stack([np.ones(y.size), r.values[:, s0:].T])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    np.testing.assert_allclose(res_span, y - A @ coef, atol=1e-9)


# --------------------------------------------------------------------------
# tau_c and the lag sets, derived from generator constants
# --------------------------------------------------------------------------
def test_tau_c_of_families_a_and_c(agent):
    family_c = _family_c()
    a = D.coupling_timescale(agent)
    assert (a.tau_c_sec, a.constant, a.constant_value) == (0.1, "AgentConfig.tau", 0.1)
    c = D.coupling_timescale(family_c)
    assert c.constant == "AgentConfig.sl_rate" and c.constant_value == 10.0
    assert c.tau_c_sec == pytest.approx(0.1)
    for system in (agent, family_c):
        plan = D.lag_plan(system)
        assert plan.nas_lags == (1, 2) and plan.iim_lags == (1, 2)  # the v1 lag set
        assert plan.l_max == 2 and plan.resolved and plan.dt == 0.05
        assert plan.basis_lags("NAS") == (0, 1, 2)
        assert plan.basis_taus == (0.1, 0.3, 1.0)


def test_tau_c_follows_the_generator_constants():
    meta = {"dynamics": "rate", "config": g.AgentConfig(tau=0.3).to_dict(), "dt": 0.05}
    plan = D.lag_plan(meta)
    assert plan.tau_c_sec == 0.3 and plan.l_max == 6 and plan.nas_lags == (1, 2, 3, 6)
    meta = {"dynamics": "stuart_landau", "config": g.AgentConfig(sl_rate=5.0).to_dict(),
            "dt": 0.05}
    assert D.lag_plan(meta).nas_lags == (1, 2, 3, 4)
    for meta in ({"dynamics": "rate"}, {"dynamics": "rate", "config": {"tau": 0}},
                 {"family": "human_eeg", "dt": 0.002}):
        with pytest.raises(D.DeclaredInputsError):
            D.coupling_timescale(meta)
    nc = D.coupling_timescale({"family": "null_calibration", "dt": 0.05})
    assert nc.tau_c_sec == g.AgentConfig().tau


def test_tau_c_and_lags_of_the_hopf_model_and_its_views():
    from impact_pipeline.bench import forward
    from impact_pipeline.bench import whole_brain as wb

    w = wb.simulate_whole_brain(wb.WholeBrainConfig(duration_sec=4.0), seed=SEED)
    tc = D.coupling_timescale(w)
    assert tc.constant == "WholeBrainConfig.rate" and tc.tau_c_sec == pytest.approx(0.1)
    plan = D.lag_plan(w)
    assert plan.dt == pytest.approx(0.004) and plan.l_max == 25
    assert plan.nas_lags == (1, 3, 9, 25) and plan.iim_lags == tuple(range(1, 26))
    assert plan.resolved
    eeg = forward.eeg_forward(w, seed=SEED)
    assert D.lag_plan(eeg).nas_lags == (1, 3, 9, 25)
    bold = forward.bold_forward(w, tr=2.0, seed=SEED)
    bp = D.lag_plan(bold)
    assert bp.tau_c_sec == pytest.approx(0.1) and bp.dt == 2.0 and not bp.resolved
    meta500 = {"dynamics": "hopf_whole_brain", "dt": 0.002,
               "whole_brain_config": wb.WholeBrainConfig(fs_out=500.0).to_dict()}
    assert D.lag_plan(meta500).nas_lags == (1, 4, 14, 50)


def test_lag_rules_and_the_resolvability_condition():
    assert D.l_max(0.1, 0.004) == 25 and D.l_max(0.3, 0.05) == 6
    assert D.l_max(0.1, 0.05) == 2 and D.l_max(0.1, 2.0) == 1
    assert D.nas_lag_set(0.1, 0.05) == (1, 2)
    assert D.nas_lag_set(0.1, 1 / 250) == (1, 3, 9, 25)
    assert D.nas_lag_set(0.1, 1 / 500) == (1, 4, 14, 50)
    assert D.nas_lag_set(0.1, 0.1) == (1,)
    assert D.iim_lag_set(0.1, 1 / 250) == tuple(range(1, 26))
    assert D.basis_lags(0.1, 0.05, "IIM") == (0, 1, 2)
    assert D.basis_taus(0.1) == (0.1, 0.3, 1.0)
    assert D.sampling_resolved(0.1, 0.05)  # dt = tau_c / 2 is resolved
    assert not D.sampling_resolved(0.1, 0.051)
    assert not D.sampling_resolved(0.1, 2.0)  # BOLD at TR 2 s
    lo, hi = D.TAU_C_SENSITIVITY_SEC
    assert not D.sampling_resolved(lo, 0.05) and D.nas_lag_set(hi, 0.05) == (1, 2, 3, 4)
    for bad in ((0.0, 0.05), (0.1, 0.0), (float("nan"), 0.05), (0.1, -1.0)):
        with pytest.raises(D.DeclaredInputsError):
            D.l_max(*bad)
    with pytest.raises(D.DeclaredInputsError):
        D.lag_set(0.1, 0.05, "SRPI")
    assert D.lag_plan(tau_c=0.1, dt=0.05).to_dict()["nas_lags"] == [1, 2]
