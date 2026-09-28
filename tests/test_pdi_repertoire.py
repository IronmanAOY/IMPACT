"""Known-answer, null and property tests for compute_PDI(mode='repertoire').

PDI as the repertoire of distinguishable states: labelled (decoded mutual
information about a declared state/context) and unlabelled (number of
recurring, separated multi-node states) variants. The simulators below are
small numpy models written for these tests only.
"""

import json

import numpy as np
import pandas as pd
import pytest

from impact_pipeline import mpc_metrics as mm

REP = dict(mode="repertoire", return_details=True)
FAST = dict(REP, null_surrogates=9)


def _visits(rng, n_states, n_time, dwell, min_dwell=5):
    """Piecewise-constant state sequence: random states, exponential dwell."""
    labels = np.zeros(n_time, dtype=int)
    t = 0
    while t < n_time:
        d = int(rng.exponential(dwell)) + min_dwell
        labels[t : t + d] = rng.integers(n_states)
        t += d
    return labels


def _planted_states(seed, k, n_nodes=16, n_time=3000, dwell=30.0, noise=0.5):
    """K attractor patterns (random +-1 over nodes) visited stochastically."""
    rng = np.random.default_rng(seed)
    pats = rng.choice([-1.0, 1.0], size=(k, n_nodes))
    labels = _visits(rng, k, n_time, dwell)
    drive = pats[labels].T
    x = np.zeros_like(drive)
    for t in range(1, n_time):
        x[:, t] = 0.7 * x[:, t - 1] + 0.3 * drive[:, t]
    return x + noise * rng.standard_normal((n_nodes, n_time)), labels


def _hypersynchronous(seed, n_nodes=16, n_time=3000, dt=0.01, freq=3.0, noise=0.05):
    """One common oscillation in every node (positive gains) + small noise."""
    rng = np.random.default_rng(seed)
    tt = np.arange(n_time) * dt
    common = np.sin(2 * np.pi * freq * tt + rng.uniform(0, 2 * np.pi))
    gains = rng.uniform(0.8, 1.2, size=(n_nodes, 1))
    return gains * common[None, :] + noise * rng.standard_normal((n_nodes, n_time))


def _hypersynchronous_resonance(seed, n_nodes=16, n_time=3000, period=23.7):
    """Noise-driven narrow-band AR(2) source shared by every node."""
    rng = np.random.default_rng(seed)
    r = 0.97
    a1, a2 = 2 * r * np.cos(2 * np.pi / period), -r * r
    s = np.zeros(n_time)
    e = rng.standard_normal(n_time)
    for t in range(2, n_time):
        s[t] = a1 * s[t - 1] + a2 * s[t - 2] + e[t]
    s /= s.std()
    gains = 1.0 + 0.2 * rng.standard_normal((n_nodes, 1))
    return gains * s[None, :] + 0.3 * rng.standard_normal((n_nodes, n_time))


def _independent_noise(seed, n_nodes=16, n_time=3000, ar=0.5):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n_nodes, n_time))
    x = np.zeros_like(e)
    for t in range(1, n_time):
        x[:, t] = ar * x[:, t - 1] + e[:, t]
    return x


def _travelling_wave(seed, n_nodes=16, n_time=3000, period=37.3, noise=0.3):
    """Stereotyped repeating pattern: one wave sweeping the nodes (phase jitter)."""
    rng = np.random.default_rng(seed)
    ph = np.cumsum(2 * np.pi / period + 0.1 * rng.standard_normal(n_time))
    lag = 2 * np.pi * np.arange(n_nodes) / n_nodes
    x = np.sin(ph[None, :] - lag[:, None])
    return x + noise * rng.standard_normal((n_nodes, n_time))


def _repeating_motif(seed, n_nodes=16, n_time=3000, motif_len=60, noise=0.3):
    """
    One smooth two-dimensional spatio-temporal motif, restarted every
    50-60 samples (each restart cuts the previous repeat short), so the
    trajectory never rests: a single stereotyped pattern with jittered phase.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(motif_len)
    lat = np.stack(
        [np.sin(2 * np.pi * t / motif_len), np.sin(4 * np.pi * t / motif_len + 1.0)]
    )
    motif = (rng.standard_normal((n_nodes, 2)) / np.sqrt(2)) @ lat
    x = np.zeros((n_nodes, n_time))
    pos = 0
    while pos < n_time:
        seg = motif[:, : min(motif_len, n_time - pos)]
        x[:, pos : pos + seg.shape[1]] = seg
        pos += motif_len - int(rng.integers(0, 11))
    return x + noise * rng.standard_normal((n_nodes, n_time))


def _independent_bistable(seed, n_nodes=3, n_time=3000, dwell=30.0, noise=0.3):
    """Independent two-state (telegraph) nodes: 2**n combinations, no coordination."""
    rng = np.random.default_rng(seed)
    x = np.stack(
        [
            np.where(_visits(rng, 2, n_time, dwell) == 1, 1.0, -1.0)
            for _ in range(n_nodes)
        ]
    )
    return x + noise * rng.standard_normal((n_nodes, n_time))


def _label_events(labels, tr, prefix="ctx"):
    change = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [labels.size]])
    return pd.DataFrame(
        {
            "onset": starts * tr,
            "duration": (ends - starts) * tr,
            "trial_type": [f"{prefix}{labels[s]}" for s in starts],
        }
    )


# --------------------------------------------------------------------------
# labelled variant
# --------------------------------------------------------------------------


def test_labelled_planted_states_give_log2_k_bits_increasing_with_k():
    res = []
    for k in (2, 4, 8):
        x, labels = _planted_states(0, k)
        d = mm.compute_PDI(x, state_labels=labels, null_seed=0, **REP)
        assert d["defined"] and d["variant"] == "labelled" and d["n_states"] == k
        # Decoding is near perfect, so I equals the label entropy (<= log2 K).
        assert d["accuracy"] > 0.97
        assert d["I_bits"] == pytest.approx(d["label_entropy_bits"], abs=0.05)
        assert abs(d["I_bits"] - np.log2(k)) < 0.2
        assert d["effective_states"] == pytest.approx(k, rel=0.15)
        assert d["value"] == pytest.approx(d["I_bits"] - d["PDI_null_mean"])
        assert d["value"] > np.log2(k) - 0.5 and d["PDI_z"] > 5
        res.append(d["value"])
    print("labelled repertoire, planted K=2,4,8: excess bits", np.round(res, 3))
    assert res[0] < res[1] < res[2]


@pytest.mark.parametrize(
    "make",
    [
        _hypersynchronous,
        _hypersynchronous_resonance,
        _independent_noise,
        _travelling_wave,
    ],
    ids=["hypersync", "hypersync_resonance", "independent_noise", "travelling_wave"],
)
def test_labelled_null_systems_carry_no_state_information(make):
    # Context labels that the dynamics ignore: nothing is decodable.
    x = make(0)
    labels = _visits(np.random.default_rng(7), 4, x.shape[1], 30.0)
    d = mm.compute_PDI(x, state_labels=labels, null_seed=1, **REP)
    assert d["defined"]
    assert d["I_bits"] < 0.1 and abs(d["value"]) < 0.1
    assert not d["PDI_z"] > 3.0


def test_block_permutation_null_absorbs_autocorrelation_bias():
    # Strongly structured 8-state data with an unrelated, autocorrelated label
    # sequence: the plug-in MI is biased upwards (few independent label runs),
    # the block-permuted null has the same bias, so the excess is about 0.
    exc = []
    for seed in range(3):
        x, _ = _planted_states(seed, 8)
        other = _visits(np.random.default_rng(100 + seed), 4, x.shape[1], 30.0)
        d = mm.compute_PDI(x, state_labels=other, null_seed=seed, **REP)
        assert d["PDI_null_mean"] > 0.02
        assert abs(d["PDI_z"]) < 3.0
        exc.append(d["value"])
    print("unrelated labels on 8-state data: excess bits", np.round(exc, 3))
    assert abs(np.mean(exc)) < 0.1


def test_random_walk_with_block_labels_is_not_significant():
    # Non-stationary data and a block design: time-blocked, purged CV and the
    # block-permutation null keep slow drift from passing as a state code.
    zs = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        x = np.cumsum(rng.standard_normal((10, 3000)), axis=1)
        labels = (np.arange(3000) // 250) % 3
        d = mm.compute_PDI(x, state_labels=labels, null_seed=seed, **REP)
        zs.append(d["PDI_z"])
    print("random walk + block labels: z", np.round(zs, 2))
    assert max(zs) < 2.5


def test_labelled_information_degrades_gradually_with_noise_and_window():
    vals = []
    for noise in (0.5, 2.0, 5.0):
        x, labels = _planted_states(0, 4, noise=noise)
        d = mm.compute_PDI(x, state_labels=labels, null_seed=0, **FAST)
        vals.append(d["I_bits"])
    assert vals[0] > vals[1] > vals[2] > 0.2
    x, labels = _planted_states(0, 4, noise=1.0)
    single = mm.compute_PDI(x, state_labels=labels, repertoire_window=1, **FAST)
    short = mm.compute_PDI(x, state_labels=labels, repertoire_window=5, **FAST)
    assert single["I_bits"] < short["I_bits"]
    assert single["n_windows"] > short["n_windows"]


def test_events_route_equals_label_vector():
    x, labels = _planted_states(1, 4)
    tr = 0.5
    ev = _label_events(labels, tr)
    a = mm.compute_PDI(x, events=ev, tr=tr, null_seed=3, **FAST)
    b = mm.compute_PDI(
        x, state_labels=[f"ctx{v}" for v in labels], null_seed=3, **FAST
    )
    assert a["label_source"] == "events" and b["label_source"] == "state_labels"
    assert a["I_bits"] == b["I_bits"] and a["value"] == b["value"]
    assert a["n_events_used"] == len(ev) and a["n_conflict_samples"] == 0
    assert a["window_sec"] == pytest.approx(5 * tr)
    # Column mapping and list-of-rows forms are equivalent.
    cols = {c: list(ev[c]) for c in ev.columns}
    rows = ev.to_dict("records")
    for form in (cols, rows):
        c = mm.compute_PDI(x, events=form, tr=tr, null_seed=3, **FAST)
        assert c["I_bits"] == a["I_bits"]
    # Events that lead the states by 10 samples (as event onsets lead a
    # haemodynamic response): declaring the 5 s delay realigns them.
    lead = np.concatenate([labels[10:], np.full(10, labels[-1])])
    ev_lead = _label_events(lead, tr)
    d0 = mm.compute_PDI(x, events=ev_lead, tr=tr, null_seed=3, **FAST)
    d1 = mm.compute_PDI(
        x, events=ev_lead, tr=tr, repertoire_label_delay=10 * tr, null_seed=3, **FAST
    )
    assert d0["label_delay_sec"] == 0.0 and d1["label_delay_sec"] == pytest.approx(5.0)
    assert d1["I_bits"] > d0["I_bits"]
    assert d1["I_bits"] == pytest.approx(a["I_bits"], abs=0.05)


def test_event_conflicts_and_skipped_rows_are_counted_not_guessed():
    x, _ = _planted_states(0, 2, n_time=400)
    ev = pd.DataFrame(
        {
            "onset": [0.0, 50.0, 100.0, 150.0, 5.0, 300.0],
            "duration": [50.0, 50.0, 50.0, 50.0, 10.0, 0.0],
            "trial_type": ["a", "b", "a", "b", "b", "a"],
        }
    )
    d = mm.compute_PDI(x, events=ev, tr=1.0, **FAST)
    assert d["n_events_used"] == 5 and d["n_events_skipped"] == 1
    assert d["n_conflict_samples"] == 10  # 5..15 s claimed by 'a' and 'b'
    assert d["n_labelled_samples"] == 200 - 10


def test_decimal_event_onsets_land_on_the_sample_grid():
    # BIDS onsets are decimal seconds: 10 * 0.72 < 7.2 in floating point, so
    # an exact comparison moved block starts by one sample and flagged the
    # boundary samples as conflicts.
    expected = np.repeat(np.arange(20) % 2, 10)
    for tr in (0.72, 0.7, 0.1, 2.2):
        ev = pd.DataFrame(
            {
                "onset": [round(b * 10 * tr, 6) for b in range(20)],
                "duration": [round(10 * tr, 6)] * 20,
                "trial_type": ["ab"[b % 2] for b in range(20)],
            }
        )
        codes, names, info = mm._pdi_labels_from_events(ev, 200, tr, "trial_type", 0.0)
        assert names == ["a", "b"] and info["n_conflict_samples"] == 0, tr
        assert np.array_equal(codes, expected), tr
    # a delay is placed on the same grid: 7.2 s from 0.72 s covers samples 1..10
    one = [{"onset": 0.0, "duration": 7.2, "trial_type": "a"}]
    codes, _, _ = mm._pdi_labels_from_events(one, 200, 0.72, "trial_type", 0.72)
    assert np.flatnonzero(codes >= 0).tolist() == list(range(1, 11))


def test_sparse_states_are_dropped_and_reported():
    x, labels = _planted_states(0, 3, n_time=3000)
    lab = labels.astype(object)
    lab[:] = np.where(labels == 2, "B", "A")
    lab[100:108] = "rare"  # one short visit: 1 window
    lab[500:560] = "once"  # 12 windows but a single visit
    d = mm.compute_PDI(x, state_labels=lab, null_seed=0, **FAST)
    assert d["defined"] and sorted(d["states"]) == ["A", "B"]
    assert d["dropped_states"] == {
        "rare": "fewer_than_3_windows",
        "once": "fewer_than_2_visits",
    }
    assert d["n_windows_dropped_mixed"] > 0
    only = np.where(labels >= 0, "A", None).astype(object)
    only[500:560] = "once"
    u = mm.compute_PDI(x, state_labels=only, **FAST)
    assert not u["defined"] and u["undefined_reason"] == "fewer_than_two_states"
    none = mm.compute_PDI(x, state_labels=[None] * x.shape[1], **FAST)
    assert none["undefined_reason"] == "no_labelled_windows"


def test_mutual_information_and_miller_madow_known_answers():
    n = 400.0
    for k in (2, 4, 8):
        diag = np.diag(np.full(k, n / k))
        i_mm, i_plug, bias = mm._pdi_mutual_information_bits(diag)
        assert i_plug == pytest.approx(np.log2(k))
        assert bias == pytest.approx((1 - k) / (2 * n * np.log(2)))
        assert i_mm == pytest.approx(i_plug - bias)
        flat = np.full((k, k), n / (k * k))
        i_mm, i_plug, bias = mm._pdi_mutual_information_bits(flat)
        assert i_plug == pytest.approx(0.0, abs=1e-12)
        assert bias == pytest.approx((k - 1) ** 2 / (2 * n * np.log(2)))
        assert i_mm < 0.0
    assert np.isnan(mm._pdi_mutual_information_bits(np.zeros((2, 2)))[0])


def test_segment_permutation_keeps_runs_and_counts():
    rng = np.random.default_rng(0)
    y = _visits(rng, 4, 500, 12.0, min_dwell=1)
    _, lengths = mm._pdi_label_segments(y)
    runs = np.split(y, np.cumsum(lengths)[:-1])
    perm_rng, ref_rng = np.random.RandomState(0), np.random.RandomState(0)
    agree = []
    for _ in range(20):
        yp = mm._pdi_segment_permutation(y, perm_rng)
        # the original runs, each intact (label and length), in a new order
        order = ref_rng.permutation(len(runs))
        assert np.array_equal(yp, np.concatenate([runs[i] for i in order]))
        assert np.array_equal(np.bincount(yp, minlength=4), np.bincount(y, minlength=4))
        agree.append(float(np.mean(yp == y)))
    # the alignment of labels with time is destroyed (4 labels: ~1/4 agree)
    assert max(agree) < 0.6 and np.mean(agree) < 0.4


def test_blocks_separated_by_unlabelled_rest_are_separate_visits():
    # Randomised block order with an unlabelled rest after every block (the
    # usual events table lists only the task blocks). Merging "a, rest, a"
    # into one run halved the visits (2 instead of 4 per condition) and left
    # the block-permutation null so few distinct orders that it reproduced
    # the design (excess 0.30 bits for a perfectly decodable 1-bit design).
    y = np.array([0, 0, 0, 0, 1, 1, 1])
    w = np.array([0, 1, 3, 9, 10, 11, 20])
    lab, length = mm._pdi_label_segments(y, w, gap=1)
    # one unused window (1 -> 3) is within the purge gap: the same visit;
    # five (3 -> 9) or eight (11 -> 20) start a new one
    assert lab.tolist() == [0, 0, 1, 1] and length.tolist() == [3, 1, 2, 1]
    assert mm._pdi_label_segments(y, w, gap=0)[1].tolist() == [2, 1, 1, 2, 1]
    assert mm._pdi_label_segments(y)[1].tolist() == [4, 3]
    # these visits are the permutation units
    runs = np.split(y, np.cumsum(length)[:-1])
    ref = np.random.RandomState(1).permutation(len(runs))
    perm = mm._pdi_segment_permutation(y, np.random.RandomState(1), length)
    assert np.array_equal(perm, np.concatenate([runs[i] for i in ref]))
    rng = np.random.default_rng(0)
    n_nodes, blk = 16, 100
    order = ["a", "a", "b", "b", "a", "a", "b", "b"]
    pats = {c: rng.choice([-1.0, 1.0], n_nodes) for c in "ab"}
    pats["rest"] = np.zeros(n_nodes)
    seq = [c for cond in order for c in [cond] * blk + ["rest"] * blk]
    x = np.stack([pats[c] for c in seq], axis=1)
    x = x + 0.8 * rng.standard_normal(x.shape)
    ev = pd.DataFrame(
        {
            "onset": [2.0 * blk * i for i in range(len(order))],
            "duration": [float(blk)] * len(order),
            "trial_type": order,
        }
    )
    d = mm.compute_PDI(x, events=ev, tr=1.0, null_seed=0, **REP)
    assert d["n_visits_per_state"] == {"a": 4, "b": 4}
    assert d["I_bits"] > 0.98 and d["value"] > 0.6


def test_pandas_missing_labels_are_unlabelled_not_a_state():
    # Nullable pandas columns (read_csv(..., dtype="string"), convert_dtypes())
    # hold pd.NA, which must not become a state called "<NA>".
    x, labels = _planted_states(0, 2, n_time=1500)
    lab = pd.Series(np.where(labels == 1, "b", "a"), dtype="string")
    lab[lab.index % 50 < 5] = pd.NA
    d = mm.compute_PDI(x, state_labels=lab, null_seed=0, **FAST)
    assert d["defined"] and sorted(d["states"]) == ["a", "b"]
    assert d["n_labelled_samples"] == int(lab.notna().sum())
    ints = pd.Series(np.where(labels == 1, 2, 1), dtype="Int64")
    ints[:40] = pd.NA
    d = mm.compute_PDI(x, state_labels=ints, null_seed=0, **FAST)
    assert sorted(d["states"]) == ["1", "2"] and d["n_labelled_samples"] == 1460
    ev = _label_events(labels, 1.0)
    ev["trial_type"] = ev["trial_type"].astype("string")
    ev.loc[0, "trial_type"] = pd.NA
    d = mm.compute_PDI(x, events=ev, tr=1.0, null_seed=0, **FAST)
    assert d["n_events_skipped"] == 1 and d["n_events_used"] == len(ev) - 1
    assert sorted(d["states"]) == ["ctx0", "ctx1"]


# --------------------------------------------------------------------------
# unlabelled variant
# --------------------------------------------------------------------------


def test_separation_criterion_known_answers():
    rng = np.random.default_rng(0)
    n = 20000
    blocks = np.zeros(2 * n, dtype=int)
    persistent = np.repeat(np.arange(2 * n // 50) % 2, 50)

    def check(x, c, block=blocks, min_dwell=1.0):
        return mm._pdi_partition_check(np.asarray(c, float), x, block, 1e9, min_dwell)

    # Two equal-variance Gaussian states d' apart: the valley ratio falls with
    # d' and crosses the default 0.4 at d' ~ 3.85 (pairwise Bayes error ~3%).
    vals = {}
    for d in (3.0, 3.5, 4.0, 4.5):
        x = (persistent * d + rng.standard_normal(2 * n))[:, None]
        vals[d] = check(x, [[0.0], [d]])[1]
    assert vals[3.0] > vals[3.5] > vals[4.0] > vals[4.5]
    assert vals[3.5] > 0.4 > vals[4.0]
    assert vals[3.0] == pytest.approx(0.70, abs=0.05)
    assert vals[4.5] == pytest.approx(0.25, abs=0.05)
    # A cut through a continuum has no valley: uniform ~1, Gaussian > 1.
    u = rng.uniform(-1, 1, (2 * n, 1))
    assert check(u, [[-0.5], [0.5]])[1] == pytest.approx(1.0, abs=0.1)
    g = rng.standard_normal((2 * n, 1))
    assert check(g, [[-0.8], [0.8]])[1] > 1.2
    # Arcs of a ring (one stereotyped cycle) have no valley between them
    # either. The projection of curved arcs on the chord axis mimics a
    # shallow dip, deepest for thirds (~0.5), which the default 0.4 rejects.
    th = rng.uniform(0, 2 * np.pi, 2 * n)
    ring = np.c_[np.cos(th), np.sin(th)] + 0.05 * rng.standard_normal((2 * n, 2))
    arc_ratio = {}
    for k in range(2, 9):
        ang = 2 * np.pi * (np.arange(k) + 0.5) / k
        r = np.sin(np.pi / k) / (np.pi / k)  # centroid radius of an arc
        arc_ratio[k] = check(ring, r * np.c_[np.cos(ang), np.sin(ang)])[1]
    assert min(arc_ratio.values()) == pytest.approx(arc_ratio[3])
    assert 0.45 < arc_ratio[3] < 0.6
    assert min(v for k, v in arc_ratio.items() if k != 3) > 0.65
    # Persistence: two separated states that alternate every window (a fast
    # cycle) fail the dwell requirement; the same states in runs pass.
    alternating = (np.arange(2 * n) % 2 * 6.0 + rng.standard_normal(2 * n))[:, None]
    ok, _, dwell = check(alternating, [[0.0], [6.0]], min_dwell=2.0)
    assert not ok and dwell < 1.1
    runs = (persistent * 6.0 + rng.standard_normal(2 * n))[:, None]
    ok, _, dwell = check(runs, [[0.0], [6.0]], min_dwell=2.0)
    assert ok and dwell > 20
    # Recurrence: a state with fewer than 3 held-out windows fails.
    few = np.r_[np.zeros(100), np.full(2, 6.0)][:, None]
    assert not check(few, [[0.0], [6.0]], block=np.zeros(102, dtype=int))[0]


def test_unlabelled_planted_states_count_k_and_excess_is_log2_k():
    res = []
    for k in (2, 4, 8):
        d = mm.compute_PDI(_planted_states(0, k)[0], null_seed=0, **FAST)
        assert d["defined"] and d["variant"] == "unlabelled"
        assert d["n_states"] == k and d["state_bits"] == pytest.approx(np.log2(k))
        assert not d["n_states_at_cap"]
        assert len(d["state_occupancy"]) == k
        assert d["state_entropy_bits"] <= np.log2(k) + 1e-9
        res.append(d["value"])
    print("unlabelled repertoire, planted K=2,4,8: excess bits", np.round(res, 3))
    assert res[0] < res[1] < res[2]
    assert abs(res[1] - 2.0) < 0.25 and abs(res[2] - 3.0) < 0.25
    # K = 2: the circular-shift surrogates keep every node's bistability, and
    # a split along a single bistable node is itself a recurring, separated
    # pair of states, so the null is > 0 bits and the excess < 1 bit.
    assert 0.0 < res[0] <= 1.0


@pytest.mark.parametrize(
    "make",
    [_hypersynchronous, _hypersynchronous_resonance, _independent_noise],
    ids=["hypersync", "hypersync_resonance", "independent_noise"],
)
def test_unlabelled_hypersynchrony_and_noise_are_about_zero(make):
    for seed in range(2):
        d = mm.compute_PDI(make(seed), null_seed=seed, **FAST)
        assert d["n_states"] == 1, d["per_k"][:3]
        assert abs(d["value"]) < 0.35


def test_bench_hypersynchronous_generator_is_one_state():
    # The MPC-Bench adversarial hypersynchrony generator: a 3 Hz sinusoid
    # sampled at 20 Hz (dt = 0.05). With 5-sample windows the window means
    # form a lattice of 4 patterns (3 cycles per 4 windows); split in two
    # halves they dwell exactly 2 windows, so a dwell threshold of 2 was
    # decided by block-edge effects (seed 1: 2 states, +0.89 bits; a
    # 3005-sample run: 2 states, +1 bit). The default 2.5 rejects it.
    from impact_pipeline.bench.generators import make_system

    for seed in range(3):
        x = make_system("hypersynchronous", seed=seed).ts
        d = mm.compute_PDI(x, null_seed=seed, **FAST)
        assert d["min_dwell_windows"] == 2.5
        assert d["n_states"] == 1 and d["value"] == 0.0, (seed, d["per_k"][0])
        assert not d["per_k"][0]["min_dwell_windows"] >= 2.5
    for n_time in (1500, 3000, 3005):
        for freq in (1.0, 3.0):
            x = _hypersynchronous(0, n_time=n_time, dt=0.05, freq=freq)
            d = mm.compute_PDI(x, null_seed=0, **FAST)
            assert d["n_states"] == 1 and abs(d["value"]) < 0.35, (n_time, freq)


def test_phase_diffusion_removes_the_lattice_of_a_slow_oscillation():
    # A noise-free oscillation of exactly 8 windows per cycle is a lattice
    # of 8 window patterns whose halves dwell 4 windows (documented
    # limitation); with phase diffusion it is a continuum again: one state
    # and no excess over the circular-shift surrogates.
    for seed in range(3):
        rng = np.random.default_rng(seed)
        n_time = 3000
        step = 2 * np.pi / 40.0 + 0.05 * rng.standard_normal(n_time)
        ph = np.cumsum(step) + rng.uniform(0, 2 * np.pi)
        gains = rng.uniform(0.8, 1.2, size=(16, 1))
        x = gains * np.sin(ph)[None, :] + 0.05 * rng.standard_normal((16, n_time))
        d = mm.compute_PDI(x, null_seed=seed, **FAST)
        assert d["n_states"] == 1 and abs(d["value"]) < 0.35


@pytest.mark.parametrize(
    "make", [_travelling_wave, _repeating_motif], ids=["wave", "motif"]
)
def test_unlabelled_stereotyped_repeating_pattern_is_low(make):
    four = mm.compute_PDI(_planted_states(0, 4)[0], null_seed=0, **FAST)["value"]
    for seed in range(2):
        d = mm.compute_PDI(make(seed), null_seed=seed, **FAST)
        assert d["n_states"] <= 2
        assert d["value"] < 0.5 and d["value"] < four - 1.0


def test_circular_shift_null_discounts_independent_multistable_nodes():
    # Three independent bistable nodes occupy 2**3 separated, recurring
    # joint states, but none is a coordinated multi-node state: the default
    # circular-shift null (independent shifts per node) reproduces them. A
    # Fourier null would not (it removes every multimodality).
    x = _independent_bistable(0)
    d = mm.compute_PDI(x, repertoire_null="both", null_seed=0, **FAST)
    assert d["n_states"] == 8
    fam = d["null_families"]
    assert fam["circular_shift"]["excess_bits"] < 0.75
    assert fam["fourier"]["excess_bits"] > 2.5
    assert d["binding_null"] == "circular_shift"
    assert d["value"] == pytest.approx(fam["circular_shift"]["excess_bits"])
    assert d["PDI_null_method"] == "circular_shift"


def test_prediction_strength_counts_cuts_of_a_continuum():
    # Why the default criterion requires separation: prediction strength
    # rewards any reproducible partition, e.g. slices of the one-dimensional
    # trajectory of a hypersynchronous oscillator.
    x = _hypersynchronous(0)
    ps = mm.compute_PDI(x, repertoire_criterion="prediction_strength", **FAST)
    sep = mm.compute_PDI(x, **FAST)
    assert ps["criterion"] == "prediction_strength" and ps["n_states"] >= 2
    assert all("prediction_strength" in p for p in ps["per_k"])
    assert sep["n_states"] == 1
    # both criteria agree on genuinely discrete states
    xs = _planted_states(0, 4)[0]
    assert mm.compute_PDI(
        xs, repertoire_criterion="prediction_strength", **FAST
    )["n_states"] >= 4


def test_power_features_capture_oscillatory_states():
    # Three states that differ in which nodes oscillate (random phases): the
    # window mean is ~0 in every state, the log power separates them.
    rng = np.random.default_rng(0)
    n, t_len = 12, 4000
    labels = _visits(rng, 3, t_len, 60.0, min_dwell=20)
    amps = rng.choice([0.2, 1.5], size=(3, n))
    phase = rng.uniform(0, 2 * np.pi, n)
    tt = np.arange(t_len)
    x = amps[labels].T * np.sin(2 * np.pi * tt / 4.0 + phase[:, None])
    x = x + 0.3 * rng.standard_normal((n, t_len))
    kw = dict(repertoire_window=10, null_seed=0, **FAST)
    mean = mm.compute_PDI(x, state_labels=labels, **kw)
    power = mm.compute_PDI(x, state_labels=labels, repertoire_features="power", **kw)
    assert abs(mean["value"]) < 0.1
    assert power["I_bits"] == pytest.approx(np.log2(3), abs=0.15)
    unl = mm.compute_PDI(x, repertoire_features="power", **kw)
    assert unl["n_states"] == 3 and unl["features"] == "power"


def test_unlabelled_count_is_conservative_under_noise_and_long_windows():
    counts = []
    for noise in (0.5, 1.0, 3.0):
        x, _ = _planted_states(0, 4, noise=noise)
        counts.append(mm.compute_PDI(x, null_seed=0, **FAST)["n_states"])
    assert counts[0] == 4 and counts == sorted(counts, reverse=True)
    assert counts[-1] < 4  # below the separation resolution states merge
    # Windows longer than about half the state duration: states do not
    # persist for two windows and are not counted (never over-counted).
    x, _ = _planted_states(0, 4, noise=1.0)
    d = mm.compute_PDI(x, repertoire_window=20, null_seed=0, **FAST)
    assert d["n_states"] < 4


def test_short_records_cap_the_number_of_testable_states():
    x, _ = _planted_states(0, 8, n_time=300)
    d = mm.compute_PDI(x, null_seed=0, **FAST)
    n_half = d["n_windows"] // 2
    assert d["max_states_tested"] <= n_half // mm.PDI_REPERTOIRE_MIN_STATE_WINDOWS
    assert d["n_states"] <= d["max_states_tested"]
    tiny = mm.compute_PDI(x[:, :40], **FAST)
    assert not tiny["defined"] and tiny["undefined_reason"] == "insufficient_windows"
    at_cap = mm.compute_PDI(
        _planted_states(0, 8)[0], repertoire_max_states=4, null_seed=0, **FAST
    )
    assert at_cap["n_states_at_cap"] is (at_cap["n_states"] == 4)


# --------------------------------------------------------------------------
# contract, determinism and validation
# --------------------------------------------------------------------------


def test_value_fields_determinism_and_float_return():
    x, labels = _planted_states(2, 4, n_time=1500)
    for kw in ({"state_labels": labels}, {}):
        a = mm.compute_PDI(x, null_seed=5, **kw, **FAST)
        b = mm.compute_PDI(x, null_seed=5, **kw, **FAST)
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
        assert a["mode"] == "repertoire"
        assert a["value"] == pytest.approx(a["raw"] - a["PDI_null_mean"])
        assert a["PDI_calibrated"] == pytest.approx(a["PDI_excess"])
        assert a["PDI_null_n"] == 9 and a["n_surrogates_requested"] == 9
        assert a["PDI_null_seed"] == 5
        v = mm.compute_PDI(x, mode="repertoire", null_seed=5, null_surrogates=9, **kw)
        assert isinstance(v, float) and v == a["value"]
    d = mm.compute_PDI(x, state_labels=labels, **REP)
    assert d["PDI_null_n"] == mm.PDI_REPERTOIRE_DEFAULT_SURROGATES
    assert d["PDI_null_method"] == "block_label_permutation"


def test_bearer_nodes_baseline_and_legacy_default_unchanged():
    x, labels = _planted_states(0, 4, n_time=1500)
    noise = np.random.default_rng(9).standard_normal((4, x.shape[1]))
    full = np.vstack([x, noise])
    bearer = list(range(16))
    a = mm.compute_PDI(
        full, bearer_nodes=bearer, state_labels=labels, null_seed=1, **FAST
    )
    b = mm.compute_PDI(x, state_labels=labels, null_seed=1, **FAST)
    assert a["I_bits"] == b["I_bits"] and a["bearer_nodes"] == bearer
    c = mm.compute_PDI(x, baseline_ts=x[:, :500], null_seed=1, **FAST)
    assert c["baseline_ignored"] is True
    assert c["value"] == mm.compute_PDI(x, null_seed=1, **FAST)["value"]
    legacy = mm.compute_PDI(x, baseline_ts=noise[:, :800].repeat(4, axis=0))
    assert legacy == mm.compute_PDI(
        x, baseline_ts=noise[:, :800].repeat(4, axis=0), mode="legacy"
    )


def test_repertoire_undefined_reasons_and_validation():
    rng = np.random.default_rng(0)
    one = mm.compute_PDI(rng.standard_normal((1, 500)), **FAST)
    assert one["undefined_reason"] == "insufficient_regions" and np.isnan(one["value"])
    assert mm.compute_PDI(rng.standard_normal((4, 6)), **FAST)["undefined_reason"] == (
        "insufficient_timepoints"
    )
    bad = rng.standard_normal((4, 300))
    bad[0, 5] = np.nan
    assert mm.compute_PDI(bad, **FAST)["undefined_reason"] == "non_finite_timeseries"
    flat = mm.compute_PDI(np.zeros((4, 300)), **FAST)
    assert flat["undefined_reason"] == "no_variance" and flat["PDI_null_n"] == 0
    assert np.isnan(mm.compute_PDI(np.zeros((4, 300)), mode="repertoire"))
    x = rng.standard_normal((4, 300))
    lab = np.zeros(300, dtype=int)
    ev = _label_events(lab, 1.0)
    cases = [
        (dict(state_labels=lab, events=ev), "either state_labels or events"),
        (dict(events=ev), "tr .* is required"),
        (dict(events=ev, tr=0.0), "tr must be"),
        (dict(state_labels=lab[:10]), "one entry per sample"),
        (dict(repertoire_window=0), "repertoire_window"),
        (dict(repertoire_window=2.5), "repertoire_window"),
        (dict(repertoire_features="std"), "repertoire_features"),
        (dict(repertoire_folds=1), "repertoire_folds"),
        (dict(repertoire_gap=-1), "repertoire_gap"),
        (dict(repertoire_components=0), "repertoire_components"),
        (dict(repertoire_max_states=1), "repertoire_max_states"),
        (dict(repertoire_null="shuffle"), "repertoire_null"),
        (dict(repertoire_criterion="silhouette"), "repertoire_criterion"),
        (dict(repertoire_valley=0.0), "repertoire_valley"),
        (dict(repertoire_min_dwell=0.5), "repertoire_min_dwell"),
        (dict(null_surrogates=1), "null_surrogates >= 2"),
        (dict(events=[{"onset": 0.0, "duration": 5.0}], tr=1.0), "lacks required"),
        (dict(events=42, tr=1.0), None),
    ]
    for kw, match in cases:
        exc = TypeError if match is None else ValueError
        with pytest.raises(exc, match=match):
            mm.compute_PDI(x, mode="repertoire", **kw)
    with pytest.raises(ValueError, match="only used by mode='repertoire'"):
        mm.compute_PDI(x, state_labels=lab)
    with pytest.raises(ValueError, match="only used by mode='repertoire'"):
        mm.compute_PDI(x, mode="surrogate_excess", events=ev)
    assert "repertoire" in mm.PDI_MODES
