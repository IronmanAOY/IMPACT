# -*- coding: utf-8 -*-
"""RAM-PE v3 (ram-v3-2026.10): the signed cross-validated readout, its exact
shift null, the fold purge, invariance to option relabelling, the update and
window guards, and the declared non-applicability on the family-C carrier
recording. Synthetic fixtures are built here; the one bench run uses a
development seed."""
import numpy as np
import pandas as pd
import pytest

from impact_pipeline import mpc_metrics as mm
from impact_pipeline.v2 import ESTIMATOR_VERSIONS_V2
from impact_pipeline.v2 import ram_v3 as RAM
from impact_pipeline.v2 import reasons as R
from impact_pipeline.v2 import records as REC

DT = 0.05
TRIAL_SEC = 4.0
STIM_AT = 1.0
FEEDBACK_AFTER = 1.2


def bandit_run(n_trials=60, seed=0, gain=1.0, n_nodes=6, labels=("left", "right"),
               response_after=0.6, n_options=2):
    """A two-option bandit recorded on ``n_nodes`` nodes: the stimulus
    response pattern ``r_k`` (held over the 0.4 s window after the stimulus)
    moves by ``gain * b * y_k`` after outcome ``k`` (``y_k`` the choice-signed
    prediction error of the frozen v1 fit); ``gain = 0`` is the
    no-plasticity control. Returns ``(ts, events)``."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_options, n_trials)
    p_win = np.linspace(0.8, 0.3, n_options)[idx]
    rewards = np.where(rng.random(n_trials) < p_win, 1.0, -1.0)
    choices = [labels[i] for i in idx]
    stim = np.arange(n_trials) * TRIAL_SEC + STIM_AT
    response = stim + np.broadcast_to(np.asarray(response_after, float), stim.shape)
    feedback = stim + FEEDBACK_AFTER
    n_time = int(round(n_trials * TRIAL_SEC / DT))
    ts = rng.normal(size=(n_nodes, n_time))
    fit = mm._fit_rescorla_wagner(choices, rewards)
    if fit is not None and n_options == 2:
        sign = RAM.option_signs(choices, sorted(set(choices)))
        y = sign * fit["prediction_errors"]
        y = y / max(float(np.std(y)), 1e-12)
        b = rng.normal(size=n_nodes)
        b /= np.linalg.norm(b)
        r = np.zeros((n_trials, n_nodes))
        r[0] = rng.normal(size=n_nodes)
        for k in range(n_trials - 1):
            r[k + 1] = r[k] + gain * b * y[k] + 0.3 * rng.normal(size=n_nodes)
        w = int(round(0.4 / DT))
        for k, s in enumerate(np.rint(stim / DT).astype(int)):
            ts[:, s:s + w] += r[k][:, None]
    events = {
        "stimulus_onsets": stim,
        "response_onsets": response,
        "choice_onsets": feedback,
        "choices": choices,
        "rewards": rewards,
        "goal_onsets": stim - 0.6,
        "feedback_onsets": feedback,
    }
    return ts, events


def events_table(events, decoys=True, channel_column=True):
    rows = []
    for k, (s, r, f, c, rw) in enumerate(zip(
            events["stimulus_onsets"], events["response_onsets"],
            events["choice_onsets"], events["choices"], events["rewards"])):
        rows += [
            {"onset": s - 0.6, "duration": 0.3, "trial_type": "goal_cue", "trial": k},
            {"onset": s, "duration": 0.3, "trial_type": "stimulus", "trial": k},
            {"onset": r, "duration": 0.15, "trial_type": "response", "choice": c,
             "trial": k},
            {"onset": f, "duration": 0.3, "trial_type": "feedback", "choice": c,
             "reward": rw, "trial": k},
        ]
    for row in rows:
        row["impact_channel"] = "behavioural_feedback"
    if decoys:  # events of another channel that RAM-PE must not read
        for k in range(5):
            t = 2.6 + TRIAL_SEC * k
            rows += [{"onset": t, "duration": 0.1, "trial_type": "stimulus_probe",
                      "impact_channel": "agency"},
                     {"onset": t + 0.05, "duration": 0.1, "trial_type": "response",
                      "impact_channel": "agency"}]
    df = pd.DataFrame(rows).sort_values("onset", kind="mergesort")
    df = df.reset_index(drop=True)
    if not channel_column:
        df = df.drop(columns="impact_channel")
    return df


def pairs_of(ts, events):
    """``(D, y, keys)`` of the lag-1 updates, through the public steps."""
    ev, _ = RAM.ram_events(events, None)
    stim_on, stim_idx = mm._sanitize_onset_seconds(ev.stimulus_onsets, DT, ts.shape[1])
    fit = mm._fit_rescorla_wagner(list(ev.choices), ev.rewards)
    options = sorted({str(c) for c in ev.choices})
    targets, _ = RAM.stimulus_targets(
        stim_on, ev.choice_onsets,
        RAM.option_signs(ev.choices, options) * fit["prediction_errors"])
    w_s, l_s = RAM.window_samples(RAM.RAMParams(), DT)
    patterns = RAM.response_patterns(RAM.zscore_nodes(ts), stim_idx, w_s, l_s)
    return RAM.update_pairs(patterns, targets, lag=1)


def compute(ts, events, **params):
    return RAM.compute_ram_v3(ts, DT, events, params={"channel": None, **params})


# --------------------------------------------------------------------------
# declarations
# --------------------------------------------------------------------------
def test_declarations_and_bench_defaults():
    assert RAM.ESTIMATOR_VERSION == "ram-v3-2026.10"
    assert RAM.ESTIMATOR_VERSION in ESTIMATOR_VERSIONS_V2["RAM"]
    assert RAM.CONSTRUCT == "RAM-PE"
    assert RAM.NULL_FAMILY == "trial_circular_shift"
    assert RAM.FACETS_NOT_APPLICABLE == {
        "goal_alignment": "no_goal_contingency_referent",
        "feedback_magnitude": "no_graded_feedback_referent",
        "speed": "speed_not_in_construct",
    }
    p = RAM.RAMParams()
    assert (p.window_sec, p.lag_sec, p.n_folds, p.inner_folds) == (0.4, 0.0, 5, 4)
    assert p.lambdas == (0.1, 1.0, 10.0, 100.0, 1000.0)
    assert (p.h_min, p.min_updates, p.max_response_overlap) == (5, 30, 0.05)
    assert p.se_method == "shift_null_sd" and p.channel == "behavioural_feedback"
    assert RAM.SE_METHODS == ("shift_null_sd", "jackknife_trials_10")
    assert RAM.RAMParams.from_mapping(p.to_dict()) == p
    with pytest.raises(ValueError):
        RAM.RAMParams.from_mapping({"persistence_lag": 2})
    with pytest.raises(ValueError):
        RAM.RAMParams(se_method="bootstrap")
    with pytest.raises(ValueError):
        RAM.RAMParams(lambdas=(10.0, 1.0))


# --------------------------------------------------------------------------
# the exact shift null
# --------------------------------------------------------------------------
def test_the_shift_null_enumerates_every_shift():
    rng = np.random.default_rng(3)
    d = rng.normal(size=(41, 5))
    y = d @ rng.normal(size=5) + rng.normal(size=41)
    m, null, shifts = RAM.readout_with_null(d, y, h_min=5)
    assert shifts.tolist() == list(range(5, 41 - 5 + 1))
    assert null.size == 41 - 2 * 5 + 1
    assert m == pytest.approx(RAM.readout_statistic(d, y), abs=1e-12)
    for h, v in zip(shifts, null):
        again = RAM.readout_statistic(d, np.roll(y, int(h)))
        assert v == pytest.approx(again, abs=1e-10)
    m2, null2, _ = RAM.readout_with_null(d, y, h_min=5)
    assert m2 == m and np.array_equal(null2, null)  # deterministic: no random draws


def test_the_estimator_null_is_the_full_shift_set():
    ts, ev = bandit_run(n_trials=50, seed=1)
    res = compute(ts, ev)
    assert res["defined"], res["reason"]
    n = res["details"]["n_updates"]
    assert n == 49
    assert res["n_null"] == n - 2 * 5 + 1
    assert res["details"]["null_shifts"] == [5, n - 5]
    assert res["details"]["null_enumerated"] is True
    assert res["null_family"] == "trial_circular_shift"
    d, y, keys = pairs_of(ts, ev)
    m, null, _ = RAM.readout_with_null(d, y, keys, h_min=5)
    assert res["estimate"] == pytest.approx(m, abs=1e-12)
    assert res["null_mean"] == pytest.approx(np.mean(null), abs=1e-12)
    assert res["null_sd"] == pytest.approx(np.std(null, ddof=1), abs=1e-12)
    assert res["value"] == pytest.approx(m - np.mean(null), abs=1e-12)
    assert compute(ts, ev)["null_mean"] == res["null_mean"]


# --------------------------------------------------------------------------
# fold purge
# --------------------------------------------------------------------------
def test_purge_removes_the_fold_and_its_neighbours():
    keys = np.arange(20)
    test = np.arange(5, 10)
    assert RAM.purged_training_rows(keys, test, 1).tolist() == [
        i for i in range(20) if not 4 <= i <= 10]
    assert RAM.purged_training_rows(keys, test, 2).tolist() == [
        i for i in range(20) if not 3 <= i <= 11]
    assert RAM.purged_training_rows(keys, test, 0).tolist() == [
        i for i in range(20) if not 5 <= i <= 9]
    # after a deleted block (keys 5-9 missing), rows across the gap are not
    # neighbours: no purge reaches over it
    keys_gap = np.r_[0:5, 10:20]
    test_gap = np.flatnonzero((keys_gap >= 10) & (keys_gap <= 12))
    train = RAM.purged_training_rows(keys_gap, test_gap, 1)
    assert keys_gap[train].tolist() == [0, 1, 2, 3, 4] + list(range(14, 20))


def test_a_fold_is_never_trained_on_its_purged_neighbours():
    rng = np.random.default_rng(5)
    n = 50
    d = rng.normal(size=(n, 4))
    y = d @ np.array([1.0, -0.5, 0.0, 0.3]) + 0.5 * rng.normal(size=n)
    folds = RAM.contiguous_folds(n, 5)
    test = folds[2]
    base, _ = RAM.out_of_fold_predictions(d, y, purge=1)
    for neighbour in (test.min() - 1, test.max() + 1):
        y2 = y.copy()
        y2[neighbour] += 50.0
        pred, _ = RAM.out_of_fold_predictions(d, y2, purge=1)
        assert np.array_equal(pred[test], base[test])
    y3 = y.copy()
    y3[0] += 50.0  # a training row of this fold
    pred, _ = RAM.out_of_fold_predictions(d, y3, purge=1)
    assert not np.allclose(pred[test], base[test])
    # with purge 2 (a lag-2 readout) the second neighbour is excluded too
    y4 = y.copy()
    y4[test.min() - 2] += 50.0
    base2, _ = RAM.out_of_fold_predictions(d, y, purge=2)
    pred2, _ = RAM.out_of_fold_predictions(d, y4, purge=2)
    assert np.array_equal(pred2[test], base2[test])
    assert len(folds) == 5 and np.array_equal(np.concatenate(folds), np.arange(n))


def _direct_ridge(xtr, ytr, xte, lam):
    mu, sd = xtr.mean(axis=0), xtr.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    z = (xtr - mu) / sd
    b = np.linalg.solve(z.T @ z + lam * np.eye(z.shape[1]), z.T @ (ytr - ytr.mean()))
    return ((xte - mu) / sd) @ b + ytr.mean()


def _direct_out_of_fold(d, y, keys, purge):
    """The readout written out step by step: 5 contiguous folds purged by
    trial distance, lambda by an inner purged 4-fold CV (pooled inner
    Pearson r, first maximum), one ridge solve per fit."""
    def train_of(k, test):
        return np.array([i for i in range(k.size)
                         if np.min(np.abs(k[test] - k[i])) > purge], dtype=int)

    pred, chosen = np.full(y.size, np.nan), []
    for test in np.array_split(np.arange(y.size), 5):
        tr = train_of(keys, test)
        r_lam = []
        for lam in RAM.LAMBDAS:
            inner = np.full(tr.size, np.nan)
            for g in np.array_split(np.arange(tr.size), 4):
                itr = train_of(keys[tr], g)
                inner[g] = _direct_ridge(d[tr][itr], y[tr][itr], d[tr][g], lam)
            r_lam.append(np.corrcoef(inner, y[tr])[0, 1])
        best = int(np.argmax(r_lam))
        chosen.append(best)
        pred[test] = _direct_ridge(d[tr], y[tr], d[test], RAM.LAMBDAS[best])
    return pred, chosen


@pytest.mark.parametrize("purge", [1, 2])
def test_the_readout_equals_a_direct_ridge_implementation(purge):
    rng = np.random.default_rng(13)
    n, p = 57, 6
    keys = np.sort(rng.choice(np.arange(n + 12), n, replace=False))  # with gaps
    d = rng.normal(size=(n, p))
    y = 0.4 * d @ rng.normal(size=p) + rng.normal(size=n)
    want, chosen = _direct_out_of_fold(d, y, keys, purge)
    got, lam_idx = RAM.out_of_fold_predictions(d, y, keys, purge=purge)
    assert np.allclose(got, want, atol=1e-9)
    assert lam_idx[:, 0].tolist() == chosen
    assert RAM.readout_statistic(d, y, keys, purge=purge) == pytest.approx(
        np.corrcoef(want, y)[0, 1], abs=1e-10)


def test_targets_need_exactly_one_outcome_between_stimuli():
    stim = np.array([1.0, 5.0, 9.0, 13.0, 17.0])
    # before the first stimulus; k = 0; two in [9, 13); one at the onset of
    # stimulus 3 (half-open intervals); one after the last stimulus
    outcomes = np.array([0.5, 2.0, 9.0, 9.5, 13.0, 30.0])
    values = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    y, counts = RAM.stimulus_targets(stim, outcomes, values)
    assert y[0] == 20.0 and y[3] == 50.0 and y[4] == 60.0
    assert np.isnan(y[1]) and np.isnan(y[2])  # none, several
    assert counts == {"n_outcomes_before_first_stimulus": 1,
                      "n_stimuli_without_outcome": 1,
                      "n_stimuli_with_several_outcomes": 1}


def test_update_and_placebo_rows_skip_missing_patterns_and_targets():
    r = np.arange(12, dtype=float).reshape(6, 2) ** 2
    r[4] = np.nan  # a window that leaves the run
    y = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    d, yy, k = RAM.update_pairs(r, y, lag=1)
    assert k.tolist() == [0, 1] and yy.tolist() == [1.0, 2.0]
    assert np.array_equal(d, r[[1, 2]] - r[[0, 1]])
    d, yy, k = RAM.update_pairs(r, y, lag=2)
    assert k.tolist() == [0, 1, 3]
    assert np.array_equal(d, r[[2, 3, 5]] - r[[0, 1, 3]])
    # the precedence placebo: the change r_k - r_{k-1} before outcome k
    d, yy, k = RAM.placebo_pairs(r, y)
    assert k.tolist() == [1, 3] and yy.tolist() == [2.0, 4.0]
    assert np.array_equal(d, r[[1, 3]] - r[[0, 2]])
    with pytest.raises(ValueError):
        RAM.update_pairs(r, y, lag=0)


def test_a_missing_outcome_leaves_a_gap_that_no_purge_crosses():
    ts, ev = bandit_run(n_trials=50, seed=14)
    drop = 20
    keep = [i for i in range(50) if i != drop]
    ev2 = dict(ev, choice_onsets=np.asarray(ev["choice_onsets"])[keep],
               choices=[ev["choices"][i] for i in keep],
               rewards=np.asarray(ev["rewards"])[keep])
    res = compute(ts, ev2)
    assert res["defined"], res["reason"]
    assert res["details"]["n_stimuli_without_outcome"] == 1
    assert res["details"]["n_updates"] == 48 and res["n_null"] == 48 - 9
    d, y, keys = pairs_of(ts, ev2)
    assert keys.size == 48 and drop not in keys.tolist()
    m, null, _ = RAM.readout_with_null(d, y, keys, h_min=5)
    assert res["estimate"] == pytest.approx(m, abs=1e-12)
    assert res["null_mean"] == pytest.approx(np.mean(null), abs=1e-12)


def test_the_jackknife_deletes_contiguous_blocks_and_keeps_the_keys(monkeypatch):
    calls = []

    def fake(d, y, keys=None, **kw):
        calls.append(np.asarray(keys).copy())
        return float(len(calls) - 1)

    monkeypatch.setattr(RAM, "readout_statistic", fake)
    n = 47
    se, reps = RAM.jackknife_readout(np.zeros((n, 3)), np.zeros(n), np.arange(n) + 100)
    assert len(calls) == 10
    deleted = []
    for keys in calls:
        gone = sorted(set(range(100, 100 + n)) - set(keys.tolist()))
        assert gone == list(range(gone[0], gone[-1] + 1))  # contiguous
        assert keys.tolist() == sorted(keys.tolist())
        deleted += gone
    assert deleted == list(range(100, 100 + n))
    assert reps.tolist() == list(range(10))
    assert se == pytest.approx(np.sqrt(0.9 * np.sum((np.arange(10) - 4.5) ** 2)))


# --------------------------------------------------------------------------
# option relabelling
# --------------------------------------------------------------------------
@pytest.mark.parametrize("mapping", [
    {"left": "right", "right": "left"},  # swapped labels: the sign flips
    {"left": "b", "right": "a"},  # new labels in reversed sorted order
    {"left": "a", "right": "b"},  # new labels in the same order
])
def test_the_readout_is_invariant_to_option_relabelling(mapping):
    ts, ev = bandit_run(n_trials=60, seed=2)
    base = compute(ts, ev)
    ev2 = dict(ev, choices=[mapping[c] for c in ev["choices"]])
    other = compute(ts, ev2)
    assert base["defined"] and other["defined"]
    for key in ("estimate", "null_mean", "null_sd", "se", "value"):
        assert other[key] == pytest.approx(base[key], abs=1e-12), key
    assert other["details"]["se_jackknife"] == pytest.approx(
        base["details"]["se_jackknife"], abs=1e-12)
    flipped = mapping["left"] > mapping["right"]
    d, y, _ = pairs_of(ts, ev)
    _, y2, _ = pairs_of(ts, ev2)
    assert np.allclose(y2, -y if flipped else y)


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------
def test_insufficient_updates_guard():
    ts, ev = bandit_run(n_trials=31, seed=4)  # 30 updates: enough
    res = compute(ts, ev)
    assert res["defined"], res["reason"]
    assert res["details"]["n_updates"] == 30
    ts, ev = bandit_run(n_trials=30, seed=4)  # 29 updates
    res = compute(ts, ev)
    assert res["reason"] == R.INSUFFICIENT_UPDATES
    assert res["details"]["n_updates"] == 29
    ts, ev = bandit_run(n_trials=60, seed=4)
    one_option = dict(ev, choices=["left"] * len(ev["choices"]))
    assert compute(ts, one_option)["reason"] == R.INSUFFICIENT_UPDATES
    three = dict(ev, choices=["abc"[k % 3] for k in range(len(ev["choices"]))])
    assert compute(ts, three)["reason"] == "NOT_DEFINED:n_options_not_implemented"
    for res in (compute(ts, one_option), compute(ts, three)):
        assert not res["defined"]
        assert R.status_for_reason(res["reason"]) == R.UNDEFINED
        assert np.isnan(res["estimate"]) and res["se_method"] is None


def test_the_window_ends_before_the_response():
    ts, ev = bandit_run(n_trials=40, seed=6, response_after=0.6)
    res = compute(ts, ev)
    assert res["defined"]
    assert res["details"]["window_response_overlap"] == {
        "n_checked": 40, "n_overlap": 0, "fraction": 0.0}
    # a response at the window's end does not overlap the half-open window
    ts, ev = bandit_run(n_trials=40, seed=6, response_after=0.4)
    assert compute(ts, ev)["details"]["window_response_overlap"]["n_overlap"] == 0
    # responses inside the window
    ts, ev = bandit_run(n_trials=40, seed=6, response_after=0.3)
    res = compute(ts, ev)
    assert res["reason"] == "NOT_DEFINED:window_includes_response"
    assert R.status_for_reason(res["reason"]) == R.UNDEFINED
    # at most 5 % of trials may overlap
    for n_bad, defined in ((1, True), (2, True), (3, False)):
        after = np.full(40, 0.6)
        after[:n_bad] = 0.3
        ts, ev = bandit_run(n_trials=40, seed=6, response_after=after)
        res = compute(ts, ev)
        assert res["details"]["window_response_overlap"]["n_overlap"] == n_bad
        assert res["defined"] is defined
    # a longer lag moves the window onto the response
    ts, ev = bandit_run(n_trials=40, seed=6)
    late = compute(ts, ev, lag_sec=0.25)
    assert late["reason"] == "NOT_DEFINED:window_includes_response"
    # without a response log the check cannot be made
    no_resp = dict(ev, response_onsets=[])
    assert compute(ts, no_resp)["reason"] == "NOT_DEFINED:missing_response_log"


def test_undefined_inputs_carry_vocabulary_reasons():
    ts, ev = bandit_run(n_trials=40, seed=7)
    cases = {
        "NOT_DEFINED:missing_choice_reward_log": dict(
            ev, choice_onsets=[], choices=[], rewards=[]),
        "NOT_DEFINED:choice_reward_log_misaligned": dict(
            ev, rewards=ev["rewards"][:-1]),
        "NOT_DEFINED:prediction_error_model_unidentifiable": dict(
            ev, rewards=np.ones(len(ev["rewards"]))),
        "NOT_DEFINED:missing_stimulus_events": dict(ev, stimulus_onsets=[]),
    }
    for reason, events in cases.items():
        res = compute(ts, events)
        assert res["reason"] == reason
        assert R.is_reason(reason) and R.status_for_reason(reason) == R.UNDEFINED
    bad = ts.copy()
    bad[0, 10] = np.nan
    assert compute(bad, ev)["reason"] == "NOT_DEFINED:non_finite_timeseries"


# --------------------------------------------------------------------------
# readout, SE methods and the record
# --------------------------------------------------------------------------
def test_a_planted_update_is_read_out_and_its_absence_is_not():
    ts, ev = bandit_run(n_trials=80, seed=8, gain=1.0)
    on = compute(ts, ev)
    ts0, ev0 = bandit_run(n_trials=80, seed=8, gain=0.0)
    off = compute(ts0, ev0)
    assert on["value"] > 0.5 and on["details"]["p_shift"] < 0.05
    assert abs(off["value"]) < 0.3
    for res in (on, off):
        assert res["se_method"] == "shift_null_sd"
        assert res["se"] == res["null_sd"]
        assert res["se_df"] == res["n_null"] - 1
        assert res["exact"] is False
        assert res["details"]["construct"] == "RAM-PE"
        assert res["details"]["persistence"] == "not_tested"
        assert set(res["details"]["placebo"]) >= {"estimate", "excess", "p_shift"}
        assert set(res["details"]["descriptors"]) >= {
            "evoked_magnitude", "fir_latency_sec"}
    jk = compute(ts, ev, se_method="jackknife_trials_10")
    assert jk["se_method"] == "jackknife_trials_10"
    assert jk["se"] == pytest.approx(on["details"]["se_jackknife"])
    assert jk["se_df"] == 9.0
    assert jk["estimate"] == on["estimate"]


def test_the_events_table_is_read_through_its_channel():
    ts, ev = bandit_run(n_trials=40, seed=9)
    base = compute(ts, ev)
    typed = RAM.compute_ram_v3(ts, DT, events_table(ev))
    assert typed["defined"] and typed["details"]["events_source"] == "events_table"
    assert typed["details"]["n_stimuli"] == 40  # the other channel's probes are ignored
    for key in ("estimate", "null_mean", "null_sd", "se"):
        assert typed[key] == pytest.approx(base[key], abs=1e-12)
    untyped = RAM.compute_ram_v3(ts, DT, events_table(ev, decoys=False,
                                                      channel_column=False))
    assert untyped["reason"] == "NOT_DEFINED:untyped_events"
    all_events = RAM.compute_ram_v3(ts, DT, events_table(ev, decoys=False,
                                                         channel_column=False),
                                    params={"channel": None})
    assert all_events["estimate"] == pytest.approx(base["estimate"], abs=1e-12)


def test_component_fields_build_a_result_record():
    ts, ev = bandit_run(n_trials=40, seed=10)
    res = compute(ts, ev, se_method="jackknife_trials_10")
    common = dict(principle="RAM", estimator_version=RAM.ESTIMATOR_VERSION,
                  declaration_id="R", observation_stage="source",
                  protocol_id="A-RAM160", protocol_hash="0" * 64)
    rec = REC.ComponentRecord(status=R.UNDEFINED, reason=R.INCONCLUSIVE,
                              **common, **RAM.component_fields(res))
    assert rec.se_method == "jackknife_trials_10" and rec.se_df == 9.0
    assert rec.null_family == "trial_circular_shift"
    assert REC.ComponentRecord.from_dict(rec.to_dict()) == rec
    und = compute(ts, dict(ev, choices=["left"] * 40))
    rec = REC.ComponentRecord(status=R.UNDEFINED, reason=und["reason"],
                              **common, **RAM.component_fields(und))
    assert rec.estimate is None and rec.se is None and rec.n_null == 0


def test_absent_reachability_descriptor():
    assert RAM.absent_reachable(0.12, 0.4, 69) is False
    assert RAM.absent_reachable(0.01, 1.0, 69) is True
    assert RAM.absent_reachable(float("nan"), 0.4, 69) is None
    ts, ev = bandit_run(n_trials=40, seed=11)
    res = RAM.compute_ram_v3(ts, DT, ev, params={"channel": None}, reference=0.4)
    assert res["details"]["absent_reachable"] is False


# --------------------------------------------------------------------------
# family C1 and the bench path
# --------------------------------------------------------------------------
@pytest.mark.parametrize("model", ["C1", "C"])
def test_family_c1_is_declared_not_applicable(model):
    ts, ev = bandit_run(n_trials=40, seed=12)
    res = RAM.compute_ram_v3(ts, DT, ev, params={"channel": None},
                             observation_model=model)
    assert res["reason"] == R.NOT_APPLICABLE_OBSERVATION_MODEL
    assert not res["defined"]
    assert R.status_for_reason(res["reason"]) == R.UNDEFINED
    assert res["details"]["observation_model"] == "C1"
    assert "carrier" in res["details"]["not_applicable"]
    assert res["details"]["descriptive"]["defined"] is True  # reported, not judged
    fields = RAM.component_fields(res)
    assert fields["estimate"] is None and fields["se"] is None
    common = dict(principle="RAM", estimator_version=RAM.ESTIMATOR_VERSION,
                  declaration_id="R", observation_stage="source", protocol_id="C1-R",
                  protocol_hash="0" * 64, **fields)
    rec = REC.ComponentRecord(status=R.UNDEFINED, reason=res["reason"], **common)
    assert rec.c is None
    for status in (R.PRESENT, R.ABSENT):  # never decided without a construct value
        with pytest.raises(REC.RecordSchemaError):
            REC.ComponentRecord(status=status, **common)
    assert RAM.not_applicable("A") is None and RAM.not_applicable(None) is None


def test_family_a_and_c_systems_on_a_development_seed():
    from impact_pipeline.bench.witnesses import (
        build_witness_system, get_witness, load_witnesses)

    catalogue = load_witnesses()
    a = build_witness_system(get_witness("PC_nominal", catalogue), 0, "A")
    res = RAM.compute_ram_v3(a.ts, a.dt, a.events, observation_model=a.meta["family"])
    assert res["defined"], res["reason"]
    assert res["details"]["n_stimuli"] == 80
    assert res["details"]["n_updates"] == 79
    assert res["n_null"] == 70 and res["se_df"] == 69.0
    assert res["details"]["window_response_overlap"]["fraction"] == 0.0
    assert res["details"]["options"] == ["0", "1"]
    c = build_witness_system(get_witness("PC_nominal", catalogue), 0, "C")
    res = RAM.compute_ram_v3(c.ts, c.dt, c.events, observation_model=c.meta["family"],
                             describe_not_applicable=False)
    assert res["reason"] == R.NOT_APPLICABLE_OBSERVATION_MODEL
