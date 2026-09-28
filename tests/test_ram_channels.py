"""RAM impact channels, prediction-error update (Rescorla-Wagner) and typed events."""

import numpy as np
import pandas as pd
import pytest

from impact_pipeline import event_parsing as ep
from impact_pipeline import mpc_metrics as mm

TR = 0.1
WINDOW_KW = dict(
    tr=TR,
    response_model="boxcar",
    latency_method="hrf_peak",
    response_window_sec=0.5,
    goal_objective_window_sec=0.5,
    feedback_window_sec=0.5,
    quality_lag_sec=0.1,
    quality_null_samples=200,
    return_details=True,
)


def _bandit_agent(
    seed, eta, n_trials=60, n_regions=16, beta=5.0, amp=2.0, permute_rewards=False
):
    """
    Two-armed reversal bandit (p = 0.8 / 0.2, reversal every 20 trials) with
    a Q-learning agent (learning rate ``eta``, softmax ``beta``). The neural
    stimulus response is ``w * p_stim`` and the stimulus-response gain ``w``
    learns from the same prediction error (``w += eta * delta``); ``eta = 0``
    is a non-learning agent. Goal cues and valued feedback events complete
    the RAM event contract. Returns ``(ts, bundle)`` with the logged choices
    and rewards (optionally with the reward log permuted).
    """
    rng = np.random.default_rng(seed)
    goal = 0.5 + 3.0 * np.arange(n_trials)
    stim = goal + 0.8
    fb = goal + 2.0
    n_tp = int((fb[-1] + 3.0) / TR)
    noise = rng.standard_normal((n_regions, n_tp))
    ts = np.zeros((n_regions, n_tp))
    for t in range(1, n_tp):
        ts[:, t] = 0.5 * ts[:, t - 1] + noise[:, t]
    p_stim, p_goal, p_fb = (rng.standard_normal(n_regions) for _ in range(3))
    q = np.array([0.5, 0.5])
    w = 1.0
    probs = np.array([0.8, 0.2])
    choices, rewards = [], []
    for k in range(n_trials):
        if k and k % 20 == 0:
            probs = probs[::-1]
        pc = np.exp(beta * q) / np.exp(beta * q).sum()
        c = int(rng.random() < pc[1])
        r = float(rng.random() < probs[c])
        v = rng.standard_normal()
        i_g, i_s, i_f = (int(round(t / TR)) for t in (goal[k], stim[k], fb[k]))
        ts[:, i_g + 1 : i_g + 6] += (v * p_goal)[:, None]
        ts[:, i_s + 1 : i_s + 6] += (amp * w * p_stim + 0.5 * v * p_goal)[:, None]
        ts[:, i_f + 1 : i_f + 6] += ((0.5 + r) * p_fb)[:, None]
        delta = r - q[c]
        q[c] += eta * delta
        w += eta * delta
        choices.append(c)
        rewards.append(r)
    log = np.asarray(rewards)
    if permute_rewards:
        log = log[rng.permutation(n_trials)]
    bundle = dict(
        onsets=stim.tolist(),
        goal_onsets=goal.tolist(),
        feedback_onsets=fb.tolist(),
        feedback_values=list(rewards),
        choice_onsets=fb.tolist(),
        choices=choices,
        rewards=log.tolist(),
    )
    return ts, bundle


def _u(ts, bundle, **kw):
    d = mm.compute_RAM(
        ts, stimulus_onsets=bundle, update="prediction_error", **dict(WINDOW_KW, **kw)
    )
    return d


def test_channel_constants_are_mirrored():
    assert ep.IMPACT_CHANNELS == mm.RAM_IMPACT_CHANNELS
    assert ep.IMPLEMENTED_RAM_CHANNELS == mm.RAM_IMPLEMENTED_CHANNELS
    assert ep.SRPI_MIN_EVENTS_PER_CLASS == 3


def test_rescorla_wagner_fit_recovers_learning_rate():
    rng = np.random.default_rng(0)
    alpha, beta = 0.3, 5.0
    q = np.array([0.5, 0.5])
    probs = np.array([0.8, 0.2])
    choices, rewards = [], []
    for k in range(400):
        if k and k % 40 == 0:
            probs = probs[::-1]
        p1 = 1.0 / (1.0 + np.exp(-beta * (q[1] - q[0])))
        c = int(rng.random() < p1)
        r = float(rng.random() < probs[c])
        q[c] += alpha * (r - q[c])
        choices.append(c)
        rewards.append(r)
    fit = mm._fit_rescorla_wagner(choices, rewards)
    assert fit["n_options"] == 2 and fit["n_trials"] == 400
    assert abs(fit["alpha"] - alpha) <= 0.15
    assert 2.5 <= fit["beta"] <= 10.0
    assert fit["choice_informative"]
    # Prediction errors of the fitted model follow the RW recursion.
    pe = fit["prediction_errors"]
    assert pe[0] == pytest.approx(rewards[0] - 0.5)
    # Random choices carry no information about learning.
    rand = mm._fit_rescorla_wagner(
        rng.integers(0, 2, 200).tolist(), rng.integers(0, 2, 200).astype(float)
    )
    assert not rand["choice_informative"]
    # Not identifiable: one option, constant rewards, too few trials.
    assert mm._fit_rescorla_wagner([1, 1, 1, 1], [0.0, 1.0, 1.0, 0.0]) is None
    assert mm._fit_rescorla_wagner([0, 1, 0, 1], [1.0, 1.0, 1.0, 1.0]) is None
    assert mm._fit_rescorla_wagner([0, 1], [0.0, 1.0]) is None
    # String labels and reward scale do not matter.
    lab = mm._fit_rescorla_wagner(
        ["L" if c == 0 else "R" for c in choices], [10.0 * r - 5.0 for r in rewards]
    )
    assert lab["alpha"] == fit["alpha"] and lab["beta"] == fit["beta"]
    np.testing.assert_allclose(lab["prediction_errors"], 10.0 * pe)


def test_prediction_error_update_separates_learning_from_non_learning_agents():
    learn, fixed, perm = [], [], []
    for seed in range(4):
        d = _u(*_bandit_agent(seed, eta=0.3))
        assert d["undefined_reason"] is None
        assert d["update"] == "prediction_error"
        assert d["prediction_error_model"]["choice_informative"]
        assert abs(d["prediction_error_model"]["alpha"] - 0.3) <= 0.2
        learn.append((d["components"]["adaptive_update"], d["adaptive_update_p_null"]))
        d0 = _u(*_bandit_agent(seed, eta=0.0))
        fixed.append(
            (d0["components"]["adaptive_update"], d0["adaptive_update_p_null"])
        )
        dp = _u(*_bandit_agent(seed, eta=0.3, permute_rewards=True))
        perm.append((dp["components"]["adaptive_update"], dp["adaptive_update_p_null"]))
    learn, fixed, perm = (np.asarray(v) for v in (learn, fixed, perm))
    print(
        "U (score, p): eta=0.3",
        learn.round(3).tolist(),
        "eta=0",
        fixed.round(3).tolist(),
        "permuted rewards",
        perm.round(3).tolist(),
    )
    assert np.all(learn[:, 1] <= 0.05) and np.all(learn[:, 0] > 0.0)
    assert np.median(fixed[:, 1]) > 0.1 and np.median(perm[:, 1]) > 0.1
    assert learn[:, 0].min() > fixed[:, 0].mean()
    assert fixed[:, 0].mean() < 0.02 and perm[:, 0].mean() < 0.02


def test_legacy_feedback_magnitude_update_misses_the_learning_agent():
    ts, bundle = _bandit_agent(0, eta=0.3)
    legacy = mm.compute_RAM(ts, stimulus_onsets=bundle, **WINDOW_KW)
    explicit = mm.compute_RAM(
        ts, stimulus_onsets=bundle, update="feedback_magnitude", **WINDOW_KW
    )
    assert legacy["update"] == "feedback_magnitude"
    assert legacy["value"] == explicit["value"]
    pe = _u(ts, bundle)
    assert legacy["components"]["adaptive_update"] < pe["components"]["adaptive_update"]


def test_prediction_error_update_needs_a_choice_reward_log():
    ts, bundle = _bandit_agent(0, eta=0.3)
    no_log = {
        k: v
        for k, v in bundle.items()
        if k not in ("choices", "rewards", "choice_onsets")
    }
    d = _u(ts, no_log)
    assert (
        d["component_undefined_reasons"]["adaptive_update"]
        == "missing_choice_reward_log"
    )
    assert np.isnan(d["value"]) and d["undefined_reason"] == "missing_choice_reward_log"
    # With U unweighted the other components still define RAM.
    d_w = _u(ts, no_log, quality_weights=(1.0, 1.0, 0.0))
    assert d_w["undefined_reason"] is None
    bad = dict(bundle, rewards=bundle["rewards"][:-1])
    assert _u(ts, bad)["undefined_reason"] == "choice_reward_log_misaligned"
    one_option = dict(bundle, choices=[0] * len(bundle["choices"]))
    assert (
        _u(ts, one_option)["undefined_reason"]
        == "prediction_error_model_unidentifiable"
    )


def test_adaptation_locus_is_declared_in_details():
    ts, bundle = _bandit_agent(1, eta=0.3)
    assert _u(ts, bundle)["adaptation_locus"] == "undeclared"
    assert _u(ts, bundle, adaptation_locus="weights")["adaptation_locus"] == "weights"
    with pytest.raises(ValueError, match="adaptation_locus"):
        _u(ts, bundle, adaptation_locus="synapses")
    with pytest.raises(ValueError, match="update must be one of"):
        mm.compute_RAM(ts, tr=TR, stimulus_onsets=bundle, update="td")


def _typed_events(bundle, channel_of_trial):
    """BIDS-like events table with an impact_channel column per trial."""
    rows = []
    for k, (g, s, f) in enumerate(
        zip(bundle["goal_onsets"], bundle["onsets"], bundle["feedback_onsets"])
    ):
        ch = channel_of_trial(k)
        rows.append(dict(onset=g, trial_type="goal_cue", impact_channel=ch))
        rows.append(dict(onset=s, trial_type="stim", impact_channel=ch))
        rows.append(
            dict(
                onset=f,
                trial_type="feedback",
                impact_channel=ch,
                reward=bundle["feedback_values"][k],
                choice=bundle["choices"][k],
            )
        )
    return pd.DataFrame(rows)


def test_events_parser_splits_channels_and_logs_choices():
    ts, bundle = _bandit_agent(0, eta=0.3, n_trials=12)
    df = _typed_events(
        bundle, lambda k: "behavioural_feedback" if k < 8 else "perturbational"
    )
    df.loc[len(df)] = dict(onset=40.0, trial_type="stim", impact_channel="n/a")
    out = ep.events_table_to_bundle(df)
    assert out["impact_channels"] == ["behavioural_feedback", "perturbational"]
    beh = out["channel_bundles"]["behavioural_feedback"]
    assert beh["onsets"] == bundle["onsets"][:8]
    assert beh["goal_onsets"] == bundle["goal_onsets"][:8]
    assert beh["feedback_values"] == bundle["feedback_values"][:8]
    assert beh["choices"] == [str(c) for c in bundle["choices"][:8]]
    assert out["channel_bundles"]["perturbational"]["onsets"] == bundle["onsets"][8:]
    # The untyped view keeps every row (legacy behaviour), incl. unlabeled ones.
    assert len(out["onsets"]) == 13
    assert out["choice_onsets"] == bundle["feedback_onsets"]
    assert out["rewards"] == bundle["feedback_values"]


def test_choice_and_reward_on_separate_rows_are_paired():
    df = pd.DataFrame(
        [
            dict(onset=1.0, trial_type="response", choice="left"),
            dict(onset=2.0, trial_type="feedback", reward=1.0),
            dict(onset=3.0, trial_type="feedback", reward=0.0),  # no pending choice
            dict(onset=4.0, trial_type="response", choice="right"),
            dict(onset=4.5, trial_type="response", choice="left"),
            dict(onset=5.0, trial_type="feedback", reward=0.0),
            dict(onset=6.0, trial_type="response", choice=2.0, reward=1.0),
        ]
    )
    out = ep.events_table_to_bundle(df)
    assert out["choice_onsets"] == [2.0, 5.0, 6.0]
    assert out["choices"] == ["left", "left", "2"]
    assert out["rewards"] == [1.0, 0.0, 1.0]
    assert ep.events_table_to_bundle(df.drop(columns=["choice"]))["choices"] == []
    assert ep.empty_event_bundle()["channel_bundles"] == {}


def test_compute_ram_per_channel():
    ts, bundle = _bandit_agent(2, eta=0.3)
    df = _typed_events(
        bundle, lambda k: "behavioural_feedback" if k % 2 == 0 else "covert_neural"
    )
    df.loc[len(df)] = dict(onset=3.0, trial_type="stim", impact_channel="endogenous")
    df.loc[len(df)] = dict(onset=3.5, trial_type="stim", impact_channel="telepathic")
    typed = ep.events_table_to_bundle(df)
    kw = dict(WINDOW_KW, quality_null_samples=50)
    beh = mm.compute_RAM(
        ts, stimulus_onsets=typed, impact_channel="behavioural_feedback", **kw
    )
    direct = mm.compute_RAM(
        ts, stimulus_onsets=typed["channel_bundles"]["behavioural_feedback"], **kw
    )
    assert beh["impact_channel"] == "behavioural_feedback"
    assert beh["n_stimulus_events"] == 30
    assert beh["value"] == direct["value"] or (
        np.isnan(beh["value"]) and np.isnan(direct["value"])
    )
    for ch in ("perturbational", "endogenous"):
        d = mm.compute_RAM(ts, stimulus_onsets=typed, impact_channel=ch, **kw)
        assert np.isnan(d["value"]) and d["undefined_reason"] == "NOT_IMPLEMENTED"
        assert d["impact_channel"] == ch
    untyped = mm.compute_RAM(
        ts, stimulus_onsets=bundle, impact_channel="covert_neural", **kw
    )
    assert untyped["undefined_reason"] == "untyped_events"
    assert np.isnan(
        mm.compute_RAM(
            ts,
            stimulus_onsets=typed,
            impact_channel="endogenous",
            tr=TR,
            response_model="boxcar",
        )
    )
    with pytest.raises(ValueError, match="impact_channel must be one of"):
        mm.compute_RAM(ts, tr=TR, stimulus_onsets=typed, impact_channel="telepathic")
    by = mm.compute_RAM_by_channel(ts, stimulus_onsets=typed, **kw)
    assert sorted(by) == [
        "behavioural_feedback",
        "covert_neural",
        "endogenous",
        "telepathic",
    ]
    assert by["endogenous"]["undefined_reason"] == "NOT_IMPLEMENTED"
    assert by["telepathic"]["undefined_reason"] == "unknown_impact_channel"
    assert by["behavioural_feedback"]["value"] == beh["value"] or np.isnan(beh["value"])
    assert by["covert_neural"]["n_stimulus_events"] == 30
    # Legacy call (no channel) uses every event and records 'untyped'.
    legacy = mm.compute_RAM(ts, stimulus_onsets=typed, **kw)
    assert legacy["impact_channel"] == "untyped" and legacy["n_stimulus_events"] == 62


def test_readiness_reports_ram_channels(tmp_path):
    from impact_pipeline.mpc_readiness import check_mpc_readiness

    prep, bids, subj = tmp_path / "prep", tmp_path / "bids", "01"
    ts_path = prep / subj / "awake" / "audio" / f"{subj}_run-1_schaefer400_ts.npy"
    ts_path.parent.mkdir(parents=True)
    np.save(ts_path, np.random.RandomState(0).randn(300, 6))
    _, bundle = _bandit_agent(0, eta=0.3, n_trials=16)
    df = _typed_events(
        bundle, lambda k: "behavioural_feedback" if k < 12 else "perturbational"
    )
    ev = bids / f"sub-{subj}" / "func" / f"sub-{subj}_task-audioawake_run-01_events.tsv"
    ev.parent.mkdir(parents=True)
    df.to_csv(ev, sep="\t", index=False)
    table, summary = check_mpc_readiness(
        str(prep), str(bids), "schaefer400", "audio", ["awake"]
    )
    row = table.iloc[0]
    assert row["RAM_channels"] == "behavioural_feedback;perturbational"
    assert (
        row["RAM_channel_status"]
        == "behavioural_feedback:ok;perturbational:NOT_IMPLEMENTED"
    )
    assert row["RAM_channels_ready"] == "behavioural_feedback"
    assert row["n_choice_trials"] == 16
    assert summary["ram_channels"] == {
        "behavioural_feedback": {"rows": 1, "ready": 1},
        "perturbational": {"rows": 1, "ready": 0},
    }
