"""Known-answer, null and contract tests for SRPI-agency (mode='agency')."""

import numpy as np
import pandas as pd
import pytest

from impact_pipeline import event_parsing as ep
from impact_pipeline import mpc_metrics as mm

PERIOD = 12  # samples of the ongoing oscillation that defines the phase bins
KW = dict(
    tr=1.0,
    mode="agency",
    pre_window_sec=2.0,
    response_lag_sec=2.0,
    response_window_sec=4.0,
    agency_null_permutations=100,
    return_details=True,
)


def _agency_task(seed, tag=1.0, motor_ramp=0.0, n_pairs=40, n_regions=20, n_stim=2):
    """
    Reafference task: every self-caused event (onset jittered within the
    oscillation period) is replayed one period later as an other-caused event
    with the identical stimulus and phase bin. With ``tag > 0`` self-caused
    responses carry an efference-copy signature (attenuated sensory response
    plus a tag pattern); ``motor_ramp`` adds activity in the pre-event window
    of self-caused events only (motor preparation, no efference effect).
    """
    rng = np.random.default_rng(seed)
    base = 30 + 2 * PERIOD * np.arange(n_pairs)
    self_on = base + rng.integers(0, PERIOD, n_pairs)
    other_on = self_on + PERIOD
    n_tp = int(other_on[-1] + 40)
    noise = rng.standard_normal((n_regions, n_tp))
    ts = np.zeros((n_regions, n_tp))
    for t in range(1, n_tp):
        ts[:, t] = 0.5 * ts[:, t - 1] + noise[:, t]
    ts += 0.5 * np.sin(2 * np.pi * np.arange(n_tp) / PERIOD)[None, :]
    stim_pats = rng.standard_normal((n_stim, n_regions))
    tag_pat = rng.standard_normal(n_regions)
    motor_pat = rng.standard_normal(n_regions)
    sid = rng.integers(0, n_stim, n_pairs)
    rows = []
    for i in range(n_pairs):
        s, o = int(self_on[i]), int(other_on[i])
        a_s, a_o = 1.0 + 0.3 * rng.standard_normal(2)
        resp_s = a_s * ((1.0 - 0.3 * tag) * stim_pats[sid[i]] + 0.8 * tag * tag_pat)
        ts[:, s + 2 : s + 6] += resp_s[:, None]
        ts[:, o + 2 : o + 6] += (a_o * stim_pats[sid[i]])[:, None]
        if motor_ramp:
            ts[:, s - 2 : s] += motor_ramp * motor_pat[:, None]
        phase = int((s % PERIOD) // 3)
        rows.append(
            dict(
                onset=float(s),
                trial_type="self_caused",
                yoked_to=None,
                phase_bin=phase,
                stim_id=f"s{sid[i]}",
            )
        )
        rows.append(
            dict(
                onset=float(o),
                trial_type="other_caused",
                yoked_to=float(s),
                phase_bin=int((o % PERIOD) // 3),
                stim_id=f"s{sid[i]}",
            )
        )
    return ts, rows


def _random_relabel(rows, seed):
    """Swap self/other roles within each yoked pair with probability 1/2."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(0, len(rows), 2):
        s, o = dict(rows[i]), dict(rows[i + 1])
        if rng.random() < 0.5:
            s_on, o_on = o["onset"], s["onset"]
            s = dict(s, onset=s_on)
            o = dict(o, onset=o_on, yoked_to=s_on)
        out += [s, o]
    return out


def test_efference_tag_separates_self_caused_from_yoked_replays():
    for seed in range(3):
        ts, rows = _agency_task(seed, tag=1.0)
        d = mm.compute_SRPI(ts, agency_events=rows, **KW)
        assert d["undefined_reason"] is None
        assert d["mode"] == "agency"
        assert (
            d["contract"]["valid"] and d["contract"]["stimulus_identity"] == "verified"
        )
        # Cross-validated AUC well above its label-permutation null.
        assert d["separability_cv_auc"] > 0.8
        assert abs(d["separability_null_auc_mean"] - 0.5) < 0.05
        assert d["separability_p_null"] <= 1.0 / 101.0 + 1e-12
        assert d["value"] > 0.1
        assert d["SRPI_z"] > 3.0
        assert d["SRPI_excess"] == pytest.approx(d["value"])


def test_without_tag_srpi_agency_stays_within_null():
    zs, aucs = [], []
    for seed in range(6):
        ts, rows = _agency_task(seed, tag=0.0)
        d = mm.compute_SRPI(ts, agency_events=rows, **KW)
        assert d["undefined_reason"] is None
        zs.append(d["SRPI_z"])
        aucs.append(d["separability_cv_auc"])
    print("SRPI-agency without tag: z", np.round(zs, 2), "AUC", np.round(aucs, 2))
    assert np.all(np.abs(zs) < 3.0)
    assert abs(np.mean(zs)) < 1.0
    assert abs(np.mean(aucs) - 0.5) < 0.1


def test_random_labels_give_about_zero():
    vals, zs = [], []
    for seed in range(6):
        ts, rows = _agency_task(seed, tag=1.0)
        d = mm.compute_SRPI(ts, agency_events=_random_relabel(rows, 100 + seed), **KW)
        vals.append(d["value"])
        zs.append(d["SRPI_z"])
    print("SRPI-agency random labels: value", np.round(vals, 3), "z", np.round(zs, 2))
    assert np.all(np.abs(zs) < 3.0)
    assert abs(np.mean(vals)) < 0.05


def test_pre_event_state_difference_is_partialled_out():
    # Motor preparation before self-caused events, no efference effect: the
    # raw response changes (post - pre) separate the classes perfectly; the
    # pre-state partialling (refitted inside each CV fold) removes it.
    zs = []
    for seed in range(3):
        ts, rows = _agency_task(seed, tag=0.0, motor_ramp=3.0)
        d = mm.compute_SRPI(ts, agency_events=rows, **KW)
        zs.append(d["SRPI_z"])
        assert d["separability_cv_auc"] < 0.75
        raw = mm.compute_SRPI(ts, agency_events=rows, agency_pre_components=0, **KW)
        assert raw["separability_cv_auc"] > 0.95
        assert raw["SRPI_z"] > 3.0
    assert np.all(np.abs(zs) < 3.0)


def test_efference_collinear_with_pre_state_is_not_credited():
    # Documented limitation: when the pre-event state itself separates the
    # classes, the partialling removes any class difference collinear with it,
    # including a genuine efference effect (conservative, about 0).
    for seed in range(2):
        ts, rows = _agency_task(seed, tag=1.0, motor_ramp=3.0)
        d = mm.compute_SRPI(ts, agency_events=rows, **KW)
        print(
            "tag + motor preparation:",
            round(d["SRPI_z"], 2),
            round(d["separability_cv_auc"], 2),
        )
        assert d["SRPI_z"] < 3.0


def test_legacy_one_sided_srpi_misses_efference_attenuation():
    ts, rows = _agency_task(0, tag=1.0)
    self_on = [r["onset"] for r in rows if r["trial_type"] == "self_caused"]
    other_on = [r["onset"] for r in rows if r["trial_type"] == "other_caused"]
    legacy = mm.compute_SRPI(
        ts,
        tr=1.0,
        self_onsets=self_on,
        nonself_onsets=other_on,
        pre_window_sec=2.0,
        response_lag_sec=2.0,
        response_window_sec=4.0,
        return_details=True,
    )
    agency = mm.compute_SRPI(ts, agency_events=rows, **KW)
    # Self-caused responses are attenuated: the legacy one-sided reactivity is
    # 0 and its hard-zero geometric mean gives SRPI = 0 despite the separable
    # efference signature; the agency mode (two-sided, no hard zero) sees it.
    assert legacy["mode"] == "legacy"
    assert legacy["components_raw"]["reactivity_bias"] == 0.0
    assert legacy["value"] == 0.0
    assert agency["signed_reactivity_bias"] < 0.0
    assert agency["value"] > 0.1


def test_agency_mode_is_deterministic_and_ignores_legacy_onsets():
    ts, rows = _agency_task(1, tag=1.0)
    a = mm.compute_SRPI(ts, agency_events=rows, **KW)
    b = mm.compute_SRPI(
        ts,
        agency_events=pd.DataFrame(rows),
        self_onsets=[5.0],
        nonself_onsets=[9.0],
        **KW,
    )
    assert a["value"] == b["value"]
    assert a["components"] == b["components"]
    assert (
        mm.compute_SRPI(ts, agency_events=rows, **dict(KW, return_details=False))
        == a["value"]
    )


def _rows_for_contract():
    return [
        dict(
            onset=10.0,
            trial_type="self_caused",
            yoked_to=None,
            phase_bin=1,
            stim_id="a",
        ),
        dict(
            onset=22.0,
            trial_type="other_caused",
            yoked_to=10.0,
            phase_bin=1,
            stim_id="a",
        ),
        dict(
            onset=40.0,
            trial_type="Self-Caused",
            yoked_to=None,
            phase_bin=2,
            stim_id="b",
        ),
        dict(
            onset=52.0,
            trial_type="other caused",
            yoked_to=40.0,
            phase_bin=2,
            stim_id="b",
        ),
        dict(
            onset=60.0,
            trial_type="self_caused",
            yoked_to=None,
            phase_bin=0,
            stim_id="a",
        ),
    ]


def test_contract_validator_accepts_matched_design():
    rep = ep.validate_srpi_agency_contract(_rows_for_contract())
    assert rep["valid"] and rep["violations"] == []
    assert rep["n_self_caused"] == 3 and rep["n_other_caused"] == 2
    assert rep["n_pairs"] == 2 and rep["n_self_unyoked"] == 1
    assert rep["yoking_key"] == "onset"
    assert (
        rep["stimulus_identity"] == "verified" and rep["stimulus_column"] == "stim_id"
    )
    assert rep["pairs"] == [(0, 1), (2, 3)]


@pytest.mark.parametrize(
    "mutate, code",
    [
        (lambda r: r[1].update(phase_bin=3), "phase_bin_mismatch"),
        (lambda r: r[3].update(stim_id="a"), "stimulus_mismatch"),
        (lambda r: r[1].update(yoked_to=None), "missing_yoked_to"),
        (lambda r: r[3].update(yoked_to=41.5), "unresolved_yoking"),
        (lambda r: r[0].update(phase_bin=None), "missing_phase_bin"),
        (lambda r: r[3].update(stim_id=None), "missing_stimulus_identity"),
    ],
)
def test_contract_validator_rejects_violations(mutate, code):
    rows = _rows_for_contract()
    mutate(rows)
    rep = ep.validate_srpi_agency_contract(rows)
    assert not rep["valid"]
    assert code in rep["violations"]


def test_contract_validator_structural_violations():
    assert ep.validate_srpi_agency_contract(None)["violations"] == [
        "missing_agency_events"
    ]
    rows = [
        {k: v for k, v in r.items() if k != "phase_bin"} for r in _rows_for_contract()
    ]
    assert (
        "missing_column:phase_bin"
        in ep.validate_srpi_agency_contract(rows)["violations"]
    )
    only_self = [r for r in _rows_for_contract() if "self" in r["trial_type"].lower()]
    rep = ep.validate_srpi_agency_contract(only_self)
    assert "missing_other_caused_events" in rep["violations"]
    dup = _rows_for_contract() + [
        dict(
            onset=10.0,
            trial_type="self_caused",
            yoked_to=None,
            phase_bin=1,
            stim_id="a",
        )
    ]
    assert "ambiguous_yoking" in ep.validate_srpi_agency_contract(dup)["violations"]


def test_contract_yoking_by_event_id():
    rows = _rows_for_contract()
    for i, r in enumerate(rows):
        r["event_id"] = f"e{i}"
    rows[1]["yoked_to"] = "e0"
    rows[3]["yoked_to"] = "e2"
    rep = ep.validate_srpi_agency_contract(pd.DataFrame(rows))
    assert rep["valid"] and rep["yoking_key"] == "event_id" and rep["n_pairs"] == 2
    rows[3]["yoked_to"] = "e1"  # an other-caused event, not a self-caused one
    rep = ep.validate_srpi_agency_contract(rows)
    assert "unresolved_yoking" in rep["violations"]
    rows[3]["yoked_to"] = "e2"
    rows[4]["event_id"] = "e0"
    assert "duplicate_event_id" in ep.validate_srpi_agency_contract(rows)["violations"]


def test_compute_srpi_agency_undefined_on_contract_violation():
    ts, rows = _agency_task(0, tag=1.0)
    rows[1] = dict(rows[1], phase_bin=rows[1]["phase_bin"] + 1)
    d = mm.compute_SRPI(ts, agency_events=rows, **KW)
    assert np.isnan(d["value"])
    assert d["undefined_reason"] == "agency_contract_violation:phase_bin_mismatch"
    assert d["contract"]["valid"] is False
    assert np.isnan(d["SRPI_null_mean"])
    missing = mm.compute_SRPI(ts, **KW)
    assert (
        missing["undefined_reason"] == "agency_contract_violation:missing_agency_events"
    )
    assert np.isnan(
        mm.compute_SRPI(ts, agency_events=rows, **dict(KW, return_details=False))
    )


def test_agency_needs_min_events_per_class_after_windowing():
    ts, rows = _agency_task(0, tag=1.0, n_pairs=4)
    d = mm.compute_SRPI(ts, agency_events=rows, **dict(KW, min_events_per_class=5))
    assert d["undefined_reason"] == "insufficient_self_events_after_windowing"
    assert d["counts"]["n_pairs_used"] == 4


def test_agency_parameter_validation():
    ts, rows = _agency_task(0, tag=1.0, n_pairs=6)
    with pytest.raises(ValueError, match="mode must be one of"):
        mm.compute_SRPI(ts, tr=1.0, mode="content")
    with pytest.raises(ValueError, match="agency_null_permutations"):
        mm.compute_SRPI(ts, agency_events=rows, **dict(KW, agency_null_permutations=1))
    with pytest.raises(ValueError, match="agency_components"):
        mm.compute_SRPI(ts, agency_events=rows, **dict(KW, agency_components=0))


def test_events_table_parses_agency_rows_for_the_validator():
    rows = _rows_for_contract()
    df = pd.DataFrame(rows + [dict(onset=5.0, trial_type="audio_stim")])
    bundle = ep.events_table_to_bundle(df)
    ev = bundle["agency_events"]
    assert ev["trial_type"] == [
        "self_caused",
        "other_caused",
        "self_caused",
        "other_caused",
        "self_caused",
    ]
    assert ev["yoked_to"][0] is None and ev["yoked_to"][1] == 10.0
    assert bundle["self_caused_onsets"] == [10.0, 40.0, 60.0]
    assert bundle["other_caused_onsets"] == [22.0, 52.0]
    assert ep.validate_srpi_agency_contract(ev)["valid"]
    # Legacy classification still sees self_caused/other_caused as self/non-self.
    assert bundle["self_onsets"] == [10.0, 40.0, 60.0]
    assert (
        ep.events_table_to_bundle(pd.DataFrame([dict(onset=1.0, trial_type="x")]))[
            "agency_events"
        ]
        is None
    )


def test_readiness_reports_srpi_agency(tmp_path):
    from impact_pipeline.mpc_readiness import check_mpc_readiness

    prep, bids, subj = tmp_path / "prep", tmp_path / "bids", "01"
    ts_path = prep / subj / "awake" / "audio" / f"{subj}_run-1_schaefer400_ts.npy"
    ts_path.parent.mkdir(parents=True)
    np.save(ts_path, np.random.RandomState(0).randn(200, 6))
    _, rows = _agency_task(0, n_pairs=5)
    ev = bids / f"sub-{subj}" / "func" / f"sub-{subj}_task-audioawake_run-01_events.tsv"
    ev.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(ev, sep="\t", index=False)
    df, summary = check_mpc_readiness(
        str(prep), str(bids), "schaefer400", "audio", ["awake"]
    )
    row = df.iloc[0]
    assert bool(row["SRPI_agency_ready"]) and row["SRPI_agency_reason"] == "ok"
    assert (row["n_self_caused"], row["n_other_caused"], row["n_yoked_pairs"]) == (
        5,
        5,
        5,
    )
    assert summary["metrics"]["SRPI_agency"]["ready"] == 1
    rows[1]["phase_bin"] = 99
    pd.DataFrame(rows).to_csv(ev, sep="\t", index=False)
    df, _ = check_mpc_readiness(str(prep), str(bids), "schaefer400", "audio", ["awake"])
    assert (
        df.iloc[0]["SRPI_agency_reason"]
        == "agency_contract_violation:phase_bin_mismatch"
    )
    with pytest.raises(ValueError, match=">= 3"):
        check_mpc_readiness(
            str(prep),
            str(bids),
            "schaefer400",
            "audio",
            ["awake"],
            srpi_min_events_per_class=2,
        )
