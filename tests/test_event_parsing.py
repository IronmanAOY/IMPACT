"""Event layer used by RAM/SRPI: parsing contract, file resolution, parity."""
import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from impact_pipeline import event_parsing as ep  # noqa: E402
from impact_pipeline import mpc_readiness as rd  # noqa: E402
from impact_pipeline import run_synergy_ci as rs  # noqa: E402


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def test_self_nonself_patterns_have_a_single_source():
    assert rd._SELF_RE is ep.SELF_RE and rs._SELF_RE is ep.SELF_RE
    assert rd._NONSELF_RE is ep.NONSELF_RE and rs._NONSELF_RE is ep.NONSELF_RE


@pytest.mark.parametrize(
    "label,expected",
    [
        ("self", "self"),
        ("self_name", "self"),
        ("my_name", "self"),
        ("own-name", "self"),
        ("SelfName", "self"),
        ("nonself", "nonself"),
        ("non-self", "nonself"),
        ("other_name", "nonself"),
        ("stranger", "nonself"),
        ("self_other", None),  # ambiguous: neither class
        ("name", None),
        ("time", None),
        ("home", None),
        ("unknown", None),
        ("down", None),
        ("mother", None),
    ],
)
def test_self_nonself_classification_uses_whole_tokens(label, expected):
    s, n = ep.classify_self_nonself(pd.Series([label]))
    got = "self" if bool(s.iloc[0]) else ("nonself" if bool(n.iloc[0]) else None)
    assert got == expected


def test_readiness_and_compute_parse_the_same_events_identically(tmp_path):
    fn = _write(
        tmp_path / "ev.tsv",
        [
            {"onset": 1.0, "duration": 0.1, "trial_type": "self_name"},
            {"onset": 2.0, "duration": 0.1, "trial_type": "other_name"},
            {"onset": 3.0, "duration": 0.1, "trial_type": "unknown_word"},
            {"onset": 4.0, "duration": 0.1, "trial_type": "goal_cue"},
            {"onset": 5.0, "duration": 0.1, "trial_type": "audio_stim"},
            {"onset": 6.0, "duration": 0.1, "trial_type": "feedback", "reward": 1.0},
            {"onset": 7.0, "duration": 0.1, "trial_type": "feedback", "reward": 0.0},
        ],
    )
    compute = rs._events_to_ram_bundle(fn)
    df = rd._read_events_table(fn)
    ready_self, ready_non = rd._events_to_srpi_onsets(df)
    ready_ram = rd._events_to_ram_bundle(df)
    assert compute["self_onsets"] == ready_self == [1.0]
    assert compute["nonself_onsets"] == ready_non == [2.0]
    for key in ("onsets", "goal_onsets", "feedback_onsets"):
        assert compute[key] == ready_ram[key]
    assert compute["feedback_values"] == ready_ram["feedback_values"] == [1.0, 0.0]


def test_parser_has_no_implicit_stimuli_or_response_time_feedback():
    df = pd.DataFrame(
        {"onset": [0.0], "duration": [0.0], "trial_type": ["New Segment/"]}
    )
    assert ep.events_table_to_bundle(df)["onsets"] == []
    assert ep.events_table_to_bundle(df, allow_implicit_stimuli=True)["onsets"] == [0.0]

    df = pd.DataFrame(
        {
            "onset": [1.0, 2.0, 3.0, 4.0, 5.0],
            "duration": 0.0,
            "trial_type": [
                "audio_stim", "response", "feedback", "response", "feedback"
            ],
            "response_time": [np.nan, 0.5, 0.4, 0.7, 0.9],
        }
    )
    b = ep.events_table_to_bundle(df)
    # 'response' rows are actions, not feedback; response_time is no feedback value.
    assert b["feedback_onsets"] == [3.0, 5.0]
    assert b["feedback_values"] is None
    b = ep.events_table_to_bundle(df, allow_response_time_feedback=True)
    assert b["feedback_onsets"] == [3.0, 5.0]
    assert b["feedback_values"] == [0.4, 0.9]


def test_feedback_values_stay_aligned_with_their_events():
    df = pd.DataFrame(
        {
            "onset": [0.5, 1.0, 2.0, 3.0, 4.0],
            "duration": 0.0,
            "trial_type": [
                "audio_stim", "feedback", "feedback", "feedback", "feedback"
            ],
            "reward": [7.0, 1.0, np.nan, 3.0, 4.0],
        }
    )
    b = ep.events_table_to_bundle(df)
    # Old parser: onsets [1, 2, 3, 4] with values [1, 3, 4] (shifted pairing).
    assert b["feedback_onsets"] == [1.0, 3.0, 4.0]
    assert b["feedback_values"] == [1.0, 3.0, 4.0]


def test_goal_pattern_excludes_self_and_name_events():
    df = pd.DataFrame(
        {"onset": [1.0, 2.0, 3.0, 4.0], "duration": 0.0,
         "trial_type": ["goal_cue", "self", "nonself", "own_name"]}
    )
    assert ep.events_table_to_bundle(df)["goal_onsets"] == [1.0]


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("onset\tduration\n")
    return path


def test_eeg_sed_and_sed2_resolve_to_their_own_files(tmp_path):
    eeg = tmp_path / "sub-1010" / "eeg"
    ec = _touch(eeg / "sub-1010_task-awake_acq-EC_events.tsv")
    sed2 = _touch(eeg / "sub-1010_task-sed2_acq-rest_run-1_events.tsv")
    sed = _touch(eeg / "sub-1010_task-sed_acq-rest_run-1_events.tsv")
    assert ep.resolve_events_file(tmp_path, "1010", "awake") == ec
    assert ep.resolve_events_file(tmp_path, "1010", "sed") == sed
    assert ep.resolve_events_file(tmp_path, "1010", "sed2") == sed2
    # 'deep' follows the EEG preprocessing rule: sed2/rest, else sed/rest.
    assert ep.resolve_events_file(tmp_path, "1010", "deep") == sed2
    sed2.unlink()
    assert ep.resolve_events_file(tmp_path, "1010", "deep") == sed


def test_fmri_resolution_has_no_cross_session_fallback(tmp_path):
    func = tmp_path / "sub-01" / "func"
    awake = _touch(func / "sub-01_task-audioawake_run-01_events.tsv")
    _touch(func / "sub-01_task-audio_run-01_events.tsv")  # ds003171 sub-10JR style
    assert ep.resolve_events_file(tmp_path, "01", "awake") == awake
    # Old code fell back to task-audio* and returned another session's file.
    assert ep.resolve_events_file(tmp_path, "01", "deep") is None
    assert rs._resolve_events_file(tmp_path, "01", "deep") is None


def test_bids_session_folders_are_searched(tmp_path):
    func = tmp_path / "sub-02" / "ses-1" / "func"
    f = _touch(func / "sub-02_ses-1_task-self_run-1_events.tsv")
    assert ep.resolve_events_file(tmp_path, "02", "ses-1", condition="self") == f
    assert ep.resolve_events_file(tmp_path, "02", "1", condition="self") == f
    assert ep.resolve_events_file(tmp_path, "02", "ses-2", condition="self") is None


def test_load_onsets_returns_strict_bundle_and_run_id(tmp_path):
    fn = _write(
        tmp_path / "sub-03" / "func" / "sub-03_task-audioawake_run-02_events.tsv",
        [{"onset": 0.0, "duration": 0.0, "trial_type": "New Segment/"}],
    )
    bundle, run_id = rs.load_onsets(tmp_path, "03", "awake")
    assert run_id == "2" and bundle["onsets"] == []
    bundle, _ = rs.load_onsets(tmp_path, "03", "awake", allow_implicit_stimuli=True)
    assert bundle["onsets"] == [0.0]
    assert fn.exists()


def test_readiness_ram_contract_matches_compute():
    ok_bundle = {
        "onsets": list(np.arange(6.0)),
        "goal_onsets": [0.0],
        "feedback_onsets": [1.0, 2.0, 3.0],
        "feedback_values": [1.0, 0.0, 2.0],
    }
    assert rd._assess_ram(ok_bundle, require_explicit_feedback=True)[:2] == (True, "ok")
    no_goal = dict(ok_bundle, goal_onsets=[])
    assert rd._assess_ram(no_goal, True)[1] == "missing_goal_events"
    assert rd._assess_ram(no_goal, True, require_explicit_goals=False)[0]
    few = dict(ok_bundle, onsets=[0.0, 1.0])
    assert rd._assess_ram(few, True)[1] == "insufficient_stimulus_events"
    from impact_pipeline import mpc_metrics as mm

    assert ep.RAM_MIN_GOAL_RESPONSE_PAIRS == mm._RAM_MIN_GOAL_PAIRS
    assert ep.RAM_MIN_FEEDBACK_EVENTS == mm._RAM_MIN_FEEDBACK_EVENTS


def _fake_step2(captured):
    def fake(prep_out, atlas, thetas, sessions, **kw):
        captured.append(kw)
        rows = [
            {"subject": "s1", "session": ses, "theta": float(t), "S": 0.1}
            for t in thetas
            for ses in sessions
        ]
        return pd.DataFrame(rows)

    return fake


def _run(tmp_path, monkeypatch, **kw):
    captured, loads = [], []
    monkeypatch.setattr(rs, "compute_synergy_ci", _fake_step2(captured))
    prep = tmp_path / "prep"
    (prep / "s1").mkdir(parents=True, exist_ok=True)

    def fake_load(bids_root, subj, ses, condition="audio", **opts):
        loads.append(opts)
        return ep.empty_event_bundle(), None

    rs.run_s_ci(
        prep_out=prep,
        bids_root=tmp_path / "no_bids",
        figdir=tmp_path,
        atlas="eeg64",
        sessions=("awake", "deep"),
        thetas=[0.5],
        thetas_fine=[0.5],
        tr=0.004,
        load_onsets_fn=fake_load,
        **kw,
    )
    return captured, loads


def test_run_s_ci_uses_modality_ram_preset_and_forwards_params(tmp_path, monkeypatch):
    captured, loads = _run(tmp_path, monkeypatch, modality="eeg")
    ram = captured[0]["ram_params"]
    assert ram["response_model"] == "boxcar" and ram["latency_method"] == "fir"
    assert ram == rs.RAM_PARAM_PRESETS["eeg"]
    assert loads and all(opts == {} for opts in loads)

    explicit = dict(rs.RAM_PARAM_PRESETS["fmri"], quality_null_samples=7)
    captured, loads = _run(
        tmp_path, monkeypatch, modality="eeg", ram_params=explicit,
        event_options={"allow_implicit_stimuli": True},
    )
    assert captured[0]["ram_params"] == explicit
    assert all(opts == {"allow_implicit_stimuli": True} for opts in loads)


def test_run_s_ci_requires_modality_or_ram_params_for_ram(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="modality-specific"):
        _run(tmp_path, monkeypatch, modality=None)
    # RAM not requested: no RAM parameters needed.
    captured, _ = _run(
        tmp_path, monkeypatch, modality=None, mpc_metrics=("PDI",), compute_ci=False
    )
    assert captured[0]["ram_params"] is None


def test_ram_presets_are_accepted_by_compute_ram():
    from impact_pipeline import mpc_metrics as mm
    from impact_pipeline.synergy_ci import _resolve_ram_kwargs

    ts = np.random.RandomState(0).randn(4, 400)
    for preset in rs.RAM_PARAM_PRESETS.values():
        kw = _resolve_ram_kwargs(preset)
        assert set(kw) == set(preset)
        out = mm.compute_RAM(ts, tr=0.01, stimulus_onsets=[], **kw)
        assert np.isnan(out)
