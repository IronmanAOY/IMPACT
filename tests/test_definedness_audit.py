"""Tests of scripts/definedness_audit.py on a synthetic BIDS tree: metadata-only,
read-only access with a hashed access log, and the definedness rules."""
import builtins
import hashlib
import json
import os
from pathlib import Path

import pytest

import scripts.definedness_audit as da

GARBAGE = b"\x00\x01TIME-SERIES-BYTES-NEVER-READ\xff" * 10


def _json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _tsv(path: Path, header, rows=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(header)] + ["\t".join(map(str, r)) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _data(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(GARBAGE)


@pytest.fixture()
def bids(tmp_path):
    root = tmp_path / "scratch"
    a = root / "dsA"
    _json(a / "dataset_description.json", {"Name": "A", "BIDSVersion": "1.8.0"})
    _json(a / "task-game_bold.json", {"RepetitionTime": 2.0})
    _json(a / "task-rest_bold.json", {"RepetitionTime": 2.0})
    _json(a / "task-game_events.json", {"trial_type": {"Levels": {
        "goal_cue": "goal", "stimulus": "stim", "feedback": "outcome"}}})
    _data(a / "sub-01/func/sub-01_task-game_bold.nii.gz")
    _tsv(a / "sub-01/func/sub-01_task-game_events.tsv",
         ["onset", "duration", "trial_type", "reward"],
         [(1.0, 0.5, "goal_cue", "n/a"), (2.0, 0.5, "feedback", 1)])
    _data(a / "sub-01/func/sub-01_task-rest_bold.nii.gz")
    _data(a / "sub-02/func/sub-02_task-game_bold.nii.gz")
    ev2 = a / "sub-02/func/sub-02_task-game_events.tsv"
    os.symlink(a / ".git/annex/objects/missing", ev2)  # annex content not fetched
    _json(a / "derivatives/fmriprep/dataset_description.json", {"Name": "deriv"})
    b = root / "dsB"
    _json(b / "dataset_description.json", {"Name": "B"})
    _data(b / "sub-01/eeg/sub-01_task-agency_eeg.edf")
    _json(b / "sub-01/eeg/sub-01_task-agency_eeg.json", {"SamplingFrequency": 500})
    _tsv(b / "sub-01/eeg/sub-01_task-agency_events.tsv",
         ["onset", "duration", "trial_type", "yoked_to", "phase_bin", "event_id"],
         [(1.0, 0.2, "self_caused", "n/a", 1, "s0")])
    _json(b / "sub-01/eeg/sub-01_task-agency_events.json", {"trial_type": {"Levels": {
        "self_caused": "self", "other_caused": "replay"}}})
    _data(b / "sub-02/eeg/sub-02_task-noev_eeg.set")
    _json(b / "sub-02/eeg/sub-02_task-noev_eeg.json", {"PowerLineFrequency": 50})
    return root


@pytest.fixture()
def guarded_open(monkeypatch, bids):
    """Fail the test if anything under the data root that is not metadata is
    opened, or if a data-root file is opened for writing."""
    real_open = builtins.open
    root = str(bids.resolve())
    opened = []

    def _open(file, mode="r", *args, **kwargs):
        path = os.path.realpath(os.fspath(file)) if isinstance(
            file, (str, bytes, os.PathLike)) else None
        if path and path.startswith(root):
            assert da.METADATA_OPEN_RE.search(path), f"opened non-metadata {path}"
            assert set(mode) <= {"r", "b"}, f"opened {path} with mode {mode}"
            opened.append(path)
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _open)
    return opened


def _status(rec, dataset, recording_part, principle, channel):
    sub = rec[(rec["dataset"] == dataset)
              & rec["recording"].str.contains(recording_part)
              & (rec["principle"] == principle) & (rec["channel"] == channel)]
    assert len(sub) == 1, (dataset, recording_part, principle, channel)
    return sub.iloc[0]["status"], sub.iloc[0]["reason"]


def test_audit_reads_only_metadata_and_logs_hashes(bids, guarded_open, tmp_path):
    res = da.run_audit(bids, tmp_path / "out", datasets=("dsA", "dsB", "dsC"))
    s = res["summary"]
    assert s["only_metadata_opened"] and s["time_series_opened"] is False
    assert guarded_open, "the audit should have opened metadata"
    assert not any(p.endswith((".nii.gz", ".edf", ".set")) for p in guarded_open)
    log = res["log"]
    assert not log["path"].str.contains("derivatives").any()
    ev = bids / "dsA/sub-01/func/sub-01_task-game_events.tsv"
    row = log[log["path"] == "dsA/sub-01/func/sub-01_task-game_events.tsv"].iloc[0]
    assert row["sha256"] == hashlib.sha256(ev.read_bytes()).hexdigest()
    assert row["kind"] == "events_header"
    assert row["bytes_parsed"] == len(b"onset\tduration\ttrial_type\treward")
    annex = log[log["path"] == "dsA/sub-02/func/sub-02_task-game_events.tsv"]
    assert annex.iloc[0]["status"] == "annex_not_fetched"
    assert s["n_annex_not_fetched"] == 1
    assert s["datasets"]["dsC"]["present"] is False
    for name in ("definedness_recordings.csv", "definedness_matrix.csv",
                 "definedness_access_log.csv", "definedness_summary.json"):
        assert (tmp_path / "out" / name).is_file()


def test_definedness_rules_on_synthetic_datasets(bids, tmp_path):
    rec = da.run_audit(bids, tmp_path / "out", datasets=("dsA", "dsB"))["recordings"]
    # fMRI task with described goal/stimulus/feedback levels
    assert _status(rec, "dsA", "sub-01_task-game", "RAM", "untyped")[0] == da.DEFINABLE
    assert _status(rec, "dsA", "sub-01_task-game", "PDI", "legacy_baseline")[0] == (
        da.DEFINABLE)
    assert _status(rec, "dsA", "sub-01_task-game", "RAM", "behavioural_feedback") == (
        da.NOT_DEFINABLE, "no_impact_channel_column")
    assert _status(rec, "dsA", "sub-01_task-game", "RAM", "perturbational")[0] == (
        da.NOT_IMPLEMENTED)
    assert _status(rec, "dsA", "sub-01_task-game", "SRPI", "legacy_self_other") == (
        da.NOT_DEFINABLE, "no_levels:self,nonself")
    assert _status(rec, "dsA", "sub-01_task-game", "SRPI", "agency") == (
        da.NOT_DEFINABLE, "missing_columns:yoked_to,phase_bin")
    # rest run: no events of its own task
    assert _status(rec, "dsA", "sub-01_task-rest", "RAM", "untyped") == (
        da.NOT_DEFINABLE, "no_events_file")
    assert _status(rec, "dsA", "sub-01_task-rest", "IIM", "default")[0] == da.DEFINABLE
    # annexed events not fetched: unknown, never guessed
    assert _status(rec, "dsA", "sub-02_task-game", "RAM", "untyped") == (
        da.METADATA_UNAVAILABLE, "events_not_readable")
    assert _status(rec, "dsA", "sub-02_task-game", "PDI", "legacy_baseline") == (
        da.NOT_DEFINABLE, "no_rest_recording")
    # EEG agency task: SRPI agency and legacy definable, RAM not
    assert _status(rec, "dsB", "agency", "SRPI", "agency")[0] == da.DEFINABLE
    assert _status(rec, "dsB", "agency", "SRPI", "legacy_self_other")[0] == da.DEFINABLE
    # no numeric feedback value column: strict RAM is undefined whatever the levels
    assert _status(rec, "dsB", "agency", "RAM", "untyped") == (
        da.NOT_DEFINABLE, "missing_columns:feedback_value")
    # missing sampling frequency: no timing, nothing time-series based is definable
    assert _status(rec, "dsB", "noev", "PDI", "repertoire") == (
        da.NOT_DEFINABLE, "missing_SamplingFrequency")
    assert _status(rec, "dsB", "noev", "SRPI", "agency") == (
        da.NOT_DEFINABLE, "no_events_file")


def test_undescribed_levels_require_event_values(tmp_path):
    root = tmp_path / "scratch"
    d = root / "dsX"
    _json(d / "dataset_description.json", {"Name": "X"})
    _data(d / "sub-01/func/sub-01_task-t_bold.nii")
    _json(d / "sub-01/func/sub-01_task-t_bold.json", {"RepetitionTime": 1.5})
    _tsv(d / "sub-01/func/sub-01_task-t_events.tsv",
         ["onset", "duration", "trial_type", "stim_file", "reward"])
    _data(d / "sub-02/func/sub-02_task-t_bold.nii")
    _json(d / "sub-02/func/sub-02_task-t_bold.json", {"RepetitionTime": 1.5})
    _tsv(d / "sub-02/func/sub-02_task-t_events.tsv",
         ["onset", "duration", "trial_type", "stim_file"])
    rec = da.run_audit(root, tmp_path / "o", datasets=("dsX",))["recordings"]
    assert _status(rec, "dsX", "sub-01_task-t", "RAM", "untyped") == (
        da.REQUIRES_EVENT_VALUES, "levels_not_described:trial_type")
    assert _status(rec, "dsX", "sub-01_task-t", "SRPI", "legacy_self_other") == (
        da.REQUIRES_EVENT_VALUES, "levels_not_described:trial_type,stim_file")
    # the header alone rules strict RAM out: no feedback value column
    assert _status(rec, "dsX", "sub-02_task-t", "RAM", "untyped") == (
        da.NOT_DEFINABLE, "missing_columns:feedback_value")


def test_ram_and_srpi_rules_follow_the_estimator_contracts(tmp_path):
    """Regression: the RAM rule needs feedback-labelled trial types *and* a
    numeric feedback value column (strict RAM, event_parsing._ram_fields);
    SRPI self/non-self use event_parsing.classify_self_nonself; the legacy
    PDI baseline needs a rest recording of the same modality."""
    from impact_pipeline.event_parsing import events_table_to_bundle
    import pandas as pd

    root = tmp_path / "scratch"
    d = root / "dsY"
    _json(d / "dataset_description.json", {"Name": "Y"})
    _json(d / "task-t_bold.json", {"RepetitionTime": 2.0})
    _json(d / "task-rest_eeg.json", {"SamplingFrequency": 250})
    # 1) feedback level described but no value column
    _data(d / "sub-01/func/sub-01_task-t_bold.nii.gz")
    _tsv(d / "sub-01/func/sub-01_task-t_events.tsv",
         ["onset", "duration", "trial_type"])
    _json(d / "sub-01/func/sub-01_task-t_events.json", {"trial_type": {"Levels": {
        "goal_cue": "g", "stimulus": "s", "feedback": "f",
        "non_self_name": "a", "other_name": "b"}}})
    # 2) value column but no feedback-labelled trial type
    _data(d / "sub-02/func/sub-02_task-t_bold.nii.gz")
    _tsv(d / "sub-02/func/sub-02_task-t_events.tsv",
         ["onset", "duration", "trial_type", "value"])
    _json(d / "sub-02/func/sub-02_task-t_events.json", {"trial_type": {"Levels": {
        "goal_cue": "g", "stimulus": "s", "SelfName": "a", "OtherName": "b"}}})
    # sub-02 has an EEG rest recording only (not an fMRI baseline)
    _data(d / "sub-02/eeg/sub-02_task-rest_eeg.edf")
    rec = da.run_audit(root, tmp_path / "o", datasets=("dsY",))["recordings"]
    assert _status(rec, "dsY", "sub-01_task-t", "RAM", "untyped") == (
        da.NOT_DEFINABLE, "missing_columns:feedback_value")
    assert _status(rec, "dsY", "sub-02_task-t", "RAM", "untyped") == (
        da.NOT_DEFINABLE, "no_levels:feedback")
    # "non_self_name" is not a self label: only non-self levels are present
    assert _status(rec, "dsY", "sub-01_task-t", "SRPI", "legacy_self_other") == (
        da.NOT_DEFINABLE, "no_levels:self")
    # camelCase labels are split like the estimator does
    assert _status(rec, "dsY", "sub-02_task-t", "SRPI", "legacy_self_other")[0] == (
        da.DEFINABLE)
    assert _status(rec, "dsY", "sub-02_task-t", "PDI", "legacy_baseline") == (
        da.NOT_DEFINABLE, "no_rest_recording")
    assert _status(rec, "dsY", "sub-02_task-rest", "PDI", "legacy_baseline")[0] == (
        da.DEFINABLE)
    # the estimator side agrees: without a value column there are no values
    ev = pd.DataFrame({"onset": [1.0, 2.0, 3.0], "duration": 0.1,
                       "trial_type": ["goal_cue", "stimulus", "feedback"]})
    assert events_table_to_bundle(ev)["feedback_values"] is None
    ev2 = pd.DataFrame({"onset": [1.0, 2.0], "duration": 0.1,
                        "trial_type": ["goal_cue", "stimulus"], "value": [1, 0]})
    assert events_table_to_bundle(ev2)["feedback_onsets"] == []


def test_coverage_matrix_and_prior_access(bids, tmp_path):
    res = da.run_audit(bids, tmp_path / "out", datasets=("dsA", "dsB"))
    m = res["matrix"]
    row = m[(m["dataset"] == "dsA") & (m["task"] == "game") & (m["principle"] == "IIM")]
    assert int(row["n_recordings"].iloc[0]) == 2
    assert int(row[da.DEFINABLE].iloc[0]) == 2
    assert set(da.STATUSES) <= set(m.columns)
    ds = res["summary"]["datasets"]
    assert ds["dsA"]["prior_access"] == da.DEFAULT_PRIOR_ACCESS
    assert da.PRIOR_ACCESS["ds003171"] == "exploratory_calibration"
    assert set(da.EXPLORATORY_DATASETS) == {"ds003171", "ds005620", "ds006623"}


def test_output_inside_the_data_root_is_refused(bids):
    with pytest.raises(ValueError, match="outside the data root"):
        da.run_audit(bids, bids / "dsA" / "audit_out", datasets=("dsA",))


def test_reader_refuses_non_metadata(bids):
    reader = da.MetadataReader(bids)
    with pytest.raises(PermissionError):
        reader._read(bids / "dsA/sub-01/func/sub-01_task-game_bold.nii.gz", "json")


def test_bids_name_parsing():
    ents, suf = da.parse_bids_name("sub-01_ses-2_task-rest_acq-EC_run-03_bold.nii.gz")
    assert ents == {"sub": "01", "ses": "2", "task": "rest", "acq": "EC", "run": "03"}
    assert suf == "bold"
    assert da.parse_bids_name("task-game_events.json") == ({"task": "game"}, "events")


def test_default_dataset_list_matches_the_spec():
    assert set(da.DEFAULT_DATASETS) == {
        "ds003171", "ds005620", "ds006623", "ds002547",
        "ds004295", "ds005479", "ds002685", "ds002336"}
