"""
ds003171 task handling (spec D10) and atlas naming.

sub-10JR's awake audio run is labelled task-audio. Before the fix fMRI
preprocessing silently skipped it (and every 'light' run), so run-spec
building later aborted with "No time-series for 10JR/awake".
"""

import json
from pathlib import Path

import numpy as np
import pytest

from impact_pipeline import preprocessing
from impact_pipeline.dataset_catalog import (
    canonical_task_label,
    parse_state_task,
    task_labels_for_state,
)
from impact_pipeline.run_synergy_ci import load_onsets
from impact_pipeline.synergy_ci import build_ci_run_specs


def test_ds003171_task_grammar_and_explicit_alias():
    assert parse_state_task("ds003171", "10JR", "audio") == ("awake", "audio")
    assert parse_state_task("ds003171", "sub-10JR", "audio") == ("awake", "audio")
    # the alias is subject-specific, not a guess for everyone
    assert parse_state_task("ds003171", "02CB", "audio") is None
    assert parse_state_task("ds003171", "02CB", "audiolight") == ("light", "audio")
    assert parse_state_task("ds003171", "02CB", "restrecovery") == ("recovery", "rest")
    assert parse_state_task("ds003171", "02CB", "audiodeep") == ("deep", "audio")
    assert parse_state_task("ds003171", "02CB", "motor") is None
    assert canonical_task_label("ds003171", "10JR", "audio") == "audioawake"
    assert task_labels_for_state("ds003171", "10JR", "awake", "audio") == (
        "audioawake",
        "audio",
    )
    assert task_labels_for_state("ds003171", "02CB", "awake", "audio") == (
        "audioawake",
    )


def _write_bids(root: Path):
    root.mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "ds003171-mini", "BIDSVersion": "1.8.0"})
    )
    tasks = {
        "10JR": [
            "audio",
            "audiolight",
            "audiodeep",
            "audiorecovery",
            "restawake",
            "restlight",
            "restdeep",
        ],
        "02CB": [
            "audioawake",
            "audiolight",
            "audiodeep",
            "restawake",
            "restdeep",
            "motor",
        ],
    }
    for subj, labels in tasks.items():
        func = root / f"sub-{subj}" / "func"
        func.mkdir(parents=True)
        for label in labels:
            stem = f"sub-{subj}_task-{label}_run-01"
            (func / f"{stem}_bold.nii.gz").write_bytes(b"")
            (func / f"{stem}_bold.json").write_text(json.dumps({"RepetitionTime": 2.0}))
            (func / f"{stem}_events.tsv").write_text("onset\tduration\ttrial_type\n")
    return tasks


def test_run_preprocessing_maps_10jr_and_light_consistently_with_run_specs(
    tmp_path, monkeypatch
):
    bids = tmp_path / "bids"
    prep = tmp_path / "prep"
    _write_bids(bids)
    calls = []

    def fake_preprocess_subject(
        bids_root, fmriprep_deriv, subj, bf, out_dir, assume_tr=None
    ):
        calls.append(
            (subj, bf.entities["task"], Path(out_dir).relative_to(prep).as_posix())
        )
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        run = int(bf.entities["run"])
        np.save(
            Path(out_dir) / f"{subj}_run-{run}_schaefer400_ts.npy",
            np.random.RandomState(0).rand(20, 3),
        )
        return {
            "subject": subj,
            "task": bf.entities["task"],
            "run": run,
            "tr_source": "sidecar",
        }

    monkeypatch.setattr(preprocessing, "preprocess_subject", fake_preprocess_subject)
    summary = preprocessing.run_preprocessing(
        str(bids), str(tmp_path / "fmriprep"), str(prep)
    )

    assert ("10JR", "audio", "10JR/awake/audio") in calls
    assert ("10JR", "audiolight", "10JR/light/audio") in calls
    assert (
        "02CB",
        "restlight",
        "02CB/light/rest",
    ) not in calls  # 02CB has no restlight
    assert ("10JR", "restlight", "10JR/light/rest") in calls
    skipped = {(r["subject"], r["task"], r["reason"]) for r in summary["skipped_runs"]}
    assert ("02CB", "motor", "task_not_in_dataset_grammar") in skipped
    assert summary["summary"]["written_runs"] == len(calls)

    # Run-spec building (step 2) now finds 10JR/awake instead of raising.
    specs = build_ci_run_specs(str(prep), "schaefer400", ("awake", "deep"), "audio")
    assert {(s["subject"], s["session"]) for s in specs} == {
        ("10JR", "awake"),
        ("10JR", "deep"),
        ("02CB", "awake"),
        ("02CB", "deep"),
    }
    # The events resolver maps sub-10JR awake to its task-audio events (run 1),
    # so event-based run selection agrees with the preprocessed tree.
    bundle, run_id = load_onsets(str(bids), "10JR", "awake", condition="audio")
    assert run_id == "1"
    onsets = {"10JR": {"awake": (bundle, run_id)}}
    specs = build_ci_run_specs(
        str(prep),
        "schaefer400",
        ("awake",),
        "audio",
        stimulus_onsets=onsets,
        subjects=["10JR"],
    )
    assert [Path(s["ts_path"]).name for s in specs] == ["10JR_run-1_schaefer400_ts.npy"]


def test_subject_filter_accepts_sub_prefix(tmp_path, monkeypatch):
    bids = tmp_path / "bids"
    _write_bids(bids)
    seen = set()
    monkeypatch.setattr(
        preprocessing,
        "preprocess_subject",
        lambda b, f, subj, bf, out_dir, assume_tr=None: seen.add(subj)
        or {"tr_source": "sidecar"},
    )
    preprocessing.run_preprocessing(
        str(bids),
        str(tmp_path / "fmriprep"),
        str(tmp_path / "prep"),
        subjects=["sub-10JR"],
    )
    assert seen == {"10JR"}


def test_aal_atlas_is_named_aal116_with_legacy_alias():
    globs = preprocessing.get_atlas_globs()
    assert "aal116" in globs and "aal90" not in globs
    assert preprocessing.find_atlas("aal90") == globs["aal116"]
    import nibabel as nib

    labels = np.unique(np.asarray(nib.load(globs["aal116"]).dataobj))
    assert int((labels != 0).sum()) == 116


def test_atlas_root_does_not_depend_on_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(preprocessing, "ATLAS_GLOBS", None)
    globs = preprocessing.get_atlas_globs()
    assert Path(globs["schaefer400"]).exists()
    with pytest.raises(FileNotFoundError, match="IMPACT_ATLAS_DIR"):
        preprocessing.get_atlas_globs(atlas_root=tmp_path / "nowhere")
