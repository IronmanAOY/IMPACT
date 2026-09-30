"""
ds003171 task handling and atlas naming.

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


REPO = Path(__file__).resolve().parents[1]


def test_atlas_root_is_the_repository_or_impact_atlas_dir(tmp_path, monkeypatch):
    monkeypatch.delenv(preprocessing.ATLAS_DIR_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    assert preprocessing._repository_root() == REPO
    assert preprocessing._default_atlas_root() == REPO / "atlases"
    # IMPACT_ATLAS_DIR wins, also after the repository atlases were cached.
    monkeypatch.setattr(preprocessing, "ATLAS_GLOBS", None)
    preprocessing.get_atlas_globs()
    alt = tmp_path / "alt_atlases"
    for key, parts in preprocessing.ATLAS_FILES.items():
        dest = alt.joinpath(*parts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(REPO.joinpath("atlases", *parts))
    monkeypatch.setenv(preprocessing.ATLAS_DIR_ENV, str(alt))
    globs = preprocessing.get_atlas_globs()
    assert set(globs) == {"schaefer400", "aal116", "shen268"}
    assert all(str(Path(p)).startswith(str(REPO / "atlases")) for p in globs.values())
    assert preprocessing._default_atlas_root() == alt
    monkeypatch.setattr(preprocessing, "ATLAS_GLOBS", None)


def test_missing_atlas_error_lists_every_expected_file(tmp_path, monkeypatch):
    partial = tmp_path / "atlases"
    shen = partial.joinpath(*preprocessing.ATLAS_FILES["shen268"])
    shen.parent.mkdir(parents=True)
    shen.write_bytes(b"x")
    monkeypatch.setenv(preprocessing.ATLAS_DIR_ENV, str(partial))
    monkeypatch.setattr(preprocessing, "ATLAS_GLOBS", None)
    with pytest.raises(FileNotFoundError) as err:
        preprocessing.get_atlas_globs()
    msg = str(err.value)
    assert f"${preprocessing.ATLAS_DIR_ENV}" in msg and str(partial) in msg
    assert "[found] shen268" in msg
    for key in ("schaefer400", "aal116"):
        expected = partial.joinpath(*preprocessing.ATLAS_FILES[key])
        assert f"[missing] {key}: {expected}" in msg
    assert "scripts/download_atlases.sh" in msg
    monkeypatch.setattr(preprocessing, "ATLAS_GLOBS", None)


class _Img:
    def __init__(self, n_time):
        self.header = _Header()
        self.shape = (2, 2, 2, n_time)
        self.affine = np.eye(4)

    def get_fdata(self):
        return np.random.default_rng(self.shape[-1]).normal(size=self.shape)


class _Header:
    def get_zooms(self):
        return (3.0, 3.0, 3.0, 2.0)

    def get_xyzt_units(self):
        return ("mm", "sec")

    def copy(self):
        return self


def test_mean_fd_is_written_per_run_not_overwritten_per_folder(tmp_path, monkeypatch):
    """Two runs of one condition folder: each keeps its own FD."""
    from impact_pipeline.motion_model import _weighted_session_fd

    deriv = tmp_path / "fmriprep"
    func = deriv / "sub-01" / "func"
    func.mkdir(parents=True)
    n_time = {1: 30, 2: 10}
    fd_values = {1: [0.0, 0.2, 0.4], 2: [0.0, 0.9, 1.2]}  # means 0.2 and 0.7
    for run in (1, 2):
        stem = f"sub-01_task-audioawake_run-{run}"
        (func / f"{stem}_desc-preproc_bold.nii.gz").write_bytes(b"")
        (func / f"{stem}_desc-confounds_timeseries.tsv").write_text(
            "framewise_displacement\ta_comp_cor_00\n"
            + "".join(f"{v}\t{0.1 * i}\n" for i, v in enumerate(fd_values[run]))
        )
    current = {}

    def load_img(path):
        run = 1 if "run-1" in str(path) else 2
        current["n"] = n_time[run]
        return _Img(n_time[run])

    class _Masker:
        def fit_transform(self, img):
            return np.zeros((current["n"], 3))

    monkeypatch.setattr(preprocessing.image, "load_img", load_img)
    monkeypatch.setattr(preprocessing, "clean", lambda signals, **k: signals.T)
    monkeypatch.setattr(preprocessing, "NiftiLabelsMasker", lambda **k: _Masker())
    monkeypatch.setattr(
        preprocessing, "get_atlas_globs", lambda: {"schaefer400": "dummy.nii.gz"}
    )
    out = tmp_path / "prep" / "01" / "awake" / "audio"
    records = []
    for run in (1, 2):
        bf = type("BF", (), {"entities": {"task": "audioawake", "run": run}})()
        records.append(
            preprocessing.preprocess_subject(
                str(tmp_path / "bids"), str(deriv), "01", bf, str(out)
            )
        )
    assert [r["mean_fd"] for r in records] == pytest.approx([0.2, 0.7])
    assert float((out / "01_run-1_mean_fd.txt").read_text()) == pytest.approx(0.2)
    assert float((out / "01_run-2_mean_fd.txt").read_text()) == pytest.approx(0.7)
    # The folder file is the mean over runs, not the last run's value.
    assert float((out / "mean_fd.txt").read_text()) == pytest.approx(0.45)
    # The motion model weights each run's FD by that run's timepoints.
    fd = _weighted_session_fd(
        str(tmp_path / "prep"), "01", "awake", "schaefer400", "audio"
    )
    assert fd == pytest.approx((0.2 * 30 + 0.7 * 10) / 40)
