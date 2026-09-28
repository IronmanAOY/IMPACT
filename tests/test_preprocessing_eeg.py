"""
EEG preprocessing (spec D10): non-EEG channels are excluded by type, BIDS run
numbers are kept, and rest/baseline segments are written when the dataset has
them so the strict PDI anchor can be defined (otherwise it stays undefined
with a reason).

ds005620 declares VEOG, HEOG and EMG as type EEG in both the BrainVision
header and channels.tsv; before the fix they were kept as network nodes.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

import run_pipeline
from impact_pipeline.preprocessing_eeg import (
    classify_eeg_channels,
    run_preprocessing_eeg,
)

SFREQ = 250.0
SCALP = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T7",
    "T8",
    "Fz",
    "Cz",
    "Pz",
]
NON_EEG = ["VEOG", "HEOG", "EMG", "ECG"]


def _write_brainvision(eeg_dir: Path, stem: str, channels, n_samples, seed):
    rng = np.random.RandomState(seed)
    data = rng.standard_normal((n_samples, len(channels))).astype(np.float32)
    (eeg_dir / f"{stem}_eeg.eeg").write_bytes(data.tobytes())
    chan_lines = "\n".join(f"Ch{i + 1}={name},,1,µV" for i, name in enumerate(channels))
    (eeg_dir / f"{stem}_eeg.vhdr").write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n\n"
        "[Common Infos]\nCodepage=UTF-8\n"
        f"DataFile={stem}_eeg.eeg\nMarkerFile={stem}_eeg.vmrk\n"
        "DataFormat=BINARY\nDataOrientation=MULTIPLEXED\n"
        f"NumberOfChannels={len(channels)}\nSamplingInterval={int(1e6 / SFREQ)}\n\n"
        "[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n\n"
        f"[Channel Infos]\n{chan_lines}\n",
        encoding="utf-8",
    )
    (eeg_dir / f"{stem}_eeg.vmrk").write_text(
        "Brain Vision Data Exchange Marker File Version 1.0\n\n"
        f"[Common Infos]\nCodepage=UTF-8\nDataFile={stem}_eeg.eeg\n\n"
        "[Marker Infos]\nMk1=New Segment,,1,1,0\n",
        encoding="utf-8",
    )
    # ds005620 style: everything typed EEG except ECG; T7 flagged bad.
    rows = ["name\ttype\tunits\tstatus"]
    for name in channels:
        ctype = "ECG" if name == "ECG" else "EEG"
        status = "bad" if name == "T7" else "good"
        rows.append(f"{name}\t{ctype}\tµV\t{status}")
    (eeg_dir / f"{stem}_channels.tsv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )


def _make_dataset(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "eeg-mini", "BIDSVersion": "1.8.0"})
    )
    channels = SCALP + NON_EEG
    n = int(SFREQ * 12)
    specs = {
        # subject with full rest data
        "1010": [
            "task-awake_acq-EC",
            "task-awake_acq-EO",
            "task-sed2_acq-rest_run-1",
            "task-sed2_acq-rest_run-2",
            "task-sed2_acq-rest_run-3",
        ],
        # a single sed2 run: no deep baseline disjoint from the analysed run
        "1016": [
            "task-awake_acq-EC",
            "task-awake_acq-EO",
            "task-sed2_acq-rest_run-1",
            "task-sed_acq-rest_run-1",
            "task-sed_acq-rest_run-2",
        ],
    }
    for k, (subj, stems) in enumerate(specs.items()):
        eeg_dir = root / f"sub-{subj}" / "eeg"
        eeg_dir.mkdir(parents=True)
        for j, stem in enumerate(stems):
            _write_brainvision(
                eeg_dir, f"sub-{subj}_{stem}", channels, n, seed=10 * k + j
            )


def test_classify_eeg_channels_by_type_name_and_status():
    names = ["Fp1", "VEOG", "HEOG", "EMG", "ECG", "Cz", "Resp", "T7"]
    mne_types = ["eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "misc", "eeg"]
    bids = {
        "ECG": {"type": "ECG", "status": "good"},
        "T7": {"type": "EEG", "status": "bad"},
        "Cz": {"type": "EEG", "status": "good"},
    }
    cls = classify_eeg_channels(names, mne_types, bids)
    assert cls["eeg"] == ["Fp1", "Cz"]
    excluded = {rec["channel"]: rec for rec in cls["excluded"]}
    assert (
        excluded["VEOG"]["type"] == "eog"
        and excluded["VEOG"]["source"] == "name_pattern"
    )
    assert excluded["EMG"]["type"] == "emg"
    assert (
        excluded["ECG"]["type"] == "ecg" and excluded["ECG"]["source"] == "channels.tsv"
    )
    assert excluded["Resp"]["type"] == "misc"
    assert excluded["T7"]["type"] == "bad"


@pytest.fixture(scope="module")
def eeg_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("eeg")
    bids = root / "bids"
    out = root / "prep"
    _make_dataset(bids)
    cfg = run_pipeline.DATASET_CONFIGS["ds005620"]
    summary = run_preprocessing_eeg(
        bids_root=str(bids),
        out_root=str(out),
        session_rules=cfg["eeg_session_rules"],
        rest_rules=cfg["eeg_rest_rules"],
        condition_label="eeg",
        atlas_key="eeg64",
        target_sfreq=SFREQ,
        max_duration_sec=120.0,
    )
    return summary, out


def test_non_eeg_channels_are_not_network_nodes(eeg_run):
    summary, out = eeg_run
    excluded = {(r["subject"], r["channel"]) for r in summary["excluded_channels"]}
    for subj in ("1010", "1016"):
        for ch in NON_EEG + ["T7"]:
            assert (subj, ch) in excluded
    arr = np.load(out / "1010" / "awake" / "eeg" / "1010_run-1_eeg64_ts.npy")
    assert arr.shape[1] == len(SCALP) - 1  # 17 scalp channels minus bad T7
    assert {r["n_channels"] for r in summary["written_runs"]} == {len(SCALP) - 1}


def test_analysis_runs_keep_bids_run_ids_and_rest_is_disjoint(eeg_run):
    summary, out = eeg_run
    subj = out / "1010"
    assert sorted(p.name for p in (subj / "awake" / "eeg").glob("*.npy")) == [
        "1010_run-1_eeg64_ts.npy"
    ]
    assert sorted(p.name for p in (subj / "deep" / "eeg").glob("*.npy")) == [
        "1010_run-1_eeg64_ts.npy"
    ]
    assert sorted(p.name for p in (subj / "awake" / "rest").glob("*.npy")) == [
        "1010_run-1_eeg64_ts.npy"
    ]
    assert sorted(p.name for p in (subj / "deep" / "rest").glob("*.npy")) == [
        "1010_run-2_eeg64_ts.npy",
        "1010_run-3_eeg64_ts.npy",
    ]
    rows = {
        (r["session"], r["segment"], r["bids_run"]): r
        for r in summary["written_runs"]
        if r["subject"] == "1010"
    }
    assert "acq-EO" in rows[("awake", "rest", None)]["source_file"]
    assert "sed2_acq-rest_run-1" in rows[("deep", "analysis", 1)]["source_file"]
    sources = [
        r["source_file"] for r in summary["written_runs"] if r["subject"] == "1010"
    ]
    assert len(sources) == len(set(sources))  # no recording is used twice


def test_rest_baseline_comes_from_the_same_state_only(eeg_run):
    summary, out = eeg_run
    # sub-1016 has one sed2 run: lighter 'sed' runs are NOT used as deep baseline.
    assert not (out / "1016" / "deep" / "rest").exists()
    assert (out / "1016" / "awake" / "rest" / "1016_run-1_eeg64_ts.npy").exists()
    missing = [r for r in summary["missing_rest"] if r["subject"] == "1016"]
    assert missing and missing[0]["session"] == "deep"
    assert missing[0]["reason"] == "no_rest_recording_for_state"


def test_strict_pdi_anchor_defined_only_with_rest_data(eeg_run):
    from impact_pipeline.synergy_ci import compute_synergy_ci

    _summary, out = eeg_run
    cfg = run_pipeline.DATASET_CONFIGS["ds005620"]
    df = compute_synergy_ci(
        str(out),
        "eeg64",
        thetas=[0.5],
        sessions=("awake", "deep"),
        condition="eeg",
        tr=1.0 / SFREQ,
        mpc_metrics=("PDI",),
        compute_ci=False,
        pdi_params=cfg["pdi_params"],
        pdi_require_explicit_params=True,
        pdi_require_strict_baseline=True,
        pdi_primary_endpoint="anchor",
    )
    df["subject"] = df["subject"].astype(str)
    with_rest = df[df["subject"] == "1010"]
    without = df[df["subject"] == "1016"]
    assert np.isfinite(with_rest["PDI_anchor"]).all()
    assert not np.isfinite(without["PDI_anchor"]).any()
    assert without["PDI_anchor_reason"].astype(str).str.contains("baseline").all()


# -- ds005620-like sidecars (as on OpenNeuro, checked against the local copy) --
#
# - channels.tsv starts with a UTF-8 BOM and has the 9 columns BIDS-validator
#   writes; VEOG, HEOG and EMG are typed EEG, all 'good'; some subjects have no
#   EMG channel.
# - *_eeg.json declares EEGChannelCount = all channels and EOG/EMG/ECG = 0.
# - pybv-written BrainVision headers (comment lines, Codepage, 0.1 uV
#   resolution) at 5000 Hz (here 500 Hz to stay small), resampled to 250 Hz.
# - git-annex layout: .vhdr/.eeg/.vmrk are symlinks into .git/annex/objects,
#   and un-fetched content is a broken symlink.
# - awake has acq-EC, acq-EO and acq-tms (TMS-EEG, never analysed).

DS_SFREQ = 500.0
DS_SCALP = SCALP  # 17 scalp channels
DS_COLUMNS = (
    "name\ttype\tunits\tlow_cutoff\thigh_cutoff\tdescription\t"
    "sampling_frequency\tstatus\tstatus_description"
)


def _annexed(eeg_dir: Path, name: str, payload: bytes, fetched: bool = True):
    obj = eeg_dir.parents[1] / ".git" / "annex" / "objects" / name[:2] / name / name
    obj.parent.mkdir(parents=True, exist_ok=True)
    if fetched:
        obj.write_bytes(payload)
    (eeg_dir / name).symlink_to(os.path.relpath(obj, eeg_dir))


def _write_ds005620_recording(
    eeg_dir: Path, stem: str, channels, seconds, seed, fetched=True
):
    n = int(DS_SFREQ * seconds)
    rng = np.random.default_rng(seed)
    data = (rng.standard_normal((n, len(channels))) * 20.0).astype("<f4")
    # Stored in units of the 0.1 uV resolution, as pybv writes them.
    _annexed(eeg_dir, f"{stem}_eeg.eeg", (data / 0.1).astype("<f4").tobytes(), fetched)
    chans = "\n".join(f"Ch{i + 1}={nm},,0.1,µV" for i, nm in enumerate(channels))
    vhdr = (
        "Brain Vision Data Exchange Header File Version 1.0\n"
        "; Written using pybv 0.7.5\n\n[Common Infos]\nCodepage=UTF-8\n"
        f"DataFile={stem}_eeg.eeg\nMarkerFile={stem}_eeg.vmrk\nDataFormat=BINARY\n"
        "; Data orientation: MULTIPLEXED=ch1,pt1, ch2,pt1 ...\n"
        f"DataOrientation=MULTIPLEXED\nNumberOfChannels={len(channels)}\n"
        "; Sampling interval in microseconds\n"
        f"SamplingInterval={1e6 / DS_SFREQ:.1f}\n\n[Binary Infos]\n"
        "BinaryFormat=IEEE_FLOAT_32\n\n[Channel Infos]\n"
        "; Each entry: Ch<Channel number>=<Name>,<Reference channel name>,\n"
        f"{chans}\n\n[Comment]\n"
    )
    _annexed(eeg_dir, f"{stem}_eeg.vhdr", vhdr.encode("utf-8"))
    vmrk = (
        "Brain Vision Data Exchange Marker File, Version 1.0\n"
        "; Exported using pybv 0.7.5\n\n[Common Infos]\nCodepage=UTF-8\n"
        f"DataFile={stem}_eeg.eeg\n\n[Marker Infos]\n"
        "Mk1=New Segment,,1,1,0\n"
    )
    _annexed(eeg_dir, f"{stem}_eeg.vmrk", vmrk.encode("utf-8"))
    rows = [
        f"{nm}\tEEG\tµV\t0.0\t2500.0\tElectroEncephaloGram\t{DS_SFREQ}\tgood\tn/a"
        for nm in channels
    ]
    (eeg_dir / f"{stem}_channels.tsv").write_bytes(
        ("﻿" + DS_COLUMNS + "\n" + "\n".join(rows) + "\n").encode("utf-8")
    )
    task = stem.split("_task-")[1].split("_")[0]
    (eeg_dir / f"{stem}_eeg.json").write_text(
        json.dumps(
            {
                "TaskName": task,
                "Manufacturer": "Brain Products",
                "SamplingFrequency": DS_SFREQ,
                "EEGChannelCount": len(channels),
                "EOGChannelCount": 0,
                "ECGChannelCount": 0,
                "EMGChannelCount": 0,
                "MiscChannelCount": 0,
            }
        )
    )
    (eeg_dir / f"{stem}_events.tsv").write_bytes(
        "﻿onset\tduration\ttrial_type\tvalue\tsample\n0.0\t0.0002\tNew Segment/\t1\t0\n"
        .encode("utf-8")
    )


def _make_ds005620_like(root: Path):
    root.mkdir(parents=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "ds005620-like", "BIDSVersion": "1.8.0"})
    )
    stems = [
        "task-awake_acq-EC",
        "task-awake_acq-EO",
        "task-awake_acq-tms",
        "task-sed2_acq-rest_run-1",
        "task-sed2_acq-rest_run-2",
        "task-sed2_acq-rest_run-3",
        "task-sed_acq-rest_run-1",
        "task-sed_acq-rest_run-2",
    ]
    subjects = {
        "1010": DS_SCALP + ["VEOG", "HEOG", "EMG"],  # 65-channel montage
        "1074": DS_SCALP + ["VEOG", "HEOG"],  # 64 channels, no EMG
        "1099": DS_SCALP + ["VEOG", "HEOG", "EMG"],  # annex content not fetched
    }
    for k, (subj, channels) in enumerate(subjects.items()):
        eeg_dir = root / f"sub-{subj}" / "eeg"
        eeg_dir.mkdir(parents=True)
        for j, stem in enumerate(stems):
            fetched = not (subj == "1099" and stem == "task-sed2_acq-rest_run-1")
            _write_ds005620_recording(
                eeg_dir,
                f"sub-{subj}_{stem}",
                channels,
                seconds=6.0 if "tms" not in stem else 3.0,
                seed=100 * k + j,
                fetched=fetched,
            )
    return subjects


@pytest.fixture(scope="module")
def ds005620_like(tmp_path_factory):
    root = tmp_path_factory.mktemp("ds005620_like")
    bids = root / "bids"
    out = root / "prep"
    subjects = _make_ds005620_like(bids)
    cfg = run_pipeline.DATASET_CONFIGS["ds005620"]
    summary = run_preprocessing_eeg(
        bids_root=str(bids),
        out_root=str(out),
        session_rules=cfg["eeg_session_rules"],
        rest_rules=cfg["eeg_rest_rules"],
        condition_label="eeg",
        atlas_key="eeg64",
        target_sfreq=250.0,
        max_duration_sec=120.0,
    )
    return summary, out, subjects


def test_ds005620_sidecars_bom_and_eeg_typed_ocular_channels(ds005620_like):
    from impact_pipeline.preprocessing_eeg import read_bids_channels

    summary, out, subjects = ds005620_like
    # The BOM does not corrupt the first column name.
    vhdr = next((out.parent / "bids" / "sub-1010" / "eeg").glob("*acq-EC_eeg.vhdr"))
    chans = read_bids_channels(str(vhdr))
    assert "Fp1" in chans and chans["VEOG"]["type"] == "EEG"
    excluded = {
        (r["subject"], r["channel"]): r["source"] for r in summary["excluded_channels"]
    }
    for subj in ("1010", "1074"):
        for ch in ("VEOG", "HEOG"):
            assert excluded[(subj, ch)] == "name_pattern"
    assert excluded[("1010", "EMG")] == "name_pattern"
    assert ("1074", "EMG") not in excluded  # montage without EMG
    for rec in summary["written_runs"]:
        assert rec["n_channels"] == len(DS_SCALP)
        assert rec["sfreq_hz"] == 250.0
        arr = np.load(rec["output_file"])
        assert arr.shape == (rec["n_timepoints"], len(DS_SCALP))
        assert np.isfinite(arr).all()
        assert np.allclose(arr.mean(axis=0), 0.0, atol=1e-6)
        assert np.allclose(arr.std(axis=0), 1.0, atol=1e-3)


def test_ds005620_rest_writing_follows_the_state_rules(ds005620_like):
    summary, out, _subjects = ds005620_like
    for subj in ("1010", "1074"):
        rows = [r for r in summary["written_runs"] if r["subject"] == subj]
        by = {(r["session"], r["segment"]): [] for r in rows}
        for r in rows:
            by[(r["session"], r["segment"])].append(Path(r["source_file"]).name)
        assert by[("awake", "analysis")] == [f"sub-{subj}_task-awake_acq-EC_eeg.vhdr"]
        assert by[("awake", "rest")] == [f"sub-{subj}_task-awake_acq-EO_eeg.vhdr"]
        assert by[("deep", "analysis")] == [
            f"sub-{subj}_task-sed2_acq-rest_run-1_eeg.vhdr"
        ]
        assert by[("deep", "rest")] == [
            f"sub-{subj}_task-sed2_acq-rest_run-2_eeg.vhdr",
            f"sub-{subj}_task-sed2_acq-rest_run-3_eeg.vhdr",
        ]
        assert not any("acq-tms" in r["source_file"] for r in rows)
        # Symlinked (git-annex) inputs keep their BIDS names in the outputs.
        assert all(".git/annex" not in r["source_file"] for r in rows)
        rest = sorted(p.name for p in (out / subj / "deep" / "rest").glob("*.npy"))
        assert rest == [f"{subj}_run-2_eeg64_ts.npy", f"{subj}_run-3_eeg64_ts.npy"]
        assert (out / subj / "awake" / "rest" / f"{subj}_run-1_eeg64_ts.npy").exists()


def test_ds005620_unfetched_annex_content_is_recorded_not_fatal(ds005620_like):
    summary, out, _subjects = ds005620_like
    missing = [r for r in summary["missing_files"] if r["subject"] == "1099"]
    assert missing and all("sed2_acq-rest_run-1" in r["file"] for r in missing)
    assert {r["reason"] for r in missing} == {"annexed_data_not_fetched"}
    # The declared deep run (first sed2 run) is unreadable: the subject is
    # skipped with a reason; later sed2 runs are not silently promoted.
    skipped = {r["subject"]: r for r in summary["skipped_subjects"]}
    assert skipped["1099"]["reason"] in {"missing_sessions", "no_readable_runs"}
    assert not (out / "1099").exists()
    assert summary["summary"]["subjects_processed"] == 2
