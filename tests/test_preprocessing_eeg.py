"""
EEG preprocessing (spec D10): non-EEG channels are excluded by type, BIDS run
numbers are kept, and rest/baseline segments are written when the dataset has
them so the strict PDI anchor can be defined (otherwise it stays undefined
with a reason).

ds005620 declares VEOG, HEOG and EMG as type EEG in both the BrainVision
header and channels.tsv; before the fix they were kept as network nodes.
"""

import json
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
