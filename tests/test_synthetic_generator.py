"""Tests for the real-data-derived synthetic smoke-test generator.

All sources are miniature OpenNeuro-like trees written into temporary folders;
nothing under data/ or test_objects/ is read or written.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
EEG_NAMES = [f"C{i:02d}" for i in range(1, 11)] + ["VEOG", "HEOG", "EMG"]
EEG_SFREQ = 250.0
EEG_SECONDS = 24.0
FMRI_NODES = 24
TINY_IIM = [
    "--iim-max-nodes",
    "4",
    "--iim-max-timepoints",
    "40",
    "--iim-max-mechanism-size",
    "2",
    "--iim-max-purview-size",
    "2",
]


def _load_script(name: str):
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen():
    return _load_script("generate_real_derived_synth_completed")


@pytest.fixture(scope="module")
def inspect_mod():
    return _load_script("inspect_real_sources_for_synth")


def _nifti(path: Path, rng, n_time: int, shape=(6, 6, 6)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = 100.0 + 50.0 * rng.random(shape)
    x = np.empty((*shape, n_time), dtype=np.float32)
    state = np.zeros(shape)
    for t in range(n_time):
        state = 0.6 * state + rng.normal(size=shape)
        x[..., t] = base + state
    img = nib.Nifti1Image(x, affine=np.diag([3.0, 3.0, 3.0, 1.0]))
    img.header.set_zooms((3.0, 3.0, 3.0, 2.0))
    nib.save(img, str(path))


def _brainvision(stem: Path, rng, seconds: float, flat=()) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    n = int(round(EEG_SFREQ * seconds))
    x = np.cumsum(rng.normal(size=(n, len(EEG_NAMES))), axis=0) * 0.1
    x += rng.normal(size=x.shape)
    for ch in flat:
        x[:, EEG_NAMES.index(ch)] = 0.0
    eeg = stem.with_name(stem.name + "_eeg.eeg")
    vmrk = stem.with_name(stem.name + "_eeg.vmrk")
    x.astype("<f4").tofile(eeg)
    chans = "\n".join(f"Ch{i}={nm},,1,uV" for i, nm in enumerate(EEG_NAMES, start=1))
    stem.with_name(stem.name + "_eeg.vhdr").write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n[Common Infos]\n"
        f"DataFile={eeg.name}\nMarkerFile={vmrk.name}\nDataFormat=BINARY\n"
        f"DataOrientation=MULTIPLEXED\nNumberOfChannels={len(EEG_NAMES)}\n"
        f"SamplingInterval={1e6 / EEG_SFREQ:.6f}\n\n[Binary Infos]\n"
        f"BinaryFormat=IEEE_FLOAT_32\n\n[Channel Infos]\n{chans}\n",
        encoding="utf-8",
    )
    vmrk.write_text(
        "Brain Vision Data Exchange Marker File, Version 1.0\n[Common Infos]\n"
        f"DataFile={eeg.name}\n\n[Marker Infos]\nMk1=New Segment,,1,1,0\n",
        encoding="utf-8",
    )
    # Real ds005620 channels.tsv labels VEOG/HEOG/EMG as EEG, so exclusion must
    # not rely on the sidecar type alone.
    pd.DataFrame({"name": EEG_NAMES, "type": ["EEG"] * len(EEG_NAMES)}).to_csv(
        stem.with_name(stem.name + "_channels.tsv"), sep="\t", index=False
    )
    pd.DataFrame(
        {"onset": [0.0], "duration": [0.0002], "trial_type": ["New Segment/"]}
    ).to_csv(stem.with_name(stem.name + "_events.tsv"), sep="\t", index=False)


def _description(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset_description.json").write_text(json.dumps({"Name": "mini"}))


def build_mini_sources(root: Path, seed: int = 0) -> Path:
    """Miniature ds003171 / ds002547 / ds005620 / ds005479 source trees."""
    rng = np.random.default_rng(seed)
    ds1 = root / "ds003171"
    _description(ds1)
    for subj in ("02CB", "10JR"):
        func = ds1 / f"sub-{subj}" / "func"
        for ses in ("awake", "deep", "light", "recovery"):
            for kind in ("audio", "rest"):
                task = f"{kind}{ses}"
                if subj == "10JR" and task == "audioawake":
                    task = "audio"  # like the real sub-10JR
                _nifti(
                    func / f"sub-{subj}_task-{task}_run-01_bold.nii.gz",
                    rng,
                    120 if kind == "audio" else 90,
                )
    # git-annex layout: the BIDS file is a symlink into .git/annex/objects.
    link = ds1 / "sub-02CB" / "func" / "sub-02CB_task-audiodeep_run-01_bold.nii.gz"
    obj = ds1 / ".git" / "annex" / "objects" / "Xx" / "MD5E-s1--abc.nii.gz"
    obj.parent.mkdir(parents=True)
    link.rename(obj)
    link.symlink_to(obj)
    ds2 = root / "ds002547"
    _description(ds2)
    for subj, n_runs in (("01", 6), ("14", 2)):
        for ses in ("ses-1", "ses-2"):
            func = ds2 / f"sub-{subj}" / ses / "func"
            func.mkdir(parents=True, exist_ok=True)
            for task in ("self_run-1", "other"):
                pd.DataFrame(
                    {
                        "onset": np.arange(10.0, 100.0, 8.0),
                        "duration": 6.0,
                        "trial_type": "action",
                    }
                ).to_csv(
                    func / f"sub-{subj}_{ses}_task-{task}_events.tsv",
                    sep="\t",
                    index=False,
                )
        keys = [
            "ses-1:self_run-1",
            "ses-1:self_run-2",
            "ses-1:other",
            "ses-2:self_run-1",
            "ses-2:self_run-2",
            "ses-2:other",
        ]
        space = "space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        for key in keys[:n_runs]:
            ses, task = key.split(":")
            func = ds2 / "derivatives" / "fmriprep" / f"sub-{subj}" / ses / "func"
            _nifti(func / f"sub-{subj}_{ses}_task-{task}_{space}", rng, 130)
        (ds2 / "derivatives" / "fmriprep" / f"sub-{subj}.html").write_text("report")
    ds3 = root / "ds005620"
    _description(ds3)
    for subj in ("1010", "1037"):
        eeg = ds3 / f"sub-{subj}" / "eeg"
        if subj == "1037":  # awake recordings only, long enough for 3 segments
            for acq in ("EC", "EO"):
                _brainvision(
                    eeg / f"sub-{subj}_task-awake_acq-{acq}", rng, 3 * EEG_SECONDS + 1
                )
            continue
        _brainvision(eeg / f"sub-{subj}_task-awake_acq-EC", rng, EEG_SECONDS + 1)
        _brainvision(
            eeg / f"sub-{subj}_task-awake_acq-EO", rng, EEG_SECONDS + 1, flat=("C05",)
        )
        for task in ("sed", "sed2"):
            for run in (1, 2, 3):
                _brainvision(
                    eeg / f"sub-{subj}_task-{task}_acq-rest_run-{run}",
                    rng,
                    EEG_SECONDS + 0.5,
                )
    mid = root / "ds005479" / "sub-01" / "func"
    _description(root / "ds005479")
    mid.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "onset": 10.0 + 13.0 * np.arange(24),
            "duration": 4.3,
            "trial_type": ["Win Small", "Loss Big", "Win Big", "Loss Small"] * 6,
        }
    ).to_csv(mid / "sub-01_task-MID_events.tsv", sep="\t", index=False)
    return root


def _gen_args(synth: Path, source: Path) -> list[str]:
    return [
        "--synth-root",
        str(synth),
        "--source-root",
        str(source),
        "--fmri-nodes",
        str(FMRI_NODES),
        "--eeg-seconds",
        str(EEG_SECONDS),
        "--eeg-sfreq",
        str(EEG_SFREQ),
        "--no-link-repo-paths",
    ]


@pytest.fixture(scope="module")
def generated(gen, tmp_path_factory):
    base = tmp_path_factory.mktemp("synthgen")
    source = build_mini_sources(base / "sources")
    synth = base / "synth"
    synth.mkdir()
    assert gen.main(_gen_args(synth, source) + ["--skip-validation"]) == 0
    reports = synth / "test_objects" / "real_derived_synth_completed" / "reports"
    manifests = {
        ds: json.loads((reports / f"{ds}_manifest.json").read_text())
        for ds in gen.TARGETS
    }
    return {
        "base": base,
        "source": source,
        "synth": synth,
        "reports": reports,
        "manifests": manifests,
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): _sha(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_inspection_counts_top_level_subjects_and_respects_source_root(
    inspect_mod, tmp_path
):
    source = build_mini_sources(tmp_path / "sources")
    repo = tmp_path / "repo"
    # A dataset that exists only in the repository must not be picked up when an
    # explicit source root is given (no silent fallback).
    _description(repo / "data" / "scratch" / "ds004295")
    row = inspect_mod.inspect_dataset("ds002547", repo, 1, 1, source)
    assert row["available"] is True
    # derivatives/fmriprep/sub-01.html must not count as another subject.
    assert row["subjects"] == ["sub-01", "sub-14"]
    assert row["n_subjects"] == 2
    assert row["derivative_subjects"] == {"fmriprep": 2}
    assert str(source) not in json.dumps(row, default=str)
    missing = inspect_mod.inspect_dataset("ds004295", repo, 1, 1, source)
    assert missing["available"] is False
    eeg = inspect_mod.inspect_dataset("ds005620", repo, 1, 1, source)
    assert eeg["eeg"]["files"][0].get("bandpower"), eeg["eeg"]["files"][0].get("error")


def test_regeneration_inspects_missing_sources(generated):
    reports = generated["reports"]
    for ds in ("ds003171", "ds002547", "ds005620", "ds005479"):
        assert (reports / f"{ds}_source_inspection.json").exists()
    donor = json.loads((reports / "donor_statistics_report.json").read_text())
    assert "ds004295" not in donor["sources_read_by_generator"]
    assert "ds002336" not in donor["sources_read_by_generator"]


def test_manifest_records_configuration_and_is_relocatable(gen, generated):
    for ds, manifest in generated["manifests"].items():
        text = json.dumps(manifest)
        assert str(generated["base"]) not in text, f"{ds} manifest leaks absolute paths"
        assert manifest["object_kind"] == "software_smoke_test_objects"
        assert "not validation of the IMPaCT theory" in manifest["disclaimer"]
        iim = manifest["metric_configuration"]["iim"]
        assert iim == gen.IIM_VALIDATION_PARAMS
        assert manifest["metric_configuration"]["ram"]["response_model"] in {
            "hrf",
            "boxcar",
        }
        design = manifest["planted_design"]
        assert design["state_levels"]["awake"] > design["state_levels"]["deep"]
        assert design["known_answer_contrasts"]["planted"] == ["awake", "deep"]
        for run in manifest["runs"]:
            assert run["planted_level"] == design["state_levels"][run["session"]]
            assert not Path(run["task_array"]).is_absolute()
            for src in (run["task_source"], run["rest_source"]):
                assert src["source_path"].startswith(src["source_dataset"] + "/")
                assert "/sub-" in src["source_path"]
                assert ".git/annex" not in src["source_path"]


def test_eeg_sessions_distinct_sources_unique_stems_and_scalp_channels(generated):
    manifest = generated["manifests"]["ds005620"]
    assert manifest["sessions"] == ["awake", "deep", "sed"]
    runs = manifest["runs"]
    for subj in manifest["subjects"]:
        sub_runs = [r for r in runs if r["subject"] == subj]
        segments = [
            (r[role]["source_path"], r[role]["segment_start_sec"])
            for r in sub_runs
            for role in ("task_source", "rest_source")
        ]
        assert len(segments) == len(
            set(segments)
        ), f"sub-{subj} reuses a source segment"
        assert not any(
            r[role]["reused"]
            for r in sub_runs
            for role in ("task_source", "rest_source")
        )
        stems = {r["bids_data"] for r in sub_runs}
        assert len(stems) == len(sub_runs), "BIDS recordings overwritten"
        vhdrs = list(
            (generated["synth"] / manifest["bids_root"] / f"sub-{subj}" / "eeg").glob(
                "*_eeg.vhdr"
            )
        )
        assert len(vhdrs) == len(sub_runs)
        for r in sub_runs:
            chans = r["task_payload"]["channels"]
            assert not {"VEOG", "HEOG", "EMG"} & set(chans)
            assert "C05" not in chans or subj == "1037"  # flat in sub-1010 EO rest
    deep = next(r for r in runs if r["subject"] == "1010" and r["session"] == "deep")
    assert "task-sed2" in deep["task_source"]["source_path"]
    fallback = [r for r in runs if r["subject"] == "1037" and r["session"] != "awake"]
    assert fallback and all(r["task_source"]["state_fallback"] for r in fallback)
    assert manifest["source_reuse_summary"]["task_state_fallback"] == len(fallback)
    for r in runs:
        arr = np.load(generated["synth"] / r["task_array"])
        rest = np.load(generated["synth"] / r["rest_array"])
        assert np.all(arr.std(axis=0) > 1e-3) and np.all(rest.std(axis=0) > 1e-3)


def test_ds002547_pseudo_sessions_distinct_or_flagged_and_rest_not_tiled(generated):
    manifest = generated["manifests"]["ds002547"]
    runs = manifest["runs"]
    for subj in manifest["subjects"]:
        by_ses = {r["session"]: r for r in runs if r["subject"] == subj}
        assert (
            by_ses["awake"]["task_source"]["source_path"]
            != by_ses["deep"]["task_source"]["source_path"]
        )
        seen = {}
        for ses in manifest["sessions"]:
            src = by_ses[ses]["task_source"]
            if src["source_path"] in seen:
                assert src["reused"] and seen[src["source_path"]] in " ".join(
                    src["shared_with"]
                )
            seen.setdefault(src["source_path"], ses)
        rest_paths = [
            by_ses[s]["rest_source"]["source_path"] for s in manifest["sessions"]
        ]
        assert len(set(rest_paths)) == len(rest_paths)
    sub01 = [r for r in runs if r["subject"] == "01"]
    assert not any(r["task_source"]["reused"] for r in sub01)
    sub14 = [r for r in runs if r["subject"] == "14"]
    assert sum(r["task_source"]["reused"] for r in sub14) == 2
    for r in runs:
        # ds003171 donor rest runs have 90 volumes; they are not tiled to the
        # task length.
        assert r["rest_shape"][0] == 90 and r["task_shape"][0] == 130
        sidecar = json.loads((generated["synth"] / r["bids_sidecar"]).read_text())
        assert "ds003171" in sidecar["SourcesInspected"]
        assert re.fullmatch(r"[A-Za-z0-9]+", sidecar["TaskName"])


def test_bids_objects_match_analysed_arrays(gen, generated):
    synth = generated["synth"]
    for ds in gen.TARGETS:
        manifest = generated["manifests"][ds]
        desc = json.loads(
            (synth / manifest["bids_root"] / "dataset_description.json").read_text()
        )
        assert desc["DatasetType"] == "raw"
        for run in manifest["runs"][:3]:
            arr = np.load(synth / run["task_array"])
            checks = gen._bids_array_consistency(
                run,
                arr,
                synth,
                manifest["sample_interval_seconds"],
                manifest["modality"],
            )
            assert all(checks.values()), (ds, run["session"], checks)
    eeg_run = generated["manifests"]["ds005620"]["runs"][0]
    assert eeg_run["task_shape"][0] == int(
        EEG_SECONDS * EEG_SFREQ
    )  # full length, not 10 s


def test_planted_structure_is_present_in_arrays(gen, generated):
    """Generator self-check: each planted quantity is larger at awake than deep."""
    for ds, manifest in generated["manifests"].items():
        values = {c: {} for c in gen.MANIPULATION_CHECKS}
        for run in manifest["runs"]:
            for c in gen.MANIPULATION_CHECKS:
                values[c].setdefault(run["subject"], {})[run["session"]] = run[
                    "manipulation_checks"
                ][c]
        result = gen._evaluate_known_answers(
            values,
            levels=manifest["planted_design"]["state_levels"],
            contrasts=manifest["planted_design"]["known_answer_contrasts"],
            metrics=list(gen.MANIPULATION_CHECKS),
        )
        for check, summary in result["contrasts"]["planted"]["metrics"].items():
            assert summary["median_difference"] > 0, (ds, check, summary)


def test_planted_coupling_is_directed_without_equal_time_correlation(gen):
    """Integration is planted without reducing differentiation (ground truth)."""

    def stats(kappa, seed):
        z = gen._coupled_module_latents(
            20000,
            4,
            kappa=kappa,
            tau_sec=1.0,
            dt=0.1,
            rng=np.random.default_rng(seed),
        )
        c0 = np.corrcoef(z, rowvar=False)[np.triu_indices(4, 1)]
        lag = 5
        fwd = [np.corrcoef(z[:-lag, m - 1], z[lag:, m])[0, 1] for m in range(4)]
        bwd = [np.corrcoef(z[:-lag, m], z[lag:, m - 1])[0, 1] for m in range(4)]
        return np.max(np.abs(c0)), float(np.mean(fwd) - np.mean(bwd))

    eq0, dir0 = stats(0.0, 1)
    eq1, dir1 = stats(0.6, 1)
    # Equal-time covariance stays isotropic (~1000 effective samples, SE ~0.03).
    assert eq0 < 0.1 and eq1 < 0.1
    assert abs(dir0) < 0.05
    assert dir1 > 0.3  # directed lagged dependence m-1 -> m


def test_known_answer_evaluator_reports_recovery_reversal_and_null(gen):
    rng = np.random.default_rng(3)
    subjects = [f"s{i}" for i in range(8)]
    levels = {"awake": 1.0, "deep": 0.2, "a": 0.6, "b": 0.6}
    values = {"GOOD": {}, "BAD": {}, "UNDEF": {}, "NOISE": {}}
    for s in subjects:
        values["GOOD"][s] = {
            "awake": 2.0,
            "deep": 1.0,
            "a": 1.5,
            "b": 1.5 + 0.01 * rng.normal(),
        }
        values["BAD"][s] = {"awake": 1.0, "deep": 2.0, "a": 1.0, "b": 2.0}
        values["UNDEF"][s] = {"awake": np.nan, "deep": 1.0, "a": np.nan, "b": np.nan}
        values["NOISE"][s] = {k: float(rng.normal()) for k in levels}
    out = gen._evaluate_known_answers(
        values,
        levels=levels,
        contrasts={"planted": ("awake", "deep"), "null": ("a", "b")},
        metrics=list(values),
    )
    planted = out["contrasts"]["planted"]["metrics"]
    assert planted["GOOD"]["outcome"] == "recovered"
    assert planted["GOOD"]["sign_test_p_two_sided"] == pytest.approx(2 / 256)
    assert planted["BAD"]["outcome"] == "reversed"
    assert planted["UNDEF"]["outcome"] == "undefined"
    assert planted["NOISE"]["outcome"] in {
        "direction_only_not_significant",
        "not_recovered",
    }
    null = out["contrasts"]["null"]["metrics"]
    assert null["BAD"]["outcome"] == "systematic_difference_without_planted_difference"
    assert null["NOISE"]["outcome"] == "no_systematic_difference"
    assert out["monotonic"]["GOOD"]["mean_spearman_level_vs_metric"] > 0.9
    assert out["monotonic"]["BAD"]["mean_spearman_level_vs_metric"] < 0


def test_manifest_paths_rebase_for_legacy_absolute_paths(gen, tmp_path):
    rel = Path("test_objects/runs/real_derived_synth_completed/ds003171/preprocessed")
    root = tmp_path / "extracted"
    target = root / rel
    target.mkdir(parents=True)
    # Legacy (v1) manifests store absolute paths of the machine that built them.
    foreign = str(Path("/nonexistent_build_machine/impact-synergy-pipeline") / rel)
    assert gen._resolve_manifest_path(foreign, root) == target
    # Even if the original machine's copy exists, the extracted copy wins.
    original = tmp_path / "original_machine"
    (original / rel).mkdir(parents=True)
    assert gen._resolve_manifest_path(str(original / rel), root) == target
    assert gen._resolve_manifest_path(rel.as_posix(), root) == target.resolve()
    with pytest.raises(FileNotFoundError):
        gen._resolve_manifest_path("/nowhere/else/file.npy", root)
    # The build machine's own path may itself contain a 'test_objects' folder.
    nested = Path("/nonexistent/test_objects/impact-synergy-pipeline") / rel
    assert gen._resolve_manifest_path(str(nested), root) == target


def test_validate_only_on_relocated_copy_is_nondestructive(
    gen, generated, tmp_path, monkeypatch
):
    moved = tmp_path / "somewhere_else"
    shutil.copytree(generated["synth"], moved)
    reports = moved / "test_objects" / "real_derived_synth_completed" / "reports"
    before = _tree_hashes(moved)
    monkeypatch.setenv("IMPACT_SYNTH_ROOT", str(moved))
    out = tmp_path / "revalidation"
    rc = gen.main(
        [
            "--validate-only",
            "--datasets",
            "ds005620",
            "--subjects",
            "1010",
            "--validation-out",
            str(out),
        ]
        + TINY_IIM
    )
    after = _tree_hashes(moved)
    assert before == after, "validate-only modified the shipped objects or reports"
    assert not (reports / "ds005620").exists()
    summary = json.loads((out / gen.SUMMARY_NAME).read_text())
    assert "all_validated" not in summary
    assert summary["object_kind"] == "software_smoke_test_objects"
    rep = summary["validation"]["ds005620"]
    # Well-formed objects must pass the smoke gate (and exit 0).
    assert summary["smoke_test_passed"] is True, rep["smoke_test"]
    assert rc == 0
    smoke = rep["smoke_test"]
    assert smoke["dataset_checks"]["objects_and_reports_unmodified"] is True
    assert smoke["dataset_checks"]["metric_table_has_required_columns"] is True
    assert not smoke["stage_errors"]
    assert rep["iim_configuration"]["iim_max_nodes"] == 4
    assert smoke["n_rows"] == 3
    rows = {r["session"]: r for r in rep["rows"]}
    for row in rows.values():
        for name in (
            "arrays_finite",
            "no_zero_variance_nodes",
            "bids_data_matches_array",
            "task_shape_matches_manifest",
            "CI_status_consistent",
            "readiness_reasons_recorded",
        ):
            assert row["checks"][name], (row["session"], name)
        # Every value is finite or NaN with a recorded reason. The
        # 24 s EEG runs hold 3 trials, so RAM is undefined (<6 goal pairs) and
        # CI with it; that is a valid, explained outcome, not a failure.
        for metric in gen.KNOWN_ANSWER_METRICS:
            value = row["metric_values"][metric]
            reason = row["undefined_reasons"].get(metric)
            assert np.isfinite(value) or reason["reason"], (row["session"], metric)
        assert row["undefined_reasons"]["RAM"]["reason"] == (
            "insufficient_goal_response_pairs"
        )
        assert row["undefined_reasons"]["CI"]["source"] == "CI_missing"
        assert "RAM" in row["undefined_reasons"]["CI"]["reason"].split(",")
    assert smoke["undefined_metrics"]["RAM"] == {"with_reason": 3, "without_reason": 0}
    planted = rep["known_answer"]["contrasts"]["planted"]
    assert planted["sessions"] == ["awake", "deep"]
    assert set(planted["metrics"]) >= set(gen.KNOWN_ANSWER_METRICS)
    for summ in planted["metrics"].values():
        assert summ["outcome"] in {
            "recovered",
            "direction_only_not_significant",
            "not_recovered",
            "reversed",
            "undefined",
        }
    assert rep["planted_structure_verified"] in (True, False)
    # The low-level session carries more zero-lag synchrony by construction.
    opposing = rep["opposing_structure"]
    assert opposing["expected"] == "deep > awake"
    assert opposing["summary"]["n_defined_pairs"] == 1
    assert opposing["present"] is True
    assert "zero-lag synchrony" in summary["known_answer_note"]
    # The pipeline event resolver is compared against the exact events file.
    res = {r["session"]: r for r in rep["pipeline_event_resolution"]["runs"]}
    assert res["awake"]["consistent"] and res["deep"]["consistent"]
    csv = pd.read_csv(out / "ds005620" / "actual_metric_computation.csv")
    assert set(csv["session"]) == {"awake", "deep", "sed"}


def test_smoke_gate_fails_on_broken_arrays(gen, generated, tmp_path):
    moved = tmp_path / "broken"
    shutil.copytree(generated["synth"], moved)
    manifest = generated["manifests"]["ds005620"]
    run = next(
        r for r in manifest["runs"] if r["subject"] == "1010" and r["session"] == "sed"
    )
    arr = np.load(moved / run["task_array"])
    arr[:, 0] = 0.0
    np.save(moved / run["task_array"], arr)
    # A missing array must be reported as a failed check, not abort validation.
    awake = next(
        r
        for r in manifest["runs"]
        if r["subject"] == "1010" and r["session"] == "awake"
    )
    (moved / awake["rest_array"]).unlink()
    out = tmp_path / "reval"
    rc = gen.main(
        [
            "--validate-only",
            "--synth-root",
            str(moved),
            "--datasets",
            "ds005620",
            "--subjects",
            "1010",
            "--validation-out",
            str(out),
        ]
        + TINY_IIM
    )
    summary = json.loads((out / gen.SUMMARY_NAME).read_text())
    assert rc == 1 and summary["smoke_test_passed"] is False
    failed = summary["validation"]["ds005620"]["smoke_test"]["failed_checks"]
    assert "no_zero_variance_nodes" in failed and "bids_data_matches_array" in failed
    assert "rest_array_exists" in failed
    rows = {
        r["session"]: r["checks"] for r in summary["validation"]["ds005620"]["rows"]
    }
    assert rows["awake"]["rest_array_exists"] is False
    assert rows["deep"]["rest_array_exists"] is True


def _revalidate_1010(gen, synth: Path, out: Path) -> tuple[int, dict]:
    rc = gen.main(
        [
            "--validate-only",
            "--synth-root",
            str(synth),
            "--datasets",
            "ds005620",
            "--subjects",
            "1010",
            "--validation-out",
            str(out),
        ]
        + TINY_IIM
    )
    return rc, json.loads((out / gen.SUMMARY_NAME).read_text())


def test_smoke_gate_fails_when_an_undefined_value_has_no_reason(
    gen, generated, tmp_path, monkeypatch
):
    """NaN is valid only with a recorded reason."""
    real = gen.compute_synergy_ci

    def reasonless(*args, **kwargs):
        df = real(*args, **kwargs)
        df["CI_missing"] = ""  # the undefined CI no longer says why
        df["NAS"] = np.nan  # NAS undefined, but nothing records a reason
        return df

    monkeypatch.setattr(gen, "compute_synergy_ci", reasonless)
    rc, summary = _revalidate_1010(gen, generated["synth"], tmp_path / "reval")
    smoke = summary["validation"]["ds005620"]["smoke_test"]
    assert rc == 1 and summary["smoke_test_passed"] is False
    failed = smoke["failed_checks"]
    assert "CI_finite_or_reasoned" in failed and "NAS_finite_or_reasoned" in failed
    assert "CI_status_consistent" in failed
    # RAM is also NaN, but its reason is recorded: it does not fail.
    assert "RAM_finite_or_reasoned" not in failed
    assert smoke["undefined_metrics"]["NAS"] == {"with_reason": 0, "without_reason": 3}


def test_smoke_gate_fails_when_validation_modifies_the_objects(
    gen, generated, tmp_path, monkeypatch
):
    moved = tmp_path / "objects"
    shutil.copytree(generated["synth"], moved)
    manifest = generated["manifests"]["ds005620"]
    victim = moved / manifest["runs"][0]["bids_events"]
    real = gen.compute_synergy_ci

    def destructive(*args, **kwargs):
        with open(victim, "a", encoding="utf-8") as fh:
            fh.write("# touched by the metric code\n")
        return real(*args, **kwargs)

    monkeypatch.setattr(gen, "compute_synergy_ci", destructive)
    rc, summary = _revalidate_1010(gen, moved, tmp_path / "reval")
    smoke = summary["validation"]["ds005620"]["smoke_test"]
    assert rc == 1
    assert smoke["dataset_checks"]["objects_and_reports_unmodified"] is False
    assert "objects_and_reports_unmodified" in smoke["failed_checks"]
    assert smoke["modified_paths"]["modified"] == [
        manifest["runs"][0]["bids_events"]
    ]


def test_smoke_gate_reports_a_corrupt_array_instead_of_crashing(
    gen, generated, tmp_path
):
    moved = tmp_path / "corrupt"
    shutil.copytree(generated["synth"], moved)
    manifest = generated["manifests"]["ds005620"]
    run = next(
        r for r in manifest["runs"] if r["subject"] == "1010" and r["session"] == "deep"
    )
    path = moved / run["task_array"]
    path.write_bytes(path.read_bytes()[:200])  # truncated .npy
    rc, summary = _revalidate_1010(gen, moved, tmp_path / "reval")
    smoke = summary["validation"]["ds005620"]["smoke_test"]
    assert rc == 1 and summary["smoke_test_passed"] is False
    assert "task_array_readable" in smoke["failed_checks"]
    rows = {
        r["session"]: r["checks"] for r in summary["validation"]["ds005620"]["rows"]
    }
    assert rows["deep"]["task_array_readable"] is False
    assert rows["awake"]["task_array_readable"] is True


def test_undefined_reason_sources_and_ci_status_known_answers(gen):
    nan = float("nan")
    # Metric-table reason columns come first; 'ok' and NaN are not reasons.
    rec = {"IIM": nan, "IIM_undefined_reason": "constant_input", "IIM_reason": "x"}
    assert gen._undefined_reason("IIM", rec) == {
        "reason": "constant_input",
        "source": "IIM_undefined_reason",
    }
    rec = {"PDI": nan, "PDI_primary_source": "anchor", "PDI_anchor_reason": "ok"}
    assert gen._undefined_reason("PDI", rec) is None
    rec["PDI_anchor_reason"] = "missing_anchor_baseline"
    assert gen._undefined_reason("PDI", rec)["source"] == "PDI_anchor_reason"
    # As compute_synergy_ci writes an undefined PDI: the source is
    # 'undefined', the declared endpoint says which baseline reason applies.
    rec = {
        "PDI": nan,
        "PDI_primary_endpoint": "anchor",
        "PDI_primary_source": "undefined",
        "PDI_anchor_reason": "missing_deep_rest_baseline",
        "PDI_task_reason": "ok",
    }
    assert gen._undefined_reason("PDI", rec) == {
        "reason": "missing_deep_rest_baseline",
        "source": "PDI_anchor_reason",
    }
    rec.update(
        PDI_primary_endpoint="task",
        PDI_task_reason="state_pdi_estimator_undefined",
    )
    assert gen._undefined_reason("PDI", rec)["source"] == "PDI_task_reason"
    # A requested metric the code skipped ('not_computed') is not explained.
    rec = {"IIM": nan, "IIM_undefined_reason": "not_computed"}
    assert gen._undefined_reason("IIM", rec) is None
    # The direct call counts only when it is undefined as well.
    direct = {"value": 0.3, "undefined_reason": "stale"}
    assert gen._undefined_reason("RAM", {"RAM": nan}, direct=direct) is None
    direct = {"value": nan, "undefined_reason": "insufficient_goal_response_pairs"}
    assert gen._undefined_reason("RAM", {"RAM": nan}, direct=direct)["reason"] == (
        "insufficient_goal_response_pairs"
    )
    ready = {"NAS_ready": False, "NAS_reason": "too_few_regions"}
    assert gen._undefined_reason("NAS", {}, readiness=ready)["source"] == (
        "readiness.NAS_reason"
    )
    assert gen._undefined_reason("NAS", {}, readiness={"NAS_ready": True}) is None
    for value in (None, nan, "", "  ", "ok", "nan", 0.0, True):
        assert gen._reason_text(value) is None
    comps = {"RAM": 1.0, "PDI": 1.0, "NAS": 1.0, "IIM": 0.5, "SRPI": 0.2}
    ok_defined = {**comps, "CI": 0.7, "CI_defined": True, "CI_missing": nan}
    assert gen._ci_status_consistent(ok_defined)
    assert not gen._ci_status_consistent({**ok_defined, "CI_defined": False})
    # An undefined component never enters a finite CI (e.g. as 0).
    assert not gen._ci_status_consistent({**ok_defined, "SRPI": nan})
    assert not gen._ci_status_consistent({**ok_defined, "IIM_defined": False})
    undefined = {
        **comps,
        "RAM": nan,
        "CI": nan,
        "CI_defined": False,
        "CI_missing": "RAM,NAS_reference",
    }
    assert gen._ci_status_consistent(undefined)
    assert not gen._ci_status_consistent({**undefined, "CI_missing": ""})
    assert not gen._ci_status_consistent({**undefined, "CI_missing": "NAS_reference"})
    assert not gen._ci_status_consistent({**undefined, "CI_missing": "RAM,bogus"})
    assert not gen._ci_status_consistent({**undefined, "CI_defined": True})
    # A flagged-undefined IIM must be listed even when a value is stored.
    flagged = {**undefined, "IIM_defined": False, "CI_missing": "RAM,IIM"}
    assert gen._ci_status_consistent(flagged)
    assert not gen._ci_status_consistent({**flagged, "CI_missing": "RAM"})


def test_tree_fingerprint_sees_edits_and_skips_the_output_folder(gen, tmp_path):
    (tmp_path / "objects" / "a").mkdir(parents=True)
    (tmp_path / "objects" / "a" / "x.npy").write_bytes(b"123")
    out = tmp_path / "objects" / "revalidation"
    out.mkdir()
    before = gen._tree_fingerprint([tmp_path / "objects"], exclude=[out])
    (out / "report.json").write_text("{}")  # the validator's own output
    assert gen._tree_fingerprint([tmp_path / "objects"], exclude=[out]) == before
    (tmp_path / "objects" / "a" / "x.npy").write_bytes(b"1234")
    (tmp_path / "objects" / "new.txt").write_text("n")
    changes = gen._fingerprint_changes(
        before, gen._tree_fingerprint([tmp_path / "objects"], exclude=[out]), tmp_path
    )
    assert changes == {
        "added": ["objects/new.txt"],
        "removed": [],
        "modified": ["objects/a/x.npy"],
    }
    # A root that is a link to a folder is walked: edits behind it are seen.
    linked = tmp_path / "linked"
    linked.symlink_to(tmp_path / "objects", target_is_directory=True)
    before = gen._tree_fingerprint([linked], exclude=[linked / "revalidation"])
    assert str(linked / "a" / "x.npy") in before
    (tmp_path / "objects" / "a" / "x.npy").write_bytes(b"12345")
    after = gen._tree_fingerprint([linked], exclude=[linked / "revalidation"])
    assert gen._fingerprint_changes(before, after, tmp_path)["modified"] == [
        "linked/a/x.npy"
    ]


def test_validate_only_accepts_legacy_manifest_with_foreign_absolute_paths(
    gen, generated, tmp_path
):
    """Archive built by generator 1.x: absolute paths of another machine."""
    moved = tmp_path / "legacy_archive"
    shutil.copytree(generated["synth"], moved)
    reports = moved / "test_objects" / "real_derived_synth_completed" / "reports"
    v2 = generated["manifests"]["ds003171"]
    foreign = Path("/mnt/other_machine/impact-synergy-pipeline")
    legacy = {
        "dataset_id": "ds003171",
        "bids_root": str(foreign / v2["bids_root"]),
        "preprocessed_root": str(foreign / v2["preprocessed_root"]),
        "atlas": v2["atlas"],
        "condition": v2["condition"],
        "sessions": v2["sessions"],
        "validation_sessions": v2["sessions"],
        "subjects": v2["subjects"],
        "modality": v2["modality"],
        "sample_interval_seconds": v2["sample_interval_seconds"],
    }
    (reports / "ds003171_manifest.json").write_text(json.dumps(legacy))
    out = tmp_path / "reval"
    gen.main(
        [
            "--validate-only",
            "--synth-root",
            str(moved),
            "--datasets",
            "ds003171",
            "--subjects",
            "02CB",
            "--validation-out",
            str(out),
        ]
        + TINY_IIM
    )
    rep = json.loads((out / gen.SUMMARY_NAME).read_text())["validation"]["ds003171"]
    assert rep["manifest_version"] == 1
    assert rep["smoke_test"]["n_rows"] == 4
    assert rep["known_answer"]["available"] is False
    assert rep["planted_structure_verified"] is None
    assert rep["pipeline_event_resolution"]["consistent"] is None
    assert all(r["checks"]["arrays_finite"] for r in rep["rows"])


def test_subject_subset_regenerates_identical_objects(gen, generated, tmp_path):
    """A subject's objects do not depend on which other subjects are generated."""
    subset = tmp_path / "subset"
    subset.mkdir()
    args = _gen_args(subset, generated["source"]) + ["--skip-validation"]
    for ds, subj in (("ds005620", "1037"), ("ds002547", "14")):
        assert gen.main(args + ["--datasets", ds, "--subjects", subj]) == 0
        reports = subset / "test_objects" / "real_derived_synth_completed" / "reports"
        sub_manifest = json.loads((reports / f"{ds}_manifest.json").read_text())
        full = {
            r["session"]: r
            for r in generated["manifests"][ds]["runs"]
            if r["subject"] == subj
        }
        assert sub_manifest["subjects"] == [subj]
        for run in sub_manifest["runs"]:
            ref = full[run["session"]]
            rest_src = run["rest_source"]["source_path"]
            assert rest_src == ref["rest_source"]["source_path"]
            for key in ("task_array", "rest_array"):
                a = np.load(generated["synth"] / ref[key])
                b = np.load(subset / run[key])
                assert np.array_equal(a, b), (ds, subj, run["session"], key)


def test_inspection_relativizes_paths_inside_messages(inspect_mod, tmp_path):
    root = tmp_path / "sources" / "ds005620"
    row = {
        "error": f"[Errno 2] No such file or directory: '{root}/sub-1/eeg/x.vhdr'",
        "files": [str(root / "sub-1" / "a.tsv")],
        "local_root": str(root),
    }
    out = inspect_mod._relativize(row, root, "ds005620")
    text = json.dumps(out)
    assert str(tmp_path) not in text
    assert out["files"] == ["ds005620/sub-1/a.tsv"]
    assert "'ds005620/sub-1/eeg/x.vhdr'" in out["error"]
    assert out["local_root"] == "ds005620"


def test_source_allocator_flags_only_true_reuse(gen, tmp_path):
    short = tmp_path / "short.vhdr"
    short.write_text("x")
    alloc = gen._SourceAllocator(60.0, lambda _p: 10.0)
    first = alloc.allocate([(short, True)], "awake/task")
    # Shorter than one segment, but nobody else uses it: not a reuse.
    assert first["reused"] is False and first["shared_with"] == []
    second = alloc.allocate([(short, True)], "deep/task")
    assert second["reused"] is True and second["shared_with"] == ["awake/task"]
    with pytest.raises(FileNotFoundError):
        alloc.allocate([(tmp_path / "missing.vhdr", True)], "sed/task")


def test_constant_metrics_are_reported(gen):
    rows = [
        {"metric_values": {"NAS": 0.0, "IIM": 0.2, "SRPI": np.nan}},
        {"metric_values": {"NAS": 0.0, "IIM": 0.3, "SRPI": 0.5}},
        {"metric_values": {"NAS": 0.0, "IIM": np.nan, "SRPI": np.nan}},
    ]
    assert gen._constant_metrics(rows, ("NAS", "IIM", "SRPI", "RAM")) == {"NAS": 0.0}


def test_ram_parameters_are_the_pipeline_presets(gen):
    """The smoke test computes RAM with run_pipeline's own presets (no stale copy)."""
    from impact_pipeline.run_synergy_ci import RAM_PARAM_PRESETS

    assert gen.EEG_RAM_PARAMS == RAM_PARAM_PRESETS["eeg"]
    assert gen.FMRI_RAM_PARAMS == RAM_PARAM_PRESETS["fmri"]
    # quality_ridge is relative to the covariance scale since the RAM fix;
    # the old private 1e-4 left the cross-validated CCA almost unregularised.
    assert gen.EEG_RAM_PARAMS["quality_ridge"] == 1.0


@pytest.mark.parametrize("seconds,n_trials", [(24.0, 3), (60.0, 9)])
def test_eeg_events_never_overlap_and_short_runs_are_not_compressed(
    gen, seconds, n_trials
):
    dt = 1.0 / EEG_SFREQ
    mid = pd.DataFrame(
        {
            "onset": [10.0, 23.0, 36.0, 49.0],
            "duration": 4.3,
            "trial_type": ["Win Small", "Loss Big", "Win Big", "Loss Small"],
            "feedback_value": [1.05, 0.25, 1.55, 0.55],
        }
    )
    empty = pd.DataFrame({"onset": []})
    ev = gen._build_events(
        rng=np.random.default_rng(0),
        n_time=int(round(seconds / dt)),
        tr=dt,
        modality="eeg",
        mid_events=mid,
        self_template=empty,
        other_template=empty,
        session="awake",
    )
    sep, support = gen._eeg_event_separation(dt)
    assert support == pytest.approx(0.8) and sep == pytest.approx(1.0)
    onsets = np.sort(ev["onset"].to_numpy(dtype=float))
    # Onset jitter has SD 0.015 s; the planted responses never superimpose.
    assert np.diff(onsets).min() > sep - 0.1
    assert onsets.max() + support <= seconds
    counts = ev["trial_type"].value_counts().to_dict()
    # Six-second trial grid kept: a 24 s run holds 3 trials (the old fallback
    # packed 8 trials 1.4 s apart, with each goal cue on the previous feedback).
    assert counts["goal_cue"] == counts["stimulus_target"] == n_trials
    assert counts["feedback_reward"] == n_trials
    assert counts["self"] >= 3 and counts["nonself"] >= 3
    stim = ev.loc[ev["trial_type"].eq("stimulus_target"), "onset"].to_numpy()
    fb = ev.loc[ev["trial_type"].eq("feedback_reward"), "onset"].to_numpy()
    # Feedback responses start after the stimulus FIR latency search window.
    assert np.all(fb - stim > gen.EEG_RAM_PARAMS["fir_window"])
