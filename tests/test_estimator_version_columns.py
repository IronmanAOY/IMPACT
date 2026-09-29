"""
Regression (1.1.0 post-freeze fix): the step-2 outputs carry
``<P>_estimator_version`` for every principle, the version part of the exact
evidence id recorded in ``ComponentEvidence.estimator``
(``compute_<P>:<mode>@<version>``), as ``scripts/run_predictions.py``
requires (the frozen code wrote only ``<P>_estimator``, so every pipeline
table was refused with ``results lack <P>_estimator/<P>_estimator_version``).
"""
import pandas as pd

import run_pipeline
import scripts.run_predictions as rp
from impact_pipeline import evidence as E
from impact_pipeline import mpc_metrics as mm
from impact_pipeline import synergy_ci as sc
from test_verdict_wiring import TR, _layout, _run

VERSIONS = mm.ESTIMATOR_VERSIONS


def test_every_principle_has_an_estimator_version_column():
    for p in E.PRINCIPLES:
        assert f"{p}_estimator_version" in sc.MPC_EVIDENCE_COLUMNS
    i = sc.MPC_EVIDENCE_FIELDS.index("estimator")
    assert sc.MPC_EVIDENCE_FIELDS[i + 1] == "estimator_version"


def test_version_column_is_the_version_of_the_recorded_estimator_id(tmp_path):
    prep, onsets = _layout(tmp_path, with_events=True)
    from impact_pipeline.run_synergy_ci import RAM_PARAM_PRESETS

    ram = {**RAM_PARAM_PRESETS["eeg"], "quality_null_samples": 5}
    df = _run(prep, sessions=("awake",), stimulus_onsets=onsets, ram_params=ram,
              compute_ci=False, mpc_metrics=("RAM", "PDI", "NAS", "SRPI"))
    row = df.iloc[0]
    for p in ("RAM", "PDI", "NAS", "SRPI"):
        name, version = E.split_estimator(row[f"{p}_estimator"])
        assert name.startswith(f"compute_{p}:")
        assert row[f"{p}_estimator_version"] == version == VERSIONS[p]
    # a principle that was not computed has neither
    assert row["IIM_estimator"] == "" and row["IIM_estimator_version"] == ""


def test_precomputed_iim_keeps_its_own_recorded_version(tmp_path):
    """The column is the version the result was computed with (e.g. an older
    Hunter reduction), not the current code's ESTIMATOR_VERSIONS entry."""
    prep, _ = _layout(tmp_path)
    ts_path = str(prep / "s1" / "awake" / "audio" / "s1_run-1_toy_ts.npy")
    info = {"defined": True, "undefined_reason": None, "canonical": 0.4,
            "raw": 0.4, "Psi_full": 1.0, "Psi_mip_preserved": 0.6,
            "iim_algorithm_version": "iim-v3-2026.08"}
    row = sc.compute_synergy_ci(
        str(prep), "toy", [0.5], sessions=("awake",), tr=TR,
        mpc_metrics=("IIM",), compute_ci=False,
        iim_precomputed_by_path={ts_path: info}, necessity_set=("IIM",),
    ).iloc[0]
    assert row["IIM_estimator"] == "compute_IIM:bidirectional@iim-v3-2026.08"
    assert row["IIM_estimator_version"] == "iim-v3-2026.08"
    assert VERSIONS["IIM"] != "iim-v3-2026.08"


def test_step2_output_filter_keeps_the_version_columns():
    df = pd.DataFrame([{"subject": "1", "session": "awake", "theta": 0.5, "S": 0.1,
                        "NAS": 0.2, "NAS_estimator": "compute_NAS:capacity@v",
                        "NAS_estimator_version": "v", "MPC_verdict": "UNDETERMINED"}])
    out, _ = run_pipeline._apply_metric_subset(
        df, df[["subject", "session", "S"]], mpc_metrics=["NAS"], compute_ci=False)
    assert {"NAS_estimator", "NAS_estimator_version"} <= set(out.columns)


def _registry_for(df):
    """A registry that registers exactly the (estimator, version) pairs the
    pipeline emitted, named as the registry names them (no '@version')."""
    ests = []
    for p in E.PRINCIPLES:
        for est in sorted(set(df[f"{p}_estimator"])):
            name, version = E.split_estimator(est)
            ests.append({"principle": p, "estimator": name, "version": version,
                         "registration": "registered"})
    return {"estimators": ests, "datasets": [{"id": "toy"}],
            "protocols": [{"hash": df["MPC_protocol_hash"].iloc[0],
                           "necessity_set": list(E.PRINCIPLES)}]}


def test_run_predictions_accepts_the_pipeline_estimator_columns(tmp_path):
    prep, onsets = _layout(tmp_path, with_events=True)
    from impact_pipeline.run_synergy_ci import RAM_PARAM_PRESETS

    ram = {**RAM_PARAM_PRESETS["eeg"], "quality_null_samples": 5}
    df = _run(prep, sessions=("awake", "deep"), stimulus_onsets=onsets,
              ram_params=ram, compute_ci=False)
    table = df.assign(dataset="toy", protocol_hash=df["MPC_protocol_hash"])
    reg = _registry_for(table)
    assert rp.registration_problems(table, reg) == []
    # without the version column the frozen tables were refused
    probs = rp.registration_problems(
        table.drop(columns=["IIM_estimator_version"]), reg)
    assert "results lack IIM_estimator/IIM_estimator_version" in probs
    # a version column that contradicts the id is refused, never guessed
    bad = table.copy()
    bad["NAS_estimator_version"] = "nas-v0"
    assert any("inconsistent NAS_estimator / NAS_estimator_version" in p
               for p in rp.registration_problems(bad, reg))
    # a consistent but unregistered version is refused as unregistered
    bad["NAS_estimator"] = "compute_NAS:capacity@nas-v0"
    probs = rp.registration_problems(bad, reg)
    assert "unregistered NAS estimator versions ['compute_NAS:capacity@nas-v0']" in (
        probs)


def test_estimator_identity_splits_the_pipeline_id():
    ident = rp.estimator_identity
    assert ident("compute_NAS:capacity@nas-v2", "nas-v2") == (
        "compute_NAS:capacity", "nas-v2", None)
    assert ident("compute_NAS:capacity", "nas-v2") == (
        "compute_NAS:capacity", "nas-v2", None)
    # a version suffix that differs from the version column is a conflict
    assert ident("compute_NAS:capacity@nas-v2", "nas-v3")[2] == (
        "versions 'nas-v2' / 'nas-v3'")
