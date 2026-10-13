"""
Regression: the step-2 outputs carry
``<P>_estimator_version`` for every principle, the version part of the exact
evidence id recorded in ``ComponentEvidence.estimator``
(``compute_<P>:<mode>@<version>``).
"""
import pandas as pd

import run_pipeline
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
            "iim_algorithm_version": "iim-2026.08"}
    row = sc.compute_synergy_ci(
        str(prep), "toy", [0.5], sessions=("awake",), tr=TR,
        mpc_metrics=("IIM",), compute_ci=False,
        iim_precomputed_by_path={ts_path: info}, necessity_set=("IIM",),
    ).iloc[0]
    assert row["IIM_estimator"] == "compute_IIM:bidirectional@iim-2026.08"
    assert row["IIM_estimator_version"] == "iim-2026.08"
    assert VERSIONS["IIM"] != "iim-2026.08"


def test_step2_output_filter_keeps_the_version_columns():
    df = pd.DataFrame([{"subject": "1", "session": "awake", "theta": 0.5, "S": 0.1,
                        "NAS": 0.2, "NAS_estimator": "compute_NAS:capacity@v",
                        "NAS_estimator_version": "v", "MPC_verdict": "UNDETERMINED"}])
    out, _ = run_pipeline._apply_metric_subset(
        df, df[["subject", "session", "S"]], mpc_metrics=["NAS"], compute_ci=False)
    assert {"NAS_estimator", "NAS_estimator_version"} <= set(out.columns)
