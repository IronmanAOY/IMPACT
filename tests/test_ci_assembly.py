"""CI assembly: three-valued definedness, no S in CI, explicit reference.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest

from impact_pipeline.synergy_ci import (
    CI_REFERENCE_COHORT_HIGH_STATE,
    assemble_ci,
    compute_synergy_ci,
    load_ci_reference,
    resolve_ci_references,
)
from test_synergy_ci import _NAS_PARAMS, _PDI_PARAMS, _SRPI_PARAMS

COMPS = ("RAM", "PDI", "NAS", "IIM", "SRPI")


def _frame():
    rows = []
    vals = {
        ("a", "awake"): (2.0, 0.2, 0.4, 0.6, 0.8),
        ("a", "deep"): (1.0, 0.1, 0.2, 0.3, 0.4),
        ("b", "awake"): (4.0, 0.4, 0.8, 1.2, 1.6),
        ("b", "deep"): (1.0, 0.1, 0.2, np.nan, 0.4),  # IIM undefined
    }
    for (subj, ses), comp in vals.items():
        for theta, S in ((0.3, 0.001), (0.6, 0.5)):  # S differs across theta
            rows.append(
                {
                    "subject": subj,
                    "session": ses,
                    "theta": theta,
                    "S": S,
                    **dict(zip(COMPS, comp)),
                    "IIM_defined": bool(np.isfinite(comp[3])),
                }
            )
    return pd.DataFrame(rows)


def test_cohort_high_state_reference_uses_subject_means():
    df = _frame()
    refs, label = resolve_ci_references(df)
    assert label == CI_REFERENCE_COHORT_HIGH_STATE
    assert refs["RAM"] == pytest.approx((2.0 + 4.0) / 2)
    assert refs["NAS"] == pytest.approx((0.4 + 0.8) / 2)


def test_ci_uses_nas_directly_and_is_theta_invariant():
    out, refs = assemble_ci(_frame())
    for (subj, ses), grp in out.groupby(["subject", "session"]):
        ci = grp["CI"].to_numpy(dtype=float)
        assert np.all(np.isnan(ci)) or np.allclose(ci, ci[0])  # S varies, CI does not
    row = out[(out.subject == "a") & (out.session == "awake")].iloc[0]
    assert row["NAS_norm"] == pytest.approx(0.4 / refs["NAS"])
    expected = np.prod([row[k] / refs[k] for k in COMPS]) ** 0.2
    assert row["CI"] == pytest.approx(expected, rel=1e-9)
    assert (out["CI_reference"] == CI_REFERENCE_COHORT_HIGH_STATE).all()


def test_undefined_component_gives_nan_with_flags_not_zero():
    out, _ = assemble_ci(_frame())
    bad = out[(out.subject == "b") & (out.session == "deep")]
    assert bad["CI"].isna().all()
    assert (~bad["CI_defined"]).all()
    assert (bad["CI_missing"] == "IIM").all()
    good = out[~((out.subject == "b") & (out.session == "deep"))]
    assert good["CI_defined"].all() and (good["CI_missing"] == "").all()


def test_iim_defined_flag_overrides_finite_value():
    df = _frame()
    df.loc[0, "IIM_defined"] = False
    out, _ = assemble_ci(df)
    assert math.isnan(out.loc[0, "CI"]) and out.loc[0, "CI_missing"] == "IIM"


def test_external_reference_dict_and_json(tmp_path):
    refs = {"RAM": 1.0, "PDI": 0.1, "NAS": 0.2, "IIM": 0.3, "SRPI": 0.4}
    out, used = assemble_ci(_frame(), reference=refs)
    assert used == refs and (out["CI_reference"] == "external").all()
    path = tmp_path / "refs.json"
    path.write_text(json.dumps({"references": refs, "source": "test cohort"}))
    assert load_ci_reference(path) == refs
    out_j, _ = assemble_ci(_frame(), reference=str(path))
    assert out_j["CI_reference"].iloc[0].startswith("external_json:")
    np.testing.assert_allclose(out_j["CI"], out["CI"], equal_nan=True)
    # Reference choice rescales CI by one constant factor (ratios unchanged).
    base, _ = assemble_ci(_frame())
    ratio = (out["CI"] / base["CI"]).dropna().to_numpy()
    assert np.allclose(ratio, ratio[0])


def test_external_reference_missing_component_raises(tmp_path):
    path = tmp_path / "refs.json"
    path.write_text(json.dumps({"RAM": 1.0}))
    with pytest.raises(ValueError, match="missing reference means"):
        load_ci_reference(path)


def test_nonpositive_or_missing_reference_makes_ci_undefined_no_floor():
    df = _frame()
    df["SRPI"] = df["SRPI"].where(
        df["session"] != "awake", 0.0
    )  # awake mean 0 -> old floor 1e-12
    out, refs = assemble_ci(df)
    assert refs["SRPI"] == 0.0
    assert out["CI"].isna().all()
    assert out["CI_missing"].str.contains("SRPI_reference").all()
    # No high-state rows at all: undefined, not a silent fallback to all rows.
    out2, refs2 = assemble_ci(_frame(), high_state_session="missing_session")
    assert all(math.isnan(v) for v in refs2.values())
    assert out2["CI"].isna().all()


def test_compute_synergy_ci_emits_ci_status_columns(tmp_path):
    rng = np.random.RandomState(0)
    iim = {}
    for subj in ("s1", "s2"):
        for ses in ("awake", "deep"):
            d = tmp_path / subj / ses / "audio"
            d.mkdir(parents=True)
            p = d / f"{subj}_run-1_schaefer400_ts.npy"
            np.save(p, rng.randn(96, 6))
            iim[str(p)] = {"defined": True, "canonical": 0.3, "raw": 0.3}
    df = compute_synergy_ci(
        str(tmp_path),
        "schaefer400",
        [0.3, 0.6],
        sessions=("awake", "deep"),
        tr=0.1,
        pdi_params=_PDI_PARAMS,
        pdi_require_explicit_params=True,
        pdi_require_strict_baseline=True,
        nas_params=_NAS_PARAMS,
        srpi_params=_SRPI_PARAMS,
        iim_precomputed_by_path=iim,
    )
    for col in ("CI", "CI_defined", "CI_missing", "CI_reference", "NAS_norm"):
        assert col in df.columns
    # No events and no rest baselines: RAM/PDI/SRPI unmeasurable ->
    # CI undefined (NaN), never 0.
    assert df["CI"].isna().all()
    assert (~df["CI_defined"]).all()
    assert (
        df["CI_missing"].str.contains("RAM").all()
        and df["CI_missing"].str.contains("SRPI").all()
    )
    assert (df["CI_reference"] == CI_REFERENCE_COHORT_HIGH_STATE).all()


def test_cohort_reference_ignores_iim_flagged_undefined():
    df = _frame()
    # Subject a's awake IIM is flagged undefined although a value is stored.
    df.loc[(df.subject == "a") & (df.session == "awake"), "IIM_defined"] = False
    refs, _ = resolve_ci_references(df)
    assert refs["IIM"] == pytest.approx(1.2)  # only subject b's awake IIM


def test_compute_synergy_ci_end_to_end_reference_and_theta_invariance(
    tmp_path, monkeypatch
):
    """All five components defined (RAM/NAS/SRPI stubbed with deterministic
    values); checks NAS enters directly, CI is theta-invariant, and the cohort
    reference is the mean over subjects of per-subject awake means with an unequal
    number of runs per subject."""
    import impact_pipeline.synergy_ci as sc

    rng = np.random.RandomState(0)
    iim = {}
    for si, subj in enumerate(("s1", "s2", "s3")):
        for ses in ("awake", "deep"):
            n_runs = 2 if subj == "s1" else 1
            for run in range(1, n_runs + 1):
                d = tmp_path / subj / ses / "audio"
                d.mkdir(parents=True, exist_ok=True)
                p = d / f"{subj}_run-{run}_schaefer400_ts.npy"
                np.save(p, rng.randn(96, 6))
                undefined = subj == "s3" and ses == "deep"
                iim[str(p)] = (
                    {"defined": False, "undefined_reason": "t", "canonical": np.nan,
                     "raw": np.nan}
                    if undefined
                    else {"defined": True, "canonical": 0.3 + 0.1 * si, "raw": 0.3}
                )
            rest = tmp_path / subj / ses / "rest"
            rest.mkdir(parents=True, exist_ok=True)
            np.save(rest / f"{subj}_run-1_schaefer400_ts.npy", rng.randn(96, 6))

    def _stub(row):
        return lambda ts, **k: float(np.abs(ts[row, :10]).mean()) + 0.1

    monkeypatch.setattr(sc, "compute_RAM", _stub(0))
    monkeypatch.setattr(sc, "compute_SRPI", _stub(1))
    monkeypatch.setattr(sc, "compute_NAS", _stub(2))
    df = compute_synergy_ci(
        str(tmp_path), "schaefer400", [0.2, 0.5, 0.8], sessions=("awake", "deep"),
        tr=0.1, pdi_params=_PDI_PARAMS, pdi_require_explicit_params=True,
        pdi_require_strict_baseline=True, nas_params=_NAS_PARAMS,
        srpi_params=_SRPI_PARAMS, iim_precomputed_by_path=iim,
    )
    awake = df[df.session == "awake"]
    for k in COMPS:
        expected = awake.groupby("subject")[k].mean().mean()
        assert df.attrs["ci_references"][k] == pytest.approx(expected)
    refs = df.attrs["ci_references"]
    # One run per (subject, session, RAM value); CI identical across theta.
    for _, grp in df.groupby(["subject", "session", "RAM"]):
        assert grp["theta"].nunique() == 3
        ci = grp["CI"].to_numpy(dtype=float)
        assert np.all(np.isnan(ci)) or np.allclose(ci, ci[0])
    np.testing.assert_allclose(df["NAS_norm"], df["NAS"] / refs["NAS"])
    bad = df[(df.subject == "s3") & (df.session == "deep")]
    assert bad["CI"].isna().all() and (bad["CI_missing"] == "IIM").all()
    good = df[df["CI_defined"]]
    assert len(good) == len(df) - len(bad)
    for _, row in good.iterrows():
        norms = [max(row[k] / refs[k], 0.0) for k in COMPS]
        expected = 0.0 if min(norms) <= 0 else float(np.prod(norms) ** 0.2)
        assert row["CI"] == pytest.approx(expected, rel=1e-9, abs=1e-12)
