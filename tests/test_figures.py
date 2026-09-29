"""Every paper-1 figure script renders PDF + PNG from small synthetic inputs and
records its code hash and input hashes; the statistics the figures compute
match the preregistered evaluation."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIG_DIR = Path(__file__).resolve().parents[1] / "scripts" / "figures"
if str(FIG_DIR) not in sys.path:
    sys.path.insert(0, str(FIG_DIR))

import fig5_null_calibration  # noqa: E402
import figure_common as fc  # noqa: E402
import synthetic_inputs  # noqa: E402

FIGURES = {
    "fig3": "fig3_aggregation",
    "fig5": "fig5_null_calibration",
    "fig6": "fig6_iim_validation",
    "fig7": "fig7_crosstalk",
    "fig8": "fig8_witnesses",
    "fig9": "fig9_rule_audit",
    "fig10": "fig10_power",
}


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    base = tmp_path_factory.mktemp("figs")
    paths = synthetic_inputs.write_all(base / "inputs", seed=0)
    outputs = synthetic_inputs.render_all(paths, base / "out")
    return paths, outputs


@pytest.mark.parametrize("key", sorted(FIGURES))
def test_figure_outputs_and_provenance(rendered, key):
    _, outputs = rendered
    out = outputs[key]
    pdf, png = Path(out["pdf"]), Path(out["png"])
    assert pdf.read_bytes()[:5] == b"%PDF-" and pdf.stat().st_size > 1000
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert Path(out["data"]).is_file()
    prov = json.loads(Path(out["provenance"]).read_text())
    script = FIG_DIR / f"{FIGURES[key]}.py"
    assert prov["figure_code_sha256"] == fc.figure_code_hash(script)
    assert prov["figure_script"] == f"scripts/figures/{script.name}"
    for item in prov["inputs"]:
        p = Path(item["path"])
        assert item["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert prov["figure_code_sha256"].encode() in pdf.read_bytes()


def test_code_hash_changes_with_the_script(tmp_path):
    src = (FIG_DIR / "fig5_null_calibration.py").read_text()
    a = tmp_path / "fig5_null_calibration.py"
    a.write_text(src)
    h1 = fc.figure_code_hash(a)
    a.write_text(src + "\n# changed\n")
    assert fc.figure_code_hash(a) != h1


def test_pdf_output_is_deterministic(rendered, tmp_path):
    import matplotlib

    paths, _ = rendered
    before = dict(matplotlib.rcParams)
    a = fig5_null_calibration.make_figure(paths["null_rates"], tmp_path / "a")
    assert dict(matplotlib.rcParams) == before  # the figure style does not leak
    b = fig5_null_calibration.make_figure(paths["null_rates"], tmp_path / "b")
    assert Path(a["pdf"]).read_bytes() == Path(b["pdf"]).read_bytes()


def test_missing_columns_fail_clearly(tmp_path):
    bad = tmp_path / "rates.csv"
    pd.DataFrame({"null_kind": ["ar1"], "principle": ["PDI"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="lacks required columns"):
        fig5_null_calibration.make_figure(bad, tmp_path / "out")
    with pytest.raises(FileNotFoundError):
        fig5_null_calibration.make_figure(tmp_path / "nope.csv", tmp_path / "out")


def test_cli_entry_points(rendered, tmp_path):
    paths, _ = rendered
    import fig7_crosstalk
    import fig10_power

    assert fig7_crosstalk.main(["--components", paths["sweep"], "--out",
                                str(tmp_path)]) == 0
    assert fig7_crosstalk.main(["--sweep", paths["sweep"], "--out",
                                str(tmp_path / "alias")]) == 0
    assert fig10_power.main(["--power-dir", paths["power_dir"], "--recovery-dir",
                             paths["recovery_dir"], "--out", str(tmp_path),
                             "--tau", "0.1"]) == 0
    assert (tmp_path / "fig7_crosstalk.pdf").is_file()
    assert (tmp_path / "fig10_power.png").is_file()


def test_witness_figure_reads_the_evaluator_layout(rendered, tmp_path):
    """Long statuses + verdicts (evaluator layout) and the wide layout both
    render; the data table carries the status fractions per cell."""
    import fig8_witnesses

    paths, _ = rendered
    out = fig8_witnesses.make_figure(paths["witness_components"], tmp_path,
                                     paths["patchwork"], paths["witness_verdicts"])
    data = pd.read_csv(out["data"])
    st = data[data["panel"] == "status"]
    assert set(st["family"]) == {"A", "C"}
    fr = st[["PRESENT", "UNDEFINED", "ABSENT", "NO_SCALE"]].sum(axis=1)
    assert np.allclose(fr, 1.0)
    # RAM and PDI have no construct scale in the synthetic bench
    assert (st[st["principle"].isin(["RAM", "PDI"])]["NO_SCALE"] == 1.0).all()
    v = data[data["panel"] == "verdicts"]
    assert np.allclose(v[["MPC_CONSISTENT", "UNDETERMINED", "EXCLUDED"]]
                       .sum(axis=1), 1.0)


def test_rule_audit_hc9_statistics_follow_the_evaluator(rendered):
    """fig9's HC9 quantities use the evaluator's classes: positive controls
    (all_present, witness:PC_nominal) and single-deficit classes with a target
    in the necessity set, scenario none, label noise 0."""
    import fig9_rule_audit

    dec = pd.DataFrame({
        "rule": ["r"] * 8, "scenario": ["none"] * 7 + ["RAM+SRPI_missing"],
        "label_noise": [0.0] * 8,
        "class": ["all_present", "all_present", "witness:PC_nominal",
                  "single_deficit:NAS", "single_deficit:NAS",
                  "witness:W_RAM_no_plasticity", "single_deficit:PDI",
                  "all_present"],
        "decision": ["MPC_CONSISTENT", "UNDETERMINED", "MPC_CONSISTENT",
                     "MPC_CONSISTENT", "EXCLUDED", "MPC_CONSISTENT",
                     "MPC_CONSISTENT", "UNDETERMINED"],
    })
    st = fig9_rule_audit.hc9_statistics(dec, ("NAS", "IIM", "SRPI")).iloc[0]
    assert st["positive_consistent_none"] == pytest.approx(2 / 3)
    # RAM and PDI deficits are outside the anchored set
    assert st["max_single_deficit_consistent"] == pytest.approx(0.5)
    assert st["worst_class"] == "single_deficit:NAS"
    assert st["positive_consistent_missing"] == 0.0


def test_crosstalk_figure_uses_paired_witness_contrasts(rendered, tmp_path):
    import fig7_crosstalk

    paths, _ = rendered
    out = fig7_crosstalk.make_figure(paths["sweep"], tmp_path)
    data = pd.read_csv(out["data"])
    heat = data[data["panel"].isin(["A", "B"])]
    own = heat[heat["own"].astype(str) == "True"]
    # the synthetic switch removes 0.9 of its own anchored principle
    assert (own[own["principle"].isin(["NAS", "IIM", "SRPI"])]["median_delta_c"]
            > 0.6).all()
    assert set(heat["family"]) == {"A", "C"}


def test_renderer_covers_every_figure_script():
    import render_paper1_figures as rp

    stems = {p.stem for p in FIG_DIR.glob("fig[0-9]*.py")}
    assert stems == set(rp.FIGURES)
    assert set(FIGURES.values()) == stems
    layout = rp.inputs(Path("results"))
    assert all(str(v).startswith("results") for v in layout.values())
