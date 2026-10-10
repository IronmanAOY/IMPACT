#!/usr/bin/env python
"""
Render every paper-1 data figure from the confirmatory MPC-Bench results.

Reads the result folders of the confirmatory run plan below ``--results``
(``c1`` null calibration and family B, ``c2`` families A and C with the
frozen evaluator's tables, ``c3`` rule audit, ``c4`` aggregation audit, power
and rule recovery), calls each figure script and writes, into ``--out``, the
PDF, PNG, data table and provenance record of every figure plus
``FIGURES_MANIFEST.json``: per figure the script, its code hash
(``figure_code_sha256``), the git commit and dirty flag, the SHA-256 of every
input and output, and the figure's place in the paper. The figures are
descriptive; no decision of the preregistration depends on them.

Example::

    python scripts/figures/render_paper1_figures.py \\
        --results outputs/paper1_mpcbench \\
        --out outputs/paper1_figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import figure_common as fc  # noqa: E402

MANIFEST_VERSION = "paper1-figures/1.0.0"


def inputs(results: Path) -> dict:
    r = Path(results)
    return {
        "null_rates": r / "c1/null_calibration/null_calibration_rates.csv",
        "null_verdicts": r / "c1/null_calibration/null_calibration_verdicts.csv",
        "null_replicates": r / "c1/null_calibration/null_calibration_replicates.csv",
        "iim": r / "c1/iim_validation/iim_validation.csv",
        "components": r / "c2/hypotheses/components.csv",
        "verdicts": r / "c2/hypotheses/verdicts.csv",
        "patchwork": r / "c2/summary/patchwork_runs.csv",
        "audit_summary": r / "c3/summary/audit_rule_coverage_risk_aurc.csv",
        "audit_curves": r / "c3/summary/audit_risk_coverage_curves.csv",
        "audit_decisions": r / "c3/audit/audit_decisions.csv",
        "aggregation": r / "c4/audit_aggregation",
        "power": r / "c4/necessity_power/component_default_grid",
        "recovery": r / "c4/rule_recovery/continuous_default",
    }


# (stem, place in the paper, hypotheses it shows, caption draft)
FIGURES = {
    "fig7_crosstalk": (
        "main (fig:crosstalk)", ["HC3", "registry (c)"],
        "Cross-talk and dose-response on the construct scale. A, B: median "
        "paired change of c when one mechanism is removed (families A, C). "
        "C-E: mean c (+- SE, 20 seeds) against the own switch; lines: z = 0.25, "
        "delta = 0.10."),
    "fig8_witnesses": (
        "main (fig:witnesses)", ["HC4", "HC5", "HC7", "HC8"],
        "Statuses (A, B) and verdicts under the anchored necessity set (C, D) "
        "of the MPC-Bench witnesses, 20 seeds per family; E: graded patchwork "
        "with per-principle bearers (10 seeds per level), with and without the "
        "single-source rule."),
    "fig9_rule_audit": (
        "main (fig:ruleaudit)", ["HC9"],
        "Rule audit on estimated statuses (primary audit, anchored necessity "
        "set, label noise 0). A: risk-coverage; B: coverage, selective risk and "
        "error rates per rule; C: HC9 quantities with the preregistered bounds."),
    "fig5_null_calibration": (
        "supplement", ["HC1"],
        "Null calibration: false-PRESENT rates per null family, series length "
        "and size (exact 95% intervals; bound 0.07), status composition per "
        "principle and verdicts on 1,600 null systems."),
    "fig6_iim_validation": (
        "supplement", ["HC2"],
        "IIM against exact transition matrices (family B, 20 seeds): "
        "convergence of the error with T per cut mode, sampled vs exact values, "
        "calibrated margins of independent units and the feedforward star, and "
        "the ring coupling sweep."),
    "fig3_aggregation": (
        "supplement", ["legacy-index audit"],
        "Legacy compensatory index (compute_CI of release 1.0) against capped "
        "geometric mean, weakest link and arithmetic mean; compensation and "
        "definedness cases; implied component floors."),
    "fig10_power": (
        "supplement (fig:supp-power)", ["power of the paper-2 criteria", "H2"],
        "Power of the symmetric necessity criterion (script default grid) and "
        "recovery of the aggregation exponent under measurement error."),
}


def render(results, out_dir) -> dict:
    import fig3_aggregation
    import fig5_null_calibration
    import fig6_iim_validation
    import fig7_crosstalk
    import fig8_witnesses
    import fig9_rule_audit
    import fig10_power

    p = {k: str(v) for k, v in inputs(results).items()}
    missing = [k for k, v in p.items() if not Path(v).exists()]
    if missing:
        raise FileNotFoundError(f"missing result inputs: {missing}")
    out = {
        "fig5_null_calibration": fig5_null_calibration.make_figure(
            p["null_rates"], out_dir, p["null_verdicts"],
            replicates_path=p["null_replicates"]),
        "fig6_iim_validation": fig6_iim_validation.make_figure(p["iim"], out_dir),
        "fig7_crosstalk": fig7_crosstalk.make_figure(p["components"], out_dir),
        "fig8_witnesses": fig8_witnesses.make_figure(
            p["components"], out_dir, p["patchwork"], p["verdicts"]),
        "fig9_rule_audit": fig9_rule_audit.make_figure(
            p["audit_summary"], out_dir, risk_coverage_path=p["audit_curves"],
            decisions_path=p["audit_decisions"], audit="primary_Nanch", noise=0.0),
        "fig3_aggregation": fig3_aggregation.make_figure(p["aggregation"], out_dir),
        "fig10_power": fig10_power.make_figure(p["power"], p["recovery"], out_dir),
    }
    return out


def write_manifest(outputs: dict, out_dir) -> Path:
    from impact_pipeline.provenance import collect_code_version

    code = collect_code_version(fc.REPO_ROOT)
    figures = []
    for stem, paths in outputs.items():
        prov = json.loads(Path(paths["provenance"]).read_text(encoding="utf-8"))
        place, hyps, caption = FIGURES[stem]
        figures.append({
            "stem": stem,
            "paper": place,
            "shows": hyps,
            "caption_draft": caption,
            "script": prov["figure_script"],
            "figure_code_sha256": prov["figure_code_sha256"],
            "git_sha": prov.get("git_sha"),
            "git_dirty": prov.get("git_dirty"),
            "inputs": prov["inputs"],
            "outputs": {k: {"file": Path(v).name, "sha256": fc.sha256_file(v)}
                        for k, v in paths.items()},
        })
    manifest = {
        "version": MANIFEST_VERSION,
        "renderer": "scripts/figures/render_paper1_figures.py",
        "code_version": code.get("code_version"),
        "git_sha": code.get("git_sha"),
        "git_dirty": code.get("git_dirty"),
        "note": ("Descriptive figures of the confirmatory MPC-Bench results "
                 "(frozen code, tag mpcbench-freeze-v1); no preregistered "
                 "decision depends on them. figure_code_sha256 = SHA-256 over "
                 "the figure script and figure_common.py."),
        "figures": figures,
    }
    path = Path(out_dir) / "FIGURES_MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default=str(fc.REPO_ROOT / "outputs"
                                             / "paper1_mpcbench"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    outputs = render(args.results, args.out)
    path = write_manifest(outputs, args.out)
    for stem, paths in outputs.items():
        print(f"{stem}: {paths['pdf']}")
    print(f"manifest: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
