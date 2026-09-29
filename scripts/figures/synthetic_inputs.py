#!/usr/bin/env python
"""
Small deterministic synthetic inputs for every figure script (layout checks
and tests only; never figure content for the paper).

``write_all(out_dir, seed=0)`` writes, under ``out_dir``: ``audit/`` (from
``scripts/audit_aggregation.py`` with a tiny sensitivity run),
``null_calibration_rates.csv``, ``null_calibration_verdicts.csv``,
``iim_validation.csv``, ``sweep.csv`` (construct-scale components, long),
``witnesses.csv`` (wide statuses), ``witness_components.csv`` and
``witness_verdicts.csv`` (long layout of the evaluator), ``patchwork.csv``,
``rule_summary.csv``, ``rule_cases.csv``, ``audit_decisions.csv``, ``power/`` (from
``scripts/necessity_power.py``) and ``recovery/`` (from
``scripts/simulate_rule_recovery.py``), and returns their paths.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent
for _p in (_HERE, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PRINCIPLES = ("RAM", "PDI", "NAS", "IIM", "SRPI")
KNOBS = {"RAM": "eta", "PDI": "K", "NAS": "g_b", "IIM": "c_int", "SRPI": "e"}


def null_rates(rng):
    rows = []
    for kind in ("ar1", "pink", "surrogate_iid", "surrogate_linear"):
        for t in (1200, 2400):
            for p in PRINCIPLES:
                if kind == "surrogate_linear" and p in ("NAS", "IIM"):
                    continue
                n = 100
                k = int(rng.binomial(n, 0.05 + 0.02 * rng.standard_normal() ** 2))
                phat = k / n
                rows.append({"null_kind": kind, "n_time": t, "n_nodes": 8,
                             "principle": p, "n": n, "false_present_rate": phat,
                             "false_present_lo": max(0.0, phat - 0.04),
                             "false_present_hi": min(1.0, phat + 0.05),
                             "expected_rate_exchangeable_v1": 0.063})
    return pd.DataFrame(rows)


def iim_table(rng):
    exact = {"independent": 0.0, "ring": 0.04975, "all_to_all": 0.04009,
             "star": 0.03583, "chain": 0.01919}
    rows = []
    for s, ex in exact.items():
        for t in (500, 2000, 8000):
            for seed in range(3):
                err = rng.standard_normal() * 0.05 / np.sqrt(t / 100)
                rows.append({"system": s, "n_time": t, "seed": seed,
                             "delta_psi_exact": ex,
                             "delta_psi_est": max(0.0, ex + abs(err) * 0.5 + err)})
    for c in np.linspace(0, 1, 6):
        for seed in range(3):
            rows.append({"system": "ring_sweep", "n_time": 2000, "seed": seed,
                         "delta_psi_exact": np.nan, "delta_psi_est": np.nan,
                         "coupling": c, "iim_z": 8 * c + rng.standard_normal()})
    return pd.DataFrame(rows)


KNOBS_BY_SWITCH = {k: p for p, k in KNOBS.items()}
SWITCH_WITNESS = {"eta": "W_RAM_no_plasticity", "K": "W_PDI_single_attractor",
                  "g_b": "W_NAS_no_workspace", "c_int": "W_IIM_feedforward",
                  "e": "W_SRPI_no_efference"}


def _status(c, se=0.1):
    if not np.isfinite(c):
        return "UNDEFINED", "INVALID_ANCHORS"
    lo, hi = c - 1.833 * se, c + 1.833 * se
    if lo > 0.25:
        return "PRESENT", ""
    if hi < 0.10:
        return "ABSENT", ""
    return "UNDEFINED", "INCONCLUSIVE"


def component_table(rng):
    """Construct-scale components (long; the evaluator's ``components.csv``
    layout): sweeps and witnesses of two families; RAM and PDI have no
    anchor (``c`` undefined), as on the bench."""
    rows = []
    anchored = ("NAS", "IIM", "SRPI")
    for fam in ("A", "C"):
        for seed in range(4):
            for p, knob in KNOBS.items():
                for lv in np.linspace(0, 1, 5):
                    for q in PRINCIPLES:
                        eff = 1.2 * lv if q == p else 0.2 * lv * (q == "IIM")
                        c = (0.1 + eff + 0.1 * rng.standard_normal()
                             if q in anchored else np.nan)
                        st, why = _status(c)
                        rows.append({"family": fam, "design": "sweep",
                                     "witness_id": None, "bearer_mode": "system",
                                     "seed": seed, "sweep_knob": knob,
                                     "sweep_level": lv, "principle": q, "c": c,
                                     "se_c": 0.1, "status": st,
                                     "status_reason": why})
            witnesses = [("PC_nominal", None)] + [
                (w, KNOBS_BY_SWITCH[k]) for k, w in SWITCH_WITNESS.items()]
            for wid, off in witnesses:
                for q in PRINCIPLES:
                    c = (1.0 - (0.9 if q == off else 0.0) + 0.1 * rng.standard_normal()
                         if q in anchored else np.nan)
                    st, why = _status(c)
                    rows.append({"family": fam, "design": "witnesses",
                                 "witness_id": wid, "bearer_mode": "system",
                                 "seed": seed, "sweep_knob": None,
                                 "sweep_level": None, "principle": q, "c": c,
                                 "se_c": 0.1, "status": st, "status_reason": why,
                                 "intended_bits": "11111" if off is None else
                                 "".join("0" if x == off else "1"
                                         for x in PRINCIPLES)})
    return pd.DataFrame(rows)


def verdict_table(components):
    """One verdict per witness run under the anchored set (Kleene AND)."""
    w = components[components["design"] == "witnesses"]
    rows = []
    for (fam, wid, seed), sub in w.groupby(["family", "witness_id", "seed"]):
        st = sub[sub["principle"].isin(("NAS", "IIM", "SRPI"))]["status"].tolist()
        verdict = ("EXCLUDED" if "ABSENT" in st else "MPC_CONSISTENT"
                   if all(x == "PRESENT" for x in st) else "UNDETERMINED")
        rows.append({"task_id": f"{wid}-{fam}-{seed}", "family": fam,
                     "design": "witnesses", "witness_id": wid,
                     "bearer_mode": "system", "seed": seed, "verdict": "UNDETERMINED",
                     "verdict_val": verdict})
    return pd.DataFrame(rows)


def audit_decisions(rng):
    """Per-system decisions of two rules (``audit_decisions.csv`` layout)."""
    rows = []
    classes = (["all_present"] * 30 + ["single_deficit:NAS"] * 20
               + ["witness:W_IIM_feedforward"] * 20 + ["multi_deficit:3"] * 10)
    for rule, p_cons in (("impact_c", (0.3, 0.02)), ("union", (0.9, 0.6))):
        for scen in ("none", "RAM+SRPI_missing"):
            for i, cl in enumerate(classes):
                pos = cl == "all_present"
                u = rng.random()
                dec = ("MPC_CONSISTENT" if u < p_cons[0 if pos else 1] else
                       "UNDETERMINED" if u < 0.9 else "EXCLUDED")
                rows.append({"task_id": f"t{i}", "class": cl, "rule": rule,
                             "scenario": scen, "label_noise": 0.0,
                             "decision": dec})
    return pd.DataFrame(rows)


def witness_results(rng):
    wit = {"positive_control": "11111", "no_RAM": "01111", "no_PDI": "10111",
           "no_NAS": "11011", "no_IIM": "11101", "no_SRPI": "11110",
           "null_ar1": "00000", "patchwork": "11111"}
    rows = []
    for w, bits in wit.items():
        for seed in range(4):
            row = {"witness_id": w, "seed": seed, "intended_bits": "b" + bits}
            st = []
            for j, p in enumerate(PRINCIPLES):
                # construct scale (v2): c ~ 1 with the mechanism, ~ 0 without;
                # PRESENT if c - 1.645 se > 0.25, ABSENT if c + 1.645 se < 0.10
                se = 0.1
                c = (1.0 if bits[j] == "1" else 0.0) + se * rng.standard_normal()
                row[f"{p}_c"] = c
                lo, hi = c - 1.645 * se, c + 1.645 * se
                st.append("PRESENT" if lo > 0.25 else ("ABSENT" if hi < 0.10
                                                       else "UNDEFINED"))
                row[f"{p}_status"] = st[-1]
            row["verdict"] = ("EXCLUDED" if "ABSENT" in st else
                              "MPC_CONSISTENT" if all(s == "PRESENT" for s in st)
                              else "UNDETERMINED")
            rows.append(row)
    return pd.DataFrame(rows)


def patchwork_results(rng):
    rows = []
    for c in np.linspace(0, 1, 6):
        for seed in range(4):
            coherent = rng.random() < c
            rows.append({"coupling": c, "seed": seed,
                         "verdict": "MPC_CONSISTENT" if coherent else "UNDETERMINED",
                         "MPC_reason": "" if coherent else "SOURCE_INCOHERENT"})
    return pd.DataFrame(rows)


def rule_tables(rng):
    rules = ("impact_c", "geometric_mean_uncapped", "weakest_link", "count_3",
             "logistic", "IIM_only")
    scen = ("complete", "missing_20pct", "label_noise_10pct")
    summ, cases = [], []
    for i, r in enumerate(rules):
        for s in scen:
            cov = 1.0 if r != "impact_c" else 0.7 - 0.1 * scen.index(s)
            summ.append({"rule": r, "scenario": s, "coverage": cov,
                         "selective_accuracy": 0.95 - 0.05 * i - 0.03 * scen.index(s)})
        for k in range(60):
            conf = rng.random()
            cases.append({"rule": r, "scenario": "complete", "confidence": conf,
                          "correct": bool(rng.random() < 0.6 + 0.35 * conf - 0.05 * i)})
    return pd.DataFrame(summ), pd.DataFrame(cases)


def write_all(out_dir, seed=0) -> dict:
    import audit_aggregation
    import necessity_power
    import simulate_rule_recovery

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = {}
    audit_aggregation.run_audit(out / "audit", n_base=16, n_boot=10, n_units=12,
                                grid_n=15, seed=seed)
    paths["audit_dir"] = out / "audit"
    rates = null_rates(rng)
    paths["null_rates"] = out / "null_calibration_rates.csv"
    rates.to_csv(paths["null_rates"], index=False)
    paths["null_verdicts"] = out / "null_calibration_verdicts.csv"
    pd.DataFrame([{"null_kind": "ar1", "n_time": 1200, "n_nodes": 8,
                   "necessity_set": "IIM,NAS,PDI,RAM,SRPI", "n": 100,
                   "consistent_rate": 0.0}]).to_csv(paths["null_verdicts"], index=False)
    comp = component_table(rng)
    for name, df in (("iim", iim_table(rng)), ("sweep", comp),
                     ("witnesses", witness_results(rng)),
                     ("witness_components", comp),
                     ("witness_verdicts", verdict_table(comp)),
                     ("patchwork", patchwork_results(rng)),
                     ("audit_decisions", audit_decisions(rng))):
        paths[name] = out / f"{name}.csv"
        df.to_csv(paths[name], index=False)
    summ, cases = rule_tables(rng)
    paths["rule_summary"] = out / "rule_summary.csv"
    paths["rule_cases"] = out / "rule_cases.csv"
    summ.to_csv(paths["rule_summary"], index=False)
    cases.to_csv(paths["rule_cases"], index=False)
    params = dict(necessity_power.DEFAULTS)
    params.update(tau=(0.05, 0.10), label_noise=(0.0, 0.05), base_rate=(0.5,),
                  coverage=(1.0,), false_absent=(0.0,), pi_viol=(0.25,),
                  n_values=tuple(range(10, 201, 10)))
    necessity_power.run(out / "power", params)
    paths["power_dir"] = out / "power"
    simulate_rule_recovery.run(out / "recovery", p_true=(-4.0, 0.0, 1.0),
                               reliability=(0.7, 1.0), rho=(0.3,), n=(60,), reps=4,
                               seed=seed)
    paths["recovery_dir"] = out / "recovery"
    return {k: str(v) for k, v in paths.items()}


def render_all(paths: dict, out_dir) -> dict:
    """Render every figure from the inputs of :func:`write_all`."""
    import fig3_aggregation
    import fig5_null_calibration
    import fig6_iim_validation
    import fig7_crosstalk
    import fig8_witnesses
    import fig9_rule_audit
    import fig10_power

    return {
        "fig3": fig3_aggregation.make_figure(paths["audit_dir"], out_dir),
        "fig5": fig5_null_calibration.make_figure(paths["null_rates"], out_dir,
                                                  paths["null_verdicts"]),
        "fig6": fig6_iim_validation.make_figure(paths["iim"], out_dir),
        "fig7": fig7_crosstalk.make_figure(paths["sweep"], out_dir),
        "fig8": fig8_witnesses.make_figure(paths["witnesses"], out_dir,
                                           paths["patchwork"]),
        "fig9": fig9_rule_audit.make_figure(paths["rule_summary"], out_dir,
                                            cases_path=paths["rule_cases"],
                                            decisions_path=paths.get(
                                                "audit_decisions")),
        "fig10": fig10_power.make_figure(paths["power_dir"], paths["recovery_dir"],
                                         out_dir),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", default=None,
                    help="also render every figure into this directory")
    args = ap.parse_args(argv)
    paths = write_all(args.out, args.seed)
    for k, v in paths.items():
        print(f"{k}: {v}")
    if args.render:
        for k, v in render_all(paths, args.render).items():
            print(f"{k}: {v['pdf']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
