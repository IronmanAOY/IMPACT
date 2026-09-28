#!/usr/bin/env python
"""
Power of the symmetric three-outcome necessity criteria (spec V2-7).

For one principle and ``n`` determinate report-positive episodes the ABSENT
count is ``Bin(n, q_pos)`` with (``impact_pipeline.necessity``)

    q_pos = (1 - lambda) q(pi_c) + lambda q(pi_u),   q(pi) = pi s + (1 - pi) f

where ``lambda`` is the label noise (report-positive episodes that are truly
not conscious), ``pi_c`` / ``pi_u`` the true absence rates in conscious /
unconscious episodes, ``s`` the ABSENT sensitivity and ``f`` the false-ABSENT
rate of the estimator. The variation check uses the determinate
report-negative episodes, ``n_neg = floor(n (1 - b) / b)`` for base rate
``b`` (fraction of report-positive episodes), with ABSENT rate
``q_neg = (1 - lambda_neg) q(pi_u) + lambda_neg q(pi_c)``. UNDEFINED statuses
are assumed missing at random: coverage ``kappa`` converts ``n`` into the
total number of episodes ``N = ceil(n / (b kappa))``.

Two scenarios per parameter set: *necessity holds* (``pi_c = pi_nec``,
correct outcome SUPPORTED) and *necessity violated* (``pi_c = pi_viol``,
correct outcome FALSIFIED). Outcome probabilities are exact (binomial);
``--mc-reps`` adds a seeded Monte-Carlo check through
``necessity.necessity_from_statuses`` on simulated episodes.

Outputs (``--out``): ``necessity_thresholds.csv`` (``n, k_F, k_S`` per tau),
``necessity_power.csv`` (outcome probabilities per scenario and n),
``necessity_min_n.csv`` (minimum n and N per target power: first n reaching
the target and the stable n from which it stays reached), and
``necessity_power.json`` (parameters, provenance).

Example::

    python scripts/necessity_power.py --out outputs/necessity_power
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from impact_pipeline import necessity as nc  # noqa: E402

POWER_VERSION = "necessity-power/1.0.0"
DEFAULTS = {
    "tau": (0.05, 0.10),
    "alpha": 0.05,
    "pi_nec": (0.0,),
    "pi_viol": (0.15, 0.25),
    "pi_unconscious": (0.6,),
    "label_noise": (0.0, 0.02, 0.05),
    "label_noise_negative": 0.0,
    "base_rate": (0.3, 0.5),
    "coverage": (0.7, 1.0),
    "sensitivity": (0.9,),
    "false_absent": (0.0, 0.02),
    "target_power": (0.8, 0.9),
    "n_values": tuple(range(5, 401, 5)),
}


def scenario_grid(params) -> list:
    """Parameter sets (dicts) of the full factorial grid."""
    keys = ("tau", "pi_nec", "pi_viol", "pi_unconscious", "label_noise",
            "base_rate", "coverage", "sensitivity", "false_absent")
    grid = []
    for vals in itertools.product(*(params[k] for k in keys)):
        d = dict(zip(keys, (float(v) for v in vals)))
        d["alpha"] = float(params["alpha"])
        d["label_noise_negative"] = float(params["label_noise_negative"])
        grid.append(d)
    return grid


def _rates(sc, truth):
    pi_c = sc["pi_nec"] if truth == "necessity_holds" else sc["pi_viol"]
    q_pos = nc.observed_absent_rate(
        pi_c, sc["pi_unconscious"], sc["label_noise"], sc["sensitivity"],
        sc["false_absent"],
    )
    q_neg = nc.observed_absent_rate(
        sc["pi_unconscious"], pi_c, sc["label_noise_negative"], sc["sensitivity"],
        sc["false_absent"],
    )
    return pi_c, q_pos, q_neg


def n_negative_for(n, base_rate):
    return int(math.floor(int(n) * (1.0 - float(base_rate)) / float(base_rate)))


def power_table(params, thresholds=None) -> pd.DataFrame:
    """Exact outcome probabilities for every scenario, truth and n."""
    n_values = np.asarray(sorted(set(int(v) for v in params["n_values"])), dtype=int)
    if thresholds is None:
        thresholds = {
            tau: nc.necessity_threshold_table(n_values, tau, params["alpha"])
            for tau in params["tau"]
        }
    rows = []
    for sid, sc in enumerate(scenario_grid(params)):
        th = thresholds[sc["tau"]].set_index("n").loc[n_values]
        k_f = th["k_F"].to_numpy()
        k_s = th["k_S"].to_numpy()
        n_neg = np.asarray([n_negative_for(n, sc["base_rate"]) for n in n_values])
        for truth in ("necessity_holds", "necessity_violated"):
            pi_c, q_pos, q_neg = _rates(sc, truth)
            p_f = np.where(k_f <= n_values, binom.sf(k_f - 1, n_values, q_pos), 0.0)
            p_low = np.where(k_s >= 0, binom.cdf(k_s, n_values, q_pos), 0.0)
            p_var = np.where(n_neg > 0, 1.0 - (1.0 - q_neg) ** n_neg, 0.0)
            p_s = p_low * p_var
            correct = "SUPPORTED" if truth == "necessity_holds" else "FALSIFIED"
            for i, n in enumerate(n_values):
                rows.append({
                    "scenario_id": sid, **sc, "truth": truth, "pi_conscious": pi_c,
                    "q_positive": q_pos, "q_negative": q_neg, "n": int(n),
                    "n_negative": int(n_neg[i]),
                    "N_total": int(math.ceil(n / (sc["base_rate"] * sc["coverage"]))),
                    "k_F": int(k_f[i]), "k_S": int(k_s[i]),
                    "P_SUPPORTED": float(p_s[i]), "P_FALSIFIED": float(p_f[i]),
                    "P_INDETERMINATE": float(max(0.0, 1.0 - p_s[i] - p_f[i])),
                    "P_varies": float(p_var[i]),
                    "correct_outcome": correct,
                    "power": float(p_s[i] if correct == "SUPPORTED" else p_f[i]),
                    "error": float(p_f[i] if correct == "SUPPORTED" else p_s[i]),
                    "asymptotically_correct": bool(
                        q_pos < sc["tau"] if correct == "SUPPORTED"
                        else q_pos > sc["tau"]
                    ),
                })
    return pd.DataFrame(rows)


def min_n_table(power: pd.DataFrame, targets) -> pd.DataFrame:
    """
    Per scenario, truth and target power: ``n_first`` (smallest n on the grid
    with power >= target) and ``n_stable`` (smallest n from which the power
    stays >= target up to the end of the grid; discreteness makes the power
    saw-toothed), with the implied total episodes ``N_total``. NaN when the
    target is not reached on the grid.
    """
    keys = ["scenario_id", "truth"]
    rows = []
    for (sid, truth), sub in power.groupby(keys, sort=True):
        sub = sub.sort_values("n")
        first = sub.iloc[0]
        base = {k: first[k] for k in sub.columns
                if k not in ("n", "n_negative", "N_total", "k_F", "k_S",
                             "P_SUPPORTED", "P_FALSIFIED", "P_INDETERMINATE",
                             "P_varies", "power", "error")}
        pw = sub["power"].to_numpy()
        ns = sub["n"].to_numpy()
        for target in targets:
            ok = pw >= float(target)
            n_first = int(ns[np.argmax(ok)]) if ok.any() else np.nan
            tail_ok = np.flip(np.logical_and.accumulate(np.flip(ok)))
            n_stable = int(ns[np.argmax(tail_ok)]) if tail_ok.any() else np.nan

            def _N(n):
                if not np.isfinite(n):
                    return np.nan
                return int(math.ceil(n / (base["base_rate"] * base["coverage"])))

            rows.append({**base, "target_power": float(target),
                         "n_first": n_first, "n_stable": n_stable,
                         "N_total_first": _N(n_first), "N_total_stable": _N(n_stable),
                         "max_power_on_grid": float(pw.max())})
    return pd.DataFrame(rows)


def simulate_outcomes(sc, truth, n, reps, seed) -> dict:
    """
    Monte-Carlo outcome frequencies: episodes are simulated from the
    generative model (labels, true absence, misclassification, UNDEFINED
    missing at random) and decided by ``necessity_from_statuses``. The
    report-positive/negative determinate counts are fixed at ``n`` and
    ``n_negative_for(n, base_rate)`` as in the exact computation.
    """
    rng = np.random.default_rng(seed)
    pi_c = sc["pi_nec"] if truth == "necessity_holds" else sc["pi_viol"]
    n_neg = n_negative_for(n, sc["base_rate"])
    counts = {o: 0 for o in nc.NECESSITY_OUTCOMES}

    def _statuses(m, pi_own, pi_other, lam):
        wrong_class = rng.random(m) < lam
        pi = np.where(wrong_class, pi_other, pi_own)
        absent = rng.random(m) < pi
        called = np.where(absent, rng.random(m) < sc["sensitivity"],
                          rng.random(m) < sc["false_absent"])
        return np.where(called, "ABSENT", "PRESENT")

    for _ in range(int(reps)):
        st_pos = _statuses(n, pi_c, sc["pi_unconscious"], sc["label_noise"])
        st_neg = _statuses(n_neg, sc["pi_unconscious"], pi_c,
                           sc["label_noise_negative"])
        st = np.r_[st_pos, st_neg]
        lab = np.r_[np.ones(n, dtype=bool), np.zeros(n_neg, dtype=bool)]
        res = nc.necessity_from_statuses(st, lab, sc["tau"], sc["alpha"])
        counts[res["outcome"]] += 1
    return {f"MC_{k}": v / float(reps) for k, v in counts.items()}


def _provenance():
    try:
        from impact_pipeline.provenance import collect_code_version

        info = collect_code_version(REPO_ROOT)
    except Exception as exc:  # noqa: BLE001 - provenance must not fail a run
        info = {"code_version": "unknown", "error": str(exc)}
    info["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return info


def _floats(text):
    return tuple(float(v) for v in str(text).split(",") if v.strip())


def _ints(text):
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            a, b, s = (int(v) for v in part.split(":"))
            out.extend(range(a, b + 1, s))
        else:
            out.append(int(part))
    return tuple(out)


def run(out_dir, params, mc_reps=0, seed=0, mc_n=None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_values = sorted(set(int(v) for v in params["n_values"]))
    thresholds = {tau: nc.necessity_threshold_table(range(1, max(n_values) + 1), tau,
                                                    params["alpha"])
                  for tau in params["tau"]}
    pd.concat(thresholds.values(), ignore_index=True).to_csv(
        out / "necessity_thresholds.csv", index=False)
    power = power_table(params, thresholds)
    if int(mc_reps) > 0:
        n_check = set(mc_n or [n_values[len(n_values) // 2]])
        mc_rows = []
        for sid, sc in enumerate(scenario_grid(params)):
            for truth in ("necessity_holds", "necessity_violated"):
                for n in sorted(n_check):
                    mc_seed = seed + 1000 * sid + int(truth == "necessity_violated")
                    mc = simulate_outcomes(sc, truth, int(n), mc_reps, mc_seed)
                    mc_rows.append({"scenario_id": sid, "truth": truth, "n": int(n),
                                    **mc})
        power = power.merge(pd.DataFrame(mc_rows), on=["scenario_id", "truth", "n"],
                            how="left")
    power.to_csv(out / "necessity_power.csv", index=False)
    min_n = min_n_table(power, params["target_power"])
    min_n.to_csv(out / "necessity_min_n.csv", index=False)
    support_min = {str(t): nc.min_n_for_support(t, params["alpha"])
                   for t in params["tau"]}
    summary = {
        "version": POWER_VERSION,
        "params": {k: list(v) if isinstance(v, tuple) else v
                   for k, v in params.items()},
        "n_scenarios": len(scenario_grid(params)),
        "min_n_support_reachable": support_min,
        "mc_reps": int(mc_reps),
        "seed": int(seed),
        "decision_rule": "FALSIFIED if CP lower > tau; SUPPORTED if CP upper < tau and"
                         " >=1 ABSENT among report-negative; one-sided alpha each",
        "provenance": _provenance(),
    }
    with open(out / "necessity_power.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return {"summary": summary, "power": power, "min_n": min_n}


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", default=",".join(map(str, DEFAULTS["tau"])))
    ap.add_argument("--alpha", type=float, default=DEFAULTS["alpha"])
    ap.add_argument("--pi-nec", default=",".join(map(str, DEFAULTS["pi_nec"])))
    ap.add_argument("--pi-viol", default=",".join(map(str, DEFAULTS["pi_viol"])))
    ap.add_argument("--pi-unconscious",
                    default=",".join(map(str, DEFAULTS["pi_unconscious"])))
    ap.add_argument("--label-noise",
                    default=",".join(map(str, DEFAULTS["label_noise"])))
    ap.add_argument("--label-noise-negative", type=float,
                    default=DEFAULTS["label_noise_negative"])
    ap.add_argument("--base-rate", default=",".join(map(str, DEFAULTS["base_rate"])))
    ap.add_argument("--coverage", default=",".join(map(str, DEFAULTS["coverage"])))
    ap.add_argument("--sensitivity",
                    default=",".join(map(str, DEFAULTS["sensitivity"])))
    ap.add_argument("--false-absent",
                    default=",".join(map(str, DEFAULTS["false_absent"])))
    ap.add_argument("--target-power",
                    default=",".join(map(str, DEFAULTS["target_power"])))
    ap.add_argument("--n-values", default="5:400:5",
                    help="comma list and/or start:stop:step ranges")
    ap.add_argument("--mc-reps", type=int, default=0)
    ap.add_argument("--mc-n", default=None, help="n values checked by Monte Carlo")
    ap.add_argument("--seed", type=int, default=0)
    return ap


def params_from_args(args) -> dict:
    return {
        "tau": _floats(args.tau),
        "alpha": float(args.alpha),
        "pi_nec": _floats(args.pi_nec),
        "pi_viol": _floats(args.pi_viol),
        "pi_unconscious": _floats(args.pi_unconscious),
        "label_noise": _floats(args.label_noise),
        "label_noise_negative": float(args.label_noise_negative),
        "base_rate": _floats(args.base_rate),
        "coverage": _floats(args.coverage),
        "sensitivity": _floats(args.sensitivity),
        "false_absent": _floats(args.false_absent),
        "target_power": _floats(args.target_power),
        "n_values": _ints(args.n_values),
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    params = params_from_args(args)
    res = run(args.out, params, mc_reps=args.mc_reps, seed=args.seed,
              mc_n=_ints(args.mc_n) if args.mc_n else None)
    print(json.dumps(res["summary"]["min_n_support_reachable"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
