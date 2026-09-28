#!/usr/bin/env python
"""
Parameter and model recovery of the aggregation exponent ``p`` of the MPC
degree (the auxiliary hypothesis H2; Wilson & Collins 2019, *eLife* 8:e49547,
"Ten simple rules for the computational modeling of behavioral data").

Generative model per unit (episode or system):

- latent components ``c_j = expit(sigma_c z_j)``, ``z ~ N(0, R)`` with
  inter-component correlation ``rho`` (equi-correlated ``R``), ``j = 1..5``;
- outcome from the weighted power mean ``M_p(c)`` (equal weights, cap
  ``--cap``): continuous ``Y = M_p(c) + e`` with ``e`` scaled so that
  ``M_p(c)`` explains ``--signal-r2`` of the outcome variance, or binary
  ``P(Y = 1) = expit(k (M_p(c) - median))`` (``--outcome binary``);
- measurement: observed ``c^_j = c_j + u_j`` with classical error chosen for
  a component reliability ``r = var(c) / (var(c) + var(u))``; observed
  components are floored at ``--floor`` before aggregation (a power mean with
  ``p <= 0`` is 0 at any zero).

Fit: ``p`` on a grid (plus the weakest link ``p = -inf``) by profile maximum
likelihood of the outcome regression on ``M_p(c^)`` (OLS for continuous,
logistic IRLS for binary outcomes); a profile-likelihood interval
``{p : 2 (l_max - l(p)) <= chi2_1(0.95)}`` on the finite grid. Recovery
statistics per cell (true p, reliability, rho, N): mean / median estimate,
bias and RMSE (finite estimates; the weakest link counts as the grid
minimum), interval coverage and width, the fraction of intervals touching the
grid edge, and the model-recovery confusion of aggregation classes
(min-like p <= -3, harmonic-like, geometric-like |p| < 0.5, arithmetic-like
p >= 0.5).

Outputs (``--out``): ``rule_recovery_cells.csv``, ``rule_recovery_confusion.csv``,
optional ``rule_recovery_replicates.csv`` and ``rule_recovery.json``.

Example::

    python scripts/simulate_rule_recovery.py --out outputs/rule_recovery --reps 200
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
from scipy.special import expit
from scipy.stats import chi2

REPO_ROOT = Path(__file__).resolve().parents[1]
RECOVERY_VERSION = "rule-recovery/1.0.0"
N_COMPONENTS = 5
DEFAULT_P_GRID = tuple(np.round(np.arange(-8.0, 4.0001, 0.25), 4))
CLASSES = ("min_like", "harmonic_like", "geometric_like", "arithmetic_like")
CHI2_95 = float(chi2.ppf(0.95, 1))


def p_class(p) -> str:
    """Aggregation class of an exponent (``-inf`` is min-like)."""
    p = float(p)
    if p <= -3.0:
        return "min_like"
    if p <= -0.5:
        return "harmonic_like"
    if p < 0.5:
        return "geometric_like"
    return "arithmetic_like"


def power_means(X, p_values, cap=1.0):
    """
    ``(len(p_values), n)`` equal-weight power means of the rows of ``X``
    (positive entries), capped at ``cap``; ``p = 0`` geometric, ``-inf``
    minimum, ``+inf`` maximum.
    """
    X = np.asarray(X, dtype=float)
    if cap is not None:
        X = np.minimum(X, float(cap))
    out = np.empty((len(p_values), X.shape[0]))
    logx = np.log(X)
    for i, p in enumerate(p_values):
        p = float(p)
        if p == -math.inf:
            out[i] = X.min(axis=1)
        elif p == math.inf:
            out[i] = X.max(axis=1)
        elif p == 0.0:
            out[i] = np.exp(logx.mean(axis=1))
        else:
            # log-sum-exp form, stable for large |p|
            a = p * logx
            m = a.max(axis=1, keepdims=True)
            lme = m[:, 0] + np.log(np.mean(np.exp(a - m), axis=1))
            out[i] = np.exp(lme / p)
    return out


def component_sd(sigma_c=1.0, n_nodes=80) -> float:
    """SD of ``expit(sigma_c Z)``, ``Z ~ N(0, 1)``, by Gauss-Hermite quadrature."""
    x, w = np.polynomial.hermite_e.hermegauss(int(n_nodes))
    w = w / w.sum()
    v = expit(float(sigma_c) * x)
    m = float(np.sum(w * v))
    return float(math.sqrt(np.sum(w * (v - m) ** 2)))


def simulate_units(n, p_true, rho, reliability, rng, *, sigma_c=1.0, cap=1.0,
                   outcome="continuous", signal_r2=0.8, slope=10.0, floor=0.01):
    """One synthetic data set: observed components ``(n, 5)`` and outcome ``y``."""
    if not 0.0 < float(reliability) <= 1.0:
        raise ValueError("reliability must be in (0, 1]")
    if not -1.0 / (N_COMPONENTS - 1) < float(rho) < 1.0:
        raise ValueError("rho must keep the correlation matrix positive definite")
    k = N_COMPONENTS
    cov = float(rho) * np.ones((k, k)) + (1 - float(rho)) * np.eye(k)
    z = rng.multivariate_normal(np.zeros(N_COMPONENTS), cov, size=int(n))
    c = expit(float(sigma_c) * z)
    m = power_means(c, [p_true], cap)[0]
    if outcome == "continuous":
        sd_m = float(np.std(m))
        noise_sd = sd_m * math.sqrt((1.0 - signal_r2) / signal_r2) if sd_m > 0 else 1.0
        y = m + noise_sd * rng.standard_normal(int(n))
    elif outcome == "binary":
        prob = expit(float(slope) * (m - np.median(m)))
        y = (rng.random(int(n)) < prob).astype(float)
    else:
        raise ValueError("outcome must be 'continuous' or 'binary'")
    err_sd = component_sd(sigma_c) * math.sqrt((1.0 - reliability) / reliability)
    c_obs = c + err_sd * rng.standard_normal(c.shape)
    return np.maximum(c_obs, float(floor)), y


def _profile_continuous(M, y):
    yc = y - y.mean()
    syy = float(yc @ yc)
    Mc = M - M.mean(axis=1, keepdims=True)
    sxx = np.einsum("ij,ij->i", Mc, Mc)
    sxy = Mc @ yc
    with np.errstate(invalid="ignore", divide="ignore"):
        rss = np.where(sxx > 0, syy - sxy ** 2 / sxx, syy)
    rss = np.maximum(rss, 1e-300)
    return -0.5 * y.size * np.log(rss / y.size)


def _profile_binary(M, y, n_iter=30, ridge=1e-6):
    """Logistic log-likelihood of y on [1, M_p] per row of M (vectorised IRLS)."""
    P, n = M.shape
    mu_x = M.mean(axis=1, keepdims=True)
    sd_x = M.std(axis=1, keepdims=True)
    X = (M - mu_x) / np.where(sd_x > 0, sd_x, 1.0)
    beta = np.zeros((P, 2))
    for _ in range(int(n_iter)):
        eta = beta[:, :1] + beta[:, 1:] * X
        pr = expit(eta)
        w = pr * (1 - pr)
        r = y[None, :] - pr
        g0 = r.sum(axis=1)
        g1 = (r * X).sum(axis=1)
        h00 = w.sum(axis=1) + ridge
        h01 = (w * X).sum(axis=1)
        h11 = (w * X * X).sum(axis=1) + ridge
        det = h00 * h11 - h01 ** 2
        beta[:, 0] += (h11 * g0 - h01 * g1) / det
        beta[:, 1] += (-h01 * g0 + h00 * g1) / det
    eta = beta[:, :1] + beta[:, 1:] * X
    ll = np.sum(y[None, :] * eta - np.logaddexp(0.0, eta), axis=1)
    return ll


def fit_exponent(C_obs, y, p_grid=DEFAULT_P_GRID, cap=1.0, outcome="continuous",
                 include_min=True) -> dict:
    """
    Profile-likelihood estimate of ``p``: the grid value (or ``-inf``) with the
    largest outcome log-likelihood, and the profile interval on the finite
    grid. Returns ``p_hat``, ``ll``, ``ci_lo``/``ci_hi``, ``ci_at_edge``.
    """
    grid = [float(p) for p in p_grid]
    cands = grid + ([-math.inf] if include_min else [])
    M = power_means(C_obs, cands, cap)
    y = np.asarray(y, dtype=float)
    ll = _profile_continuous(M, y) if outcome == "continuous" else _profile_binary(M, y)
    best = int(np.argmax(ll))
    p_hat = cands[best]
    ll_fin = ll[: len(grid)]
    inside = 2.0 * (ll[best] - ll_fin) <= CHI2_95
    idx = np.flatnonzero(inside)
    if idx.size:
        lo, hi = grid[idx[0]], grid[idx[-1]]
        edge = bool(idx[0] == 0 or idx[-1] == len(grid) - 1)
    else:
        lo = hi = float("nan")
        edge = True
    return {"p_hat": p_hat, "ll_max": float(ll[best]), "ci_lo": lo, "ci_hi": hi,
            "ci_at_edge": edge, "grid_min": grid[0], "grid_max": grid[-1]}


def run_cell(p_true, reliability, rho, n, reps, seed, *, p_grid=DEFAULT_P_GRID,
             cap=1.0, outcome="continuous", signal_r2=0.8, floor=0.01,
             keep_replicates=False) -> tuple:
    """Recovery statistics of one design cell (``reps`` seeded replicates)."""
    rng = np.random.default_rng(seed)
    grid_min = float(min(p_grid))
    rows = []
    for r in range(int(reps)):
        C, y = simulate_units(n, p_true, rho, reliability, rng, cap=cap,
                              outcome=outcome, signal_r2=signal_r2, floor=floor)
        fit = fit_exponent(C, y, p_grid, cap, outcome)
        p_hat = fit["p_hat"]
        p_fin = grid_min if p_hat == -math.inf else p_hat
        pt_fin = grid_min if float(p_true) == -math.inf else float(p_true)
        covered = bool(
            math.isfinite(fit["ci_lo"]) and fit["ci_lo"] - 1e-9 <= pt_fin
            <= fit["ci_hi"] + 1e-9
        )
        rows.append({
            "replicate": r, "p_hat": p_hat, "p_hat_finite": p_fin,
            "ci_lo": fit["ci_lo"], "ci_hi": fit["ci_hi"],
            "ci_at_edge": fit["ci_at_edge"],
            "covered": covered, "true_class": p_class(p_true),
            "hat_class": p_class(p_hat),
        })
    rep = pd.DataFrame(rows)
    pt_fin = grid_min if float(p_true) == -math.inf else float(p_true)
    err = rep["p_hat_finite"].to_numpy() - pt_fin
    cell = {
        "p_true": float(p_true), "reliability": float(reliability), "rho": float(rho),
        "n": int(n), "reps": int(reps), "outcome": outcome, "cap": cap,
        "signal_r2": float(signal_r2), "seed": int(seed),
        "mean_p_hat": float(rep["p_hat_finite"].mean()),
        "median_p_hat": float(rep["p_hat_finite"].median()),
        "bias": float(err.mean()),
        "rmse": float(math.sqrt(np.mean(err ** 2))),
        "coverage": float(rep["covered"].mean()),
        "mean_ci_width": float((rep["ci_hi"] - rep["ci_lo"]).mean()),
        "ci_at_edge_rate": float(rep["ci_at_edge"].mean()),
        "class_accuracy": float((rep["true_class"] == rep["hat_class"]).mean()),
        "min_selected_rate": float(np.mean(np.isneginf(rep["p_hat"].to_numpy()))),
    }
    if keep_replicates:
        for k in ("p_true", "reliability", "rho", "n"):
            rep[k] = cell[k]
    return cell, rep


def cell_seed(seed, *keys) -> int:
    text = "|".join([str(int(seed))] + [repr(float(k)) for k in keys])
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def run(out_dir, *, p_true=(-4.0, -1.0, 0.0, 1.0), reliability=(0.5, 0.7, 0.9, 1.0),
        rho=(0.0, 0.3, 0.6), n=(50, 100, 200, 500), reps=200, seed=0,
        p_grid=DEFAULT_P_GRID, cap=1.0, outcome="continuous", signal_r2=0.8,
        floor=0.01, save_replicates=False) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cells, reps_all = [], []
    for pt, rel, rh, nn in itertools.product(p_true, reliability, rho, n):
        cs = cell_seed(seed, pt, rel, rh, nn)
        cell, rep = run_cell(pt, rel, rh, nn, reps, cs, p_grid=p_grid, cap=cap,
                             outcome=outcome, signal_r2=signal_r2, floor=floor,
                             keep_replicates=True)
        cells.append(cell)
        reps_all.append(rep)
    cells_df = pd.DataFrame(cells)
    cells_df.to_csv(out / "rule_recovery_cells.csv", index=False)
    rep_df = pd.concat(reps_all, ignore_index=True)
    conf = (rep_df.groupby(["reliability", "n", "true_class", "hat_class"])
            .size().rename("count").reset_index())
    conf["rate"] = conf["count"] / conf.groupby(
        ["reliability", "n", "true_class"])["count"].transform("sum")
    conf.to_csv(out / "rule_recovery_confusion.csv", index=False)
    if save_replicates:
        rep_df.to_csv(out / "rule_recovery_replicates.csv", index=False)
    try:
        from impact_pipeline.provenance import collect_code_version

        prov = collect_code_version(REPO_ROOT)
    except Exception as exc:  # noqa: BLE001 - provenance must not fail a run
        prov = {"code_version": "unknown", "error": str(exc)}
    prov["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary = {
        "version": RECOVERY_VERSION,
        "design": {"p_true": list(p_true), "reliability": list(reliability),
                   "rho": list(rho), "n": list(n), "reps": int(reps),
                   "p_grid": [float(p) for p in p_grid], "cap": cap,
                   "outcome": outcome, "signal_r2": signal_r2, "floor": floor,
                   "sigma_c": 1.0, "component_sd": component_sd(1.0)},
        "seed": int(seed),
        "n_cells": int(len(cells)),
        "provenance": prov,
    }
    with open(out / "rule_recovery.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return {"summary": summary, "cells": cells_df, "confusion": conf}


def _floats(text):
    vals = []
    for v in str(text).split(","):
        v = v.strip()
        if v:
            vals.append(-math.inf if v in ("-inf", "min") else float(v))
    return tuple(vals)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--p-true", default="-4,-1,0,1")
    ap.add_argument("--reliability", default="0.5,0.7,0.9,1.0")
    ap.add_argument("--rho", default="0,0.3,0.6")
    ap.add_argument("--n", default="50,100,200,500")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap", type=float, default=1.0,
                    help="cap of the power mean (<= 0 disables it)")
    ap.add_argument("--outcome", choices=("continuous", "binary"), default="continuous")
    ap.add_argument("--signal-r2", type=float, default=0.8)
    ap.add_argument("--floor", type=float, default=0.01)
    ap.add_argument("--save-replicates", action="store_true")
    args = ap.parse_args(argv)
    res = run(
        args.out, p_true=_floats(args.p_true), reliability=_floats(args.reliability),
        rho=_floats(args.rho), n=tuple(int(v) for v in _floats(args.n)),
        reps=args.reps, seed=args.seed, cap=args.cap if args.cap > 0 else None,
        outcome=args.outcome, signal_r2=args.signal_r2, floor=args.floor,
        save_replicates=args.save_replicates,
    )
    print(res["cells"][["p_true", "reliability", "n", "bias", "rmse", "coverage"]]
          .groupby(["p_true", "reliability"]).mean(numeric_only=True).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
