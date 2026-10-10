#!/usr/bin/env python
"""
Audit of the legacy Consciousness Index (CI) aggregation.

The legacy ``compute_CI`` is loaded *verbatim* from git commit ``4c74466``
(``git show 4c74466:src/impact_pipeline/mpc_metrics.py``) into an in-memory
module, so the audit reproduces what the released code computed, not a
re-implementation. Imports the legacy module needs but the current
environment lacks are replaced by inert stubs (recorded in the output); the
CI function itself only uses NumPy. Without git (e.g. an HPC tarball), a saved
copy can be given with ``--legacy-source``; it is verified against the git blob
id of the audited file.

Outputs (``--out``):

- ``legacy_compensation.csv``: the compensation / definedness demonstrations
  (e.g. ``CI(0.01, 10, 10, 10, 10) > 1``; ``CI(NaN, 1, 1, 1, 1) == CI(0, 1, 1,
  1, 1) == 0``), with the capped geometric mean and the weakest link as
  comparators.
- ``implied_floors.csv``: the component floor a CI threshold ``t`` implies,
  ``x_j >= (t / M**(1 - w_j))**(1 / w_j)`` for the weighted geometric mean with
  the other components at ``M`` (``t**(1/w_j)`` at the reference, ``M = 1``),
  and the power-mean analogue for ``p = -1, 1``.
- ``aggregation_grid.csv``: the aggregate over a grid of two components (the
  other three at the reference) for the legacy CI (evaluated with the legacy
  function), the capped geometric mean, the weakest link, the arithmetic mean
  and the harmonic mean (input of Figure 3).
- ``sensitivity_units.csv`` / ``sensitivity_indices.csv``: Saisana-style
  uncertainty and sensitivity analysis (Saisana, Saltelli & Tarantola 2005,
  *J. R. Stat. Soc. A* 168:307-323) of a component panel: weights
  (Dirichlet), aggregation exponent, reference anchors (log-normal), missing-
  data handling and the cap are sampled; per unit the rank distribution and
  the decision flip rate; per factor group the first-order and total Sobol
  indices of the average absolute rank shift ``R_S`` (Saltelli 2010 and
  Jansen estimators).
- ``audit_aggregation.json``: summary and provenance.

Example::

    python scripts/audit_aggregation.py --out outputs/audit_aggregation
"""
from __future__ import annotations

import argparse
import ast
import atexit
import builtins
import hashlib
import json
import linecache
import math
import shutil
import subprocess
import sys
import tempfile
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from impact_pipeline.bench.rules import power_mean  # noqa: E402

AUDIT_VERSION = "audit-aggregation/1.0.0"
LEGACY_COMMIT = "4c74466"
LEGACY_PATH = "src/impact_pipeline/mpc_metrics.py"
# git blob id of LEGACY_PATH at LEGACY_COMMIT (verifies --legacy-source copies).
LEGACY_BLOB_SHA1 = "26c106b4c5c8fa9e1bf1890fa0b85defdcd941d9"
COMPONENTS = ("RAM", "PDI", "NAS", "IIM", "SRPI")
LOAD_MODES = ("auto", "module", "ast")

# Sensitivity factor groups (Saisana et al. 2005) and their uniform dimension.
FACTOR_GROUPS = (
    ("weights", 5),
    ("aggregation", 1),
    ("reference", 5),
    ("missing_handling", 1),
    ("cap", 1),
)
AGGREGATION_LEVELS = (-math.inf, -1.0, 0.0, 1.0)
MISSING_LEVELS = ("legacy_zero", "exclude", "skip")
CAP_LEVELS = (None, 1.0)


# --------------------------------------------------------------------------
# loading the legacy compute_CI
# --------------------------------------------------------------------------
def git_blob_sha1(data: bytes) -> str:
    """git's blob id: sha1 of ``b"blob <len>\\0" + data``."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def read_legacy_source(commit=LEGACY_COMMIT, path=LEGACY_PATH, repo_root=REPO_ROOT,
                       source_file=None) -> tuple:
    """
    Legacy source text and its identity. From git (``git show
    <commit>:<path>``) or, with ``source_file``, from a saved copy that must
    match :data:`LEGACY_BLOB_SHA1` when the audited commit/path are the
    defaults. Returns ``(text, info)``.
    """
    if source_file is not None:
        data = Path(source_file).read_bytes()
        origin = f"file:{Path(source_file).resolve()}"
    else:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{commit}:{path}"],
            capture_output=True, check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"git show {commit}:{path} failed "
                f"({proc.stderr.decode(errors='replace').strip()}); "
                "run from a checkout with history or pass --legacy-source"
            )
        data = proc.stdout
        origin = f"git:{commit}:{path}"
    blob = git_blob_sha1(data)
    if (commit, path) == (LEGACY_COMMIT, LEGACY_PATH) and blob != LEGACY_BLOB_SHA1:
        raise RuntimeError(
            f"legacy source blob {blob} does not match the audited blob "
            f"{LEGACY_BLOB_SHA1} ({origin})"
        )
    info = {
        "origin": origin,
        "commit": commit,
        "path": path,
        "git_blob_sha1": blob,
        "sha256": hashlib.sha256(data).hexdigest(),
        "n_bytes": len(data),
    }
    return data.decode("utf-8"), info


class _Placeholder:
    """Stand-in for a name the legacy module imports but the env lacks."""

    def __init__(self, qualname):
        self._qualname = qualname

    def __call__(self, *args, **kwargs):
        raise RuntimeError(f"legacy dependency {self._qualname} is not available")

    def __getattr__(self, name):
        return _Placeholder(f"{self._qualname}.{name}")


def _stub_module(name, real=None, stubbed=None):
    mod = types.ModuleType(name)
    if real is not None:
        mod.__dict__.update(real.__dict__)

    def __getattr__(attr):
        if stubbed is not None:
            stubbed.append(f"{name}.{attr}")
        return _Placeholder(f"{name}.{attr}")

    mod.__getattr__ = __getattr__
    return mod


def _tolerant_import(report):
    real_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        try:
            mod = real_import(name, globals, locals, fromlist, level)
        except ImportError as exc:
            report["stubbed_modules"].append(f"{name} ({type(exc).__name__})")
            return _stub_module(name, stubbed=report["stubbed_names"])
        missing = [n for n in (fromlist or ()) if n != "*" and not hasattr(mod, n)]
        if missing:
            report["stubbed_names"].extend(f"{name}.{n}" for n in missing)
            return _stub_module(name, real=mod, stubbed=report["stubbed_names"])
        return mod

    return _import


_LEGACY_TMP = []


def _legacy_tmpdir() -> Path:
    """Process-lifetime temp dir holding the legacy module file (numba's
    ``cache=True`` decorators need a real source file); removed at exit."""
    if not _LEGACY_TMP:
        path = Path(tempfile.mkdtemp(prefix="impact_legacy_"))
        atexit.register(shutil.rmtree, path, ignore_errors=True)
        _LEGACY_TMP.append(path)
    return _LEGACY_TMP[0]


def _exec_module(src, name, import_fn):
    path = _legacy_tmpdir() / f"{name}.py"
    path.write_text(src, encoding="utf-8")
    filename = str(path)
    mod = types.ModuleType(name)
    mod.__file__ = filename
    env = dict(builtins.__dict__)
    env["__import__"] = import_fn
    mod.__dict__["__builtins__"] = env
    exec(compile(src, filename, "exec"), mod.__dict__)
    return mod


def _ast_module(src, name, filename):
    """Only ``compute_CI`` (plus NumPy) from the legacy source."""
    tree = ast.parse(src, filename=filename)
    funcs = [
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "compute_CI"
    ]
    if len(funcs) != 1:
        raise RuntimeError("compute_CI not found in the legacy source")
    fn_src = ast.get_source_segment(src, funcs[0])
    mod = types.ModuleType(name)
    mod.__file__ = filename
    mod.__dict__.update({"np": np, "math": math})
    linecache.cache[filename] = (len(fn_src), None, fn_src.splitlines(True), filename)
    exec(compile(fn_src, filename, "exec"), mod.__dict__)
    return mod


def load_legacy_compute_ci(commit=LEGACY_COMMIT, repo_root=REPO_ROOT, mode="auto",
                           source_file=None):
    """
    ``(compute_CI, info)``: the legacy function executed from the legacy
    source in a fresh temporary module (``mode="module"``: the whole file,
    written to a temp dir and executed with imports the environment lacks
    replaced by stubs; it is not registered in ``sys.modules``; ``"ast"``:
    only the ``compute_CI`` definition with NumPy; ``"auto"``: module,
    falling back to ast). ``info`` records the source identity, the load mode
    and any stubbed imports.
    """
    if mode not in LOAD_MODES:
        raise ValueError(f"mode must be one of {LOAD_MODES}")
    src, info = read_legacy_source(commit, LEGACY_PATH, repo_root, source_file)
    name = f"impact_legacy_{commit}_mpc_metrics"
    filename = f"<legacy {info['origin']}>"
    report = {"stubbed_modules": [], "stubbed_names": []}
    fallback_error = None
    mod = None
    if mode in ("auto", "module"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mod = _exec_module(src, name, _tolerant_import(report))
            used = "module"
        except Exception as exc:  # noqa: BLE001 - recorded, then ast fallback
            if mode == "module":
                raise
            fallback_error = f"{type(exc).__name__}: {exc}"
    if mod is None:
        mod = _ast_module(src, name, filename)
        used = "ast"
    fn = getattr(mod, "compute_CI")
    info.update(
        load_mode=used,
        module_name=name,
        stubbed_modules=sorted(set(report["stubbed_modules"])),
        stubbed_names=sorted(set(report["stubbed_names"])),
        fallback_error=fallback_error,
    )
    return fn, info


# --------------------------------------------------------------------------
# compensation demonstration
# --------------------------------------------------------------------------
def demonstration_cases():
    """Labelled component vectors (RAM, PDI, NAS, IIM, SRPI; reference = 1)."""
    nan = float("nan")
    return [
        ("reference", (1, 1, 1, 1, 1), None,
         "all components at the reference"),
        ("compensation", (0.01, 10, 10, 10, 10), None,
         "RAM at 1% of the reference, the rest at 10x: CI above the reference"),
        ("compensation_to_reference", (1e-4, 10, 10, 10, 10), None,
         "RAM at 0.01% of the reference still gives CI = 1"),
        ("undefined_component", (nan, 1, 1, 1, 1), None,
         "RAM undefined (NaN): hard zero"),
        ("measured_zero", (0, 1, 1, 1, 1), None,
         "RAM measured as exactly 0: the same hard zero"),
        ("declared_undefined", (1, 1, 1, 1, 1), {"RAM": False},
         "RAM declared undefined through `defined`: hard zero"),
        ("below_null_clipped", (-0.5, 1, 1, 1, 1), None,
         "RAM below the null (negative): clipped to 0, the same hard zero"),
        ("tiny_positive", (1e-6, 1, 1, 1, 1), None,
         "RAM at 1e-6 of the reference: CI 0.063, versus 0 for NaN"),
        ("eps_discontinuity", (1e-13, 1, 1, 1, 1), None,
         "RAM = 1e-13: CI jumps from 0 (at 0) to (1e-13 + 1e-12)**(1/5)"),
    ]


def _comparator(vals, p):
    x = np.asarray(vals, dtype=float)
    return float(power_mean(x[None, :], p=p, cap=1.0)[0])


def compensation_table(legacy_ci) -> pd.DataFrame:
    rows = []
    for label, vals, defined, note in demonstration_cases():
        kw = {"defined": defined} if defined is not None else {}
        ci = float(legacy_ci(*vals, **kw))
        x = np.asarray(vals, dtype=float)
        if defined:
            x = x.copy()
            for k, ok in defined.items():
                if not ok:
                    x[COMPONENTS.index(k)] = np.nan
        row = {"case": label}
        row.update({k: float(v) for k, v in zip(COMPONENTS, vals)})
        row.update(
            legacy_CI=ci,
            capped_geometric=_comparator(x, 0.0),
            weakest_link=_comparator(x, -np.inf),
            above_reference=bool(ci > 1.0 + 1e-12),
            note=note,
        )
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# implied floors
# --------------------------------------------------------------------------
def implied_floor(t, w, others=1.0, p=0.0):
    """
    Smallest value of one component (weight ``w``) for which the weighted
    power mean (exponent ``p``; ``p = 0`` geometric) reaches ``t`` when the
    other components (total weight ``1 - w``) all equal ``others``. ``0`` means
    the threshold is met whatever the component (full compensation); ``inf``
    means it cannot be met.
    """
    t, w, m, p = float(t), float(w), float(others), float(p)
    if not (0 < w <= 1) or t <= 0 or m < 0:
        raise ValueError("need 0 < w <= 1, t > 0 and others >= 0")
    if w == 1.0:
        return t
    if p == 0.0:
        if m == 0.0:
            return math.inf
        return (t / m ** (1.0 - w)) ** (1.0 / w)
    rest = (1.0 - w) * (m ** p if m > 0 else (math.inf if p < 0 else 0.0))
    rhs = (t ** p - rest) / w
    if p > 0:
        return 0.0 if rhs <= 0 else rhs ** (1.0 / p)
    return math.inf if rhs <= 0 else rhs ** (1.0 / p)


def implied_floor_table(thresholds=(0.1, 0.25, 0.5, 0.75, 0.9),
                        weights=(0.1, 0.2, 0.25, 1.0 / 3.0, 0.5),
                        others=(1.0, 2.0, 10.0),
                        powers=(0.0, -1.0, 1.0)) -> pd.DataFrame:
    rows = []
    for p in powers:
        for m in others:
            for w in weights:
                for t in thresholds:
                    f = implied_floor(t, w, m, p)
                    rows.append({
                        "p": p, "others": m, "weight": w, "threshold": t,
                        "floor": f,
                        "floor_formula": "(t/M^(1-w))^(1/w)" if p == 0 else
                        "((t^p-(1-w)M^p)/w)^(1/p)",
                        "fully_compensable": bool(f == 0.0),
                    })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# aggregation grid (Figure 3)
# --------------------------------------------------------------------------
GRID_RULES = ("legacy_CI", "capped_geometric", "weakest_link", "arithmetic", "harmonic")


def aggregation_grid(legacy_ci, lo=0.0, hi=2.0, n=41,
                     pair=("RAM", "PDI")) -> pd.DataFrame:
    """Aggregates over an ``n x n`` grid of two components (others at 1)."""
    a = np.linspace(lo, hi, int(n))
    ia, ib = COMPONENTS.index(pair[0]), COMPONENTS.index(pair[1])
    A, B = np.meshgrid(a, a, indexing="ij")
    X = np.ones((A.size, 5))
    X[:, ia], X[:, ib] = A.ravel(), B.ravel()
    legacy = np.asarray([float(legacy_ci(*row)) for row in X])
    vals = {
        "legacy_CI": legacy,
        "capped_geometric": power_mean(X, 0.0, cap=1.0),
        "weakest_link": power_mean(X, -np.inf, cap=1.0),
        "arithmetic": power_mean(X, 1.0, cap=1.0),
        "harmonic": power_mean(X, -1.0, cap=1.0),
    }
    frames = []
    for rule in GRID_RULES:
        frames.append(pd.DataFrame({
            "rule": rule, "x_name": pair[0], "y_name": pair[1],
            "x": X[:, ia], "y": X[:, ib], "value": vals[rule],
        }))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Saisana-style uncertainty and sensitivity analysis
# --------------------------------------------------------------------------
def synthetic_panel(n_units=40, seed=0, missing_rate=0.05) -> pd.DataFrame:
    """
    Deterministic demonstration panel of reference-normalised components:
    log-normal around the reference with inter-component correlation 0.3 and
    ``missing_rate`` undefined entries.
    """
    rng = np.random.default_rng(seed)
    cov = 0.3 * np.ones((5, 5)) + 0.7 * np.eye(5)
    z = rng.multivariate_normal(np.zeros(5), cov, size=int(n_units))
    x = np.exp(0.6 * z - 0.2)
    x[rng.random(x.shape) < float(missing_rate)] = np.nan
    df = pd.DataFrame(x, columns=list(COMPONENTS))
    df.insert(0, "unit", [f"u{i:03d}" for i in range(int(n_units))])
    return df


def _group_slices():
    out, start = {}, 0
    for name, dim in FACTOR_GROUPS:
        out[name] = slice(start, start + dim)
        start += dim
    return out, start


def _config_from_uniforms(u, kappa=20.0, ref_sigma=0.2):
    from scipy.stats import gamma, norm

    sl, _ = _group_slices()
    g = gamma.ppf(np.clip(u[sl["weights"]], 1e-12, 1 - 1e-12), float(kappa))
    w = g / g.sum()
    p = AGGREGATION_LEVELS[min(int(u[sl["aggregation"]][0] * len(AGGREGATION_LEVELS)),
                               len(AGGREGATION_LEVELS) - 1)]
    ref_u = np.clip(u[sl["reference"]], 1e-12, 1 - 1e-12)
    ref = np.exp(float(ref_sigma) * norm.ppf(ref_u))
    miss = MISSING_LEVELS[min(int(u[sl["missing_handling"]][0] * len(MISSING_LEVELS)),
                              len(MISSING_LEVELS) - 1)]
    cap = CAP_LEVELS[min(int(u[sl["cap"]][0] * len(CAP_LEVELS)), len(CAP_LEVELS) - 1)]
    return {"weights": w, "p": p, "reference": ref, "missing": miss, "cap": cap}


def aggregate_panel(X, weights, p=0.0, reference=None, missing="legacy_zero", cap=None):
    """
    Composite scores of a panel ``X`` (units x 5, reference-normalised) under
    one configuration: components divided by ``reference``; missing entries
    count as 0 (``legacy_zero``), make the score undefined (``exclude``) or
    are skipped with the weights renormalised over the observed components
    (``skip``); weighted power mean with exponent ``p`` and optional cap.
    """
    X = np.asarray(X, dtype=float)
    ref = np.ones(5) if reference is None else np.asarray(reference, dtype=float)
    Y = X / ref[None, :]
    w = np.asarray(weights, dtype=float)
    if missing == "legacy_zero":
        return power_mean(np.where(np.isfinite(Y), Y, 0.0), p, cap=cap, weights=w)
    if missing == "exclude":
        return power_mean(Y, p, cap=cap, weights=w)
    if missing != "skip":
        raise ValueError(f"missing must be one of {MISSING_LEVELS}")
    out = np.full(Y.shape[0], np.nan)
    for i, row in enumerate(Y):
        ok = np.isfinite(row) & (w > 0)
        if not ok.any():
            continue
        full = np.where(ok, row, 1.0)
        out[i] = power_mean(full[None, :], p, cap=cap, weights=np.where(ok, w, 0.0))[0]
    return out


def _ranks(scores):
    s = np.asarray(scores, dtype=float)
    r = np.full(s.shape, np.nan)
    ok = np.isfinite(s)
    r[ok] = pd.Series(-s[ok]).rank(method="average").to_numpy()
    return r


def _rank_shift(ranks, ref_ranks):
    ok = np.isfinite(ranks) & np.isfinite(ref_ranks)
    return float(np.mean(np.abs(ranks[ok] - ref_ranks[ok]))) if ok.any() else np.nan


def sobol_indices(fA, fB, fAB, idx=None) -> dict:
    """
    First-order and total Sobol indices per factor (group) ``g`` from model
    outputs on the base matrices A and B and on AB_g (A with the columns of
    group g taken from B): ``S_g = mean(f_B (f_ABg - f_A)) / V`` (Saltelli et
    al. 2010, *Comput. Phys. Commun.* 181:259-270, Table 2 (b)) and
    ``S_Tg = mean((f_A - f_ABg)**2) / (2 V)`` (Jansen 1999), with ``V`` the
    variance of the pooled ``f_A`` and ``f_B``. Outputs are centred on their
    pooled mean first (the first-order estimator stays unbiased, since
    ``E[f_ABg - f_A] = 0``, and its variance no longer grows with the output
    mean). ``idx`` selects base samples (bootstrap). Returns
    ``{g: (S_g, S_Tg)}`` (NaN when ``V`` is 0).
    """
    fA, fB = np.asarray(fA, dtype=float), np.asarray(fB, dtype=float)
    idx = np.arange(fA.size) if idx is None else np.asarray(idx)
    fa, fb = fA[idx], fB[idx]
    centre = float(np.mean(np.r_[fa, fb])) if fa.size else 0.0
    fa, fb = fa - centre, fb - centre
    var = np.var(np.r_[fa, fb], ddof=1)
    out = {}
    for name, f_ab in fAB.items():
        fg = np.asarray(f_ab, dtype=float)[idx] - centre
        if not var > 0:
            out[name] = (np.nan, np.nan)
            continue
        out[name] = (float(np.mean(fb * (fg - fa)) / var),
                     float(np.mean((fa - fg) ** 2) / (2.0 * var)))
    return out


def sensitivity_analysis(panel: pd.DataFrame, n_base=256, seed=0, threshold=0.5,
                         kappa=20.0, ref_sigma=0.2, n_boot=200) -> tuple:
    """
    Uncertainty and variance-based sensitivity analysis of the composite.

    Baseline configuration: equal weights, geometric mean (``p = 0``),
    reference 1, legacy zero for missing entries, no cap (the legacy CI).
    ``n_base`` base samples of the uniform factor vector give matrices A and
    B; for every factor group g the matrix AB_g takes group g from B
    (Saltelli 2010). The scalar output is the average absolute rank shift
    ``R_S`` against the baseline; S_g = mean(f_B (f_ABg - f_A)) / V and
    S_Tg = mean((f_A - f_ABg)**2) / (2 V) (Jansen), with bootstrap SEs over
    the base samples. Per unit: median rank, 5-95% rank band and the flip
    rate of the decision ``score >= threshold`` against the baseline decision
    (a run in which the unit is undefined counts as a flip).
    Returns ``(units_df, indices_df, meta)``.
    """
    X = panel[list(COMPONENTS)].to_numpy(dtype=float)
    units = panel["unit"].astype(str).tolist() if "unit" in panel else [
        f"u{i:03d}" for i in range(X.shape[0])]
    base_scores = aggregate_panel(X, np.full(5, 0.2), 0.0, None, "legacy_zero", None)
    base_ranks = _ranks(base_scores)
    base_dec = base_scores >= threshold
    rng = np.random.default_rng(seed)
    sl, dim = _group_slices()
    A = rng.random((int(n_base), dim))
    B = rng.random((int(n_base), dim))

    def _evaluate(U, keep=False):
        f = np.empty(U.shape[0])
        rank_rows, flip_rows = [], []
        for i, u in enumerate(U):
            cfg = _config_from_uniforms(u, kappa, ref_sigma)
            s = aggregate_panel(X, cfg["weights"], cfg["p"], cfg["reference"],
                                cfg["missing"], cfg["cap"])
            r = _ranks(s)
            f[i] = _rank_shift(r, base_ranks)
            if keep:
                rank_rows.append(r)
                flip_rows.append(~np.isfinite(s) | ((s >= threshold) != base_dec))
        return f, rank_rows, flip_rows

    fA, rank_rows, flip_rows = _evaluate(A, keep=True)
    fB, _, _ = _evaluate(B)
    fAB = {}
    for name, _dim in FACTOR_GROUPS:
        ABg = A.copy()
        ABg[:, sl[name]] = B[:, sl[name]]
        fAB[name], _, _ = _evaluate(ABg)

    def _indices(idx):
        return sobol_indices(fA, fB, fAB, idx)

    point = _indices(np.arange(int(n_base)))
    boot_rng = np.random.default_rng(int(seed) + 1)
    boots = [_indices(boot_rng.integers(0, int(n_base), int(n_base)))
             for _ in range(int(n_boot))]
    rows = []
    for name, _dim in FACTOR_GROUPS:
        s1 = np.asarray([b[name][0] for b in boots], dtype=float)
        st = np.asarray([b[name][1] for b in boots], dtype=float)
        rows.append({
            "factor": name,
            "S_first": point[name][0],
            "S_first_se": float(np.nanstd(s1, ddof=1)) if n_boot > 1 else np.nan,
            "S_total": point[name][1],
            "S_total_se": float(np.nanstd(st, ddof=1)) if n_boot > 1 else np.nan,
        })
    indices = pd.DataFrame(rows)
    R = np.vstack(rank_rows)
    F = np.vstack(flip_rows)
    unit_rows = []
    for j, unit in enumerate(units):
        col = R[:, j]
        ok = np.isfinite(col)
        unit_rows.append({
            "unit": unit,
            "baseline_score": float(base_scores[j]),
            "baseline_rank": float(base_ranks[j]),
            "median_rank": float(np.median(col[ok])) if ok.any() else np.nan,
            "rank_p05": float(np.quantile(col[ok], 0.05)) if ok.any() else np.nan,
            "rank_p95": float(np.quantile(col[ok], 0.95)) if ok.any() else np.nan,
            "undefined_rate": float(np.mean(~ok)),
            "decision_flip_rate": float(np.mean(F[:, j])),
            "n_missing_components": int(np.sum(~np.isfinite(X[j]))),
        })
    meta = {
        "n_base": int(n_base),
        "n_model_runs": int(n_base) * (len(FACTOR_GROUPS) + 2),
        "seed": int(seed),
        "threshold": float(threshold),
        "kappa": float(kappa),
        "ref_sigma": float(ref_sigma),
        "n_boot": int(n_boot),
        "R_S_mean": float(np.mean(fA)),
        "R_S_var": float(np.var(np.r_[fA, fB], ddof=1)),
        "factor_groups": [{"name": n, "dim": d} for n, d in FACTOR_GROUPS],
        "aggregation_levels": [str(p) for p in AGGREGATION_LEVELS],
        "missing_levels": list(MISSING_LEVELS),
        "cap_levels": [str(c) for c in CAP_LEVELS],
    }
    return pd.DataFrame(unit_rows), indices, meta


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None if math.isnan(obj) else ("inf" if obj > 0 else "-inf")
    return obj


def code_provenance() -> dict:
    try:
        from impact_pipeline.provenance import collect_code_version

        info = collect_code_version(REPO_ROOT)
    except Exception as exc:  # noqa: BLE001 - provenance must not fail a run
        info = {"code_version": "unknown", "error": str(exc)}
    info["script"] = str(Path(__file__).resolve().relative_to(REPO_ROOT))
    info["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return info


def run_audit(out_dir, *, components_csv=None, commit=LEGACY_COMMIT, mode="auto",
              legacy_source=None, n_base=256, n_boot=200, seed=0, threshold=0.5,
              n_units=40, grid_n=41) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    legacy_ci, legacy_info = load_legacy_compute_ci(
        commit, REPO_ROOT, mode, legacy_source
    )
    comp = compensation_table(legacy_ci)
    comp.to_csv(out / "legacy_compensation.csv", index=False)
    floors = implied_floor_table()
    floors.to_csv(out / "implied_floors.csv", index=False)
    grid = aggregation_grid(legacy_ci, n=grid_n)
    grid.to_csv(out / "aggregation_grid.csv", index=False)
    if components_csv is not None:
        panel = pd.read_csv(components_csv)
        panel_info = {
            "source": str(Path(components_csv).resolve()),
            "sha256": hashlib.sha256(Path(components_csv).read_bytes()).hexdigest(),
        }
    else:
        panel = synthetic_panel(n_units, seed)
        panel_info = {"source": "synthetic_panel", "n_units": int(n_units),
                      "seed": int(seed)}
    missing_cols = [c for c in COMPONENTS if c not in panel.columns]
    if missing_cols:
        raise ValueError(f"components table lacks columns {missing_cols}")
    units, indices, sens_meta = sensitivity_analysis(
        panel, n_base=n_base, seed=seed, threshold=threshold, n_boot=n_boot
    )
    units.to_csv(out / "sensitivity_units.csv", index=False)
    indices.to_csv(out / "sensitivity_indices.csv", index=False)
    by_case = comp.set_index("case")["legacy_CI"]
    summary = {
        "audit_version": AUDIT_VERSION,
        "legacy": legacy_info,
        "headline": {
            "CI(0.01,10,10,10,10)": float(by_case["compensation"]),
            "CI(1e-4,10,10,10,10)": float(by_case["compensation_to_reference"]),
            "CI(NaN,1,1,1,1)": float(by_case["undefined_component"]),
            "CI(0,1,1,1,1)": float(by_case["measured_zero"]),
            "undefined_equals_measured_zero": bool(
                by_case["undefined_component"] == by_case["measured_zero"]),
            "implied_floor_t0.5_w0.2": implied_floor(0.5, 0.2),
        },
        "panel": panel_info,
        "sensitivity": sens_meta,
        "outputs": sorted(p.name for p in out.glob("*.csv")),
        "provenance": code_provenance(),
    }
    with open(out / "audit_aggregation.json", "w", encoding="utf-8") as fh:
        json.dump(_sanitize(summary), fh, indent=2, default=_json_default)
    return summary


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--components", default=None,
                    help="CSV with RAM,PDI,NAS,IIM,SRPI (reference-normalised) and "
                         "optional 'unit'; default: a seeded synthetic panel")
    ap.add_argument("--commit", default=LEGACY_COMMIT)
    ap.add_argument("--load-mode", choices=LOAD_MODES, default="auto")
    ap.add_argument("--legacy-source", default=None,
                    help="saved copy of the legacy mpc_metrics.py "
                         "(verified by its git blob id)")
    ap.add_argument("--n-base", type=int, default=256)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--n-units", type=int, default=40)
    ap.add_argument("--grid-n", type=int, default=41)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_audit(
        args.out, components_csv=args.components, commit=args.commit,
        mode=args.load_mode, legacy_source=args.legacy_source, n_base=args.n_base,
        n_boot=args.n_boot, seed=args.seed, threshold=args.threshold,
        n_units=args.n_units, grid_n=args.grid_n,
    )
    print(json.dumps(_sanitize(summary["headline"]), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
