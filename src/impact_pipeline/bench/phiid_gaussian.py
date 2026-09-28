"""
Closed-form MMI-PhiID and Phi_R for stationary Gaussian VAR(1) systems.

For X_{t+1} = A X_t + eps, eps ~ N(0, Sigma_eps), the stationary covariance
Sigma solves Sigma = A Sigma A' + Sigma_eps and the lag-one covariance is
Cov(X_{t+1}, X_t) = A Sigma. Mutual information between any past subset U and
future subset V is Gaussian:

    I(U; V) = 1/2 log( |S_UU| |S_VV| / |S_(U,V)| ).

Integrated Information Decomposition (PhiID; Mediano et al. 2021, 2025) of a
bipartition (X1, X2) uses the double-redundancy lattice of the two-source
PID lattice {r = {1}{2}, x = {1}, y = {2}, s = {12}} for past and future.
With minimum mutual information (MMI) redundancy the double redundancies are

    I_cap(a -> b) = I(X^a_t; X^b_{t+1})                   a, b in {x, y, s}
    I_cap(r -> b) = min_i I(X^i_t; X^b_{t+1})             b in {x, y, s}
    I_cap(a -> r) = min_j I(X^a_t; X^j_{t+1})             a in {x, y, s}
    I_cap(r -> r) = min_{i,j} I(X^i_t; X^j_{t+1})         (MMI double redundancy)

and the 16 atoms follow by Moebius inversion on the product lattice. The
whole-minus-sum integrated information is
Phi_WMS = I(X_t; X_{t+1}) - sum_k I(X^k_t; X^k_{t+1}) and the revised measure
is Phi_R = Phi_WMS + I_cap(r -> r) (Mediano et al. 2021, arXiv:2109.13186).
For systems with more than two parts' worth of nodes and no declared
partition, Phi_R is minimised over all bipartitions (unnormalised MIB).
"""

from __future__ import annotations

import itertools
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import linalg

LATTICE = ("r", "x", "y", "s")
# Zeta matrix of the two-source PID lattice: Z[a', a] = 1 if a' <= a.
_ZETA = np.array(
    [
        [1, 1, 1, 1],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
    ],
    dtype=float,
)
_MOEBIUS = np.linalg.inv(_ZETA)
ATOM_NAMES = tuple(f"{a}t{b}" for a in LATTICE for b in LATTICE)


def _as_system(A, noise_cov):
    A = np.atleast_2d(np.asarray(A, dtype=float))
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError(f"A must be square, got {A.shape}")
    radius = float(np.max(np.abs(np.linalg.eigvals(A)))) if n else 0.0
    if radius >= 1.0:
        raise ValueError(
            f"VAR(1) is not stationary (spectral radius {radius:.4f} >= 1)"
        )
    S = (
        np.eye(n)
        if noise_cov is None
        else np.atleast_2d(np.asarray(noise_cov, dtype=float))
    )
    if S.shape != (n, n):
        raise ValueError(f"noise_cov must be {n}x{n}, got {S.shape}")
    if not np.allclose(S, S.T) or np.min(np.linalg.eigvalsh(S)) <= 0:
        raise ValueError("noise_cov must be symmetric positive definite")
    return A, S


def var1_stationary_cov(A, noise_cov=None) -> np.ndarray:
    """Stationary covariance of a VAR(1) (discrete Lyapunov equation)."""
    A, S = _as_system(A, noise_cov)
    sigma = linalg.solve_discrete_lyapunov(A, S)
    return 0.5 * (sigma + sigma.T)


def var1_joint_cov(A, noise_cov=None) -> np.ndarray:
    """Covariance of the stacked vector (X_t, X_{t+1}) (2n x 2n)."""
    A, S = _as_system(A, noise_cov)
    sigma = var1_stationary_cov(A, S)
    cross = sigma @ A.T  # Cov(X_t, X_{t+1})
    return np.block([[sigma, cross], [cross.T, sigma]])


def gaussian_mi(
    cov, idx_a: Sequence[int], idx_b: Sequence[int], base: float = 2.0
) -> float:
    """I(Z_a; Z_b) of a zero-mean Gaussian with covariance ``cov``."""
    cov = np.asarray(cov, dtype=float)
    a = list(idx_a)
    b = list(idx_b)
    if not a or not b:
        return 0.0
    _, ld_a = np.linalg.slogdet(cov[np.ix_(a, a)])
    _, ld_b = np.linalg.slogdet(cov[np.ix_(b, b)])
    ab = a + b
    _, ld_ab = np.linalg.slogdet(cov[np.ix_(ab, ab)])
    return float(0.5 * (ld_a + ld_b - ld_ab) / np.log(base))


def _parts(n: int, partition) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    if partition is None:
        if n != 2:
            raise ValueError("partition is required unless the system has 2 nodes")
        return (0,), (1,)
    p1, p2 = (tuple(int(i) for i in part) for part in partition)
    if not p1 or not p2 or set(p1) & set(p2) or sorted(p1 + p2) != list(range(n)):
        raise ValueError(f"partition must split range({n}) into two non-empty parts")
    return p1, p2


def phiid_mmi_var1(A, noise_cov=None, partition=None, base: float = 2.0) -> dict:
    """
    MMI-PhiID of a stationary Gaussian VAR(1) for one bipartition.

    Returns the 16 atoms (``'rtr'`` ... ``'sts'``, past lattice node first),
    the double-redundancy matrix ``I_cap`` (rows past, columns future, order
    r, x, y, s), the mutual informations, ``phi_wms``, ``rtr`` and ``phi_r``
    (units: log ``base``; bits by default).
    """
    A, S = _as_system(A, noise_cov)
    n = A.shape[0]
    p1, p2 = _parts(n, partition)
    joint = var1_joint_cov(A, S)
    past = {"x": list(p1), "y": list(p2), "s": list(p1) + list(p2)}
    fut = {k: [n + i for i in v] for k, v in past.items()}

    def mi(a, b):
        return gaussian_mi(joint, past[a], fut[b], base=base)

    single = {(a, b): mi(a, b) for a in ("x", "y", "s") for b in ("x", "y", "s")}
    icap = np.zeros((4, 4))
    pos = {k: i for i, k in enumerate(LATTICE)}
    for (a, b), v in single.items():
        icap[pos[a], pos[b]] = v
    for b in ("x", "y", "s"):
        icap[pos["r"], pos[b]] = min(single[("x", b)], single[("y", b)])
    for a in ("x", "y", "s"):
        icap[pos[a], pos["r"]] = min(single[(a, "x")], single[(a, "y")])
    icap[0, 0] = min(
        single[("x", "x")], single[("x", "y")], single[("y", "x")], single[("y", "y")]
    )
    atoms_mat = _MOEBIUS.T @ icap @ _MOEBIUS
    atoms = {
        f"{a}t{b}": float(atoms_mat[pos[a], pos[b]]) for a in LATTICE for b in LATTICE
    }
    tdmi = single[("s", "s")]
    phi_wms = tdmi - single[("x", "x")] - single[("y", "y")]
    rtr = float(icap[0, 0])
    return {
        "partition": (p1, p2),
        "atoms": atoms,
        "I_cap": icap,
        "mutual_information": {f"{a}->{b}": float(v) for (a, b), v in single.items()},
        "tdmi": float(tdmi),
        "phi_wms": float(phi_wms),
        "rtr": rtr,
        "phi_r": float(phi_wms + rtr),
        "base": float(base),
    }


def bipartitions(n: int):
    """All unordered bipartitions of range(n) into two non-empty parts."""
    nodes = tuple(range(int(n)))
    out = []
    for k in range(1, n // 2 + 1):
        for part in itertools.combinations(nodes, k):
            rest = tuple(i for i in nodes if i not in part)
            if k == n - k and part[0] > rest[0]:
                continue
            out.append((part, rest))
    return out


def phi_r_var1(A, noise_cov=None, partition=None, base: float = 2.0) -> dict:
    """
    Phi_R of a VAR(1). With ``partition=None`` and n > 2 nodes, every
    bipartition is evaluated and the minimum Phi_R (with its partition) is
    returned; n = 2 uses the single bipartition.
    """
    A, S = _as_system(A, noise_cov)
    n = A.shape[0]
    if n < 2:
        raise ValueError("Phi_R needs at least 2 nodes")
    if partition is not None or n == 2:
        res = phiid_mmi_var1(A, S, partition=partition, base=base)
        return {
            "phi_r": res["phi_r"],
            "phi_wms": res["phi_wms"],
            "partition": res["partition"],
            "per_partition": {str(res["partition"]): res["phi_r"]},
        }
    per = {}
    best = None
    for part in bipartitions(n):
        res = phiid_mmi_var1(A, S, partition=part, base=base)
        per[str(res["partition"])] = res["phi_r"]
        if best is None or res["phi_r"] < best["phi_r"]:
            best = res
    return {
        "phi_r": best["phi_r"],
        "phi_wms": best["phi_wms"],
        "partition": best["partition"],
        "per_partition": per,
    }


def fit_var1(ts, ridge: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Least-squares VAR(1) fit of a nodes x time array (demeaned); returns
    (A, residual covariance)."""
    x = np.asarray(ts, dtype=float)
    if x.ndim != 2 or x.shape[1] < x.shape[0] + 2:
        raise ValueError("ts must be nodes x time with more samples than nodes")
    x = x - x.mean(axis=1, keepdims=True)
    past, fut = x[:, :-1], x[:, 1:]
    gram = past @ past.T + float(ridge) * np.eye(x.shape[0])
    A = np.linalg.solve(gram, past @ fut.T).T
    resid = fut - A @ past
    noise = resid @ resid.T / (resid.shape[1] - x.shape[0])
    return A, 0.5 * (noise + noise.T)


def phi_r_from_timeseries(
    ts, partition=None, base: float = 2.0, ridge: float = 0.0
) -> dict:
    """Gaussian Phi_R of the VAR(1) fitted to a nodes x time array (external
    single marker for the rule audit). NaN when the fit is not stationary."""
    try:
        A, S = fit_var1(ts, ridge=ridge)
        out = phi_r_var1(A, S, partition=partition, base=base)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return {"phi_r": float("nan"), "reason": str(exc)}
    out["A"] = A
    out["noise_cov"] = S
    return out


def phi_r_marker(
    ts, macro_nodes: Optional[Dict[str, Sequence[int]]] = None, **kwargs
) -> float:
    """Phi_R of macro-node means (declared grain) or of the given nodes."""
    x = np.asarray(ts, dtype=float)
    if macro_nodes:
        x = np.stack(
            [x[np.asarray(v, dtype=int)].mean(axis=0) for v in macro_nodes.values()]
        )
    return float(phi_r_from_timeseries(x, **kwargs)["phi_r"])
