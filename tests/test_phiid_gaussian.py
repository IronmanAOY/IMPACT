"""Known-answer tests for the closed-form Gaussian MMI-PhiID / Phi_R of VAR(1)
systems (MPC-Bench external marker). Expected values are derived analytically
here, independently of the Lyapunov solver used by the implementation."""

import itertools

import numpy as np
import pytest

from impact_pipeline.bench import phiid_gaussian as pg

LN2 = np.log(2.0)


def _symmetric_expected(a, c, sigma2=1.0):
    """
    A = [[a, c], [c, a]], noise sigma2 * I. Eigenvectors (1, +-1)/sqrt2 with
    eigenvalues l+- = a +- c, so Sigma = sigma2 * sum_k s_k v_k v_k' with
    s_k = 1 / (1 - l_k^2) and Cov(X_{t+1}, X_t) = A Sigma. Returns bits.
    """
    lp, lm = a + c, a - c
    sp, sm = 1.0 / (1.0 - lp**2), 1.0 / (1.0 - lm**2)
    rho_same = (lp * sp + lm * sm) / (sp + sm)
    rho_cross = (lp * sp - lm * sm) / (sp + sm)
    i_same = -0.5 * np.log(1.0 - rho_same**2)
    i_cross = -0.5 * np.log(1.0 - rho_cross**2)
    i_total = 0.5 * np.log(sp * sm)
    phi_wms = i_total - 2.0 * i_same
    rtr = min(i_same, i_cross)
    return {
        "tdmi": i_total / LN2,
        "i_same": i_same / LN2,
        "i_cross": i_cross / LN2,
        "phi_wms": phi_wms / LN2,
        "rtr": rtr / LN2,
        "phi_r": (phi_wms + rtr) / LN2,
    }


def test_independent_processes_have_zero_integration():
    A = np.diag([0.8, -0.3])
    res = pg.phiid_mmi_var1(A, noise_cov=np.diag([1.0, 2.5]))
    assert res["phi_wms"] == pytest.approx(0.0, abs=1e-12)
    assert res["phi_r"] == pytest.approx(0.0, abs=1e-12)
    assert res["rtr"] == pytest.approx(0.0, abs=1e-12)
    mi = res["mutual_information"]
    assert mi["x->y"] == pytest.approx(0.0, abs=1e-12)
    assert mi["y->x"] == pytest.approx(0.0, abs=1e-12)
    # Every atom involving the other unit's future vanishes.
    assert res["atoms"]["xty"] == pytest.approx(0.0, abs=1e-12)
    assert res["atoms"]["ytx"] == pytest.approx(0.0, abs=1e-12)


def test_pure_cross_coupling_known_value():
    # a = 0, c = 0.5: Sigma = I / (1 - c^2) is diagonal, each future is
    # predicted only by the other past: Phi_R = -log2(1 - c^2).
    c = 0.5
    res = pg.phiid_mmi_var1([[0.0, c], [c, 0.0]])
    assert res["phi_r"] == pytest.approx(-np.log2(1.0 - c**2), rel=1e-10)
    assert res["phi_wms"] == pytest.approx(-np.log2(1.0 - c**2), rel=1e-10)
    assert res["rtr"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "a,c,sigma2",
    [
        (0.4, 0.2, 1.0),
        (0.2, 0.2, 1.0),
        (0.3, -0.25, 2.0),
        (-0.5, 0.3, 0.5),
        (0.1, 0.45, 1.0),
    ],
)
def test_symmetric_two_node_var_matches_closed_form(a, c, sigma2):
    exp = _symmetric_expected(a, c, sigma2)
    res = pg.phiid_mmi_var1([[a, c], [c, a]], noise_cov=sigma2 * np.eye(2))
    assert res["tdmi"] == pytest.approx(exp["tdmi"], rel=1e-9)
    assert res["mutual_information"]["x->x"] == pytest.approx(exp["i_same"], rel=1e-9)
    assert res["mutual_information"]["x->y"] == pytest.approx(exp["i_cross"], rel=1e-9)
    assert res["phi_wms"] == pytest.approx(exp["phi_wms"], rel=1e-9, abs=1e-12)
    assert res["rtr"] == pytest.approx(exp["rtr"], rel=1e-9, abs=1e-12)
    assert res["phi_r"] == pytest.approx(exp["phi_r"], rel=1e-9, abs=1e-12)


def test_common_noise_only_has_zero_wms_and_redundancy_correction():
    # A = aI with correlated innovations: Phi_WMS = 0 exactly and
    # Phi_R = rtr = I(X1_t; X2_{t+1}) = -1/2 log(1 - a^2 r^2).
    a, r = 0.6, 0.5
    res = pg.phiid_mmi_var1(a * np.eye(2), noise_cov=[[1.0, r], [r, 1.0]])
    assert res["phi_wms"] == pytest.approx(0.0, abs=1e-12)
    assert res["phi_r"] == pytest.approx(-0.5 * np.log2(1.0 - a * a * r * r), rel=1e-10)


def test_atoms_invert_the_double_redundancy_lattice():
    res = pg.phiid_mmi_var1(
        [[0.3, 0.4], [-0.2, 0.5]], noise_cov=[[1.0, 0.2], [0.2, 1.5]]
    )
    atoms = res["atoms"]
    assert len(atoms) == 16
    assert sum(atoms.values()) == pytest.approx(res["tdmi"], rel=1e-10)
    order = pg.LATTICE
    below = {"r": ("r",), "x": ("r", "x"), "y": ("r", "y"), "s": ("r", "x", "y", "s")}
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            total = sum(atoms[f"{p}t{f}"] for p in below[a] for f in below[b])
            assert total == pytest.approx(res["I_cap"][i, j], abs=1e-12)


def test_node_relabelling_invariance_and_units():
    A = np.array([[0.3, 0.4], [-0.2, 0.5]])
    S = np.array([[1.0, 0.2], [0.2, 1.5]])
    P = np.array([[0, 1], [1, 0]])
    r1 = pg.phiid_mmi_var1(A, S)
    r2 = pg.phiid_mmi_var1(P @ A @ P.T, P @ S @ P.T)
    assert r1["phi_r"] == pytest.approx(r2["phi_r"], rel=1e-10)
    nats = pg.phiid_mmi_var1(A, S, base=np.e)
    assert nats["phi_r"] == pytest.approx(r1["phi_r"] * LN2, rel=1e-10)


def test_multinode_minimum_over_bipartitions():
    A = np.array([[0.3, 0.2, 0.0], [0.2, 0.3, 0.2], [0.0, 0.2, 0.3]])
    res = pg.phi_r_var1(A)
    parts = pg.bipartitions(3)
    assert len(parts) == 3
    vals = [pg.phiid_mmi_var1(A, partition=p)["phi_r"] for p in parts]
    assert res["phi_r"] == pytest.approx(min(vals), rel=1e-12)
    assert len(res["per_partition"]) == 3
    # A block-diagonal system has a zero-integration bipartition.
    B = np.zeros((4, 4))
    B[:2, :2] = [[0.3, 0.4], [0.4, 0.3]]
    B[2:, 2:] = [[0.5, 0.1], [0.1, 0.5]]
    assert pg.phi_r_var1(B)["phi_r"] == pytest.approx(0.0, abs=1e-12)
    assert len(pg.bipartitions(4)) == 7


def test_fitted_var_recovers_analytic_phi_r():
    a, c = 0.4, 0.3
    A = np.array([[a, c], [c, a]])
    rng = np.random.default_rng(0)
    T = 200_000
    x = np.zeros((2, T))
    eps = rng.standard_normal((2, T))
    for t in range(1, T):
        x[:, t] = A @ x[:, t - 1] + eps[:, t]
    A_hat, S_hat = pg.fit_var1(x)
    assert np.allclose(A_hat, A, atol=0.01)
    assert np.allclose(S_hat, np.eye(2), atol=0.02)
    est = pg.phi_r_from_timeseries(x)["phi_r"]
    assert est == pytest.approx(_symmetric_expected(a, c)["phi_r"], abs=0.01)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        pg.phiid_mmi_var1([[1.0, 0.1], [0.1, 0.2]])  # not stationary
    with pytest.raises(ValueError):
        pg.phiid_mmi_var1(0.2 * np.eye(3))  # partition required for n != 2
    with pytest.raises(ValueError):
        pg.phiid_mmi_var1(0.2 * np.eye(3), partition=((0,), (0, 1, 2)))
    with pytest.raises(ValueError):
        pg.phiid_mmi_var1(0.2 * np.eye(2), noise_cov=[[1.0, 2.0], [2.0, 1.0]])
    assert np.isnan(pg.phi_r_from_timeseries(np.ones((2, 2)))["phi_r"])


def test_gaussian_mi_matches_correlation_formula():
    for rho in (0.0, 0.3, -0.7, 0.95):
        cov = np.array([[1.0, rho], [rho, 1.0]])
        assert pg.gaussian_mi(cov, [0], [1]) == pytest.approx(
            -0.5 * np.log2(1.0 - rho**2), abs=1e-12
        )
    # Chain rule on a random 4-variable Gaussian.
    rng = np.random.default_rng(1)
    m = rng.standard_normal((4, 4))
    cov = m @ m.T + 4 * np.eye(4)
    for a, b in itertools.permutations(range(4), 2):
        assert pg.gaussian_mi(cov, [a], [b]) == pytest.approx(
            pg.gaussian_mi(cov, [b], [a])
        )
