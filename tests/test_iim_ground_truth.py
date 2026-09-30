"""Ground-truth and null tests for IIM.

Small systems (2-4 binary nodes) keep the runtime low while exercising the full
estimator: TPM estimation, mechanism/purview MIP search, system cuts and
surrogate-null calibration.
"""

import numpy as np
import pytest

from impact_pipeline import mpc_metrics as mm


def _independent_noise(n, t, seed):
    return np.random.RandomState(seed).randn(n, t)


def _var1(n, t, seed, coupling, burn=200):
    """All-to-all VAR(1) without self-coupling: x_t = c (1 1^T - I) x_{t-1} + e_t."""
    rng = np.random.RandomState(seed)
    a = coupling * (np.ones((n, n)) - np.eye(n))
    x = np.zeros((n, t + burn))
    for k in range(1, t + burn):
        x[:, k] = a @ x[:, k - 1] + rng.randn(n)
    return x[:, burn:]


def _kinetic_ising(n, t, seed, coupling, burn=200):
    """Parallel-update kinetic Ising model, uniform coupling J, no self-coupling."""
    rng = np.random.RandomState(seed)
    w = coupling * (np.ones((n, n)) - np.eye(n))
    s = rng.choice([-1.0, 1.0], size=n)
    out = np.zeros((n, t + burn))
    for k in range(t + burn):
        p_up = 1.0 / (1.0 + np.exp(-2.0 * (w @ s)))
        s = np.where(rng.rand(n) < p_up, 1.0, -1.0)
        out[:, k] = s
    return out[:, burn:] + 1e-3 * rng.randn(n, t)


def _copy_wires(t, seed, mapping, flip=0.05):
    """Two binary nodes; x_i(t+1) = x_{mapping[i]}(t) with flip noise (reducible)."""
    rng = np.random.RandomState(seed)
    x = np.zeros((2, t))
    x[:, 0] = rng.randint(0, 2, size=2)
    for k in range(1, t):
        for i in range(2):
            v = x[mapping[i], k - 1]
            x[i, k] = 1.0 - v if rng.rand() < flip else v
    return x + 1e-6 * rng.randn(2, t)


def _calibrated(ts, seed, n_surrogates=8):
    return mm.compute_IIM(
        ts,
        bins=2,
        return_details=True,
        null_surrogates=n_surrogates,
        null_seed=seed,
    )


SEEDS = (0, 1, 2)


def test_crossed_wires_are_as_reducible_as_identity_wires():
    # Two independent copy-wires are reducible whether each node copies itself
    # (identity) or the other node (swap). The swap is only recognised as
    # reducible if the crossed pairing (M1->Z2, M2->Z1) is searched; with a
    # single pairing orientation Psi_full(swap) was ~0.16 vs ~0 for identity.
    identity = mm.compute_IIM(_copy_wires(2000, 0, (0, 1)), bins=2, return_details=True)
    swap = mm.compute_IIM(_copy_wires(2000, 0, (1, 0)), bins=2, return_details=True)
    assert identity["Psi_full"] < 5e-3
    assert swap["Psi_full"] < 5e-3


@pytest.mark.parametrize("perm", [(1, 0, 2), (2, 1, 0), (1, 2, 0)])
def test_iim_is_invariant_to_node_labelling(perm):
    ts = _kinetic_ising(3, 400, 5, 0.3)
    base = mm.compute_IIM(ts, bins=2, return_details=True)
    permuted = mm.compute_IIM(ts[list(perm)], bins=2, return_details=True)
    assert base["defined"] and permuted["defined"]
    for key in ("Psi_full", "Psi_mip_preserved", "raw"):
        assert permuted[key] == pytest.approx(base[key], rel=1e-9, abs=1e-12)


def test_calibrated_iim_is_null_for_noise_and_positive_for_coupled_systems():
    noise = [_calibrated(_independent_noise(3, 300, s), s) for s in SEEDS]
    var = [_calibrated(_var1(3, 300, s, 0.4), s) for s in SEEDS]
    ising = [_calibrated(_kinetic_ising(3, 300, s, 0.4), s) for s in SEEDS]

    for info in noise + var + ising:
        assert info["defined"] is True
        assert info["IIM_null_calibrated"] is True
        assert info["IIM_null_n"] == 8

    noise_cal = np.mean([d["canonical_calibrated"] for d in noise])
    noise_excess = np.mean([d["IIM_excess"] for d in noise])
    noise_z = np.mean([d["IIM_z"] for d in noise])
    # Independent nodes: calibrated IIM ~ 0 and the excess is centred on 0.
    assert noise_cal < 1e-3
    assert abs(noise_excess) < 1e-3
    assert abs(noise_z) < 2.0

    for coupled in (var, ising):
        cal = np.asarray([d["canonical_calibrated"] for d in coupled])
        z = np.asarray([d["IIM_z"] for d in coupled])
        assert np.all(cal > 10.0 * max(noise_cal, 1e-4))
        assert np.all(z > 3.0)


def test_uncalibrated_ratio_does_not_separate_noise_from_coupling():
    # Documents why the calibration is needed: the scale-free ratio IIM_raw is
    # O(1) for independent noise, so it cannot be used as evidence on its own.
    noise = mm.compute_IIM(_independent_noise(3, 300, 0), bins=2, return_details=True)
    assert noise["defined"] is True
    assert noise["canonical"] > 0.2
    assert noise["Psi_full"] < 0.01


@pytest.mark.parametrize(
    "make, couplings",
    [
        (_kinetic_ising, (0.0, 0.2, 0.4, 0.6)),
        (_var1, (0.0, 0.2, 0.4)),
    ],
    ids=["kinetic_ising", "var1"],
)
def test_calibrated_iim_increases_with_coupling(make, couplings):
    means = []
    for c in couplings:
        vals = [
            _calibrated(make(3, 300, s, c), s)["canonical_calibrated"] for s in SEEDS
        ]
        means.append(float(np.mean(vals)))
    assert means[0] < 1e-3
    assert all(b > a for a, b in zip(means[:-1], means[1:])), means


def test_null_fields_are_consistent():
    info = _calibrated(_var1(3, 300, 0, 0.4), 0, n_surrogates=5)
    assert info["IIM_raw"] == pytest.approx(info["raw"], rel=1e-12)
    assert info["Delta_Psi"] == pytest.approx(
        info["Psi_full"] - info["Psi_mip_preserved"]
    )
    assert len(info["Delta_Psi_null"]) == 5
    assert info["Delta_Psi_null_mean"] == pytest.approx(np.mean(info["Delta_Psi_null"]))
    assert info["IIM_z"] == pytest.approx(
        (info["IIM_raw"] - info["IIM_null_mean"]) / info["IIM_null_sd"], rel=1e-9
    )
    assert info["IIM_excess"] == pytest.approx(
        info["Delta_Psi"] - info["Delta_Psi_null_mean"]
    )
    assert info["canonical_calibrated"] == pytest.approx(max(info["IIM_excess"], 0.0))
    assert info["IIM_calibrated"] == info["canonical_calibrated"]
    # With calibration requested, the returned value is the calibrated canonical IIM.
    assert info["value"] == pytest.approx(info["canonical_calibrated"])
    assert 1.0 / 6.0 <= info["IIM_null_p"] <= 1.0
    # Legacy fields keep their uncalibrated meaning.
    assert info["canonical"] == pytest.approx(np.clip(info["raw"], 0.0, 1.0))


def test_uncalibrated_run_reports_nan_null_fields():
    info = mm.compute_IIM(_independent_noise(3, 120, 0), bins=2, return_details=True)
    assert info["IIM_null_calibrated"] is False
    assert info["IIM_null_n"] == 0
    assert np.isnan(info["IIM_z"])
    assert np.isnan(info["canonical_calibrated"])
    assert info["value"] == pytest.approx(info["canonical"])


def _reference_node_tpm(disc, base, alpha, shrink):
    """Loop-based state-by-node TPM (Laplace, or Hausser-Strimmer shrinkage of
    each row towards the node's own transition p(x_i'|x_i))."""
    n, t = disc.shape
    cur, nxt = disc[:, :-1].T, disc[:, 1:].T
    states = np.array(list(np.ndindex(*([base] * n))))
    keys = [int(np.ravel_multi_index(tuple(c), [base] * n)) for c in cur]
    tpm = np.ones((len(states), len(states)))
    for i in range(n):
        own = np.full((base, base), alpha)
        for a, b in zip(cur[:, i], nxt[:, i]):
            own[a, b] += 1.0
        own /= own.sum(axis=1, keepdims=True)
        for s, sv in enumerate(states):
            nxt_i = [nxt[k, i] for k in range(len(keys)) if keys[k] == s]
            counts = np.bincount(np.asarray(nxt_i, dtype=int), minlength=base)
            n_s = len(nxt_i)
            if not shrink:
                p = (counts + alpha) / (n_s + alpha * base)
            elif n_s <= 1:
                p = own[sv[i]]
            else:
                ml = counts / n_s
                target = own[sv[i]]
                den = (n_s - 1) * np.sum((target - ml) ** 2)
                lam = 1.0 if den <= 0 else np.clip((1 - np.sum(ml**2)) / den, 0, 1)
                p = lam * target + (1 - lam) * ml
            tpm[s] *= p[states[:, i]]
    return tpm / tpm.sum(axis=1, keepdims=True)


@pytest.mark.parametrize("estimator", ["node_shrinkage", "node_laplace"])
def test_state_by_node_tpm_matches_reference(estimator):
    rng = np.random.RandomState(1)
    disc = rng.randint(0, 3, size=(3, 60)).astype(np.int16)
    disc[2] = np.roll(disc[0], 1)  # a lagged cross-node dependence
    _, tpm, _ = mm._iim_build_states_and_tpm(disc, 3, 1, 1e-3, estimator=estimator)
    ref = _reference_node_tpm(disc, 3, 1e-3, shrink=(estimator == "node_shrinkage"))
    np.testing.assert_allclose(tpm, ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(tpm.sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_size_one_mechanisms_contribute_zero():
    # Documented convention: a mechanism (or purview) of size 1 has no
    # bipartition into two non-empty parts, so it contributes exactly 0 to Psi.
    info = mm.compute_IIM(
        _kinetic_ising(3, 300, 0, 0.4),
        bins=2,
        max_mechanism_size=1,
        return_details=True,
    )
    assert info["defined"] is False
    assert info["undefined_reason"] == "nonpositive_psi_full"
    assert info["Psi_full"] == 0.0


def test_parallel_matches_serial():
    ts = _var1(4, 200, 3, 0.25)
    serial = mm.compute_IIM(
        ts, bins=2, return_details=True, phase1_parallel_workers=None
    )
    parallel = mm.compute_IIM(
        ts,
        bins=2,
        return_details=True,
        phase1_parallel_workers=2,
        phase1_chunk_size=2,
    )
    assert parallel["phase_parallel_enabled"] is True
    assert parallel["phase_parallel_fallbacks"] == []
    for key in ("Psi_full", "Psi_mip_preserved", "raw"):
        assert parallel[key] == pytest.approx(serial[key], rel=1e-10, abs=1e-14)
