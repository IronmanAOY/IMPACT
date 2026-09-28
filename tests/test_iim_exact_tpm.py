"""
Exact IIM on known TPMs (stream I2): compute_IIM_from_tpm, directional cuts,
the per-unit TPM estimator and bearer-restricted subsystems.

Known-answer systems are parallel-update kinetic Ising networks
(P(s_i'=1|s) = sigmoid(2 beta sum_j J_ij sigma_j), sigma = 2s-1), whose TPMs are
exact and conditionally independent (state-by-node, as in IIT):

* independent units (self-coupling only): Psi_full = 0 exactly, so IIM = 0;
* recurrent ring / all-to-all: IIM > 0 under both cut families;
* feedforward fan-out (a self-coupled hub driving the other units): observed
  Delta_Psi = 0 (within 1e-12) under directional cuts, because the cut that
  severs the direction without connections leaves the TPM unchanged, but
  clearly > 0 (0.036 bits for n=3, 0.051 bits for n=4 at J=1, beta=1) under the
  bidirectional cuts of compute_IIM, which also noise the hub's outputs.
"""

import numpy as np
import pytest

from impact_pipeline import iim_xp
from impact_pipeline import mpc_metrics as mm

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _ising_tpm(coupling, beta=1.0):
    n = coupling.shape[0]
    states = mm.iim_state_table(n, 2)
    sigma = 2.0 * states - 1.0
    p_up = 1.0 / (1.0 + np.exp(-2.0 * beta * (sigma @ coupling.T)))
    return mm.iim_tpm_from_unit_probabilities(p_up, 2)


def _networks(n):
    self_c = 0.6 * np.eye(n)
    ring = self_c + 0.6 * (np.roll(np.eye(n), 1, 1) + np.roll(np.eye(n), -1, 1))
    all_to_all = self_c + 0.5 * (np.ones((n, n)) - np.eye(n))
    star = self_c.copy()
    star[1:, 0] = 1.0  # unit 0 drives every other unit, nothing drives unit 0
    chain = self_c.copy()
    for i in range(1, n):
        chain[i, i - 1] = 1.0  # 0 -> 1 -> ... -> n-1
    return {
        "independent": self_c,
        "ring": ring,
        "all_to_all": all_to_all,
        "ff_star": star,
        "ff_chain": chain,
    }


def _exact(coupling, cut_mode="bidirectional", **kwargs):
    return mm.compute_IIM_from_tpm(
        _ising_tpm(coupling), cut_mode=cut_mode, return_details=True, **kwargs
    )


def _sample(tpm, n, t, seed):
    """Binary trajectory (nodes x time) of the Markov chain with TPM ``tpm``."""
    rng = np.random.default_rng(seed)
    cdf = np.cumsum(tpm, axis=1)
    states = mm.iim_state_table(n, 2)
    s = int(rng.integers(tpm.shape[0]))
    u = rng.random(t)
    keys = np.empty(t, dtype=np.int64)
    for k in range(t):
        keys[k] = s
        s = min(int(np.searchsorted(cdf[s], u[k], side="right")), tpm.shape[0] - 1)
    return states[keys].T.astype(float)


# ---------------------------------------------------------------------------
# Helpers: state table, TPM builder, stationary distribution
# ---------------------------------------------------------------------------


def test_state_table_and_unit_probability_tpm():
    np.testing.assert_array_equal(
        mm.iim_state_table(2, 2), [[0, 0], [0, 1], [1, 0], [1, 1]]
    )
    rng = np.random.default_rng(0)
    p_up = rng.random((4, 2))
    tpm = mm.iim_tpm_from_unit_probabilities(p_up)
    states = mm.iim_state_table(2, 2)
    for s in range(4):
        for s2 in range(4):
            want = np.prod(
                [p_up[s, i] if states[s2, i] else 1.0 - p_up[s, i] for i in range(2)]
            )
            assert tpm[s, s2] == pytest.approx(want, abs=1e-15)
    # base 3 with explicit per-state distributions
    p3 = rng.random((9, 2, 3))
    p3 /= p3.sum(axis=2, keepdims=True)
    t3 = mm.iim_tpm_from_unit_probabilities(p3, base=3)
    assert t3.shape == (9, 9)
    np.testing.assert_allclose(t3.sum(axis=1), 1.0, atol=1e-12)
    assert mm.iim_xp.state_by_node_deviation(t3, 2, 3) < 1e-12
    with pytest.raises(ValueError):
        mm.iim_tpm_from_unit_probabilities(np.full((5, 2), 0.5))
    with pytest.raises(ValueError):
        mm.iim_tpm_from_unit_probabilities(np.full((4, 2, 2), 0.7))


def test_stationary_distribution_known_answers():
    a, b = 0.2, 0.05
    two_state = np.array([[1 - a, a], [b, 1 - b]])
    np.testing.assert_allclose(
        mm.iim_stationary_distribution(two_state),
        [b / (a + b), a / (a + b)],
        atol=1e-12,
    )
    # periodic swap chain: the long-run average from a uniform start is uniform
    np.testing.assert_allclose(
        mm.iim_stationary_distribution(np.array([[0.0, 1.0], [1.0, 0.0]])),
        [0.5, 0.5],
        atol=1e-12,
    )
    # independent units: product of the per-unit stationary distributions
    tpm = _ising_tpm(np.diag([0.3, 0.8]), beta=1.0)
    pi = mm.iim_stationary_distribution(tpm)
    np.testing.assert_allclose(pi @ tpm, pi, atol=1e-12)
    assert pi.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Known answers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [3, 4])
@pytest.mark.parametrize("cut_mode", ["bidirectional", "directional"])
def test_independent_units_have_zero_iim(n, cut_mode):
    info = _exact(_networks(n)["independent"], cut_mode)
    assert info["defined"] is True
    assert abs(info["Psi_full"]) < 1e-12
    assert abs(info["Delta_Psi"]) < 1e-12
    assert info["value"] == pytest.approx(0.0, abs=1e-12)
    # Psi_full is rounding noise (~1e-17): the ratio is 0 by definition
    assert info["psi_full_is_zero"] is True
    assert info["raw"] == 0.0 and info["canonical"] == 0.0
    assert info["tpm_state_by_node"] is True


def test_independent_units_with_three_states_and_uniform_weights():
    # Units with arbitrary own-state dynamics (base 3) and no cross-talk.
    rng = np.random.default_rng(4)
    own = rng.random((3, 3, 3))
    own /= own.sum(axis=2, keepdims=True)  # own[i][x_i] -> distribution of x_i'
    states = mm.iim_state_table(3, 3)
    probs = np.stack([own[i][states[:, i]] for i in range(3)], axis=1)
    tpm = mm.iim_tpm_from_unit_probabilities(probs, base=3)
    for mode in ("bidirectional", "directional"):
        info = mm.compute_IIM_from_tpm(
            tpm, base=3, cut_mode=mode, state_weights="uniform", return_details=True
        )
        assert abs(info["Delta_Psi"]) < 1e-12 and abs(info["Psi_full"]) < 1e-12


@pytest.mark.parametrize("name,n", [("ring", 4), ("all_to_all", 3), ("all_to_all", 4)])
@pytest.mark.parametrize("cut_mode", ["bidirectional", "directional"])
def test_recurrent_networks_are_integrated(name, n, cut_mode):
    info = _exact(_networks(n)[name], cut_mode)
    assert info["Psi_full"] > 0.05
    assert info["Delta_Psi"] > 1e-2
    assert info["value"] == pytest.approx(info["Delta_Psi"])
    assert 0.0 < info["canonical"] <= 1.0


@pytest.mark.parametrize("n", [3, 4])
@pytest.mark.parametrize("name", ["ff_star", "ff_chain"])
def test_feedforward_is_zero_under_directional_but_positive_under_bidirectional(
    n, name
):
    coupling = _networks(n)[name]
    bidir = _exact(coupling, "bidirectional")
    direc = _exact(coupling, "directional")
    assert bidir["Psi_full"] == pytest.approx(direc["Psi_full"], rel=1e-12)
    assert bidir["Psi_full"] > 0.05
    # Observed: bidirectional cuts also noise the feedforward connections.
    assert bidir["Delta_Psi"] > 1e-2
    # Observed: a directional cut severing a direction without connections
    # leaves the TPM (and Psi) unchanged, so Delta_Psi is 0 to rounding.
    assert abs(direc["Delta_Psi"]) < 1e-12
    assert direc["value"] == pytest.approx(0.0, abs=1e-12)
    part_a, part_b = direc["mip_cut"]
    # J[i, j] = influence of j on i: the severed direction A -> B is empty.
    assert np.abs(coupling[np.ix_(part_b, part_a)]).sum() == 0.0
    assert direc["n_cuts_evaluated"] == 2 * bidir["n_cuts_evaluated"]


@pytest.mark.parametrize("cut_mode", ["bidirectional", "directional"])
@pytest.mark.parametrize("perm", [(1, 0, 2, 3), (3, 2, 1, 0), (2, 0, 3, 1)])
def test_orientation_and_labelling_invariance(cut_mode, perm):
    """Relabelling units (and hence the orientation of every cut) changes nothing."""
    rng = np.random.default_rng(7)
    coupling = 0.8 * rng.standard_normal((4, 4))
    coupling[0, 1:] = 0.0  # unit 0 receives nothing: asymmetric, partly feedforward
    base = _exact(coupling, cut_mode)
    idx = list(perm)
    relabelled = _exact(coupling[np.ix_(idx, idx)], cut_mode)
    for key in ("Psi_full", "Psi_mip_preserved", "Delta_Psi", "raw"):
        assert relabelled[key] == pytest.approx(base[key], rel=1e-10, abs=1e-13), key
    # both orientations of every bipartition are scored in directional mode
    keys = set(base["cut_scores"])
    if cut_mode == "directional":
        for key in keys:
            left, right = key.split("|")
            assert f"{right}|{left}" in keys


def test_exact_numba_and_xp_kernels_agree():
    coupling = _networks(4)["ring"]
    for mode in ("bidirectional", "directional"):
        a = _exact(coupling, mode, psi_kernel="numba")
        b = _exact(coupling, mode, psi_kernel="xp")
        assert a["psi_kernel"] == "numba" and b["psi_kernel"] == "xp"
        for key in ("Psi_full", "Psi_mip_preserved", "Delta_Psi"):
            assert a[key] == pytest.approx(b[key], rel=1e-12, abs=1e-14)
        for key in a["cut_scores"]:
            assert a["cut_scores"][key] == pytest.approx(
                b["cut_scores"][key], rel=1e-12, abs=1e-14
            )


def test_directional_cut_matches_per_unit_noising_reference():
    rng = np.random.default_rng(3)
    n = 3
    p_up = rng.random((8, n))
    tpm = mm.iim_tpm_from_unit_probabilities(p_up)
    states = mm.iim_state_table(n, 2)
    for part_a, part_b in mm._iim_all_system_cuts(n, "all", "directional"):
        ref_p = p_up.copy()
        key_b = iim_xp.subset_keys(states, part_b, 2)
        for i in part_b:  # unit i in B: its A-inputs replaced by uniform noise
            for kb in np.unique(key_b):
                rows = key_b == kb
                ref_p[rows, i] = p_up[rows, i].mean()
        ref = mm.iim_tpm_from_unit_probabilities(ref_p)
        got = mm._iim_build_cut_tpm_for_mode(
            tpm, states, 2, part_a, part_b, cut_mode="directional"
        )
        np.testing.assert_allclose(got, ref, atol=1e-14)
        # bidirectional tensor form == the legacy host builder
        np.testing.assert_allclose(
            iim_xp.cut_tpm(tpm, n, 2, part_a, part_b, "bidirectional"),
            mm._iim_build_cut_tpm(tpm, states, 2, part_a, part_b),
            atol=1e-14,
        )


# ---------------------------------------------------------------------------
# Agreement with compute_IIM on trajectories sampled from the same TPM
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("estimator", ["node_shrinkage", "per_unit"])
def test_sampled_iim_converges_to_the_exact_value(estimator):
    n = 3
    tpm = _ising_tpm(_networks(n)["all_to_all"])
    exact = mm.compute_IIM_from_tpm(tpm, return_details=True)
    errors = {}
    for t in (1000, 30000):
        errs = []
        for seed in range(3):
            info = mm.compute_IIM(
                _sample(tpm, n, t, seed), bins=2, tpm_estimator=estimator,
                return_details=True,
            )
            delta = info["Psi_full"] - info["Psi_mip_preserved"]
            errs.append(
                (
                    abs(info["Psi_full"] - exact["Psi_full"]),
                    abs(delta - exact["Delta_Psi"]),
                    abs(info["raw"] - exact["raw"]),
                )
            )
        errors[t] = np.mean(errs, axis=0)
    # finite-sample bias shrinks with T ...
    assert np.all(errors[30000] < errors[1000])
    # ... and the long-run estimate agrees with the exact IIM
    assert errors[30000][0] < 0.05 * exact["Psi_full"]
    assert errors[30000][1] < 0.10 * exact["Delta_Psi"]
    assert errors[30000][2] < 0.10


def test_sampled_independent_units_expose_the_ratio_null_floor():
    """Delta_Psi -> 0 = exact value, but the scale-free ratio does not."""
    n = 3
    tpm = _ising_tpm(_networks(n)["independent"])
    exact = mm.compute_IIM_from_tpm(tpm, return_details=True)
    assert exact["Delta_Psi"] == pytest.approx(0.0, abs=1e-12)
    info = mm.compute_IIM(_sample(tpm, n, 30000, 0), bins=2, return_details=True)
    assert abs(info["Psi_full"] - info["Psi_mip_preserved"]) < 1e-3
    assert info["Psi_full"] < 1e-3
    # the uncalibrated ratio stays O(1) (why compute_IIM needs the null calibration)
    assert info["raw"] > 0.2


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def test_input_validation_and_kwargs():
    tpm = _ising_tpm(_networks(3)["ring"])
    with pytest.raises(ValueError, match="rows must sum to 1"):
        mm.compute_IIM_from_tpm(tpm * 1.01)
    with pytest.raises(ValueError, match="base\\*\\*n"):
        mm.compute_IIM_from_tpm(np.full((6, 6), 1.0 / 6.0))
    with pytest.raises(ValueError, match="does not match base"):
        mm.compute_IIM_from_tpm(tpm, bins=3)
    with pytest.raises(ValueError, match="not supported for a known TPM"):
        mm.compute_IIM_from_tpm(tpm, max_nodes=2)
    with pytest.raises(TypeError, match="unexpected keyword"):
        mm.compute_IIM_from_tpm(tpm, no_such_option=1)
    with pytest.raises(ValueError, match="cut_mode"):
        mm.compute_IIM_from_tpm(tpm, cut_mode="sideways")
    # a joint (not state-by-node) TPM is fine bidirectionally, not directionally
    rng = np.random.default_rng(1)
    joint = rng.random((8, 8))
    joint /= joint.sum(axis=1, keepdims=True)
    info = mm.compute_IIM_from_tpm(joint, return_details=True)
    assert info["tpm_state_by_node"] is False
    with pytest.raises(ValueError, match="state-by-node"):
        mm.compute_IIM_from_tpm(joint, cut_mode="directional")
    # data-estimation keywords are accepted and reported as ignored
    info = mm.compute_IIM_from_tpm(
        tpm, return_details=True, lag_trs=1, tpm_estimator="node_laplace", bins=2
    )
    assert info["ignored_iim_kwargs"] == ["lag_trs", "tpm_estimator"]
    assert mm.compute_IIM_from_tpm(tpm) == pytest.approx(info["value"])
    assert mm.compute_IIM_from_tpm(tpm, scale=2.0) == pytest.approx(2.0 * info["value"])


def test_state_weights_options():
    tpm = _ising_tpm(_networks(3)["all_to_all"])
    stat = mm.compute_IIM_from_tpm(tpm, return_details=True)
    assert stat["state_weights_mode"] == "stationary"
    explicit = mm.compute_IIM_from_tpm(
        tpm, state_weights=np.asarray(stat["state_weights"]) * 7.0, return_details=True
    )
    assert explicit["Psi_full"] == pytest.approx(stat["Psi_full"], rel=1e-12)
    uniform = mm.compute_IIM_from_tpm(tpm, state_weights="uniform", return_details=True)
    assert uniform["state_weights_mode"] == "uniform"
    # the Psi average only runs over states with positive weight
    one_state = np.zeros(8)
    one_state[5] = 1.0
    single = mm.compute_IIM_from_tpm(tpm, state_weights=one_state, return_details=True)
    assert single["Psi_full"] != pytest.approx(uniform["Psi_full"])
    with pytest.raises(ValueError, match="state_weights"):
        mm.compute_IIM_from_tpm(tpm, state_weights=-one_state)


# ---------------------------------------------------------------------------
# compute_IIM options: per-unit estimator, directional cuts, bearer nodes
# ---------------------------------------------------------------------------


def test_per_unit_estimator_is_the_ml_conditional_frequency():
    rng = np.random.RandomState(0)
    disc = rng.randint(0, 2, size=(3, 40))
    disc[:, :4] = 0  # make state 0 frequent
    disc[:, 4:] = np.where(disc[:, 4:] == 0, 0, 1)
    curr, tpm, states = mm._iim_build_states_and_tpm(
        disc, 2, 1, 1e-3, estimator="per_unit"
    )
    keys = iim_xp.subset_keys(curr, (0, 1, 2), 2)
    nxt = disc[:, 1:].T
    for s in range(8):
        rows = np.flatnonzero(keys == s)
        for i in range(3):
            own = disc[i, :-1]
            self_p = np.array(
                [
                    [
                        (np.sum((own == a) & (disc[i, 1:] == b)) + 1e-3)
                        / (np.sum(own == a) + 2e-3)
                        for b in range(2)
                    ]
                    for a in range(2)
                ]
            )
            want = (
                np.bincount(nxt[rows, i], minlength=2) / rows.size
                if rows.size
                else self_p[states[s, i]]
            )
            marg = np.array(
                [tpm[s, states[:, i] == b].sum() for b in range(2)]
            )
            np.testing.assert_allclose(marg, want, atol=1e-12)
    assert mm.iim_xp.state_by_node_deviation(tpm, 3, 2) < 1e-12


def test_compute_iim_per_unit_and_directional_options():
    tpm = _ising_tpm(_networks(3)["ff_star"])
    ts = _sample(tpm, 3, 4000, 1)
    bidir = mm.compute_IIM(ts, bins=2, tpm_estimator="per_unit", return_details=True)
    direc = mm.compute_IIM(
        ts, bins=2, tpm_estimator="per_unit", cut_mode="directional",
        return_details=True,
    )
    assert bidir["tpm_estimator"] == direc["tpm_estimator"] == "per_unit"
    assert direc["cut_mode"] == "directional"
    assert direc["n_cuts_evaluated"] == 2 * bidir["n_cuts_evaluated"]
    delta_b = bidir["Psi_full"] - bidir["Psi_mip_preserved"]
    delta_d = direc["Psi_full"] - direc["Psi_mip_preserved"]
    assert delta_d < 0.25 * delta_b  # the feedforward star is ~reducible directionally
    with pytest.raises(ValueError, match="state-by-node"):
        mm.compute_IIM(
            ts, bins=2, tpm_estimator="joint_laplace", cut_mode="directional"
        )


def test_bearer_nodes_restrict_the_subsystem():
    rng = np.random.RandomState(2)
    ts = rng.randn(5, 150)
    ts[0] *= 10.0  # the highest-variance node is not a bearer node
    prep = mm.prepare_iim_problem(ts, bins=2, max_nodes=2, bearer_nodes=[1, 2, 4])
    assert prep["bearer_nodes"] == [1, 2, 4]
    assert set(prep["selected_nodes"]) <= {1, 2, 4}
    assert 0 not in prep["selected_nodes"]
    # all bearer nodes selected: identical to computing on the bearer rows
    info = mm.compute_IIM(ts, bins=2, bearer_nodes=[4, 1, 2], return_details=True)
    direct = mm.compute_IIM(ts[[1, 2, 4]], bins=2, return_details=True)
    assert info["selected_nodes"] == [1, 2, 4]
    assert info["bearer_nodes"] == [1, 2, 4]
    for key in ("Psi_full", "Psi_mip_preserved", "raw"):
        assert info[key] == pytest.approx(direct[key], rel=1e-12, abs=1e-15)
    with pytest.raises(ValueError, match="not bearer nodes"):
        mm.prepare_iim_problem(ts, bins=2, bearer_nodes=[1, 2], node_indices=[0, 1])
    with pytest.raises(ValueError, match="bearer_nodes"):
        mm.prepare_iim_problem(ts, bins=2, bearer_nodes=[1, 1])
    single = mm.compute_IIM(ts, bins=2, bearer_nodes=[3], return_details=True)
    assert single["defined"] is False
    assert single["undefined_reason"] == "insufficient_bearer_nodes"


def test_cut_mode_and_bearer_subsystem_enter_the_checkpoint_signature(tmp_path):
    rng = np.random.RandomState(5)
    ts = rng.randn(4, 120)
    ckpt = str(tmp_path / "iim.json")
    first = mm.compute_IIM(
        ts, bins=2, return_details=True, checkpoint_path=ckpt, bearer_nodes=[0, 1, 2]
    )
    again = mm.compute_IIM(
        ts, bins=2, return_details=True, checkpoint_path=ckpt, bearer_nodes=[0, 1, 2]
    )
    assert again["checkpoint_resumed"] is True
    assert first["Psi_full"] == pytest.approx(again["Psi_full"])
    # a different bearer set changes the subsystem (selected_nodes)
    other = mm.compute_IIM(
        ts, bins=2, return_details=True, checkpoint_path=ckpt, bearer_nodes=[1, 2, 3]
    )
    assert other["checkpoint_resumed"] is False
    # the cut family changes the cut scores
    directional = mm.compute_IIM(
        ts, bins=2, return_details=True, checkpoint_path=ckpt,
        bearer_nodes=[1, 2, 3], cut_mode="directional",
    )
    assert directional["checkpoint_resumed"] is False
