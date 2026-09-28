"""
Independent brute-force reference for the exact IIM (review of stream I2).

``compute_IIM_from_tpm`` and both Psi kernels (numba host kernel, array-module
xp kernel) are checked against a direct transcription of the definitions,
written with plain loops over state tuples and sharing no code with
``mpc_metrics`` or ``iim_xp``:

* effect repertoire p(Z'|M=m): next-state distribution averaged uniformly over
  the current states consistent with m, marginalised to Z;
* cause repertoire p(Z|M'=m): posterior over current states (uniform prior)
  from the likelihood P(M'=m | s), marginalised to Z;
* phi_e / phi_c(M, m) = max over purviews Z of the minimum Jensen-Shannon
  divergence (bits) between p(Z|m) and p(Z1|m1) x p(Z2|m2) over every
  mechanism bipartition and every ordered purview bipartition (|M|,|Z| >= 2);
* Psi = sum_M (1/|M|) sum_m w(m) min(phi_e, phi_c), w = state weights summed
  per mechanism state (zero-weight states excluded);
* bidirectional cut: T_cut = p_A(s'_A|s_A) p_B(s'_B|s_B), each part's marginal
  averaged uniformly over the other part's current state;
* directional cut (A, B): every unit j in B has its A-inputs replaced by
  independent uniform noise, p_j(s'_j|s) -> mean over s_A of p_j(s'_j|s);
* Delta_Psi = Psi_full - max over cuts of Psi_cut.
"""

import itertools
import math

import numpy as np
import pytest

from impact_pipeline import mpc_metrics as mm

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _states(n, k):
    return list(itertools.product(range(k), repeat=n))  # node 0 most significant


def _jsd_bits(p, q):
    p = np.asarray(p, float) / np.sum(p)
    q = np.asarray(q, float) / np.sum(q)
    out = 0.0
    for a, b in zip(p, q):
        m = 0.5 * (a + b)
        if a > 0:
            out += 0.5 * a * math.log2(a / m)
        if b > 0:
            out += 0.5 * b * math.log2(b / m)
    return out


def _marginal(dist, states, subset, k):
    vals = list(itertools.product(range(k), repeat=len(subset)))
    out = np.zeros(len(vals))
    for w, s in zip(dist, states):
        out[vals.index(tuple(s[x] for x in subset))] += w
    return out


def _effect(T, states, M, m, Z, k):
    rows = [i for i, s in enumerate(states) if tuple(s[x] for x in M) == m]
    return _marginal(T[rows].mean(axis=0), states, Z, k)


def _cause(T, states, M, m, Z, k):
    cols = [j for j, s in enumerate(states) if tuple(s[x] for x in M) == m]
    lik = T[:, cols].sum(axis=1)
    return _marginal(lik / lik.sum(), states, Z, k)


def _ordered_bipartitions(nodes):
    return [
        (a, tuple(x for x in nodes if x not in a))
        for r in range(1, len(nodes))
        for a in itertools.combinations(nodes, r)
    ]


def _phi(T, states, M, m, Z, k, rep):
    if len(M) < 2 or len(Z) < 2:
        return 0.0
    whole = rep(T, states, M, m, Z, k)
    zvals = list(itertools.product(range(k), repeat=len(Z)))
    best = np.inf
    for m1_nodes, m2_nodes in _ordered_bipartitions(M):
        m1 = tuple(m[M.index(x)] for x in m1_nodes)
        m2 = tuple(m[M.index(x)] for x in m2_nodes)
        for z1, z2 in _ordered_bipartitions(Z):
            p1 = rep(T, states, m1_nodes, m1, z1, k)
            p2 = rep(T, states, m2_nodes, m2, z2, k)
            v1 = list(itertools.product(range(k), repeat=len(z1)))
            v2 = list(itertools.product(range(k), repeat=len(z2)))
            q = np.array(
                [
                    p1[v1.index(tuple(z[Z.index(x)] for x in z1))]
                    * p2[v2.index(tuple(z[Z.index(x)] for x in z2))]
                    for z in zvals
                ]
            )
            best = min(best, _jsd_bits(whole, q))
    return best


def _psi(T, n, k, weights):
    states = _states(n, k)
    subsets = [
        c for r in range(1, n + 1) for c in itertools.combinations(range(n), r)
    ]
    total = 0.0
    for M in subsets:
        mech_w = {}
        for w, s in zip(weights, states):
            if w > 0:
                key = tuple(s[x] for x in M)
                mech_w[key] = mech_w.get(key, 0.0) + w
        norm = sum(mech_w.values())
        acc = 0.0
        for m, w in mech_w.items():
            phi_e = max(_phi(T, states, M, m, Z, k, _effect) for Z in subsets)
            phi_c = max(_phi(T, states, M, m, Z, k, _cause) for Z in subsets)
            acc += w / norm * min(phi_e, phi_c)
        total += acc / len(M)
    return total


def _unit_probs(T, n, k):
    states = _states(n, k)
    P = np.zeros((len(states), n, k))
    for i in range(len(states)):
        for j, s2 in enumerate(states):
            for u in range(n):
                P[i, u, s2[u]] += T[i, j]
    return P


def _tpm_from_units(P, n, k):
    states = _states(n, k)
    return np.array(
        [
            [np.prod([P[i, u, s2[u]] for u in range(n)]) for s2 in states]
            for i in range(len(states))
        ]
    )


def _cut_bidirectional(T, n, k, A, B):
    states = _states(n, k)

    def part(nodes):
        vals = list(itertools.product(range(k), repeat=len(nodes)))
        out = np.zeros((len(vals), len(vals)))
        for i, s in enumerate(states):
            a = vals.index(tuple(s[x] for x in nodes))
            for j, s2 in enumerate(states):
                out[a, vals.index(tuple(s2[x] for x in nodes))] += T[i, j]
        return vals, out / out.sum(axis=1, keepdims=True)

    va, pa = part(A)
    vb, pb = part(B)

    def code(vals, state, nodes):
        return vals.index(tuple(state[x] for x in nodes))

    return np.array(
        [
            [
                pa[code(va, s, A), code(va, s2, A)]
                * pb[code(vb, s, B), code(vb, s2, B)]
                for s2 in states
            ]
            for s in states
        ]
    )


def _cut_directional(T, n, k, A, B):
    states = _states(n, k)
    P = _unit_probs(T, n, k)
    Q = P.copy()
    for i, s in enumerate(states):
        same_b = [
            i2 for i2, s2 in enumerate(states) if all(s2[x] == s[x] for x in B)
        ]
        for u in B:
            Q[i, u] = P[same_b, u].mean(axis=0)
    return _tpm_from_units(Q, n, k)


def _reference(T, n, k, weights, cut_mode):
    nodes = tuple(range(n))
    cuts = []
    for r in range(1, n // 2 + 1):
        for a in itertools.combinations(nodes, r):
            b = tuple(x for x in nodes if x not in a)
            if r == n - r and a[0] > b[0]:
                continue
            cuts.append((a, b))
    if cut_mode == "directional":
        cuts = [c for a, b in cuts for c in ((a, b), (b, a))]
    build = _cut_bidirectional if cut_mode == "bidirectional" else _cut_directional
    scores = {
        f"{','.join(map(str, a))}|{','.join(map(str, b))}": _psi(
            build(T, n, k, a, b), n, k, weights
        )
        for a, b in cuts
    }
    full = _psi(T, n, k, weights)
    return full, scores, full - max(scores.values())


def _random_state_by_node(rng, n, k):
    P = rng.random((k ** n, n, k)) ** 2
    P /= P.sum(axis=2, keepdims=True)
    return _tpm_from_units(P, n, k)


def _random_weights(rng, n_states):
    w = rng.random(n_states)
    w[rng.integers(n_states)] = 0.0  # one state never visited
    return w / w.sum()


def test_state_order_is_big_endian():
    np.testing.assert_array_equal(mm.iim_state_table(3, 2), np.array(_states(3, 2)))
    np.testing.assert_array_equal(mm.iim_state_table(2, 3), np.array(_states(2, 3)))


@pytest.mark.parametrize("n,k,seed", [(2, 2, 0), (2, 3, 1), (3, 2, 2), (3, 2, 3)])
@pytest.mark.parametrize("cut_mode", ["bidirectional", "directional"])
@pytest.mark.parametrize("kernel", ["numba", "xp"])
def test_exact_iim_matches_the_definitions(n, k, seed, cut_mode, kernel):
    rng = np.random.default_rng(seed)
    T = _random_state_by_node(rng, n, k)
    w = _random_weights(rng, k ** n)
    full, scores, delta = _reference(T, n, k, w, cut_mode)
    got = mm.compute_IIM_from_tpm(
        T, state_weights=w, base=k, cut_mode=cut_mode, psi_kernel=kernel,
        return_details=True,
    )
    assert full > 1e-3  # a non-trivial system
    assert got["Psi_full"] == pytest.approx(full, rel=1e-10, abs=1e-14)
    assert got["Delta_Psi"] == pytest.approx(delta, rel=1e-10, abs=1e-14)
    assert set(got["cut_scores"]) == set(scores)
    for key, val in scores.items():
        assert got["cut_scores"][key] == pytest.approx(val, rel=1e-10, abs=1e-14), key


@pytest.mark.parametrize("kernel", ["numba", "xp"])
def test_exact_iim_of_a_joint_tpm_matches_the_definitions(kernel):
    # Not conditionally independent: only the bidirectional cut family applies.
    rng = np.random.default_rng(5)
    T = rng.random((8, 8)) ** 3
    T /= T.sum(axis=1, keepdims=True)
    w = _random_weights(rng, 8)
    full, scores, delta = _reference(T, 3, 2, w, "bidirectional")
    got = mm.compute_IIM_from_tpm(
        T, state_weights=w, psi_kernel=kernel, return_details=True
    )
    assert got["tpm_state_by_node"] is False
    assert got["Psi_full"] == pytest.approx(full, rel=1e-10, abs=1e-14)
    assert got["Delta_Psi"] == pytest.approx(delta, rel=1e-10, abs=1e-14)


def test_recurrent_xor_loop_documents_that_zero_does_not_imply_feedforward():
    """
    s0' = s1 XOR s2, s1' = s0, s2' = s1 is strongly connected (0 <-> 1, 1 -> 2
    -> 0), yet its directional IIM is exactly 0: the cut severing 0 -> {1, 2}
    preserves Psi. Observed with both the pipeline and the reference, so it is
    a property of the Psi proxy (mechanism-level min(phi_e, phi_c) sums), not
    an implementation error. Feedforward => Delta_Psi = 0 holds; the converse
    does not.
    """
    states = _states(3, 2)
    P = np.zeros((8, 3, 2))
    for i, s in enumerate(states):
        for u, bit in enumerate((s[1] ^ s[2], s[0], s[1])):
            P[i, u, bit] = 1.0
    T = _tpm_from_units(P, 3, 2)
    w = np.full(8, 1.0 / 8.0)
    for mode in ("bidirectional", "directional"):
        full, scores, delta = _reference(T, 3, 2, w, mode)
        got = mm.compute_IIM_from_tpm(T, cut_mode=mode, return_details=True)
        assert got["state_weights"] == pytest.approx(list(w))  # permutation chain
        assert got["Psi_full"] == pytest.approx(full, abs=1e-12)
        assert got["Delta_Psi"] == pytest.approx(delta, abs=1e-12)
    assert got["Delta_Psi"] == pytest.approx(0.0, abs=1e-12)
    assert got["cut_scores"]["0|1,2"] == pytest.approx(got["Psi_full"], abs=1e-12)
    bidir = mm.compute_IIM_from_tpm(T, return_details=True)
    assert bidir["Delta_Psi"] > 0.1


@pytest.mark.parametrize("rate", [1e-6, 1e-10, 1e-14, 1e-15])
def test_stationary_distribution_of_slowly_mixing_chains(rate):
    """The stopping rule must not mistake a slow mode for convergence."""
    a, b = rate, 3.0 * rate
    T = np.array([[1.0 - a, a], [b, 1.0 - b]])
    np.testing.assert_allclose(
        mm.iim_stationary_distribution(T), [0.75, 0.25], atol=1e-9
    )
    # reducible chain (two closed classes): the uniform start keeps its split
    np.testing.assert_allclose(
        mm.iim_stationary_distribution(np.eye(3)), [1 / 3] * 3, atol=1e-12
    )
    # transient state 0 drains into the absorbing states 1 and 2 (1:3)
    T3 = np.array([[0.2, 0.2, 0.6], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(
        mm.iim_stationary_distribution(T3),
        [0.0, 1 / 3 + 1 / 3 * 0.25, 1 / 3 + 1 / 3 * 0.75],
        atol=1e-12,
    )
