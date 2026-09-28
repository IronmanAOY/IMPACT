"""Known-answer tests for the exploratory legacy statistic S (HypergraphSynergy)."""

import numpy as np
import pytest

from impact_pipeline.utils import HypergraphSynergy


def _block_series(n_time=400, block=8, n_noise=12, seed=0, amp=1.0):
    rng = np.random.RandomState(seed)
    shared = rng.randn(n_time) * amp
    ts = np.column_stack([shared] * block + [rng.randn(n_time) for _ in range(n_noise)])
    return ts, shared


def test_known_answer_single_clique():
    ts, shared = _block_series()
    n_nodes = ts.shape[1]
    theta = 0.5  # noise correlations are ~N(0, 1/400): far below 0.5
    S = HypergraphSynergy.compute(ts, theta)
    E = 8 * 7 // 2  # all block pairs, no other edges
    integ = np.var(shared)  # I: pair means of identical signals
    Bc = 8 / n_nodes  # one component of size 8 (>5)
    Bal = 1 - abs(E - 1) / (E + 1)
    assert S == pytest.approx(integ * Bc * Bal, rel=1e-12)
    # Documented behaviour: Bal ~ 2G/E, so S ~ 2*I*Bc/E (inverse edge count).
    assert S == pytest.approx(2 * integ * Bc / (E + 1), rel=1e-12)


def test_small_cliques_and_no_edges_give_zero():
    ts, _ = _block_series(block=5)
    # A component of 5 nodes is not "large".
    assert HypergraphSynergy.compute(ts, 0.5) == 0.0
    rng = np.random.RandomState(1)
    assert HypergraphSynergy.compute(rng.randn(400, 10), 0.9) == 0.0


def test_invariant_to_node_order_and_scales_with_amplitude_squared():
    ts, _ = _block_series(seed=2)
    perm = np.random.RandomState(3).permutation(ts.shape[1])
    s0 = HypergraphSynergy.compute(ts, 0.5)
    assert HypergraphSynergy.compute(ts[:, perm], 0.5) == pytest.approx(s0, rel=1e-12)
    assert HypergraphSynergy.compute(ts * 10.0, 0.5) == pytest.approx(
        100.0 * s0, rel=1e-9
    )


def test_s_decreases_with_more_suprathreshold_edges():
    # A larger clique has more edges (66 vs 28):
    # Balance ~ 2/E shrinks faster than Bc grows.
    rng = np.random.RandomState(4)
    shared = rng.randn(500)
    one = np.column_stack([shared] * 8 + [rng.randn(500) for _ in range(8)])
    two = np.column_stack([shared] * 12 + [rng.randn(500) for _ in range(4)])
    assert HypergraphSynergy.compute(two, 0.5) < HypergraphSynergy.compute(one, 0.5)
