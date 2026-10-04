# -*- coding: utf-8 -*-
"""PDI v3 (pdi-v3-2026.10): the Fisher-axis valley, the content bearer and a
mis-declared access node, split-offset averaging, the concordance flag
(never an exact path), the null and SE fields, and the declared
non-applicability on the family-C carrier recording. All fixtures are
synthetic."""
import numpy as np
import pytest

from impact_pipeline import mpc_metrics as mm
from impact_pipeline.bench.export import block_keep
from impact_pipeline.v2 import ESTIMATOR_VERSIONS_V2
from impact_pipeline.v2 import pdi_v3 as PDI
from impact_pipeline.v2 import reasons as R
from impact_pipeline.v2 import records as REC

FAST = {"n_null": 2, "max_states": 4}


def segments(rng, n_time, lo=40, hi=120):
    """Alternating 0/1 labels in runs of ``lo``-``hi`` samples."""
    lab = np.empty(n_time, dtype=int)
    t, s = 0, 0
    while t < n_time:
        n = int(rng.integers(lo, hi))
        lab[t:t + n] = s
        s = 1 - s
        t += n
    return lab


def two_state_run(seed, n_nodes=3, n_time=2000, sep=2.5):
    """Two recurring states (two Gaussian clusters of window patterns) with
    dwell times of 8-24 windows."""
    rng = np.random.default_rng(seed)
    lab = segments(rng, n_time)
    pats = np.zeros((2, n_nodes))
    pats[0, : n_nodes // 2 + 1] = sep
    pats[1, n_nodes // 2 + 1:] = sep
    return pats[lab].T + rng.normal(size=(n_nodes, n_time))


def continuum_run(seed, n_nodes=3, n_time=2000, amp=1.5):
    """A slow Gaussian latent (AR(1), time constant 100 samples) projected on
    one pattern: a continuum of the same variance without attractors."""
    rng = np.random.default_rng(seed)
    a = np.exp(-1 / 100.0)
    e = rng.normal(size=n_time) * np.sqrt(1 - a * a)
    z = np.zeros(n_time)
    for t in range(1, n_time):
        z[t] = a * z[t - 1] + e[t]
    pat = np.linspace(-1, 1, n_nodes)
    return amp * pat[:, None] * z[None, :] + rng.normal(size=(n_nodes, n_time))


def ignition_run(seed, n_time=2000):
    """Nodes 0-1 (the access nodes) alternate between an ignited and a quiet
    state; nodes 2-3 carry one content state (noise only)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(4, n_time))
    x[:2] += 3.0 * segments(rng, n_time)[None, :]
    return x


def noise_run(seed, n_nodes=4, n_time=2000):
    return np.random.default_rng(seed).normal(size=(n_nodes, n_time))


# --------------------------------------------------------------------------
# declarations
# --------------------------------------------------------------------------
def test_declarations_and_bench_defaults():
    assert PDI.ESTIMATOR_VERSION == "pdi-v3-2026.10"
    assert PDI.ESTIMATOR_VERSION in ESTIMATOR_VERSIONS_V2["PDI"]
    p = PDI.PDIParams()
    assert (p.window, p.features, p.n_components, p.n_blocks, p.gap) == (
        5, "mean", 10, 6, 1)
    assert p.offsets == (0.0, 0.25, 0.5, 0.75)
    assert (p.max_states, p.n_init, p.min_state_windows) == (12, 8, 3)
    assert (p.min_dwell, p.valley, p.bandwidth, p.n_null) == (2.5, 0.4, 0.1, 19)
    assert p.bearer == "non_workspace"
    assert p.n_counts == 44
    assert PDI.JACKKNIFE_GROUPS == 10
    assert PDI.SE_METHOD == "jackknife_contiguous_10"
    # the status rule dispatches the concordance route on this method name
    assert PDI.SE_METHOD_CONCORDANT == "concordant"
    assert PDI.SE_METHODS == (PDI.SE_METHOD, PDI.SE_METHOD_CONCORDANT)
    assert PDI.NULL_FAMILY == "circular_shift"
    assert PDI.PDIParams.from_mapping(p.to_dict()) == p
    for bad in ({"offsets": (0.0, 1.0)}, {"bearer": "hub"}, {"window": 2.5},
                {"features": "phase"}):
        with pytest.raises(ValueError):
            PDI.PDIParams(**bad)
    with pytest.raises(ValueError):
        PDI.PDIParams.from_mapping({"exact": True})


# --------------------------------------------------------------------------
# geometry: Ledoit-Wolf, Fisher axis, valley
# --------------------------------------------------------------------------
def test_ledoit_wolf_matches_the_textbook_formula():
    rng = np.random.default_rng(0)
    for n, p in ((7, 3), (50, 5), (400, 10), (30, 1)):
        x = rng.normal(size=(n, p)) @ rng.normal(size=(p, p)) + 2.0
        cov, a = PDI.ledoit_wolf(x)
        xc = x - x.mean(axis=0)
        s = xc.T @ xc / n
        if p == 1:
            assert a == 0.0 and np.allclose(cov, s)
            continue
        mu = np.trace(s) / p
        d2 = np.sum((s - mu * np.eye(p)) ** 2)
        b2 = sum(np.sum((np.outer(v, v) - s) ** 2) for v in xc) / n ** 2
        a_want = min(b2, d2) / d2
        assert a == pytest.approx(a_want, rel=1e-10)
        assert 0.0 <= a <= 1.0
        assert np.allclose(cov, (1 - a_want) * s + a_want * mu * np.eye(p))
        assert np.all(np.linalg.eigvalsh(cov) > 0)


def test_the_fisher_axis_is_affine_invariant():
    rng = np.random.default_rng(1)
    c_i, c_j = rng.normal(size=3), rng.normal(size=3)
    s = np.cov(rng.normal(size=(3, 50)))
    pts = rng.normal(size=(20, 3))
    w, den = PDI.fisher_axis(c_i, c_j, np.linalg.inv(s))
    t = (pts - c_i) @ w / den
    assert ((c_j - c_i) @ w) / den == pytest.approx(1.0)
    a, b = rng.normal(size=(3, 3)), rng.normal(size=3)
    w2, den2 = PDI.fisher_axis(a @ c_i + b, a @ c_j + b, np.linalg.inv(a @ s @ a.T))
    t2 = (pts @ a.T + b - (a @ c_i + b)) @ w2 / den2
    assert np.allclose(t, t2)


def _gaussian_states(rng, n_windows, cov, centres, run=20):
    lab = np.repeat(np.arange(n_windows // run) % len(centres), run)
    x = np.asarray(centres)[lab] + rng.multivariate_normal(np.zeros(2), cov, lab.size)
    return x, lab


def test_the_fisher_axis_valley_separates_states_with_shared_covariance():
    """Two Gaussian states whose shared within-state covariance is elongated
    across the centroid axis: the v1 centroid-axis valley is filled, the
    Fisher-axis valley is not; one Gaussian cut in two fails both."""
    rng = np.random.default_rng(2)
    cov = 0.16 * np.array([[1.0, 0.98], [0.98, 1.0]])
    centres = [[0.0, 0.0], [1.0, 0.0]]
    x_fit, lab_fit = _gaussian_states(rng, 400, cov, centres)
    x_test, lab_test = _gaussian_states(rng, 400, cov, centres)
    block = np.zeros(lab_test.size, dtype=np.int64)
    c = np.stack([x_fit[lab_fit == k].mean(axis=0) for k in (0, 1)])
    p = PDI.PDIParams()
    fisher_ok, fisher_ratio, dwell = PDI.partition_check(c, x_fit, lab_fit, x_test,
                                                         block, p)
    v1_ok, v1_ratio, _ = mm._pdi_partition_check(c, x_test, block, p.valley,
                                                 p.min_dwell)
    assert fisher_ok and fisher_ratio <= p.valley and dwell >= p.min_dwell
    assert not v1_ok and v1_ratio > p.valley
    # one Gaussian state split at its median: no valley on any axis
    one, _ = _gaussian_states(rng, 400, cov, [[0.0, 0.0]])
    cut = (one[:, 0] > np.median(one[:, 0])).astype(int)
    c1 = np.stack([one[cut == k].mean(axis=0) for k in (0, 1)])
    one_test, _ = _gaussian_states(rng, 400, cov, [[0.0, 0.0]])
    ok, ratio, _ = PDI.partition_check(c1, one, cut, one_test,
                                       np.zeros(400, dtype=np.int64),
                                       PDI.PDIParams(min_dwell=1.0))
    assert not ok


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_two_gaussian_states_count_two_and_a_continuum_counts_one(seed):
    two = PDI.split_averaged_bits(two_state_run(seed), None, seed=0)
    assert two["counts"] == [2, 2, 2, 2] and two["bits"] == 1.0
    one = PDI.split_averaged_bits(continuum_run(seed), None, seed=0)
    assert one["counts"] == [1, 1, 1, 1] and one["bits"] == 0.0
    res = PDI.count_states(two_state_run(seed), offset=0.0, seed=0, describe=True)
    k2 = [d for d in res["per_k"] if d["k"] == 2][0]
    assert k2["passed"] and k2["valley_ratio"] <= 0.4
    assert res["n_components"] == 3 and res["k_max"] == 12


# --------------------------------------------------------------------------
# content bearer
# --------------------------------------------------------------------------
def test_the_content_bearer_excludes_the_declared_workspace_nodes():
    rows, reading, excluded = PDI.content_bearer(30, [12, 13, 14, 15, 16, 17])
    assert rows.tolist() == list(range(12)) + list(range(18, 30))
    assert reading == "content" and excluded == [12, 13, 14, 15, 16, 17]
    rows, reading, excluded = PDI.content_bearer(30, None)
    assert rows.tolist() == list(range(30)) and reading == "upper_bound"
    rows, reading, _ = PDI.content_bearer(30, [12, 13], bearer="full")
    assert rows.tolist() == list(range(30)) and reading == "upper_bound"
    with pytest.raises(ValueError):
        PDI.content_bearer(30, [30])


def test_access_declaration_and_a_mis_declared_access_node():
    x = ignition_run(0)
    right = PDI.compute_pdi_v3(x, workspace_nodes=[0, 1], params=FAST)
    assert right["defined"], right["reason"]
    assert right["details"]["bearer_nodes"] == [2, 3]
    assert right["details"]["bearer_reading"] == "content"
    assert right["details"]["counted_bearer"] == "non_workspace"
    assert right["details"]["content_specific"] is True
    assert right["estimate"] == 0.0  # the ignition is not a content state
    full = right["details"]["full_bearer"]
    assert full["bits"] == 1.0 and full["reading"] == "upper_bound"
    assert full["status"] == "never"  # point value only
    # the wrong declaration (content nodes declared as the access nodes)
    wrong = PDI.compute_pdi_v3(x, workspace_nodes=[2, 3], params=FAST)
    assert wrong["details"]["bearer_nodes"] == [0, 1]
    assert wrong["estimate"] == 1.0  # the access state is now counted
    # a broken access node spoils only the reported upper bound
    broken = x.copy()
    broken[0, 100] = np.nan
    res = PDI.compute_pdi_v3(broken, workspace_nodes=[0, 1], params=FAST)
    assert res["defined"] and res["estimate"] == 0.0
    assert res["details"]["full_bearer"]["undefined_reason"] == "non_finite_timeseries"
    # no declaration: only the upper-bound reading exists; the protocol's
    # declared bearer stays non_workspace, but the full bearer was counted,
    # so the route's bearer cell must be the full one
    none = PDI.compute_pdi_v3(x, workspace_nodes=None, params=FAST)
    assert none["details"]["bearer_reading"] == "upper_bound"
    assert none["details"]["bearer"] == "non_workspace"
    assert none["details"]["counted_bearer"] == "full"
    assert none["details"]["content_specific"] is False
    assert none["details"]["bearer_nodes"] == [0, 1, 2, 3]
    assert "full_bearer" not in none["details"]
    full = PDI.compute_pdi_v3(x, workspace_nodes=[0, 1],
                              params=dict(FAST, bearer="full"))
    assert full["details"]["counted_bearer"] == "full"
    assert full["details"]["bearer_nodes"] == [0, 1, 2, 3]
    assert full["estimate"] == none["estimate"]


# --------------------------------------------------------------------------
# split offsets
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n_win", [12, 61, 600, 2372])
def test_the_unrotated_split_is_the_v1_split(n_win):
    half, block = PDI.split_halves(n_win, 6, 1, 0.0)
    v1_half, v1_block = mm._pdi_half_blocks(n_win, 6, 1)
    assert np.array_equal(half, v1_half) and np.array_equal(block, v1_block)


@pytest.mark.parametrize("offset", [0.0, 0.25, 0.5, 0.75])
@pytest.mark.parametrize("n_win", [61, 600, 2372])
def test_rotated_splits_keep_the_halves_apart(offset, n_win):
    half, block = PDI.split_halves(n_win, 6, 1, offset)
    sh = int(round(offset * n_win / 6))
    for t in range(n_win - 1):
        if half[t] >= 0 and half[t + 1] >= 0:
            assert half[t] == half[t + 1]  # no window next to the other half
    for b in range(6):
        assert set(half[block == b].tolist()) == {b % 2}
    assert int(np.sum(half < 0)) == (6 if sh > 0 else 5)
    if sh > 0:
        assert half[sh] == -1  # the first window of the rotated block 0
        v1_half, _ = mm._pdi_half_blocks(n_win, 6, 1)
        assert not np.array_equal(half, v1_half)


@pytest.mark.parametrize("offset", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("n_win", [61, 600])
def test_dwell_never_pairs_windows_across_the_wrapped_block(offset, n_win):
    """The rotation wraps the last block around the end of the run (its
    windows sit at both ends). The dwell check pairs consecutive rows of a
    half that share a block, so every such pair must be adjacent in time."""
    half, block = PDI.split_halves(n_win, 6, 1, offset)
    wrapped = block[0]
    assert wrapped == 5 and block[n_win - 1] == wrapped
    for h in (0, 1):
        win = np.flatnonzero(half == h)
        blk = block[win]
        same = blk[1:] == blk[:-1]
        assert np.all(np.diff(win)[same] == 1)


def test_b_is_the_mean_over_offsets_of_log2_counts(monkeypatch):
    per_offset = {0.0: 6, 0.25: 5, 0.5: 6, 0.75: 4}

    def fake(x, *, offset=0.0, params=None, seed=0, describe=False):
        k = per_offset[offset]
        return {"n_states": k, "bits": float(np.log2(k)), "k_max": 12,
                "n_windows": 100, "n_components": 3, "offset": offset,
                "undefined_reason": None}

    monkeypatch.setattr(PDI, "count_states", fake)
    res = PDI.split_averaged_bits(np.zeros((3, 10)))
    assert res["counts"] == [6, 5, 6, 4]
    assert res["bits"] == pytest.approx(np.mean(np.log2([6, 5, 6, 4])))


# --------------------------------------------------------------------------
# concordance, SE and the never-exact rule
# --------------------------------------------------------------------------
def test_the_concordance_flag_needs_all_44_counts_to_agree():
    assert PDI.counts_concordant([[3] * 4] * 11, expected=44)
    differing = [[3] * 4 for _ in range(11)]
    differing[7][2] = 4
    assert not PDI.counts_concordant(differing, expected=44)
    assert not PDI.counts_concordant([[3] * 4] * 10, expected=44)
    undefined = [[3] * 4 for _ in range(11)]
    undefined[0][0] = None
    assert not PDI.counts_concordant(undefined, expected=44)


class _CountScript:
    """``count_states`` replacement: every count is 1 except those listed by
    call number (1-based), in the estimator's call order."""

    def __init__(self, special, default=1):
        self.special, self.default, self.n = dict(special), default, 0

    def __call__(self, x, *, offset=0.0, params=None, seed=0, describe=False):
        self.n += 1
        k = self.special.get(self.n, self.default(offset) if callable(self.default)
                             else self.default)
        out = {"n_states": k, "bits": float(np.log2(k)), "k_max": 4,
               "n_windows": x.shape[1] // 5, "n_components": 3, "offset": offset,
               "undefined_reason": None}
        if describe:
            out["per_k"] = []
        return out


def _scripted(monkeypatch, special, default=1):
    script = _CountScript(special, default)
    monkeypatch.setattr(PDI, "count_states", script)
    res = PDI.compute_pdi_v3(noise_run(0), params=FAST)
    return res, script


def test_concordant_counts_give_se_zero_with_their_own_method(monkeypatch):
    # observed counts are calls 1-4, the two surrogates 5-12, the jackknife
    # replicates 13-52 (no access nodes declared, so no full-bearer counts)
    res, script = _scripted(monkeypatch, {})
    assert script.n == 4 + 2 * 4 + 10 * 4
    assert res["concordant"] is True and res["details"]["n_concordance_counts"] == 44
    assert res["se"] == 0.0 and res["se_method"] == PDI.SE_METHOD_CONCORDANT
    assert res["se_df"] is None  # no sampling SE, so no df either
    assert res["details"]["se_jackknife"] == 0.0
    assert res["details"]["se_df_jackknife"] == 9.0
    assert res["exact"] is False
    # a surrogate count does not enter the concordance rule
    res, _ = _scripted(monkeypatch, {6: 2})
    assert res["concordant"] is True and res["null_mean"] > 0
    # one replicate count out of 44 breaks it
    res, _ = _scripted(monkeypatch, {20: 2})
    assert res["concordant"] is False
    assert res["se_method"] == PDI.SE_METHOD and res["se"] > 0
    assert res["se_df"] == 9.0
    assert res["exact"] is False
    # so does one observed count
    res, _ = _scripted(monkeypatch, {3: 2})
    assert res["concordant"] is False and res["se_method"] == PDI.SE_METHOD


def test_a_zero_se_without_concordance_is_not_marked_concordant(monkeypatch):
    # counts differ between offsets but every data set gives the same B: the
    # jackknife SE is 0, the component is not concordant and keeps the plain
    # jackknife method, so the status rule reads it as NO_SAMPLING_SE
    res, _ = _scripted(monkeypatch, {}, default=lambda o: 2 if o == 0.0 else 1)
    assert res["se"] == 0.0
    assert res["concordant"] is False and res["se_method"] == PDI.SE_METHOD
    assert res["exact"] is False


def test_a_concordant_component_is_never_exact():
    res = PDI.compute_pdi_v3(noise_run(1), params=FAST)
    assert res["defined"], res["reason"]
    assert res["details"]["counts"] == [1, 1, 1, 1]
    assert res["concordant"] is True and res["se"] == 0.0
    assert res["se_method"] == PDI.SE_METHOD_CONCORDANT and res["se_df"] is None
    assert res["exact"] is False
    fields = PDI.component_fields(res)
    assert fields["se"] == 0.0 and fields["se_method"] == PDI.SE_METHOD_CONCORDANT
    assert fields["se_df"] is None
    assert fields["details"]["exact"] is False and fields["details"]["concordant"]
    assert "exact" not in PDI.PDIParams().to_dict()  # no exact path to declare
    rec = REC.ComponentRecord(principle="PDI", status=R.UNDEFINED,
                              reason=R.NO_SAMPLING_SE,
                              estimator_version=PDI.ESTIMATOR_VERSION,
                              declaration_id="R", observation_stage="source",
                              protocol_id="A-R", protocol_hash="0" * 64, **fields)
    assert REC.ComponentRecord.from_dict(rec.to_dict()) == rec


def test_null_and_jackknife_fields():
    x = two_state_run(0)
    params = {"n_null": 3, "max_states": 4}
    res = PDI.compute_pdi_v3(x, params=params, null_seed=7)
    assert res["defined"], res["reason"]
    assert res["estimate"] == 1.0
    assert res["n_null"] == 3 and res["null_family"] == "circular_shift"
    null = np.asarray(res["details"]["null_values"])
    assert null.size == 3 and np.all(np.isfinite(null))
    assert res["null_mean"] == pytest.approx(null.mean())
    assert res["null_sd"] == pytest.approx(null.std(ddof=1))
    assert res["value"] == pytest.approx(res["estimate"] - null.mean())
    reps = np.asarray(res["details"]["jackknife_bits"])
    assert reps.size == 10 and len(res["details"]["jackknife_counts"]) == 10
    se = np.sqrt(0.9 * np.sum((reps - reps.mean()) ** 2))
    assert res["details"]["se_jackknife"] == (0.0 if se < 1e-9 else pytest.approx(se))
    assert res["details"]["se_df_jackknife"] == 9.0
    if res["concordant"]:  # no sampling SE on the concordance route
        assert res["se"] == 0.0 and res["se_df"] is None
    else:
        assert res["se"] == res["details"]["se_jackknife"] and res["se_df"] == 9.0
    assert len(res["details"]["state_occupancy"]) == 2
    # the replicates use the v1 contiguous-block rule
    for g in range(10):
        keep = PDI.jackknife_keep(x.shape[1], 10, g)
        assert np.array_equal(keep, block_keep(x.shape[1], 10, g))
    again = PDI.compute_pdi_v3(x, params=params, null_seed=7)
    assert again["details"]["null_values"] == res["details"]["null_values"]
    assert again["se"] == res["se"]


def test_partitions_for_the_oracle_ami_are_optional():
    x = ignition_run(1)
    res = PDI.compute_pdi_v3(x, workspace_nodes=[0, 1], params=FAST,
                             return_partition=True, dt=0.05)
    part = res["details"]["full_bearer"]["partition"]
    assert part["n_states"] == 2 and part["window_samples"] == 5
    assert len(part["labels"]) == x.shape[1] // 5
    assert res["details"]["partition"]["n_states"] == 1
    assert res["details"]["min_countable_dwell_sec"] == pytest.approx(0.625)
    assert "partition" not in PDI.compute_pdi_v3(x, workspace_nodes=[0, 1],
                                                 params=FAST)["details"]


# --------------------------------------------------------------------------
# definedness and family C1
# --------------------------------------------------------------------------
def test_undefined_inputs_carry_vocabulary_reasons():
    x = noise_run(2)
    cases = {
        "NOT_DEFINED:insufficient_regions": x[:1],
        "NOT_DEFINED:insufficient_timepoints": x[:, :9],
        "NOT_DEFINED:no_variance": np.ones((4, 500)),
        "NOT_DEFINED:insufficient_windows": x[:, :40],
    }
    nan = x.copy()
    nan[1, 3] = np.nan
    cases["NOT_DEFINED:non_finite_timeseries"] = nan
    for reason, data in cases.items():
        res = PDI.compute_pdi_v3(data, params=FAST)
        assert res["reason"] == reason
        assert not res["defined"] and res["exact"] is False
        assert R.status_for_reason(reason) == R.UNDEFINED
    # a declaration that leaves fewer than two content nodes
    res = PDI.compute_pdi_v3(x, workspace_nodes=[0, 1, 2], params=FAST)
    assert res["reason"] == "NOT_DEFINED:insufficient_regions"


@pytest.mark.parametrize("model", ["C1", "C"])
def test_family_c1_is_declared_not_applicable(model):
    x = two_state_run(3)
    res = PDI.compute_pdi_v3(x, params=FAST, observation_model=model)
    assert res["reason"] == R.NOT_APPLICABLE_OBSERVATION_MODEL
    assert not res["defined"] and res["n_null"] == 0 and res["se_method"] is None
    assert R.status_for_reason(res["reason"]) == R.UNDEFINED
    assert res["details"]["observation_model"] == "C1"
    assert "phase-coded" in res["details"]["not_applicable"]
    assert res["details"]["descriptive"]["counts"] == [2, 2, 2, 2]  # point value only
    fields = PDI.component_fields(res)
    assert fields["estimate"] is None and fields["se"] is None
    common = dict(principle="PDI", estimator_version=PDI.ESTIMATOR_VERSION,
                  declaration_id="R", observation_stage="source", protocol_id="C1-R",
                  protocol_hash="0" * 64, **fields)
    REC.ComponentRecord(status=R.UNDEFINED, reason=res["reason"], **common)
    for status in (R.PRESENT, R.ABSENT):
        with pytest.raises(REC.RecordSchemaError):
            REC.ComponentRecord(status=status, **common)
    assert PDI.not_applicable("A") is None
