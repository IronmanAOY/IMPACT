# -*- coding: utf-8 -*-
"""
Monte-Carlo check of the error-control argument E1-E4 of the status rule
``tost-v2`` (preregistration v2, section 3.5) on synthetic (c, se) drawn
under its assumption (A-SE): ``(c - theta) / se_c ~ t_df``. A draw is
``c = theta + sigma Z`` and ``se_c = sigma sqrt(X / df)`` with
``Z ~ N(0, 1)`` and ``X ~ chi2(df)`` independent, so the pivot is exactly
Student t with ``df`` degrees of freedom.

* E1: theta <= z gives P(PRESENT) <= alpha; for NAS, the intersection-union
  over the directions (one direction at the boundary).
* E2: |theta| >= delta gives P(ABSENT) <= alpha_A; for NAS, the union over
  the directions at alpha_A / 2 (the boundary case: both directions at
  delta), and the same union without the split exceeds alpha_A.
* E3: one principle of N_anch at theta = z gives P(MPC_CONSISTENT) <= alpha.
* E4: every principle of N_anch at |theta| >= delta gives
  P(EXCLUDED) <= |N_anch| alpha_A <= alpha, with independent and with fully
  dependent components and for an anchored subset.

The verdict is the strong-Kleene AND of the component codes; the last test
checks on the same draws that the full verdict path gives the same verdicts
as the vectorised computation, and that UNDEFINED never makes a verdict
EXCLUDED. Every rate is checked against its bound plus four Monte-Carlo
standard errors. Seeds are development seeds.
"""
import math

import numpy as np
import pytest

from impact_pipeline import evidence as v1
from impact_pipeline import evidence_v2 as E

Z, DELTA = 0.25, 0.10
ALPHA, ALPHA_A = 0.05, 0.01
DF = 9
N = 400_000
RULE = E.StatusRule()


def draw(rng, theta, sigma, n=N, df=DF, z=None, x=None):
    z = rng.standard_normal(n) if z is None else z
    x = rng.chisquare(df, n) if x is None else x
    return theta + sigma * z, sigma * np.sqrt(x / df)


def tol(p, n=N):
    return 4.0 * math.sqrt(p * (1.0 - p) / n)


def rate(codes, code):
    return float(np.mean(np.asarray(codes) == code))


def component(c, se):
    return E.status_c(c, se, DF, rule=RULE).codes


def nas(c_r, se_r, c_b, se_b, rule=RULE):
    return E.status_c_directional(np.stack([c_r, c_b], -1), np.stack([se_r, se_b], -1),
                                  DF, rule=rule).codes


def test_pivot_is_student_t():
    rng = np.random.default_rng(401)
    c, se = draw(rng, 0.0, 0.1)
    q = E.quantile(0.05, DF)
    assert abs(float(np.mean(c / se > q)) - 0.05) < tol(0.05)


# --------------------------------------------------------------------------
# E1 and E2: one component
# --------------------------------------------------------------------------
@pytest.mark.parametrize("theta, sigma, exact", [
    (Z, 0.10, True), (Z, 0.30, True), (0.20, 0.10, False), (0.0, 0.10, False),
    (-0.5, 0.20, False),
])
def test_e1_component_false_present(theta, sigma, exact):
    rng = np.random.default_rng(402)
    p = rate(component(*draw(rng, theta, sigma)), E.CODE_T)
    assert p <= ALPHA + tol(ALPHA)
    if exact:  # at the boundary the level is attained
        assert abs(p - ALPHA) < tol(ALPHA)


@pytest.mark.parametrize("theta, sigma, exact", [
    (DELTA, 0.005, True), (-DELTA, 0.005, True), (DELTA, 0.01, True),
    (DELTA, 0.02, False), (DELTA, 0.03, False), (0.15, 0.01, False),
    (-0.2, 0.02, False), (0.5, 0.05, False),
])
def test_e2_component_false_absent(theta, sigma, exact):
    rng = np.random.default_rng(403)
    p = rate(component(*draw(rng, theta, sigma)), E.CODE_F)
    assert p <= ALPHA_A + tol(ALPHA_A)
    if exact:  # the other one-sided test never fails: the level is attained
        assert abs(p - ALPHA_A) < tol(ALPHA_A)


def test_absent_only_where_the_tost_region_is_not_empty():
    """theta = 0 with sigma well above s_A = 0.035: ABSENT occurs only on
    draws whose estimated se_c is below s_A (q_A se_c < delta), and
    ABSENT_NOT_REACHABLE marks exactly the undecided draws above it (unless
    the null model is violated); NULL_MODEL_VIOLATED is rare at the null."""
    rng = np.random.default_rng(404)
    c, se = draw(rng, 0.0, 0.1)
    out = E.status_c(c, se, DF, rule=RULE)
    reach = out.q_absent * out.se < DELTA
    absent = out.codes == E.CODE_F
    assert np.all(reach[absent]) and rate(out.codes, E.CODE_F) < ALPHA_A
    und = out.codes == E.CODE_U
    assert all(r is not None for r in out.reason[und])
    anr = out.reason == "ABSENT_NOT_REACHABLE"
    nmv = out.reason == "NULL_MODEL_VIOLATED"
    assert np.array_equal(anr, und & ~reach & ~nmv)
    assert float(np.mean(nmv)) < ALPHA  # c + q_P se < -delta: beyond an alpha tail


# --------------------------------------------------------------------------
# NAS: the boundary case
# --------------------------------------------------------------------------
def test_e1_nas_intersection_union_present():
    rng = np.random.default_rng(405)
    c_r, s_r = draw(rng, Z, 0.1)
    c_b, s_b = draw(rng, 3.0, 0.1)
    p = rate(nas(c_r, s_r, c_b, s_b), E.CODE_T)  # theta_NAS = min = z
    assert p <= ALPHA + tol(ALPHA) and abs(p - ALPHA) < tol(ALPHA)
    c_b, s_b = draw(rng, Z, 0.1)
    assert rate(nas(c_r, s_r, c_b, s_b), E.CODE_T) <= ALPHA ** 2 + tol(ALPHA ** 2)


def test_e2_nas_union_absent_at_the_boundary():
    rng = np.random.default_rng(406)
    c_r, s_r = draw(rng, DELTA, 0.01)
    c_b, s_b = draw(rng, DELTA, 0.01)
    independent = rate(nas(c_r, s_r, c_b, s_b), E.CODE_F)
    want = 1.0 - (1.0 - ALPHA_A / 2) ** 2  # 0.009975
    assert independent <= ALPHA_A + tol(ALPHA_A)
    assert abs(independent - want) < tol(want)
    dependent = rate(nas(c_r, s_r, c_r, s_r), E.CODE_F)  # identical directions
    assert abs(dependent - ALPHA_A / 2) < tol(ALPHA_A / 2)
    c_b, s_b = draw(rng, 1.0, 0.01)  # only one direction near the margin
    one = rate(nas(c_r, s_r, c_b, s_b), E.CODE_F)
    assert abs(one - ALPHA_A / 2) < tol(ALPHA_A / 2)
    # the same union without the split over the directions exceeds alpha_A
    unsplit = E.directional_codes(np.stack([
        E.status_c(c_r, s_r, DF, rule=RULE).codes,
        E.status_c(draw(rng, DELTA, 0.01)[0], s_r, DF, rule=RULE).codes], -1))
    p_unsplit = rate(unsplit, E.CODE_F)
    assert p_unsplit > ALPHA_A + tol(ALPHA_A)
    assert abs(p_unsplit - (1.0 - (1.0 - ALPHA_A) ** 2)) < tol(0.02)


# --------------------------------------------------------------------------
# E3 and E4: verdicts
# --------------------------------------------------------------------------
def verdict_codes(thetas, sigmas, rng, mask=None, dependent=False):
    """Verdict codes over the five principles (NAS with two directions)."""
    z = rng.standard_normal(N) if dependent else None
    x = rng.chisquare(DF, N) if dependent else None
    cols = []
    for p in ("RAM", "PDI", "NAS", "IIM", "SRPI"):
        th, sg = thetas[p], sigmas[p]
        if p == "NAS":
            c_r, s_r = draw(rng, th[0], sg, z=z, x=x)
            c_b, s_b = draw(rng, th[1], sg, z=z, x=x)
            cols.append(nas(c_r, s_r, c_b, s_b))
        else:
            cols.append(component(*draw(rng, th, sg, z=z, x=x)))
    codes = np.stack(cols, -1)
    return E.kleene_verdict_codes(codes, mask), codes


@pytest.mark.parametrize("which", ["RAM", "NAS", "SRPI"])
def test_e3_verdict_false_consistent(which):
    rng = np.random.default_rng(407)
    thetas = {"RAM": 3.0, "PDI": 3.0, "NAS": (3.0, 3.0), "IIM": 3.0, "SRPI": 3.0}
    thetas[which] = (Z, 3.0) if which == "NAS" else Z
    sigmas = {p: 0.1 for p in thetas}
    v, _ = verdict_codes(thetas, sigmas, rng)
    p = rate(v, E.CODE_T)
    assert p <= ALPHA + tol(ALPHA) and abs(p - ALPHA) < tol(ALPHA)


@pytest.mark.parametrize("dependent", [False, True], ids=["independent", "dependent"])
def test_e4_verdict_false_excluded(dependent):
    rng = np.random.default_rng(408)
    thetas = {"RAM": DELTA, "PDI": -DELTA, "NAS": (DELTA, DELTA), "IIM": DELTA,
              "SRPI": DELTA}
    sigmas = {p: 0.01 for p in thetas}
    v, codes = verdict_codes(thetas, sigmas, rng, dependent=dependent)
    p = rate(v, E.CODE_F)
    assert p <= len(E.PRINCIPLES) * ALPHA_A + tol(ALPHA)
    if not dependent:
        want = 1.0 - (1 - ALPHA_A) ** 4 * (1 - ALPHA_A / 2) ** 2
        assert abs(p - want) < tol(want)
    # EXCLUDED iff some component is ABSENT: UNDEFINED never counts
    assert np.array_equal(v == E.CODE_F, np.any(codes == E.CODE_F, axis=-1))


def test_e4_holds_for_an_anchored_subset_and_any_anchoring_outcome():
    rng = np.random.default_rng(409)
    thetas = {"RAM": DELTA, "PDI": DELTA, "NAS": (DELTA, 2.0), "IIM": 0.3,
              "SRPI": DELTA}
    sigmas = {p: 0.01 for p in thetas}
    mask = np.array([True, False, True, False, True])  # N_anch = RAM, NAS, SRPI
    v, codes = verdict_codes(thetas, sigmas, rng, mask=mask)
    p = rate(v, E.CODE_F)
    assert p <= 3 * ALPHA_A + tol(3 * ALPHA_A)
    assert not np.any(v[np.all(codes[:, mask] != E.CODE_F, axis=-1)] == E.CODE_F)


def test_full_verdict_path_agrees_with_the_vectorised_rule():
    """On the same draws, verdict_v3 over evidence items gives the verdicts
    and component codes of the vectorised computation."""
    rng = np.random.default_rng(410)
    n = 1500
    ref = {"kind": "external", "scale": "excess",
           "values": {"RAM": 1.0, "PDI": 1.0, "IIM": 1.0, "SRPI": 1.0,
                      "NAS:receive": 1.0, "NAS:return": 1.0}}
    proto = E.ProtocolV3(reference=ref, directions={"NAS": ["receive", "return"]})
    draws = {}
    for key, theta in (("RAM", DELTA), ("PDI", 0.6), ("NAS:receive", DELTA),
                       ("NAS:return", Z), ("IIM", -DELTA), ("SRPI", 0.0)):
        draws[key] = draw(rng, theta, float(rng.choice([0.01, 0.05, 0.2])), n=n)
    cols = [component(*draws["RAM"]), component(*draws["PDI"]),
            nas(*draws["NAS:receive"], *draws["NAS:return"]),
            component(*draws["IIM"]), component(*draws["SRPI"])]
    codes = np.stack(cols, -1)
    want = E.kleene_verdict_codes(codes)
    code_of = {v1.Verdict.MPC_CONSISTENT: E.CODE_T, v1.Verdict.UNDETERMINED: E.CODE_U,
               v1.Verdict.EXCLUDED: E.CODE_F}
    status_code = {v1.ComponentStatus.PRESENT: E.CODE_T,
                   v1.ComponentStatus.UNDEFINED: E.CODE_U,
                   v1.ComponentStatus.ABSENT: E.CODE_F}

    def ev(p, key, i, direction=None):
        c, se = draws[key]
        return E.ComponentEvidenceV2(principle=p, estimate=float(c[i]), null_mean=0.0,
                                     se=float(se[i]), se_df=float(DF),
                                     direction=direction, protocol_id=proto.protocol_id)

    for i in range(n):
        evidence = {
            "RAM": ev("RAM", "RAM", i), "PDI": ev("PDI", "PDI", i),
            "NAS": [ev("NAS", "NAS:receive", i, "receive"),
                    ev("NAS", "NAS:return", i, "return")],
            "IIM": ev("IIM", "IIM", i), "SRPI": ev("SRPI", "SRPI", i),
        }
        verdict = E.mpc_verdict(evidence, proto)
        assert code_of[verdict.verdict] == want[i]
        got = [status_code[verdict.component_status[p]] for p in E.PRINCIPLES]
        assert got == list(codes[i])
        if verdict.verdict is v1.Verdict.EXCLUDED:
            assert any(r.startswith("ABSENT:") for r in verdict.reasons)
