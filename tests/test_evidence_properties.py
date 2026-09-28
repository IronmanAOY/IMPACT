"""
Randomised property tests of the three-valued MPC verdict (spec P1, P2', P3',
P4, permutation symmetry, channel disjunction, reason-code decomposability).

The Kleene core (``evidence.kleene_verdict_codes`` / ``kleene_or_codes``) is
checked on >= 1e5 numpy-seeded vectors per property; the object path
(``evidence.mpc_verdict`` on ComponentEvidence) is checked against that core,
for missingness safety, veto and reason-code decomposability on 1e5 random
evidence configurations. Counts are printed (``pytest -s``).
"""
import dataclasses
import itertools

import numpy as np
import pytest

from impact_pipeline import evidence as E

N_VEC = 200_000
N_OBJ = 100_000
J = len(E.PRINCIPLES)
T, U, F = E.CODE_T, E.CODE_U, E.CODE_F


def _statuses(rng, n, p=(0.45, 0.3, 0.25)):
    return rng.choice(np.array([T, U, F], dtype=np.int8), size=(n, J), p=p)


def _necessity_masks(rng, n):
    mask = rng.random((n, J)) < 0.7
    empty = ~mask.any(axis=1)
    mask[empty, rng.integers(0, J, size=int(empty.sum()))] = True
    return mask


def _verdict(s, mask):
    return E.kleene_verdict_codes(s, mask)


def _report(name, **counts):
    print(f"\n[{name}] " + ", ".join(f"{k}={v}" for k, v in counts.items()))


def test_P1_missingness_safety_vectorised():
    rng = np.random.default_rng(101)
    s, mask = _statuses(rng, N_VEC), _necessity_masks(rng, N_VEC)
    v = _verdict(s, mask)
    j = rng.integers(0, J, size=N_VEC)
    s2 = s.copy()
    s2[np.arange(N_VEC), j] = U
    v2 = _verdict(s2, mask)
    # never turns UNDETERMINED / NOT_ATTRIBUTED into ATTRIBUTED ...
    assert not np.any((v != T) & (v2 == T))
    # ... and never creates a determinate verdict that differs from the original
    assert not np.any((v2 != U) & (v2 != v))
    _report(
        "P1", vectors=N_VEC, changed_to_undetermined=int(np.sum(v2 != v)),
        determinate_kept=int(np.sum((v2 != U) & (v2 == v))),
    )


def test_P2prime_resolving_undefined_never_reverses_a_determinate_verdict():
    rng = np.random.default_rng(102)
    s, mask = _statuses(rng, N_VEC), _necessity_masks(rng, N_VEC)
    has_u = (s == U).any(axis=1)
    s, mask = s[has_u], mask[has_u]
    n = s.shape[0]
    v = _verdict(s, mask)
    # resolve one random undefined component to T or F
    u_pos = np.where(s == U, rng.random(s.shape), -1.0).argmax(axis=1)
    s2 = s.copy()
    s2[np.arange(n), u_pos] = rng.choice(np.array([T, F], dtype=np.int8), size=n)
    v2 = _verdict(s2, mask)
    det = v != U
    assert np.all(v2[det] == v[det])
    _report("P2'", vectors=n, determinate_before=int(det.sum()),
            newly_determinate=int(np.sum(~det & (v2 != U))))


def test_P3prime_determinate_iff_all_completions_agree():
    rng = np.random.default_rng(103)
    s, mask = _statuses(rng, N_VEC), _necessity_masks(rng, N_VEC)
    v = _verdict(s, mask)
    completions = []
    for bits in itertools.product((T, F), repeat=J):
        c = np.where(s == U, np.asarray(bits, dtype=np.int8)[None, :], s)
        completions.append(_verdict(c, mask))
    comp = np.stack(completions, axis=1)
    agree = comp.min(axis=1) == comp.max(axis=1)
    det = v != U
    assert np.array_equal(det, agree)
    assert np.all(v[det] == comp[det, 0])
    _report("P3'", vectors=N_VEC, completions_per_vector=2 ** J,
            determinate=int(det.sum()), undetermined=int((~det).sum()))


def test_P4_veto_non_compensation():
    rng = np.random.default_rng(104)
    s, mask = _statuses(rng, N_VEC), _necessity_masks(rng, N_VEC)
    # make many rows otherwise maximally strong: all other components PRESENT
    strong = rng.random(N_VEC) < 0.5
    j = rng.integers(0, J, size=N_VEC)
    s[strong] = T
    s[strong, j[strong]] = F
    mask[strong, j[strong]] = True
    v = _verdict(s, mask)
    vetoed = ((s == F) & mask).any(axis=1)
    assert np.all(v[vetoed] == F)
    assert not np.any(v[~vetoed] == F)
    _report("P4", vectors=N_VEC, vetoed=int(vetoed.sum()),
            vetoed_with_all_others_present=int(strong.sum()))


def test_permutation_symmetry():
    rng = np.random.default_rng(105)
    s, mask = _statuses(rng, N_VEC), _necessity_masks(rng, N_VEC)
    v = _verdict(s, mask)
    perm = np.argsort(rng.random((N_VEC, J)), axis=1)  # one permutation per row
    rows = np.arange(N_VEC)[:, None]
    v2 = _verdict(s[rows, perm], mask[rows, perm])
    assert np.array_equal(v, v2)
    _report("permutation", vectors=N_VEC, distinct_permutations=int(
        np.unique(perm, axis=0).shape[0]))


def test_channel_disjunction():
    rng = np.random.default_rng(106)
    C = 3
    ch = rng.choice(np.array([T, U, F], dtype=np.int8), size=(N_VEC, J, C),
                    p=(0.3, 0.35, 0.35))
    has = rng.random((N_VEC, J, C)) < 0.75
    mask = _necessity_masks(rng, N_VEC)
    p = E.kleene_or_codes(ch, has)
    real = np.where(has, ch, 0)
    any_t = ((real == T) & has).any(axis=2)
    all_f = np.where(has, ch == F, True).all(axis=2) & has.any(axis=2)
    missing = ~has.any(axis=2)
    assert np.array_equal(p == T, any_t)
    assert np.array_equal(p == F, all_f)
    assert np.all(p[missing] == U)
    v = _verdict(p, mask)
    # adding a PRESENT channel to a principle makes it PRESENT and never
    # lowers the verdict (T > U > F)
    j = rng.integers(0, J, size=N_VEC)
    ch2 = np.concatenate([ch, np.full((N_VEC, J, 1), F, dtype=np.int8)], axis=2)
    has2 = np.concatenate([has, np.zeros((N_VEC, J, 1), dtype=bool)], axis=2)
    ch2[np.arange(N_VEC), j, C] = T
    has2[np.arange(N_VEC), j, C] = True
    p2 = E.kleene_or_codes(ch2, has2)
    assert np.all(p2[np.arange(N_VEC), j] == T)
    v2 = _verdict(p2, mask)
    assert np.all(v2 >= v)
    # adding an ABSENT channel never changes a principle that has evidence
    ch2[np.arange(N_VEC), j, C] = F
    p3 = E.kleene_or_codes(ch2, has2)
    keep = ~missing[np.arange(N_VEC), j]
    assert np.array_equal(p3[keep], p[keep])
    _report("channels", vectors=N_VEC, principles_present_via_one_channel=int(
        np.sum(any_t & ((real == T).sum(axis=2) == 1))),
        principles_absent=int(all_f.sum()), missing=int(missing.sum()))


# --------------------------------------------------------------------------
# object path: mpc_verdict on ComponentEvidence
# --------------------------------------------------------------------------
_TYPES = ("T", "F", "I", "N", "D", "X", "G", "V")
_TYPE_CODE = {"T": T, "F": F}


def _alphabet():
    """Pre-built evidence items: (principle, type, channel) -> ComponentEvidence."""
    out = {}
    for p in E.PRINCIPLES:
        for c in ("c0", "c1", "c2"):
            base = dict(principle=p, channel=c, null_mean=0.0, null_sd=1.0,
                        bearer_id="b", protocol_id="proto", estimator="est_ok")
            out[p, "T", c] = E.ComponentEvidence(estimate=5.0, **base)
            out[p, "F", c] = E.ComponentEvidence(estimate=0.2, **base)
            out[p, "I", c] = E.ComponentEvidence(estimate=1.3, **base)
            out[p, "N", c] = E.ComponentEvidence(
                estimate=1.0, **{**base, "null_mean": float("nan")})
            out[p, "D", c] = E.ComponentEvidence(
                estimate=float("nan"), defined=False, reason="missing_events", **base)
            out[p, "X", c] = E.ComponentEvidence(
                estimate=float("nan"), defined=False, reason="NOT_IMPLEMENTED", **base)
            out[p, "G", c] = E.ComponentEvidence(
                estimate=3.0, **{**base, "null_sd": 0.0})
            out[p, "V", c] = E.ComponentEvidence(
                estimate=5.0, **{**base, "estimator": "est_unvalidated"})
            out[p, "T@other", c] = E.ComponentEvidence(
                estimate=5.0, **{**base, "bearer_id": "other"})
    return out


_REGISTRY = E.ApplicabilityRegistry(
    [{"principle": p, "estimator": "est_ok"} for p in E.PRINCIPLES]
)


def test_object_path_matches_kleene_core_and_reasons_decompose():
    rng = np.random.default_rng(107)
    alpha = _alphabet()
    n_ch = rng.choice([0, 1, 2, 3], size=(N_OBJ, J), p=(0.04, 0.64, 0.22, 0.1))
    # PRESENT-heavy so that all three verdicts are well represented
    p_types = np.array([0.5, 0.1] + [0.4 / 6] * 6)
    types = rng.choice(len(_TYPES), size=(N_OBJ, J, 3), p=p_types / p_types.sum())
    mismatch = rng.random(N_OBJ) < 0.05
    mismatch_p = rng.integers(0, J, size=N_OBJ)
    masks = _necessity_masks(rng, N_OBJ)
    undef_p = rng.integers(0, J, size=N_OBJ)

    single_cache = {}
    counts = {"ATTRIBUTED": 0, "NOT_ATTRIBUTED": 0, "UNDETERMINED": 0}
    n_global = n_p1_changed = 0
    code_table = np.array([_TYPE_CODE.get(t, U) for t in _TYPES], dtype=np.int8)
    for i in range(N_OBJ):
        ev = {}
        keys = {}
        for j, p in enumerate(E.PRINCIPLES):
            k = int(n_ch[i, j])
            if k == 0:
                continue
            tt = [_TYPES[t] for t in types[i, j, :k]]
            if mismatch[i] and j == mismatch_p[i]:
                tt[0] = "T@other"
            ev[p] = [alpha[p, t, f"c{c}"] for c, t in enumerate(tt)]
            keys[p] = tuple(tt)
        nset = tuple(p for j, p in enumerate(E.PRINCIPLES) if masks[i, j])
        out = E.mpc_verdict(ev, necessity_set=nset, registry=_REGISTRY)
        counts[out.verdict.value] += 1

        # 1) the verdict equals the vectorised Kleene core on the channel codes
        pcodes = np.full(J, U, dtype=np.int8)
        for j, p in enumerate(E.PRINCIPLES):
            if p in keys:
                pcodes[j] = max(
                    T if t == "T@other" else int(code_table[_TYPES.index(t)])
                    for t in keys[p]
                )
        expected = int(E.kleene_verdict_codes(pcodes[None, :], masks[i][None, :])[0])
        in_nset = set(nset)
        gated_bearers = {e.bearer_id for p in nset for e in ev.get(p, [])}
        if len(gated_bearers) > 1:  # Principle 0: one bearer
            expected = U
        got = {"ATTRIBUTED": T, "UNDETERMINED": U, "NOT_ATTRIBUTED": F}[
            out.verdict.value]
        assert got == expected, (i, keys, nset, out.reasons)

        # 2) reason-code decomposability
        assert E.verdict_from_reasons(out.reasons) == out.verdict
        glob = [r for r in out.reasons if E.parse_reason(r)[0] in E.GLOBAL_REASON_KINDS]
        n_global += bool(glob)
        union = []
        for p in nset:
            key = (p, keys.get(p))
            if key not in single_cache:
                single = E.mpc_verdict(
                    {p: ev[p]} if p in ev else {}, necessity_set=(p,),
                    registry=_REGISTRY, require_same_bearer=False,
                )
                single_cache[key] = single.reasons
            union.extend(single_cache[key])
        assert out.reasons == glob + union, (out.reasons, glob, union)
        assert all(E.parse_reason(r)[1] in in_nset for r in union)

        # 3) P1 and P4 on the object path. Setting a component undefined keeps
        # its declarations (bearer, protocol, estimator, channel).
        p_u = E.PRINCIPLES[undef_p[i]]
        ev_u = dict(ev)
        if p_u in ev:
            ev_u[p_u] = [
                dataclasses.replace(e, estimate=float("nan"), defined=False,
                                    reason="dropped")
                for e in ev[p_u]
            ]
        out_u = E.mpc_verdict(ev_u, necessity_set=nset, registry=_REGISTRY)
        if out.verdict != E.Verdict.ATTRIBUTED:
            assert out_u.verdict != E.Verdict.ATTRIBUTED
        if out_u.verdict != E.Verdict.UNDETERMINED:
            assert out_u.verdict == out.verdict
        n_p1_changed += out_u.verdict != out.verdict
        if not glob and any(E.parse_reason(r)[0] == E.REASON_ABSENT for r in union):
            assert out.verdict == E.Verdict.NOT_ATTRIBUTED
    _report("object path", configurations=N_OBJ, **counts,
            with_global_reason=n_global, p1_changed_to_undetermined=n_p1_changed,
            distinct_single_principle_evidence=len(single_cache))
    assert min(counts.values()) > 1000


def test_object_path_permutation_symmetry_of_evidence_order():
    """Permuting the mapping order, the channel order and the necessity-set
    order changes neither the verdict nor the set of reasons."""
    rng = np.random.default_rng(108)
    alpha = _alphabet()
    n = 20_000
    for i in range(n):
        ev = {}
        for p in E.PRINCIPLES:
            k = int(rng.integers(0, 4))
            if k:
                ev[p] = [alpha[p, _TYPES[int(t)], f"c{c}"]
                         for c, t in enumerate(rng.integers(0, 7, size=k))]
        nset = [p for p in E.PRINCIPLES if rng.random() < 0.7] or ["IIM"]
        a = E.mpc_verdict(ev, necessity_set=nset)
        order = list(ev.items())
        rng.shuffle(order)
        ev2 = {p: list(items)[::-1] for p, items in order}
        b = E.mpc_verdict(ev2, necessity_set=nset[::-1])
        assert a.verdict == b.verdict
        assert set(a.reasons) == set(b.reasons)
    _report("object permutation", configurations=n)


@pytest.mark.parametrize("n", [100_000])
def test_component_status_array_matches_scalar(n):
    rng = np.random.default_rng(109)
    est = rng.normal(0, 3, n)
    nm = rng.normal(0, 1, n)
    nsd = rng.gamma(2.0, 0.5, n)
    se = np.abs(rng.normal(0, 0.5, n)) * (rng.random(n) < 0.7)
    est[rng.random(n) < 0.02] = np.nan
    nsd[rng.random(n) < 0.02] = 0.0
    codes, margins = E.component_status_array(est, nm, nsd, se)
    mismatches = 0
    for i in range(0, n, 10):  # scalar path on every 10th draw
        st, m, _ = E.component_status(E.ComponentEvidence("RAM", est[i], nm[i], nsd[i],
                                                          se=se[i]))
        mismatches += E._STATUS_TO_CODE[st] != codes[i]
        if np.isfinite(m) or np.isfinite(margins[i]):
            assert m == pytest.approx(margins[i], rel=1e-12)
    assert mismatches == 0
    _report("status array", draws=n, present=int(np.sum(codes == T)),
            absent=int(np.sum(codes == F)), undefined=int(np.sum(codes == U)))
