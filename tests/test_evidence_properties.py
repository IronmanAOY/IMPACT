"""
Randomised property tests of the three-valued MPC verdict under the v2
semantics (exclusion rule, construct-scale status, declared channels):
missingness safety (P1), resolving UNDEFINED never reverses a determinate
verdict (P2'), a verdict is determinate iff all completions agree (P3'),
veto, permutation symmetry, channel disjunction, reason-code
decomposability, plus "a declared channel without an item is never ABSENT"
and "a missing sampling SE gives UNDEFINED".

The Kleene core (``evidence.kleene_verdict_codes`` / ``kleene_or_codes``) is
checked on >= 1e5 numpy-seeded vectors per property; the object path
(``evidence.mpc_verdict`` on ComponentEvidence under a Protocol) is checked
against that core, for missingness safety, veto and reason-code
decomposability on 1e5 random evidence configurations. Counts are printed
(``pytest -s``).
"""
import dataclasses
import itertools
import math

import numpy as np
import pytest

from impact_pipeline import evidence as E

N_VEC = 200_000
N_OBJ = 100_000
N_CHAN = 60_000
J = len(E.PRINCIPLES)
T, U, F = E.CODE_T, E.CODE_U, E.CODE_F
VERDICT_CODE = {"MPC_CONSISTENT": T, "UNDETERMINED": U, "EXCLUDED": F}


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


# --------------------------------------------------------------------------
# Kleene core (vectorised)
# --------------------------------------------------------------------------
def test_P1_missingness_safety_vectorised():
    rng = np.random.default_rng(101)
    s, mask = _statuses(rng, N_VEC), _necessity_masks(rng, N_VEC)
    v = _verdict(s, mask)
    j = rng.integers(0, J, size=N_VEC)
    s2 = s.copy()
    s2[np.arange(N_VEC), j] = U
    v2 = _verdict(s2, mask)
    # never turns UNDETERMINED / EXCLUDED into MPC_CONSISTENT ...
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
    perm = np.argsort(rng.random((N_VEC, J)), axis=1)
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
    j = rng.integers(0, J, size=N_VEC)
    ch2 = np.concatenate([ch, np.full((N_VEC, J, 1), F, dtype=np.int8)], axis=2)
    has2 = np.concatenate([has, np.zeros((N_VEC, J, 1), dtype=bool)], axis=2)
    ch2[np.arange(N_VEC), j, C] = T
    has2[np.arange(N_VEC), j, C] = True
    p2 = E.kleene_or_codes(ch2, has2)
    assert np.all(p2[np.arange(N_VEC), j] == T)
    v2 = _verdict(p2, mask)
    assert np.all(v2 >= v)
    ch2[np.arange(N_VEC), j, C] = F
    p3 = E.kleene_or_codes(ch2, has2)
    keep = ~missing[np.arange(N_VEC), j]
    assert np.array_equal(p3[keep], p[keep])
    _report("channels", vectors=N_VEC, principles_present_via_one_channel=int(
        np.sum(any_t & ((real == T).sum(axis=2) == 1))),
        principles_absent=int(all_f.sum()), missing=int(missing.sum()))


# --------------------------------------------------------------------------
# object path: mpc_verdict on ComponentEvidence (v2 construct-scale status)
# --------------------------------------------------------------------------
# Evidence types (reference 1 above an analytic null 0, so c = estimate):
# T present, F absent, I inconclusive, N no null, D undefined, X channel not
# implemented, S no sampling SE, A invalid anchors, E exact (no SE needed,
# present), V unvalidated estimator.
_TYPES = ("T", "F", "I", "N", "D", "X", "S", "A", "E", "V")
_TYPE_CODE = {"T": T, "F": F, "E": T}
_REASON_OF = {"I": "INCONCLUSIVE", "N": "NO_NULL_CALIBRATION", "S": "NO_SAMPLING_SE",
              "A": "INVALID_ANCHORS", "D": "UNDEFINED", "X": "NOT_IMPLEMENTED",
              "V": "ESTIMATOR_NOT_VALIDATED"}


def _alphabet(protocol_id="proto"):
    """Pre-built evidence items: (principle, type, channel) -> ComponentEvidence."""
    out = {}
    for p in E.PRINCIPLES:
        for c in ("c0", "c1", "c2"):
            base = dict(principle=p, channel=c, null_mean=0.0, null_sd=0.3, n_null=50,
                        reference=1.0, se=0.05, bearer_id="b",
                        protocol_id=protocol_id, estimator="est_ok@v1",
                        substrate="synthetic_rate")
            mk = E.ComponentEvidence
            out[p, "T", c] = mk(estimate=1.0, **base)
            out[p, "F", c] = mk(estimate=-0.1, **base)
            out[p, "I", c] = mk(estimate=0.2, **base)
            out[p, "N", c] = mk(estimate=1.0, **{**base, "null_mean": float("nan")})
            out[p, "D", c] = mk(estimate=float("nan"), defined=False,
                                reason="missing_events", **base)
            out[p, "X", c] = mk(estimate=float("nan"), defined=False,
                                reason="NOT_IMPLEMENTED", **base)
            out[p, "S", c] = mk(estimate=1.0, **{**base, "se": 0.0})
            out[p, "A", c] = mk(estimate=1.0, **{**base, "reference": -0.5})
            out[p, "E", c] = mk(estimate=1.0, **{**base, "se": 0.0, "exact": True})
            out[p, "V", c] = mk(estimate=1.0, **{**base, "estimator": "est_bad@v1"})
            out[p, "T@other", c] = mk(estimate=1.0, **{**base, "bearer_id": "other"})
    return out


_REGISTRY = E.ApplicabilityRegistry([
    {"principle": p, "estimator": "est_ok", "version": "v1",
     "substrate": "synthetic_rate",
     "evidence": {"run_id": "prop", "null_false_present_rate": 0.05,
                  "recovery_slope": 1.0}}
    for p in E.PRINCIPLES
])


def test_alphabet_statuses_are_as_labelled():
    alpha = _alphabet()
    for t in _TYPES:
        ev = alpha["RAM", t, "c0"]
        v = E.mpc_verdict({"RAM": [ev]}, necessity_set=("RAM",), registry=_REGISTRY)
        expected = _TYPE_CODE.get(t, U)
        assert VERDICT_CODE[v.verdict.value] == expected, t
        if expected == U:
            assert E.parse_reason(v.reasons[0])[0] == _REASON_OF[t], (t, v.reasons)


def test_object_path_matches_kleene_core_and_reasons_decompose():
    rng = np.random.default_rng(107)
    alpha = _alphabet()
    n_ch = rng.choice([0, 1, 2, 3], size=(N_OBJ, J), p=(0.04, 0.64, 0.22, 0.1))
    p_types = np.array([0.5, 0.12] + [0.38 / 8] * 8)
    types = rng.choice(len(_TYPES), size=(N_OBJ, J, 3), p=p_types / p_types.sum())
    mismatch = rng.random(N_OBJ) < 0.05
    mismatch_p = rng.integers(0, J, size=N_OBJ)
    masks = _necessity_masks(rng, N_OBJ)
    undef_p = rng.integers(0, J, size=N_OBJ)
    drop_se = rng.random(N_OBJ) < 0.5

    single_cache = {}
    counts = {"MPC_CONSISTENT": 0, "EXCLUDED": 0, "UNDETERMINED": 0}
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
        gated_bearers = {e.bearer_id for p in nset for e in ev.get(p, [])}
        if len(gated_bearers) > 1:  # single-source constraint: one bearer
            expected = U
        got = VERDICT_CODE[out.verdict.value]
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
        assert all(E.parse_reason(r)[1] in set(nset) for r in union)

        # 3) P1 on the object path: making a component undefined (or dropping
        # its sampling SE) keeps its declarations and never helps the verdict.
        p_u = E.PRINCIPLES[undef_p[i]]
        ev_u = dict(ev)
        if p_u in ev:
            if drop_se[i]:
                ev_u[p_u] = [dataclasses.replace(e, se=float("nan"), exact=False)
                             for e in ev[p_u]]
            else:
                ev_u[p_u] = [dataclasses.replace(e, estimate=float("nan"),
                                                 defined=False, reason="dropped")
                             for e in ev[p_u]]
        out_u = E.mpc_verdict(ev_u, necessity_set=nset, registry=_REGISTRY)
        if out.verdict != E.Verdict.MPC_CONSISTENT:
            assert out_u.verdict != E.Verdict.MPC_CONSISTENT
        if out_u.verdict != E.Verdict.UNDETERMINED:
            assert out_u.verdict == out.verdict
        n_p1_changed += out_u.verdict != out.verdict
        # 4) veto on the object path
        if not glob and any(E.parse_reason(r)[0] == E.REASON_ABSENT for r in union):
            assert out.verdict == E.Verdict.EXCLUDED
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
                         for c, t in enumerate(rng.integers(0, 9, size=k))]
        nset = [p for p in E.PRINCIPLES if rng.random() < 0.7] or ["IIM"]
        a = E.mpc_verdict(ev, necessity_set=nset)
        order = list(ev.items())
        rng.shuffle(order)
        ev2 = {p: list(items)[::-1] for p, items in order}
        b = E.mpc_verdict(ev2, necessity_set=nset[::-1])
        assert a.verdict == b.verdict
        assert set(a.reasons) == set(b.reasons)
    _report("object permutation", configurations=n)


def test_declared_channel_without_item_is_never_absent():
    """
    Under a protocol that declares channels, a principle with a declared
    channel that has no evidence item is never ABSENT, and the verdict is
    never EXCLUDED because of it: its status is the Kleene OR of the other
    channels and U. Checked against the Kleene core with the missing channels
    as U.
    """
    rng = np.random.default_rng(109)
    protocols = []
    for _ in range(64):
        declared = {p: tuple(c for c in ("c0", "c1", "c2") if rng.random() < 0.6)
                    or ("c0",) for p in E.PRINCIPLES}
        nset = [p for p in E.PRINCIPLES if rng.random() < 0.7] or ["NAS"]
        protocols.append(E.Protocol(necessity_set=nset, channels=declared))
    alphabets = {pr.hash: _alphabet(pr.protocol_id) for pr in protocols}
    types_det = ("T", "F", "I", "S")
    n_missing = n_would_be_absent = 0
    counts = {"MPC_CONSISTENT": 0, "EXCLUDED": 0, "UNDETERMINED": 0}
    for i in range(N_CHAN):
        proto = protocols[int(rng.integers(len(protocols)))]
        alpha = alphabets[proto.hash]
        ev, pcodes = {}, np.full(J, U, dtype=np.int8)
        for j, p in enumerate(E.PRINCIPLES):
            items, codes = [], []
            for c in proto.channels[p]:
                if rng.random() < 0.3:  # declared channel left without an item
                    codes.append(U)
                    n_missing += p in proto.necessity_set
                    continue
                t = types_det[int(rng.integers(len(types_det)))]
                items.append(alpha[p, t, c])
                codes.append(_TYPE_CODE.get(t, U))
            if rng.random() < 0.2:  # evidence of an undeclared channel: ignored
                extra = next(c for c in ("c0", "c1", "c2")
                             if c not in proto.channels[p]) if len(
                    proto.channels[p]) < 3 else None
                if extra is not None:
                    items.append(alpha[p, "F", extra])
            if items:
                ev[p] = items
            pcodes[j] = max(codes)
        out = E.mpc_verdict(ev, proto)
        counts[out.verdict.value] += 1
        mask = np.array([p in proto.necessity_set for p in E.PRINCIPLES])
        assert VERDICT_CODE[out.verdict.value] == int(
            E.kleene_verdict_codes(pcodes[None, :], mask[None, :])[0])
        for p in E.PRINCIPLES:
            chans = out.channels.get(p, {})
            if any(chans.get(c) is None or out.channel_reasons.get(p, {}).get(c)
                   == E.REASON_MISSING_CHANNEL for c in proto.channels[p]):
                missing = [c for c in proto.channels[p]
                           if out.channel_reasons.get(p, {}).get(c)
                           == E.REASON_MISSING_CHANNEL]
                if missing:
                    assert out.component_status[p] != E.ComponentStatus.ABSENT
                    others = [chans[c] for c in proto.channels[p] if c not in missing]
                    n_would_be_absent += bool(others) and all(
                        s == E.ComponentStatus.ABSENT for s in others)
    _report("declared channel without item", configurations=N_CHAN,
            missing_declared_channels_in_N=n_missing,
            principles_absent_but_for_the_missing_channel=n_would_be_absent, **counts)
    assert n_would_be_absent > 1000 and min(counts.values()) > 500


def test_missing_sampling_se_gives_undefined():
    """se 0/NaN/None on an empirical estimate is UNDEFINED (NO_SAMPLING_SE)
    whatever the estimate, null and anchors; exact estimates keep a status."""
    rng = np.random.default_rng(110)
    n = N_VEC // 2
    est = rng.normal(0.5, 1.0, n)
    nm = rng.normal(0.0, 0.2, n)
    nsd = rng.gamma(2.0, 0.2, n)
    ref = nm + rng.gamma(2.0, 0.5, n)
    se = rng.choice(np.array([0.0, np.nan]), size=n)
    codes, margins = E.component_status_array(est, nm, nsd, se, n_null=30,
                                              reference=ref)
    assert np.all(codes == U) and np.all(np.isnan(margins))
    codes_ex, _ = E.component_status_array(est, nm, nsd, se, n_null=30,
                                           reference=ref, exact=True)
    assert np.any(codes_ex != U)
    n_scalar = 0
    for i in range(0, n, 10):  # the scalar path, with its reason
        for s in (0.0, float("nan"), None):
            a = E.component_assessment(E.ComponentEvidence(
                "IIM", est[i], nm[i], nsd[i], se=s, n_null=30, reference=ref[i]))
            assert a.status == E.ComponentStatus.UNDEFINED
            assert a.reason == E.REASON_NO_SAMPLING_SE
            assert math.isfinite(a.c)
            n_scalar += 1
    # in the verdict: a component without SE can never be decisive
    ev = {p: [E.ComponentEvidence(p, 1.0, 0.0, 0.1, se=0.05, n_null=30,
                                  reference=1.0)] for p in E.PRINCIPLES}
    for p in E.PRINCIPLES:
        for est_p in (-5.0, 0.0, 1.0, 5.0):
            ev_p = dict(ev)
            ev_p[p] = [E.ComponentEvidence(p, est_p, 0.0, 0.1, se=0.0, n_null=30,
                                           reference=1.0)]
            v = E.mpc_verdict(ev_p)
            assert v.verdict == E.Verdict.UNDETERMINED
            assert v.reasons == [f"NO_SAMPLING_SE:{p}"]
    _report("missing se", vectors=n, scalar_checks=n_scalar,
            exact_determinate=int(np.sum(codes_ex != U)))


@pytest.mark.parametrize("n", [100_000])
def test_component_status_array_matches_scalar(n):
    rng = np.random.default_rng(111)
    est = rng.normal(0.4, 1.0, n)
    nm = rng.normal(0, 0.3, n)
    nsd = rng.gamma(2.0, 0.2, n)
    k = rng.integers(0, 60, n)
    ref = nm + rng.normal(1.0, 0.8, n)  # some anchors invalid (ref <= null)
    rse = np.abs(rng.normal(0, 0.1, n)) * (rng.random(n) < 0.5)
    se = np.abs(rng.normal(0, 0.3, n)) * (rng.random(n) < 0.8)
    exact = rng.random(n) < 0.1
    est[rng.random(n) < 0.02] = np.nan
    nsd[rng.random(n) < 0.02] = np.nan
    rse[rng.random(n) < 0.05] = np.nan
    se[rng.random(n) < 0.02] = -0.1
    for scale in ("estimate", "excess"):
        codes, margins = E.component_status_array(
            est, nm, nsd, se, n_null=k, reference=ref, reference_se=rse,
            exact=exact, reference_scale=scale, cutoff=(0.3, 0.05), alpha=0.05)
        mismatches = 0
        for i in range(n):
            st, m, _ = E.component_status(
                E.ComponentEvidence("RAM", est[i], nm[i], nsd[i], se=se[i],
                                    n_null=int(k[i]), reference=ref[i],
                                    reference_se=rse[i], exact=bool(exact[i]),
                                    reference_scale=scale),
                cutoff=(0.3, 0.05), alpha=0.05)
            mismatches += E._STATUS_TO_CODE[st] != codes[i]
            if np.isfinite(m) or np.isfinite(margins[i]):
                assert m == margins[i]
        assert mismatches == 0
        _report(f"status array ({scale})", draws=n,
                present=int(np.sum(codes == T)), absent=int(np.sum(codes == F)),
                undefined=int(np.sum(codes == U)))
