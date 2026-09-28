"""Calibration and preregistration tooling of MPC-Bench: the anchor-validity
rule of the reference, protocol-declared estimator options in the bench, the
stored jackknife replicates, the construct-scale analysis helpers
(bench.analysis), scripts/calibrate_bench.py, scripts/iim_validation.py and
the decision logic of scripts/bench_hypotheses.py on synthetic records."""

import json

import numpy as np
import pandas as pd
import pytest

import scripts.bench_hypotheses as bh
import scripts.calibrate_bench as cb
import scripts.iim_validation as iv
from impact_pipeline import evidence as ev
from impact_pipeline.bench import analysis as A
from impact_pipeline.bench import export
from impact_pipeline.bench import generators as g
from impact_pipeline.bench import reference as ref
from impact_pipeline.bench.generators import PRINCIPLES

FAMILIES = {
    "RAM": "onset_jitter",
    "PDI": "circular_shift",
    "NAS": "block_circular_shift",
    "IIM": "circular_shift",
    "SRPI": "yoked_label_permutation",
}
MODES = {
    "RAM": {"update": "prediction_error"},
    "PDI": {"mode": "repertoire"},
    "NAS": {"mode": "capacity"},
    "SRPI": {"mode": "agency"},
    "IIM": {"cut_mode": "bidirectional"},
}
SMALL = g.AgentConfig(n_trials=16, n_reafference_pairs=6)


def _template(**kw):
    base = dict(
        null_families=FAMILIES,
        estimators={
            "RAM": {"update": "prediction_error"},
            "PDI": {"mode": "repertoire"},
            "NAS": {"mode": "capacity"},
            "SRPI": {"mode": "agency"},
            "IIM": {"cut_mode": "bidirectional"},
        },
        name="template",
    )
    base.update(kw)
    return ev.Protocol(**base)


def _record(task_id, *, family="A", design="witnesses", witness=None, seed=0,
            c_true=None, split="development", anchor=None, noise=0.0, rng=None,
            cell_id=None, knob=None, level=None, bits=None, bearer_mode="system",
            se=0.02, freeze_tag=None):
    """Synthetic bench record whose component excess is c_true x anchor."""
    rng = rng if rng is not None else np.random.default_rng(seed)
    anchor = anchor or {p: 1.0 for p in PRINCIPLES}
    c_true = c_true or {p: 1.0 for p in PRINCIPLES}
    comps = {}
    for p in PRINCIPLES:
        exc = c_true[p] * anchor[p] + noise * rng.standard_normal()
        reps = list(exc + 0.2 + se / 3.0 * rng.standard_normal(10))
        comps[p] = {
            "estimate": 0.2 + exc, "null_mean": 0.2, "null_sd": 0.01,
            "n_null": 19, "null_family": FAMILIES[p], "se": se, "se_n": 10,
            "se_replicates": reps, "defined": True, "statistic": "raw",
            "bearer_id": "system" if bearer_mode == "system" else f"pw_{p}",
        }
    return {
        "task_id": task_id, "status": "ok", "design": design, "family": family,
        "generator": "family_a", "cell_id": cell_id or witness, "witness_id": witness,
        "sweep_knob": knob, "sweep_level": level, "bearer_mode": bearer_mode,
        "seed": seed, "split": split, "null_surrogates": 19, "se_groups": 10,
        "bench_version": "2.0.0", "generator_version": "g",
        "intended_bits": bits or [1, 1, 1, 1, 1],
        "components": comps, "estimator_modes": MODES,
        "system": {"substrate": "synthetic_rate"},
        "provenance": {"code_version": {"git_sha": "abc", "freeze_tag": freeze_tag}},
    }


def _pc_records(n=8, family="A", split="development", excess=None, noise=0.02,
                seed0=900):
    rng = np.random.default_rng(1)
    exc = excess or {p: 1.0 for p in PRINCIPLES}
    return [
        _record(f"witness-{family}-PC_nominal-s{seed0 + i:05d}", family=family,
                witness="PC_nominal", seed=seed0 + i, split=split,
                c_true={p: exc[p] for p in PRINCIPLES}, noise=noise, rng=rng)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# reference anchor validity
# ---------------------------------------------------------------------------
def test_anchor_needs_a_credibly_positive_excess():
    recs = _pc_records(excess={"RAM": 0.0, "PDI": 0.0, "NAS": 1.0, "IIM": 1.0,
                               "SRPI": 1.0}, noise=0.05)
    reference, summary = ref.reference_from_records(recs, alpha=0.05)
    assert set(reference["values"]) == {"NAS", "IIM", "SRPI"}
    assert summary["no_anchor"] == ["RAM", "PDI"]
    assert "not credibly above" in summary["per_principle"]["RAM"]["anchor"]
    assert summary["per_principle"]["NAS"]["lower_bound"] > 0
    assert "no anchor: RAM,PDI" in reference["source"]
    # the estimate scale keeps the raw mean (no excess test)
    est, _ = ref.reference_from_records(recs, scale="estimate")
    assert set(est["values"]) == set(PRINCIPLES)
    # every principle without an anchor: refused
    zero = _pc_records(excess={p: 0.0 for p in PRINCIPLES}, noise=0.05)
    with pytest.raises(ValueError, match="valid reference anchor"):
        ref.reference_from_records(zero)
    with pytest.raises(ValueError, match="alpha"):
        ref.reference_from_records(recs, alpha=0.7)


def test_confirmatory_reference_only_when_allowed_and_not_mixed():
    conf = _pc_records(family="C", split="confirmatory", seed0=19000)
    with pytest.raises(ValueError, match="held-out"):
        ref.reference_from_records(conf)
    reference, summary = ref.reference_from_records(conf, allow_confirmatory=True)
    assert summary["splits"] == ["confirmatory"]
    with pytest.raises(ValueError, match="mixes"):
        ref.reference_from_records(conf + _pc_records(), allow_confirmatory=True)


# ---------------------------------------------------------------------------
# protocol estimator options in the bench
# ---------------------------------------------------------------------------
def test_protocol_params_merge_and_conflicts():
    proto = _template(estimators={
        "IIM": {"cut_mode": "directional"},
        "RAM": {"update": "prediction_error", "update_fallback": "feedback_magnitude",
                "require_explicit_feedback": True},
        "PDI": {"mode": "repertoire", "repertoire_null": "circular_shift"},
    })
    out = export.protocol_params(proto, {"NAS": {"tau": 0.2}})
    assert out["IIM"] == {"cut_mode": "directional"}
    assert out["PDI"] == {"repertoire_null": "circular_shift"}
    assert out["RAM"] == {} and out["NAS"] == {"tau": 0.2}
    assert export.protocol_params(None, {"x": 1}) == {"x": 1}
    with pytest.raises(ValueError, match="differs"):
        export.protocol_params(proto, {"IIM": {"cut_mode": "bidirectional"}})
    with pytest.raises(ValueError, match="bench computes"):
        export.protocol_params(_template(estimators={"PDI": {"mode": "legacy"}}))


def test_run_in_memory_keeps_replicates_and_records_the_iim_grain(monkeypatch):
    from impact_pipeline import mpc_metrics as mm

    calls = []

    def fake_iim(x, **kw):
        calls.append(kw.get("cut_mode"))
        v = float(np.mean(x[0]))
        return {"Delta_Psi": v, "raw": v, "value": v, "defined": True,
                "Delta_Psi_null_mean": 0.0, "Delta_Psi_null_sd": 1.0,
                "IIM_null_n": kw.get("null_surrogates", 0),
                "IIM_null_method": "circular_shift"}

    monkeypatch.setattr(mm, "compute_IIM", fake_iim)
    s = g.simulate_family_a(None, SMALL, seed=0)
    res = export.run_in_memory(s, metrics=("IIM",), null_surrogates=3, se_groups=4,
                               params={"IIM": {"cut_mode": "directional"}})
    c = res["components"]["IIM"]
    assert len(c["se_replicates"]) == 4
    assert c["se"] == pytest.approx(export.jackknife_se(c["se_replicates"])[0])
    modes = res["estimator_modes"]["IIM"]
    assert modes["cut_mode"] == "directional" and modes["bins"] == 2
    assert set(calls) == {"directional"}
    assert export._component_mode("IIM", modes) == "directional"


# ---------------------------------------------------------------------------
# analysis helpers
# ---------------------------------------------------------------------------
def test_status_from_bounds_is_the_evidence_rule():
    rng = np.random.default_rng(3)
    n = 400
    c = rng.normal(0.4, 0.6, n)
    se = rng.uniform(0.01, 0.4, n)
    df = rng.choice([4.0, 9.0, np.inf], n)
    for z, d, alpha in ((0.25, 0.10, 0.05), (0.4, 0.2, 0.025)):
        got = A.status_from_bounds(c, se, df, z, d, alpha)
        for i in range(n):
            item = ev.ComponentEvidence(
                principle="NAS", estimate=c[i], null_mean=0.0, null_sd=0.0,
                n_null=0, se=se[i], se_df=None if not np.isfinite(df[i]) else df[i],
                reference=1.0, reference_scale="excess")
            a = ev.component_assessment(item, cutoff=(z, d), alpha=alpha)
            assert a.status.value == got[i]
    with pytest.raises(ValueError):
        A.status_from_bounds(c, se, df, 0.1, 0.2, 0.05)


def test_assess_and_grid_on_synthetic_records():
    anchors = {p: 1.0 for p in PRINCIPLES}
    proto = _template(reference={"kind": "external", "values": anchors})
    recs = [_record("w-a", witness="PC_nominal"),
            _record("w-b", witness="W_NAS_no_workspace",
                    c_true={**{p: 1.0 for p in PRINCIPLES}, "NAS": 0.0})]
    comp, verd = A.assess(recs, proto)
    assert set(verd["verdict"]) == {"MPC_CONSISTENT", "EXCLUDED"}
    nas = comp[(comp["task_id"] == "w-b") & (comp["principle"] == "NAS")].iloc[0]
    assert nas["status"] == "ABSENT" and nas["c"] == pytest.approx(0.0)
    raw = A.assess_components(A.components_raw(recs), proto.reference, proto)
    merged = raw.merge(comp, on=["task_id", "principle"], suffixes=("", "_l"))
    assert (merged["status"] == merged["status_l"]).all()
    grid = A.cutoff_grid(raw, ["witness_id", "principle"], (0.25, 0.5), (0.1,), (0.05,))
    assert set(grid["z"]) == {0.25, 0.5} and (grid["n"] == 1).all()
    with pytest.raises(ValueError, match="allowed splits"):
        A.check_split([_record("x", split="confirmatory")], ("development",))
    assert A.same_except_reference(proto, _template(name="other"))
    assert not A.same_except_reference(proto, _template(alpha=0.01))


# ---------------------------------------------------------------------------
# scripts
# ---------------------------------------------------------------------------
def _write(dirpath, recs):
    dirpath.mkdir(parents=True, exist_ok=True)
    with open(dirpath / "results.jsonl", "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")


def test_calibrate_bench_runs_on_development_records(tmp_path):
    _write(tmp_path / "ref", _pc_records())
    rng = np.random.default_rng(5)
    sweep = []
    for knob, target in A.KNOB_TARGET.items():
        for li, lev in enumerate((0.0, 0.5, 1.0)):
            for s in range(3):
                c = {p: 1.0 for p in PRINCIPLES}
                c[target] = lev
                sweep.append(_record(f"sweep-A-{knob}-l{li:02d}-s{s:05d}",
                                     design="sweep", knob=knob, level=lev, seed=s,
                                     c_true=c, noise=0.02, rng=rng))
    wit = [_record(f"witness-A-N_ar1-s{s:05d}", witness="N_ar1", seed=s,
                   c_true={p: 0.0 for p in PRINCIPLES}, noise=0.01, rng=rng)
           for s in range(4)]
    wit += _witness_set("A", range(4), rng)
    _write(tmp_path / "rec", sweep + wit)
    tpl = tmp_path / "tpl.json"
    _template().to_json(tpl)
    res = cb.run(tmp_path / "out", template=str(tpl), reference_dirs=[tmp_path / "ref"],
                 record_dirs=[tmp_path / "rec"])
    s = res["summary"]
    assert s["consistency"]["assess_raw_vs_layer_agree"]
    assert set(s["validated_subset"]) == set(PRINCIPLES)
    ct = pd.read_csv(tmp_path / "out" / "witness_contrasts.csv")
    own = ct[ct["own"]]
    assert (own["median_delta_c"] > 0.9).all() and len(own) == 5
    for name in ("dose_response_slopes.csv", "null_cutoff_grid.csv",
                 "entry_criteria.csv", "candidate_protocol.json"):
        assert (tmp_path / "out" / name).is_file()
    _write(tmp_path / "bad", [_record("x", split="confirmatory")])
    with pytest.raises(ValueError, match="allowed splits"):
        cb.run(tmp_path / "out2", template=str(tpl), reference_dirs=[tmp_path / "ref"],
               record_dirs=[tmp_path / "bad"])


def test_iim_validation_smoke(tmp_path):
    res = iv.run(tmp_path, kinds=("independent",), n=3, n_times=(200,), seeds=(0,),
                 cut_modes=("directional",), null_surrogates=2, sweep_kind=None)
    df = res["results"]
    assert list(df["status"]) == ["ok"] and df["delta_psi_exact"].abs().max() < 1e-9
    assert {"iim_z", "null_mean", "null_sd"} <= set(df.columns)
    again = iv.run(tmp_path, kinds=("independent",), n=3, n_times=(200,), seeds=(0,),
                   cut_modes=("directional",), null_surrogates=2, sweep_kind=None)
    assert again["summary"]["n_run"] == 0  # resumed: nothing left to do
    with pytest.raises(ValueError, match="seeds"):
        iv.run(tmp_path, kinds=("independent",), seeds=(10000,))
    # serial runs leave process-wide logging as they found it
    import logging

    assert logging.root.manager.disable == logging.NOTSET


# ---------------------------------------------------------------------------
# bench_hypotheses decision logic (synthetic, development mode)
# ---------------------------------------------------------------------------
def _hyp_protocols(tmp_path, nval=("NAS", "IIM", "SRPI")):
    anchors = {"NAS": 1.0, "IIM": 1.0, "SRPI": 1.0}
    full = _template(reference={"kind": "external", "values": anchors,
                                "se": {p: 0.001 for p in anchors}}, name="bench")
    val = full.replace(necessity_set=tuple(nval), name="bench-anchored")
    pf, pv = tmp_path / "p.json", tmp_path / "pv.json"
    full.to_json(pf)
    val.to_json(pv)
    return pf, pv


def _witness_set(family, seeds, rng, target_c=0.0, split="development"):
    recs = []
    for s in seeds:
        recs.append(_record(f"witness-{family}-PC_nominal-s{s:05d}", family=family,
                            witness="PC_nominal", seed=s, rng=rng, split=split))
        for w, t in bh.WITNESS_TARGET.items():
            c = {p: 1.0 for p in PRINCIPLES}
            c[t] = target_c
            recs.append(_record(f"witness-{family}-{w}-s{s:05d}", family=family,
                                witness=w, seed=s, c_true=c, rng=rng, split=split,
                                bits=[int(p != t) for p in PRINCIPLES]))
    return recs


def test_hypotheses_on_ideal_synthetic_records(tmp_path, monkeypatch):
    pf, pv = _hyp_protocols(tmp_path)
    monkeypatch.setattr(bh, "PROTOCOL", pf)
    monkeypatch.setattr(bh, "PROTOCOL_ANCHORED", pv)
    rng = np.random.default_rng(11)
    recs_a = _witness_set("A", range(10), rng)
    recs_c = _witness_set("C", range(10), rng)
    _write(tmp_path / "A", recs_a)
    _write(tmp_path / "C", recs_c)
    _write(tmp_path / "refC", _pc_records(family="C", seed0=19000))
    s = bh.run(tmp_path / "out", bench_a=[tmp_path / "A"], bench_c=[tmp_path / "C"],
               reference_c=[tmp_path / "refC"], allow_development=True)
    assert s["status"].startswith("DEVELOPMENT")
    out = s["outcomes"]
    assert out["HC4"] == "SUPPORTED"
    assert s["results"]["HC4"]["n_principles_irredundant"] == 3
    assert out["HC5"] == "SUPPORTED"
    assert out["HC8"] == "SUPPORTED"
    assert out["HC1"] == "NOT_EVALUABLE"  # no null systems given
    assert out["HC10"] in ("SUPPORTED", "INDETERMINATE")
    assert (tmp_path / "out" / "hypotheses.json").is_file()
    # a witness whose target stays present falsifies HC4 and HC8
    bad = _witness_set("A", range(10), rng, target_c=1.0)
    _write(tmp_path / "A2", bad)
    s2 = bh.run(tmp_path / "out2", bench_a=[tmp_path / "A2"], allow_development=True)
    assert s2["outcomes"]["HC4"] == "FALSIFIED" and s2["outcomes"]["HC8"] == "FALSIFIED"
    assert s2["outcomes"]["HC5"] == "FALSIFIED"


def test_hypotheses_refuse_development_records_and_unfrozen_protocols(tmp_path,
                                                                      monkeypatch):
    pf, pv = _hyp_protocols(tmp_path)
    monkeypatch.setattr(bh, "PROTOCOL", pf)
    monkeypatch.setattr(bh, "PROTOCOL_ANCHORED", pv)
    _write(tmp_path / "A", _witness_set("A", range(2), np.random.default_rng(0)))
    with pytest.raises(ValueError, match="frozen hash"):
        bh.run(tmp_path / "o", bench_a=[tmp_path / "A"])
    monkeypatch.setattr(bh, "EXPECTED_HASHES", {
        "mpc_bench_v1": ev.Protocol.from_json(pf).hash,
        "mpc_bench_v1_anchored": ev.Protocol.from_json(pv).hash})
    with pytest.raises(ValueError, match="allowed splits"):
        bh.run(tmp_path / "o", bench_a=[tmp_path / "A"])
    conf = _witness_set("A", range(2), np.random.default_rng(0), split="confirmatory")
    _write(tmp_path / "Aconf", conf)
    with pytest.raises(ValueError, match="freeze tag"):
        bh.run(tmp_path / "o", bench_a=[tmp_path / "Aconf"])
    # an anchored protocol that differs in more than the necessity set
    other = ev.Protocol.from_json(pv).replace(alpha=0.01)
    other.to_json(pv)
    with pytest.raises(ValueError, match="except for its necessity set"):
        bh.run(tmp_path / "o", allow_development=True)


def test_frozen_protocol_hashes_match_the_files():
    """The hashes the evaluator checks are those of the committed files."""
    for key, path in (("mpc_bench_v1", bh.PROTOCOL),
                      ("mpc_bench_v1_anchored", bh.PROTOCOL_ANCHORED)):
        want = bh.EXPECTED_HASHES[key]
        assert want is not None, f"{key}: hash not frozen"
        assert ev.Protocol.from_json(path).hash == want


def test_hc2_decisions_on_a_synthetic_table(tmp_path):
    rows = []
    rng = np.random.default_rng(2)
    exact = {"independent": 0.0, "ring": 0.04, "all_to_all": 0.09,
             "feedforward_star": 0.02, "xor_loop": 0.4}
    for sysname, ex in exact.items():
        for cm in ("bidirectional", "directional"):
            e = 0.0 if (sysname == "feedforward_star" and cm == "directional") else ex
            for t, err in ((1000, 0.02), (3000, 0.01), (10000, 0.005)):
                for s in range(5):
                    z = rng.normal(0, 1) if e == 0.0 else 10.0
                    rows.append({"system": sysname, "cut_mode": cm, "n_time": t,
                                 "seed": s, "status": "ok", "coupling": np.nan,
                                 "delta_psi_exact": e, "delta_psi_est": e + err,
                                 "null_mean": 0.0, "null_sd": 0.001, "iim_z": z})
    for cm in ("bidirectional", "directional"):
        for cp in (0.0, 0.3, 0.6):
            for s in range(5):
                rows.append({"system": "ring", "cut_mode": cm, "n_time": 10000,
                             "seed": s, "status": "ok", "coupling": cp,
                             "delta_psi_exact": cp / 10, "delta_psi_est": cp / 10,
                             "null_mean": 0.0, "null_sd": 0.001, "iim_z": 1.0})
    pd.DataFrame(rows).to_csv(tmp_path / "iim_validation.csv", index=False)
    pd.DataFrame([{"system": k, "cut_mode": cm,
                   "delta_psi_exact": 0.0 if (k == "feedforward_star"
                                              and cm == "directional") else v}
                  for k, v in exact.items() for cm in ("bidirectional",
                                                       "directional")]
                 ).to_csv(tmp_path / "iim_validation_exact.csv", index=False)
    res = bh.hc2(tmp_path)
    assert res["parts"]["a_exact"]["outcome"] == "SUPPORTED"
    assert res["parts"]["b_bias_shrinks"]["outcome"] == "SUPPORTED"
    assert res["parts"]["e_coupling_monotone"]["outcome"] == "SUPPORTED"
