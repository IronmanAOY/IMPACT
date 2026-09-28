"""MPC-Bench v2 runner: development / confirmatory split for the held-out
sets (family C, whole-brain, adversarial), new designs, exact-TPM IIM,
jackknife SEs, graded patchworks and the evidence-layer compatibility shim
(v1 and v2 verdict names)."""

import dataclasses
import json
import sys
import types
from enum import Enum

import numpy as np
import pandas as pd
import pytest

from impact_pipeline.bench import compat, export, run_bench
from impact_pipeline.bench import generators as g
from impact_pipeline.bench.factorial import factorial_tasks
from impact_pipeline.bench.patchwork import (
    PATCHWORK_MODULES,
    patchwork_sweep_levels,
    simulate_patchwork,
)

SMALL = g.AgentConfig(n_trials=16, n_reafference_pairs=6)


# --------------------------------------------------------------------------
# Development / confirmatory split
# --------------------------------------------------------------------------


def test_held_out_sets_need_the_confirmatory_path():
    adv = run_bench.adversarial_tasks([0])
    wb = run_bench.whole_brain_tasks(
        [0], observations=("source",), g_levels=(0.0,), lesions=("hub",)
    )
    assert {t.generator for t in adv} == {
        f"adversarial_{k}"
        for k in (
            "parity_grid",
            "hypersynchrony",
            "common_driver",
            "reflex_arc",
            "random_label_self_other",
            "scrambled_feedback",
        )
    }
    assert [t.cell_id for t in wb] == ["G0", "lesion_hub", "lesion_random_matched_hub"]
    for tasks in (adv, wb, factorial_tasks([0], family="C", cells=["b11111"])):
        assert all(run_bench.split_of(t) == "confirmatory" for t in tasks)
        with pytest.raises(ValueError, match="held out"):
            run_bench.check_seed_policy(tasks, confirmatory=False)
    dev = factorial_tasks([0], cells=["b11111"]) + run_bench.patchwork_sweep_tasks([0])
    run_bench.check_seed_policy(dev, confirmatory=False)
    assert {run_bench.split_of(t) for t in dev} == {"development"}
    conf = run_bench.adversarial_tasks([10000])
    run_bench.check_seed_policy(conf, confirmatory=True)
    # run_tasks itself refuses held-out tasks without the confirmatory path.
    with pytest.raises(ValueError, match="held out"):
        run_bench.run_tasks(adv[:1], "unused", confirmatory=False)
    for design in ("adversarial", "whole_brain"):
        assert run_bench.main([design, "--seeds", "0", "--list"]) == 2
    assert run_bench.main(["adversarial", "--seeds", "10000", "--list"]) == 2
    assert (
        run_bench.main(["adversarial", "--seeds", "10000", "--confirmatory", "--list"])
        == 0
    )


def test_patchwork_sweep_design_and_cli(capsys):
    tasks = run_bench.patchwork_sweep_tasks([0, 1], n_levels=3)
    assert len(tasks) == 3 * 2 * 2
    assert sorted({t.sweep_level for t in tasks}) == [0.0, 0.5, 1.0]
    assert {t.bearer_mode for t in tasks} == {"system", "principle"}
    s = run_bench.build_system(tasks[-1])
    assert s.meta["patchwork_inter_module_coupling"] == tasks[-1].sweep_level
    assert (
        run_bench.main(["patchwork_sweep", "--seeds", "0", "--levels", "4", "--list"])
        == 0
    )
    assert len(capsys.readouterr().out.split()) == 8


def test_manipulation_design_runs_on_dev_seeds(tmp_path):
    out = tmp_path / "manip"
    rc = run_bench.main(
        [
            "manipulation",
            "--seeds",
            "0",
            "--out",
            str(out),
            "--config",
            json.dumps({"n_trials": 40, "n_reafference_pairs": 15}),
        ]
    )
    assert rc == 0
    df = pd.read_csv(out / run_bench.MANIPULATION_CSV)
    assert set(df["family"]) == {"A", "C"} and df["passed"].all()
    man = json.loads((out / "manipulation_manifest.json").read_text())
    assert man["all_passed"] and man["check_version"].startswith("mpc-bench-manip")


def test_manipulation_design_keeps_the_seed_policy(tmp_path):
    """Oracle-only checks still respect the split: confirmatory seeds only
    through the confirmatory path (freeze tag and clean tree checked before
    anything is simulated), no reserved seeds, no dev seeds labelled
    confirmatory."""
    out = str(tmp_path / "manip")
    assert run_bench.main(["manipulation", "--seeds", "10000", "--out", out]) == 2
    assert run_bench.main(["manipulation", "--seeds", "5000", "--out", out]) == 2
    assert (
        run_bench.main(["manipulation", "--seeds", "0", "--confirmatory", "--out", out])
        == 2
    )
    for extra in ([], ["--freeze-tag", "no-such-freeze-tag-bench2"]):
        argv = ["manipulation", "--seeds", "10000", "--confirmatory", "--out", out]
        assert run_bench.main(argv + extra) == 2
    assert not (tmp_path / "manip").exists()


def test_nas_capacity_jackknife_measures_the_direction_of_the_estimate(monkeypatch):
    """NAS capacity reports the transfer entropy of the direction with the
    smaller null z, a choice that depends on the surrogates. The jackknife
    replicates (2 surrogates each) must re-measure the direction of the
    estimate, not the one their own small null happens to pick."""
    from impact_pipeline import mpc_metrics as mm

    def fake_nas(
        ts,
        tr=None,
        workspace_nodes=None,
        return_details=False,
        null_surrogates=0,
        null_seed=None,
        mode="legacy",
        **kw,
    ):
        x = np.asarray(ts, dtype=float)
        te_in, te_out = 10.0 + float(x[0].mean()), float(x[1].mean())
        # The full run's null picks 'out'; the replicates' 2-surrogate nulls
        # pick 'in' (a surrogate-driven switch of the reported direction).
        lim = "in" if int(null_surrogates) == 2 else "out"
        raw = te_in if lim == "in" else te_out
        return {
            "value": raw,
            "raw": raw,
            "defined": True,
            "undefined_reason": None,
            "mode": mode,
            "limiting_direction": lim,
            "transfer": {"te_in": te_in, "te_out": te_out},
            "NAS_null_n": 19,
            "NAS_null_mean": 0.0,
            "NAS_null_sd": 1.0,
            "NAS_null_method": "block_circular_shift",
        }

    monkeypatch.setattr(mm, "compute_NAS", fake_nas)
    s = g.simulate_family_a(None, SMALL, seed=0)
    res = export.run_in_memory(s, metrics=("NAS",), null_surrogates=19, se_groups=4)
    c = res["components"]["NAS"]
    x = export.bearer_view(s, "NAS")["ts"]
    assert c["estimate"] == pytest.approx(x[1].mean())
    reps = [x[1, export.block_keep(x.shape[1], 4, k)].mean() for k in range(4)]
    assert c["se"] == pytest.approx(export.jackknife_se(reps)[0], rel=1e-12)
    assert c["se_statistic"] == "te_out" and c["se_n"] == 4


def test_exact_iim_of_a_declared_tpm():
    pytest.importorskip("impact_pipeline.mpc_metrics")
    from impact_pipeline.bench.adversarial import parity_grid_system

    s = parity_grid_system(seed=0, n=4, config=SMALL)
    ex = run_bench.exact_iim(s)
    assert ex["exact"] and ex["state"] == 0 and ex["value"] > 0.5
    assert run_bench.exact_iim(g.simulate_family_a(None, SMALL, 0)) is None


def test_run_task_records_split_se_and_v2_verdict(tmp_path):
    tasks = factorial_tasks(
        [0], cells=["b11111"], config={"n_trials": 16, "n_reafference_pairs": 6}
    )
    recs = run_bench.run_tasks(
        tasks,
        tmp_path / "run",
        metrics=("NAS", "SRPI"),
        null_surrogates=2,
        se_groups=3,
        markers=False,
    )
    rec = recs[0]
    assert rec["status"] == "ok" and rec["split"] == "development"
    assert rec["schema"] == run_bench.RESULT_SCHEMA
    for p in ("NAS", "SRPI"):
        c = rec["components"][p]
        if c["defined"]:
            assert c["se_method"] == "jackknife_delete_group_3" and c["se_n"] >= 2
    assert rec["verdict"]["verdict"] in compat.VERDICTS
    df = pd.read_csv(tmp_path / "run" / run_bench.RESULTS_CSV)
    assert {"split", "se_groups", "NAS_se", "freeze_tag"} <= set(df.columns)


def test_jackknife_helpers_known_answers():
    se, n = export.jackknife_se([1.0, 2.0, 3.0, np.nan])
    # G = 3: sqrt(2/3 * ((1-2)^2 + 0 + 1)) = sqrt(4/3)
    assert n == 3 and se == pytest.approx(np.sqrt(4.0 / 3.0))
    assert np.isnan(export.jackknife_se([1.0])[0])
    keep = export.block_keep(10, 5, 1)
    assert list(keep) == [0, 1, 4, 5, 6, 7, 8, 9]
    s = g.simulate_family_a(None, SMALL, seed=1)
    ev = export.events_without_group(s.events, "RAM", 4, 0)
    tr = pd.to_numeric(ev["trial"], errors="coerce").dropna()
    assert not (tr % 4 == 0).any() and len(tr.unique()) == 12
    ev = export.events_without_group(s.events, "SRPI", 3, 2)
    assert (ev.trial_type == "self_caused").sum() == 4
    assert (ev.trial_type == "other_caused").sum() == 4
    assert set(ev.loc[ev.trial_type == "other_caused", "yoked_to"]) <= set(
        ev.loc[ev.trial_type == "self_caused", "event_id"]
    )
    with pytest.raises(ValueError):
        export.run_in_memory(s, metrics=("NAS",), se_groups=1)


def test_systems_without_events_leave_event_components_undefined():
    from impact_pipeline.bench import whole_brain as wb

    s = wb.simulate_whole_brain(wb.WholeBrainConfig(duration_sec=4.0), seed=0)
    res = export.run_in_memory(s, metrics=("RAM", "SRPI"))
    for p in ("RAM", "SRPI"):
        assert not res["components"][p]["defined"]
        assert res["components"][p]["reason"] == "no_events"


# --------------------------------------------------------------------------
# Graded patchwork
# --------------------------------------------------------------------------


def test_graded_patchwork_couples_modules_monotonically():
    from impact_pipeline.bench.manipulation import granger_gain

    cfg = g.AgentConfig(n_trials=40, n_reafference_pairs=15)
    assert patchwork_sweep_levels(3) == [0.0, 0.5, 1.0]
    base = simulate_patchwork(None, SMALL, seed=4)
    same = simulate_patchwork(None, SMALL, seed=4, inter_module_coupling=0.0)
    assert np.array_equal(base.ts, same.ts)  # lambda = 0: the disconnected patchwork
    dep = []
    for lam in (0.0, 0.4, 1.0):
        vals = []
        for seed in (0, 1):
            s = simulate_patchwork(None, cfg, seed, inter_module_coupling=lam)
            adj = s.oracle["unit_adjacency"]
            bear = s.meta["principle_bearers"]
            a, b = np.asarray(bear["RAM"]), np.asarray(bear["PDI"])
            assert (np.any(adj[np.ix_(a, b)] > 0)) == (lam > 0)
            assert s.oracle["modules_connected"] == (lam > 0)
            assert s.oracle["inter_module_coupling"] == lam
            mods = {p: s.ts[np.asarray(v)].mean(axis=0) for p, v in bear.items()}
            ring = list(PATCHWORK_MODULES) + [PATCHWORK_MODULES[0]]
            vals.append(
                np.mean(
                    [
                        granger_gain(mods[y], mods[x]) + granger_gain(mods[x], mods[y])
                        for x, y in zip(ring[:-1], ring[1:])
                    ]
                )
            )
        dep.append(np.mean(vals))
    assert dep[0] < 0.005 and dep[0] < dep[1] < dep[2]
    with pytest.raises(ValueError):
        simulate_patchwork(None, SMALL, 0, inter_module_coupling=1.5)


# --------------------------------------------------------------------------
# Compatibility shim (evidence layer v1 / v2 names)
# --------------------------------------------------------------------------


def test_verdict_names_accept_either_naming():
    class V1(str, Enum):
        ATTRIBUTED = "ATTRIBUTED"
        NOT_ATTRIBUTED = "NOT_ATTRIBUTED"

    assert compat.verdict_name(V1.ATTRIBUTED) == "MPC_CONSISTENT"
    assert compat.verdict_name("NOT_ATTRIBUTED") == "EXCLUDED"
    assert compat.verdict_name("excluded") == "EXCLUDED"
    assert compat.verdict_name(None) is None
    with pytest.raises(ValueError):
        compat.verdict_name("CONSCIOUS")
    assert compat.normalise_counts({"ATTRIBUTED": 2, "MPC_CONSISTENT": 1}) == {
        "MPC_CONSISTENT": 3,
        "EXCLUDED": 0,
        "UNDETERMINED": 0,
    }


def _fake_v2_layer():
    """A v2-style evidence module: Protocol object, v2 names, extra fields."""
    mod = types.ModuleType("impact_pipeline.evidence")

    class Verdict(str, Enum):
        EXCLUDED = "EXCLUDED"
        MPC_CONSISTENT = "MPC_CONSISTENT"
        UNDETERMINED = "UNDETERMINED"

    @dataclasses.dataclass(frozen=True)
    class ComponentEvidence:
        principle: str
        estimate: float
        null_mean: float = float("nan")
        null_sd: float = float("nan")
        se: float = 0.0
        reference: float = None
        exact: bool = False
        channel: str = "default"
        defined: bool = True
        reason: str = None
        bearer_id: str = None

    @dataclasses.dataclass(frozen=True)
    class Protocol:
        necessity_set: tuple
        channels: dict
        alpha: float = 0.05

    calls = []

    def mpc_verdict(evidence, protocol):
        calls.append((evidence, protocol))
        assert isinstance(protocol, Protocol)
        ok = all(ev.se > 0 for items in evidence.values() for ev in items)
        v = Verdict.MPC_CONSISTENT if ok else Verdict.UNDETERMINED
        return types.SimpleNamespace(
            verdict=v, reasons=[], component_status={}, margins={}
        )

    mod.Verdict, mod.ComponentEvidence, mod.Protocol = (
        Verdict,
        ComponentEvidence,
        Protocol,
    )
    mod.mpc_verdict, mod.calls = mpc_verdict, calls
    return mod


def test_evidence_verdict_drives_a_v2_layer(monkeypatch):
    import impact_pipeline

    fake = _fake_v2_layer()
    monkeypatch.setitem(sys.modules, "impact_pipeline.evidence", fake)
    monkeypatch.setattr(impact_pipeline, "evidence", fake, raising=False)
    result = {
        "components": {
            p: {
                "estimate": 1.0,
                "null_mean": 0.0,
                "null_sd": 0.1,
                "se": 0.05,
                "defined": True,
                "reason": None,
                "bearer_id": "system",
                "statistic": "raw",
                "null_family": "x",
                "n_null": 10,
                "reference": 1.2,
            }
            for p in ("NAS", "SRPI")
        }
    }
    out = export.evidence_verdict(result, {"substrate": "synthetic_rate"})
    assert out["verdict"] == "MPC_CONSISTENT"
    ev, proto = fake.calls[-1]
    assert proto.necessity_set == tuple(g.PRINCIPLES)
    assert proto.channels == {"NAS": ("default",), "SRPI": ("default",)}
    assert ev["NAS"][0].reference == 1.2 and ev["NAS"][0].se == 0.05
    # Fields the installed dataclass lacks (null_family, n_null, ...) are
    # dropped instead of raising.
    assert not hasattr(ev["NAS"][0], "null_family")


def test_evidence_verdict_on_the_installed_v1_layer_uses_v2_names():
    pytest.importorskip("impact_pipeline.evidence")
    result = {
        "components": {
            "NAS": {
                "estimate": 5.0,
                "null_mean": 0.0,
                "null_sd": 1.0,
                "defined": True,
                "statistic": "raw",
                "n_null": 10,
            }
        }
    }
    out = export.evidence_verdict(result, {}, necessity_set=("NAS",))
    assert out["verdict"] in ("MPC_CONSISTENT", "UNDETERMINED")
    assert out["verdict"] not in ("ATTRIBUTED", "NOT_ATTRIBUTED")
