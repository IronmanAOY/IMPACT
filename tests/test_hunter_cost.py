"""
Hunter IIM cost guard (1.1.0 post-freeze fix): closed-form Psi-evaluation
counts, the ceiling at build-campaign, the estimate in the manifest and the
measured calibration from shard timings. Nothing here changes an estimator:
the counts are checked against the enumerations of
mpc_metrics.prepare_iim_problem.
"""

import dataclasses
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from impact_pipeline import hunter_cost as hc
from impact_pipeline import hunter_iim
from impact_pipeline import mpc_metrics as mm
from impact_pipeline.execution_profiles import get_execution_profile

REPO = Path(__file__).resolve().parents[1]


pytestmark = pytest.mark.usefixtures("clean_hunter_env")


# ---------------------------------------------------------------------------
# Closed-form counts
# ---------------------------------------------------------------------------


def _enumerated(n, max_mech, max_purv, n_parts, cut_mode, partition_mode):
    nodes = tuple(range(n))
    mechs = mm._iim_enumerate_subsets(nodes, n if max_mech is None else max_mech)
    purvs = mm._iim_enumerate_subsets(nodes, n if max_purv is None else max_purv)
    cuts = mm._iim_all_system_cuts(n, partition_mode, cut_mode)
    n_eval = len(cuts) if n_parts is None else min(n_parts, len(cuts))
    bip = lambda s: len(mm._iim_enumerate_bipartitions(s))  # noqa: E731
    per_tpm = sum(bip(m) for m in mechs) * sum(bip(z) for z in purvs)
    return mechs, purvs, cuts, per_tpm * (1 + n_eval)


@pytest.mark.parametrize("cut_mode", ["bidirectional", "directional"])
@pytest.mark.parametrize("partition_mode", ["all", "balanced"])
def test_closed_form_counts_match_the_iim_enumerations(cut_mode, partition_mode):
    for n, max_mech, max_purv, n_parts in itertools.product(
        range(2, 8), (None, 1, 2, 3), (None, 2), (None, 1, 5)
    ):
        mechs, purvs, cuts, total = _enumerated(
            n, max_mech, max_purv, n_parts, cut_mode, partition_mode
        )
        cost = hc.iim_run_cost(
            n,
            max_mechanism_size=max_mech,
            max_purview_size=max_purv,
            n_parts=n_parts,
            cut_mode=cut_mode,
            partition_mode=partition_mode,
        )
        case = (n, max_mech, max_purv, n_parts)
        assert cost["n_mechanisms"] == len(mechs), case
        assert cost["n_purviews"] == len(purvs), case
        assert cost["n_cuts_total"] == len(cuts), case
        assert cost["psi_evaluations"] == total, case
        assert hc.enumerated_run_cost(mechs, purvs, cost["n_cuts_evaluated"])[
            "psi_evaluations"
        ] == total, case
        # a mechanism shard is a slice of the size-ordered enumeration
        bips = [hc.bipartitions_of_size(len(m)) for m in mechs]
        for a, b in hunter_iim._split_evenly(len(mechs), 4):
            assert hc.subset_range_bipartition_sum(n, max_mech, a, b) == sum(
                bips[a:b]
            ), case


def test_counts_reproduce_the_measured_growth():
    # exhaustive IIM: 1.7 s (4 nodes), 46 s (5), ~27 min (6) on one core
    per_run = {n: hc.iim_run_cost(n)["psi_evaluations"] for n in range(4, 11)}
    assert per_run[4] == 5_000 and per_run[5] == 129_600 and per_run[6] == 2_899_232
    for m in hc.REFERENCE_MEASUREMENTS:
        assert per_run[m["n_nodes"]] == m["psi_evaluations"]
        assert m["wall_seconds"] / m["psi_evaluations"] <= (
            hc.REFERENCE_SECONDS_PER_PSI_EVAL
        )
    growth = [per_run[n + 1] / per_run[n] for n in range(4, 10)]
    assert all(15 < g < 40 for g in growth)  # "roughly 30x per added node"
    assert per_run[10] > hc.DEFAULT_MAX_PSI_EVALS  # the 10-node default


def test_planned_structure_matches_prepare_iim_problem():
    rng = np.random.RandomState(0)
    cases = itertools.product(
        (1, 2, 3, 6, 9),
        (2, 3, 4),
        (None, 2, 5),
        ("reduce_bins_first", "reduce_nodes_first", "error"),
        (None, (0, 2, 4)),
    )
    for n_regions, bins, max_nodes, policy, bearer in cases:
        if bearer is not None and n_regions <= max(bearer):
            continue
        prep = mm.prepare_iim_problem(
            rng.rand(n_regions, 40),
            bins=bins,
            max_nodes=max_nodes,
            max_state_space=64,
            state_budget_policy=policy,
            bearer_nodes=None if bearer is None else list(bearer),
        )
        plan = hc.planned_iim_structure(
            n_regions,
            bins=bins,
            max_nodes=max_nodes,
            max_state_space=64,
            state_budget_policy=policy,
            n_bearer=None if bearer is None else len(bearer),
        )
        case = (n_regions, bins, max_nodes, policy, bearer)
        assert plan["defined"] == bool(prep["defined"]), case
        if plan["defined"]:
            assert plan["n_nodes"] == prep["n_nodes_used"], case
            assert plan["bins"] == prep["bins_used"], case
            cost = hc.iim_run_cost(plan["n_nodes"])
            assert cost["n_mechanisms"] == len(prep["mechanisms_all"]), case
            assert cost["n_cuts_evaluated"] == len(prep["cuts_eval"]), case
        else:
            assert plan["reason"] == prep["undefined_reason"], case


def test_default_state_budget_gives_the_ten_node_subsystem():
    prep = mm.prepare_iim_problem(np.random.RandomState(1).rand(12, 30), bins=3)
    plan = hc.planned_iim_structure(12, bins=3)
    assert (plan["n_nodes"], plan["bins"]) == (10, 2)
    assert (prep["n_nodes_used"], prep["bins_used"]) == (10, 2)


def test_ceiling_and_rate_resolution():
    default = hc.resolve_psi_eval_ceiling(None, env={})
    assert default == (hc.DEFAULT_MAX_PSI_EVALS, "default")
    value, source = hc.resolve_psi_eval_ceiling(None, env={hc.MAX_PSI_EVALS_ENV: "2e9"})
    assert value == 2_000_000_000 and hc.MAX_PSI_EVALS_ENV in source
    value, source = hc.resolve_psi_eval_ceiling(
        "1e12", env={hc.MAX_PSI_EVALS_ENV: "5"}
    )
    assert value == 10**12 and "--hunter-max-psi-evals" in source
    for bad in ("0", "-3", "abc", "nan", "inf"):
        with pytest.raises(ValueError):
            hc.parse_count(bad)
    assert hc.resolve_seconds_per_psi_eval(None, env={}) == (None, "reference")
    rate, source = hc.resolve_seconds_per_psi_eval(
        None, env={hc.SECONDS_PER_PSI_EVAL_ENV: "2.5e-6"}
    )
    assert rate == 2.5e-6 and hc.SECONDS_PER_PSI_EVAL_ENV in source
    with pytest.raises(ValueError):
        hc.parse_rate("0")


def test_runtime_estimate_reference_and_measured():
    ref = hc.runtime_estimate(
        psi_evaluations=10**9,
        max_phase1_shard_psi_evaluations=10**6,
        max_cut_shard_psi_evaluations=10**8,
        workers_per_task=20,
        shards_per_node=4,
        cut_walltime_seconds=1800,
        phase1_walltime_seconds=1800,
    )
    assert ref["rate_source"] == "reference"
    rate = hc.REFERENCE_SECONDS_PER_PSI_EVAL / 20
    assert ref["shard_seconds_per_psi_evaluation"] == pytest.approx(rate)
    assert ref["node_hours"] == pytest.approx(10**9 * rate / 4 / 3600)
    assert ref["single_core_hours_at_reference_rate"] == pytest.approx(
        10**9 * hc.REFERENCE_SECONDS_PER_PSI_EVAL / 3600
    )
    # 1e8 * 2.8e-5 s = 2800 s > 30 min walltime; 1e6 * 2.8e-5 s = 28 s is fine
    assert ref["stages_exceeding_walltime"] == ["cut-shard"]
    measured = hc.runtime_estimate(
        psi_evaluations=10**9,
        max_phase1_shard_psi_evaluations=0,
        max_cut_shard_psi_evaluations=0,
        workers_per_task=20,
        shards_per_node=4,
        seconds_per_psi_evaluation=1e-6,
        rate_source="measured (test)",
    )
    assert measured["shard_hours"] == pytest.approx(1e3 / 3600)
    assert measured["node_hours"] == pytest.approx(1e3 / 4 / 3600)
    assert measured["rate_source"] == "measured (test)"


def test_calibration_from_records():
    recs = [
        {"wall_seconds": 10.0, "cpu_seconds": 100.0, "psi_evaluations": 2_000_000},
        {"wall_seconds": 30.0, "cpu_seconds": 300.0, "psi_evaluations": 2_000_000},
        {"wall_seconds": 5.0, "psi_evaluations": 0},  # skipped/undefined shard
        {"wall_seconds": 7.0},  # record of an older build
    ]
    cal = hc.calibration_from_records(recs)
    assert cal["n_shards"] == 2 and cal["psi_evaluations"] == 4_000_000
    assert cal["shard_wall_seconds_per_psi_evaluation"] == pytest.approx(1e-5)
    assert cal["cpu_seconds_per_psi_evaluation"] == pytest.approx(1e-4)
    assert cal["per_shard_wall_seconds_per_psi_evaluation"]["max"] == pytest.approx(
        1.5e-5
    )
    assert cal["overhead_dominated"] is False
    tiny = hc.calibration_from_records([{"wall_seconds": 1.0, "psi_evaluations": 10}])
    assert tiny["overhead_dominated"] is True
    empty = hc.calibration_from_records([])
    assert empty["shard_wall_seconds_per_psi_evaluation"] is None


def test_cost_table_cli(capsys):
    assert hc.main(["--nodes", "4-6", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["psi_evaluations"] for r in rows] == [5_000, 129_600, 2_899_232]
    assert hc.main(["--nodes", "10", "--runs", "2", "--null", "1"]) == 0
    out = capsys.readouterr().out
    assert "ABOVE" in out and "7.4 years" in out


# ---------------------------------------------------------------------------
# The guard in prepare_hunter_campaign
# ---------------------------------------------------------------------------


def _layout(tmp_path, *, n_regions=5, n_time=64, subjects=("s1", "s2")):
    data_dir = tmp_path / "prep"
    rng = np.random.RandomState(11)
    paths = []
    for subj in subjects:
        run_dir = data_dir / subj / "awake" / "audio"
        run_dir.mkdir(parents=True)
        p = run_dir / f"{subj}_run-1_schaefer400_ts.npy"
        np.save(p, rng.rand(n_time, n_regions), allow_pickle=False)
        paths.append(p)
    return data_dir, paths


def _build(tmp_path, data_dir, **kwargs):
    profile = dataclasses.replace(
        get_execution_profile("hunter"),
        hunter_phase1_shards_per_run=2,
        hunter_cut_shards_per_run=2,
        hunter_phase1_workers_per_task=1,
        hunter_shared_memory=False,
    )
    options = dict(
        iim_bins=2,
        iim_lag_trs=1,
        iim_n_parts=4,
        iim_max_timepoints=None,
        iim_max_nodes=4,
        iim_max_mechanism_size=None,
        iim_max_purview_size=None,
    )
    options.update(kwargs)
    return hunter_iim.prepare_hunter_campaign(
        data_dir=data_dir,
        atlas="schaefer400",
        sessions=("awake",),
        condition="audio",
        stimulus_onsets=None,
        subjects=None,
        campaign_dir=tmp_path / "campaign",
        execution_profile=profile,
        step2_context={"hardware_target": "cpu"},
        **options,
    )


def test_build_refuses_above_the_ceiling_before_writing_anything(tmp_path):
    data_dir, _ = _layout(tmp_path)
    # 4 nodes, all sizes, 4 cuts: 625 * 5 = 3125 evaluations per run, 2 runs
    with pytest.raises(hc.HunterCostError) as err:
        _build(tmp_path, data_dir, max_psi_evaluations=6000)
    msg = str(err.value)
    assert "6.25e+03" in msg and "--hunter-allow-large" in msg
    assert "--hunter-max-psi-evals" in msg and "section 10a" in msg
    assert not (tmp_path / "campaign").exists()


def test_env_ceiling_refuses_and_allow_large_builds(tmp_path, monkeypatch, caplog):
    data_dir, _ = _layout(tmp_path)
    monkeypatch.setenv(hc.MAX_PSI_EVALS_ENV, "1000")
    with pytest.raises(hc.HunterCostError, match=hc.MAX_PSI_EVALS_ENV):
        _build(tmp_path, data_dir)
    assert not (tmp_path / "campaign").exists()

    with caplog.at_level("INFO", logger="impact_pipeline.hunter_iim"):
        manifest = _build(tmp_path, data_dir, allow_large=True)
    guard = manifest["iim_cost_estimate"]["guard"]
    assert guard == {
        "max_psi_evaluations": 1000,
        "source": f"environment ({hc.MAX_PSI_EVALS_ENV})",
        "allow_large": True,
        "exceeds_ceiling": True,
        "refused": False,
    }
    assert "ABOVE the ceiling, built because --hunter-allow-large" in caplog.text
    assert "is built because allow_large" in caplog.text
    assert (tmp_path / "campaign" / "pbs" / "00_submit_all.sh").exists()


def test_default_ceiling_refuses_the_exhaustive_ten_node_default(tmp_path):
    data_dir, _ = _layout(tmp_path, n_regions=12, subjects=("s1",))
    with pytest.raises(hc.HunterCostError, match="4.16e\\+11"):
        _build(tmp_path, data_dir, iim_bins=3, iim_max_nodes=None, iim_n_parts=None)
    assert not (tmp_path / "campaign").exists()


def test_manifest_records_preflight_and_enumerated_estimates(tmp_path):
    data_dir, paths = _layout(tmp_path, n_time=80)
    manifest = _build(
        tmp_path,
        data_dir,
        iim_null_surrogates=2,
        iim_bootstrap_n=2,
        seconds_per_psi_evaluation=1e-6,
    )
    est = manifest["iim_cost_estimate"]
    per_run = hc.iim_run_cost(4, n_parts=4)["psi_evaluations"]
    assert per_run == 625 * 5
    # 2 real runs x (1 + 2 surrogate + 2 bootstrap) replicate runs
    assert est["preflight"]["psi_evaluations"] == 2 * 5 * per_run
    assert est["psi_evaluations"] == sum(
        r["iim_cost"]["psi_evaluations"] for r in manifest["runs"] if r["defined"]
    )
    assert est["psi_evaluations"] <= est["preflight"]["psi_evaluations"]
    assert all(r["defined"] for r in manifest["runs"])
    assert est["psi_evaluations"] == est["preflight"]["psi_evaluations"]
    assert est["runs"] == {
        "real": 2,
        "null_per_run": 2,
        "bootstrap_per_run": 2,
        "total_iim_runs": 10,
    }
    assert est["guard"]["source"] == "default" and not est["guard"]["exceeds_ceiling"]
    rt = est["runtime"]
    assert rt["rate_source"] == "measured (--hunter-seconds-per-psi-eval)"
    assert rt["shard_hours"] == pytest.approx(est["psi_evaluations"] * 1e-6 / 3600)
    # 15 mechanisms in 2 shards (8 + 7, by size): the second holds 2 pairs,
    # 4 triples and the 4-node mechanism; 25 purview bipartitions
    assert est["max_phase1_shard_psi_evaluations"] == (2 * 1 + 4 * 3 + 7) * 25
    assert est["max_cut_shard_psi_evaluations"] == 2 * 625
    plan = json.loads(
        (tmp_path / "campaign" / "pbs" / "campaign_plan.json").read_text()
    )
    assert plan["iim_cost_estimate"]["psi_evaluations"] == est["psi_evaluations"]
    assert plan["iim_cost_estimate"]["guard"]["refused"] is False


def test_undefined_runs_cost_nothing_in_the_enumerated_estimate(tmp_path):
    data_dir, _ = _layout(tmp_path)
    short = data_dir / "s3" / "awake" / "audio"
    short.mkdir(parents=True)
    np.save(short / "s3_run-1_schaefer400_ts.npy", np.random.rand(1, 5))
    manifest = _build(tmp_path, data_dir)
    est = manifest["iim_cost_estimate"]
    per_run = hc.iim_run_cost(4, n_parts=4)["psi_evaluations"]
    # the preflight cannot see the too-short series: an upper bound
    assert est["preflight"]["psi_evaluations"] == 3 * per_run
    assert est["psi_evaluations"] == 2 * per_run
    undefined = [s for s in est["structures"] if not s["defined"]]
    assert undefined == [
        {"defined": False, "n_real_runs": 1, "reason": "insufficient_shape"}
    ]


# ---------------------------------------------------------------------------
# Measured calibration from the shard timings
# ---------------------------------------------------------------------------


def test_shard_timings_give_the_measured_calibration(tmp_path):
    data_dir, _ = _layout(tmp_path)
    manifest = _build(tmp_path, data_dir)
    campaign = tmp_path / "campaign"
    for i in range(len(manifest["phase1_tasks"])):
        hunter_iim.run_phase1_shard(campaign, i)
    for i in range(len(manifest["cut_tasks"])):
        hunter_iim.run_cut_shard(campaign, i)
    hunter_iim.run_reduce_all(campaign)

    cal = json.loads((campaign / "cost_calibration.json").read_text())
    assert cal["schema"] == hc.CALIBRATION_SCHEMA
    # every Psi evaluation of the campaign is attributed to exactly one shard
    assert cal["psi_evaluations"] == manifest["iim_cost_estimate"]["psi_evaluations"]
    assert cal["n_shards"] == len(manifest["phase1_tasks"]) + len(manifest["cut_tasks"])
    assert cal["stages"]["phase1-shard"]["psi_evaluations"] == 2 * 625
    assert cal["shard_wall_seconds_per_psi_evaluation"] > 0
    assert cal["overhead_dominated"] is True  # a tiny campaign
    assert cal["context"]["psi_kernels"] == ["numba"]
    assert cal["context"]["iim_structures"] == [
        {
            "n_nodes": 4,
            "bins": 2,
            "max_mechanism_size": 4,
            "max_purview_size": 4,
            "n_cuts_evaluated": 4,
        }
    ]
    status = hunter_iim.campaign_status(campaign)
    assert status["cost_calibration"]["psi_evaluations"] == cal["psi_evaluations"]

    # the measured rate feeds the estimate of the next build
    rate = cal["shard_wall_seconds_per_psi_evaluation"]
    rebuilt = _build(tmp_path / "next", data_dir, seconds_per_psi_evaluation=rate)
    rt = rebuilt["iim_cost_estimate"]["runtime"]
    assert rt["shard_seconds_per_psi_evaluation"] == pytest.approx(rate)
    assert rt["shard_hours"] == pytest.approx(cal["psi_evaluations"] * rate / 3600)


def test_resumed_cut_shard_counts_only_the_computed_cuts(tmp_path):
    data_dir, _ = _layout(tmp_path, subjects=("s1",))
    manifest = _build(tmp_path, data_dir, iim_n_parts=None)
    campaign = tmp_path / "campaign"
    reference = hunter_iim.run_cut_shard(campaign, 0)
    n_cuts = len(reference["cut_scores"])
    assert n_cuts >= 2
    shard_dir = campaign / "runs" / manifest["cut_tasks"][0]["run_key"] / "cut_shards"
    (shard_dir / "shard_0000.json").unlink()
    first = sorted(reference["cut_scores"])[0]
    (shard_dir / "shard_0000.partial.json").write_text(
        json.dumps(
            {
                "identity": reference["identity"],
                "cut_scores": {first: reference["cut_scores"][first]},
            }
        )
    )
    resumed = hunter_iim.run_cut_shard(campaign, 0)
    per_tpm = manifest["runs"][0]["iim_cost"]["psi_evaluations_per_tpm"]
    assert resumed["timing"]["resumed_cuts"] == 1
    assert resumed["timing"]["computed_cuts"] == n_cuts - 1
    assert resumed["timing"]["psi_evaluations"] == (n_cuts - 1) * per_tpm
    assert reference["timing"]["psi_evaluations"] == n_cuts * per_tpm


# ---------------------------------------------------------------------------
# Command line: run_pipeline.py and the smoke-test helper
# ---------------------------------------------------------------------------


def _cli_layout(tmp_path, n_regions=5):
    synth_root = tmp_path / "synth"
    bids = synth_root / "test_objects" / "datasets" / "ds003171"
    out = synth_root / "test_objects" / "runs" / "ds003171"
    bids.mkdir(parents=True)
    (bids / "dataset_description.json").write_text(
        json.dumps(
            {"Name": "synthetic", "SyntheticData": True, "DatasetType": "synthetic"}
        )
    )
    rng = np.random.RandomState(0)
    for subj in ("01",):
        (bids / f"sub-{subj}" / "func").mkdir(parents=True)
        for ses in ("awake", "deep"):
            d = out / "preprocessed" / subj / ses / "audio"
            d.mkdir(parents=True)
            np.save(d / f"{subj}_run-1_schaefer400_ts.npy", rng.rand(60, n_regions))
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith("IMPACT_HUNTER_")},
        "IMPACT_SYNTH_ROOT": str(synth_root),
        "IMPACT_SKIP_ENV_CHECK": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    return bids, out, env


def test_cli_flags_refuse_and_allow_large(tmp_path):
    bids, out, env = _cli_layout(tmp_path)
    base = [
        sys.executable,
        str(REPO / "run_pipeline.py"),
        "--execution-mode", "hunter",
        "--hunter-stage", "build-campaign",
        "--dataset-id", "ds003171",
        "--data-origin", "dummy",
        "--bids-root", str(bids),
        "--out-dir", str(out),
        "--mpc-metrics", "IIM",
        "--no-ci",
        "--iim-max-nodes", "4",
        "--iim-n-parts", "4",
        "--hunter-max-psi-evals", "1e3",
    ]
    refused = subprocess.run(base, capture_output=True, text=True, timeout=300, env=env)
    assert refused.returncode != 0
    assert "HunterCostError" in refused.stderr and "refused" in refused.stderr
    assert "IIM cost estimate" in refused.stderr
    campaign = out / "cache" / "hunter_iim_campaign"
    assert not (campaign / "campaign_manifest.json").exists()

    built = subprocess.run(
        base + ["--hunter-allow-large", "--hunter-seconds-per-psi-eval", "2e-6"],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert built.returncode == 0, built.stderr[-2000:]
    manifest = json.loads((campaign / "campaign_manifest.json").read_text())
    est = manifest["iim_cost_estimate"]
    assert est["guard"]["exceeds_ceiling"] and est["guard"]["allow_large"]
    assert est["runtime"]["shard_seconds_per_psi_evaluation"] == pytest.approx(2e-6)
    prov = json.loads((out / "cache" / "provenance_manifest.json").read_text())
    hunter = prov["parameters"]["hunter"]
    assert hunter["max_psi_evals"] == 1000 and hunter["allow_large"] is True

    bad = subprocess.run(
        base[:-1] + ["zero"], capture_output=True, text=True, timeout=300, env=env
    )
    assert bad.returncode == 2 and "--hunter-max-psi-evals" in bad.stderr


def test_smoke_helper_accepts_a_calibration_size(tmp_path):
    bids, out, env = _cli_layout(tmp_path, n_regions=6)
    env.pop("IMPACT_HUNTER_SETUP_FILE", None)
    proc = subprocess.run(
        [
            "bash",
            str(REPO / "scripts" / "hunter" / "hunter_smoke_test.sh"),
            "--bids-root", str(bids),
            "--out-dir", str(out),
            "--subjects", "01",
            "--hardware-target", "cpu",
            "--python", sys.executable,
            "--iim-max-nodes", "5",
            "--iim-max-mechanism-size", "all",
            "--iim-max-purview-size", "all",
            "--iim-n-parts", "all",
        ],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    campaign = out / "cache" / "hunter_iim_smoke"
    manifest = json.loads((campaign / "campaign_manifest.json").read_text())
    assert all(r["n_nodes_used"] == 5 for r in manifest["runs"])
    assert all(r["iim_max_mechanism_size"] is None for r in manifest["runs"])
    assert all(r["n_cuts_evaluated"] == 15 for r in manifest["runs"])
    # 2 runs x 129,600 evaluations (exhaustive 5-node IIM)
    assert manifest["iim_cost_estimate"]["psi_evaluations"] == 2 * 129_600
    assert "cost_calibration.json" in proc.stdout


# ---------------------------------------------------------------------------
# Documentation (runbook, helper README, preregistration errata)
# ---------------------------------------------------------------------------


def test_runbook_and_readme_document_the_sizing_procedure():
    runbook = (REPO / "docs" / "HLRS_HUNTER_RUNBOOK.md").read_text(encoding="utf-8")
    assert "## 10a. Size the IIM configuration before the full campaign" in runbook
    # prominent: linked from the top of the runbook, before the first section
    head = runbook.split("## 1. What runs on Hunter", 1)[0]
    assert "#10a-size-the-iim-configuration-before-the-full-campaign" in head
    section = runbook.split("## 10a.", 1)[1].split("## 11.", 1)[0]
    flat = " ".join(section.split())  # prose is wrapped
    for token in (
        "1.7 s",
        "46 s",
        "27 min",
        "10-node default is infeasible",
        "--iim-max-nodes 6",
        "cost_calibration.json",
        "shard_wall_seconds_per_psi_evaluation",
        "--hunter-seconds-per-psi-eval",
        "--hunter-max-psi-evals",
        "--hunter-allow-large",
        "What to report back to the author",
        "impact_pipeline.hunter_cost",
    ):
        assert token in flat, token
    readme = (REPO / "scripts" / "hunter" / "README.md").read_text(encoding="utf-8")
    for token in (
        "## IIM cost guard and sizing",
        hc.MAX_PSI_EVALS_ENV,
        hc.SECONDS_PER_PSI_EVAL_ENV,
        "--hunter-allow-large",
        "--iim-max-nodes 6",
    ):
        assert token in readme, token
    # the numbers quoted in the runbook come from the cost model
    for n, text in ((7, "5.97e7"), (8, "1.17e9"), (9, "2.23e10"), (10, "4.16e11")):
        assert f"| {n} | {text} |" in section, n
        assert hc.iim_run_cost(n)["psi_evaluations"] == pytest.approx(
            float(text), rel=5e-3
        ), n


def test_preregistration_errata_is_documentation_only():
    prereg = (
        REPO / "docs" / "preregistration" / "MPC_BENCH_PREREGISTRATION.md"
    ).read_text(encoding="utf-8")
    title = "## Errata (documentation only; no change to hypotheses or decision rules)"
    assert prereg.count(title) == 1
    frozen, errata = prereg.split(title)
    # the frozen text is unchanged; the errata carry the corrected numbers
    assert "209 regions" in frozen and "280 per family" in frozen
    assert "2026-09-29" in errata
    assert "76" in errata and "265" in errata and "260 per family" in errata
    from impact_pipeline.bench import whole_brain
    from impact_pipeline.bench.run_bench import witness_tasks

    prov = whole_brain.load_connectome().provenance
    assert (prov["n_nodes"], prov["n_edges"]) == (76, 265)
    assert prov["edge_confidence_min"] == 209
    seeds = range(10000, 10020)
    assert len(witness_tasks(seeds, family="A")) == 260
    assert len(witness_tasks(seeds, family="C")) == 260


# ---------------------------------------------------------------------------
# Walltime syntax, invalid sizes, malformed timings, build commands, errata
# ---------------------------------------------------------------------------


def test_walltime_for_the_estimate_follows_the_scheduler_syntax():
    pbs = {"scheduler": "pbs", "cut_time": "24:00:00", "phase1_time": "00:25:00"}
    assert hunter_iim._walltime_seconds(pbs, "cut_time") == 86400
    assert hunter_iim._walltime_seconds(pbs, "phase1_time") == 1500
    assert hunter_iim._walltime_seconds(pbs, "reduce_time") is None
    # Slurm --time: a bare number is minutes and after "D-" the first field
    # is hours (D-H:M is not PBS's D-H:M:S read with seconds last)
    for text, seconds in (
        ("90", 5400),
        ("30:15", 1815),
        ("02:00:00", 7200),
        ("1-12", 129600),
        ("1-12:30", 131400),
        ("1-12:30:05", 131405),
    ):
        slurm = {"scheduler": "slurm", "cut_time": text}
        assert hunter_iim._walltime_seconds(slurm, "cut_time") == seconds, text
    for text in ("UNLIMITED", "1:2:3:4", ""):
        slurm = {"scheduler": "slurm", "cut_time": text}
        assert hunter_iim._walltime_seconds(slurm, "cut_time") is None, text


def test_cost_table_cli_rejects_invalid_sizes(capsys):
    for argv in (
        ["--nodes", "1-3"],
        ["--nodes", "x"],
        ["--n-parts", "0"],
        ["--max-mechanism-size", "-1"],
        ["--null", "-1"],
        ["--runs", "0"],
    ):
        with pytest.raises(SystemExit) as exc:
            hc.main(argv)
        assert exc.value.code == 2, argv
    capsys.readouterr()
    assert hc.main(["--nodes", "10", "--max-mechanism-size", "3",
                    "--max-purview-size", "3", "--n-parts", "64", "--json"]) == 0
    (row,) = json.loads(capsys.readouterr().out)
    # the runbook's example: 405 x 405 per TPM, 1 + 64 TPMs
    assert row["psi_evaluations_per_tpm"] == 164_025
    assert row["tpm_evaluations"] == 65


def test_status_survives_a_malformed_timing_record(tmp_path):
    data_dir, _ = _layout(tmp_path, subjects=("s1",))
    manifest = _build(tmp_path, data_dir)
    campaign = tmp_path / "campaign"
    done = hunter_iim.run_phase1_shard(campaign, 0)
    bad = campaign / "timing" / hunter_iim.CUT_STAGE
    bad.mkdir(parents=True, exist_ok=True)
    (bad / "broken.json").write_text(
        json.dumps({"wall_seconds": 1.0, "psi_evaluations": "n/a"})
    )
    status = hunter_iim.campaign_status(campaign)
    cal = status["cost_calibration"]
    # the malformed record is ignored; the completed shard is counted
    assert cal["n_shards"] == 1
    assert cal["psi_evaluations"] == done["timing"]["psi_evaluations"] > 0
    assert cal["psi_evaluations"] < manifest["iim_cost_estimate"]["psi_evaluations"]
    assert hc.calibration_from_records([None, [1], {"psi_evaluations": None}])[
        "n_shards"
    ] == 0


def test_documented_build_commands_set_the_iim_subsystem_size():
    # Without --iim-max-nodes the build uses the 10-node default, which the
    # cost guard refuses; every documented build command must set it.
    docs = [
        REPO / "README.md",
        REPO / "docs" / "HLRS_HUNTER_RUNBOOK.md",
        REPO / "scripts" / "hunter" / "README.md",
    ]
    n_commands = 0
    for path in docs:
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            continued = line.rstrip().endswith("\\")
            if "--hunter-stage build-campaign" not in line or not continued:
                continue
            command = [line]
            j = i
            while lines[j].rstrip().endswith("\\"):
                j += 1
                command.append(lines[j])
            n_commands += 1
            assert "--iim-max-nodes" in " ".join(command), (path.name, i + 1)
    assert n_commands >= 4


def test_errata_leave_the_frozen_preregistration_text_unchanged():
    rel = "docs/preregistration/MPC_BENCH_PREREGISTRATION.md"
    try:
        frozen_at_tag = subprocess.run(
            ["git", "show", f"mpcbench-freeze-v1:{rel}"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git is not available")
    if frozen_at_tag.returncode != 0:
        pytest.skip("freeze tag not available in this checkout")
    title = "## Errata (documentation only; no change to hypotheses or decision rules)"
    current = (REPO / rel).read_text(encoding="utf-8")
    frozen_now = current.split(title, 1)[0]
    assert frozen_now.rstrip("\n") == frozen_at_tag.stdout.rstrip("\n")
