"""MPC-Bench export layout (discovered by synergy_ci and the events resolver
exactly like preprocessed data), the in-memory runner and the bench runner
(seed policy, confirmatory guard, process pool, resume, PBS template)."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from impact_pipeline.bench import export, run_bench
from impact_pipeline.bench import generators as g
from impact_pipeline.bench.factorial import factorial_tasks
from impact_pipeline.bench.patchwork import simulate_patchwork
from impact_pipeline.event_parsing import read_events_table, resolve_events_file
from impact_pipeline.synergy_ci import build_ci_run_specs, discover_ci_subjects

SMALL = g.AgentConfig(n_trials=16, n_reafference_pairs=6)
SMALL_DICT = {"n_trials": 16, "n_reafference_pairs": 6}
SESSIONS = ("nominal", "noplast")


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    root = tmp_path_factory.mktemp("bench_export")
    cfg = SMALL.replace(rest_sec=30.0)
    items = []
    systems = {}
    for seed in (1, 2):
        subj = f"s{seed:05d}"
        for ses, kn in zip(
            SESSIONS, (g.NOMINAL_KNOBS, g.NOMINAL_KNOBS.replace(eta=0.0))
        ):
            s = g.simulate_family_a(kn, cfg, seed=seed)
            items.append((subj, ses, s))
            systems[(subj, ses)] = s
    manifest = export.export_bench(items, root, provenance={"test": True})
    return root, manifest, systems


def test_export_layout_is_discovered_by_synergy_ci(exported):
    root, manifest, systems = exported
    prep = root / "prep"
    assert discover_ci_subjects(str(prep)) == ["s00001", "s00002"]
    specs = build_ci_run_specs(
        str(prep), export.DEFAULT_ATLAS, SESSIONS, export.DEFAULT_CONDITION
    )
    assert [(s["subject"], s["session"]) for s in specs] == [
        ("s00001", "nominal"),
        ("s00001", "noplast"),
        ("s00002", "nominal"),
        ("s00002", "noplast"),
    ]
    for spec in specs:
        arr = np.load(spec["ts_path"])
        sysm = systems[(spec["subject"], spec["session"])]
        assert arr.shape == (sysm.n_time, sysm.n_nodes)  # time x nodes
        assert np.allclose(arr, sysm.ts.T)
    # The macro-node (IIM grain) array has its own atlas key and glob.
    macro = build_ci_run_specs(
        str(prep), export.DEFAULT_MACRO_ATLAS, SESSIONS, export.DEFAULT_CONDITION
    )
    assert len(macro) == 4
    assert np.load(macro[0]["ts_path"]).shape[1] == len(
        systems[("s00001", "nominal")].meta["iim_macro_nodes"]
    )
    # Event-based run selection matches the exported run id.
    onsets = export.load_export_onsets(root, ["s00001"], SESSIONS)
    assert onsets["s00001"]["nominal"][1] == "1"
    sel = build_ci_run_specs(
        str(prep),
        export.DEFAULT_ATLAS,
        SESSIONS,
        export.DEFAULT_CONDITION,
        stimulus_onsets=onsets,
        subjects=["sub-s00001"],
    )
    assert [Path(s["ts_path"]).name for s in sel] == ["s00001_run-1_bench_ts.npy"] * 2
    # PDI state-matched rest baseline lives under <subj>/<session>/rest.
    rest = sorted((prep / "s00001" / "nominal" / "rest").glob("s00001_*_bench_ts.npy"))
    assert len(rest) == 1 and np.load(rest[0]).shape == (600, 30)
    assert manifest["runs"][0]["paths"]["ts"].startswith("prep/")


def test_export_events_and_sidecars_match_pipeline_readers(exported):
    from impact_pipeline import run_synergy_ci as rsc
    from impact_pipeline.provenance import assert_origin_matches_dataset

    root, _, systems = exported
    bids = root / "bids"
    fn = resolve_events_file(bids, "s00002", "noplast", condition="bench")
    assert fn is not None and fn.name == "sub-s00002_task-benchnoplast_run-1_events.tsv"
    table = read_events_table(fn)
    ref = systems[("s00002", "noplast")].events
    assert list(table.columns) == list(g.EVENT_COLUMNS)
    assert np.allclose(table["onset"], ref["onset"])
    assert list(table["trial_type"]) == list(ref["trial_type"])
    assert table.loc[ref["yoked_to"].isna().to_numpy(), "yoked_to"].isna().all()
    bundle, run_id = rsc.load_onsets(str(bids), "s00002", "noplast", condition="bench")
    assert run_id == "1"
    assert len(bundle["goal_onsets"]) == SMALL.n_trials
    assert len(bundle["feedback_values"]) == SMALL.n_trials
    assert (
        len(bundle["self_onsets"])
        == len(bundle["nonself_onsets"])
        == SMALL.n_reafference_pairs
    )
    assert rsc._infer_sample_interval_seconds(str(bids)) == pytest.approx(SMALL.dt)
    desc = json.loads((bids / "dataset_description.json").read_text())
    assert desc["SyntheticData"] is True
    with pytest.raises(ValueError):
        assert_origin_matches_dataset(bids, "real")
    side = json.loads(
        fn.with_name(fn.name.replace("_events.tsv", "_bold.json")).read_text()
    )
    assert side["RepetitionTime"] == pytest.approx(SMALL.dt)
    declared = side["MPCBenchDeclared"]
    assert (
        declared["workspace_nodes"]
        == systems[("s00002", "noplast")].meta["workspace_nodes"]
    )
    # Hidden oracle channels are only under <root>/oracle.
    oracle_files = list((root / "oracle").rglob("*_oracle.*"))
    assert len(oracle_files) == 8
    for tree in ("prep", "bids"):
        assert not list((root / tree).rglob("*oracle*"))
        for js in (root / tree).rglob("*.json"):
            text = js.read_text()
            assert "q_by_trial" not in text and "intended_bits" not in text
    orc = json.loads(
        (
            root / "oracle" / "s00001" / "noplast" / "s00001_run-1_oracle.json"
        ).read_text()
    )
    assert orc["intended_bits"] == [0, 1, 1, 1, 1]
    npz = np.load(root / "oracle" / "s00001" / "noplast" / "s00001_run-1_oracle.npz")
    assert np.all(npz["q_by_trial"] == 0)


def test_compute_synergy_ci_runs_unchanged_on_export(exported):
    root, _, systems = exported
    df = export.run_export_with_synergy_ci(
        root, SESSIONS, subjects=["s00001"], metrics=("NAS", "SRPI")
    )
    assert len(df) == 2 and set(df["session"]) == set(SESSIONS)
    assert df["NAS"].notna().all() and df["SRPI"].notna().all()
    assert set(df["data_origin"]) == {"dummy"}
    iim = export.run_export_with_synergy_ci(
        root,
        ("nominal",),
        subjects=["s00001"],
        metrics=("IIM",),
        atlas=export.DEFAULT_MACRO_ATLAS,
    )
    assert bool(iim["IIM_defined"].iloc[0])
    params = export.synergy_ci_params(systems[("s00001", "nominal")].meta)
    assert (
        params["nas_params"]["workspace_nodes"]
        == systems[("s00001", "nominal")].meta["workspace_nodes"]
    )
    assert params["srpi_params"]["modality"] == "eeg"


def test_run_in_memory_components_and_nulls():
    s = g.simulate_family_a(None, SMALL, seed=3)
    res = export.run_in_memory(s, metrics=("PDI", "NAS", "SRPI"), null_surrogates=0)
    comps = res["components"]
    assert set(comps) == {"PDI", "NAS", "SRPI"}
    for c in comps.values():
        assert set(c) >= {
            "estimate",
            "value",
            "null_mean",
            "null_sd",
            "n_null",
            "defined",
            "reason",
            "bearer_id",
            "seconds",
        }
        assert c["bearer_id"] == "system" and c["n_null"] == 0
    cal = export.run_in_memory(
        s, metrics=("NAS", "SRPI", "IIM"), null_surrogates=3, null_seed=5
    )["components"]
    assert cal["NAS"]["n_null"] == 3 and np.isfinite(cal["NAS"]["null_sd"])
    assert cal["NAS"]["null_family"] == "circular_shift"
    assert cal["IIM"]["statistic"] == "Delta_Psi_bits" and cal["IIM"]["n_null"] == 3
    assert cal["SRPI"]["n_null"] == 3
    assert cal["SRPI"]["null_family"] in (
        "circular_shift_fallback",
        "nulls.component_null",
    )
    again = export.run_in_memory(s, metrics=("NAS",), null_surrogates=3, null_seed=5)
    assert again["components"]["NAS"]["null_mean"] == cal["NAS"]["null_mean"]
    with pytest.raises(ValueError):
        export.run_in_memory(s, metrics=("PHI",))
    with pytest.raises(ValueError):
        export.run_in_memory(s, metrics=("NAS",), params={"XYZ": {}})


def test_bearer_views_on_patchwork():
    s = simulate_patchwork(None, SMALL, seed=2)
    sys_view = export.bearer_view(s, "IIM", "system")
    assert len(sys_view["macro_nodes"]) == 5 and sys_view["bearer_id"] == "system"
    iim = export.bearer_view(s, "IIM", "principle")
    assert iim["bearer_id"] == "principle:IIM" and len(iim["macro_nodes"]) == 3
    assert iim["nodes"] == s.meta["principle_bearers"]["IIM"]
    nas = export.bearer_view(s, "NAS", "principle")
    hub = s.meta["principle_workspace_nodes"]["NAS"]
    assert [nas["nodes"][i] for i in nas["workspace_nodes"]] == hub
    res = export.run_in_memory(s, metrics=("IIM", "SRPI"), bearer_mode="principle")
    ids = {c["bearer_id"] for c in res["components"].values()}
    assert ids == {"principle:IIM", "principle:SRPI"}
    # Systems without per-principle bearers keep the system view.
    a = g.simulate_family_a(None, SMALL, seed=2)
    assert export.bearer_view(a, "NAS", "principle")["bearer_id"] == "system"
    with pytest.raises(ValueError):
        export.bearer_view(a, "NAS", "module")


def test_optional_modes_are_detected_by_signature():
    def old(ts, tr=None):
        return 0.0

    def new(ts, tr=None, mode="default", events=None):
        return 0.0

    ev = pd.DataFrame({"onset": [1.0]})
    assert export._optional_kwargs("SRPI", old, ev) == ({}, {})
    kw, used = export._optional_kwargs("SRPI", new, ev)
    assert kw["mode"] == "agency" and used == {"mode": "agency", "events": "dataframe"}


def test_evidence_verdict_degrades_without_evidence_layer(monkeypatch):
    import sys

    s = g.simulate_family_a(None, SMALL, seed=3)
    res = export.run_in_memory(s, metrics=("NAS",), null_surrogates=2)
    monkeypatch.setitem(sys.modules, "impact_pipeline.evidence", None)
    out = export.evidence_verdict(res, s.meta)
    assert out["verdict"] is None and out["reasons"] == ["evidence_layer_unavailable"]


def test_evidence_verdict_uses_evidence_layer_when_available():
    pytest.importorskip("impact_pipeline.evidence")
    s = g.simulate_family_a(None, SMALL, seed=3)
    res = export.run_in_memory(s, metrics=("NAS", "SRPI"), null_surrogates=2)
    out = export.evidence_verdict(res, s.meta)
    assert out["verdict"] in (None, "ATTRIBUTED", "NOT_ATTRIBUTED", "UNDETERMINED")


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def test_seed_policy():
    assert run_bench.seed_set(0) == "dev" and run_bench.seed_set(999) == "dev"
    assert run_bench.seed_set(5000) == "reserved"
    assert run_bench.seed_set(10000) == "confirmatory"
    dev_a = factorial_tasks([0], cells=["b11111"])
    dev_c = factorial_tasks([0], cells=["b11111"], family="C")
    conf_c = factorial_tasks([10000], cells=["b11111"], family="C")
    run_bench.check_seed_policy(dev_a, confirmatory=False)
    run_bench.check_seed_policy(conf_c, confirmatory=True)
    with pytest.raises(ValueError, match="held out"):
        run_bench.check_seed_policy(dev_c, confirmatory=False)
    with pytest.raises(ValueError, match="confirmatory"):
        run_bench.check_seed_policy(factorial_tasks([10000], cells=["b11111"]), False)
    with pytest.raises(ValueError, match=">= 10000"):
        run_bench.check_seed_policy(dev_a, confirmatory=True)
    with pytest.raises(ValueError, match="reserved"):
        run_bench.check_seed_policy(factorial_tasks([1500], cells=["b11111"]), False)


def _git(cwd, *args):
    subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.org",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.org",
            "HOME": str(cwd),
            "PATH": "/usr/bin:/bin:/usr/local/bin",
        },
    )


def test_confirmatory_guard_requires_clean_tree(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "a.py").write_text("x = 1\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    _git(repo, "tag", "freeze-v1")
    info = run_bench.confirmatory_guard(repo, freeze_tag="freeze-v1")
    assert info["confirmatory"] and len(info["git_sha"]) == 40
    assert info["freeze_tag"] == "freeze-v1" and "freeze-v1" in info["tags_at_head"]
    with pytest.raises(RuntimeError, match="not found"):
        run_bench.confirmatory_guard(repo, freeze_tag="nope")
    (repo / "src" / "new.py").write_text("y = 2\n")
    with pytest.raises(RuntimeError, match="untracked"):
        run_bench.confirmatory_guard(repo)
    (repo / "src" / "new.py").unlink()
    (repo / "src" / "a.py").write_text("x = 2\n")
    with pytest.raises(RuntimeError, match="clean"):
        run_bench.confirmatory_guard(repo)
    with pytest.raises(RuntimeError):
        run_bench.confirmatory_guard(tmp_path / "not_a_repo")


def test_run_tasks_writes_results_with_provenance_and_resumes(tmp_path):
    tasks = factorial_tasks([0], cells=["b11111", "b01111"], config=SMALL_DICT)
    out = tmp_path / "run"
    recs = run_bench.run_tasks(
        tasks, out, n_workers=1, metrics=("NAS", "SRPI"), null_surrogates=2
    )
    assert len(recs) == 2 and all(r["status"] == "ok" for r in recs)
    lines = (out / run_bench.RESULTS_JSONL).read_text().strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["schema"] == run_bench.RESULT_SCHEMA
    assert rec["seed_set"] == "dev" and rec["intended_bits"] in (
        [1] * 5,
        [0, 1, 1, 1, 1],
    )
    assert "code_version" in rec["provenance"]
    assert rec["timing"]["simulate_s"] > 0 and "NAS" in rec["timing"]["estimators_s"]
    assert "LZc" in rec["markers"] and "PhiR_bits" in rec["markers"]
    assert rec["verdict"]["verdict"] is None or rec["verdict"]["verdict"] in (
        "ATTRIBUTED",
        "NOT_ATTRIBUTED",
        "UNDETERMINED",
    )
    df = pd.read_csv(out / run_bench.RESULTS_CSV)
    assert list(df["cell_id"]) == ["b01111", "b11111"]
    assert set(df["intended_bits"]) == {"b01111", "b11111"}
    assert {"NAS_z", "SRPI_estimate", "marker_LZc", "oracle_q_range", "git_sha"} <= set(
        df.columns
    )
    manifest = json.loads((out / run_bench.MANIFEST).read_text())
    assert manifest["n_run"] == 2 and "estimator_params" in manifest
    assert manifest["runtime"]["python"]
    again = run_bench.run_tasks(tasks, out, metrics=("NAS", "SRPI"), null_surrogates=2)
    assert again == []
    # Serial runs must not leave process-wide logging disabled.
    import logging

    assert logging.root.manager.disable == logging.NOTSET
    assert json.loads((out / run_bench.MANIFEST).read_text())["n_skipped_done"] == 2


def test_process_pool_and_shards_merge(tmp_path, monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    tasks = run_bench.witness_tasks(
        [0], witness_ids=["PC_nominal", "N_ar1", "O_inert"], config=SMALL_DICT
    )
    parts = [run_bench.shard(tasks, i, 2) for i in range(2)]
    assert sorted(t.task_id for p in parts for t in p) == sorted(
        t.task_id for t in tasks
    )
    assert not set(t.task_id for t in parts[0]) & set(t.task_id for t in parts[1])
    for i, part in enumerate(parts):
        run_bench.run_tasks(
            part,
            tmp_path / f"shard-{i:04d}",
            n_workers=2,
            metrics=("SRPI",),
            markers=False,
        )
    import os

    assert "OMP_NUM_THREADS" not in os.environ  # pool env is scoped to the pool
    df = run_bench.merge_results(tmp_path)
    assert sorted(df["task_id"]) == sorted(t.task_id for t in tasks)
    assert (df["status"] == "ok").all()


def test_run_task_records_errors_instead_of_raising():
    bad = factorial_tasks([0], cells=["b11111"], config={"n_trials": 0})[0]
    rec = run_bench.run_task(bad, metrics=("SRPI",))
    assert rec["status"] == "error" and "n_trials" in rec["error"]


def test_pbs_template_and_cli(tmp_path, capsys):
    script = run_bench.pbs_array_script(
        ["factorial", "--seeds", "0-19"], n_shards=4, group_list="grp"
    )
    assert script.startswith("#!/bin/bash")
    for line in (
        "#PBS -J 0-3",
        "#PBS -r y",
        "#PBS -l select=1:node_type=mi300a",
        "#PBS -W group_list=grp",
        'cd "$PBS_O_WORKDIR"',
        "--n-shards 4",
    ):
        assert line in script
    single = run_bench.pbs_array_script(["factorial"], n_shards=1)
    assert "#PBS -J" not in single and "--shard-index 0" in single
    target = tmp_path / "bench.pbs"
    rc = run_bench.main(
        [
            "factorial",
            "--seeds",
            "0-1",
            "--n-shards",
            "3",
            "--out",
            "x",
            "--pbs-template",
            str(target),
        ]
    )
    assert rc == 0
    text = target.read_text()
    assert "#PBS -J 0-2" in text and " --out x" not in text and "--seeds 0-1" in text
    assert run_bench.main(["factorial", "--seeds", "0", "--list"]) == 0
    listed = capsys.readouterr().out.split()
    assert len([t for t in listed if t.startswith("factorial-A-")]) == 32
    assert run_bench.main(["factorial", "--family", "C", "--seeds", "0", "--list"]) == 2
    assert run_bench.parse_seeds("0-2,7") == [0, 1, 2, 7]


def test_timing_report_measures_each_generator():
    rep = run_bench.timing_report(seeds=(0,), config=SMALL_DICT, binary_steps=500)
    rows = rep["timings"]
    assert {"family_A", "family_C", "patchwork"} <= set(rows)
    assert sum(k.startswith("family_B_") for k in rows) == len(g.BINARY_NETWORK_KINDS)
    assert all(r["mean_s"] > 0 for r in rows.values())
    assert rows["family_A"]["shape"][0] == 30
