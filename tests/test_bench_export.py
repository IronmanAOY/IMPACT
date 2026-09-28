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


COMPONENT_FIELDS = {
    "estimate",
    "value",
    "null_mean",
    "null_sd",
    "n_null",
    "null_family",
    "null_impl",
    "defined",
    "reason",
    "bearer_id",
    "seconds",
}


def test_run_in_memory_components_and_nulls():
    """Legacy estimator modes: no null unless ``null_surrogates > 0``."""
    s = g.simulate_family_a(None, SMALL, seed=3)
    res = export.run_in_memory(
        s, metrics=("PDI", "NAS", "SRPI"), null_surrogates=0, use_optional_modes=False
    )
    comps = res["components"]
    assert set(comps) == {"PDI", "NAS", "SRPI"}
    for c in comps.values():
        assert set(c) >= COMPONENT_FIELDS
        assert c["bearer_id"] == "system" and c["n_null"] == 0
        assert c["null_family"] is None and c["null_impl"] is None
    cal = export.run_in_memory(
        s,
        metrics=("NAS", "SRPI", "IIM"),
        null_surrogates=3,
        null_seed=5,
        use_optional_modes=False,
    )["components"]
    assert cal["NAS"]["n_null"] == 3 and np.isfinite(cal["NAS"]["null_sd"])
    assert cal["NAS"]["null_family"] == "circular_shift"
    assert cal["NAS"]["null_impl"] == "estimator"
    assert cal["IIM"]["statistic"] == "Delta_Psi_bits" and cal["IIM"]["n_null"] == 3
    assert cal["SRPI"]["n_null"] + cal["SRPI"]["n_null_failed"] == 3
    assert cal["SRPI"]["null_family"] == "label_permutation"
    assert cal["SRPI"]["null_impl"] in ("bench_local", "impact_pipeline.nulls")
    again = export.run_in_memory(
        s, metrics=("NAS",), null_surrogates=3, null_seed=5, use_optional_modes=False
    )
    assert again["components"]["NAS"]["null_mean"] == cal["NAS"]["null_mean"]
    with pytest.raises(ValueError):
        export.run_in_memory(s, metrics=("PHI",))
    with pytest.raises(ValueError):
        export.run_in_memory(s, metrics=("NAS",), params={"XYZ": {}})


def test_self_calibrating_modes_record_their_own_null():
    """PDI repertoire, NAS capacity and SRPI agency return their own null
    family even without ``null_surrogates``; SRPI-agency receives the agency
    events (yoked replays) and is defined on the nominal agent."""
    s = g.simulate_family_a(None, SMALL, seed=3)
    res = export.run_in_memory(s, metrics=("PDI", "NAS", "SRPI"), null_surrogates=0)
    modes = res["estimator_modes"]
    if modes["SRPI"].get("mode") != "agency":
        pytest.skip("installed compute_SRPI has no agency mode")
    assert modes["SRPI"]["agency_events"] == "bundle"
    assert modes["PDI"] == {"mode": "repertoire"}
    comps = res["components"]
    for p in ("PDI", "NAS", "SRPI"):
        c = comps[p]
        assert set(c) >= COMPONENT_FIELDS
        assert c["defined"], (p, c["reason"])
        # the repertoire null counts states (integers): its SD can be 0
        assert c["n_null"] >= 2 and np.isfinite(c["null_sd"]) and c["null_sd"] >= 0
        assert c["null_impl"] == "estimator" and c["null_family"]
        assert c["statistic"] == "raw"
    assert comps["NAS"]["null_sd"] > 0 and comps["SRPI"]["null_sd"] > 0
    assert comps["PDI"]["null_family"] == "circular_shift"
    assert comps["SRPI"]["null_family"] == "yoked_label_permutation"
    # The returned value of SRPI-agency is the excess over its own null.
    srpi = comps["SRPI"]
    assert srpi["value"] == pytest.approx(srpi["estimate"] - srpi["null_mean"])


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


def test_pdi_provenance_does_not_claim_events_it_did_not_pass():
    # compute_PDI accepts ``events`` (labelled mode='repertoire'), but the
    # bench never passes them to PDI, so the recorded modes must not say so.
    s = g.simulate_family_a(None, SMALL, seed=3)
    res = export.run_in_memory(s, metrics=("PDI",), null_surrogates=2)
    assert "events" not in res["estimator_modes"]["PDI"]
    assert res["estimator_modes"]["PDI"] == export.OPTIONAL_MODES["PDI"]


def test_evidence_verdict_degrades_without_evidence_layer(monkeypatch):
    import sys

    import impact_pipeline

    s = g.simulate_family_a(None, SMALL, seed=3)
    res = export.run_in_memory(s, metrics=("NAS",), null_surrogates=2)
    # Hide the module even when another test has already imported it (the
    # package attribute would otherwise satisfy the import).
    monkeypatch.setitem(sys.modules, "impact_pipeline.evidence", None)
    monkeypatch.delattr(impact_pipeline, "evidence", raising=False)
    out = export.evidence_verdict(res, s.meta)
    assert out["verdict"] is None and out["reasons"] == ["evidence_layer_unavailable"]


def test_evidence_verdict_uses_evidence_layer_when_available():
    pytest.importorskip("impact_pipeline.evidence")
    s = g.simulate_family_a(None, SMALL, seed=3)
    res = export.run_in_memory(s, metrics=("NAS", "SRPI"), null_surrogates=2)
    out = export.evidence_verdict(res, s.meta)
    # The layer is installed: its API must be driven without error.
    assert out["verdict"] in ("MPC_CONSISTENT", "EXCLUDED", "UNDETERMINED")
    assert not any(r.startswith("evidence_layer_") for r in out["reasons"])
    assert set(out["component_status"]) >= {"NAS", "SRPI"}


def _without_nulls_module(monkeypatch):
    import sys

    import impact_pipeline

    monkeypatch.setitem(sys.modules, "impact_pipeline.nulls", None)
    monkeypatch.delattr(impact_pipeline, "nulls", raising=False)


def test_local_event_surrogates_keep_the_declared_structure():
    s = g.simulate_family_a(None, SMALL, seed=6)
    ev = s.events
    rng = np.random.default_rng(0)
    t_max = s.n_time * s.dt
    kw = export._event_null_kwargs("RAM", s.n_time, s.dt)
    assert kw == {"common": True, "t_max": t_max, "min_shift": 0.1 * t_max}
    shifted = export._local_event_surrogate("onset_jitter", ev, rng, **kw)
    d = np.mod(shifted["onset"].to_numpy() - ev["onset"].to_numpy(), t_max)
    # One rigid shift for every event, >= 10% of the run from either end.
    assert np.allclose(d, d[0]) and 0.1 * t_max <= d[0] <= 0.9 * t_max
    assert shifted["onset"].between(0, t_max, inclusive="left").all()
    assert list(shifted["trial_type"]) == list(ev["trial_type"])
    kw = export._event_null_kwargs("SRPI", s.n_time, s.dt)
    perm = export._local_event_surrogate("label_permutation", ev, rng, **kw)
    assert np.array_equal(perm["onset"].to_numpy(), ev["onset"].to_numpy())
    agency = ev["trial_type"].isin(export.SRPI_NULL_LABELS).to_numpy()
    assert list(perm.loc[~agency, "trial_type"]) == list(ev.loc[~agency, "trial_type"])
    before = ev[agency].groupby(["phase_bin", "trial_type"]).size()
    after = perm[agency].groupby(["phase_bin", "trial_type"]).size()
    pd.testing.assert_series_equal(before, after)
    # Across draws the labels are really re-dealt.
    draws = [
        export._local_event_surrogate("label_permutation", ev, rng, **kw)
        for _ in range(5)
    ]
    assert any(
        not np.array_equal(p["trial_type"].to_numpy(), ev["trial_type"].to_numpy())
        for p in draws
    )
    with pytest.raises(ValueError):
        export._local_event_surrogate("circular_shift", ev, rng)


def test_event_nulls_feed_surrogate_events_to_the_estimator(monkeypatch):
    _without_nulls_module(monkeypatch)
    s = g.simulate_family_a(None, SMALL, seed=6)
    seen = []

    def fn(ts, ev):
        seen.append(ev)
        return {"value": float(len(ev))}

    vals, kind, impl, n_failed = export._event_null(
        "SRPI", fn, s.ts, s.events, 4, 3, s.dt
    )
    assert (kind, impl, n_failed) == ("label_permutation", "bench_local", 0)
    assert len(vals) == 4 and len(seen) == 4
    assert all(isinstance(e, pd.DataFrame) and e is not s.events for e in seen)
    again = export._event_null("SRPI", fn, s.ts, s.events, 4, 3, s.dt)
    assert again[0] == vals

    def failing(ts, ev):
        raise ValueError("degenerate surrogate")

    vals, _, _, n_failed = export._event_null(
        "RAM", failing, s.ts, s.events, 3, 0, s.dt
    )
    assert n_failed == 3 and np.all(np.isnan(vals))


def test_event_nulls_use_the_installed_nulls_module(monkeypatch):
    """With impact_pipeline.nulls installed, its ComponentNull result is used
    with the declared kind and options; its errors are not masked."""
    import sys
    import types
    from typing import NamedTuple

    class ComponentNull(NamedTuple):
        null_mean: float
        null_sd: float
        samples: np.ndarray
        n_failed: int

    calls = []

    def component_null(fn, ts, events, kind="circular_shift", n=100, seed=0, **kw):
        calls.append((kind, n, seed, kw))
        vals = np.array([float(fn(ts, events)["value"]) + i for i in range(n - 1)])
        return ComponentNull(float(vals.mean()), float(vals.std(ddof=1)), vals, 1)

    fake = types.ModuleType("impact_pipeline.nulls")
    fake.component_null = component_null
    monkeypatch.setitem(sys.modules, "impact_pipeline.nulls", fake)
    monkeypatch.setattr("impact_pipeline.nulls", fake, raising=False)
    s = g.simulate_family_a(None, SMALL, seed=6)
    vals, kind, impl, n_failed = export._event_null(
        "RAM", lambda ts, ev: {"value": 1.0}, s.ts, s.events, 4, 9, s.dt
    )
    assert (kind, impl, n_failed) == ("onset_jitter", "impact_pipeline.nulls", 1)
    assert vals == [1.0, 2.0, 3.0]
    t_max = s.n_time * s.dt
    assert calls[-1] == (
        "onset_jitter",
        4,
        9,
        {"common": True, "t_max": t_max, "min_shift": 0.1 * t_max},
    )
    # Legacy SRPI (self/non-self onsets) uses the event-table label
    # permutation of the installed module.
    res = export.run_in_memory(
        s, metrics=("SRPI",), null_surrogates=4, use_optional_modes=False
    )
    comp = res["components"]["SRPI"]
    assert comp["null_impl"] == "impact_pipeline.nulls"
    assert comp["null_family"] == "label_permutation" and comp["n_null"] == 3
    assert calls[-1][3] == {
        "among": export.SRPI_NULL_LABELS,
        "stratify_col": "phase_bin",
    }
    # SRPI-agency calibrates itself (yoked permutation): the event-table
    # null, which would break the yoking, is not called.
    n_calls = len(calls)
    agency = export.run_in_memory(s, metrics=("SRPI",), null_surrogates=4)
    if agency["estimator_modes"]["SRPI"].get("mode") == "agency":
        assert len(calls) == n_calls
        assert agency["components"]["SRPI"]["null_impl"] == "estimator"
    # RAM (no self-calibrating mode) goes through the installed module.
    ram = export.run_in_memory(s, metrics=("RAM",), null_surrogates=4)["components"]
    if ram["RAM"]["defined"]:
        assert ram["RAM"]["null_impl"] == "impact_pipeline.nulls"
        assert calls[-1][0] == "onset_jitter"

    def broken(*args, **kwargs):
        raise TypeError("API drift")

    fake.component_null = broken
    with pytest.raises(TypeError, match="API drift"):
        export._event_null(
            "RAM", lambda ts, ev: {"value": 1.0}, s.ts, s.events, 2, 0, s.dt
        )


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
    # The code-freeze tag is required.
    with pytest.raises(RuntimeError, match="freeze tag"):
        run_bench.confirmatory_guard(repo)
    (repo / "src" / "new.py").write_text("y = 2\n")
    with pytest.raises(RuntimeError, match="untracked"):
        run_bench.confirmatory_guard(repo, freeze_tag="freeze-v1")
    (repo / "src" / "new.py").unlink()
    (repo / "src" / "a.py").write_text("x = 2\n")
    with pytest.raises(RuntimeError, match="clean"):
        run_bench.confirmatory_guard(repo, freeze_tag="freeze-v1")
    # Later commits may touch other paths but not the frozen code.
    (repo / "src" / "a.py").write_text("x = 1\n")
    (repo / "NOTES.md").write_text("results\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "docs")
    info = run_bench.confirmatory_guard(repo, freeze_tag="freeze-v1")
    assert info["freeze_tag_sha"] != info["git_sha"]
    (repo / "src" / "a.py").write_text("x = 3\n")
    _git(repo, "commit", "-q", "-am", "post-freeze code change")
    with pytest.raises(RuntimeError, match="changed since freeze tag"):
        run_bench.confirmatory_guard(repo, freeze_tag="freeze-v1")
    with pytest.raises(RuntimeError):
        run_bench.confirmatory_guard(tmp_path / "not_a_repo", freeze_tag="freeze-v1")


def test_confirmatory_runs_need_guard_provenance(tmp_path):
    tasks = factorial_tasks([10000], cells=["b11111"], config=SMALL_DICT)
    with pytest.raises(RuntimeError, match="confirmatory_guard"):
        run_bench.run_tasks(tasks, tmp_path / "run", confirmatory=True)
    assert not (tmp_path / "run").exists()
    # The CLI refuses --confirmatory without the freeze tag (exit code 2).
    rc = run_bench.main(
        ["factorial", "--seeds", "10000", "--cells", "b11111", "--confirmatory"]
        + ["--out", str(tmp_path / "cli")]
    )
    assert rc == 2 and not (tmp_path / "cli" / run_bench.RESULTS_JSONL).exists()


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
        "MPC_CONSISTENT",
        "EXCLUDED",
        "UNDETERMINED",
    )
    df = pd.read_csv(out / run_bench.RESULTS_CSV)
    assert list(df["cell_id"]) == ["b01111", "b11111"]
    assert set(df["intended_bits"]) == {"b01111", "b11111"}
    assert {
        "NAS_z",
        "SRPI_estimate",
        "SRPI_null_family",
        "marker_LZc",
        "oracle_q_range",
        "git_sha",
    } <= set(df.columns)
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
    rep = run_bench.timing_report(
        seeds=(0,), config=SMALL_DICT, binary_steps=500, whole_brain_sec=4.0
    )
    rows = rep["timings"]
    assert {"family_A", "family_C", "patchwork", "patchwork_graded_lambda1"} <= set(
        rows
    )
    assert sum(k.startswith("family_B_") for k in rows) == len(g.BINARY_NETWORK_KINDS)
    assert sum(k.startswith("adversarial_") for k in rows) == 6
    assert {
        "whole_brain_source_4s",
        "whole_brain_eeg_forward",
        "whole_brain_bold_forward",
    } <= set(rows)
    assert all(r["mean_s"] > 0 for r in rows.values())
    assert rows["family_A"]["shape"][0] == 30
    assert rows["whole_brain_eeg_forward"]["shape"][0] == 64
