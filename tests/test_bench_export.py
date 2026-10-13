"""MPC-Bench export layout (discovered by synergy_ci and the events resolver
exactly like preprocessed data) and the in-memory runner."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from impact_pipeline.bench import export
from impact_pipeline.bench import generators as g
from impact_pipeline.bench.patchwork import simulate_patchwork
from impact_pipeline.event_parsing import read_events_table, resolve_events_file
from impact_pipeline.synergy_ci import build_ci_run_specs, discover_ci_subjects

SMALL = g.AgentConfig(n_trials=16, n_reafference_pairs=6)
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
    assert modes["SRPI"]["mode"] == "agency"
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


# ---------------------------------------------------------------------------
# in-memory replicates and jackknife
# ---------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# jackknife SEs
# --------------------------------------------------------------------------
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
