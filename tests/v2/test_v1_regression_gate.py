# -*- coding: utf-8 -*-
"""The v1 regression gate (integrity audit IA-1) and the environment lock.

Fast tests cover the gate's static checks, its comparison helpers and the
re-judging of the stored records (where the stored v1 outputs exist). The
from-scratch re-runs are marked ``slow`` (``MPCBENCH_RUN_SLOW=1``)."""
import json
import math
import os

import pytest

from scripts.v2 import env_lock as EL
from scripts.v2 import regression_gate as G


# --------------------------------------------------------------------------
# static checks
# --------------------------------------------------------------------------
def test_read_only_v1_files_are_unchanged():
    c = G.check_read_only_files()
    assert c["status"] == G.PASS, c
    assert c["details"]["n_files"] == 23


def test_the_read_only_manifest_is_the_plans_list(work_plan):
    assert sorted(G.READ_ONLY_V1_SHA256) == sorted(work_plan["read_only_v1_files"])


def test_a_changed_read_only_file_fails_the_gate(monkeypatch):
    table = dict(G.READ_ONLY_V1_SHA256)
    table["src/impact_pipeline/evidence.py"] = "0" * 64
    table["docs/does_not_exist.md"] = "0" * 64
    monkeypatch.setattr(G, "READ_ONLY_V1_SHA256", table)
    c = G.check_read_only_files()
    assert c["status"] == G.FAIL
    assert c["details"]["changed"] == ["src/impact_pipeline/evidence.py"]
    assert c["details"]["missing"] == ["docs/does_not_exist.md"]


def test_frozen_v1_protocol_hashes_are_unchanged():
    c = G.check_v1_protocols()
    assert c["status"] == G.PASS, c
    rows = {r["file"]: r["protocol_hash"][:8] for r in c["details"]["protocols"]}
    assert rows == {"protocols/mpc_bench_v1.json": "855f6a44",
                    "protocols/mpc_bench_v1_anchored.json": "780581f5",
                    "protocols/mpc_default_v1.json": "383eb310"}
    if c["details"]["freeze_tag_present"]:
        assert all(r["equals_freeze_tag"] for r in c["details"]["protocols"])


def test_a_changed_protocol_hash_fails_the_gate(monkeypatch):
    table = dict(G.V1_PROTOCOLS)
    table["protocols/mpc_default_v1.json"] = "f" * 64
    monkeypatch.setattr(G, "V1_PROTOCOLS", table)
    assert G.check_v1_protocols()["status"] == G.FAIL


def test_v1_version_constants_are_unchanged(monkeypatch):
    assert G.check_v1_constants()["status"] == G.PASS
    table = dict(G.V1_CONSTANTS)
    table["impact_pipeline.bench.BENCH_VERSION"] = "9.9.9"
    monkeypatch.setattr(G, "V1_CONSTANTS", table)
    c = G.check_v1_constants()
    assert c["status"] == G.FAIL
    assert "impact_pipeline.bench.BENCH_VERSION" in c["details"]["changed"]


def test_gate_without_stored_outputs_fails_unless_skipped(monkeypatch):
    monkeypatch.setattr(G, "locate_v1_outputs", lambda explicit=None, env=None: None)

    def names(rep, status):
        return {c["name"] for c in rep["checks"] if c["status"] == status}

    rep = G.run_gate(quick=True, allow_env_mismatch=True)
    assert not rep["ok"]
    assert names(rep, G.FAIL) == set(G.STORED_CHECKS)
    rep = G.run_gate(quick=True, skip_stored=True, allow_env_mismatch=True)
    assert rep["ok"], names(rep, G.FAIL)
    assert names(rep, G.SKIPPED) == set(G.STORED_CHECKS)
    only = ["read_only_files", "stratified_reruns"]
    rep = G.run_gate(skip_stored=True, allow_env_mismatch=True, only=only)
    assert [c["name"] for c in rep["checks"]] == only
    with pytest.raises(ValueError, match="unknown checks"):
        G.run_gate(only=["read_only_file"])
    assert G.main(["--quick", "--allow-env-mismatch"]) == 2  # outputs missing


def test_explicit_outputs_path_must_hold_the_v1_outputs(tmp_path):
    with pytest.raises(FileNotFoundError):
        G.locate_v1_outputs(tmp_path)
    (tmp_path / "c2").mkdir()
    assert G.locate_v1_outputs(tmp_path) == tmp_path
    assert G.locate_v1_outputs(None, env={G.V1_OUTPUTS_ENV: str(tmp_path)}) == tmp_path


# --------------------------------------------------------------------------
# comparison helpers
# --------------------------------------------------------------------------
def test_strip_volatile_removes_wall_times_only():
    rec = {"timing": {"total_s": 1.0}, "components": {"NAS": {
        "seconds": 1.2, "se_seconds": 3.0, "estimate": 0.1}},
        "markers": {"LZc": 0.9, "LZc_seconds": 0.1}, "auditor_wall_s": 5.0,
        "exact_iim": {"value": 0.2, "seconds": 0.01}, "status": "ok"}
    assert G.strip_volatile(rec) == {
        "components": {"NAS": {"estimate": 0.1}}, "markers": {"LZc": 0.9},
        "exact_iim": {"value": 0.2}, "status": "ok"}


def test_diff_is_exact_and_lists_paths():
    a = {"x": 1.0, "y": [1, 2, {"z": None}], "n": float("nan")}
    assert G.diff(a, {"x": 1.0, "y": [1, 2, {"z": None}], "n": float("nan")}) == []
    b = {"x": math.nextafter(1.0, 2.0), "y": [1, 2], "w": 0}
    d = G.diff(a, b)
    assert any(p.startswith("/x:") for p in d)
    assert any(p.startswith("/y: length") for p in d)
    assert any("/n: only in rerun" in p for p in d)
    assert any("/w: only in stored" in p for p in d)
    assert G.diff({"x": 1.0}, {"x": 1.0 + 1e-15}, rtol=1e-12) == []
    assert G.diff({"x": True}, {"x": 1}) != []
    assert G.diff({"x": "1"}, {"x": 1}) != []
    assert G.diff({"x": None}, {"x": 0}) != []


def test_diff_tells_ints_from_floats_unless_told_otherwise():
    # a record written as 1 is not the record written as 1.0
    assert G.diff({"x": 1}, {"x": 1.0}) != []
    assert G.diff({"x": [2, {"y": 3.0}]}, {"x": [2.0, {"y": 3}]}) != []
    assert G.diff({"x": 1}, {"x": 1.0}, strict_types=False) == []
    # tables read from CSV: ints and floats mix, floats carry 16 digits
    assert G.diff({"x": 1}, {"x": 1.0 + 1e-15}, rtol=1e-12, strict_types=False) == []
    assert G.diff({"x": 1}, {"x": 1.001}, rtol=1e-12, strict_types=False) != []
    assert G.diff({"x": True}, {"x": 1}, strict_types=False) != []
    assert G.diff({"x": float("nan")}, {"x": float("nan")}, rtol=1e-12) == []
    assert G.diff({"x": float("inf")}, {"x": float("inf")}) == []
    assert G.diff({"x": float("inf")}, {"x": 1e308}, rtol=1e-12) != []


# --------------------------------------------------------------------------
# environment lock
# --------------------------------------------------------------------------
def test_environment_lock_matches_the_recorded_versions():
    """Exact versions on the locked platform (the machine of the v1 and v2
    runs), minor versions elsewhere (what environment.yml pins)."""
    env = EL.environment_versions()
    assert EL.check_versions(env) == [], EL.check_versions(env)
    if env["platform"] == EL.LOCK["platform"]:
        assert env["python"] == EL.LOCK["python"]
        assert env["packages"] == EL.LOCK["packages"]


def test_the_lock_agrees_with_environment_yml():
    minors = EL.environment_yml_minors()
    assert minors["python"] == ".".join(EL.LOCK["python"].split(".")[:2])
    for name, version in EL.LOCK["packages"].items():
        assert minors[name] == ".".join(version.split(".")[:2]), name


def test_version_mismatches_are_reported():
    locked = EL.LOCK["packages"]
    cur = {"platform": EL.LOCK["platform"], "python": EL.LOCK["python"],
           "packages": dict(locked, numpy="2.2.5")}
    assert EL.check_versions(cur) == [f"numpy 2.2.5 != locked {locked['numpy']}"]
    assert EL.check_versions(cur, strict=False) == []
    cur["packages"]["scipy"] = "1.16.0"
    assert len(EL.check_versions(cur, strict=False)) == 1
    cur = {"platform": "linux-x86_64", "python": "3.10.2",
           "packages": dict(locked, pandas="2.3.0")}
    assert EL.check_versions(cur) == []  # not the locked platform: minors only
    cur["packages"]["numba"] = None
    assert EL.check_versions(cur) == [f"numba None != locked {locked['numba']}"]


def test_v2_code_imports_neither_mne_nor_sklearn(repo_root):
    files = EL.v2_source_files(repo_root)
    rels = {str(f.relative_to(repo_root)) for f in files}
    assert "src/impact_pipeline/v2/reasons.py" in rels
    assert "scripts/v2/regression_gate.py" in rels
    assert EL.forbidden_imports(files) == []


def test_forbidden_imports_are_detected(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text("import numpy\n"
                 "import mne.io\n"
                 "from sklearn.covariance import x\n"
                 "import importlib\n"
                 "m = importlib.import_module('sklearn.linear_model')\n"
                 "from . import mne\n", encoding="utf-8")
    found = [(d["line"], d["module"]) for d in EL.forbidden_imports([f])]
    assert sorted(found) == [(2, "mne.io"), (3, "sklearn.covariance"),
                             (5, "sklearn")]


def test_env_lock_cli(capsys):
    assert EL.main([]) == (0 if not EL.report()["version_mismatches"] else 1)
    assert "forbidden imports: 0" in capsys.readouterr().out


# --------------------------------------------------------------------------
# re-judging of the stored v1 records (needs the stored outputs)
# --------------------------------------------------------------------------
def test_stored_records_rejudged_without_differences(v1_outputs):
    rep = G.run_gate(v1_outputs, quick=True, allow_env_mismatch=True)
    by_name = {c["name"]: c for c in rep["checks"]}
    static = ("read_only_files", "v1_protocols", "v1_constants")
    for name in (*static, *G.STORED_CHECKS):
        assert by_name[name]["status"] == G.PASS, (name, by_name[name]["summary"])
    tot = by_name["rejudge_records"]["details"]["totals"]
    assert tot["with_verdict"] > 4000
    assert tot["status_diffs"] == tot["verdict_diffs"] == tot["other_diffs"] == 0
    for case in by_name["rejudge_evaluator"]["details"]["cases"].values():
        assert case["components_byte_identical"] and case["verdicts_byte_identical"]
        assert case["family_c_protocols_ok"]
    nullcal = by_name["rejudge_null_calibration"]["details"]
    assert nullcal["rows"] == 7200
    assert nullcal["status_rule"] == "v2"  # the stored run's rule, not a default
    assert nullcal["tables_byte_identical"] == {"rates": True, "verdicts": True}


def test_a_changed_null_calibration_status_is_detected(v1_outputs, monkeypatch):
    """The null-calibration re-judging is not vacuous: one flipped
    component status shows up in the row count and in the rebuilt rate
    table."""
    import scripts.null_calibration as NC

    real = NC.classify
    flipped = []

    def classify(principle, comp, protocol, **kw):
        st, reason, impl, a = real(principle, comp, protocol, **kw)
        if not flipped and st == "UNDEFINED" and principle == "NAS":
            flipped.append(principle)
            return "ABSENT", None, impl, a
        return st, reason, impl, a

    monkeypatch.setattr(NC, "classify", classify)
    c = G.check_rejudge_null_calibration(v1_outputs)
    assert flipped and c["status"] == G.FAIL
    assert c["details"]["tables_byte_identical"]["rates"] is False
    assert "tables DIFFER" in c["summary"]
    assert c["summary"].startswith("7200 null-calibration components re-classified: "
                                   "1 status")


def test_a_changed_stored_status_is_detected(v1_outputs):
    """Re-judging is not vacuous: a stored record whose status was altered
    is reported."""
    src = v1_outputs / "c2/bench/A/witnesses/results.jsonl"
    rec = G.load_jsonl(src)[0]
    assert G.diff(G.rejudge_bench_record(rec), rec["verdict"]) == []
    status = rec["verdict"]["component_status"]
    p = next(iter(status))
    status[p] = "ABSENT" if status[p] != "ABSENT" else "PRESENT"
    assert G.diff(G.rejudge_bench_record(rec), rec["verdict"]) != []


def test_stratified_sample_is_deterministic_and_covers_every_stratum(v1_outputs):
    sample = G.stratified_selection(v1_outputs)
    assert sample == G.stratified_selection(v1_outputs)
    assert len(sample) == sum(k for _f, k, _req in G.STRATA) == 29
    assert {f for f, _t in sample} == {f for f, _k, _req in G.STRATA}
    # system-bearer patchworks (about 20 min each) come from the witness
    # strata only; the graded patchworks are sampled with principle bearers
    pw = [t for f, t in sample if "patchwork_sweep" in f]
    assert len(pw) == 2 and all("-principle-" in t for t in pw)
    tasks = [t for _f, t in sample]
    assert len(set(tasks)) == len(tasks)
    assert not {t for _l, _r, t in G.VERIFICATION_BENCH} & set(tasks)
    assert G.BOLD_ERROR_TASK[1] not in tasks
    adv = [t for f, t in sample if f.endswith("adversarial/results.jsonl")]
    kinds = {t.split("-")[1] for t in adv}
    # every adversarial kind except the parity grid, which a verification
    # re-run already covers
    assert len(kinds) == 5 and "parity_grid" not in kinds


def test_verification_jobs_are_the_14_reruns(v1_outputs):
    jobs = G.verification_jobs(v1_outputs)
    kinds = [j["kind"] for j in jobs]
    # three bench tasks, three IIM tasks, two null-calibration replicates
    # with 3 + 5 components
    assert (kinds.count("bench"), kinds.count("iim"), kinds.count("nullcal")) == (
        3, 3, 2)
    applicable = {"surrogate_linear": 3, "ar1": 5}
    assert sum(applicable[j["task"][0]] for j in jobs if j["kind"] == "nullcal") == 8
    for j in jobs:
        if j["kind"] == "bench":
            assert j["stored"]["seed"] >= 10000  # v1 seeds, replayed not reused
            path = G._record_protocol_path(j["stored"])
            assert path.name == "mpc_bench_v1.json"


# --------------------------------------------------------------------------
# from-scratch re-runs (slow)
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_full_regression_gate(v1_outputs):
    workers = int(os.environ.get("MPCBENCH_GATE_WORKERS", "4"))
    rep = G.run_gate(v1_outputs, workers=workers)
    failed = [(c["name"], c["summary"]) for c in rep["checks"] if c["status"] != G.PASS]
    assert rep["ok"] and not failed, failed
    by_name = {c["name"]: c for c in rep["checks"]}
    assert by_name["verification_reruns"]["summary"].startswith("14/14")
    assert by_name["stratified_reruns"]["summary"].startswith("29/29")
    assert "ValueError: compute_NAS invalid band" in by_name["bold_error"]["summary"]


def test_gate_cli_quick(v1_outputs, tmp_path, capsys):
    report = tmp_path / "gate.json"
    rc = G.main(["--quick", "--allow-env-mismatch", "--report", str(report),
                 "--v1-outputs", str(v1_outputs)])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "regression gate: PASS" in out
    rep = json.loads(report.read_text(encoding="utf-8"))
    assert rep["ok"] and rep["version"] == G.GATE_VERSION
    assert [c["name"] for c in rep["checks"]] == [
        "environment", "read_only_files", "v1_protocols", "v1_constants",
        *G.STORED_CHECKS]
