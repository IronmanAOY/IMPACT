# -*- coding: utf-8 -*-
"""
The development calibration runner after the calibration decisions (plans
only, nothing is simulated):

* the held-out-regime reference anchors of the forward views still plan
  120 tasks on 900-939 and carry the v1 quadrant comparator on the Hopf
  anchor condition, which HO-6 does not withhold there;
* the development-regime reference anchors never carry it, and IIM on the
  Hopf sensor views above G = 0 stays withheld (HO-6);
* the constants file is strict JSON: a non-finite value is written as null.
"""
import json
import math

import numpy as np
import pytest

from impact_pipeline.bench.designs_v2 import forward as FW
from impact_pipeline.v2 import seeds as S
from scripts.v2 import dev_calibration as DC

QUADRANT_VIEWS = {("eeg64", FW.IIM_V1_QUADRANTS), ("eeg64_noref", FW.IIM_V1_QUADRANTS)}


def _forms(task):
    return {(s.view, s.estimator_form) for s in task.scorings}


def test_the_held_out_reference_anchors_plan_with_the_quadrant_comparator():
    run = DC.get_item("anchors_forward_held_out").runs[0]
    assert run.released and run.purpose == FW.REFERENCE
    p = DC.plan_run(run)
    assert p["n_tasks"] == 3 * 40
    assert p["by_design"] == {FW.ANCHOR_REPLICATION_DESIGN: 120}
    assert "held_out_iim_withheld" not in p and "unpriced_designs" not in p
    # the tasks the run would execute once the release is logged (a release
    # entry needs only its id here)
    withheld = []
    tasks = DC.runner_tasks(run, release={"release_id": "r-test"}, withheld=withheld)
    assert withheld == []
    assert len(tasks) == 120 and {t.seed for t in tasks} == set(S.REFERENCE_BLOCK)
    for t in tasks:
        assert t.tags["held_out_release"] == "r-test" and not t.has_held_out
        if t.tags["arm"] == FW.ARM_HOPF:
            assert t.system == FW.g_nom_label()
            assert _forms(t) >= QUADRANT_VIEWS
            quad = [s for s in t.scorings if s.estimator_form == FW.IIM_V1_QUADRANTS]
            assert all(s.principles == ("IIM",) for s in quad)
            assert {s.protocol_key for s in quad} == {
                "hopf-eeg64+iim_v1_quadrants", "hopf-eeg64_noref+iim_v1_quadrants"}
        else:
            assert not any(f == FW.IIM_V1_QUADRANTS for _v, f in _forms(t))


def test_the_development_reference_anchors_keep_the_ho6_withholding():
    run = DC.get_item("anchors_forward_dev_regime").runs[0]
    assert run.purpose == FW.REFERENCE_DEVELOPMENT
    withheld = []
    tasks = DC.runner_tasks(run, withheld=withheld)
    hopf = [t for t in tasks if t.tags["arm"] == FW.ARM_HOPF]
    assert len(hopf) == 40
    for t in hopf:
        assert not any(f == FW.IIM_V1_QUADRANTS for _v, f in _forms(t))
        for s in t.scorings:
            if s.view in DC.HO6_VIEWS:
                assert "IIM" not in s.principles, (t.task_id, s.scoring_id)
    # IIM of the four sensor views is withheld on every Hopf run, the
    # comparator is not planned there at all
    assert len(withheld) == 40 * len(DC.HO6_VIEWS)
    assert not any(FW.IIM_V1_QUADRANTS in w for w in withheld)


def test_the_constants_are_written_as_strict_json(tmp_path):
    payload = {"a": float("nan"), "b": [1.5, float("inf"), -float("inf")],
               "c": {"d": np.float64("nan"), "e": np.float64(0.25), "f": (2, 3)},
               "g": "text", "h": None, "i": np.float32(np.nan), "j": 7}
    path = tmp_path / DC.CONSTANTS_JSON
    DC.write_constants(path, payload)
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text

    def refuse(name):
        raise ValueError(f"non-standard JSON constant {name}")

    got = json.loads(text, parse_constant=refuse)
    assert got == {"a": None, "b": [1.5, None, None],
                   "c": {"d": None, "e": 0.25, "f": [2, 3]},
                   "g": "text", "h": None, "i": None, "j": 7}
    # finite values are written exactly as before
    finite = {"x": [0.1, 2.0, np.float64(1e-300)], "y": {"z": 3}}
    DC.write_constants(path, finite)
    assert path.read_text(encoding="utf-8") == json.dumps(
        finite, indent=2, sort_keys=True, default=float) + "\n"
    assert DC.strict_json({"k": math.nan}) == {"k": None}


def test_a_non_finite_value_cannot_reach_the_file(tmp_path, monkeypatch):
    monkeypatch.setattr(DC, "strict_json", lambda obj: obj)
    with pytest.raises(ValueError):
        DC.write_constants(tmp_path / "c.json", {"a": float("nan")})


def test_the_oracle_item_runs_on_the_workers_and_records_the_reported_checks(
        tmp_path, monkeypatch):
    calls = []

    def fake(seeds, out, *, families, workers):
        calls.append((list(seeds), tuple(families), workers))
        return {"complete": True, "all_usable": False,
                "not_usable": ["A:ADV_NAS_staggered_tau10:tau10/x"],
                "reported": {"pc_half_between": {"A:K": False}}}

    monkeypatch.setattr(DC.RB, "run_manipulation", fake)
    out = DC.run_item("oracle_checks", root=tmp_path, workers=12, limit=2)
    assert calls == [([320, 321], DC.ORACLE_FAMILIES, 12)]
    assert out["manipulation"]["not_usable"] == ["A:ADV_NAS_staggered_tau10:tau10/x"]
    assert out["manipulation"]["reported"] == {"pc_half_between": {"A:K": False}}
    prov = json.loads((tmp_path / "oracle_checks" / "manipulation"
                       / DC.PROVENANCE_JSON).read_text(encoding="utf-8"))
    assert prov["outcome"]["reported"] == {"pc_half_between": {"A:K": False}}
    item = DC.get_item("oracle_checks")
    assert "slow-context" in item.description and "PC_half" in item.description
    assert DC.plan_run(item.runs[0])["cpu_s"] == 40 * DC.ORACLE_CPU_S_PER_SEED
