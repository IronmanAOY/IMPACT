"""The shipped protocols (protocols/*.json): they load, their hashes match the
README, their declarations match the code that uses them, and the default
protocol runs through compute_synergy_ci."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from impact_pipeline import evidence as E
from impact_pipeline import mpc_metrics as mm

REPO = Path(__file__).resolve().parents[1]
PROTOCOLS = REPO / "protocols"
DEFAULT = PROTOCOLS / "mpc_default_v1.json"
# opt-in (1.1.0 post-freeze): RAM declared on its behavioural channel only
BEHAVIOURAL_RAM = PROTOCOLS / "mpc_behavioural_ram_v1.json"
BENCH = PROTOCOLS / "mpc_bench_v1.json"
ANCHORED = PROTOCOLS / "mpc_bench_v1_anchored.json"


def test_protocols_load_and_their_hashes_are_documented():
    readme = (PROTOCOLS / "README.md").read_text(encoding="utf-8")
    for path in (DEFAULT, BEHAVIOURAL_RAM, BENCH, ANCHORED):
        proto = E.Protocol.from_json(path)
        # canonical: the file is the protocol's own serialisation
        assert json.loads(path.read_text()) == proto.to_dict()
        row = re.search(rf"^\| `{path.name}` \|.*\| `([0-9a-f]{{64}})` \|$", readme,
                        flags=re.M)
        assert row, path.name
        assert row.group(1) == proto.hash, path.name
    summary = json.loads((PROTOCOLS / "mpc_bench_v1_reference_summary.json")
                         .read_text())
    assert summary["protocol_hash"] == E.Protocol.from_json(BENCH).hash


def test_default_protocol_declarations():
    p = E.Protocol.from_json(DEFAULT)
    assert p.necessity_set == E.PRINCIPLES
    assert p.reference == {"kind": "cohort_high_state", "session": "awake"}
    assert p.source_rule == "single_source" and p.alpha == 0.05
    assert all(p.cutoff_for(q) == (0.25, 0.10) for q in E.PRINCIPLES)
    est = {q: p.estimator_options(q) for q in E.PRINCIPLES}
    assert est["RAM"]["update"] == "prediction_error"
    assert est["RAM"]["update_fallback"] == "feedback_magnitude"
    assert est["RAM"]["require_explicit_feedback"] is True
    assert est["PDI"]["mode"] == "repertoire"
    assert est["NAS"]["mode"] == "capacity"
    assert est["IIM"] == {"cut_mode": "bidirectional",
                          "tpm_estimator": "node_shrinkage"}
    assert est["SRPI"] == {"mode": "agency", "mode_fallback": "legacy"}
    # SRPI's null family depends on the mode each run uses: not declared
    assert "SRPI" not in p.null_families
    assert p.null_families["PDI"] == "circular_shift"
    assert p.null_families["NAS"] == "block_circular_shift"
    # declared RAM channels include the ones without an estimator
    unimplemented = set(mm.RAM_IMPACT_CHANNELS) - set(mm.RAM_IMPLEMENTED_CHANNELS)
    assert unimplemented <= set(p.channels_for("RAM"))


def test_ram_is_never_absent_under_the_default_protocol():
    """RAM's unimplemented declared channels are UNDEFINED, so a credibly
    absent behavioural channel alone can never exclude."""
    p = E.Protocol.from_json(DEFAULT)
    ev = {q: [E.ComponentEvidence(q, 1.0, 0.0, 0.1, se=0.05, n_null=30,
                                  reference=1.0, reference_scale="excess",
                                  null_family=p.null_families.get(q))]
          for q in E.PRINCIPLES}
    ev["RAM"] = [
        E.ComponentEvidence("RAM", -0.5, 0.0, 0.1, se=0.05, n_null=30,
                            reference=1.0, reference_scale="excess",
                            null_family="onset_jitter"),
        *[E.ComponentEvidence("RAM", float("nan"), channel=ch, defined=False,
                              reason="NOT_IMPLEMENTED")
          for ch in ("perturbational", "endogenous")],
    ]
    v = E.mpc_verdict(ev, p)
    assert v.channels["RAM"]["default"] == E.ComponentStatus.ABSENT
    assert v.component_status["RAM"] == E.ComponentStatus.UNDEFINED
    assert v.verdict == E.Verdict.UNDETERMINED
    assert "NOT_IMPLEMENTED:RAM:perturbational" in v.reasons
    # a protocol declaring only the behavioural (default) channel would exclude
    only_default = p.to_dict()
    only_default["channels"]["RAM"] = ["default"]
    v = E.mpc_verdict(ev, E.Protocol.from_dict(only_default))
    assert v.verdict == E.Verdict.EXCLUDED and "ABSENT:RAM" in v.reasons


def test_bench_protocol_matches_the_bench_runner():
    from impact_pipeline.bench import export
    from impact_pipeline.bench.reference import check_modes

    p = E.Protocol.from_json(BENCH)
    assert p.name == "mpc-bench-v1" and p.necessity_set == E.PRINCIPLES
    est = {q: p.estimator_options(q) for q in p.estimators}
    assert est["IIM"] == {
        "cut_mode": "bidirectional", "tpm_estimator": "node_shrinkage"}
    assert {q: v for q, v in est.items() if q != "IIM"} == {
        q: dict(v) for q, v in export.OPTIONAL_MODES.items()}
    check_modes([{"estimator_modes": export.OPTIONAL_MODES}], p.to_dict()["estimators"])
    # the declared options are what the bench computes (IIM options pass through)
    assert export.protocol_params(p)["IIM"] == est["IIM"]
    assert p.null_families["RAM"] == export.EVENT_NULL_KINDS["RAM"]
    assert p.null_families["SRPI"] == "yoked_label_permutation"
    assert p.null_families["NAS"] == "block_circular_shift"
    assert all(p.cutoff_for(q) == (0.25, 0.10) for q in E.PRINCIPLES)
    assert p.alpha == 0.05
    ref = p.reference
    assert ref["kind"] == "external" and ref["scale"] == "excess"
    # anchor rule: RAM and PDI are not credibly above their nulls on the
    # development positive control, so they have no anchor
    assert set(ref["values"]) == {"NAS", "IIM", "SRPI"}
    assert "development seeds 900-919 (n=20)" in ref["source"]
    summary = json.loads((PROTOCOLS / "mpc_bench_v1_reference_summary.json")
                         .read_text())
    assert summary["no_anchor"] == ["RAM", "PDI"]
    for q, v in ref["values"].items():
        assert summary["per_principle"][q]["lower_bound"] > 0
        assert v == pytest.approx(summary["per_principle"][q]["mean"])


def test_anchored_bench_protocol_differs_only_in_the_necessity_set():
    from impact_pipeline.bench.analysis import same_except_reference

    full = E.Protocol.from_json(BENCH)
    anch = E.Protocol.from_json(ANCHORED)
    assert anch.necessity_set == tuple(sorted(
        full.reference["values"], key=E.PRINCIPLES.index))
    assert anch.name == "mpc-bench-v1-anchored"
    assert same_except_reference(full, anch.replace(necessity_set=full.necessity_set))
    assert anch.reference == full.reference


def test_default_protocol_runs_through_the_pipeline(tmp_path):
    from test_verdict_wiring import _layout, _run

    from impact_pipeline.run_synergy_ci import RAM_PARAM_PRESETS

    prep, onsets = _layout(tmp_path, with_events=True)
    ram = {**RAM_PARAM_PRESETS["eeg"], "quality_null_samples": 5}
    df = _run(prep, sessions=("awake",), stimulus_onsets=onsets, ram_params=ram,
              nas_params=None, protocol=str(DEFAULT), compute_ci=False,
              mpc_metrics=("RAM", "PDI", "SRPI"))
    row = df.iloc[0]
    assert row["MPC_protocol_hash"] == E.Protocol.from_json(DEFAULT).hash
    assert row["RAM_estimator"].startswith("compute_RAM:feedback_magnitude@")
    assert row["RAM_mode_reason"] == (
        "no_choice_reward_log:prediction_error->feedback_magnitude")
    assert row["SRPI_estimator"].startswith("compute_SRPI:legacy@")
    assert row["SRPI_mode_reason"] == "no_agency_events:agency->legacy"
    assert row["PDI_estimator"].startswith("compute_PDI:repertoire@")
    assert "NOT_IMPLEMENTED:RAM:perturbational" in row["MPC_reason"]
    assert row["RAM_status"] != "ABSENT"
    assert np.isfinite(row["PDI_estimate"])
    with pytest.raises(ValueError):
        E.Protocol.from_dict({**json.loads(DEFAULT.read_text()), "unknown": 1})
