"""The shipped applicability registry (protocols/applicability_registry_v1.json):
it loads under the strict entry criteria, every validated entry carries the
evidence of the preregistered criteria, excluded combinations and empirical
substrates are refused, and the file is documented in protocols/README.md."""

import hashlib
import json
import re
from pathlib import Path

import pytest

from impact_pipeline import evidence as E
from impact_pipeline.mpc_metrics import ESTIMATOR_VERSIONS

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "protocols" / "applicability_registry_v1.json"


@pytest.fixture(scope="module")
def payload():
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def registry():
    return E.ApplicabilityRegistry.from_json(REGISTRY, strict=True)


def _query(entry):
    reg = entry["regime"]
    q = {"n_time": reg["T_min"], "n_nodes": reg["nodes_min"],
         "tr": reg["fs_or_tr"]["min"]}
    if "bins" in reg:
        q["bins"] = reg["bins"]
    return q


def _grain(entry):
    g = entry["grain"]
    return None if g == "*" else g


def test_loads_strictly_and_only_validated_entries(payload, registry):
    assert payload["schema"] == E.REGISTRY_SCHEMA
    assert registry.entries, "the registry has no validated entry"
    assert all(e.status == "validated" and e.verified for e in registry.entries)
    assert all(e["status"] == "validated" for e in payload["entries"])
    # entry criteria as preregistered: one-sided alpha + 0.02, positive slope,
    # cross-talk ratio below 1
    assert payload["criteria"] == {"alpha": 0.05, "false_present_tolerance": 0.02,
                                   "two_sided": False, "min_recovery_slope": 0.0,
                                   "cross_talk_max": 1.0}


def test_every_entry_meets_the_preregistered_criteria(payload):
    tol = payload["criteria"]["alpha"] + payload["criteria"]["false_present_tolerance"]
    for e in payload["entries"]:
        ev = e["evidence"]
        assert e["version"] == ESTIMATOR_VERSIONS[e["principle"]]
        assert e["substrate"] in E.REGISTRY_SUBSTRATES
        assert e["substrate"] not in E.EMPIRICAL_SUBSTRATES
        assert ev["run_id"].startswith("mpcbench-freeze-v1:confirmatory")
        assert ev["anchor"]["one_sided_lower_bound"] > 0          # (0)
        assert ev["null_false_present_rate"] <= tol               # (a)
        assert ev["recovery_spearman_rho"] > 0                    # (b1)
        assert ev["recovery_spearman_p"] < payload["criteria"]["alpha"]
        assert ev["recovery_slope"] > 0
        assert ev["off_mechanism_present"]["rate"] <= tol         # (b2)
        assert 0 <= ev["cross_talk"] < 1                          # (c)
        own = [c for c in ev["witness_contrasts"] if c["own"]]
        other = [abs(c["median_delta_c"]) for c in ev["witness_contrasts"]
                 if not c["own"] and not c["declared_dependency"]]
        assert len(own) == 1 and abs(own[0]["median_delta_c"]) > max(other)


def test_entries_match_their_own_regime_only(payload, registry):
    for e in payload["entries"]:
        est = f"{e['estimator']}@{e['version']}"
        q = _query(e)
        assert registry.is_validated(e["principle"], est, e["substrate"],
                                     _grain(e), **q) == (True, None)
        # another version, a shorter series or another size is not covered
        ok, why = registry.is_validated(e["principle"], f"{e['estimator']}@other",
                                        e["substrate"], _grain(e), **q)
        assert not ok and why == "NO_ENTRY"
        ok, why = registry.is_validated(e["principle"], est, e["substrate"],
                                        _grain(e), **{**q, "n_time": 1000})
        assert not ok and why.startswith("REGIME_MISMATCH")
        ok, why = registry.is_validated(e["principle"], est, e["substrate"],
                                        _grain(e), **{**q, "n_nodes": 76})
        assert not ok and why.startswith("REGIME_MISMATCH")


def test_excluded_combinations_are_documented_and_refused(payload, registry):
    assert payload["excluded"]
    validated = {(e["principle"], e["estimator"], e["substrate"])
                 for e in payload["entries"]}
    for x in payload["excluded"]:
        assert x["failed_criteria"], x
        assert (x["principle"], x["estimator"], x["substrate"]) not in validated
        ok, _ = registry.is_validated(x["principle"],
                                      f"{x['estimator']}@{x['version']}",
                                      x["substrate"], None, n_time=12000,
                                      n_nodes=30, tr=0.05, bins=2)
        assert not ok


@pytest.mark.parametrize("substrate", ["eeg", "fmri"])
def test_no_empirical_substrate_is_covered(payload, registry, substrate):
    """Human EEG/fMRI need forward-model validation; v1 has none, so every
    empirical component is ESTIMATOR_NOT_VALIDATED under this registry."""
    for e in payload["entries"]:
        ok, why = registry.is_validated(e["principle"],
                                        f"{e['estimator']}@{e['version']}",
                                        substrate, _grain(e), **_query(e))
        assert not ok and why == "NO_ENTRY"


def test_the_verdict_layer_refuses_unregistered_evidence(payload, registry):
    e = payload["entries"][0]
    item = E.ComponentEvidence(
        e["principle"], 1.0, 0.0, 0.1, se=0.05, n_null=30, reference=1.0,
        reference_scale="excess", substrate="eeg",
        estimator=f"{e['estimator']}@{e['version']}",
    )
    v = E.mpc_verdict({e["principle"]: [item]}, necessity_set=(e["principle"],),
                      registry=registry)
    assert v.component_status[e["principle"]] == E.ComponentStatus.UNDEFINED
    assert any("ESTIMATOR_NOT_VALIDATED" in r for r in v.reasons)


def test_no_local_paths_and_documented_hash(payload):
    text = REGISTRY.read_text(encoding="utf-8")
    assert not re.search(r"/(Users|Volumes|private|home)/", text)
    readme = (REPO / "protocols" / "README.md").read_text(encoding="utf-8")
    pattern = r"^\| `applicability_registry_v1\.json` \|.*\| `([0-9a-f]{64})` \|$"
    row = re.search(pattern, readme, flags=re.M)
    assert row, "applicability_registry_v1.json is not listed in protocols/README.md"
    assert row.group(1) == hashlib.sha256(REGISTRY.read_bytes()).hexdigest()
