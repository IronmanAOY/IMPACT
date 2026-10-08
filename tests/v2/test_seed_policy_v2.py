# -*- coding: utf-8 -*-
"""Seed policy v2, the seed map, the reserved random streams and the v2
confirmatory guard."""
import copy
import json

import numpy as np
import pytest

from impact_pipeline.v2 import provenance as P
from impact_pipeline.v2 import seeds as S


# --------------------------------------------------------------------------
# split of a seed
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 320, 900, 939, 980, 999, np.int64(5)])
def test_development_seeds(seed):
    assert S.split_of(seed) == "development"


@pytest.mark.parametrize("seed", [20000, 20019, 20499, 20900, 20939, 10**9])
def test_confirmatory_seeds(seed):
    assert S.split_of(seed) == "confirmatory"


@pytest.mark.parametrize("seed", [10000, 10005, 15000, 19000, 19019, 19999])
def test_the_v1_block_is_never_reused(seed):
    with pytest.raises(S.SeedPolicyError, match="v1 confirmatory block"):
        S.split_of(seed)


@pytest.mark.parametrize("seed", [1000, 5000, 9999])
def test_unassigned_seeds_are_refused(seed):
    with pytest.raises(S.SeedPolicyError, match="unassigned"):
        S.split_of(seed)


@pytest.mark.parametrize("seed", [-1, True, False, 1.0, "3", None])
def test_non_seeds_are_refused(seed):
    with pytest.raises(S.SeedPolicyError):
        S.split_of(seed)


# --------------------------------------------------------------------------
# runs
# --------------------------------------------------------------------------
def test_one_split_per_run():
    assert S.check_seeds(range(320, 332)) == "development"
    assert S.check_seeds(range(20000, 20020)) == "confirmatory"
    with pytest.raises(S.SeedPolicyError, match="mixed"):
        S.check_seeds([999, 20000])
    with pytest.raises(S.SeedPolicyError, match="mixed"):
        S.check_seeds([20000, 20001, 5])
    with pytest.raises(S.SeedPolicyError, match="no seeds"):
        S.check_seeds([])


def test_confirmatory_runs_refuse_seeds_below_20000():
    S.assert_confirmatory([20000, 20044])
    with pytest.raises(S.SeedPolicyError, match="confirmatory run uses seeds >= 20000"):
        S.assert_confirmatory([0, 1, 2])
    with pytest.raises(S.SeedPolicyError, match="v1 confirmatory block"):
        S.assert_confirmatory([10000, 10001])
    with pytest.raises(S.SeedPolicyError, match="v1 confirmatory block"):
        S.assert_confirmatory([20000, 19999])
    with pytest.raises(S.SeedPolicyError):
        S.check_seeds([20000], split="held_out")


def test_development_runs_refuse_confirmatory_seeds():
    S.assert_development([0, 999])
    with pytest.raises(S.SeedPolicyError, match="development run uses seeds 0-999"):
        S.assert_development([20000])


def test_smoke_seeds():
    assert [s for s in range(1000) if S.is_smoke_seed(s)] == [980, 981, 982, 983, 984]
    assert list(S.REFERENCE_BLOCK) == list(range(900, 940))


# --------------------------------------------------------------------------
# random streams
# --------------------------------------------------------------------------
def test_reserved_stream_keys():
    assert S.STREAM_KEYS == {
        "label_error": 41, "cue_jitter": 42, "staggered_driver": 43}
    for name, key in S.STREAM_KEYS.items():
        ss = S.stream_seed_sequence(321, name)
        assert ss.entropy == [321, key]
        a = np.random.default_rng(ss).random(4)
        b = np.random.default_rng(np.random.SeedSequence([321, key])).random(4)
        assert np.array_equal(a, b)
    with pytest.raises(S.SeedPolicyError):
        S.stream_seed_sequence(321, "noise")


def test_twin_streams_never_collide_with_reserved_keys():
    assert S.twin_seed_sequence(20003, 1).entropy == [20003, 1]
    assert S.twin_seed_sequence(20003, 30).entropy == [20003, 30]
    # 31 and 32 are named by the benchmark design for its substrate nulls
    for r in (0, 31, 32, 40, 41, 42, 43, -1, True):
        with pytest.raises(S.SeedPolicyError):
            S.twin_seed_sequence(20003, r)
    twins = {tuple(S.twin_seed_sequence(7, r).entropy)
             for r in range(S.TWIN_REPLICATE_MIN, S.TWIN_REPLICATE_MAX + 1)}
    reserved = {tuple(S.stream_seed_sequence(7, n).entropy) for n in S.STREAM_KEYS}
    assert not twins & reserved
    assert S.TWIN_REPLICATE_MAX < min(S.STREAM_KEYS.values())
    assert not twins & {(7, 31), (7, 32)}


def test_structural_stream_is_the_v1_stream():
    a = np.random.default_rng(S.structural_seed_sequence(12)).random(3)
    b = np.random.default_rng(np.random.SeedSequence(12)).random(3)
    assert np.array_equal(a, b)
    # the v1 generators spawn their streams from SeedSequence(seed); a v2 key
    # [seed, r] is a different entropy pool
    child = np.random.SeedSequence(12).spawn(1)[0]
    assert not np.array_equal(
        np.random.default_rng(child).random(3),
        np.random.default_rng(S.twin_seed_sequence(12, 1)).random(3))


# --------------------------------------------------------------------------
# seed map
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def seed_map():
    return S.load_seed_map()


def test_seed_map_matches_the_design(seed_map):
    dev = {(e["min"], e["max"]): e for e in seed_map["development"]["assignments"]}
    assert set(dev) == {(900, 939), (320, 399), (400, 439), (804, 819), (820, 824),
                        (850, 899), (940, 979), (980, 984)}
    assert dev[(980, 984)]["outputs_discarded_unread"] is True
    assert dev[(900, 939)]["reuse_allowed"] is True
    parts = [(p["min"], p["max"]) for p in dev[(320, 399)]["parts"]]
    assert parts == [(320, 331), (332, 335), (336, 339), (340, 351), (352, 371),
                     (372, 383), (384, 399)]
    used = [(b["min"], b["max"]) for b in seed_map["development"]["used_before_v2"]]
    assert used == [(0, 39), (100, 119), (200, 279), (300, 319), (440, 449), (500, 803),
                    (900, 919), (985, 998)]
    conf = {(e["min"], e["max"]): e for e in seed_map["confirmatory"]["assignments"]}
    fam_b = [(p["min"], p["max"], p["use"]) for p in conf[(20000, 20499)]["parts"]]
    assert fam_b == [(20000, 20199, "HCv2-2"), (20240, 20279, "HCv2-12(a)"),
                     (20280, 20319, "HCv2-11"), (20320, 20419, "HCv2-13"),
                     (20420, 20459, "HCv2-12(b)"), (20460, 20499, "HCv2-12(c, d)")]
    assert conf[(20900, 20919)]["extendable_to"] == 20939
    # CD-11: the family-A anchor replication block is extended
    assert conf[(20900, 20919)]["extended"] == [
        {"design": "A_anchors", "max": 20939,
         "decided": "CD-11: A-H IIM replication power 0.565 at 20 seeds"}]
    for key in [(20000, 20039), (20000, 20019), (20000, 20044), (20000, 20045),
                (20000, 20009), (20000, 20060), (20000, 20149)]:
        assert key in conf
    # the HCv2-1 resize: the two family-A null witnesses on 46 seeds
    assert "N_independent_noise and N_ar1" in conf[(20000, 20045)]["use"]
    twins = {(e["min"], e["max"]): e for e in seed_map["development"]["assignments"]}
    assert twins[(820, 824)]["use"].endswith("r = 0..6 (RAM-only r = 0..7)")
    assert seed_map["streams"]["keys"] == S.STREAM_KEYS
    assert seed_map["freeze_tag"] == "mpcbench-freeze-v2"


def test_seed_uses(seed_map):
    assert any("reference blocks" in u for u in S.seed_uses(905, seed_map))
    assert S.seed_uses(325, seed_map)[-1].endswith("witnesses")
    assert S.seed_uses(985, seed_map) == []  # used before v2, not reassigned
    assert "family B: HCv2-2" in S.seed_uses(20100, seed_map)
    assert any("anchor replication" in u for u in S.seed_uses(20930, seed_map))
    with pytest.raises(S.SeedPolicyError):
        S.seed_uses(10003, seed_map)


def _mutated(seed_map, fn):
    m = copy.deepcopy(seed_map)
    fn(m)
    return m


def _family_b(m):
    (fam_b,) = [e for e in m["confirmatory"]["assignments"] if e["use"] == "family B"]
    return fam_b


@pytest.mark.parametrize("mutate, match", [
    (lambda m: m["development"]["assignments"].append(
        {"min": 990, "max": 1005, "use": "x"}), "not a development block"),
    (lambda m: m["development"]["assignments"].append(
        {"min": 10, "max": 12, "use": "x"}), "reuses seeds used before v2"),
    (lambda m: m["development"]["assignments"].append(
        {"min": 400, "max": 401, "use": "x"}), "overlap"),
    (lambda m: m["confirmatory"]["assignments"].append(
        {"min": 19990, "max": 20010, "use": "x"}), "not a confirmatory block"),
    (lambda m: _family_b(m)["parts"].append(
        {"min": 20100, "max": 20120, "use": "x"}), "overlap"),
    (lambda m: _family_b(m)["parts"].append(
        {"min": 20490, "max": 20510, "use": "x"}), "outside its block"),
    (lambda m: m["streams"]["keys"].update(label_error=3), "stream keys"),
    (lambda m: m["streams"]["twin"].update(r_max=45), "twin replicate range"),
    (lambda m: m["streams"]["twin"].update(r_max=40), "twin replicate range"),
    (lambda m: m["development"]["used_before_v2"].append(
        {"min": 10000, "max": 10019}), "used_before_v2"),
    (lambda m: m["policy"].update(never_reused=[]), "never_reused"),
    (lambda m: m["policy"].update(confirmatory={"min": 10000}), "policy.confirmatory"),
    (lambda m: m["policy"].update(one_split_per_run=False), "one_split_per_run"),
    (lambda m: m.update(schema="x"), "schema"),
    (lambda m: m["development"]["assignments"][-1].pop("outputs_discarded_unread"),
     "smoke block"),
    (lambda m: m["confirmatory"]["assignments"][-1].update(extendable_to=20000),
     "extendable_to"),
])
def test_seed_map_validation_refuses_violations(seed_map, mutate, match):
    with pytest.raises(S.SeedPolicyError, match=match):
        S.validate_seed_map(_mutated(seed_map, mutate))


def test_seed_map_file_is_plain_json(repo_root):
    path = repo_root / "protocols" / "v2" / "seed_map_v2.json"
    text = path.read_text(encoding="utf-8")
    assert json.loads(text)["schema"] == S.SEED_MAP_SCHEMA


# --------------------------------------------------------------------------
# v2 confirmatory guard
# --------------------------------------------------------------------------
def _frozen_repo(git_repo, tag="mpcbench-freeze-v2"):
    git_repo.write("src/pkg/a.py", "x = 1\n")
    git_repo.write("scripts/run.py", "print('run')\n")
    git_repo.write("docs/notes.md", "notes\n")
    git_repo.commit("initial")
    git_repo.git("tag", tag)
    return git_repo


def test_guard_accepts_a_clean_checkout_of_the_tag(git_repo):
    repo = _frozen_repo(git_repo)
    info = P.confirmatory_guard(range(20000, 20020), repo.root)
    assert info["confirmatory"] is True
    assert info["freeze_tag"] == "mpcbench-freeze-v2"
    assert info["trees"]["src"] == info["freeze_tag_trees"]["src"]
    assert info["trees"]["scripts"] == repo.git("rev-parse", "HEAD:scripts")
    # a later commit that touches neither src/ nor scripts/ keeps the trees
    repo.write("docs/notes.md", "more notes\n")
    repo.commit("docs")
    assert P.confirmatory_guard([20001], repo.root)["freeze_tag_commit"] != repo.git(
        "rev-parse", "HEAD")


def test_guard_refuses_seeds_below_20000(git_repo):
    repo = _frozen_repo(git_repo)
    for seeds in ([0, 1], [10000], [20000, 999]):
        with pytest.raises(P.ConfirmatoryGuardError, match="seed policy"):
            P.confirmatory_guard(seeds, repo.root)


def test_guard_refuses_a_dirty_tree(git_repo):
    repo = _frozen_repo(git_repo)
    repo.write("src/pkg/a.py", "x = 2\n")
    with pytest.raises(P.ConfirmatoryGuardError, match="tracked files modified"):
        P.confirmatory_guard([20000], repo.root)


def test_guard_refuses_untracked_code(git_repo):
    repo = _frozen_repo(git_repo)
    repo.write("scripts/extra.py", "print('x')\n")
    with pytest.raises(P.ConfirmatoryGuardError, match="untracked or modified"):
        P.confirmatory_guard([20000], repo.root)


def test_guard_accepts_an_annotated_tag(git_repo):
    repo = _frozen_repo(git_repo, tag="some-other-tag")
    repo.git("tag", "-a", "mpcbench-freeze-v2", "-m", "freeze")
    info = P.confirmatory_guard([20000], repo.root)
    assert info["freeze_tag_commit"] == repo.git("rev-parse", "HEAD")


def test_guard_refuses_a_branch_named_like_the_tag(git_repo):
    repo = _frozen_repo(git_repo, tag="some-other-tag")
    repo.git("branch", "mpcbench-freeze-v2")
    with pytest.raises(P.ConfirmatoryGuardError, match="not found"):
        P.confirmatory_guard([20000], repo.root)


def test_guard_refuses_a_missing_or_wrong_tag(git_repo):
    repo = _frozen_repo(git_repo, tag="some-other-tag")
    with pytest.raises(P.ConfirmatoryGuardError, match="not found"):
        P.confirmatory_guard([20000], repo.root)
    repo.git("tag", "mpcbench-freeze-v2")
    repo.write("src/pkg/a.py", "x = 3\n")
    repo.commit("code change after the freeze")
    with pytest.raises(P.ConfirmatoryGuardError, match="differ from freeze tag"):
        P.confirmatory_guard([20000], repo.root)


def test_guard_refuses_a_directory_without_git(tmp_path):
    with pytest.raises(P.ConfirmatoryGuardError, match="git checkout"):
        P.confirmatory_guard([20000], tmp_path)


def test_run_provenance_derives_the_split(git_repo):
    repo = _frozen_repo(git_repo)
    dev = P.run_provenance(repo.root, seeds=[320, 321])
    assert dev["split"] == "development" and dev["code"]["confirmatory"] is False
    assert set(dev["code"]["trees"]) == {"src", "scripts", "protocols"}
    conf = P.run_provenance(repo.root, seeds=[20000], confirmatory=True)
    assert conf["split"] == "confirmatory" and conf["code"]["confirmatory"] is True
    with pytest.raises(S.SeedPolicyError):
        P.run_provenance(repo.root, seeds=[999, 20000])
