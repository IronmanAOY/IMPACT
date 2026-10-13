# -*- coding: utf-8 -*-
"""Seed policy, the reserved random streams and the confirmatory guard."""
import numpy as np
import pytest

from impact_pipeline.bench import run_hygiene as P
from impact_pipeline.bench import seeds as S


# --------------------------------------------------------------------------
# split of a seed
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 320, 900, 939, 980, 999, np.int64(5)])
def test_development_seeds(seed):
    assert S.split_of(seed) == "development"


@pytest.mark.parametrize("seed", [20000, 20019, 20499, 20900, 20939, 10**9])
def test_confirmatory_seeds(seed):
    assert S.split_of(seed) == "confirmatory"


@pytest.mark.parametrize("seed", [1000, 5000, 9999, 10000, 15000, 19999])
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
    with pytest.raises(S.SeedPolicyError, match="unassigned"):
        S.assert_confirmatory([10000, 10001])
    with pytest.raises(S.SeedPolicyError, match="unassigned"):
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


def test_structural_stream_is_the_seed_sequence_of_the_seed():
    a = np.random.default_rng(S.structural_seed_sequence(12)).random(3)
    b = np.random.default_rng(np.random.SeedSequence(12)).random(3)
    assert np.array_equal(a, b)
    # the generators spawn their streams from SeedSequence(seed); a twin key
    # [seed, r] is a different entropy pool
    child = np.random.SeedSequence(12).spawn(1)[0]
    assert not np.array_equal(
        np.random.default_rng(child).random(3),
        np.random.default_rng(S.twin_seed_sequence(12, 1)).random(3))


# --------------------------------------------------------------------------
# confirmatory guard
# --------------------------------------------------------------------------
TAG = "bench-confirmatory-1"


def _tagged_repo(git_repo, tag=TAG):
    git_repo.write("src/pkg/a.py", "x = 1\n")
    git_repo.write("scripts/run.py", "print('run')\n")
    git_repo.write("docs/notes.md", "notes\n")
    git_repo.commit("initial")
    git_repo.git("tag", tag)
    return git_repo


def test_guard_accepts_a_clean_checkout_of_the_tag(git_repo):
    repo = _tagged_repo(git_repo)
    info = P.confirmatory_guard(range(20000, 20020), repo.root, TAG)
    assert info["confirmatory"] is True
    assert info["freeze_tag"] == TAG
    assert info["trees"]["src"] == info["freeze_tag_trees"]["src"]
    assert info["trees"]["scripts"] == repo.git("rev-parse", "HEAD:scripts")
    # a later commit that touches neither src/ nor scripts/ keeps the trees
    repo.write("docs/notes.md", "more notes\n")
    repo.commit("docs")
    assert P.confirmatory_guard([20001], repo.root, TAG)["freeze_tag_commit"] != repo.git(
        "rev-parse", "HEAD")


def test_guard_refuses_seeds_below_20000(git_repo):
    repo = _tagged_repo(git_repo)
    for seeds in ([0, 1], [10000], [20000, 999]):
        with pytest.raises(P.ConfirmatoryGuardError, match="seed policy"):
            P.confirmatory_guard(seeds, repo.root, TAG)


def test_guard_refuses_a_dirty_tree(git_repo):
    repo = _tagged_repo(git_repo)
    repo.write("src/pkg/a.py", "x = 2\n")
    with pytest.raises(P.ConfirmatoryGuardError, match="tracked files modified"):
        P.confirmatory_guard([20000], repo.root, TAG)


def test_guard_refuses_untracked_code(git_repo):
    repo = _tagged_repo(git_repo)
    repo.write("scripts/extra.py", "print('x')\n")
    with pytest.raises(P.ConfirmatoryGuardError, match="untracked or modified"):
        P.confirmatory_guard([20000], repo.root, TAG)


def test_guard_accepts_an_annotated_tag(git_repo):
    repo = _tagged_repo(git_repo, tag="some-other-tag")
    repo.git("tag", "-a", TAG, "-m", "freeze")
    info = P.confirmatory_guard([20000], repo.root, TAG)
    assert info["freeze_tag_commit"] == repo.git("rev-parse", "HEAD")


def test_guard_refuses_a_branch_named_like_the_tag(git_repo):
    repo = _tagged_repo(git_repo, tag="some-other-tag")
    repo.git("branch", TAG)
    with pytest.raises(P.ConfirmatoryGuardError, match="not found"):
        P.confirmatory_guard([20000], repo.root, TAG)


def test_guard_refuses_a_missing_or_wrong_tag(git_repo):
    repo = _tagged_repo(git_repo, tag="some-other-tag")
    with pytest.raises(P.ConfirmatoryGuardError, match="not found"):
        P.confirmatory_guard([20000], repo.root, TAG)
    repo.git("tag", TAG)
    repo.write("src/pkg/a.py", "x = 3\n")
    repo.commit("code change after the freeze")
    with pytest.raises(P.ConfirmatoryGuardError, match="differ from freeze tag"):
        P.confirmatory_guard([20000], repo.root, TAG)


def test_guard_needs_a_tag(git_repo):
    repo = _tagged_repo(git_repo)
    for tag in (None, ""):
        with pytest.raises(P.ConfirmatoryGuardError, match="needs a freeze tag"):
            P.confirmatory_guard([20000], repo.root, tag)


def test_guard_refuses_a_directory_without_git(tmp_path):
    with pytest.raises(P.ConfirmatoryGuardError, match="git checkout"):
        P.confirmatory_guard([20000], tmp_path, TAG)


def test_run_provenance_derives_the_split(git_repo):
    repo = _tagged_repo(git_repo)
    dev = P.run_provenance(repo.root, seeds=[320, 321])
    assert dev["split"] == "development" and dev["code"]["confirmatory"] is False
    assert set(dev["code"]["trees"]) == {"src", "scripts", "protocols"}
    conf = P.run_provenance(repo.root, seeds=[20000], confirmatory=True,
                            freeze_tag=TAG)
    assert conf["split"] == "confirmatory" and conf["code"]["confirmatory"] is True
    with pytest.raises(S.SeedPolicyError):
        P.run_provenance(repo.root, seeds=[999, 20000])
