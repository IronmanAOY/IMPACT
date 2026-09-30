# -*- coding: utf-8 -*-
"""File ownership of the v2 work plan: every path belongs to exactly one work
package, no read-only v1 file is claimed or changed, and every change on
this branch is owned."""
import json
import os

import pytest

from impact_pipeline.v2 import ownership as own

WP00_FILES = [
    "src/impact_pipeline/v2/__init__.py",
    "src/impact_pipeline/v2/reasons.py",
    "src/impact_pipeline/v2/records.py",
    "src/impact_pipeline/v2/seeds.py",
    "src/impact_pipeline/v2/provenance.py",
    "src/impact_pipeline/v2/ownership.py",
    "scripts/v2/regression_gate.py",
    "scripts/v2/env_lock.py",
    "protocols/v2/README.md",
    "protocols/v2/seed_map_v2.json",
    "tests/v2/__init__.py",
    "tests/v2/conftest.py",
    "tests/v2/test_reasons.py",
    "tests/v2/test_records_schema.py",
    "tests/v2/test_seed_policy_v2.py",
    "tests/v2/test_v1_regression_gate.py",
    "tests/v2/test_ownership.py",
]


def plan(*packages, read_only=()):
    return {
        "schema": own.WORK_PACKAGES_SCHEMA,
        "work_packages": [
            {"id": wid, "owner_files": list(own), "edits_existing": list(edit),
             "depends_on": []}
            for wid, own, edit in packages
        ],
        "read_only_v1_files": list(read_only),
    }


# --------------------------------------------------------------------------
# synthetic plans
# --------------------------------------------------------------------------
def test_a_clean_plan_has_no_conflicts():
    p = plan(("A", ["src/a.py", "gen/"], []), ("B", ["src/b.py"], ["README.md"]),
             ("C", ["README2.md", "src/c.py"], ["src/c.py"]))
    own.validate_work_packages(p)
    assert own.conflicts(p) == []
    assert own.claims(p)["src/c.py"] == {"C"}  # owner and editor at once counts once


def test_a_path_claimed_twice_is_a_conflict():
    p = plan(("A", ["src/a.py"], []), ("B", ["src/b.py"], ["src/a.py"]))
    (c,) = own.conflicts(p)
    assert (c["path"], c["owners"], c["kind"]) == ("src/a.py", ["A", "B"], "same_path")


def test_a_path_inside_another_package_directory_is_a_conflict():
    p = plan(("A", ["protocols/v2/generated/"], []),
             ("B", ["protocols/v2/generated/x.json", "protocols/v2/README.md"], []))
    (c,) = own.conflicts(p)
    assert c["path"] == "protocols/v2/generated/x.json"
    assert c["owners"] == ["A", "B"]
    assert own.owners_of("protocols/v2/generated/y/z.json", p) == {"A"}
    assert own.owners_of("protocols/v2/README.md", p) == {"B"}
    assert own.owners_of("protocols/v2/other.json", p) == set()


def test_read_only_claims_and_changes():
    p = plan(("A", ["src/a.py", "src/v1.py"], []), ("B", ["src/b.py", "out/"], []),
             read_only=["src/v1.py", "docs/v1.md"])
    assert own.read_only_claims(p) == [{"path": "src/v1.py", "owners": ["A"]}]
    changed = ["src/a.py", "src/b.py", "out/x.json", "docs/v1.md", "src/unowned.py"]
    res = own.check_changes(changed, p)
    assert res["read_only"] == ["docs/v1.md"]
    assert res["unowned"] == ["src/unowned.py"]
    assert res["foreign"] == [] and res["ok"] is False
    res = own.check_changes(["src/a.py", "src/b.py"], p, package="A")
    assert res["foreign"] == ["src/b.py"] and not res["ok"]
    assert own.check_changes(["src/a.py", "./src/a.py"], p, package="A")["ok"]
    with pytest.raises(own.OwnershipError):
        own.check_changes([], p, package="Z")


@pytest.mark.parametrize("bad", [
    {"schema": "x", "work_packages": [{"id": "A"}]},
    {"schema": own.WORK_PACKAGES_SCHEMA, "work_packages": []},
    {"schema": own.WORK_PACKAGES_SCHEMA, "work_packages": [{"id": "A"}, {"id": "A"}]},
    {"schema": own.WORK_PACKAGES_SCHEMA,
     "work_packages": [{"id": "A", "owner_files": ["../escape.py"]}]},
    {"schema": own.WORK_PACKAGES_SCHEMA,
     "work_packages": [{"id": "A", "owner_files": ["/abs.py"]}]},
    {"schema": own.WORK_PACKAGES_SCHEMA,
     "work_packages": [{"id": "A", "depends_on": ["B"]}]},
    {"schema": own.WORK_PACKAGES_SCHEMA,
     "work_packages": [{"id": "A", "owner_files": "src/a.py"}]},
])
def test_malformed_plans_are_refused(bad):
    with pytest.raises(own.OwnershipError):
        own.validate_work_packages(bad)


def test_the_plan_is_located_through_the_environment(tmp_path):
    f = tmp_path / "wp.json"
    f.write_text(json.dumps(plan(("A", ["x.py"], []))), encoding="utf-8")
    assert own.locate_work_packages(tmp_path, env={own.WORK_PACKAGES_ENV: str(f)}) == f
    assert own.load_work_packages(f)["work_packages"][0]["id"] == "A"
    missing = {own.WORK_PACKAGES_ENV: str(tmp_path / "no")}
    with pytest.raises(FileNotFoundError):
        own.locate_work_packages(tmp_path, env=missing)
    assert own.locate_work_packages(tmp_path, env={}) is None  # not a git checkout


def test_changed_paths_of_a_branch(git_repo):
    git_repo.write("src/a.py", "a\n")
    git_repo.write(".gitignore", "ignored/\n")
    base = git_repo.commit("base")
    git_repo.write("src/b.py", "b\n")
    git_repo.commit("add b")
    git_repo.write("src/a.py", "a2\n")  # unstaged edit
    git_repo.write("new/c.py", "c\n")  # untracked
    git_repo.write("ignored/d.py", "d\n")  # ignored
    changed = own.changed_paths(git_repo.root, base)
    assert changed == ["new/c.py", "src/a.py", "src/b.py"]
    with pytest.raises(RuntimeError):
        own.changed_paths(git_repo.root, "no-such-revision")


# --------------------------------------------------------------------------
# the real plan
# --------------------------------------------------------------------------
def test_no_path_is_claimed_by_two_work_packages(work_plan):
    assert own.conflicts(work_plan) == []


def test_no_work_package_claims_a_read_only_v1_file(work_plan):
    assert own.read_only_claims(work_plan) == []
    ro = set(work_plan["read_only_v1_files"])
    assert "src/impact_pipeline/evidence.py" in ro
    # the one v1 file that receives additive edits, with a single owner
    generators = "src/impact_pipeline/bench/generators.py"
    assert generators not in ro
    assert own.owners_of(generators, work_plan) == {"WP02"}


def test_this_package_owns_the_skeleton(work_plan, repo_root):
    (wp00,) = [w for w in work_plan["work_packages"] if w["id"] == "WP00"]
    assert sorted(wp00["owner_files"]) == sorted(WP00_FILES)
    assert wp00["edits_existing"] == []
    for rel in WP00_FILES:
        assert (repo_root / rel).is_file(), rel
        assert own.owners_of(rel, work_plan) == {"WP00"}


def test_every_change_on_this_branch_is_owned(work_plan, repo_root):
    """Every path changed since the design's base commit (committed or not)
    is owned by a work package, and no read-only v1 file changed. With
    ``MPCBENCH_WP`` set, every change must belong to that package."""
    base = work_plan.get("base_commit_at_design")
    try:
        paths = own.changed_paths(repo_root, base)
    except RuntimeError as exc:
        pytest.skip(f"cannot diff against the design base: {exc}")
    package = os.environ.get("MPCBENCH_WP") or None
    res = own.check_changes(paths, work_plan, package=package)
    assert res["read_only"] == []
    assert res["unowned"] == []
    assert res["foreign"] == []
