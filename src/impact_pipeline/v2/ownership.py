"""
File ownership of the v2 work plan.

The v2 revision is built as independent work packages, each in its own git
worktree and branch. ``work_packages.json`` (next to the v2 design document)
lists, per package, the paths it owns (``owner_files``; a path ending in
``/`` owns a directory) and the existing files it edits
(``edits_existing``), plus the v1 files nobody may edit
(``read_only_v1_files``). This module checks that

* no path is claimed by two packages (also through a directory claim);
* no package claims a read-only v1 file;
* every path changed since the design's base commit is owned by some
  package (or by one given package) and no read-only v1 file changed.

The plan file lives under ``docs/manuscript/``, which is not versioned, so
:func:`locate_work_packages` also looks in the main checkout of a linked
worktree and honours ``MPCBENCH_V2_WORK_PACKAGES``.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Set

REPO_ROOT = Path(__file__).resolve().parents[3]
WORK_PACKAGES_RELPATH = Path("docs/manuscript/research/v2_design/work_packages.json")
WORK_PACKAGES_ENV = "MPCBENCH_V2_WORK_PACKAGES"
WORK_PACKAGES_SCHEMA = "mpcbench-v2-work-packages/1"


class OwnershipError(ValueError):
    """A malformed work plan."""


def _git(repo_root, *args) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except Exception:  # noqa: BLE001
        return None
    return proc.stdout if proc.returncode == 0 else None


def main_checkout_root(repo_root=REPO_ROOT) -> Optional[Path]:
    """The main working tree of the repository ``repo_root`` belongs to (the
    parent of the common git directory), or None outside git."""
    common = _git(repo_root, "rev-parse", "--path-format=absolute", "--git-common-dir")
    if not common:
        return None
    path = Path(common.strip())
    return path.parent if path.name == ".git" else None


def locate_work_packages(repo_root=REPO_ROOT, env=None) -> Optional[Path]:
    """``$MPCBENCH_V2_WORK_PACKAGES``, else the plan in this checkout, else
    the plan in the main checkout of this worktree; None when absent."""
    env = os.environ if env is None else env
    explicit = str(env.get(WORK_PACKAGES_ENV) or "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"{WORK_PACKAGES_ENV}={explicit!r} is not a file")
        return p
    candidates = [Path(repo_root) / WORK_PACKAGES_RELPATH]
    main = main_checkout_root(repo_root)
    if main is not None:
        candidates.append(main / WORK_PACKAGES_RELPATH)
    for c in candidates:
        if c.is_file():
            return c
    return None


def _norm(path: str) -> str:
    p = str(path).replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    if not p or p.startswith("/") or ".." in p.split("/"):
        raise OwnershipError(f"not a repository-relative path: {path!r}")
    return p


def validate_work_packages(payload: Mapping) -> Mapping:
    """Minimal structural check: schema, unique package ids, lists of
    repository-relative paths, ``depends_on`` naming known packages."""
    if payload.get("schema") != WORK_PACKAGES_SCHEMA:
        raise OwnershipError(f"schema must be {WORK_PACKAGES_SCHEMA!r}")
    wps = payload.get("work_packages")
    if not isinstance(wps, list) or not wps:
        raise OwnershipError("work_packages must be a non-empty list")
    ids = [w.get("id") for w in wps]
    if len(set(ids)) != len(ids) or not all(isinstance(i, str) and i for i in ids):
        raise OwnershipError("work package ids must be unique non-empty strings")
    for w in wps:
        for key in ("owner_files", "edits_existing"):
            vals = w.get(key, [])
            if not isinstance(vals, list):
                raise OwnershipError(f"{w['id']}.{key} must be a list")
            for v in vals:
                _norm(v)
        unknown = set(w.get("depends_on", [])) - set(ids)
        if unknown:
            raise OwnershipError(
                f"{w['id']} depends on unknown packages {sorted(unknown)}")
    for v in payload.get("read_only_v1_files", []):
        _norm(v)
    return payload


def load_work_packages(path=None, repo_root=REPO_ROOT) -> Mapping:
    """The validated work plan (default: :func:`locate_work_packages`)."""
    p = Path(path) if path is not None else locate_work_packages(repo_root)
    if p is None:
        raise FileNotFoundError(
            f"work plan not found ({WORK_PACKAGES_RELPATH}; set {WORK_PACKAGES_ENV})"
        )
    return validate_work_packages(json.loads(Path(p).read_text(encoding="utf-8")))


def claims(plan: Mapping) -> Dict[str, Set[str]]:
    """``{path: {package ids}}`` over ``owner_files`` and ``edits_existing``
    (a package listing a path in both counts once)."""
    out: Dict[str, Set[str]] = {}
    for w in plan["work_packages"]:
        for key in ("owner_files", "edits_existing"):
            for p in w.get(key, []):
                out.setdefault(_norm(p), set()).add(w["id"])
    return out


def _is_dir_claim(path: str) -> bool:
    return path.endswith("/")


def _under(path: str, directory: str) -> bool:
    return path.startswith(directory) and path != directory


def conflicts(plan: Mapping) -> List[dict]:
    """Paths claimed by more than one package: the same path twice, or a
    path inside a directory claimed by another package. Empty when the plan
    assigns every path to exactly one package."""
    table = claims(plan)
    out = []
    for path, owners in sorted(table.items()):
        if len(owners) > 1:
            out.append({"path": path, "owners": sorted(owners), "kind": "same_path"})
    dirs = [(d, o) for d, o in table.items() if _is_dir_claim(d)]
    for path, owners in sorted(table.items()):
        for d, downers in dirs:
            if _under(path, d) and owners != downers:
                out.append({"path": path, "owners": sorted(owners | downers),
                            "kind": f"inside {d}"})
    return out


def read_only_claims(plan: Mapping) -> List[dict]:
    """Read-only v1 files that some package claims (must be empty)."""
    ro = {_norm(p) for p in plan.get("read_only_v1_files", [])}
    return [{"path": p, "owners": sorted(o)} for p, o in sorted(claims(plan).items())
            if p in ro]


def owners_of(path: str, plan: Mapping) -> Set[str]:
    """Packages owning ``path`` (exact claim or a claimed directory above it)."""
    p = _norm(path)
    table = claims(plan)
    found = set(table.get(p, set()))
    for d, owners in table.items():
        if _is_dir_claim(d) and _under(p, d):
            found |= owners
    return found


def changed_paths(repo_root=REPO_ROOT, base: Optional[str] = None) -> List[str]:
    """
    Paths that differ between the merge base of ``base`` and HEAD and the
    working tree (committed, staged and unstaged changes, both sides of
    renames) plus untracked files that are not ignored. ``base`` is a
    revision, usually the plan's ``base_commit_at_design``. Raises
    ``RuntimeError`` when git cannot answer.
    """
    root = Path(repo_root)
    if base is None:
        raise ValueError("changed_paths needs a base revision")
    merge_base = _git(root, "merge-base", base, "HEAD")
    if merge_base is None:
        raise RuntimeError(f"no merge base between {base!r} and HEAD")
    diff = _git(root, "diff", "--name-only", "--no-renames", merge_base.strip())
    untracked = _git(root, "ls-files", "--others", "--exclude-standard")
    if diff is None or untracked is None:
        raise RuntimeError("git diff / ls-files failed")
    paths = {ln.strip() for ln in (diff + "\n" + untracked).splitlines() if ln.strip()}
    return sorted(paths)


def check_changes(paths: Iterable[str], plan: Mapping,
                  package: Optional[str] = None) -> dict:
    """
    Classify changed paths: ``read_only`` (a read-only v1 file changed),
    ``unowned`` (no package owns it) and ``foreign`` (owned, but not by
    ``package`` when one is given). ``ok`` is True iff all three are empty.
    """
    ro = {_norm(p) for p in plan.get("read_only_v1_files", [])}
    ids = {w["id"] for w in plan["work_packages"]}
    if package is not None and package not in ids:
        raise OwnershipError(f"unknown package {package!r}")
    out = {"read_only": [], "unowned": [], "foreign": []}
    for raw in paths:
        p = _norm(raw)
        if p in ro:
            out["read_only"].append(p)
            continue
        owners = owners_of(p, plan)
        if not owners:
            out["unowned"].append(p)
        elif package is not None and package not in owners:
            out["foreign"].append(p)
    out["ok"] = not (out["read_only"] or out["unowned"] or out["foreign"])
    return out


__all__ = [
    "OwnershipError",
    "WORK_PACKAGES_ENV",
    "WORK_PACKAGES_RELPATH",
    "WORK_PACKAGES_SCHEMA",
    "changed_paths",
    "check_changes",
    "claims",
    "conflicts",
    "load_work_packages",
    "locate_work_packages",
    "main_checkout_root",
    "owners_of",
    "read_only_claims",
    "validate_work_packages",
]
