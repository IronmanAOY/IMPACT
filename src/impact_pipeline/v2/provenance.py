"""
Run hygiene of MPC-Bench v2.

* **Code identity by tree.** :func:`code_identity` records the commit, the
  dirty state and the git tree SHAs of ``src/``, ``scripts/`` and
  ``protocols/``. A history rewrite changes commit SHAs but not trees (as
  happened in v1), so the v2 confirmatory guard (:func:`confirmatory_guard`)
  compares the trees of ``src/`` and ``scripts/`` with those of the freeze tag
  ``mpcbench-freeze-v2``; it also refuses a dirty tree and any seed below
  20000.
* **Duplicate detector.** :func:`array_sha256` hashes a time series (dtype,
  shape and C-order bytes), recorded as ``sha256(ts)`` and ``sha256(raw_ts)``.
* **Timing with load.** :func:`timed` measures wall time and records the load
  average at the end (v1 timings were taken under heavy load).
* **Environment.** :func:`environment_versions` records Python and the
  pinned packages (numpy, scipy, numba, pandas) without importing them;
  ``scripts/v2/env_lock.py`` checks them against the recorded lock.
* **Protocol files.** :func:`protocol_file_record` records a protocol file's
  byte hash and its ``evidence.Protocol`` hash.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from impact_pipeline.v2 import FREEZE_TAG_V2
from impact_pipeline.v2 import seeds as S

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCKED_PACKAGES = ("numpy", "scipy", "numba", "pandas")
TREE_PATHS = ("src", "scripts", "protocols")
GUARDED_TREES = ("src", "scripts")


class ConfirmatoryGuardError(RuntimeError):
    """The checkout or the seeds do not allow a v2 confirmatory run."""


# --------------------------------------------------------------------------
# hashes
# --------------------------------------------------------------------------
def array_sha256(a) -> str:
    """SHA-256 of an array: a header with dtype and shape, then the C-order
    bytes, so equal hashes mean equal arrays (duplicate detector)."""
    arr = np.ascontiguousarray(np.asarray(a))
    h = hashlib.sha256()
    h.update(f"{arr.dtype.str}|{arr.shape}|".encode("ascii"))
    h.update(arr.tobytes())
    return h.hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def protocol_file_record(path) -> dict:
    """``{path, sha256, protocol_hash, name}`` of a protocol file; the
    protocol hash is ``evidence.Protocol.hash`` for schema ``/2`` files and
    None when the file is not a v1-layer protocol (for example a ``/3``
    protocol before its layer exists)."""
    p = Path(path)
    out = {"path": str(p), "sha256": file_sha256(p), "protocol_hash": None,
           "name": None}
    try:
        from impact_pipeline import evidence as ev

        proto = ev.Protocol.from_json(p)
    except Exception:  # noqa: BLE001 - recorded as unavailable
        return out
    out["protocol_hash"] = proto.hash
    out["name"] = proto.name
    return out


# --------------------------------------------------------------------------
# git
# --------------------------------------------------------------------------
def _git(repo_root, *args) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except Exception:  # noqa: BLE001
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def tree_shas(repo_root=REPO_ROOT, rev: str = "HEAD",
              paths: Sequence[str] = TREE_PATHS) -> dict:
    """Git tree SHAs ``{path: sha or None}`` of ``paths`` at ``rev``."""
    return {p: _git(repo_root, "rev-parse", f"{rev}:{p}") for p in paths}


def code_identity(repo_root=REPO_ROOT) -> dict:
    """
    The code that produces a run: package version, commit, dirty flag
    (tracked modifications), untracked files under the guarded trees, branch,
    tags at HEAD and the tree SHAs of ``src/``, ``scripts/`` and
    ``protocols/``.
    """
    from impact_pipeline.provenance import collect_code_version

    root = Path(repo_root)
    info = dict(collect_code_version(root))
    untracked = _git(root, "status", "--porcelain", "--untracked-files=all",
                     "--", *GUARDED_TREES)
    info["untracked_or_modified_in_guarded_trees"] = (
        None if untracked is None else bool(untracked.strip())
    )
    info["tags_at_head"] = (_git(root, "tag", "--points-at", "HEAD") or "").split()
    info["trees"] = (tree_shas(root) if info.get("git_sha")
                     else {p: None for p in TREE_PATHS})
    return info


def confirmatory_guard(seeds: Iterable, repo_root=REPO_ROOT,
                       freeze_tag: str = FREEZE_TAG_V2) -> dict:
    """
    Code identity of a v2 confirmatory run. Raises
    :class:`ConfirmatoryGuardError` unless every seed is confirmatory
    (>= 20000, no development seed mixed in), the checkout is a git work tree
    with no modified tracked files and no untracked or modified files under
    ``src/`` or ``scripts/``, and the freeze tag exists (as a tag, not as a
    branch of that name) and has the same ``src/`` and ``scripts/`` trees as
    HEAD. Returns :func:`code_identity`
    with ``freeze_tag``, ``freeze_tag_commit`` and ``confirmatory = True``.
    """
    try:
        S.assert_confirmatory(seeds)
    except S.SeedPolicyError as exc:
        raise ConfirmatoryGuardError(f"seed policy: {exc}") from exc
    root = Path(repo_root)
    info = code_identity(root)
    if not info.get("git_sha"):
        raise ConfirmatoryGuardError("a confirmatory run needs a git checkout")
    if info.get("git_dirty") is not False:
        raise ConfirmatoryGuardError("a confirmatory run needs a clean tree "
                                     "(tracked files modified)")
    if info.get("untracked_or_modified_in_guarded_trees") is not False:
        raise ConfirmatoryGuardError("a confirmatory run needs a clean tree "
                                     "(untracked or modified files under src/ "
                                     "or scripts/)")
    if not freeze_tag:
        raise ConfirmatoryGuardError("a confirmatory run needs the freeze tag")
    # the tag namespace only: a branch or another ref with the tag's name
    # does not stand in for the freeze tag
    tag_commit = _git(root, "rev-parse", "--verify", "--quiet",
                      f"refs/tags/{freeze_tag}^{{commit}}")
    if not tag_commit:
        raise ConfirmatoryGuardError(f"freeze tag {freeze_tag!r} not found")
    tag_trees = tree_shas(root, tag_commit, GUARDED_TREES)
    differ = [p for p in GUARDED_TREES
              if tag_trees[p] is None or tag_trees[p] != info["trees"].get(p)]
    if differ:
        raise ConfirmatoryGuardError(
            f"{'/, '.join(differ)}/ differ from freeze tag {freeze_tag!r}: "
            "confirmatory runs use the frozen code"
        )
    info["freeze_tag"] = str(freeze_tag)
    info["freeze_tag_commit"] = tag_commit
    info["freeze_tag_trees"] = tag_trees
    info["confirmatory"] = True
    return info


# --------------------------------------------------------------------------
# timing and environment
# --------------------------------------------------------------------------
def load_average():
    """``(1, 5, 15)``-minute load average, or None where unavailable."""
    try:
        return tuple(float(x) for x in os.getloadavg())
    except (AttributeError, OSError):
        return None


@contextlib.contextmanager
def timed():
    """``with timed() as t: ...`` fills ``t['seconds']`` (wall time) and
    ``t['load_average']`` (at the end) when the block exits, also on error."""
    out = {"seconds": None, "load_average": None}
    t0 = time.perf_counter()
    try:
        yield out
    finally:
        out["seconds"] = round(time.perf_counter() - t0, 4)
        out["load_average"] = load_average()


def environment_versions(packages: Sequence[str] = LOCKED_PACKAGES) -> dict:
    """Python version and the installed versions of ``packages`` (read from
    the package metadata; nothing is imported), plus the platform."""
    from importlib import metadata

    versions = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "implementation": sys.implementation.name,
        "platform": f"{sys.platform}-{platform.machine()}",
        "packages": versions,
    }


def run_provenance(repo_root=REPO_ROOT, protocol_files: Sequence = (),
                   seeds: Optional[Iterable] = None, confirmatory: bool = False,
                   freeze_tag: str = FREEZE_TAG_V2) -> dict:
    """
    Provenance block of a v2 run: the code identity (through
    :func:`confirmatory_guard` when ``confirmatory``), the environment, the
    protocol files and the run's split derived from its seeds.
    """
    seeds = None if seeds is None else list(seeds)
    if confirmatory:
        code = confirmatory_guard(seeds or [], repo_root, freeze_tag)
    else:
        code = code_identity(repo_root)
        code["confirmatory"] = False
    return {
        "code": code,
        "environment": environment_versions(),
        "protocol_files": [protocol_file_record(p) for p in protocol_files],
        "split": None if not seeds else S.check_seeds(seeds),
        "created_unix": time.time(),
    }


__all__ = [
    "ConfirmatoryGuardError",
    "GUARDED_TREES",
    "LOCKED_PACKAGES",
    "REPO_ROOT",
    "TREE_PATHS",
    "array_sha256",
    "code_identity",
    "confirmatory_guard",
    "environment_versions",
    "file_sha256",
    "load_average",
    "protocol_file_record",
    "run_provenance",
    "timed",
    "tree_shas",
]
