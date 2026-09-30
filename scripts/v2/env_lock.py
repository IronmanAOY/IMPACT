#!/usr/bin/env python
"""
Environment lock of MPC-Bench v2.

v2 adds no package. The numerical stack that the v1 results and the v2
confirmatory run depend on is pinned here (recorded on the machine of the v1
and v2 runs, osx-arm64, conda environment ``impact-synergy-clean``), and every
v2 run records its environment next to its results
(``impact_pipeline.v2.provenance.environment_versions``).

Checks:

* ``--check``: the installed numpy, scipy, numba and pandas (and Python)
  equal the lock. On the locked platform the versions must match exactly;
  elsewhere (for example the CI runners) the minor versions must match, which
  is what ``environment.yml`` pins.
* v2 code does not import ``mne`` or ``scikit-learn`` (inverse solutions and
  shrinkage are written in numpy): every v2 source file is scanned.

Exit status 0 when every check passes, 1 otherwise. ``--json`` prints the
full report.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from impact_pipeline.v2.provenance import (  # noqa: E402
    LOCKED_PACKAGES,
    environment_versions,
)

LOCK = {
    "platform": "darwin-arm64",
    "python": "3.10.19",
    "packages": {
        "numpy": "2.2.6",
        "scipy": "1.15.2",
        "numba": "0.61.2",
        "pandas": "2.3.3",
    },
}
FORBIDDEN_MODULES = ("mne", "sklearn")
# v2 source files (globs relative to the repository root).
V2_SOURCES = (
    "src/impact_pipeline/v2/**/*.py",
    "src/impact_pipeline/evidence_v2.py",
    "src/impact_pipeline/bench/*_v2.py",
    "src/impact_pipeline/bench/designs_v2/**/*.py",
    "scripts/v2/**/*.py",
    "scripts/*_v2.py",
)
_IMPORT_CALL = re.compile(
    r"""(?:import_module|__import__)\(\s*['"](%s)(?:\.[\w.]*)?['"]"""
    % "|".join(FORBIDDEN_MODULES)
)


def _minor(version: Optional[str]) -> Optional[str]:
    if version is None:
        return None
    parts = str(version).split(".")
    return ".".join(parts[:2])


def check_versions(current: Optional[dict] = None, lock: dict = LOCK,
                   strict: Optional[bool] = None) -> List[str]:
    """
    Mismatches between the environment and the lock (empty when it matches).
    ``strict`` compares exact versions and defaults to True on the locked
    platform and False elsewhere (minor versions only).
    """
    cur = environment_versions() if current is None else current
    if strict is None:
        strict = cur.get("platform") == lock["platform"]
    cmp = (lambda v: v) if strict else _minor
    out = []
    if cmp(cur.get("python")) != cmp(lock["python"]):
        out.append(f"python {cur.get('python')} != locked {lock['python']}")
    for name in LOCKED_PACKAGES:
        have = (cur.get("packages") or {}).get(name)
        want = lock["packages"][name]
        if cmp(have) != cmp(want):
            out.append(f"{name} {have} != locked {want}")
    return out


def v2_source_files(repo_root=REPO_ROOT,
                    patterns: Iterable[str] = V2_SOURCES) -> List[Path]:
    root = Path(repo_root)
    files = set()
    for pat in patterns:
        files.update(p for p in root.glob(pat)
                     if p.is_file() and "__pycache__" not in p.parts)
    return sorted(files)


def forbidden_imports(files: Iterable[Path], modules=FORBIDDEN_MODULES) -> List[dict]:
    """Imports of ``modules`` (``import m``, ``from m[...] import``,
    ``importlib.import_module('m')``) in the given Python files."""
    out = []
    for path in files:
        text = Path(path).read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            out.append({"file": str(path), "line": exc.lineno,
                        "module": "<syntax error>"})
            continue
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            for name in names:
                if name.split(".")[0] in modules:
                    out.append({"file": str(path), "line": node.lineno, "module": name})
        for m in _IMPORT_CALL.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            out.append({"file": str(path), "line": line, "module": m.group(1)})
    return out


def environment_yml_minors(path=REPO_ROOT / "environment.yml") -> dict:
    """Minor versions pinned in ``environment.yml`` for the locked packages
    (``- numpy=2.2.*`` -> ``2.2``) and Python."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*-\s*([A-Za-z0-9_.-]+)\s*=\s*([0-9]+\.[0-9]+)", line)
        if m and m.group(1) in (*LOCKED_PACKAGES, "python"):
            out[m.group(1)] = m.group(2)
    return out


def report(repo_root=REPO_ROOT) -> dict:
    env = environment_versions()
    files = v2_source_files(repo_root)
    return {
        "lock": LOCK,
        "environment": env,
        "strict": env.get("platform") == LOCK["platform"],
        "version_mismatches": check_versions(env),
        "v2_source_files": [str(Path(f).relative_to(repo_root)) for f in files],
        "forbidden_imports": forbidden_imports(files),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="exit 1 on a version mismatch or a forbidden import "
                         "(the default action)")
    ap.add_argument("--json", action="store_true", help="print the full report")
    args = ap.parse_args(argv)
    rep = report()
    ok = not rep["version_mismatches"] and not rep["forbidden_imports"]
    if args.json:
        print(json.dumps(rep, indent=2))
    else:
        mode = "exact" if rep["strict"] else "minor versions"
        print(f"environment lock ({mode}): "
              f"{'ok' if not rep['version_mismatches'] else 'MISMATCH'}")
        for m in rep["version_mismatches"]:
            print(f"  {m}")
        print(f"v2 sources scanned: {len(rep['v2_source_files'])}; forbidden imports: "
              f"{len(rep['forbidden_imports'])}")
        for f in rep["forbidden_imports"]:
            print(f"  {f['file']}:{f['line']} imports {f['module']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
