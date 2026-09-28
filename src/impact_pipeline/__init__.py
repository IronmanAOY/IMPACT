"""IMPaCT metric pipeline (RAM, PDI, NAS, IIM, SRPI) for EEG and fMRI data."""

from __future__ import annotations

import re
from pathlib import Path

_DIST_NAME = "impact-synergy-pipeline"


def _version_from_pyproject() -> str | None:
    """Version declared in the checkout's pyproject.toml (source tree runs)."""
    path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    # Only the [project] table of this distribution counts.
    section = re.search(r"(?ms)^\[project\]\s*$(.*?)(?=^\[|\Z)", text)
    if section is None:
        return None
    body = section.group(1)
    name = re.search(r'(?m)^name\s*=\s*"([^"]+)"', body)
    version = re.search(r'(?m)^version\s*=\s*"([^"]+)"', body)
    if name is None or version is None or name.group(1) != _DIST_NAME:
        return None
    return version.group(1)


def _resolve_version() -> str:
    """
    Version of the code that is actually imported: the pyproject.toml next to
    the package when it is imported from a source checkout (PYTHONPATH, e.g.
    the HLRS Hunter venv route, or an editable install), else the installed
    distribution metadata (importlib.metadata); '0+unknown' if neither is
    available. The checkout comes first so that an unrelated installed wheel
    of another version cannot mislabel a checkout run.
    """
    declared = _version_from_pyproject()
    if declared:
        return declared
    try:
        from importlib import metadata as importlib_metadata

        return str(importlib_metadata.version(_DIST_NAME))
    except Exception:
        return "0+unknown"


__version__ = _resolve_version()

__all__ = ["__version__"]
