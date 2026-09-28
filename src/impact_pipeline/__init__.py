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
    Installed distribution metadata (importlib.metadata) first; for a source
    checkout used via PYTHONPATH (e.g. the HLRS Hunter venv route) the
    pyproject.toml next to the package; '0+unknown' if neither is available.
    """
    try:
        from importlib import metadata as importlib_metadata

        return str(importlib_metadata.version(_DIST_NAME))
    except Exception:
        pass
    return _version_from_pyproject() or "0+unknown"


__version__ = _resolve_version()

__all__ = ["__version__"]
