"""User-facing documentation stays consistent with the code it describes."""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOCS = (
    "README.md",
    "CHANGELOG.md",
    "docs/metrics.md",
    "docs/ARCHITECTURE.md",
    "docs/HLRS_HUNTER_RUNBOOK.md",
    "docs/synthetic_data.md",
    "scripts/hunter/README.md",
)
_FENCE = re.compile(r"```.*?```", flags=re.S)
_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def _slug(heading):
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


def _anchors(path):
    text = _FENCE.sub("", path.read_text(encoding="utf-8"))
    return {
        _slug(m.group(1)) for m in re.finditer(r"^#+\s+(.*)$", text, flags=re.M)
    }


@pytest.mark.parametrize("doc", DOCS)
def test_relative_links_and_anchors_resolve(doc):
    path = REPO / doc
    text = _FENCE.sub("", path.read_text(encoding="utf-8"))
    broken = []
    for link in _LINK.findall(text):
        if link.startswith(("http://", "https://", "mailto:")):
            continue
        target, _, frag = link.partition("#")
        dest = (path.parent / target).resolve() if target else path
        if not dest.exists():
            broken.append(link)
        elif frag and dest.suffix == ".md" and frag not in _anchors(dest):
            broken.append(link)
    assert not broken, f"{doc}: broken links {broken}"


def test_metrics_doc_lists_every_surrogate_family():
    from impact_pipeline import mpc_metrics, nulls

    text = (REPO / "docs" / "metrics.md").read_text(encoding="utf-8")
    section = text.split("## 7. Null calibration", 1)[1].split("## 8.", 1)[0]
    rows = set(re.findall(r"^\| `([a-z_]+)`", section, flags=re.M))
    rows |= set(re.findall(r"^\| `[a-z_]+`, `([a-z_]+)`", section, flags=re.M))
    missing = (set(mpc_metrics.SURROGATE_METHODS) | set(nulls.SURROGATE_KINDS)) - rows
    assert not missing, f"surrogate families not documented: {sorted(missing)}"


def test_release_version_is_consistent():
    import impact_pipeline

    version = impact_pipeline.__version__
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(rf'^version = "{re.escape(version)}"$', pyproject, flags=re.M)
    citation = (REPO / "CITATION.cff").read_text(encoding="utf-8")
    assert re.search(rf'^version: "?{re.escape(version)}"?$', citation, flags=re.M)
    changelog = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    first = re.search(r"^## \[([^\]]+)\]", changelog, flags=re.M)
    assert first and first.group(1) == version


def test_documented_code_version_example_is_not_doubled():
    """IMPACT_CODE_VERSION is a suffix: the package version is added to it."""
    from impact_pipeline import provenance

    setup = (REPO / "scripts" / "hunter" / "hunter_pbs_setup.sh").read_text()
    runbook = (REPO / "docs" / "HLRS_HUNTER_RUNBOOK.md").read_text(encoding="utf-8")
    examples = re.findall(r"IMPACT_CODE_VERSION=(\S+?)`?(?:\s|$)", setup + runbook)
    assert examples
    pkg = provenance.package_version()
    for value in examples:
        text = provenance.format_code_version(
            {"package_version": pkg, "declared_version": value}
        )
        assert text.count(pkg) == 1, (value, text)


def test_hunter_defaults_in_docs_match_the_code():
    from impact_pipeline import iim_xp
    from impact_pipeline.execution_profiles import (
        HUNTER_EXECUTION_PROFILE,
        HunterPBSProfile,
    )

    pbs = HunterPBSProfile()
    readme = (REPO / "scripts" / "hunter" / "README.md").read_text(encoding="utf-8")
    runbook = (REPO / "docs" / "HLRS_HUNTER_RUNBOOK.md").read_text(encoding="utf-8")
    hours = [
        int(t.split(":")[0]) for t in (pbs.phase1_time, pbs.cut_time, pbs.reduce_time)
    ]
    assert f"{hours[0]}h / {hours[1]}h / {hours[2]}h" in readme
    assert f"{hours[0]} h / {hours[1]} h / {hours[2]} h" in runbook
    shards = (
        f"{HUNTER_EXECUTION_PROFILE.hunter_phase1_shards_per_run} / "
        f"{HUNTER_EXECUTION_PROFILE.hunter_cut_shards_per_run}"
    )
    assert shards in readme and shards in runbook
    assert f"{pbs.max_array_size} (PBS default)" in readme
    assert f"{pbs.cores_per_node} / {pbs.gpus_per_node}" in readme
    assert f"| {iim_xp._DEFAULT_MAX_ELEMENTS} |" in runbook
    assert iim_xp._DEFAULT_MAX_ELEMENTS == 2**23
    assert iim_xp._DEFAULT_CACHE_ELEMENTS == 2**27
    assert "2^23 / 2^27" in readme
