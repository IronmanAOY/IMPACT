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
    "protocols/README.md",
    "docs/preregistration/README.md",
    "docs/preregistration/MPC_BENCH_PREREGISTRATION.md",
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


# --------------------------------------------------------------------------
# documented CLI flags, verdict names, reason codes and columns
# --------------------------------------------------------------------------
USER_DOCS = (
    "README.md",
    "docs/metrics.md",
    "docs/ARCHITECTURE.md",
    "docs/HLRS_HUNTER_RUNBOOK.md",
    "scripts/hunter/README.md",
    "protocols/README.md",
    "docs/preregistration/README.md",
    "docs/preregistration/MPC_BENCH_PREREGISTRATION.md",
)
_FENCED = re.compile(r"```[a-z]*\n(.*?)```", flags=re.S)


@pytest.fixture(scope="module")
def pipeline_flags():
    import subprocess
    import sys

    text = subprocess.run(
        [sys.executable, str(REPO / "run_pipeline.py"), "--help"],
        capture_output=True, text=True, timeout=180,
    ).stdout
    return set(re.findall(r"(--[a-z0-9][a-z0-9-]*)", text))


def _commands(text, script):
    """Arguments of the commands (continuation lines joined) in fenced blocks
    that run script (the text after the script name)."""
    out = []
    for block in _FENCED.findall(text):
        joined = re.sub(r"\\\n\s*", " ", block)
        out += [ln.split(script, 1)[1] for ln in joined.splitlines() if script in ln]
    return out


def test_documented_pipeline_flags_exist(pipeline_flags):
    new = {"--protocol", "--bootstrap-se", "--bootstrap-block-len",
           "--null-surrogates", "--hunter-iim-null-surrogates",
           "--hunter-iim-bootstrap-se"}
    assert new <= pipeline_flags
    for doc in ("README.md", "docs/HLRS_HUNTER_RUNBOOK.md"):
        text = (REPO / doc).read_text(encoding="utf-8")
        for flag in new:
            assert f"`{flag}" in text or f"{flag} " in text, (doc, flag)
    bad = []
    for doc in USER_DOCS:
        text = (REPO / doc).read_text(encoding="utf-8")
        for cmd in _commands(text, "run_pipeline.py"):
            bad += [(doc, f) for f in re.findall(r"(--[a-z0-9][a-z0-9-]*)", cmd)
                    if f not in pipeline_flags]
    assert not bad, f"undocumented run_pipeline flags in commands: {bad}"


def test_documented_bench_flags_exist():
    from impact_pipeline.bench.run_bench import build_parser

    flags = set(build_parser()._option_string_actions)
    bad = []
    for doc in USER_DOCS:
        text = (REPO / doc).read_text(encoding="utf-8")
        for cmd in _commands(text, "run_bench.py"):
            bad += [(doc, f) for f in re.findall(r"(--[a-z0-9][a-z0-9-]*)", cmd)
                    if f not in flags]
    assert not bad, f"unknown run_bench flags in documented commands: {bad}"


def test_docs_use_the_v2_verdict_names():
    for doc in USER_DOCS:
        text = (REPO / doc).read_text(encoding="utf-8")
        assert not re.search(r"(?<![A-Z_])(NOT_)?ATTRIBUTED\b", text), doc
    # the changelog mentions the v1 names only where it says they were removed
    for line in (REPO / "CHANGELOG.md").read_text(encoding="utf-8").splitlines():
        if re.search(r"(?<![A-Z_])(NOT_)?ATTRIBUTED\b", line):
            assert "verdict names" in line or "evidence-layer build" in line, line


def test_metrics_doc_lists_every_reason_code_in_its_format():
    from impact_pipeline import evidence as E

    text = (REPO / "docs" / "metrics.md").read_text(encoding="utf-8")
    table = text.split("### 8.4 Reason codes", 1)[1].split("Decomposition rule", 1)[0]
    for kind in E.GLOBAL_REASON_KINDS:
        assert f"`{kind}" in table, kind
    for kind in E.PRINCIPLE_REASON_KINDS:
        assert f"`{kind}:<P>" in table, kind
    # bare codes are <KIND>:<P>, never nested under UNDEFINED
    for kind in E._BARE_PRINCIPLE_REASONS:
        assert f"`{kind}:<P>`" in table, kind
        assert f"UNDEFINED:<P>:{kind}" not in text, kind
    for detail in ("DEGENERATE_NULL", "INVALID_SE", "NULL_FAMILY_MISMATCH"):
        assert detail in table, detail


def test_metrics_doc_lists_every_evidence_column():
    from impact_pipeline import synergy_ci as sc

    text = (REPO / "docs" / "metrics.md").read_text(encoding="utf-8")
    section = text.split("## 11. Output columns", 1)[1]
    for col in sc.MPC_VERDICT_COLUMNS:
        assert col in section, col
    for field in sc.MPC_EVIDENCE_FIELDS:
        assert f"`<P>_{field}`" in section, field
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    for field in sc.MPC_EVIDENCE_FIELDS:
        assert f"<P>_{field}" in readme, field


def test_pdi_repertoire_and_protocols_are_documented():
    text = (REPO / "docs" / "metrics.md").read_text(encoding="utf-8")
    assert '### 3.3 `mode="repertoire"`' in text
    for doc in ("README.md", "docs/metrics.md", "CHANGELOG.md"):
        body = (REPO / doc).read_text(encoding="utf-8")
        assert "repertoire" in body and "protocols/" in body, doc
