"""User-facing documentation stays consistent with the code it describes:
links, release metadata, documented command-line flags, reason codes, output
columns and protocols. The TOST section of the metrics reference documents
every reason and flag of the central vocabulary (and nothing else), every SE
method with its principles, the status rule block, the order of the checks
and the implied-precision table; the protocol table quotes the hash of the
benchmark template."""

import json
import re
from pathlib import Path

import pytest

from impact_pipeline import evidence as E
from impact_pipeline import reasons as R
from impact_pipeline import testability as T

REPO = Path(__file__).resolve().parents[1]
METRICS = REPO / "docs" / "metrics.md"
DOCS = (
    "README.md",
    "CHANGELOG.md",
    "docs/metrics.md",
    "docs/ARCHITECTURE.md",
    "docs/HLRS_HUNTER_RUNBOOK.md",
    "docs/synthetic_data.md",
    "scripts/hunter/README.md",
    "protocols/README.md",
    "protocols/examples/README.md",
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
    first = re.search(r"^## \[([^\]]+)\] - (\S+)$", changelog, flags=re.M)
    assert first and first.group(1) == version
    assert re.search(rf'^date-released: "?{first.group(2)}"?$', citation, flags=re.M)
    # container labels and the image names in the build examples
    for name in ("Dockerfile", "Singularity"):
        text = (REPO / name).read_text(encoding="utf-8")
        label = rf'org\.opencontainers\.image\.version[ =]"?{re.escape(version)}"?( \\)?$'
        assert re.search(label, text, flags=re.M), name
        tags = re.findall(r"impact-synergy-pipeline[:_](\d+\.\d+\.\d+)", text)
        assert tags and set(tags) == {version}, (name, tags)
    # the README and the runbook clone the release tag of this version
    runbook = (REPO / "docs" / "HLRS_HUNTER_RUNBOOK.md").read_text(encoding="utf-8")
    assert runbook.startswith(f"# HLRS Hunter Runbook: IMPaCT Synergy Pipeline {version}\n")
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    for doc, text in (("README.md", readme), ("runbook", runbook)):
        clones = re.findall(r"git clone --branch (\S+)", text)
        assert clones and set(clones) == {f"v{version}"}, (doc, clones)


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
    from impact_pipeline.bench.runner import build_parser

    parser = build_parser()
    flags = set(parser._option_string_actions)
    for action in parser._subparsers._group_actions:
        for sub in action.choices.values():
            flags |= set(sub._option_string_actions)
    bad = []
    for doc in USER_DOCS:
        text = (REPO / doc).read_text(encoding="utf-8")
        for cmd in _commands(text, "run_bench.py"):
            bad += [(doc, f) for f in re.findall(r"(--[a-z0-9][a-z0-9-]*)", cmd)
                    if f not in flags]
    assert not bad, f"unknown run_bench flags in documented commands: {bad}"


_SCRIPT = re.compile(r"scripts/([a-z_]+\.py)")
_FLAG = re.compile(r"(?<![\w-])(--[a-z0-9][a-z0-9-]*)")


def _script_flags(name):
    """Flags of scripts/<name> and of each of its subcommands, from --help."""
    import subprocess
    import sys

    def help_text(*args):
        return subprocess.run(
            [sys.executable, str(REPO / "scripts" / name), *args, "--help"],
            capture_output=True, text=True, timeout=180, cwd=str(REPO),
        ).stdout

    text = help_text()
    flags = set(_FLAG.findall(text))
    subs = re.search(r"\{([a-z_,]+)\}", text)
    for sub in subs.group(1).split(",") if subs else ():
        flags |= set(_FLAG.findall(help_text(sub)))
    return flags


def test_documented_script_flags_exist():
    """Every flag in a documented command of a Python script under scripts/
    exists in that script's parser (run_bench.py has its own test above)."""
    used = {}
    for doc in DOCS:
        text = (REPO / doc).read_text(encoding="utf-8")
        for block in _FENCED.findall(text):
            for line in re.sub(r"\\\n\s*", " ", block).splitlines():
                for m in _SCRIPT.finditer(line):
                    if m.group(1) != "run_bench.py":
                        used.setdefault(m.group(1), set()).update(
                            (doc, f) for f in _FLAG.findall(line[m.end():]))
    assert {"null_calibration.py", "iim_validation.py",
            "build_example_protocols.py", "compute_empirical_reference.py"} <= set(used)
    bad = []
    for name, uses in sorted(used.items()):
        if uses:
            flags = _script_flags(name)
            bad += [(doc, name, f) for doc, f in sorted(uses) if f not in flags]
    assert not bad, f"unknown script flags in documented commands: {bad}"


def test_docs_use_the_current_verdict_names():
    for doc in USER_DOCS:
        text = (REPO / doc).read_text(encoding="utf-8")
        assert not re.search(r"(?<![A-Z_])(NOT_)?ATTRIBUTED\b", text), doc
    # the changelog mentions the legacy names only where it says they were removed
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
    section = text.split("## 13. Output columns", 1)[1]
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


# --------------------------------------------------------------------------
# default MPC protocol: mpc_default.json; the behavioural-RAM protocol is
# opt-in only
# --------------------------------------------------------------------------
# the documents that describe the default protocol
PROTOCOL_DOCS = (
    "README.md",
    "docs/metrics.md",
    "docs/ARCHITECTURE.md",
    "docs/HLRS_HUNTER_RUNBOOK.md",
    "scripts/hunter/README.md",
    "protocols/README.md",
    "protocols/examples/README.md",
)


def _flat(doc):
    return " ".join((REPO / doc).read_text(encoding="utf-8").split())


@pytest.mark.parametrize("doc", PROTOCOL_DOCS)
def test_docs_name_the_default_and_the_behavioural_ram_caveat(doc):
    text = _flat(doc)
    assert "mpc_default.json" in text, doc
    assert "NO_DECLARED_WORKSPACE" in text, doc
    if "mpc_behavioural_ram" in text:
        # opt-in only, with the interpretation of an ABSENT RAM stated here or
        # in the section of protocols/README.md that the document links to
        assert "opt-in" in text, doc
        assert ("not absence of responsiveness" in text
                or "README.md#mpc_behavioural_ramjson-opt-in" in text), doc
    readme = _flat("protocols/README.md")
    assert "not absence of responsiveness" in readme


def test_documented_hunter_campaigns_pass_a_protocol_with_a_hub():
    """A Hunter campaign that computes the evidence must pass a protocol
    derived from the default with a declared NAS hub (the default declares
    none, so NAS would be UNDEFINED in every run); never the bare default or
    the behavioural-RAM protocol. A preprocessing-only build (discarded) is
    exempt."""
    seen = 0
    for doc in ("README.md", "docs/HLRS_HUNTER_RUNBOOK.md", "scripts/hunter/README.md"):
        text = (REPO / doc).read_text(encoding="utf-8")
        flat = " ".join(text.split())
        assert "protocols/examples/" in flat, doc
        assert "mpc_default_schaefer400_7networks_hub.json" in flat, doc
        for cmd in _commands(text, "run_pipeline.py"):
            if "build-campaign" not in cmd or "--run-preprocessing" in cmd:
                continue
            seen += 1
            m = re.search(r"--protocol\s+(\S+)", cmd)
            assert m, (doc, cmd)
            value = m.group(1).strip("\"'")
            assert value not in ("protocols/mpc_default.json",
                                 "protocols/mpc_behavioural_ram.json"), (doc, cmd)
    assert seen >= 3


# --------------------------------------------------------------------------
# the TOST rule and the protocol fields in the metrics reference
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bench_metrics():
    return METRICS.read_text(encoding="utf-8")


def _section(text, number):
    """Subsection ``number`` (e.g. "9.5") up to the next heading."""
    m = re.search(rf"^### {re.escape(number)} .*?(?=^##+ |\Z)", text,
                  flags=re.M | re.S)
    assert m, f"section {number} missing"
    return m.group(0)


def _table_codes(block):
    return re.findall(r"^\| `([A-Z][A-Z0-9_]*)` \|", block, flags=re.M)


def test_bench_metrics_doc_lists_every_reason_and_flag(bench_metrics):
    reasons_part, flags_part = _section(bench_metrics, "9.5").split(
        "Flags are markers", 1)
    reasons = _table_codes(reasons_part)
    flags = _table_codes(flags_part)
    assert sorted(reasons) == sorted(R.REASONS)
    assert sorted(flags) == sorted(R.FLAGS)
    assert len(set(reasons)) == len(reasons) and len(set(flags)) == len(flags)
    for code in reasons:
        assert R.REASONS[code].status == "UNDEFINED"
    assert "UNDEFINED is never counted as ABSENT" in bench_metrics


def test_bench_metrics_doc_lists_every_se_method(bench_metrics):
    rows = re.findall(r"^\| `([A-Za-z0-9_]+)` \| ([A-Z, ]+) \| (.+) \|$",
                      _section(bench_metrics, "9.3"), flags=re.M)
    assert sorted(name for name, _, _ in rows) == sorted(E.SE_METHODS)
    for name, principles, _ in rows:
        assert tuple(p.strip() for p in principles.split(",")) == (
            E.SE_METHODS[name].principles), name


def test_bench_metrics_doc_quotes_the_status_rule_block(bench_metrics):
    blocks = re.findall(r"```json\n(.*?)\n```", _section(bench_metrics, "10.2"),
                        flags=re.S)
    assert [json.loads(b) for b in blocks] == [E.StatusRule().to_dict()]


def test_bench_metrics_doc_lists_the_order_of_the_checks(bench_metrics):
    part = _section(bench_metrics, "9.2").split("**Order of the checks**", 1)[1]
    items = re.findall(r"^\d+\. (.*?)(?=^\d+\. |\Z)", part, flags=re.M | re.S)
    codes = [re.search(r"`([A-Z_]+)[`:]", item).group(1) for item in items[:10]]
    assert codes == [code for _, code in E.CHECK_ORDER]


def test_bench_metrics_doc_precision_table_matches_the_code(bench_metrics):
    rows = re.findall(r"^\| (\d+) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \|$",
                      _section(bench_metrics, "9.2"), flags=re.M)
    assert [int(r[0]) for r in rows] == [9, 12, 19, 199]
    for df, s_a, s_a_nas, s_p in rows:
        df = int(df)
        assert f"{T.absent_precision(df):.3f}" == s_a
        assert f"{T.absent_precision(df, members=2):.3f}" == s_a_nas
        assert f"{T.present_precision(df):.3f}" == s_p


def test_protocols_readme_quotes_the_bench_template_and_its_hash():
    readme = (REPO / "protocols" / "README.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| `bench/([a-z0-9_]+\.json)` \| .* \| `([0-9a-f]{64})` \|$",
                      readme, flags=re.M)
    assert [name for name, _ in rows] == ["template.json"]
    for name, quoted in rows:
        assert E.resolve_protocol(REPO / "protocols" / "bench" / name).hash == quoted


def test_bench_metrics_doc_names_the_schemas_and_versions(bench_metrics):
    for token in (E.PROTOCOL_SCHEMA, E.STATUS_RULE_VERSION,
                  T.PRECISION_SCHEMA, T.ANCHORS_SCHEMA, "mpc-bench-result/1"):
        assert token in bench_metrics, token
