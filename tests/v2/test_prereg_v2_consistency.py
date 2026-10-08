# -*- coding: utf-8 -*-
"""The v2 preregistration agrees with the files it quotes.

* every hypothesis and part of ``protocols/v2/hypotheses_v2.json`` appears in
  the preregistration with the same rule name, parameters and label;
* every protocol hash and file SHA-256 quoted equals the generated protocols
  (recomputed from the files, and as the build manifest lists them), and no
  other 64-hex value in the documents is unknown;
* the seed map quoted equals ``protocols/v2/seed_map_v2.json``;
* the held-out predictions file was committed before the logged release and
  has not changed since (skipped where the development outputs are absent,
  for example on a fresh clone);
* the hypotheses file has status ``final``;
* the relative links and anchors of the new documents resolve.
"""
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

from impact_pipeline.v2 import hypothesis_engine as HE
from impact_pipeline.v2 import ownership as own

REPO = Path(__file__).resolve().parents[2]
PREREG_DIR = REPO / "docs" / "preregistration"
PREREG = PREREG_DIR / "MPC_BENCH_PREREGISTRATION_V2.md"
COMPANIONS = (
    PREREG_DIR / "v2" / "testability_table.md",
    PREREG_DIR / "v2" / "development_expectations.md",
    PREREG_DIR / "v2" / "operating_characteristics.md",
)
HYPOTHESES = REPO / "protocols" / "v2" / "hypotheses_v2.json"
HELD_OUT = REPO / "protocols" / "v2" / "held_out_predictions_v2.json"
SEED_MAP = REPO / "protocols" / "v2" / "seed_map_v2.json"
GENERATED = REPO / "protocols" / "v2" / "generated"
MANIFEST = GENERATED / "build_manifest.json"
HELD_OUT_REL = "protocols/v2/held_out_predictions_v2.json"
RELEASE_LOG_REL = Path("outputs/mpcbench_v2/dev_calibration/held_out_release_log.jsonl")
OTHER_FILES = (
    "protocols/v2/hypotheses_v2.json",
    "protocols/v2/held_out_predictions_v2.json",
    "protocols/v2/seed_map_v2.json",
    "protocols/v2/mpc_bench_v2_template.json",
    "protocols/v2/mpc_default_v2.json",
)
HEX64 = re.compile(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _between(text: str, name: str) -> str:
    begin, end = f"<!-- {name}:begin -->", f"<!-- {name}:end -->"
    assert text.count(begin) == 1 and text.count(end) == 1, name
    return text.split(begin, 1)[1].split(end, 1)[0]


@pytest.fixture(scope="module")
def prereg() -> str:
    return PREREG.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def spec() -> dict:
    return json.loads(HYPOTHESES.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# hypotheses
# --------------------------------------------------------------------------
def test_the_hypotheses_file_is_final(spec):
    assert spec["status"] == "final"


def _blocks(text: str, kind: str) -> dict:
    """``{id: block}`` for the ``<!-- part ID -->`` (or hypothesis) markers;
    a block runs to the next marker or heading."""
    marker = re.compile(r"^<!-- (part|hypothesis) (.+?) -->$", flags=re.M)
    stops = sorted(
        {m.start() for m in marker.finditer(text)}
        | {m.start() for m in re.finditer(r"^#{1,4} ", text, flags=re.M)}
        | {len(text)}
    )
    out = {}
    for m in marker.finditer(text):
        if m.group(1) != kind:
            continue
        end = next(s for s in stops if s > m.start())
        assert m.group(2) not in out, f"{kind} {m.group(2)} quoted twice"
        out[m.group(2)] = text[m.end():end]
    return out


def test_every_hypothesis_appears_once_with_its_title(prereg, spec):
    blocks = _blocks(prereg, "hypothesis")
    ids = [h["id"] for h in spec["hypotheses"]]
    assert set(blocks) == set(ids)
    for h in spec["hypotheses"]:
        assert f"#### {h['id']}: {h['title']}\n" in prereg, h["id"]
        if h.get("statement"):
            assert f"- Statement: {h['statement']}\n" in blocks[h["id"]], h["id"]
        if h.get("expectation"):
            assert f"- Development expectation: {h['expectation']}\n" in blocks[h["id"]]
        if h.get("prediction"):
            assert f"predicted outcome {h['prediction']}" in blocks[h["id"]], h["id"]


def test_every_part_appears_with_its_rule_parameters_and_label(prereg, spec):
    blocks = _blocks(prereg, "part")
    parts = [p for h in spec["hypotheses"] for p in h.get("parts") or []]
    assert len(parts) == len({p["id"] for p in parts})
    assert set(blocks) == {p["id"] for p in parts}
    for p in parts:
        block = blocks[p["id"]]
        head = re.search(r"^\*\*Part `(.+?)`\*\* \((\w+); label: (\S+?);", block, flags=re.M)
        assert head and head.group(1) == p["id"], p["id"]
        assert head.group(2) == (p.get("role") or "decisive"), p["id"]
        assert head.group(3) == (p.get("label") or "none"), p["id"]
        rules = re.findall(r"^- Rule: `([a-z0-9_]+)`$", block, flags=re.M)
        assert rules == [p["rule"]], p["id"]
        params = re.findall(r"^- Parameters: `(.*)`$", block, flags=re.M)
        assert len(params) == 1, p["id"]
        assert json.loads(params[0]) == (p.get("params") or {}), p["id"]
        assert f"- Text: {p['text']}\n" in block, p["id"]
        if p.get("prediction"):
            assert f"predicted: {p['prediction']} (stated)" in block, p["id"]
        data = re.search(r"```json\n(.*?)\n```", block, flags=re.S)
        assert data and json.loads(data.group(1)) == (p.get("data") or {}), p["id"]


def test_every_rule_of_the_hypotheses_is_defined_in_the_rule_table(prereg, spec):
    section = prereg.split("### 6.1 ", 1)[1].split("### 6.2 ", 1)[0]
    defined = set()
    for row in re.findall(r"^\| (`[a-z0-9_]+`(?:, `[a-z0-9_]+`)*) \|", section, flags=re.M):
        defined |= set(re.findall(r"`([a-z0-9_]+)`", row))
    used = {p["rule"] for h in spec["hypotheses"] for p in h.get("parts") or []}
    assert used <= defined, sorted(used - defined)
    assert used <= set(HE.RULES), sorted(used - set(HE.RULES))


def test_the_conventions_are_quoted(prereg, spec):
    c = spec["conventions"]
    text = " ".join(prereg.split())
    assert f"`alpha = {c['alpha']}`" in text
    assert f"`alpha_absent = {c['alpha_absent']}`" in text
    assert f"`z = {c['z']}`" in text
    assert f"`delta = {c['delta']:.2f}`" in text
    assert f"false-PRESENT bound `alpha + 0.02 = {c['false_present_bound']}`" in text


# --------------------------------------------------------------------------
# protocols and file hashes
# --------------------------------------------------------------------------
def test_quoted_protocol_hashes_equal_the_generated_protocols(prereg, manifest):
    protos = dict(HE.load_protocol_files([GENERATED]))
    rows = re.findall(
        r"^\| `([^`]+)` \| `([^`]+)` \| `([0-9a-f]{64})` \| `([0-9a-f]{64})` \|$",
        _between(prereg, "frozen-protocols"), flags=re.M)
    assert {r[0] for r in rows} == set(manifest["protocols"]) == set(protos)
    assert len(rows) == len(manifest["protocols"])
    for key, name, phash, fsha in rows:
        assert name == manifest["protocols"][key]["file"], key
        assert phash == manifest["protocols"][key]["hash"] == protos[key].hash, key
        assert fsha == manifest["files"][name] == _sha(GENERATED / name), key


def test_quoted_generated_tables_equal_the_files(prereg, manifest):
    rows = dict(re.findall(r"^\| `([^`]+)` \| `([0-9a-f]{64})` \|$",
                           _between(prereg, "frozen-tables"), flags=re.M))
    protocol_files = {v["file"] for v in manifest["protocols"].values()}
    expected = {n for n in manifest["files"] if n not in protocol_files}
    assert set(rows) == expected | {MANIFEST.name}
    for name, sha in rows.items():
        assert sha == _sha(GENERATED / name), name
        if name != MANIFEST.name:
            assert sha == manifest["files"][name], name


def test_quoted_file_hashes_equal_the_files(prereg):
    """Each of these files is quoted with its SHA-256: the first hash after
    the file's name on a line, on every line that has one."""
    for rel in OTHER_FILES:
        quoted = []
        for line in prereg.splitlines():
            at = line.find(f"`{rel}`")
            if at < 0:
                continue
            m = HEX64.search(line, at)
            if m:
                quoted.append(m.group(0))
        assert quoted, f"{rel}: no SHA-256 quoted"
        assert set(quoted) == {_sha(REPO / rel)}, rel


def _known_hashes(manifest) -> set:
    known = {v["hash"] for v in manifest["protocols"].values()}
    known |= set(manifest["files"].values())
    known.add(_sha(MANIFEST))
    known.add(manifest["decisions_sha256"])
    known |= {_sha(REPO / rel) for rel in OTHER_FILES}
    known |= {p["sha256"] for r in manifest.get("releases") or [] for p in r["predictions"]}
    evidence = json.loads((GENERATED / "calibration_evidence.json").read_text())
    known.add(evidence["constants"]["sha256"])
    known.add(HE.spec_sha256(json.loads(HYPOTHESES.read_text(encoding="utf-8"))))
    return known


def test_the_spec_hash_of_the_hypotheses_file_is_quoted(prereg, spec):
    """Section 13 gives the canonical spec hash, which the evaluator and the
    audit record, next to the file SHA-256."""
    line = next(ln for ln in prereg.splitlines()
                if ln.startswith("| Hypotheses | `protocols/v2/hypotheses_v2.json`"))
    assert _sha(HYPOTHESES) in line
    assert f"spec hash `{HE.spec_sha256(spec)}`" in line


def test_no_unknown_hash_is_quoted_in_the_preregistration(prereg, manifest):
    unknown = sorted(set(HEX64.findall(prereg)) - _known_hashes(manifest))
    assert not unknown, unknown


def test_the_testability_companion_names_the_generated_table():
    text = COMPANIONS[0].read_text(encoding="utf-8")
    assert _sha(GENERATED / "testability_table.json") in text
    table = json.loads((GENERATED / "testability_table.json").read_text())
    rows = re.findall(r"^\| `([^`]+)` \| (absent|present) \| (\w+) \| `([^`]+)` \| (\d+) \|",
                      text.split("## Rows", 1)[1].split("## Status rates", 1)[0], flags=re.M)
    assert sorted(rows) == sorted(
        (r["family"], r["kind"], r["principle"], r["witness"], str(r["n"]))
        for r in table["rows"])


# --------------------------------------------------------------------------
# seed map
# --------------------------------------------------------------------------
def test_the_quoted_seed_map_equals_the_seed_map_file(prereg):
    sm = json.loads(SEED_MAP.read_text(encoding="utf-8"))
    section = _between(prereg, "seed-map")
    rows = re.findall(r"^\| (development|confirmatory) \| `(\d+)-(\d+)` \| (.*) \|$",
                      section, flags=re.M)
    expected = []
    for split in ("development", "confirmatory"):
        for a in sm[split]["assignments"]:
            expected.append((split, a["min"], a["max"], a["use"]))
            for part in a.get("parts") or []:
                expected.append((split, part["min"], part["max"], part["use"]))
    got = [(s, int(lo), int(hi), use.replace("\\|", "|")) for s, lo, hi, use in rows]
    assert [g[:3] for g in got] == [e[:3] for e in expected]
    for g, e in zip(got, expected):
        assert g[3].startswith(e[3]), (g, e)
    pol = sm["policy"]
    flat = " ".join(section.split())
    assert (f"development seeds {pol['development']['min']}-{pol['development']['max']}; "
            f"confirmatory seeds >= {pol['confirmatory']['min']}") in flat
    for b in pol["never_reused"]:
        assert f"{b['min']}-{b['max']}" in flat
    used = ", ".join(f"{b['min']}-{b['max']}" for b in sm["development"]["used_before_v2"])
    assert used in flat
    assert f"`{sm['freeze_tag']}`" in flat
    for key in ("development_regime", "held_out_regime"):
        assert str(sm["lead_field_seeds"][key]) in flat


# --------------------------------------------------------------------------
# held-out predictions and their release
# --------------------------------------------------------------------------
def _git(*args) -> str:
    try:
        proc = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                              text=True, check=False)
    except FileNotFoundError:
        pytest.skip("git is not available")
    if proc.returncode != 0:
        raise AssertionError(f"git {' '.join(args)}: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _release_log() -> Path:
    roots = [REPO]
    main = own.main_checkout_root(REPO)
    if main is not None:
        roots.append(main)
    for root in roots:
        path = Path(root) / RELEASE_LOG_REL
        if path.is_file():
            return path
    pytest.skip(f"no release log ({RELEASE_LOG_REL}): the development outputs are "
                "not versioned and are absent here (for example on a fresh clone)")


def test_the_held_out_predictions_were_committed_before_their_release():
    log = _release_log()
    entries = [json.loads(ln) for ln in log.read_text(encoding="utf-8").splitlines()
               if ln.strip()]
    named = [(e, p) for e in entries for p in e.get("predictions") or []
             if p.get("path") == HELD_OUT_REL]
    assert named, f"{log}: no release names {HELD_OUT_REL}"
    if not _git("rev-parse", "--is-inside-work-tree") == "true":
        pytest.skip("not a git checkout")
    history = _git("log", "--format=%H %ct", "--", HELD_OUT_REL).splitlines()
    assert history, f"{HELD_OUT_REL} is not committed"
    last_commit, last_time = history[0].split()
    for entry, pred in named:
        assert pred["tracked"] is True and pred["clean"] is True
        # the predictions the release names are the file as it is now ...
        assert pred["sha256"] == _sha(HELD_OUT), "the released predictions changed"
        # ... the file has not been committed again since that commit ...
        assert last_commit == pred["commit"], "the predictions file changed after its release"
        # ... and that commit precedes the release, in history and in time
        _git("merge-base", "--is-ancestor", pred["commit"], entry["head"])
        assert int(last_time) < float(entry["released_unix"])
        assert entry["schema"] == "mpc-bench-held-out-release/1"


def test_the_preregistration_names_the_release(prereg, manifest):
    releases = manifest.get("releases") or []
    assert releases, "the build manifest names no release"
    flat = " ".join(prereg.split())
    for r in releases:
        assert f"`{r['release_id']}`" in flat
        for p in r["predictions"]:
            assert f"`{p['commit'][:7]}`" in flat
            assert p["sha256"] == _sha(HELD_OUT)


# --------------------------------------------------------------------------
# links
# --------------------------------------------------------------------------
_FENCE = re.compile(r"```.*?```", flags=re.S)
_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def _slug(heading: str) -> str:
    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


def _anchors(path: Path) -> set:
    text = _FENCE.sub("", path.read_text(encoding="utf-8"))
    return {_slug(m.group(1)) for m in re.finditer(r"^#+\s+(.*)$", text, flags=re.M)}


@pytest.mark.parametrize("doc", (PREREG,) + COMPANIONS, ids=lambda p: p.name)
def test_relative_links_and_anchors_resolve(doc):
    text = _FENCE.sub("", doc.read_text(encoding="utf-8"))
    broken = []
    for link in _LINK.findall(text):
        if link.startswith(("http://", "https://", "mailto:")):
            continue
        target, _, frag = link.partition("#")
        dest = (doc.parent / target).resolve() if target else doc
        if not dest.exists():
            broken.append(link)
        elif frag and dest.suffix == ".md" and frag not in _anchors(dest):
            broken.append(link)
    assert not broken, f"{doc.name}: broken links {broken}"


def test_the_companions_are_flagged_as_development():
    for doc in COMPANIONS:
        assert "DEVELOPMENT - NOT A RESULT" in doc.read_text(encoding="utf-8"), doc.name
