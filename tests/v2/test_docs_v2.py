# -*- coding: utf-8 -*-
"""
``docs/metrics_v2.md`` stays consistent with the v2 evidence layer: every
reason and flag of the central vocabulary is documented (and nothing else),
every SE method with its principles, the status rule block, the order of the
checks, the implied-precision table and the hashes of the shipped /3
protocols. Its links and anchors are checked with the other documents in
``tests/test_docs_consistency.py``.
"""
import json
import re
from pathlib import Path

import pytest

from impact_pipeline import evidence_v2 as E
from impact_pipeline.v2 import reasons as R
from impact_pipeline.v2 import testability as T

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "docs" / "metrics_v2.md"


@pytest.fixture(scope="module")
def text():
    return DOC.read_text(encoding="utf-8")


def section(text, number):
    m = re.search(rf"^## {number}\. .*?(?=^## |\Z)", text, flags=re.M | re.S)
    assert m, f"section {number} missing"
    return m.group(0)


def _table_codes(block):
    return re.findall(r"^\| `([A-Z][A-Z0-9_]*)` \|", block, flags=re.M)


def test_every_reason_and_flag_is_documented(text):
    s6 = section(text, 6)
    reasons_part, flags_part = s6.split("Flags are markers", 1)
    reasons = _table_codes(reasons_part)
    flags = _table_codes(flags_part)
    assert sorted(reasons) == sorted(R.REASONS)
    assert sorted(flags) == sorted(R.FLAGS)
    assert len(set(reasons)) == len(reasons) and len(set(flags)) == len(flags)
    for code in reasons:
        assert R.REASONS[code].status == "UNDEFINED"
    assert "UNDEFINED is never counted as ABSENT" in text


def test_every_se_method_is_documented(text):
    rows = re.findall(r"^\| `([A-Za-z0-9_]+)` \| ([A-Z, ]+) \| (.+) \|$",
                      section(text, 3), flags=re.M)
    assert sorted(name for name, _, _ in rows) == sorted(E.SE_METHODS)
    for name, principles, _ in rows:
        assert tuple(p.strip() for p in principles.split(",")) == (
            E.SE_METHODS[name].principles), name


def test_status_rule_block_is_documented(text):
    blocks = re.findall(r"```json\n(.*?)\n```", section(text, 5), flags=re.S)
    assert [json.loads(b) for b in blocks] == [E.StatusRule().to_dict()]


def test_order_of_the_checks_is_documented(text):
    part = section(text, 2).split("**Order of the checks**", 1)[1]
    items = re.findall(r"^\d+\. (.*?)(?=^\d+\. |\Z)", part, flags=re.M | re.S)
    codes = [re.search(r"`([A-Z_]+)[`:]", item).group(1) for item in items[:10]]
    assert codes == [code for _, code in E.CHECK_ORDER]


def test_implied_precision_table_matches_the_code(text):
    rows = re.findall(r"^\| (\d+) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \|$",
                      section(text, 2), flags=re.M)
    assert [int(r[0]) for r in rows] == [9, 12, 19, 199]
    for df, s_a, s_a_nas, s_p in rows:
        df = int(df)
        assert f"{T.absent_precision(df):.3f}" == s_a
        assert f"{T.absent_precision(df, members=2):.3f}" == s_a_nas
        assert f"{T.present_precision(df):.3f}" == s_p


def test_shipped_v3_protocols_and_hashes_are_quoted(text):
    rows = re.findall(r"^\| `([a-z0-9_]+\.json)` \| .* \| `([0-9a-f]{64})` \|$",
                      section(text, 5), flags=re.M)
    shipped = sorted(
        p.name for p in (REPO / "protocols" / "v2").glob("*.json")
        if json.loads(p.read_text(encoding="utf-8")).get("schema")
        == E.PROTOCOL_SCHEMA_V3)
    assert sorted(name for name, _ in rows) == shipped
    for name, quoted in rows:
        assert E.load_protocol(REPO / "protocols" / "v2" / name).hash == quoted


def test_schemas_and_versions_are_named(text):
    for token in (E.PROTOCOL_SCHEMA_V3, E.PROTOCOL_SCHEMA_V2, E.STATUS_RULE_VERSION,
                  T.PRECISION_SCHEMA, T.ANCHORS_SCHEMA, "mpc-bench-result/3"):
        assert token in text, token
