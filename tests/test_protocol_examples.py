"""
``protocols/mpc_default.json`` as the command-line default for empirical
data; the opt-in ``protocols/mpc_behavioural_ram.json`` (RAM declared on its
behavioural channel only; never a default); and the example derived
protocols with a declared NAS hub (``protocols/examples/``, derived from the
default by ``scripts/build_example_protocols.py``, which can also derive them
from the behavioural-RAM protocol).
"""
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import run_pipeline
import scripts.build_example_protocols as bep
from impact_pipeline import evidence as E
from impact_pipeline import mpc_metrics as mm

REPO = Path(__file__).resolve().parents[1]
PROTOCOLS = REPO / "protocols"
DEFAULT = PROTOCOLS / "mpc_default.json"
BRAM = PROTOCOLS / "mpc_behavioural_ram.json"
EXAMPLES = PROTOCOLS / "examples"
# the default protocol: its evidence.Protocol hash and the SHA-256 of the
# file's bytes
DEFAULT_HASH = "0fbfe90dc00e5f703f2fac60207f1399066b59c4317fb30be5a337cf051c842b"
DEFAULT_FILE_SHA256 = (
    "724750b2aa4f12492d771271107ba7919ccafa57983bdefc72bbbf1fe87dc6d6")
# the opt-in behavioural-RAM protocol
BRAM_HASH = "328101d46c46646ea2075f6daa6af75ac484cca40fed8e60fb377de0d339d9e0"


# --------------------------------------------------------------------------
# the default and the opt-in behavioural-RAM protocol
# --------------------------------------------------------------------------
def test_default_protocol_bytes_and_hash():
    assert hashlib.sha256(DEFAULT.read_bytes()).hexdigest() == DEFAULT_FILE_SHA256
    default = E.Protocol.from_json(DEFAULT)
    assert default.hash == DEFAULT_HASH and default.name == "mpc-default"
    assert set(default.channels_for("RAM")) == {
        "default", "perturbational", "endogenous"}


def test_behavioural_ram_differs_from_the_default_only_in_the_ram_channels():
    default, bram = E.Protocol.from_json(DEFAULT), E.Protocol.from_json(BRAM)
    assert bram.hash == BRAM_HASH
    assert json.loads(BRAM.read_text()) == bram.to_dict()  # canonical file
    assert bram.name == "mpc-behavioural-ram"
    assert "default" not in bram.name and "default" not in BRAM.name
    assert bram.channels_for("RAM") == ("default",)
    a, b = default.to_dict(), bram.to_dict()
    for d in (a, b):
        d["channels"].pop("RAM")
        d.pop("name")
    assert a == b


def test_behavioural_ram_declares_only_implemented_ram_channels():
    bram = E.Protocol.from_json(BRAM)
    implemented = {"default", *mm.RAM_IMPLEMENTED_CHANNELS}
    assert set(bram.channels_for("RAM")) <= implemented
    for p in E.PRINCIPLES:
        assert bram.channels_for(p) == ("default",)


def test_ram_can_be_absent_under_behavioural_ram_but_not_under_the_default():
    """Under the default a credibly absent behavioural channel can never
    make RAM ABSENT (its unimplemented declared channels are UNDEFINED); under
    the opt-in behavioural-RAM protocol it does."""
    def evidence(proto):
        ev = {q: [E.ComponentEvidence(q, 1.0, 0.0, 0.1, se=0.05, n_null=30,
                                      reference=1.0, reference_scale="excess",
                                      null_family=proto.null_families.get(q))]
              for q in E.PRINCIPLES}
        ev["RAM"] = [E.ComponentEvidence(
            "RAM", -0.5, 0.0, 0.1, se=0.05, n_null=30, reference=1.0,
            reference_scale="excess", null_family="onset_jitter")]
        ev["RAM"] += [E.ComponentEvidence("RAM", float("nan"), channel=ch,
                                          defined=False, reason="NOT_IMPLEMENTED")
                      for ch in (proto.channels_for("RAM") or ()) if ch != "default"]
        return ev

    default, bram = E.Protocol.from_json(DEFAULT), E.Protocol.from_json(BRAM)
    v = E.mpc_verdict(evidence(default), default)
    assert v.channels["RAM"]["default"] == E.ComponentStatus.ABSENT
    assert v.component_status["RAM"] == E.ComponentStatus.UNDEFINED
    assert v.verdict != E.Verdict.EXCLUDED
    v = E.mpc_verdict(evidence(bram), bram)
    assert v.component_status["RAM"] == E.ComponentStatus.ABSENT
    assert v.verdict == E.Verdict.EXCLUDED and "ABSENT:RAM" in v.reasons


def test_protocols_readme_documents_the_default_and_the_opt_in():
    readme = (PROTOCOLS / "README.md").read_text(encoding="utf-8")
    rows = {m.group(1): m.group(0) for m in re.finditer(
        r"^\| `([^`]+\.json)` \|.*$", readme, flags=re.M)}
    assert DEFAULT_HASH in rows["mpc_default.json"]
    assert "default" in rows["mpc_default.json"]
    bram_row = rows["mpc_behavioural_ram.json"]
    assert BRAM_HASH in bram_row and "opt-in" in bram_row
    # nothing calls the behavioural-RAM protocol a default
    assert "default" not in bram_row.lower()
    assert "## `mpc_behavioural_ram.json` (opt-in)" in readme
    text = " ".join(readme.split())
    assert "behavioural-RAM results" in text
    assert "not absence of responsiveness" in text
    assert "NO_DECLARED_WORKSPACE" in readme


def test_no_document_calls_the_behavioural_ram_protocol_a_default():
    docs = ["README.md", "CHANGELOG.md", "docs/metrics.md", "docs/ARCHITECTURE.md",
            "docs/HLRS_HUNTER_RUNBOOK.md", "scripts/hunter/README.md",
            "protocols/README.md", "protocols/examples/README.md"]
    # "default ...: mpc_behavioural_ram" / "mpc_behavioural_ram is the default"
    pattern = re.compile(
        r"default( protocol| for [a-z ]+)?( is|:)? `?(protocols/)?"
        r"mpc_behavioural_ram"
        r"|mpc_behavioural_ram(\.json)?`? (is|as|becomes) (the |a |an )?"
        r"(empirical |command-line )?default",
        flags=re.I)
    for doc in docs:
        text = " ".join((REPO / doc).read_text(encoding="utf-8").split())
        assert not pattern.search(text), (doc, pattern.search(text).group(0))


# --------------------------------------------------------------------------
# default protocol of command-line runs
# --------------------------------------------------------------------------
def test_cli_default_protocol_resolution():
    r = run_pipeline.resolve_cli_protocol
    assert Path(run_pipeline.DEFAULT_EMPIRICAL_PROTOCOL) == DEFAULT
    assert E.Protocol.from_json(run_pipeline.DEFAULT_EMPIRICAL_PROTOCOL).hash == (
        DEFAULT_HASH)
    assert r(None, "real") == str(DEFAULT)
    assert r(None) == str(DEFAULT)
    assert r(None, "study") == str(DEFAULT)
    # dummy/synthetic data and an explicit necessity set keep the flag-built one
    assert r(None, "dummy") is None
    assert r(None, "real", necessity_set="NAS,IIM") is None
    # explicit choices win: the behavioural-RAM protocol only when asked for
    assert r(str(BRAM), "real") == str(BRAM)
    assert r(str(BRAM), "dummy") == str(BRAM)
    assert r(str(DEFAULT), "real") == str(DEFAULT)
    assert r("none", "real") is None and r("FLAGS", "real") is None


@pytest.mark.parametrize("extra, expected", [
    ([], str(DEFAULT)),
    (["--protocol", str(BRAM)], str(BRAM)),
    (["--data-origin", "dummy"], "default protocol from flags"),
])
def test_cli_run_uses_the_default_protocol(tmp_path, extra, expected):
    """The command line resolves the protocol before anything else runs (the
    unknown dataset then stops the run right after the protocol is logged)."""
    import os

    env = {**os.environ, "IMPACT_SKIP_ENV_CHECK": "1", "OMP_NUM_THREADS": "1"}
    proc = subprocess.run(
        [sys.executable, str(REPO / "run_pipeline.py"), "--out-dir",
         str(tmp_path / "out"), "--dataset-id", "no_such_dataset", *extra],
        capture_output=True, text=True, timeout=300, env=env,
    )
    assert proc.returncode != 0 and "no_such_dataset" in proc.stderr
    line = next(ln for ln in proc.stderr.splitlines() if "MPC protocol:" in ln)
    assert expected in line
    if expected == str(DEFAULT):
        assert DEFAULT_HASH[:12] in line
    if expected == str(BRAM):
        assert BRAM_HASH[:12] in line


def test_cli_help_names_the_default_protocol():
    text = subprocess.run(
        [sys.executable, str(REPO / "run_pipeline.py"), "--help"],
        capture_output=True, text=True, timeout=180,
    ).stdout
    text = " ".join(text.split())
    assert "Default for real data: protocols/mpc_default.json" in text


# --------------------------------------------------------------------------
# examples
# --------------------------------------------------------------------------
EXAMPLE_FILES = (bep.SCHAEFER_FILE, bep.EEG64_FILE)
# evidence.Protocol hashes of the shipped examples (derived from the default)
EXAMPLE_HASHES = {
    bep.SCHAEFER_FILE:
        "a83ecf5949718c540eca08b110a7bf2d5c47d910457a8e889c5c5a0907b11e98",
    bep.EEG64_FILE:
        "5aca34b77744082a51d4c4fa72860181fa5c6dd7e49dbee8f03aa01076d8be38",
}


def test_shipped_examples_are_a_fresh_build_from_the_default():
    assert Path(bep.BASE_PROTOCOL) == DEFAULT
    assert bep.example_names(DEFAULT) == {
        "schaefer_file": bep.SCHAEFER_FILE, "schaefer_name": bep.SCHAEFER_NAME,
        "eeg_file": bep.EEG64_FILE, "eeg_name": bep.EEG64_NAME,
        "eeg_custom_name": bep.EEG_CUSTOM_NAME}
    assert bep.SCHAEFER_FILE == "mpc_default_schaefer400_7networks_hub.json"
    assert bep.EEG64_FILE == "mpc_default_eeg64_hub.json"
    assert bep.main(["--check"]) == 0
    built = bep.build_examples()
    assert sorted(built) == sorted(
        [*EXAMPLE_FILES, *(n.replace(".json", ".derivation.json")
                           for n in EXAMPLE_FILES)])
    for name, payload in built.items():
        assert json.loads((EXAMPLES / name).read_text(encoding="utf-8")) == payload
    # the directory holds exactly this build (no leftovers)
    assert sorted(p.name for p in EXAMPLES.glob("*.json")) == sorted(built)


def test_check_reports_stale_example_files(tmp_path, monkeypatch):
    shipped = tmp_path / "examples"
    shutil.copytree(EXAMPLES, shipped)
    monkeypatch.setattr(bep, "EXAMPLES_DIR", shipped)
    assert bep.main(["--check"]) == 0
    (shipped / "mpc_default_old_eeg64_hub.json").write_text("{}")
    assert bep.main(["--check"]) == 1
    (shipped / "mpc_default_old_eeg64_hub.json").unlink()
    payload = json.loads((shipped / bep.EEG64_FILE).read_text())
    payload["name"] = "tampered"
    (shipped / bep.EEG64_FILE).write_text(json.dumps(payload))
    assert bep.main(["--check"]) == 1


@pytest.mark.parametrize("name", EXAMPLE_FILES)
def test_examples_differ_from_the_default_only_in_the_hub_and_name(name):
    proto = E.Protocol.from_json(EXAMPLES / name)
    assert json.loads((EXAMPLES / name).read_text()) == proto.to_dict()
    assert proto.name.startswith("EXAMPLE-mpc-default-")
    assert proto.hash == EXAMPLE_HASHES[name]
    base = E.Protocol.from_json(DEFAULT)
    # RAM keeps the unimplemented channels of the default protocol
    assert proto.channels_for("RAM") == base.channels_for("RAM")
    a, b = base.to_dict(), proto.to_dict()
    hub = b["estimators"]["NAS"].pop("workspace_nodes")
    assert hub == sorted(set(hub)) and hub
    a.pop("name"), b.pop("name")
    assert a == b
    side_path = EXAMPLES / name.replace(".json", ".derivation.json")
    side = json.loads(side_path.read_text())
    assert side["protocol_hash"] == proto.hash
    assert side["base_protocol"] == "protocols/mpc_default.json"
    assert side["base_protocol_hash"] == base.hash == DEFAULT_HASH
    assert side["base_protocol_name"] == "mpc-default"
    assert side["base_ram_channels"] == ["default", "perturbational", "endogenous"]
    assert "ram_channel_caveat" not in side
    assert side["workspace_nodes"] == hub and side["n_hub"] == len(hub)
    assert side["status"].startswith("EXAMPLE - not preregistered")


def test_examples_can_be_derived_from_the_behavioural_ram_protocol(tmp_path):
    """--base derives the same hubs from the opt-in behavioural-RAM protocol,
    under names taken from it (never 'default'), with the interpretation
    caveat in the sidecar; such a build needs --out-dir and --check verifies
    it."""
    # never into the shipped directory, with or without --check
    assert bep.main(["--base", str(BRAM)]) == 2
    assert bep.main(["--base", str(BRAM), "--check"]) == 2
    out = tmp_path / "bram"
    assert bep.main(["--base", str(BRAM), "--out-dir", str(out)]) == 0
    assert bep.main(["--base", str(BRAM), "--out-dir", str(out), "--check"]) == 0
    names = bep.example_names(BRAM)
    assert names["schaefer_file"] == (
        "mpc_behavioural_ram_schaefer400_7networks_hub.json")
    assert names["eeg_name"] == (
        "EXAMPLE-mpc-behavioural-ram-eeg64-frontoparietal-hub")
    assert sorted(p.name for p in out.glob("*.json")) == sorted(
        [names["schaefer_file"], names["eeg_file"],
         names["schaefer_file"].replace(".json", ".derivation.json"),
         names["eeg_file"].replace(".json", ".derivation.json")])
    base = E.Protocol.from_json(BRAM)
    for key, shipped in (("schaefer_file", bep.SCHAEFER_FILE),
                         ("eeg_file", bep.EEG64_FILE)):
        proto = E.Protocol.from_json(out / names[key])
        assert "default" not in proto.name
        assert proto.channels_for("RAM") == ("default",)
        # same hub as the shipped example derived from the default
        assert proto.estimator_options("NAS") == E.Protocol.from_json(
            EXAMPLES / shipped).estimator_options("NAS")
        a, b = base.to_dict(), proto.to_dict()
        b["estimators"]["NAS"].pop("workspace_nodes")
        a.pop("name"), b.pop("name")
        assert a == b
        side = json.loads((out / names[key].replace(
            ".json", ".derivation.json")).read_text())
        assert side["base_protocol_hash"] == BRAM_HASH
        assert side["base_ram_channels"] == ["default"]
        caveat = " ".join(side["ram_channel_caveat"].split())
        assert "not absence of responsiveness" in caveat
        assert "behavioural-RAM results" in caveat and "preregister" in caveat
    # a modified file fails the check
    (out / names["eeg_file"]).write_text("{}")
    assert bep.main(["--base", str(BRAM), "--out-dir", str(out), "--check"]) == 1


def test_schaefer_example_hub_is_the_declared_networks():
    proto = E.Protocol.from_json(EXAMPLES / bep.SCHAEFER_FILE)
    hub = set(proto.estimator_options("NAS")["workspace_nodes"])
    rows = bep.read_schaefer_order()
    assert len(rows) == 400
    for i, r in enumerate(rows):
        assert r["value"] == i + 1
        in_hub = r["network"] in {"Cont", "DorsAttn", "SalVentAttn"}
        assert (i in hub) == in_hub, r["name"]
    assert len(hub) == 145
    side = json.loads((EXAMPLES / bep.SCHAEFER_FILE.replace(
        ".json", ".derivation.json")).read_text())
    assert side["source"]["sha256"] == bep._sha256(bep.SCHAEFER_ORDER)
    assert side["hub_labels"] == [rows[i]["name"] for i in sorted(hub)]


def test_eeg_example_hub_is_in_sorted_channel_order():
    proto = E.Protocol.from_json(EXAMPLES / bep.EEG64_FILE)
    hub = proto.estimator_options("NAS")["workspace_nodes"]
    order = sorted(bep.EEG64_MONTAGE)
    assert len(order) == 64 and len(set(order)) == 64
    assert {order[i] for i in hub} == set(bep.EEG64_HUB)
    # the pipeline's EEG preprocessing writes the channels in sorted order
    src = (REPO / "src" / "impact_pipeline" / "preprocessing_eeg.py").read_text()
    assert "common_channels = sorted(" in src and "raw.pick(common_channels)" in src


def test_example_builder_refuses_bad_hubs(tmp_path):
    with pytest.raises(ValueError, match="not in the montage"):
        bep.eeg_hub(bep.EEG64_MONTAGE, ["Fz", "XYZ"])
    with pytest.raises(ValueError, match="unknown networks"):
        bep.schaefer_hub(["Nope"])
    with pytest.raises(ValueError, match="capacity"):
        legacy = E.Protocol.from_json(DEFAULT).to_dict()
        legacy["estimators"]["NAS"] = {"mode": "legacy"}
        bep.derived_protocol(legacy, [0], "x")
    # a custom montage needs an output directory (the shipped files stay)
    assert bep.main(["--eeg-hub", "Fz,Pz"]) == 2
    chans = tmp_path / "channels.tsv"
    chans.write_text("name\ttype\nFz\tEEG\nPz\tEEG\nCz\tEEG\nEOG1\tEOG\n")
    assert bep.main(["--eeg-channels", str(chans), "--eeg-hub", "Fz,Pz",
                     "--out-dir", str(tmp_path / "out")]) == 0
    side = json.loads((tmp_path / "out" / bep.EEG64_FILE.replace(
        ".json", ".derivation.json")).read_text())
    assert side["channel_order"] == ["Cz", "Fz", "Pz"]
    assert side["workspace_nodes"] == [1, 2]
    # a custom build is never named like the shipped example
    assert side["protocol_name"] == bep.EEG_CUSTOM_NAME != bep.EEG64_NAME
    assert side["hub_rule"] == "declared sensors Fz, Pz"


def test_channels_tsv_is_read_as_the_eeg_preprocessing_reads_it(tmp_path):
    """A BIDS channels.tsv as OpenNeuro ships it (ds005620): a UTF-8 BOM,
    ocular channels typed EEG, a bad channel. The node set must be the one
    preprocessing_eeg keeps, or the derived indices name other channels."""
    from impact_pipeline.preprocessing_eeg import classify_eeg_channels

    rows = [("Fz", "EEG", "good"), ("VEOG", "EEG", "good"), ("Pz", "EEG", "good"),
            ("HEOG", "EEG", "good"), ("Cz", "EEG", "bad"), ("Oz", "EEG", "good"),
            ("ECG", "ECG", "good"), ("F3", "EEG", "good")]
    chans = tmp_path / "sub-01_channels.tsv"
    chans.write_bytes(("﻿name\ttype\tunits\tstatus\n" + "".join(
        f"{n}\t{t}\tµV\t{s}\n" for n, t, s in rows)).encode("utf-8"))
    names = bep._read_channels(chans)
    assert names == ["Fz", "Pz", "Oz", "F3"]
    expected = classify_eeg_channels(
        [r[0] for r in rows], ["eeg"] * len(rows),
        {n: {"type": t, "status": s} for n, t, s in rows})["eeg"]
    assert names == expected
    nodes, labels, order = bep.eeg_hub(names, ["Fz", "Pz"])
    assert order == ["F3", "Fz", "Oz", "Pz"] and nodes == [1, 3]
    assert bep.main(["--eeg-channels", str(chans), "--eeg-hub", "Fz,Pz",
                     "--out-dir", str(tmp_path / "out")]) == 0
    # a plain list (one name per line, BOM tolerated) is the node set as given
    plain = tmp_path / "channels.txt"
    plain.write_bytes("﻿Fz\nPz\nVEOG\n".encode("utf-8"))
    assert bep._read_channels(plain) == ["Fz", "Pz", "VEOG"]


def test_eeg_example_indices_do_not_carry_over_to_another_layout():
    """The shipped BioSemi-64 indices are in range on a 62-channel 10-10
    layout (ds005620-like: TP9/TP10, no AF7/AF8/P9/P10) but name other
    channels, which is why the README and the derivation say so."""
    shipped = E.Protocol.from_json(EXAMPLES / bep.EEG64_FILE).estimator_options(
        "NAS")["workspace_nodes"]
    other = sorted((set(bep.EEG64_MONTAGE) - {"AF7", "AF8", "P9", "P10"})
                   | {"TP9", "TP10"})
    assert len(other) == 62 and max(shipped) < 62
    assert {other[i] for i in shipped} != set(bep.EEG64_HUB)
    nodes, _, _ = bep.eeg_hub(other, bep.EEG64_HUB)
    assert nodes != shipped and {other[i] for i in nodes} == set(bep.EEG64_HUB)
    text = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    assert "It does not hold for ds005620" in text


def test_examples_readme_says_they_must_be_preregistered():
    text = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    assert "EXAMPLES, NOT PREREGISTERED" in text
    assert "must be preregistered" in text
    for name in EXAMPLE_FILES:
        assert name in text
    flat = " ".join(text.split())
    assert "mpc_default.json" in flat
    # the behavioural-RAM derivation is documented as opt-in, with its caveat
    assert "--base protocols/mpc_behavioural_ram.json" in flat
    assert "not absence of responsiveness" in flat
