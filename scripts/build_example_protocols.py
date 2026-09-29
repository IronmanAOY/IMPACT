#!/usr/bin/env python
"""
Example derived protocols with a declared NAS hub (``protocols/examples/``).

NAS ``mode='capacity'`` needs a declared hub (``workspace_nodes``). The
default empirical protocol ``protocols/mpc_default_v1.json`` (the
preregistered protocol) declares none, so NAS is UNDEFINED there
(``UNDEFINED:NAS:NO_DECLARED_WORKSPACE``). A derived protocol declares the hub
for one grain. This script writes two **examples** of such protocols (the base
protocol with ``estimators.NAS.workspace_nodes`` and a new name, nothing else
changed) and a ``.derivation.json`` sidecar per example that records how the
hub was obtained:

- ``mpc_default_v1_schaefer400_7networks_hub.json``: Schaefer-2018 400
  parcels / 7 networks (the pipeline's ``schaefer400`` grain). Hub = the
  parcels whose network label in
  ``atlases/schaefer_2018/Schaefer2018_400Parcels_7Networks_order.txt`` is
  one of ``Cont``, ``DorsAttn``, ``SalVentAttn`` (fronto-parietal control,
  dorsal attention, salience/ventral attention). Node index = label value - 1,
  the column order of the pipeline's ``schaefer400`` time series
  (``NiftiLabelsMasker`` on the label image, all 400 labels present).
- ``mpc_default_v1_eeg64_hub.json``: a 64-channel 10-20 (10-10 labels)
  montage, the BioSemi-64 layout (MNE ``biosemi64``). The pipeline's EEG
  preprocessing keeps the channels common to a subject's runs **in sorted
  order** (``preprocessing_eeg``: ``raw.pick(sorted(common))``), so node
  index = position of the channel name in ``sorted(montage)``. Hub = the
  fronto-parietal sensors F3, F1, Fz, F2, F4, P3, P1, Pz, P2, P4. Not valid
  for the pipeline's EEG dataset ds005620 (BrainVision 10-10 layout with
  TP9/TP10 and VEOG/HEOG, 62 scalp channels kept): derive its hub from its
  channels.tsv with ``--eeg-channels`` (classified as the preprocessing does).

These hub choices are illustrations of the mechanics, not recommendations:
a hub is a substantive measurement decision that must be preregistered
(with its derivation) before any confirmatory use, and scalp-EEG NAS needs
the forward-model validation of the applicability registry first (volume
conduction mixes hub and periphery). See ``protocols/examples/README.md``.

``--base`` derives the same examples from another base protocol, e.g. the
opt-in ``protocols/mpc_behavioural_ram_v1.json`` (RAM declared on its
behavioural channel only, so RAM can be ABSENT on behavioural evidence alone:
a substantive choice that has to be preregistered, and its results are
behavioural-RAM results). File and protocol names are taken from the base
protocol's name (``mpc_behavioural_ram_v1_*``, ``EXAMPLE-mpc-behavioural-ram-v1-*``);
a build from another base needs ``--out-dir`` (``protocols/examples/`` holds
the examples derived from the default protocol only).

Usage::

    python scripts/build_example_protocols.py            # (re)write the examples
    python scripts/build_example_protocols.py --check    # verify the shipped files
    # another montage / hub (writes to --out-dir, never over the shipped ones
    # unless asked):
    python scripts/build_example_protocols.py --eeg-channels channels.tsv \\
        --eeg-hub F3,Fz,F4,P3,Pz,P4 --out-dir my_protocols
    # derived from the opt-in behavioural-RAM protocol (and verified):
    python scripts/build_example_protocols.py \\
        --base protocols/mpc_behavioural_ram_v1.json --out-dir my_protocols
    python scripts/build_example_protocols.py \\
        --base protocols/mpc_behavioural_ram_v1.json --out-dir my_protocols --check
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from impact_pipeline import evidence as E  # noqa: E402

# the preregistered default protocol (hash 383eb310..., unchanged since the
# freeze); the shipped examples are derived from it
BASE_PROTOCOL = REPO_ROOT / "protocols" / "mpc_default_v1.json"
# opt-in alternative base (--base): RAM declared on its behavioural channel only
BEHAVIOURAL_RAM_PROTOCOL = REPO_ROOT / "protocols" / "mpc_behavioural_ram_v1.json"
EXAMPLES_DIR = REPO_ROOT / "protocols" / "examples"
SCHAEFER_ORDER = (
    REPO_ROOT / "atlases" / "schaefer_2018"
    / "Schaefer2018_400Parcels_7Networks_order.txt"
)
SCHAEFER_HUB_NETWORKS = ("Cont", "DorsAttn", "SalVentAttn")
SCHAEFER_N_PARCELS = 400
SCHAEFER_FILE = "mpc_default_v1_schaefer400_7networks_hub.json"
SCHAEFER_NAME = "EXAMPLE-mpc-default-v1-schaefer400-7networks-hub"
# BioSemi-64 channel labels (10-20 system, 10-10 positions; MNE montage
# 'biosemi64', in the manufacturer's order).
EEG64_MONTAGE = (
    "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1",
    "C1", "C3", "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7",
    "P9", "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz", "Fpz", "Fp2",
    "AF8", "AF4", "AFz", "Fz", "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4",
    "FC2", "FCz", "Cz", "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2",
    "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
)
EEG64_HUB = ("F3", "F1", "Fz", "F2", "F4", "P3", "P1", "Pz", "P2", "P4")
EEG64_FILE = "mpc_default_v1_eeg64_hub.json"
EEG64_NAME = "EXAMPLE-mpc-default-v1-eeg64-frontoparietal-hub"
# name of a build with another montage or hub (--eeg-channels / --eeg-hub), so
# that it is never mistaken for the shipped example (its hash differs anyway)
EEG_CUSTOM_NAME = "EXAMPLE-mpc-default-v1-eeg-custom-hub"
EXAMPLE_STATUS = (
    "EXAMPLE - not preregistered. The hub is a substantive measurement choice: "
    "preregister it (with this derivation) before any confirmatory use."
)


def example_names(base_path: Path = BASE_PROTOCOL) -> Dict[str, str]:
    """
    File and protocol names of the examples derived from ``base_path``, from
    the base protocol's name (its file stem when unnamed): ``mpc-default-v1``
    gives the shipped names (:data:`SCHAEFER_FILE`, :data:`EEG64_NAME`, ...),
    ``mpc-behavioural-ram-v1`` gives ``mpc_behavioural_ram_v1_*.json`` and
    ``EXAMPLE-mpc-behavioural-ram-v1-*``.
    """
    base = E.Protocol.from_json(base_path)
    label = str(base.name or Path(base_path).stem).strip()
    label = "-".join(label.replace("_", "-").split())
    prefix = label.replace("-", "_").replace(".", "_")
    return {
        "schaefer_file": f"{prefix}_schaefer400_7networks_hub.json",
        "schaefer_name": f"EXAMPLE-{label}-schaefer400-7networks-hub",
        "eeg_file": f"{prefix}_eeg64_hub.json",
        "eeg_name": f"EXAMPLE-{label}-eeg64-frontoparietal-hub",
        "eeg_custom_name": f"EXAMPLE-{label}-eeg-custom-hub",
    }


def _ram_channel_note(base) -> Optional[str]:
    """
    Interpretation note of a base protocol whose RAM declares only implemented
    channels (e.g. mpc_behavioural_ram_v1.json): RAM can then be ABSENT on
    those measured channels alone. None for the default protocol (RAM keeps
    its unimplemented channels, so behavioural non-response cannot exclude).
    """
    from impact_pipeline.mpc_metrics import RAM_IMPLEMENTED_CHANNELS

    chans = base.channels_for("RAM")
    if "RAM" not in base.necessity_set or not chans:
        return None
    if set(chans) - {"default", *RAM_IMPLEMENTED_CHANNELS}:
        return None
    return (
        f"RAM declares only implemented channels {list(chans)}: RAM can be "
        "ABSENT when these measured channels are, so an ABSENT RAM means no "
        "responsiveness-and-adaptation above the null in the measured "
        "channels ('default': the recorded behaviour), not absence of "
        "responsiveness (covert, perturbational and endogenous responsiveness "
        "are unmeasured). Opt-in only: a substantive choice to preregister; "
        "report the results as behavioural-RAM results (protocols/README.md)."
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def read_schaefer_order(path: Path = SCHAEFER_ORDER) -> List[dict]:
    """
    Rows of a Schaefer ``*_order.txt`` label table (label value, name,
    hemisphere, network), sorted by label value. Refuses gaps or duplicates,
    since the node index is ``label value - 1``.
    """
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        value, name = int(parts[0]), parts[1].strip()
        # e.g. 7Networks_LH_Vis_1 -> hemisphere LH, network Vis
        fields = name.split("_")
        if len(fields) < 4:
            raise ValueError(f"unexpected Schaefer label name {name!r}")
        rows.append({"value": value, "name": name, "hemisphere": fields[1],
                     "network": fields[2]})
    rows.sort(key=lambda r: r["value"])
    values = [r["value"] for r in rows]
    if values != list(range(1, len(rows) + 1)):
        raise ValueError("Schaefer label values must be 1..n without gaps")
    return rows


def schaefer_hub(networks: Sequence[str] = SCHAEFER_HUB_NETWORKS,
                 path: Path = SCHAEFER_ORDER) -> tuple:
    """``(workspace_nodes, labels, n_nodes)`` of the parcels in ``networks``."""
    rows = read_schaefer_order(path)
    known = sorted({r["network"] for r in rows})
    unknown = sorted(set(networks) - set(known))
    if unknown:
        raise ValueError(f"unknown networks {unknown}; the file has {known}")
    hub = [r for r in rows if r["network"] in set(networks)]
    return [r["value"] - 1 for r in hub], [r["name"] for r in hub], len(rows)


def eeg_node_order(channels: Sequence[str]) -> List[str]:
    """Node order of the pipeline's EEG time series: sorted channel names."""
    names = [str(c).strip() for c in channels]
    if len(set(names)) != len(names) or any(not n for n in names):
        raise ValueError("EEG channel names must be distinct and non-empty")
    return sorted(names)


def eeg_hub(channels: Sequence[str], hub: Sequence[str]) -> tuple:
    """``(workspace_nodes, hub labels in node order, node order)``."""
    order = eeg_node_order(channels)
    missing = sorted(set(hub) - set(order))
    if missing:
        raise ValueError(f"hub channels {missing} are not in the montage")
    idx = sorted(order.index(c) for c in set(hub))
    return idx, [order[i] for i in idx], order


def derived_protocol(base, workspace_nodes: Sequence[int], name: str):
    """``base`` (Protocol, dict or path) with ``estimators.NAS.workspace_nodes``
    and ``name`` replaced; everything else unchanged. NAS must be in
    ``mode='capacity'``."""
    proto = E.resolve_protocol(base)
    est = {p: proto.estimator_options(p) for p in proto.estimators}
    nas = dict(est.get("NAS") or {})
    if nas.get("mode") != "capacity":
        raise ValueError("the base protocol's NAS mode must be 'capacity'")
    hub = sorted(int(i) for i in workspace_nodes)
    if not hub or len(set(hub)) != len(hub) or hub[0] < 0:
        raise ValueError("workspace_nodes must be distinct non-negative indices")
    nas["workspace_nodes"] = hub
    est["NAS"] = nas
    return proto.replace(estimators=est, name=str(name))


def _derivation(proto, file_name, base_path, extra) -> dict:
    base = E.Protocol.from_json(base_path)
    out = {
        "status": EXAMPLE_STATUS,
        "protocol_file": file_name,
        "protocol_name": proto.name,
        "protocol_hash": proto.hash,
        "base_protocol": _rel(base_path),
        "base_protocol_name": base.name,
        "base_protocol_hash": base.hash,
        "base_ram_channels": list(base.channels_for("RAM") or ()),
        "changes_from_base": ["estimators.NAS.workspace_nodes", "name"],
        "generator": "scripts/build_example_protocols.py",
    }
    note = _ram_channel_note(base)
    if note is not None:
        out["ram_channel_caveat"] = note
    out.update(extra)
    return out


def build_examples(base_path: Path = BASE_PROTOCOL,
                   schaefer_order: Path = SCHAEFER_ORDER,
                   eeg_channels: Sequence[str] = EEG64_MONTAGE,
                   eeg_hub_channels: Sequence[str] = EEG64_HUB,
                   eeg_montage_label: str = "BioSemi-64 (MNE 'biosemi64')",
                   eeg_name: Optional[str] = None,
                   ) -> Dict[str, dict]:
    """``{file name: JSON payload}`` of the example protocols and sidecars
    derived from ``base_path`` (names from :func:`example_names`; ``eeg_name``
    overrides the EEG protocol name)."""
    names = example_names(base_path)
    schaefer_file, eeg_file = names["schaefer_file"], names["eeg_file"]
    out = {}
    nodes, labels, n = schaefer_hub(SCHAEFER_HUB_NETWORKS, schaefer_order)
    if n != SCHAEFER_N_PARCELS:
        raise ValueError(f"expected {SCHAEFER_N_PARCELS} parcels, got {n}")
    proto = derived_protocol(base_path, nodes, names["schaefer_name"])
    out[schaefer_file] = proto.to_dict()
    out[schaefer_file.replace(".json", ".derivation.json")] = _derivation(
        proto, schaefer_file, base_path, {
            "grain": "schaefer400",
            "n_nodes": n,
            "node_order": (
                "label value - 1 (columns of the pipeline's schaefer400 time "
                "series: NiftiLabelsMasker on "
                "Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz "
                "with all 400 labels present)"
            ),
            "hub_rule": (
                "parcels whose 7-network label is one of "
                + ", ".join(SCHAEFER_HUB_NETWORKS)
            ),
            "hub_networks": list(SCHAEFER_HUB_NETWORKS),
            "source": {"file": _rel(schaefer_order),
                       "sha256": _sha256(schaefer_order)},
            "n_hub": len(nodes),
            "hub_labels": labels,
            "workspace_nodes": nodes,
        })
    nodes, labels, order = eeg_hub(eeg_channels, eeg_hub_channels)
    proto = derived_protocol(base_path, nodes, eeg_name or names["eeg_name"])
    out[eeg_file] = proto.to_dict()
    out[eeg_file.replace(".json", ".derivation.json")] = _derivation(
        proto, eeg_file, base_path, {
            "grain": "eeg64",
            "n_nodes": len(order),
            "montage": eeg_montage_label,
            "node_order": (
                "sorted channel names (the pipeline's EEG preprocessing picks "
                "the channels common to a subject's runs in sorted order); "
                "valid only for recordings whose kept scalp EEG channels are "
                "exactly channel_order; on another layout the indices can be "
                "in range but name other channels, e.g. ds005620 (62 scalp "
                "channels): derive its hub from its channels.tsv with "
                "--eeg-channels"
            ),
            "channel_order": order,
            "hub_rule": (
                "fronto-parietal sensors "
                if tuple(eeg_hub_channels) == EEG64_HUB else "declared sensors "
            ) + ", ".join(eeg_hub_channels),
            "n_hub": len(nodes),
            "hub_labels": labels,
            "workspace_nodes": nodes,
            "caveat": (
                "scalp EEG: volume conduction mixes hub and periphery; NAS on "
                "sensor data needs the forward-model validation of the "
                "applicability registry (eeg_like_forward) first"
            ),
        })
    return out


def _dump(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def _read_channels(path: Path) -> List[str]:
    """
    Channel names from a text file (one per line, taken as the node set) or a
    BIDS channels.tsv (``name`` column). A channels.tsv is classified as the
    pipeline's EEG preprocessing does
    (``preprocessing_eeg.classify_eeg_channels``): the ``type`` column (EEG
    when absent), names that denote EOG/EMG/ECG sensors excluded even when
    typed EEG (e.g. VEOG/HEOG in ds005620), ``status=bad`` excluded. A UTF-8
    byte-order mark is ignored (BIDS files often carry one).
    """
    text = Path(path).read_text(encoding="utf-8-sig")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"no channels in {path}")
    head = [c.strip() for c in lines[0].split("\t")]
    if "name" not in head:
        return [ln.strip() for ln in lines]
    from impact_pipeline.preprocessing_eeg import classify_eeg_channels

    names, types, bids = [], [], {}
    for row in csv.DictReader(io.StringIO("\n".join(lines)), delimiter="\t"):
        row = {str(k).strip(): v for k, v in row.items() if k is not None}
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        typ = str(row.get("type") or "").strip()
        names.append(name)
        types.append(typ.lower() or "eeg")
        bids[name] = {"type": typ,
                      "status": str(row.get("status") or "").strip().lower()}
    return classify_eeg_channels(names, types, bids)["eeg"]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="build_example_protocols.py",
        description=__doc__.split("\n\n")[0],
    )
    ap.add_argument("--check", action="store_true",
                    help="verify that the shipped examples equal a fresh build")
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default protocols/examples)")
    ap.add_argument("--base", default=str(BASE_PROTOCOL),
                    help="base protocol (default protocols/mpc_default_v1.json, "
                         "the preregistered default; e.g. "
                         "protocols/mpc_behavioural_ram_v1.json, opt-in: needs "
                         "--out-dir)")
    ap.add_argument("--eeg-channels", default=None,
                    help="channel list (one per line, or a BIDS channels.tsv) "
                         "instead of the BioSemi-64 labels")
    ap.add_argument("--eeg-hub", default=None,
                    help="comma-separated hub channels (default "
                         + ",".join(EEG64_HUB) + ")")
    return ap


def _same_file(a: Path, b: Path) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return False


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    base = Path(args.base)
    other_base = not _same_file(base, BASE_PROTOCOL)
    custom = args.eeg_channels is not None or args.eeg_hub is not None
    if (custom or other_base) and args.out_dir is None:
        print("a custom montage or hub, or another --base, needs --out-dir "
              "(protocols/examples/ holds the examples derived from the default "
              "protocol only; they are not overwritten)", file=sys.stderr)
        return 2
    channels = (EEG64_MONTAGE if args.eeg_channels is None
                else _read_channels(Path(args.eeg_channels)))
    hub = (EEG64_HUB if args.eeg_hub is None
           else tuple(c.strip() for c in args.eeg_hub.split(",") if c.strip()))
    files = build_examples(
        base, eeg_channels=channels, eeg_hub_channels=hub,
        eeg_montage_label=(str(args.eeg_channels) if args.eeg_channels
                           else "BioSemi-64 (MNE 'biosemi64')"),
        eeg_name=example_names(base)["eeg_custom_name"] if custom else None,
    )
    out_dir = Path(args.out_dir) if args.out_dir else EXAMPLES_DIR
    if args.check:
        bad = [name for name, payload in files.items()
               if not (out_dir / name).is_file()
               or json.loads((out_dir / name).read_text(encoding="utf-8")) != payload]
        for name in bad:
            print(f"differs from a fresh build: {out_dir / name}", file=sys.stderr)
        stale = []
        if _same_file(out_dir, EXAMPLES_DIR):
            # the shipped directory holds exactly this build (no stale examples,
            # e.g. of an earlier base protocol)
            stale = sorted(p.name for p in out_dir.glob("*.json")
                           if p.name not in files)
            for name in stale:
                print(f"not part of a fresh build: {out_dir / name}",
                      file=sys.stderr)
        return 1 if bad or stale else 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        (out_dir / name).write_text(_dump(payload), encoding="utf-8")
        print(out_dir / name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
