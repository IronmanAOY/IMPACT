#!/usr/bin/env python
"""
Protocol builder of MPC-Bench v2: the family protocols of the confirmatory
run, written reproducibly from a calibration-decisions file and the
development outputs of ``scripts/v2/dev_calibration.py``.

The builder applies the preregistered rules to the development outputs
(anchor validity and the specificity gate, ``N_anch``, the testability
gates with empirical ``pi0`` and ``pi1``, the concordance-route admission
rule, the SE twin calibration, the mechanism-on labels) and the decisions
that the rules leave to the calibration (the CD decisions: SE methods,
``N_min``, the paper-2 regime and the other declared parameters of
:data:`DECISIONS`). It takes no decision: a decision missing from the file
is *pending*, the code's provisional value is used, and the build is marked
not ready for the freeze. Same inputs, same bytes: no output depends on the
clock, the machine or the order in which files are found.

Outputs (``--out``, default ``protocols/v2/generated/``)
--------------------------------------------------------
``mpc_bench_v2_<key>.json``
    every protocol a record of the confirmatory run names (the key
    convention of :func:`impact_pipeline.v2.hypothesis_engine.protocol_file_name`):
    the family protocols of families A and C1 and of the RAM-only arm with
    their forms (anchors and ``N_anch`` from the development reference block
    900-939, SE methods and IIM settings of the decisions, the admitted
    concordance cells, the precision block of the testability gates); the
    held-out declarations ``A-P``, ``A-Q10``, ``A-Q25``, ``A-J`` and the
    null-calibration protocol ``A-none``, copies of the generated ``A-R``
    with their declaration (``A-none``: none); the forward views and the v1
    quadrant forms of the Hopf EEG views (their own anchors from the
    reference block at the held-out regime, released after the held-out
    predictions were committed); the family-B protocols
    ``B-<anchor>-<cut>-<null family>[-values]`` exactly as the family-B
    validation builds them, and ``B``, the family-B protocol that carries the
    family-B testability rows (no record names it). IIM's
    ``report_cut_modes`` is empty wherever no scoring reads the reported cut
    (:data:`NO_REPORTED_CUT`, the forward views). Every protocol is named
    ``mpc-bench-v2-<key>``.
``forward_anchors.json``
    anchor validity per (arm, view, principle), the ``--anchors`` input of
    ``scripts/v2/build_registry_v3.py`` (criterion FM0).
``testability_table.json``
    every testability row (also below the minimum run count) with its sources.
``mechanism_on.json``
    the CD-8 labels as a list of entries in the hypotheses-file format
    (usable as the evaluator's ``--mechanism-on`` on development records):
    the rule's entries (witness doses from the v2 system catalogue), then
    the dose-only entries of the cells the held-out rule leaves without
    development rows (C1 IIM).
``declared_dependencies.json``
    the CD-7 declared dependencies of HCv2-22 (iii).
``calibration_decisions.json``
    the decisions as applied: value, pending or decided, the rule's
    suggestion, the deviation note and the code-consistency check.
``calibration_evidence.json``
    what the rules computed (development numbers, flagged as such): anchors,
    SE calibration on the twins (CD-2 to CD-4), the IIM occupancy gate
    (CD-3), the concordance battery (CD-5), the HCv2-14 development
    expectation (CD-6), the off-target rates behind the declared
    dependencies (CD-7), the mechanism-on medians (CD-8), the adversaries'
    potency (CD-9), the replication power of every anchor (CD-11), the
    oracle checks (CD-13), the null-calibration RAM-PE definedness, the IIM
    grain cap and the CD-1 constants, and the rules' suggestion for every
    decision they inform.
``build_manifest.json``
    hashes of every input and output, ``freeze_ready`` and the blocking and
    downstream items.

Commands
--------
::

    python scripts/v2/build_protocols_v2.py template --out decisions.json
    python scripts/v2/build_protocols_v2.py build --decisions decisions.json \\
        [--dev-root outputs/mpcbench_v2/dev_calibration] [--out DIR]
        [--require-freeze-ready]
    python scripts/v2/build_protocols_v2.py check [--dir DIR]

``template`` writes a decisions file with every decision pending and, where
development outputs exist, the rule's suggestion beside it. ``build`` refuses
development outputs on seeds of 1000 or more, records on the smoke seeds
980-984, records of held-out conditions, held-out-regime anchors without a
logged release and IIM on the Hopf arm's v2 sensor views above G = 0 (the
held-out G sweep of the sensor pipeline); a reference block without its 40
positive-control or own-lesion seeds, a release whose predictions changed
since it was logged, development records run under another IIM macro-node
cap than the declared one, a pending decision or a decision that differs
from the code or, without a deviation note, from its rule keep the build
from being ready for the freeze. With ``--require-freeze-ready`` it writes
nothing unless the build is ready. A build replaces the files of an
earlier build in its directory and refuses a directory with protocol files
that no build wrote. ``check`` re-reads a build and verifies its manifest.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import sys
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from impact_pipeline import evidence_v2 as E  # noqa: E402
from impact_pipeline.bench import designs_v2 as D  # noqa: E402
from impact_pipeline.bench import run_bench_v2 as RB  # noqa: E402
from impact_pipeline.v2 import PRINCIPLES  # noqa: E402
from impact_pipeline.v2 import hypothesis_engine as HE  # noqa: E402
from impact_pipeline.v2 import reasons as R  # noqa: E402
from impact_pipeline.v2 import records as REC  # noqa: E402
from impact_pipeline.v2 import seeds as S  # noqa: E402
from impact_pipeline.v2 import testability as T  # noqa: E402
from scripts.v2 import dev_calibration as DC  # noqa: E402

BUILDER_VERSION = "mpc-bench-protocol-builder/1.0.0"
DECISIONS_SCHEMA = "mpc-bench-calibration-decisions/1"
EVIDENCE_SCHEMA = "mpc-bench-calibration-evidence/1"
MANIFEST_SCHEMA = "mpc-bench-protocol-build/1"
GENERATED_DIR = RB.GENERATED_DIR
MANIFEST = "build_manifest.json"
FORWARD_ANCHORS = "forward_anchors.json"
TESTABILITY = "testability_table.json"
MECHANISM_ON = "mechanism_on.json"
DEPENDENCIES = "declared_dependencies.json"
DECISIONS_RECORD = "calibration_decisions.json"
EVIDENCE = "calibration_evidence.json"
STATUS_DRAFT, STATUS_FINAL = "draft", "final"
CUTOFF = E.DEFAULT_CUTOFF
Z, DELTA = float(CUTOFF[0]), float(CUTOFF[1])
ALPHA_A = E.DEFAULT_ALPHA_ABSENT
DEVELOPMENT_FLAG = HE.DEVELOPMENT_FLAG
BOOTSTRAP_SEED = 20261006  # resampling stream of the replication-power check
BOOTSTRAP_DRAWS = 2000

# --------------------------------------------------------------------------
# declared tables (one place each)
# --------------------------------------------------------------------------
# protocols that are copies of the generated A-R with another declaration
# (the held-out declarations and the null-calibration classification)
A_R = "A-R"
DERIVED_FROM_A_R = OrderedDict([("A-P", "P"), ("A-Q10", "Q10"), ("A-Q25", "Q25"),
                                ("A-J", "J"), ("A-none", "none")])
# protocols whose IIM cut is read by no scoring beside the primary one: the
# reported cut is not computed there (the forward views declare it already)
NO_REPORTED_CUT = tuple(DERIVED_FROM_A_R)
# the anchor designs: development reference block per family protocol
ANCHOR_DESIGNS = ("A_anchors", "C1_anchors", "RAM160_anchors")
FORWARD_ANCHOR_DESIGN = "forward_anchor_replication"
# the v1 quadrant comparator forms of the Hopf EEG views: their own anchors
# on the forward anchor condition (validity only), never their base view's
FORWARD_OWN_ANCHOR_FORMS = ("hopf-eeg64+iim_v1_quadrants",
                            "hopf-eeg64_noref+iim_v1_quadrants")
# the protocols whose anchor status a design's replication block tests
# (CD-11, HCv2-6): the family protocols, and every forward view and quadrant
# form with its own anchors (validity only, HCv2-6 (b))
PRIMARY_PROTOCOLS_OF_ANCHOR_DESIGN = {
    "A_anchors": ("A-R", "A-H"), "C1_anchors": ("C1-R", "C1-H"),
    "RAM160_anchors": ("A-RAM160",),
    FORWARD_ANCHOR_DESIGN: ("fwdA-eeg64", "fwdA-eeglow", "fwdA-source",
                            "fwdA_bold-bold", "fwdA_bold-source", "hopf-bold",
                            "hopf-eeg64", "hopf-eeg64_noref", "hopf-eeglow",
                            "hopf-mne_template", "hopf-source")
    + FORWARD_OWN_ANCHOR_FORMS}
VALIDITY_ONLY_ANCHOR_DESIGNS = (FORWARD_ANCHOR_DESIGN,)
VALID = "valid"  # the status of a valid anchor where only validity is tested
# the protocols of the gated parts (HCv2-14 (c, d), HCv2-22 (ii)) and the
# designs whose development rows enter the testability table
GATE_PROTOCOLS = ("A-R", "A-H", "C1-R", "C1-H", "A-RAM160")
TESTABILITY_DESIGNS = ("A_anchors", "C1_anchors", "RAM160_anchors", "A_witnesses",
                       "C1_witnesses", "RAM160")
TWIN_DESIGNS = ("A_twins", "C1_twins", "RAM160_twins")
ADVERSARY_DESIGNS = ("A_adversaries",)
# CD-9: an adversary is potent against an estimator when its paired ratio
# with PC_nominal on the same seed is >= 0.5 in >= 50 % of the development
# seeds or it is PRESENT in >= 50 % (reported, never used to select
# conditions)
POTENCY_C, POTENCY_SHARE = 0.5, 0.5
# the principle whose mechanism a knob switches (mechanism-on dose ratio)
OWN_KNOB = {"RAM": "eta", "PDI": "K", "NAS": "g_b", "IIM": "c_int", "SRPI": "e"}
SWEEP_DESIGNS = ("A_sweeps", "C1_sweeps")
FACTORIAL_DESIGNS = ("A_factorial", "C1_factorial")
WITNESS_DESIGNS = ("A_witnesses", "C1_witnesses")
MECHANISM_DESIGNS = SWEEP_DESIGNS + FACTORIAL_DESIGNS + WITNESS_DESIGNS
# HCv2-5 scores no patchwork: its mechanisms sit on separate bearers, so no
# witness row of it is labelled (unmatched rows are not on)
MECHANISM_EXCLUDED_GENERATORS = ("patchwork",)
# the family-principle cells whose development rows the held-out rule
# withholds on the mechanism-on designs (HO-4: IIM on C1). Their labels are
# the dose condition alone, placed after the rule entries.
DOSE_ONLY_MECHANISM_CELLS = (("C1", "IIM"),)
# substrate of a record's source view (the evidence item's substrate)
FAMILY_SUBSTRATE = {"A": "synthetic_rate", "C1": "stuart_landau"}
# the concordance-route admission rule (design 2.4 item 4, CD-5)
CONCORDANCE_MIN_RUNS = DC.CONCORDANCE_MIN_RUNS
FORWARD_SUBSTRATES = ("eeg_like_forward", "bold_like_forward")
# The admitted cell is the family-A battery's (A-R, source view, the
# declared access bearer). A protocol carries no route where the same cell
# key would name other runs: a PDI block with a declared access module (the
# mis-declared access form, another bearer) and the BOLD forward arm, whose
# slow-context agents and longer windows are another regime than the
# battery's even on the source view.
NO_CONCORDANCE_ROUTE_PREFIXES = ("fwdA_bold-",)
# the SE calibration rules (design 2.1 item 9, 2.2 item 5, 2.3 item 4)
KAPPA_BOUNDS = (0.8, 1.25)
KAPPA_LEVEL = 0.90
TAIL_MAX = 0.02
# HCv2-4: a class enters the calibration with c defined in >= 80 % of its
# twin sessions
DEFINED_SHARE_MIN = 0.8
# The statement of HCv2-4 (a), reported beside the CD-2 reading: the point in
# KAPPA_BOUNDS and the 90 % interval inside these wider bounds. With the
# development twins (5 networks x 7 sessions, df 30 per class) the 90 %
# interval spans a factor of about 1.54, so the interval lies inside
# KAPPA_BOUNDS only for a kappa point in about [0.97, 0.98].
KAPPA_WIDE_BOUNDS = (0.67, 1.5)
# family-B testability row (HCv2-11 (c))
FAMILY_B_GATE = {"family": "B", "principle": "IIM", "witness": "feedforward_star",
                 "kind": T.KIND_ABSENT, "min_coupling": 0.2, "cut": "directional"}


class BuildError(ValueError):
    """Inputs the builder refuses (development policy, malformed decisions)."""


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    raise TypeError(f"not JSON-serialisable: {type(obj).__name__}")


def _clean(obj):
    """Plain JSON: mappings with string keys, lists, finite floats (non-finite
    as null), ints, strings, booleans and null."""
    if isinstance(obj, Mapping):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        items = sorted(obj, key=str) if isinstance(obj, (set, frozenset)) else obj
        return [_clean(v) for v in items]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if obj is None or isinstance(obj, str):
        return obj
    raise TypeError(f"not JSON-serialisable: {type(obj).__name__}")


def _f(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# decisions
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DecisionSpec:
    """A calibration decision: its CD, meaning, the value the code runs with
    until it is taken (``provisional``), the validator, and where the code
    holds the same value (``code``: a description and a function returning
    the code's value), which the decision must equal."""

    name: str
    cd: str
    meaning: str
    provisional: Callable[[], object]
    validate: Callable[[object], object]
    code: Optional[Tuple[str, Callable[[], object]]] = None
    tier: str = "A"


def _one_of(allowed):
    allowed = tuple(allowed)

    def check(v):
        if v not in allowed:
            raise BuildError(f"must be one of {allowed}, got {v!r}")
        return v
    return check


def _int_ge(lo):
    def check(v):
        if isinstance(v, bool) or not isinstance(v, int) or v < lo:
            raise BuildError(f"must be an integer >= {lo}, got {v!r}")
        return int(v)
    return check


def _pairs(v):
    if not isinstance(v, (list, tuple)):
        raise BuildError("must be a list of [witness, principle] pairs")
    out = []
    for p in v:
        if (not isinstance(p, (list, tuple)) or len(p) != 2
                or p[1] not in PRINCIPLES or not isinstance(p[0], str)):
            raise BuildError(f"not a [witness, principle] pair: {p!r}")
        out.append([str(p[0]), str(p[1])])
    return sorted(out)


def _concordance_value(v):
    if v in ("rule", "none"):
        return v
    if isinstance(v, list):
        out = []
        for c in v:
            if not isinstance(c, Mapping):
                raise BuildError("an explicit concordance cell is an object")
            need = {"substrate", "observation_stage", "view", "bearer"}
            keys = need | {"absent", "present"}
            if set(c) - keys or need - set(c):
                raise BuildError(f"concordance cell keys are {sorted(keys)}")
            out.append({k: c[k] for k in sorted(c)})
        return out
    raise BuildError("must be 'rule', 'none' or a list of cells")


def _mechanism_value(v):
    if v in ("rule", "structural"):
        return v
    raise BuildError("must be 'rule' (the CD-8 table of the development "
                     "medians) or 'structural' (the draft of the hypotheses file)")


def _replication_value(v):
    """Each anchor design to true or false; the forward anchor design
    (its views' validity-only replication, decided once the released
    reference block gives its power) may be given beside them, as each
    forward arm to true or false: one anchor run serves every view of its
    arm, so the arm is what extends."""
    from impact_pipeline.bench.designs_v2 import anchors as AN

    need = set(AN.ANCHOR_DESIGNS)
    allowed = need | {FORWARD_ANCHOR_DESIGN}
    arms = set(AN.FORWARD_ARMS)
    fwd = v.get(FORWARD_ANCHOR_DESIGN, {}) if isinstance(v, Mapping) else None
    if (not isinstance(v, Mapping) or not need <= set(v) <= allowed
            or not all(isinstance(v[k], bool) for k in need)
            or not isinstance(fwd, Mapping)
            or (FORWARD_ANCHOR_DESIGN in v and set(fwd) != arms)
            or not all(isinstance(x, bool) for x in fwd.values())):
        raise BuildError(f"must map each of {sorted(need)} to true or false (and "
                         f"optionally {FORWARD_ANCHOR_DESIGN} each of "
                         f"{sorted(arms)} to true or false)")
    return {k: ({a: bool(fwd[a]) for a in sorted(fwd)} if k == FORWARD_ANCHOR_DESIGN
                else bool(v[k])) for k in sorted(v)}


def _regime_value(v):
    if not isinstance(v, Mapping):
        raise BuildError("the paper-2 regime is an object")
    keys = {"n_low", "reference", "tr_s", "duration_s", "declaration", "source"}
    if set(v) - keys or "n_low" not in v:
        raise BuildError(f"paper-2 regime keys are {sorted(keys)} (n_low required)")
    out = dict(v)
    out["n_low"] = _int_ge(8)(v["n_low"])
    return {k: out[k] for k in sorted(out)}


def _seed_value(v):
    if v is None:
        return None
    return _int_ge(0)(v)


def _hypotheses_spec() -> dict:
    return json.loads(HE.SPEC_PATH.read_text(encoding="utf-8"))


def _draft_dependencies():
    return _pairs((_hypotheses_spec().get("declared_dependencies") or {})
                  .get("pairs") or [])


def _nas():
    from impact_pipeline.v2 import nas_v3
    return nas_v3


def _iim():
    from impact_pipeline.v2 import iim_v5
    return iim_v5


def _ram():
    from impact_pipeline.v2 import ram_v3
    return ram_v3


def _forward_n_low():
    from impact_pipeline.bench import forward_v2 as F2
    return F2.N_LOW_DEFAULT


def _replication_code():
    """The extensions the code runs: the anchor designs', and the forward
    arms' under the forward anchor design."""
    from impact_pipeline.bench.designs_v2 import anchors as AN
    out = dict(AN.CALIBRATION_PENDING["replication_extended"]["value"])
    out[FORWARD_ANCHOR_DESIGN] = dict(
        AN.CALIBRATION_PENDING["forward_replication_extended"]["value"])
    return out


DECISIONS: Tuple[DecisionSpec, ...] = (
    DecisionSpec("nas_se_method", "CD-2",
                 "NAS v3 SE method (twins 820-824: the calibrated method with the "
                 "largest df; ties and failure: contiguous G = 10)",
                 lambda: _nas().SE_METHOD_DEFAULT,
                 lambda v: _one_of(_nas().SE_METHODS)(v)),
    DecisionSpec("iim_se_method", "CD-3",
                 "IIM v5 SE method (block bootstrap; the contiguous jackknife G = 10 "
                 "if the bootstrap fails its twin calibration)",
                 lambda: _iim().SE_METHOD_DEFAULT,
                 lambda v: _one_of(_iim().SE_METHODS)(v)),
    DecisionSpec("iim_bootstrap_se_df", "CD-3",
                 "df of the IIM block bootstrap SE (12; 9 if the one-sided 1 % tail "
                 "exceeds 0.02 on the twins)",
                 lambda: float(_iim().BOOTSTRAP_SE_DF),
                 lambda v: float(_one_of(_iim().BOOTSTRAP_SE_DF_ALLOWED)(float(v))),
                 ("impact_pipeline.v2.iim_v5.BOOTSTRAP_SE_DF (the family-B path)",
                  lambda: float(_iim().BOOTSTRAP_SE_DF))),
    DecisionSpec("iim_n_min", "CD-3",
                 "IIM occupancy gate N_min (25; raised to 50 or 100 only if the "
                 "development occupancy cells fail their 90 % criteria)",
                 lambda: int(_iim().N_MIN_DEFAULT),
                 lambda v: _one_of(_iim().N_MIN_ALLOWED)(v),
                 ("impact_pipeline.v2.iim_v5.N_MIN_DEFAULT (family-B protocols and "
                  "occupancy cells)", lambda: int(_iim().N_MIN_DEFAULT))),
    DecisionSpec("ram_se_method", "CD-4",
                 "RAM-PE v3 SE method (shift-null SD; the trial jackknife if the "
                 "twins' kappa interval lies outside [0.8, 1.25])",
                 lambda: _ram().RAMParams().se_method,
                 lambda v: _one_of(_ram().SE_METHODS)(v)),
    DecisionSpec("pdi_concordance", "CD-5",
                 "PDI concordance route: 'rule' (the admission rule on the batteries), "
                 "'none', or the admitted cells",
                 lambda: "none", _concordance_value),
    DecisionSpec("declared_dependencies", "CD-7",
                 "witness-principle pairs whose off-target change is a declared "
                 "structural dependency (HCv2-22 (iii))",
                 _draft_dependencies, _pairs),
    DecisionSpec("mechanism_on", "CD-8",
                 "mechanism-on labels: 'rule' (dose >= 0.5 x nominal and development "
                 "median c >= 2 delta) or 'structural' (the hypotheses-file draft)",
                 lambda: "structural", _mechanism_value),
    DecisionSpec("replication_extended", "CD-11",
                 "anchor designs (and forward arms) whose replication block "
                 "extends to 20900-20939",
                 _replication_code, _replication_value,
                 ("impact_pipeline.bench.designs_v2.anchors.CALIBRATION_PENDING "
                  "(replication_extended; forward_replication_extended per arm)",
                  _replication_code)),
    DecisionSpec("paper2_regime", "CD-12",
                 "paper-2 regime from the BIDS metadata: n_low and reference scheme "
                 "(TR, duration and declaration of the paper-2 confirmatory data)",
                 lambda: {"n_low": int(_forward_n_low()), "reference": "average"},
                 _regime_value,
                 ("impact_pipeline.bench.forward_v2.N_LOW_DEFAULT (n_low)",
                  lambda: {"n_low": int(_forward_n_low())})),
    DecisionSpec("pdi_kmeans_seed", "open decision",
                 "k-means seed of every PDI v3 count (an integer, or null for the "
                 "task's null seed)",
                 lambda: RB.CALIBRATION_PENDING["pdi_kmeans_seed"]["value"],
                 _seed_value,
                 ("impact_pipeline.bench.run_bench_v2.CALIBRATION_PENDING",
                  lambda: RB.CALIBRATION_PENDING["pdi_kmeans_seed"]["value"])),
    DecisionSpec("null_calibration_ram", "open decision",
                 "RAM-PE v3 on the null-calibration cells, where it is "
                 "UNDEFINED(INSUFFICIENT_UPDATES): 'keep' the rows, 'drop' RAM from "
                 "those cells, or 'require_defined' rows in HCv2-1 and HCv2-3",
                 lambda: "keep",
                 lambda v: _one_of(("keep", "drop", "require_defined"))(v)),
    DecisionSpec("iim_max_macro_nodes", "run setting",
                 "the cap on the IIM v5 macro grain of every run (recorded in every "
                 "record's settings)",
                 lambda: RB.DEFAULT_SETTINGS.iim_max_macro_nodes,
                 _int_ge(1),
                 ("impact_pipeline.bench.run_bench_v2.IIM_MAX_MACRO_NODES (the "
                  "default run setting)",
                  lambda: RB.DEFAULT_SETTINGS.iim_max_macro_nodes)),
    DecisionSpec("forward_anchor_regime", "design 3.6",
                 "regime of the forward views' anchors: 'held_out' (the frozen "
                 "choice; needs the release) or 'development' (development builds "
                 "only)",
                 lambda: "held_out",
                 lambda v: _one_of(("held_out", "development"))(v)),
    DecisionSpec("srpi_v3_pairs", "CD-14",
                 "SRPI v3 pair count n* of the SRPI-only arm (Tier B: only if SRPI "
                 "v3 is admitted; 60, 90 or 120 by the power rule)",
                 lambda: None,
                 lambda v: None if v is None else _one_of((60, 90, 120))(v),
                 tier="B"),
)
DECISION_NAMES = tuple(d.name for d in DECISIONS)


@dataclass(frozen=True)
class Decisions:
    """The decisions as applied: ``values`` (decided or provisional),
    ``pending`` (names without a decision), ``deviation`` notes, the file's
    status and hash."""

    values: Mapping
    pending: Tuple[str, ...]
    deviation: Mapping
    notes: Mapping
    status: str
    sha256: Optional[str]

    def __getitem__(self, name):
        return self.values[name]


def load_decisions(source=None) -> Decisions:
    """Decisions from a file or mapping (None: every decision pending)."""
    payload = {}
    sha = None
    if source is not None:
        if isinstance(source, Mapping):
            payload = dict(source)
        else:
            text = Path(source).read_text(encoding="utf-8")
            sha = sha256_text(text)
            payload = json.loads(text)
    if payload and payload.get("schema") != DECISIONS_SCHEMA:
        raise BuildError(f"a decisions file has schema {DECISIONS_SCHEMA!r}")
    status = payload.get("status", STATUS_DRAFT)
    if status not in (STATUS_DRAFT, STATUS_FINAL):
        raise BuildError("decisions status must be draft or final")
    raw = dict(payload.get("decisions") or {})
    unknown = sorted(set(raw) - set(DECISION_NAMES))
    if unknown:
        raise BuildError(f"unknown decisions {unknown}; one of {DECISION_NAMES}")
    values, pending, deviation, notes = {}, [], {}, {}
    for spec in DECISIONS:
        entry = raw.get(spec.name)
        if entry is not None and not isinstance(entry, Mapping):
            raise BuildError(f"decision {spec.name}: an object with 'value'")
        val = None if entry is None else entry.get("value")
        # a decision is taken when it has a value; a null value is a decision
        # only where null is admissible and the entry says "decided": true
        # (the PDI k-means seed: null means the task's null seed)
        if val is None and not (entry is not None and entry.get("decided") is True):
            pending.append(spec.name)
            val = spec.provisional()
        try:
            values[spec.name] = spec.validate(val)
        except BuildError as exc:
            raise BuildError(f"decision {spec.name}: {exc}") from None
        if entry:
            if entry.get("deviation"):
                deviation[spec.name] = str(entry["deviation"])
            if entry.get("note"):
                notes[spec.name] = str(entry["note"])
    return Decisions(values, tuple(pending), deviation, notes, status, sha)


# --------------------------------------------------------------------------
# development outputs
# --------------------------------------------------------------------------
@dataclass
class Source:
    path: str
    sha256: str
    item: str
    run: str
    tagged: bool
    n_records: int


class DevData:
    """The development records under a calibration root, checked against the
    development policy: development seeds only (a seed of 1000 or more is
    refused), no record on a smoke seed, no record of a held-out condition,
    held-out-regime anchors only under a logged release, no family-B cell of
    the held-out regime, no IIM on the held-out G sweep of the Hopf arm's v2
    sensor views (:func:`dev_calibration.ho6_withheld`)."""

    def __init__(self, root, *, files: Optional[Sequence] = None):
        self.root = Path(root)
        self.records: List[Tuple[Source, REC.TaskRecord]] = []
        self.sources: List[Source] = []
        self.releases = DC.read_release_log(self.root) if self.root.exists() else []
        self.release_ids = {r["release_id"] for r in self.releases}
        paths = ([Path(p) for p in files] if files is not None
                 else (DC.result_files(self.root) if self.root.exists() else []))
        for p in sorted(paths, key=lambda q: str(q)):
            self._read(Path(p))

    def _read(self, path: Path) -> None:
        try:
            rel = path.relative_to(self.root)
            head = rel.parts[0]
            run = rel.parts[1] if len(rel.parts) > 2 else ""
        except ValueError:
            rel, head, run = path, "", ""
        latest, _info = RB.read_results(path)
        src = Source(str(rel), _file_sha(path), head.split("@", 1)[0], run,
                     "@" in head, len(latest))
        for tid in sorted(latest):
            rec = latest[tid]
            self._check(rec, path)
            self.records.append((src, rec))
        self.sources.append(src)

    def _check(self, rec: REC.TaskRecord, path: Path) -> None:
        if int(rec.seed) > S.DEV_SEED_MAX or rec.split != S.DEVELOPMENT:
            raise BuildError(f"{path}: {rec.task_id} has seed {rec.seed}; the builder "
                             f"reads development seeds 0-{S.DEV_SEED_MAX} only")
        if S.is_smoke_seed(rec.seed):
            raise BuildError(f"{path}: {rec.task_id} is on a smoke seed ({rec.seed}); "
                             "smoke outputs are discarded and never read")
        cfg = rec.config or {}
        if cfg.get("held_out") or any((s.details or {}).get("held_out")
                                      for s in rec.scorings):
            raise BuildError(f"{path}: {rec.task_id} is a held-out condition; held-out "
                             "outputs never enter the calibration")
        tags = cfg.get("tags") or {}
        if tags.get("regime") == "held_out":
            rid = tags.get("held_out_release")
            if rid not in self.release_ids:
                raise BuildError(f"{path}: {rec.task_id} is at the held-out regime "
                                 "without a logged release")
        for s in rec.scorings:
            if DC.HO6_PRINCIPLE in s.components and DC.ho6_withheld(tags, s.view):
                raise BuildError(f"{path}: {rec.task_id} has IIM on {s.view} at G = "
                                 f"{tags.get('dose')}: the G sweep of the v2 sensor "
                                 "pipeline is held out (HO-6)")
        if rec.design == "family_b":
            cell = (cfg.get("cell") or {})
            fake = type("C", (), {"system": cell.get("system"),
                                  "params": tuple((cell.get("params") or {}).items())})
            if DC.family_b_held_out(fake):
                raise BuildError(f"{path}: {rec.task_id} is a family-B cell of the "
                                 "held-out regime")

    def select(self, designs: Iterable[str] = (), *, tagged: Optional[bool] = False,
               items: Optional[Iterable[str]] = None
               ) -> List[Tuple[Source, REC.TaskRecord]]:
        ds = set(designs)
        its = None if items is None else set(items)
        return [(s, r) for s, r in self.records
                if (not ds or r.design in ds) and (tagged is None or s.tagged == tagged)
                and (its is None or s.item in its)]


def _file_sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# re-judging stored components under a protocol
# --------------------------------------------------------------------------
def excess_of(comp, direction: Optional[str] = None) -> float:
    """``estimate - null_mean`` of a component (or of one NAS direction);
    NaN when the estimator is undefined or a value is missing."""
    det = comp.details or {}
    if direction is not None:
        m = (det.get("members") or {}).get(direction) or {}
        if not m.get("defined", False):
            return float("nan")
        e, nm = _f(m.get("estimate")), _f(m.get("null_mean"))
    else:
        if not det.get("defined", False):
            return float("nan")
        e, nm = _f(comp.estimate), _f(comp.null_mean)
    if e is None or nm is None:
        return float("nan")
    return e - nm


def _substrate(rec, scoring) -> Optional[str]:
    sub = (scoring.details or {}).get("substrate")
    return sub if sub else FAMILY_SUBSTRATE.get(rec.family)


def _bearer_id(scoring, principle) -> str:
    """The bearer id the runner gives a component's evidence: the
    principle's own bearer in the principle-bearer mode (the patchwork
    declares one per principle), else the system."""
    mode = (scoring.details or {}).get("bearer_mode")
    return f"principle:{principle}" if mode == "principle" else "system"


def evidence_items(rec, scoring, comp, proto, *, se_df_override=None) -> Optional[list]:
    """The evidence items of a stored component as the runner built them
    (None when the estimator produced no output: an estimator error or a
    declared not-applicable observation model keeps its stored status)."""
    det = comp.details or {}
    if "estimator_id" not in det:
        return None
    common = dict(principle=comp.principle, channel="default",
                  bearer_id=_bearer_id(scoring, comp.principle),
                  protocol_id=proto.protocol_id, substrate=_substrate(rec, scoring),
                  observation_stage=scoring.observation_stage, view=scoring.view,
                  estimator=det["estimator_id"])
    members = det.get("members")
    rows = []
    if members:
        for d in sorted(members):
            rows.append(dict(members[d]))
    else:
        est = det.get("estimator") or {}
        rows.append({"direction": None, "estimate": comp.estimate,
                     "null_mean": comp.null_mean, "null_sd": comp.null_sd,
                     "n_null": comp.n_null, "null_family": comp.null_family,
                     "se": comp.se, "se_df": comp.se_df, "se_method": comp.se_method,
                     "defined": det.get("defined", True),
                     "reason": det.get("estimator_reason"), "exact": False,
                     "p_ind": det.get("p_ind"),
                     "content_bearer": est.get("counted_bearer")
                     if comp.principle == "PDI" else None})
    out = []
    for m in rows:
        se_df = m.get("se_df")
        if (se_df_override is not None and comp.principle == "IIM"
                and m.get("se_method") == _iim().SE_METHOD_DEFAULT):
            se_df = float(se_df_override)
        nan = float("nan")
        out.append(E.ComponentEvidenceV2(
            estimate=nan if m.get("estimate") is None else float(m["estimate"]),
            null_mean=nan if m.get("null_mean") is None else float(m["null_mean"]),
            null_sd=nan if m.get("null_sd") is None else float(m["null_sd"]),
            se=0.0 if m.get("se") is None else float(m["se"]), se_df=se_df,
            defined=bool(m.get("defined", True)), reason=m.get("reason"),
            null_family=m.get("null_family"), n_null=int(m.get("n_null") or 0),
            exact=bool(m.get("exact", False)), se_method=m.get("se_method"),
            direction=m.get("direction"), p_ind=m.get("p_ind"),
            content_bearer=m.get("content_bearer"), **common))
    return out


@dataclass
class Judged:
    """A stored component re-judged under a protocol."""

    status: str
    reason: Optional[str]
    flags: Tuple[str, ...]
    c: Optional[float]
    se_c: Optional[float]
    df_c: Optional[float]
    members: Mapping
    se_method: Optional[str]
    concordant: bool


def rejudge(rec, scoring, comp, proto, *, se_df_override=None) -> Judged:
    items = evidence_items(rec, scoring, comp, proto, se_df_override=se_df_override)
    if items is None:
        return Judged(comp.status, comp.reason, tuple(comp.flags), _f(comp.c),
                      _f(comp.se_c), _f(comp.df_c), {}, comp.se_method, False)
    pa = E.assess_principle(comp.principle, items, proto)
    f = pa.record_fields()
    members = {}
    if pa.deciding is not None and pa.deciding.members:
        for m in pa.deciding.members:
            members[m.direction] = {"c": _f(m.c), "se": _f(m.se), "df": _f(m.df),
                                    "status": m.status.value}
    return Judged(f["status"], f["reason"], tuple(f["flags"]), _f(f["c"]),
                  _f(f["se_c"]), _f(f["df_c"]), members, comp.se_method,
                  comp.se_method == E.SE_METHOD_CONCORDANT)


def _undefined_items(rec, scoring, comp, proto) -> list:
    """The evidence of a component without estimator output (an estimator
    error or a not-applicable observation model) as the runner builds it:
    one undefined item per direction with the stored reason."""
    p = comp.principle
    dirs = proto.directions_for(p)
    common = dict(principle=p, channel="default", bearer_id=_bearer_id(scoring, p),
                  protocol_id=proto.protocol_id, substrate=_substrate(rec, scoring),
                  observation_stage=scoring.observation_stage, view=scoring.view)
    return [E.ComponentEvidenceV2(
        estimate=float("nan"), defined=False,
        reason=comp.reason or R.NOT_APPLICABLE_OBSERVATION_MODEL,
        estimator=f"compute_{p}:undefined@{comp.estimator_version}", direction=d,
        **common) for d in (dirs or (None,))]


def _iim_se_df(proto) -> Optional[float]:
    """The bootstrap ``se_df`` an IIM block of the protocol declares."""
    try:
        opts = proto.estimator_options("IIM") or {}
    except (KeyError, ValueError):
        return None
    v = _f(opts.get("bootstrap_se_df"))
    return v


def _rejudged_component(comp, verdict, proto):
    """A stored component with the status fields of ``verdict`` (the runner's
    component record under ``proto``): status, reason, flags, ``c``, the NAS
    directions, ``se_c``, ``df_c``, the deciding assessment and, for a
    directional principle, the member reported (the direction of the
    smaller ``c``). An estimator error keeps its record."""
    det = dict(comp.details or {})
    if "error" in det:
        return dataclasses.replace(comp, protocol_hash=proto.hash)
    p = comp.principle
    f = verdict.construct[p]
    upd = dict(status=f["status"], reason=f["reason"], flags=tuple(f["flags"]),
               c=f["c"], se_c=f["se_c"], df_c=f["df_c"], protocol_hash=proto.hash)
    if p == "NAS":
        upd["c_R"], upd["c_B"] = f.get("c_R"), f.get("c_B")
    deciding = verdict.principle_assessment.get(p)
    members = det.get("members") or {}
    if len(members) > 1 and deciding is not None and deciding.members:
        cs = {m.direction: m.c for m in deciding.members if math.isfinite(m.c)}
        if cs:
            m = members.get(str(min(cs, key=cs.get))) or {}
            upd.update(estimate=m.get("estimate"), null_mean=m.get("null_mean"),
                       null_sd=m.get("null_sd"), n_null=int(m.get("n_null") or 0),
                       null_family=m.get("null_family"), se=m.get("se"),
                       se_df=m.get("se_df"), se_method=m.get("se_method"))
            det["defined"] = bool(m.get("defined"))
            det["estimator_reason"] = m.get("reason")
            if m.get("p_ind") is not None:
                det["p_ind"] = m["p_ind"]
            else:
                det.pop("p_ind", None)
    if deciding is not None:
        det["assessment"] = RB._sanitize(deciding.to_dict())
    else:
        det.pop("assessment", None)
    upd["details"] = det
    return dataclasses.replace(comp, **upd)


def rejudge_record(rec, protos: Mapping):
    """A stored task record judged again under ``protos`` (``{key:
    ProtocolV3}``): every scoring whose protocol key is given gets its
    components and verdict from the stored estimator outputs as the runner
    judges them under that protocol (the IIM bootstrap ``se_df`` of the
    protocol applied); other scorings are kept. The scoring details record
    the hash the record was judged under before (``rejudged_from``)."""
    scorings = []
    for s in rec.scorings:
        proto = protos.get(s.protocol_id)
        if proto is None:
            scorings.append(s)
            continue
        df = _iim_se_df(proto)
        evidence = {}
        for p, comp in s.components.items():
            items = evidence_items(rec, s, comp, proto, se_df_override=df)
            evidence[p] = (items if items is not None
                           else _undefined_items(rec, s, comp, proto))
        verdict = E.verdict_v3(evidence, proto)
        comps = {p: _rejudged_component(comp, verdict, proto)
                 for p, comp in s.components.items()}
        details = dict(s.details or {}, rejudged_from=s.protocol_hash)
        scorings.append(dataclasses.replace(
            s, protocol_hash=proto.hash, components=comps, details=details,
            verdict=RB._verdict_block(verdict) if s.verdict is not None else None))
    return dataclasses.replace(rec, scorings=tuple(scorings))


# --------------------------------------------------------------------------
# anchors
# --------------------------------------------------------------------------
def _is_positive_control(rec) -> bool:
    return rec.system == "PC_nominal"


def _is_forward_anchor(rec) -> bool:
    """A forward reference task of the view's anchor condition (PC_nominal
    or ``G_nom``; the forward design tags it ``anchor``)."""
    return bool(((rec.config or {}).get("tags") or {}).get("anchor"))


def release_problems(data: "DevData", repo_root=REPO_ROOT) -> List[str]:
    """What blocks the freeze in the held-out release of the records read:
    a release whose predictions files changed since it was logged (SHA-256
    or an uncommitted change), are gone, or that names none."""
    used = sorted({((r.config or {}).get("tags") or {}).get("held_out_release")
                   for _s, r in data.records} - {None}, key=str)
    by_id = {r.get("release_id"): r for r in data.releases}
    out = []
    for rid in used:
        entry = by_id.get(rid)
        if entry is None:  # refused when the records were read
            continue
        logged = list(entry.get("predictions") or ())
        if not logged:
            out.append(f"held-out release {rid}: names no predictions file")
            continue
        try:
            now = {s["path"]: s for s in DC.predictions_state(
                [p["path"] for p in logged], repo_root)}
        except DC.CalibrationError as exc:
            out.append(f"held-out release {rid}: {exc}")
            continue
        changed = [p["path"] for p in logged
                   if now[p["path"]]["sha256"] != p.get("sha256")
                   or not now[p["path"]]["clean"]]
        if changed:
            out.append(f"held-out release {rid}: predictions changed since the "
                       f"release: {changed}")
    return out


def anchor_series(pairs, key: str, *, positive: Callable, lesion: Optional[Callable],
                  seeds: Sequence[int]) -> Dict[str, Dict[str, np.ndarray]]:
    """Per principle (NAS per direction, ``NAS:receive``) the excesses of the
    positive control and of the own lesion on the reference seeds (NaN for a
    missing seed) from the scorings of protocol ``key``."""
    index = {int(s): i for i, s in enumerate(seeds)}
    out: Dict[str, Dict[str, np.ndarray]] = {}

    def slot(name):
        return out.setdefault(name, {"pc": np.full(len(seeds), np.nan),
                                     "lesion": np.full(len(seeds), np.nan)})

    for _src, rec in pairs:
        if rec.replicate or int(rec.seed) not in index:
            continue
        i = index[int(rec.seed)]
        for s in rec.scorings:
            if s.protocol_id != key:
                continue
            for p, comp in s.components.items():
                role = None
                if positive(rec):
                    role = "pc"
                elif lesion is not None and lesion(rec, p):
                    role = "lesion"
                if role is None:
                    continue
                if p == "NAS" and (comp.details or {}).get("members"):
                    for d in E.NAS_DIRECTIONS:
                        slot(f"NAS:{d}")[role][i] = excess_of(comp, d)
                else:
                    slot(p)[role][i] = excess_of(comp)
    return out


def anchor_entries(series: Mapping, *,
                   with_specificity: bool = True) -> Tuple[dict, dict]:
    """``(entries, reference)``: the anchor entry of every declared principle
    (validity, specificity, status) and the external reference values and
    SEs of the valid ones (NAS per direction)."""
    entries, values, ses = {}, {}, {}
    for p in PRINCIPLES:
        if p == "NAS":
            members, ok = {}, True
            for d in E.NAS_DIRECTIONS:
                s = series.get(f"NAS:{d}")
                pc = s["pc"] if s else np.full(T.ANCHOR_BLOCK_SIZE, np.nan)
                v = T.anchor_validity(pc)
                spec = None
                if (with_specificity and s is not None
                        and np.isfinite(s["lesion"]).any()):
                    spec = T.anchor_specificity(s["pc"], s["lesion"], anchor=v["mean"])
                members[d] = T.anchor_entry(v, spec)
                ok &= bool(v["valid"])
            entries[p] = T.combine_anchor_entries(members)
            if entries[p]["valid"]:
                for d in E.NAS_DIRECTIONS:
                    vd = members[d]["validity"]
                    values[f"NAS:{d}"], ses[f"NAS:{d}"] = vd["mean"], vd["se"]
            continue
        s = series.get(p)
        pc = s["pc"] if s else np.full(T.ANCHOR_BLOCK_SIZE, np.nan)
        v = T.anchor_validity(pc)
        spec = None
        if with_specificity and s is not None and np.isfinite(s["lesion"]).any():
            spec = T.anchor_specificity(s["pc"], s["lesion"], anchor=v["mean"])
        entries[p] = T.anchor_entry(v, spec)
        if entries[p]["valid"]:
            values[p], ses[p] = v["mean"], v["se"]
    return entries, {"values": values, "se": ses}


def reference_block(ref: Mapping, source: str) -> dict:
    if not ref["values"]:
        return {"kind": E.REFERENCE_PENDING,
                "note": "no valid anchor on the development reference block "
                        f"({source})"}
    return {"kind": "external", "scale": "excess", "values": dict(ref["values"]),
            "se": dict(ref["se"]), "source": source}


# --------------------------------------------------------------------------
# protocol assembly
# --------------------------------------------------------------------------
def confirmatory_plan_principles() -> Dict[str, set]:
    """``{protocol key: principles scored}`` over every scoring of the
    confirmatory runner plan (all merged design modules)."""
    tasks = D.build_plan(list(D.designs()), S.CONFIRMATORY)
    for t in tasks:
        RB.load_design_module(t.design_module)
    out: Dict[str, set] = {}
    for t in tasks:
        for s in t.scorings:
            out.setdefault(s.protocol_key, set()).update(s.principles)
    return out


def confirmatory_protocol_keys() -> List[str]:
    """Every protocol key a record of the confirmatory run names: the runner
    plan of every merged design module and the family-B keys."""
    keys = set(confirmatory_plan_principles())
    keys.update(family_b_protocols())
    return sorted(keys)


def family_b_protocols() -> Dict[str, object]:
    """``{key: ProtocolV3}`` of every family-B protocol the validation builds
    in memory (cut mode x null family x anchor x SE contract)."""
    from impact_pipeline.bench.designs_v2 import family_b as FB

    out = {}
    for cut in FB.CUT_MODES:
        for fam in _iim().NULL_FAMILIES:
            for anchor in (FB.PRIMARY_ANCHOR, FB.SECOND_ANCHOR):
                for with_se in (True, False):
                    p = FB.family_b_protocol(cut, anchor=anchor, null_family=fam,
                                             sampling_se=with_se)
                    out[HE.protocol_key(p)] = p
    return out


def _load_modules() -> None:
    for name in D.available_modules():
        RB.load_design_module(name)


def draft_payload(key: str) -> dict:
    """The runner's draft of a key (template, declaration, form options, the
    design modules' drafts)."""
    _load_modules()
    return RB.draft_protocol(key).to_dict()


def apply_estimator_decisions(payload: dict, dec: Decisions) -> dict:
    """The SE methods and IIM settings of the decisions in every estimator
    block the protocol has; an IIM block that declares no SE (``se_method``
    None, a value-only scoring) keeps it."""
    est = payload.setdefault("estimators", {})
    methods = payload.setdefault("se_methods", {})
    if "NAS" in est:
        est["NAS"]["se_method"] = dec["nas_se_method"]
        methods["NAS"] = [dec["nas_se_method"]]
    if "IIM" in est:
        blk = est["IIM"]
        if "se_method" not in blk or blk["se_method"] is not None:
            blk["se_method"] = dec["iim_se_method"]
        blk["n_min"] = int(dec["iim_n_min"])
        blk["bootstrap_se_df"] = float(dec["iim_bootstrap_se_df"])
        methods["IIM"] = [dec["iim_se_method"]]
    if "RAM" in est:
        est["RAM"]["se_method"] = dec["ram_se_method"]
        methods["RAM"] = [dec["ram_se_method"]]
    return payload


def _named(payload: dict, key: str) -> dict:
    payload["name"] = f"mpc-bench-v2-{key}"
    return payload


def _scores(payload: Mapping, principle: str) -> bool:
    return principle in (payload.get("estimators") or {})


def carries_concordance_route(key: str, payload: Mapping) -> bool:
    """Whether a protocol carries the admitted concordance cells: it scores
    PDI, its PDI block declares no access module and it is no protocol of
    the BOLD forward arm (:data:`NO_CONCORDANCE_ROUTE_PREFIXES`)."""
    if not _scores(payload, "PDI"):
        return False
    if (payload["estimators"]["PDI"] or {}).get("access_module") is not None:
        return False
    return not str(key).startswith(NO_CONCORDANCE_ROUTE_PREFIXES)


def assemble(key: str, base: dict, dec: Decisions, *, reference: dict, anchors: dict,
             concordance: Sequence = (), precision: Optional[dict] = None) -> object:
    """A protocol from a draft payload: the decisions, the reference, the
    anchors block with ``N_anch`` as the necessity set, the admitted
    concordance cells (:func:`carries_concordance_route`) and the precision
    block."""
    payload = apply_estimator_decisions(copy.deepcopy(base), dec)
    payload["reference"] = reference
    payload["anchors"] = anchors
    n_anch = list(anchors["necessity_set"]) or list(E.PRINCIPLES)
    payload["necessity_set"] = n_anch
    payload["concordance_route"] = (list(concordance)
                                    if carries_concordance_route(key, payload) else [])
    payload["precision"] = precision
    if key in NO_REPORTED_CUT and "IIM" in payload["estimators"]:
        payload["estimators"]["IIM"]["report_cut_modes"] = []
    return E.ProtocolV3.from_dict(_named(payload, key))


def derived_copy(base_proto, key: str, declaration: str) -> object:
    """A copy of a generated protocol with another declaration (the held-out
    declarations and the null-calibration classification ``A-none``): its
    anchors, cutoffs, estimators, SE methods, concordance cells and precision
    block kept, the reported IIM cut dropped."""
    from impact_pipeline.bench.designs_v2 import null_calibration as NCV

    if key == NCV.PROTOCOL_KEY:
        payload = NCV.classification_protocol(base_proto.to_dict())
    else:
        payload = base_proto.to_dict()
        payload["shared_inputs_declaration"] = {
            "id": declaration, "shared_inputs": E.DECLARATION_LEVELS[declaration]}
    if "IIM" in (payload.get("estimators") or {}):
        payload["estimators"]["IIM"]["report_cut_modes"] = []
    return E.ProtocolV3.from_dict(_named(payload, key))


# --------------------------------------------------------------------------
# testability, concordance, SE calibration, mechanism-on
# --------------------------------------------------------------------------
def _witness_targets() -> Dict[str, str]:
    return dict(_hypotheses_spec().get("witness_targets") or {})


def _decided_methods(dec: Decisions) -> Dict[str, str]:
    return {"NAS": dec["nas_se_method"], "IIM": dec["iim_se_method"],
            "RAM": dec["ram_se_method"]}


def judged_rows(data: DevData, protos: Mapping, designs: Sequence[str], *,
                dec: Decisions, replicate0: bool = True) -> List[dict]:
    """Every component of the selected development records re-judged under
    the generated protocol its scoring names. Where a task was run more than
    once (a tagged re-run, for example with the decided SE methods), each
    (task, scoring, principle) is taken once: the run whose SE method is the
    decided one first, then the untagged run, then the first by path."""
    want = _decided_methods(dec)
    best: Dict[tuple, tuple] = {}
    for src, rec in data.select(designs, tagged=None):
        if replicate0 and rec.replicate:
            continue
        for s in rec.scorings:
            if s.protocol_id not in protos:
                continue
            for p, comp in s.components.items():
                key = (rec.task_id, s.scoring_id, p)
                rank = (comp.se_method != want.get(p, comp.se_method), src.tagged,
                        src.path)
                if key not in best or rank < best[key][0]:
                    best[key] = (rank, src, rec, s, p, comp)
    out = []
    for key in sorted(best):
        _rank, src, rec, s, p, comp = best[key]
        j = rejudge(rec, s, comp, protos[s.protocol_id],
                    se_df_override=dec["iim_bootstrap_se_df"])
        out.append({"task_id": rec.task_id, "design": rec.design,
                    "family": rec.family, "system": rec.system,
                    "seed": int(rec.seed), "protocol": s.protocol_id,
                    "declaration": s.declaration_id, "form": s.estimator_form,
                    "view": s.view, "principle": p,
                    "tags": rec.config.get("tags") or {},
                    "se_method": comp.se_method, "source": src.path,
                    "judged": j, "comp": comp, "scoring": s, "record": rec})
    return out


def testability(rows: Sequence[dict], dec: Decisions) -> Tuple[List[dict], List[dict]]:
    """``(rows, table)``: the precision rows with at least
    :data:`testability.MIN_RUNS` runs, and the whole table (every gate, its
    run count, sources and the SE methods it was computed with)."""
    targets = _witness_targets()
    wanted = []
    for key in GATE_PROTOCOLS:
        for w, p in sorted(targets.items()):
            wanted.append((key, p, w, T.KIND_ABSENT))
        for p in PRINCIPLES:
            wanted.append((key, p, "PC_nominal", T.KIND_PRESENT))
    decided = {"NAS": dec["nas_se_method"], "IIM": dec["iim_se_method"],
               "RAM": dec["ram_se_method"]}
    by = {}
    for r in rows:
        if r["form"] != D.PRIMARY_FORM or r["design"] not in TESTABILITY_DESIGNS:
            continue
        by.setdefault((r["protocol"], r["principle"], r["system"]), []).append(r)
    good, table = [], []
    for key, p, w, kind in wanted:
        # every run counts, an estimator error too (it is UNDEFINED, never
        # ABSENT or PRESENT)
        runs = by.get((key, p, w), [])
        methods = sorted({str(r["se_method"]) for r in runs
                          if r["se_method"] is not None})
        entry = {"family": key, "principle": p, "witness": w, "kind": kind,
                 "n": len(runs), "sources": sorted({r["design"] for r in runs}),
                 "se_methods": methods,
                 "se_method_matches_decision": (p not in decided or not methods
                                                or methods == [decided[p]])}
        if len(runs) >= T.MIN_RUNS:
            row = T.testability_row(
                family=key, principle=p, witness=w, kind=kind,
                runs=[{"status": r["judged"].status, "reason": r["judged"].reason,
                       "flags": r["judged"].flags, "se_c": r["judged"].se_c,
                       "df_c": r["judged"].df_c, "seed": r["seed"]} for r in runs],
                members=2 if (p == "NAS" and kind == T.KIND_ABSENT) else 1)
            good.append(row)
            entry["row"] = row
        else:
            entry["row"] = None
            entry["why"] = f"{len(runs)} development runs; the gate needs {T.MIN_RUNS}"
        table.append(entry)
    return good, table


def family_b_rows(data: DevData, se_df: Optional[float] = None,
                  protos: Optional[Mapping] = None) -> Tuple[Optional[dict], dict]:
    """The family-B testability row (HCv2-11 (c)): IIM-dir ABSENT on the
    feed-forward star at coupling >= 0.2 on the exact family-B scale. With
    ``se_df`` each stored IIM component is judged again under its family-B
    protocol (``protos``, default :func:`family_b_protocols`) at that block
    bootstrap df (the decided ``iim_bootstrap_se_df``); without it, or for a
    record whose protocol is not known, the recorded status counts."""
    g = FAMILY_B_GATE
    if se_df is not None and protos is None:
        protos = family_b_protocols()
    runs = []
    for _src, rec in data.select(("family_b",)):
        cell = (rec.config or {}).get("cell") or {}
        if cell.get("hypothesis") != "HCv2-11" or cell.get("system") != g["witness"]:
            continue
        if float((cell.get("params") or {}).get("coupling", 0.0)) < g["min_coupling"]:
            continue
        for s in rec.scorings:
            if s.estimator_form != g["cut"]:
                continue
            c = s.components.get("IIM")
            if c is None:
                continue
            proto = None if se_df is None else protos.get(s.protocol_id)
            if proto is not None:
                j = rejudge(rec, s, c, proto, se_df_override=se_df)
                runs.append({"status": j.status, "reason": j.reason, "flags": j.flags,
                             "se_c": j.se_c, "df_c": j.df_c, "seed": rec.seed})
                continue
            runs.append({"status": c.status, "reason": c.reason, "flags": c.flags,
                         "se_c": c.se_c, "df_c": c.df_c, "seed": rec.seed})
    info = {"family": g["family"], "principle": g["principle"], "witness": g["witness"],
            "kind": g["kind"], "n": len(runs), "sources": ["family_b"],
            "se_df": se_df}
    if len(runs) < T.MIN_RUNS:
        info["row"] = None
        info["why"] = f"{len(runs)} development runs; the gate needs {T.MIN_RUNS}"
        return None, info
    row = T.testability_row(family=g["family"], principle=g["principle"],
                            witness=g["witness"], kind=g["kind"], runs=runs)
    info["row"] = row
    return row, info


def concordance_evidence(rows: Sequence[dict]) -> List[dict]:
    """Per (substrate, observation stage, view, bearer) cell of PDI: the
    content-on runs and their concordant ABSENTs, the no-content runs and
    their concordant PRESENTs (battery classes, and the forward family-A
    conditions except the BOLD arm's source view), and the rule's admission
    (0 events among >= 300 runs; a forward cell's admission is
    provisional). Where a view has no reference yet (c undefined), a
    concordant count at one state (zero excess) on a content-on run is a
    concordant ABSENT, and a concordant no-content run with a positive
    excess is recorded as undetermined."""
    from impact_pipeline.bench.designs_v2 import forward as FW

    content = {c[0]: c[1] for c in DC.BATTERY_CLASSES}
    fwd_on = {"PC_nominal", "PC_nominal_K2", "PC_nominal_K3"}
    fwd_off = {"W_PDI_no_multistability", "N_ar1", "PC_nominal_K1"}
    cells: Dict[tuple, dict] = {}
    for r in rows:
        if r["principle"] != "PDI" or r["form"] != D.PRIMARY_FORM:
            continue
        tags = r["tags"]
        if r["design"] == DC.BATTERY_DESIGN:
            kind = content.get(tags.get("battery_class"))
        elif r["design"] in FW.ADMISSION_RECORD_DESIGNS:
            kind = (DC.CONTENT_ON if r["system"] in fwd_on else DC.CONTENT_OFF
                    if r["system"] in fwd_off else None)
        else:
            kind = None
        if kind is None:
            continue
        j = r["judged"]
        sub = _substrate(r["record"], r["scoring"])
        est = (r["comp"].details or {}).get("estimator") or {}
        bearer = est.get("counted_bearer")
        if not sub or not bearer:
            continue  # no estimator output: no cell to count it in
        if (str(r["protocol"]).startswith(NO_CONCORDANCE_ROUTE_PREFIXES)
                and sub not in FORWARD_SUBSTRATES):
            continue  # the BOLD arm's source view: not the battery's regime
        key = (sub, r["scoring"].observation_stage, r["view"], bearer)
        cell = cells.setdefault(key, {"on": 0, "on_events": 0, "off": 0,
                                      "off_events": 0, "on_concordant": 0,
                                      "off_concordant": 0, "off_undetermined": 0,
                                      "classes": set()})
        cell["classes"].add(r["system"])
        c = j.c
        # without a reference c is None; a concordant count at one state
        # (zero excess) is then still an ABSENT event, a positive excess on
        # a no-content run an event that cannot be told (recorded)
        x = excess_of(r["comp"]) if j.concordant and c is None else float("nan")
        if kind == DC.CONTENT_ON:
            cell["on"] += 1
            cell["on_concordant"] += int(j.concordant)
            cell["on_events"] += int(j.concordant and (
                abs(c) < DELTA if c is not None else x == 0.0))
        else:
            cell["off"] += 1
            cell["off_concordant"] += int(j.concordant)
            cell["off_events"] += int(j.concordant and c is not None and c > Z)
            cell["off_undetermined"] += int(j.concordant and c is None and x > 0.0)
    out = []
    for (sub, stage, view, bearer), c in sorted(cells.items(),
                                                key=lambda kv: str(kv[0])):
        absent = c["on"] >= CONCORDANCE_MIN_RUNS and c["on_events"] == 0
        present = c["off"] >= CONCORDANCE_MIN_RUNS and c["off_events"] == 0
        out.append({
            "substrate": sub, "observation_stage": stage, "view": view,
            "bearer": bearer, "min_runs": CONCORDANCE_MIN_RUNS,
            "content_on_runs": c["on"],
            "concordant_absent": c["on_events"],
            "content_on_concordant": c["on_concordant"],
            "no_content_runs": c["off"], "concordant_present": c["off_events"],
            "no_content_concordant": c["off_concordant"],
            "concordant_present_undetermined": c["off_undetermined"],
            "cp_upper_absent": _f(HE.cp_upper(c["on_events"], c["on"], 0.05))
            if c["on"] else None,
            "cp_upper_present": _f(HE.cp_upper(c["off_events"], c["off"], 0.05))
            if c["off"] else None,
            "classes": sorted(c["classes"]),
            "rule_absent": bool(absent), "rule_present": bool(present),
            "provisional": sub in FORWARD_SUBSTRATES,
        })
    return out


def concordance_cells(evidence: Sequence[dict], dec: Decisions) -> List[dict]:
    """The admitted cells the protocols carry: the rule's (``rule``), none,
    or the decision's explicit cells."""
    val = dec["pdi_concordance"]
    if val == "none":
        return []
    if val == "rule":
        chosen = [{"substrate": e["substrate"], "observation_stage":
                   e["observation_stage"], "view": e["view"], "bearer": e["bearer"],
                   "absent": e["rule_absent"], "present": e["rule_present"],
                   "provisional": e["provisional"],
                   "battery": {"content_on_runs": e["content_on_runs"],
                               "concordant_absent": e["concordant_absent"],
                               "no_content_runs": e["no_content_runs"],
                               "concordant_present": e["concordant_present"]}}
                  for e in evidence if e["rule_absent"] or e["rule_present"]]
    else:
        chosen = []
        for c in val:
            chosen.append({"substrate": c["substrate"],
                           "observation_stage": c["observation_stage"],
                           "view": c["view"], "bearer": c["bearer"],
                           "absent": bool(c.get("absent", False)),
                           "present": bool(c.get("present", False)),
                           "provisional": c["substrate"] in FORWARD_SUBSTRATES,
                           "battery": {}})
    return [{"principle": "PDI", **c} for c in chosen]


def _twin_q(alpha, df):
    return float(E.quantile(alpha, df if df and math.isfinite(df) else math.inf))


def se_calibration(data: DevData, protos: Mapping, dec: Decisions) -> dict:
    """SE calibration on the development twins (CD-2 to CD-4): per (principle,
    SE method, protocol, class[, direction]) the pooled within-network kappa,
    its 90 % chi-square interval and the ``q_A`` tail rates (the IIM block
    bootstrap also at the lowered df 9); the methods the rules suggest.
    An admitted concordant PDI component has no sampling SE (its error
    control is the battery): it forms no cell but counts among the class's
    sessions, so a cell's ``defined_share`` is the share of the class's twin
    sessions with a defined ``c`` and SE under the cell's method. A cell is
    eligible for the rules with a share >= 0.8 (HCv2-4) and a defined kappa
    (an RMS of the SEs of 0 leaves kappa undefined)."""
    iim_boot = _iim().SE_METHOD_DEFAULT
    admitted: Dict[tuple, object] = {}
    cal: Dict[tuple, Dict[int, list]] = {}
    sessions: Dict[tuple, set] = {}
    # a twin session enters once per SE method: a task run again (a tagged
    # re-run with the decided protocols) is taken from the untagged run,
    # else from the first run by path
    pairs = sorted(data.select(TWIN_DESIGNS, tagged=None),
                   key=lambda sr: (sr[0].tagged, sr[0].path))
    seen = set()
    for _src, rec in pairs:
        for s in rec.scorings:
            base = protos.get(s.protocol_id)
            if base is None:
                continue
            for p, comp in s.components.items():
                # every twin session of the class counts, an undefined
                # component (no SE method) and a concordant one among them
                for d in (E.NAS_DIRECTIONS if p == "NAS" else (None,)):
                    sessions.setdefault((p, s.protocol_id, rec.system, d), set()).add(
                        (rec.task_id, s.scoring_id))
                method = comp.se_method
                if method is None or method == E.SE_METHOD_CONCORDANT:
                    continue
                once = (rec.task_id, s.scoring_id, p, method)
                if once in seen:
                    continue
                seen.add(once)
                akey = (s.protocol_id, p, method)
                if akey not in admitted:
                    admitted[akey] = _admit_method(base, p, method)
                proto = admitted[akey]
                j = rejudge(rec, s, comp, proto)
                if p == "NAS":
                    groups = [(d, dict(j.members[d])) for d in sorted(j.members)]
                else:
                    m = {"c": j.c, "se": j.se_c, "df": j.df_c}
                    if p == "IIM" and method == iim_boot:
                        j9 = rejudge(rec, s, comp, proto, se_df_override=9.0)
                        m["df9"] = j9.df_c
                    groups = [(None, m)]
                for d, m in groups:
                    if m.get("c") is None or m.get("se") is None:
                        continue
                    k = (p, method, s.protocol_id, rec.system, d)
                    cal.setdefault(k, {}).setdefault(int(rec.seed), []).append(m)
    cells = []
    for (p, method, key, system, d), nets in sorted(cal.items(),
                                                    key=lambda kv: str(kv[0])):
        members = 2 if p == "NAS" else 1
        groups, ses, n = [], [], 0
        n_defined = sum(len(ms) for ms in nets.values())
        n_sessions = len(sessions.get((p, key, system, d), ())) or n_defined
        ev = {"below": [], "above": [], "below_df9": [], "above_df9": []}
        for _seed, ms in sorted(nets.items()):
            if len(ms) < 2:
                continue
            c = np.array([m["c"] for m in ms], dtype=float)
            groups.append(c)
            ses.extend(float(m["se"]) for m in ms)
            mean = float(c.mean())
            for m in ms:
                for suffix, df in (("", m.get("df")), ("_df9", m.get("df9"))):
                    if suffix and "df9" not in m:
                        continue
                    q = _twin_q(ALPHA_A / members, df)
                    ev["below" + suffix].append(m["c"] + q * m["se"] < mean)
                    ev["above" + suffix].append(m["c"] - q * m["se"] > mean)
            n += len(ms)
        sd, df = HE.pooled_within_sd(groups)
        rms = HE.rms(ses) if ses else float("nan")
        kap = sd / rms if df and rms > 0 else float("nan")
        lo, hi = (HE.kappa_interval(kap, df, KAPPA_LEVEL) if df
                  else (float("nan"), float("nan")))
        tails = {k: (float(np.mean(v)) if v else None) for k, v in ev.items()
                 if v or not k.endswith("_df9")}
        lo_f, hi_f, kap_f = _f(lo), _f(hi), _f(kap)
        both = lo_f is not None and hi_f is not None
        point_inside = bool(kap_f is not None
                            and KAPPA_BOUNDS[0] <= kap_f <= KAPPA_BOUNDS[1])
        share = n_defined / n_sessions
        cells.append({
            "principle": p, "se_method": method, "protocol": key, "class": system,
            "direction": d, "networks": len(groups), "sessions": n, "df": df,
            "kappa": kap_f, "interval": [lo_f, hi_f], "tails": tails,
            "defined_sessions": n_defined, "class_sessions": n_sessions,
            "defined_share": share,
            "eligible": bool(share >= DEFINED_SHARE_MIN and kap_f is not None),
            # CD-2 (and CD-4): the interval inside [0.8, 1.25]; entirely
            # outside it is the falsification criterion of HCv2-4
            "inside": bool(both and lo_f >= KAPPA_BOUNDS[0] and hi_f <= KAPPA_BOUNDS[1]),
            "outside": bool(both and (hi_f < KAPPA_BOUNDS[0] or lo_f > KAPPA_BOUNDS[1])),
            # the statement of HCv2-4 (a), reported beside it
            "hcv2_4_a": bool(both and point_inside and lo_f >= KAPPA_WIDE_BOUNDS[0]
                             and hi_f <= KAPPA_WIDE_BOUNDS[1]),
            "tails_ok": all(tails.get(k) is not None and tails[k] <= TAIL_MAX
                            for k in ("below", "above"))})
    return {"cells": cells, "suggestions": _se_suggestions(cells, data, protos)}


def _admit_method(proto, principle, method):
    methods = dict(proto.se_methods)
    have = set(methods.get(principle) or ())
    if method in have or method not in E.SE_METHODS:
        return proto
    payload = proto.to_dict()
    payload["se_methods"][principle] = sorted(have | {method})
    return E.ProtocolV3.from_dict(payload)


def calibrated(cell: Mapping) -> bool:
    """A method calibrated in a class (PRE_DATA_COMMITMENTS section 2, the
    binding reading of CD-2 to CD-4): the HCv2-4 (a) statement holds (kappa
    point in [0.8, 1.25], its 90 % interval inside [0.67, 1.5]) and both
    one-sided ``q_A`` tail rates are <= 0.02."""
    return bool(cell["hcv2_4_a"] and cell["tails_ok"])


def _class_of(cell: Mapping) -> tuple:
    return (cell["protocol"], cell["class"], cell["direction"])


def _keep_or_switch(cells: Sequence[dict], principle: str, default: str,
                    fallback: str) -> dict:
    """CD-3 and CD-4: keep the default method unless it fails to be
    calibrated in some eligible class and the fallback is calibrated in
    every class (each eligible class of either method); a method that is
    not better is never swapped in."""
    mine = [c for c in cells if c["principle"] == principle and c["eligible"]]
    dflt = {_class_of(c): c for c in mine if c["se_method"] == default}
    fb = {_class_of(c): c for c in mine if c["se_method"] == fallback}
    default_fails = sorted(k for k, c in dflt.items() if not calibrated(c))
    fallback_ok = bool(fb) and all(k in fb and calibrated(fb[k])
                                   for k in set(dflt) | set(fb))
    all_default = [c for c in cells if c["principle"] == principle
                   and c["se_method"] == default]
    return {
        "value": (None if not dflt else fallback if default_fails and fallback_ok
                  else default),
        "classes": len(dflt), "calibrated_classes": len(dflt) - len(default_fails),
        "fallback": fallback, "fallback_classes": len(fb),
        "fallback_calibrated_classes": sum(calibrated(c) for c in fb.values()),
        "fallback_calibrated_everywhere": fallback_ok,
        "not_eligible_classes": len(all_default) - len(dflt),
        # the falsification side of HCv2-4 and the literal reading of design
        # 2.1 item 9 (interval inside [0.8, 1.25]), reported beside the rule
        "failing_classes": sum(c["outside"] for c in dflt.values()),
        "classes_not_inside": sum(not c["inside"] for c in dflt.values())}


def _se_suggestions(cells: Sequence[dict], data: DevData, protos: Mapping) -> dict:
    """The rules' suggestions for the SE decisions from the eligible twin
    cells (:func:`calibrated`; cells below the defined share or without a
    kappa are left out)."""
    nas = _nas()
    out = {}
    by_method: Dict[str, List[dict]] = {}
    for c in cells:
        if c["principle"] == "NAS" and c["eligible"]:
            by_method.setdefault(c["se_method"], []).append(c)
    # every class with an eligible cell of some method: a method without an
    # eligible cell in one of them is not calibrated there
    classes = {_class_of(c) for cs in by_method.values() for c in cs}
    good = [m for m, cs in by_method.items()
            if {_class_of(c) for c in cs if calibrated(c)} >= classes]
    inside = [m for m, cs in by_method.items()
              if cs and all(c["inside"] and c["tails_ok"] for c in cs)]

    def rank(m):
        scheme, groups = m.split("_")[1], int(m.rsplit("_", 1)[1])
        return (groups - 1, scheme == "contiguous", -groups)

    out["nas_se_method"] = {
        "value": (None if not by_method else max(good, key=rank) if good
                  else nas.SE_METHOD_DEFAULT),
        "candidates": sorted(by_method), "calibrated": sorted(good),
        "calibrated_classes": {m: sum(calibrated(c) for c in cs)
                               for m, cs in sorted(by_method.items())},
        "classes": {m: len(cs) for m, cs in sorted(by_method.items())},
        "rule": "the method calibrated in every class (HCv2-4 (a) statement: kappa "
                "point in [0.8, 1.25] and its 90 % interval inside [0.67, 1.5]; q_A "
                "tails <= 0.02) with the largest df; ties go to the contiguous "
                "scheme; none calibrated: contiguous G = 10",
        "calibrated_by_the_interval_inside_reading": sorted(inside),
        "note": "reported beside the rule: the literal reading of design 2.1 item 9 "
                "(interval inside [0.8, 1.25]); at df 30 it holds only for a kappa "
                "point in about [0.97, 0.98] (PRE_DATA_COMMITMENTS section 2)"}
    iim = _iim()
    out["iim_se_method"] = dict(
        _keep_or_switch(cells, "IIM", iim.SE_METHOD_DEFAULT, DC.IIM_SE_FALLBACK),
        rule="block bootstrap unless it fails to be calibrated in some class and "
             "the contiguous jackknife G = 10 is calibrated in every class")
    boot = [c for c in cells if c["principle"] == "IIM" and c["eligible"]
            and c["se_method"] == iim.SE_METHOD_DEFAULT]
    tail_bad = [c for c in boot if not c["tails_ok"]]
    out["iim_bootstrap_se_df"] = {
        "value": None if not boot else (9.0 if tail_bad else 12.0),
        "rule": "12; 9 if a one-sided q_A tail of the bootstrap exceeds 0.02",
        "classes_with_tail_above": len(tail_bad)}
    out["ram_se_method"] = dict(
        _keep_or_switch(cells, "RAM", "shift_null_sd", DC.RAM_SE_FALLBACK),
        rule="shift-null SD unless it fails to be calibrated in some class and the "
             "trial jackknife G = 10 is calibrated in every class")
    return out


def occupancy_evidence(data: DevData) -> dict:
    """CD-3 occupancy gate on the development cells of HCv2-12 (c), per
    ``N_min`` the cells were designated and scored at (the default, and the
    raised values of the optional item ``occupancy_n_min``): cells
    designated ``undefined`` (``T pi_min <= N_min / 2``) must be
    ``INSUFFICIENT_OCCUPANCY`` in >= 90 %, cells designated ``defined``
    (``>= 4 N_min``) defined in >= 90 %. The suggestion is the smallest
    ``N_min`` at which every cell passes, the default first."""
    default = int(_iim().N_MIN_DEFAULT)
    cells: Dict[tuple, dict] = {}
    for _src, rec in data.select(("family_b",), tagged=None):
        cell = (rec.config or {}).get("cell") or {}
        exp = cell.get("expected") or {}
        des = exp.get("occupancy")
        if "c" not in (cell.get("parts") or []) or des is None:
            continue
        n_min = int(exp.get("n_min") or default)
        for s in rec.scorings:
            if s.estimator_form != "directional":
                continue
            c = s.components.get("IIM")
            if c is None:
                continue
            row = cells.setdefault((n_min, cell["cell_id"]), {
                "cell": cell["cell_id"], "designation": des,
                "t_pi_min": exp.get("t_pi_min"), "n_min": n_min, "n": 0,
                "seeds": set(), "insufficient": 0, "defined": 0})
            if int(rec.seed) in row["seeds"]:
                continue  # one run per seed (a re-run of the same cell)
            row["seeds"].add(int(rec.seed))
            row["n"] += 1
            row["insufficient"] += int(R.parse_reason(c.reason)[0]
                                       == R.INSUFFICIENT_OCCUPANCY
                                       if c.reason else False)
            row["defined"] += int(bool((c.details or {}).get("defined")))
    out = []
    for key in sorted(cells):
        r = cells[key]
        del r["seeds"]
        rate = (r["insufficient"] if r["designation"] == "undefined"
                else r["defined"]) / r["n"] if r["n"] else None
        r["rate"] = rate
        r["pass"] = rate is not None and rate >= 0.9
        out.append(r)
    by_n: Dict[int, List[dict]] = {}
    for r in out:
        by_n.setdefault(r["n_min"], []).append(r)
    all_pass = {n: all(r["pass"] for r in rs) for n, rs in sorted(by_n.items())}
    passing = [n for n in sorted(all_pass) if all_pass[n] and n >= default]
    if default not in all_pass:
        suggestion = None
    elif all_pass[default]:
        suggestion = default
    elif passing:
        suggestion = passing[0]
    else:
        suggestion = ("raise N_min (50 or 100): run the optional item occupancy_n_min "
                      "and build again")
    return {"cells": out, "all_pass": all_pass.get(default),
            "all_pass_by_n_min": {str(n): v for n, v in all_pass.items()},
            "suggestion": suggestion,
            "rule": f"N_min stays {default} unless a development occupancy cell fails "
                    "its 90 % criterion; then the smallest raised N_min (50, 100) at "
                    "which every cell passes"}


def witness_dose(system: str, principle: str) -> Optional[float]:
    """The dose ratio of a witness system's ``principle`` mechanism, from the
    v2 system catalogue (``bench/witnesses_v2.yaml``): an agent witness's
    knobs against the nominal knobs (0 where the knobs switch the mechanism
    off, ``Knobs.bits``; else the principle's own knob over its nominal
    value); a witness without knobs (the nulls, the hypersynchronous
    oscillator) by its intended pattern. None for a system the catalogue
    does not list as a witness, a patchwork
    (:data:`MECHANISM_EXCLUDED_GENERATORS`) or a principle the pattern
    leaves unspecified."""
    from impact_pipeline.bench import adversarial_v2 as AV
    from impact_pipeline.bench.generators import NOMINAL_KNOBS, knobs_from_dict

    try:
        entry = AV.get_entry(system)
    except KeyError:
        return None
    gen = entry.get("generator")
    if entry["kind"] != "witness" or gen in MECHANISM_EXCLUDED_GENERATORS:
        return None
    i = PRINCIPLES.index(principle)
    if gen == "family_a":
        knobs = knobs_from_dict(entry.get("knobs") or {})
        if not knobs.bits()[i]:
            return 0.0
        own = OWN_KNOB[principle]
        return float(getattr(knobs, own)) / float(getattr(NOMINAL_KNOBS, own))
    pattern = entry.get("intended_pattern") or ()
    v = pattern[i] if i < len(pattern) else None
    return None if v is None else float(v)


def dose_ratio(design: str, system: str, tags: Mapping, principle: str
               ) -> Optional[float]:
    """The dose of ``principle``'s mechanism in a row of the mechanism-on
    designs relative to its nominal dose: the swept level of the principle's
    own knob over its nominal value (1 for another knob), the factorial
    cell's bit, the witness's catalogue dose (:func:`witness_dose`); None
    outside these designs or where no dose is defined."""
    from impact_pipeline.bench.generators import NOMINAL_KNOBS

    nominal = NOMINAL_KNOBS.to_dict()
    if design in SWEEP_DESIGNS:
        knob, level = tags.get("sweep_knob"), _f(tags.get("sweep_level"))
        if knob is None or level is None:
            return None
        return (level / float(nominal[knob]) if knob == OWN_KNOB.get(principle)
                and float(nominal[knob]) else 1.0)
    if design in FACTORIAL_DESIGNS:
        # the cell's mechanism bits in PRINCIPLES order (Knobs.bits)
        bits = tags.get("bits") or []
        if len(bits) != len(PRINCIPLES):
            return None
        return float(bits[PRINCIPLES.index(principle)])
    if design in WITNESS_DESIGNS:
        return witness_dose(system, principle)
    return None


def _dose_on(ratio: float) -> bool:
    return ratio >= T.MECHANISM_ON_DOSE_RATIO - 1e-9


def dose_only_plan() -> List[dict]:
    """The rows of the mechanism-on designs whose development output the
    held-out rule withholds (:data:`DOSE_ONLY_MECHANISM_CELLS`): per
    (family, declaration, principle, design, system) of the development plan
    the task tags (no task is run)."""
    out, seen = [], set()
    tasks = RB.build_tasks(list(MECHANISM_DESIGNS), S.DEVELOPMENT)
    for t in tasks:
        for fam, p in DOSE_ONLY_MECHANISM_CELLS:
            if t.family != fam:
                continue
            for s in t.scorings:
                if s.estimator_form != D.PRIMARY_FORM:
                    continue
                k = (fam, s.declaration_id, p, t.design, t.system)
                if k not in seen:
                    seen.add(k)
                    out.append({"family": fam, "declaration": s.declaration_id,
                                "principle": p, "design": t.design,
                                "system": t.system, "tags": dict(t.tags or {})})
    return sorted(out, key=lambda e: str((e["family"], e["declaration"],
                                         e["principle"], e["design"], e["system"])))


def mechanism_on(rows: Sequence[dict], dose_only: Sequence[Mapping] = ()
                 ) -> Tuple[List[dict], List[dict]]:
    """The CD-8 labels: per (family, declaration, principle, design, system)
    of the dry-run sweeps, factorial and witnesses (every witness the
    catalogue gives a dose, :func:`dose_ratio`), the development median c
    and the dose ratio; "on" iff the ratio >= 0.5 and the median >= 2 delta.
    The rows of ``dose_only`` (:func:`dose_only_plan`; cells without
    development rows) follow the rule's entries, labelled by the dose
    condition alone. Returns the entries (hypotheses-file format, first
    match decides) and the evidence."""
    groups: Dict[tuple, List[float]] = {}
    meta: Dict[tuple, dict] = {}
    for r in rows:
        if r["form"] != D.PRIMARY_FORM or r["record"].replicate:
            continue
        d, p = r["design"], r["principle"]
        if d not in MECHANISM_DESIGNS:
            continue
        ratio = dose_ratio(d, r["system"], r["tags"], p)
        if ratio is None:
            continue
        k = (r["family"], r["declaration"], p, d, r["system"])
        groups.setdefault(k, [])
        if r["judged"].c is not None:
            groups[k].append(r["judged"].c)
        meta[k] = {"dose_ratio": ratio}
    entries, evidence = [], []
    for k in sorted(groups, key=str):
        fam, decl, p, d, system = k
        vals = groups[k]
        med = float(np.median(vals)) if vals else None
        ratio = meta[k]["dose_ratio"]
        on = bool(_dose_on(ratio) and med is not None
                  and med >= T.MECHANISM_ON_C_RATIO * DELTA)
        entries.append({"principle": p, "family": fam, "declaration": decl,
                        "where": {"design": d, "system": system}, "on": on})
        evidence.append({"family": fam, "declaration": decl, "principle": p,
                         "design": d, "system": system, "n": len(vals),
                         "median_c": med, "dose_ratio": ratio, "on": on})
    for row in dose_only:
        k = (row["family"], row["declaration"], row["principle"], row["design"],
             row["system"])
        if k in groups:
            continue
        ratio = dose_ratio(row["design"], row["system"], row.get("tags") or {},
                           row["principle"])
        if ratio is None:
            continue
        fam, decl, p, d, system = k
        on = _dose_on(ratio)
        entries.append({"principle": p, "family": fam, "declaration": decl,
                        "where": {"design": d, "system": system}, "on": on})
        evidence.append({"family": fam, "declaration": decl, "principle": p,
                         "design": d, "system": system, "n": 0, "median_c": None,
                         "dose_ratio": ratio, "on": on, "dose_only": True})
    return entries, evidence


def iim_agents_expectation(rows: Sequence[dict]) -> dict:
    """The development expectation of HCv2-14 (CD-6) under the v2 estimator:
    per family protocol the paired IIM-dir contrast PC_nominal minus
    W_IIM_feedforward (median, share >= 0.5, n), the PRESENT rates of the
    positive control, the feed-forward witness and O_inert, and Spearman's
    rho of c with c_int over the dry-run sweep."""
    from scipy import stats as st

    by: Dict[tuple, Dict[int, float]] = {}
    status: Dict[tuple, List[str]] = {}
    sweep: Dict[str, List[Tuple[float, float]]] = {}
    for r in rows:
        if r["principle"] != "IIM" or r["form"] != D.PRIMARY_FORM:
            continue
        if r["record"].replicate:
            continue
        key = r["protocol"]
        if r["design"] in WITNESS_DESIGNS + ANCHOR_DESIGNS:
            if r["judged"].c is not None:
                by.setdefault((key, r["system"]), {})[r["seed"]] = r["judged"].c
            status.setdefault((key, r["system"]), []).append(r["judged"].status)
        if r["design"] in SWEEP_DESIGNS and r["tags"].get("sweep_knob") == "c_int":
            if r["judged"].c is not None:
                sweep.setdefault(key, []).append((float(r["tags"]["sweep_level"]),
                                                  r["judged"].c))
    out = {"flag": DEVELOPMENT_FLAG, "protocols": {}}
    for key in sorted({k[0] for k in list(by) + list(status)}):
        pc, ff = by.get((key, "PC_nominal"), {}), by.get((key, "W_IIM_feedforward"), {})
        common = sorted(set(pc) & set(ff))
        delta = [pc[s] - ff[s] for s in common]

        def rate(system):
            st_ = status.get((key, system), [])
            return {"n": len(st_), "present": (sum(x == R.PRESENT for x in st_)
                                               / len(st_)) if st_ else None}

        entry = {"paired_delta_c": {"n": len(delta),
                                    "median": (float(np.median(delta)) if delta
                                               else None),
                                    "share_ge_0.5": (float(np.mean(np.asarray(delta)
                                                                   >= 0.5))
                                                     if delta else None)},
                 "present": {s: rate(s) for s in ("PC_nominal", "W_IIM_feedforward",
                                                  "O_inert")}}
        pts = sweep.get(key, [])
        if len(pts) >= 3:
            rho, pval = st.spearmanr([a for a, _ in pts], [b for _, b in pts])
            entry["spearman_c_int"] = {"n": len(pts), "rho": _f(rho),
                                       "p_two_sided": _f(pval)}
        out["protocols"][key] = entry
    return out


def replication_power(series_by_key: Mapping, entries_by_key: Mapping) -> dict:
    """CD-11 for the anchors: per primary protocol of each anchor design
    (:data:`PRIMARY_PROTOCOLS_OF_ANCHOR_DESIGN`) and principle the
    probability that its development status (valid and specific; valid,
    non-specific; invalid) recurs on a replication block of 20 (and 40)
    seeds, by resampling the reference-block seeds (fixed stream). The
    forward views and forms replicate validity only (HCv2-6 (b)); their
    status is valid or invalid. A protocol without reference-block runs
    (a forward view before the release) has no entry."""
    from impact_pipeline.bench.designs_v2 import anchors as AN

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out = {}
    for design, keys in PRIMARY_PROTOCOLS_OF_ANCHOR_DESIGN.items():
        validity_only = design in VALIDITY_ONLY_ANCHOR_DESIGNS
        per = {}
        for key in keys:
            series = series_by_key.get(key)
            entries = entries_by_key.get(key)
            if series is None or entries is None:
                continue
            for p in PRINCIPLES:
                names = ([f"NAS:{d}" for d in E.NAS_DIRECTIONS] if p == "NAS" else [p])
                if not all(n in series for n in names):
                    continue
                want = (entries[p]["status"] if not validity_only
                        else VALID if entries[p]["valid"] else T.ANCHOR_INVALID)
                res = {}
                for size in (20, 40):
                    idx = rng.integers(0, T.ANCHOR_BLOCK_SIZE, (BOOTSTRAP_DRAWS, size))
                    st_ = _resampled_statuses(series, names, idx,
                                              validity_only=validity_only)
                    res[str(size)] = float(np.mean(st_ == want))
                per[f"{key}|{p}"] = {"development_status": want, "power": res}
        low = sorted(k for k, v in per.items() if v["power"]["20"] < 0.9)
        out[design] = {"anchors": per, "below_0.9_at_20": low,
                       "suggest_extended": bool(low), "validity_only": validity_only}
        if design == FORWARD_ANCHOR_DESIGN:
            # one anchor run serves every view of its arm: the arm extends
            below = {_forward_arm_view_of_key(k.split("|")[0])[0] for k in low}
            out[design]["suggest_extended_by_arm"] = {
                arm: arm in below for arm in sorted(AN.FORWARD_ARMS)}
    return out


def replication_suggestion(replication: Mapping) -> Optional[dict]:
    """The CD-11 suggestion from :func:`replication_power`: extend a design
    whose replication power at 20 seeds is below 0.9 somewhere. The family
    designs need their anchors (else no suggestion); the forward design
    joins once its reference block exists (after the release), per arm:
    an arm extends where one of its views or forms is below 0.9."""
    designs = [d for d in replication if d not in VALIDITY_ONLY_ANCHOR_DESIGNS
               or replication[d]["anchors"]]
    if not all(replication[d]["anchors"] for d in designs):
        return None
    value = {d: (replication[d]["suggest_extended_by_arm"]
                 if d == FORWARD_ANCHOR_DESIGN else replication[d]["suggest_extended"])
             for d in sorted(designs)}
    return {"value": value,
            "rule": "extend where the replication power at 20 seeds is below 0.9 "
                    "(a forward arm where one of its views is)"}


def _lower_bounds(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per row of ``x`` (NaN = missing): the count, the mean and the one-sided
    95 % Student-t lower bound of the mean (NaN with fewer than two values)."""
    n = np.sum(np.isfinite(x), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        # rows without (two) finite values are NaN by design
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(np.where(np.isfinite(x), x, np.nan), axis=1)
        sd = np.nanstd(np.where(np.isfinite(x), x, np.nan), axis=1, ddof=1)
        qs = {int(k): (E.quantile(T.ANCHOR_ALPHA, int(k) - 1) if k >= 2 else np.nan)
              for k in np.unique(n)}
        q = np.array([qs[int(k)] for k in n], dtype=float)
        lower = np.where(n >= 2, mean - q * sd / np.sqrt(np.maximum(n, 1)), np.nan)
    return n, mean, lower


def _resampled_statuses(series, names, idx: np.ndarray, *,
                        validity_only: bool = False) -> np.ndarray:
    """The anchor status of each resampled block (rows of ``idx``): valid iff
    at least 36/40 of the block's values are finite and the t lower bound is
    > 0; specific iff the paired contrast's mean is >= 0.5 x the anchor and
    its lower bound > 0 (every direction of NAS). ``validity_only``: valid
    (:data:`VALID`) or invalid."""
    size = idx.shape[1]
    min_finite = math.ceil(T.ANCHOR_MIN_FINITE * size / T.ANCHOR_BLOCK_SIZE)
    valid = np.ones(idx.shape[0], dtype=bool)
    specific = np.ones(idx.shape[0], dtype=bool)
    for name in names:
        pc, les = series[name]["pc"][idx], series[name]["lesion"][idx]
        n, mean, lower = _lower_bounds(pc)
        valid &= (n >= min_finite) & np.nan_to_num(lower > 0, nan=False)
        if validity_only:
            continue
        d = np.where(np.isfinite(pc) & np.isfinite(les), pc - les, np.nan)
        _nd, dm, dl = _lower_bounds(d)
        with np.errstate(invalid="ignore"):
            specific &= np.nan_to_num((dm >= T.SPECIFICITY_RATIO * mean) & (dl > 0),
                                      nan=False)
    if validity_only:
        return np.where(valid, VALID, T.ANCHOR_INVALID)
    return np.where(~valid, T.ANCHOR_INVALID,
                    np.where(specific, T.ANCHOR_VALID_SPECIFIC,
                             T.ANCHOR_VALID_NONSPECIFIC))


def dependency_evidence(rows: Sequence[dict], protos: Mapping) -> List[dict]:
    """HCv2-22 (iii) on development data: per (protocol, single-deficit
    witness, other principle of ``N_anch``) the share of seeds with
    ``|c_p(w) - c_p(PC)| < z`` (paired on the seed)."""
    targets = _witness_targets()
    by: Dict[tuple, Dict[int, float]] = {}
    for r in rows:
        if r["form"] != D.PRIMARY_FORM or r["design"] not in TESTABILITY_DESIGNS:
            continue
        if r["judged"].c is not None:
            by.setdefault((r["protocol"], r["system"], r["principle"]),
                          {})[r["seed"]] = r["judged"].c
    out = []
    for key in GATE_PROTOCOLS:
        proto = protos.get(key)
        if proto is None:
            continue
        n_anch = _necessity_set(proto) or set()
        for w, target in sorted(targets.items()):
            for p in PRINCIPLES:
                if p == target or p not in n_anch:
                    continue
                a, b = by.get((key, w, p), {}), by.get((key, "PC_nominal", p), {})
                common = sorted(set(a) & set(b))
                if not common:
                    continue
                d = np.array([abs(a[s] - b[s]) for s in common])
                out.append({"protocol": key, "witness": w, "principle": p,
                            "n": len(common), "share_within_z": float(np.mean(d < Z))})
    return out


def paired_ratio(comp, pc) -> Optional[float]:
    """The paired ratio ``r_P(X, s) = excess_P(X, s) / excess_P(PC, s)`` of a
    component and the positive control's on the same seed (NAS: the smaller
    of the two directions' ratios); None without a positive control
    excess."""
    if comp.principle == "NAS" and (comp.details or {}).get("members"):
        ratios = []
        for d in E.NAS_DIRECTIONS:
            a, b = excess_of(comp, d), excess_of(pc, d)
            if not (math.isfinite(a) and math.isfinite(b) and b > 0):
                return None
            ratios.append(a / b)
        return min(ratios)
    a, b = excess_of(comp), excess_of(pc)
    if not (math.isfinite(a) and math.isfinite(b) and b > 0):
        return None
    return a / b


def adversary_potency(rows: Sequence[dict]) -> List[dict]:
    """CD-9: per kept adversary (and variant), declaration and principle the
    share of development seeds whose paired ratio with PC_nominal on the
    same seed (same protocol and declaration) is >= 0.5, the PRESENT rate,
    and whether the adversary is potent (either share >= 0.5); the design's
    potency is the one under H. The share of seeds with ``c >= 0.5`` is
    reported beside it. Reported only, never used to select conditions."""
    pcs: Dict[tuple, object] = {}
    for r in rows:
        if (r["system"] == "PC_nominal" and r["form"] == D.PRIMARY_FORM
                and r["design"] in WITNESS_DESIGNS + ANCHOR_DESIGNS
                and not r["record"].replicate):
            pcs[(r["family"], r["protocol"], r["declaration"], r["principle"],
                 r["seed"])] = r["comp"]
    groups: Dict[tuple, List[dict]] = {}
    for r in rows:
        if r["design"] not in ADVERSARY_DESIGNS or r["form"] != D.PRIMARY_FORM:
            continue
        variant = ((r["record"].config or {}).get("params") or {}).get("variant")
        k = (r["system"], variant or "", r["declaration"], r["principle"])
        groups.setdefault(k, []).append(r)
    out = []
    for (system, variant, decl, p), rs in sorted(groups.items()):
        js = [r["judged"] for r in rs]
        ratios = []
        for r in rs:
            pc = pcs.get((r["family"], r["protocol"], decl, p, r["seed"]))
            v = None if pc is None else paired_ratio(r["comp"], pc)
            if v is not None:
                ratios.append(v)
        share_r = (float(np.mean(np.asarray(ratios) >= POTENCY_C)) if ratios
                   else None)
        cs = [j.c for j in js if j.c is not None]
        share_c = float(np.mean(np.asarray(cs) >= POTENCY_C)) if cs else None
        present = float(np.mean([j.status == R.PRESENT for j in js]))
        out.append({"adversary": system, "variant": variant or None,
                    "declaration": decl, "principle": p, "n": len(js),
                    "n_paired": len(ratios),
                    "median_paired_ratio": (float(np.median(ratios)) if ratios
                                            else None),
                    "share_paired_ratio_ge_0.5": share_r,
                    "share_c_ge_0.5": share_c, "present_rate": present,
                    "under_h": decl == "H",
                    "potent": bool((share_r or 0.0) >= POTENCY_SHARE
                                   or present >= POTENCY_SHARE)})
    return out


def oracle_checks(data: DevData) -> Optional[dict]:
    """CD-13: the development oracle checks (prerequisite M and the
    realisation checks): checks not usable and checks near their threshold
    (flagged; pass rates near 0.9)."""
    import csv

    path = data.root / "oracle_checks" / "manipulation" / "manipulation_summary.csv"
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    def label(r):
        return "/".join(x for x in (r.get("kind"), r.get("family"), r.get("switch"),
                                    r.get("system_id"), r.get("variant"),
                                    r.get("check")) if x)

    return {"file": str(path.relative_to(data.root)), "sha256": _file_sha(path),
            "n_checks": len(rows),
            "not_usable": sorted(label(r) for r in rows
                                 if str(r.get("usable")).lower() != "true"),
            "near_threshold": sorted(label(r) for r in rows
                                     if str(r.get("near_threshold")).lower() == "true")}


def null_calibration_ram(data: DevData) -> dict:
    """RAM-PE on the development null-calibration cells: rows and their
    UNDEFINED reasons (the evidence of the declared option)."""
    reasons: Dict[str, int] = {}
    n = defined = 0
    for _src, rec in data.select(("null_calibration",)):
        for s in rec.scorings:
            c = s.components.get("RAM")
            if c is None:
                continue
            n += 1
            defined += int(bool((c.details or {}).get("defined")))
            if c.status == R.UNDEFINED:
                code = R.parse_reason(c.reason)[0] if c.reason else "?"
                reasons[code] = reasons.get(code, 0) + 1
    return {"n": n, "estimator_defined": defined, "undefined_reasons": reasons,
            "options": {"keep": "the rows stay (UNDEFINED, never PRESENT)",
                        "drop": "the null-calibration scorings leave RAM out "
                                "(designs_v2.null_calibration); HCv2-1 and HCv2-3 "
                                "then have no RAM cells",
                        "require_defined": "HCv2-1 and HCv2-3 count only defined RAM "
                                           "rows (protocols/v2/hypotheses_v2.json)"}}


def grain_cap_evidence(data: DevData) -> dict:
    caps, refused = set(), 0
    for _src, rec in data.records:
        st_ = (rec.config or {}).get("settings") or {}
        if "iim_max_macro_nodes" in st_:
            caps.add(st_["iim_max_macro_nodes"])
        for s in rec.scorings:
            c = s.components.get("IIM")
            if c is not None and c.reason and "MacroGrainTooLargeError" in c.reason:
                refused += 1
    return {"caps_in_records": sorted(caps, key=lambda v: (v is None, v)),
            "components_refused_by_the_cap": refused}


# --------------------------------------------------------------------------
# the build
# --------------------------------------------------------------------------
def anchored_keys() -> List[str]:
    """The protocol keys that carry their own anchors: the scorings of the
    reference-block designs (families A and C1, the RAM-only arm with every
    form that has its own anchor, and every forward view) and the v1
    quadrant forms of the Hopf EEG views (:data:`FORWARD_OWN_ANCHOR_FORMS`,
    scored on the anchor condition of the released reference block only)."""
    tasks = RB.build_tasks(list(ANCHOR_DESIGNS), S.DEVELOPMENT)
    tasks += RB.build_tasks([FORWARD_ANCHOR_DESIGN], S.DEVELOPMENT,
                            purpose="reference_development")
    return sorted(set(RB.protocol_keys(tasks)) | set(FORWARD_OWN_ANCHOR_FORMS))


def reference_plan() -> Dict[Tuple[str, str], set]:
    """``{(protocol key, system): principles}`` the development reference
    blocks of families A and C1 and of the RAM-only arm score."""
    out: Dict[Tuple[str, str], set] = {}
    for t in RB.build_tasks(list(ANCHOR_DESIGNS), S.DEVELOPMENT):
        for s in t.scorings:
            out.setdefault((s.protocol_key, t.system), set()).update(s.principles)
    return out


def lesion_coverage(pairs, key: str, plan: Mapping) -> Dict[str, int]:
    """Per principle whose own lesion the reference plan scores under
    ``key``: the reference seeds on which the lesion's record has that
    component (the specificity gate pairs PC and lesion by seed; a missing
    lesion run must not pass as a non-specific anchor)."""
    from impact_pipeline.bench.designs_v2 import anchors as AN

    out = {}
    for p, lesion in sorted(AN.OWN_LESION.items()):
        if p not in plan.get((key, lesion), ()):
            continue
        out[p] = len({int(r.seed) for _s, r in pairs
                      if r.system == lesion and not r.replicate
                      and int(r.seed) in S.REFERENCE_BLOCK
                      and any(s.protocol_id == key and p in s.components
                              for s in r.scorings)})
    return out


def build(decisions=None, dev_root=DC.DEFAULT_ROOT, *, files: Optional[Sequence] = None,
          keys: Optional[Sequence[str]] = None) -> dict:
    """Everything the builder writes, in memory: ``{"protocols": {key:
    ProtocolV3}, "files": {name: payload}, "manifest": ..., "evidence": ...,
    "decisions": ...}``. ``keys`` replaces the planned protocol keys (tests)."""
    from impact_pipeline.bench.designs_v2 import anchors as AN

    dec = decisions if isinstance(decisions, Decisions) else load_decisions(decisions)
    data = DevData(dev_root, files=files)
    planned = list(keys) if keys is not None else confirmatory_protocol_keys()
    fb = family_b_protocols()
    own_keys = [k for k in anchored_keys() if keys is None or k in planned]
    blocking: List[str] = []
    seeds = list(S.REFERENCE_BLOCK)

    # 1. anchors: the development reference blocks (validity and the
    # specificity gate) and the forward views' anchor conditions (validity)
    regime = dec["forward_anchor_regime"]
    purpose = "reference" if regime == "held_out" else "reference_development"
    ref_pairs = data.select(ANCHOR_DESIGNS)
    fwd_pairs = [(s, r) for s, r in data.select((FORWARD_ANCHOR_DESIGN,))
                 if (r.config.get("tags") or {}).get("purpose") == purpose]
    fwd_keys = set(_forward_keys())
    ref_plan = reference_plan()
    series_by_key, entries_by_key, refs, anchor_info = {}, {}, {}, {}
    forward_anchor_rows = []
    for key in own_keys:
        forward = key in fwd_keys
        pairs = fwd_pairs if forward else ref_pairs
        # the forward views anchor on their anchor condition only (PC_nominal
        # or G_nom; design 3.6), validity only
        positive = _is_forward_anchor if forward else _is_positive_control
        series = anchor_series(
            pairs, key, positive=positive,
            lesion=None if forward else (lambda r, p: r.system == AN.OWN_LESION.get(p)),
            seeds=seeds)
        entries, ref = anchor_entries(series, with_specificity=not forward)
        series_by_key[key], entries_by_key[key], refs[key] = series, entries, ref
        n_ref = len({int(r.seed) for _s, r in pairs
                     if positive(r) and not r.replicate
                     and int(r.seed) in S.REFERENCE_BLOCK
                     and any(s.protocol_id == key for s in r.scorings)})
        anchor_info[key] = {"n_reference_seeds": n_ref,
                            "regime": regime if forward else None,
                            "status": {p: entries[p]["status"] for p in PRINCIPLES}}
        if n_ref < T.ANCHOR_BLOCK_SIZE:
            blocking.append(f"anchors of {key}: {n_ref} of {T.ANCHOR_BLOCK_SIZE} "
                            "reference seeds")
        if not forward:
            cover = lesion_coverage(pairs, key, ref_plan)
            anchor_info[key]["n_lesion_seeds"] = cover
            for p, n_les in cover.items():
                if n_les < T.ANCHOR_BLOCK_SIZE:
                    blocking.append(f"anchors of {key}: own lesion of {p} on {n_les} "
                                    f"of {T.ANCHOR_BLOCK_SIZE} reference seeds")
        if forward and _split(key)[1] == D.PRIMARY_FORM:
            arm, view = _forward_arm_view_of_key(key)
            for p in _forward_principles(key):
                e = entries[p]
                validity = (e.get("validity") if p != "NAS" else
                            {d: e["members"][d]["validity"] for d in E.NAS_DIRECTIONS})
                forward_anchor_rows.append({"arm": arm, "view": view, "principle": p,
                                            "valid": bool(e["valid"]), "protocol": key,
                                            "regime": regime, "validity": validity})
    if regime != "held_out":
        blocking.append("forward anchors at the development regime (the frozen "
                        "anchors are the held-out regime's, after the release)")
    blocking.extend(release_problems(data))

    def source_text(key):
        info = anchor_info.get(key, {})
        where = (f"forward reference block 900-939 at the {info.get('regime')} regime"
                 if key in fwd_keys else "development reference block 900-939")
        return (f"MPC-Bench v2 {where}: the positive control's mean excess over its "
                f"null (n = {info.get('n_reference_seeds', 0)} seeds), scale excess, "
                f"anchor rule {json.dumps(T.ANCHOR_RULE, sort_keys=True)}; "
                f"{BUILDER_VERSION}")

    drafts: Dict[str, dict] = {}

    def draft(key):
        if key not in drafts:
            drafts[key] = draft_payload(key)
        return copy.deepcopy(drafts[key])

    def anchored(key, concordance=(), precision=None):
        return assemble(key, draft(key), dec,
                        reference=reference_block(refs[key], source_text(key)),
                        anchors=T.anchors_block(entries_by_key[key]),
                        concordance=concordance, precision=precision)

    # 2. the concordance battery judged on the first-stage protocols
    stage1 = {k: anchored(k) for k in own_keys}
    dev_designs = sorted(set(TESTABILITY_DESIGNS) | set(SWEEP_DESIGNS)
                         | set(FACTORIAL_DESIGNS) | set(ADVERSARY_DESIGNS)
                         | {DC.BATTERY_DESIGN} | set(_forward_admission_designs()))
    conc_ev = concordance_evidence(judged_rows(data, stage1, dev_designs, dec=dec))
    cells = concordance_cells(conc_ev, dec)

    # 3. the testability rows under the protocols with the concordance cells,
    # then the precision blocks
    stage2 = {k: anchored(k, concordance=cells) for k in own_keys}
    rows2 = judged_rows(data, stage2, dev_designs, dec=dec)
    prec_rows, table = testability(rows2, dec)
    fb_row, fb_info = family_b_rows(data, dec["iim_bootstrap_se_df"], fb)
    table.append(fb_info)
    for t in table:
        if not t.get("se_method_matches_decision", True):
            blocking.append(f"testability {t['family']}/{t['principle']}/"
                            f"{t['witness']}: computed with the SE methods "
                            f"{t['se_methods']}, not the decided one (re-run with the "
                            "protocols of the decisions)")
    gate_missing = _gates_without_rows(table, prec_rows + ([fb_row] if fb_row else []),
                                       stage2)
    blocking.extend(f"testability gate without a row: {g}" for g in gate_missing)
    final: Dict[str, object] = {}
    for key in own_keys:
        mine = [r for r in prec_rows if r["family"] == key]
        final[key] = anchored(key, concordance=cells, precision=T.precision_block(mine))

    # 4. copies of A-R, inherited forms, family B
    inherited: Dict[str, str] = {}
    for key, decl in DERIVED_FROM_A_R.items():
        if key in planned and A_R in final:
            final[key] = derived_copy(final[A_R], key, decl)
    for key in planned:
        if key in final or key in fb:
            continue
        base, _form = _split(key)
        if base not in final:
            blocking.append(f"protocol {key}: no reference block and no generated "
                            "base protocol")
            continue
        src = final[base].to_dict()
        payload = apply_estimator_decisions(draft(key), dec)
        payload["reference"] = src["reference"]
        payload["anchors"] = src["anchors"]
        payload["necessity_set"] = src["necessity_set"]
        payload["concordance_route"] = (src["concordance_route"]
                                        if carries_concordance_route(key, payload)
                                        else [])
        payload["precision"] = None
        final[key] = E.ProtocolV3.from_dict(_named(payload, key))
        inherited[key] = base
    final.update(fb)
    final["B"] = _family_b_carrier(fb_row)
    for key in planned:
        if key not in final:
            blocking.append(f"protocol {key} not built")
        elif final[key].reference.get("kind") == E.REFERENCE_PENDING:
            blocking.append(f"protocol {key}: anchors pending")
    notes = _report_cut_notes(planned, final)

    # 5. evidence
    rows_final = judged_rows(data, final, dev_designs, dec=dec)
    mech_entries, mech_evidence = mechanism_on(rows_final, dose_only_plan())
    se_cal = se_calibration(data, final, dec)
    occupancy = occupancy_evidence(data)
    replication = replication_power(series_by_key, entries_by_key)
    suggestions = dict(se_cal["suggestions"])
    suggestions["iim_n_min"] = {
        "value": occupancy["suggestion"] if isinstance(occupancy["suggestion"], int)
        else None, "rule": occupancy["rule"], "all_pass": occupancy["all_pass"]}
    rule_cells = concordance_cells(conc_ev, _with(dec, pdi_concordance="rule"))
    suggestions["pdi_concordance"] = {
        "value": None, "rule": "0 concordant ABSENTs among >= 300 content-on runs; "
                               "0 concordant PRESENTs among >= 300 no-content runs",
        "rule_cells": rule_cells, "decided_cells": cells,
        "cells_with_enough_content_on_runs": sum(
            e["content_on_runs"] >= CONCORDANCE_MIN_RUNS for e in conc_ev),
        "cells_with_enough_no_content_runs": sum(
            e["no_content_runs"] >= CONCORDANCE_MIN_RUNS for e in conc_ev),
        "consistent": _canon(rule_cells) == _canon(cells)}
    suggestions["mechanism_on"] = {"value": "rule", "rule": "design 4.7 (iii)"}
    rep_sug = replication_suggestion(replication)
    if rep_sug is not None:
        suggestions["replication_extended"] = rep_sug
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "flag": DEVELOPMENT_FLAG,
        "anchors": {k: {**anchor_info[k], "entries": entries_by_key[k],
                        "reference": refs[k]} for k in sorted(anchor_info)},
        "inherited_anchors": inherited,
        "se_calibration": se_cal,
        "occupancy_gate": occupancy,
        "concordance_battery": conc_ev,
        "iim_agents_expectation": iim_agents_expectation(rows_final),
        "replication_power": replication,
        "dependency_rates": dependency_evidence(rows_final, final),
        "mechanism_on": mech_evidence,
        "adversary_potency": adversary_potency(rows_final),
        "oracle_checks": oracle_checks(data),
        "null_calibration_ram": null_calibration_ram(data),
        "iim_grain_cap": grain_cap_evidence(data),
        "constants": _constants(data),
        "suggestions": suggestions,
    }

    # 6. decisions against the code, the rules and the development runs; the
    # freeze
    cap = dec["iim_max_macro_nodes"]
    other_caps = [c for c in evidence["iim_grain_cap"]["caps_in_records"] if c != cap]
    if other_caps:
        blocking.append(f"development records ran with the IIM macro-node cap "
                        f"{other_caps}, not the declared {cap}")
    dec_record, dec_blocking = decisions_record(dec, suggestions)
    blocking.extend(dec_blocking)
    if dec.status != STATUS_FINAL:
        blocking.append("the decisions file is a draft")
    downstream = _downstream(dec, mech_entries)
    mech_payload = (mech_entries if dec["mechanism_on"] == "rule"
                    else list((_hypotheses_spec().get("mechanism_on") or {})
                              .get("entries") or []))
    files = OrderedDict()
    for key in sorted(final):
        files[HE.protocol_file_name(key)] = final[key].to_dict()
    files[FORWARD_ANCHORS] = {
        "schema": "mpc-bench-forward-anchors/1", "regime": regime,
        "anchors": sorted(forward_anchor_rows,
                          key=lambda a: (a["arm"], a["view"], a["principle"]))}
    files[TESTABILITY] = {
        "schema": "mpc-bench-testability-table/1", "flag": DEVELOPMENT_FLAG,
        "threshold": T.DECISIVE_THRESHOLD, "min_runs": T.MIN_RUNS,
        "rows": sorted(table, key=lambda t: (t["family"], t["principle"],
                                             t["witness"], t["kind"]))}
    files[MECHANISM_ON] = mech_payload
    files[DEPENDENCIES] = {"schema": "mpc-bench-declared-dependencies/1",
                           "status": ("pending" if "declared_dependencies" in
                                      dec.pending else "final"),
                           "pairs": dec["declared_dependencies"]}
    files[DECISIONS_RECORD] = dec_record
    files[EVIDENCE] = evidence
    files = OrderedDict((k, _clean(v)) for k, v in files.items())
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "builder_version": BUILDER_VERSION,
        "decisions_sha256": dec.sha256,
        "decisions_status": dec.status,
        "pending_decisions": list(dec.pending),
        "inputs": [{"path": s.path, "sha256": s.sha256, "n_records": s.n_records}
                   for s in sorted(data.sources, key=lambda s: s.path)],
        "releases": [{"release_id": r["release_id"], "predictions": r["predictions"]}
                     for r in data.releases],
        "planned_protocol_keys": planned,
        "protocols": {k: {"file": HE.protocol_file_name(k), "hash": final[k].hash}
                      for k in sorted(final)},
        "files": {name: sha256_text(_dump(payload)) for name, payload in files.items()},
        "run_settings": {"iim_max_macro_nodes": dec["iim_max_macro_nodes"],
                         "pdi_kmeans_seed": dec["pdi_kmeans_seed"]},
        "freeze_ready": not blocking,
        "blocking": sorted(set(blocking)),
        "downstream": downstream,
        "notes": notes,
    }
    return {"protocols": final, "files": files, "manifest": _clean(manifest),
            "evidence": evidence, "decisions": dec}


def _with(dec: Decisions, **values) -> Decisions:
    return Decisions({**dict(dec.values), **values}, dec.pending, dec.deviation,
                     dec.notes, dec.status, dec.sha256)


def _forward_keys() -> List[str]:
    from impact_pipeline.bench.designs_v2 import forward as FW

    return sorted(FW.protocol_options())


def _forward_arm_view_of_key(key: str) -> Tuple[str, str]:
    from impact_pipeline.bench.designs_v2 import forward as FW

    base, _form = _split(key)
    for arm, prefix in FW.PROTOCOL_PREFIX_OF_ARM.items():
        view = base[len(prefix) + 1:]
        if base.startswith(prefix + "-") and view in FW.VIEWS_OF_ARM[arm]:
            return arm, view
    raise BuildError(f"{key} is not a forward protocol")


def _forward_principles(key: str) -> List[str]:
    return _forward_scored(key)


def _forward_scored(key: str) -> List[str]:
    from impact_pipeline.bench.designs_v2 import forward as FW

    arm, view = _forward_arm_view_of_key(key)
    out = set()
    for s in FW._scorings(arm, view):
        if FW.protocol_key_of(arm, view, s["estimator_form"]) == key:
            out.update(s["principles"])
    return sorted(out)


def _report_cut_notes(planned: Sequence[str], final: Mapping,
                      scored: Optional[Mapping] = None) -> List[str]:
    """Protocols of the runner plan that score IIM and keep a reported cut
    although no form scoring of the plan reads it beside them (reported for
    review; the family-B protocols record both cuts by design)."""
    out = []
    plan = set(planned)
    scored = confirmatory_plan_principles() if scored is None else scored
    for key in sorted(plan):
        p = final.get(key)
        if p is None or "+" in key or "IIM" not in scored.get(key, ()):
            continue
        cuts = list((p.estimator_options("IIM") or {}).get("report_cut_modes") or [])
        if cuts and f"{key}+iim_bidirectional" not in plan:
            out.append(f"{key}: reported IIM cut {cuts} kept; no "
                       f"{key}+iim_bidirectional scoring in the plan")
    return out


def _split(key: str) -> Tuple[str, str]:
    base, sep, form = str(key).partition("+")
    return base, (form if sep else D.PRIMARY_FORM)


def _forward_admission_designs():
    from impact_pipeline.bench.designs_v2 import forward as FW
    return tuple(FW.ADMISSION_RECORD_DESIGNS)


def _family_b_carrier(row: Optional[dict]):
    """The family-B protocol ``B``: the primary family-B protocol (directional
    cut, primary anchor, circular-shift null, with the SE contract) under the
    key ``B`` with the family-B testability rows; no record names it."""
    from impact_pipeline.bench.designs_v2 import family_b as FB

    primary = FB.family_b_protocol(_iim().PRIMARY_CUT_MODE)
    payload = primary.to_dict()
    payload["precision"] = T.precision_block([row] if row else [])
    return E.ProtocolV3.from_dict(_named(payload, "B"))


def _part_systems(part: Mapping, vocab: Mapping) -> Optional[List[str]]:
    """The systems a part's data selects (``data.where.system``, vocabulary
    references resolved); None when the part does not restrict them."""
    sel = ((part.get("data") or {}).get("where") or {}).get("system")
    if sel is None:
        return None
    out: List[str] = []
    for v in (sel if isinstance(sel, list) else [sel]):
        if isinstance(v, str) and v.startswith("@"):
            v = vocab.get(v[1:], v)
        out.extend(str(x) for x in (v if isinstance(v, list) else [v]))
    return out


def _necessity_set(proto) -> Optional[set]:
    """``N_anch`` of a protocol as the hypothesis engine reads it (the
    anchors block's, else the protocol's necessity set)."""
    if proto is None:
        return None
    anchors = proto.anchors if isinstance(proto.anchors, Mapping) else {}
    if anchors.get("necessity_set") is not None:
        return set(anchors["necessity_set"])
    return set(proto.necessity_set)


def _target_kept(proto, principle: Optional[str],
                 target_filter: Optional[str]) -> bool:
    """Whether a part's ``target_filter`` keeps the cells of a witness whose
    target is ``principle`` under ``proto`` (no filter or no protocol: kept):
    ``in_n_anch`` keeps a principle of ``N_anch``, ``valid_anchor`` a
    principle with a valid anchor (specific or not); a witness without a
    target is dropped, as the engine drops its rows."""
    if not target_filter or proto is None:
        return True
    if principle is None:
        return False
    if target_filter == "in_n_anch":
        return principle in (_necessity_set(proto) or ())
    return proto.anchor_status(principle) in (T.ANCHOR_VALID_SPECIFIC,
                                              T.ANCHOR_VALID_NONSPECIFIC)


def _gates_without_rows(table: Sequence[dict], rows: Sequence[dict],
                        protos: Optional[Mapping] = None) -> List[str]:
    """The gates the hypotheses file declares without a precision row, the
    templates expanded over the gate protocols and the witness targets of
    the systems the part selects: a part gate needs its row; a per-cell gate
    a row for every cell with development runs (a cell without any, such as
    a not-applicable principle on C1, is not a cell of the part). A part's
    ``target_filter`` drops the cells whose witness target it drops under
    the generated protocol of the cell (``protos``)."""
    protos = protos or {}
    have = {(r["family"], r["principle"], r["witness"], r["kind"]) for r in rows}
    seen = {(t["family"], t["principle"], t["witness"], t["kind"]) for t in table
            if t.get("n")}
    spec = _hypotheses_spec()
    vocab = spec.get("vocabulary") or {}
    targets = _witness_targets()
    missing = []
    for h in spec.get("hypotheses") or ():
        for part in h.get("parts") or ():
            g = part.get("gate")
            if not g or part.get("status") == "removed":
                continue
            fam = str(g.get("family"))
            if fam.startswith("@"):
                fam = str(vocab.get(fam[1:], fam))
            fams = list(GATE_PROTOCOLS) if fam == "{protocol_id}" else [fam]
            if g.get("witness") == "{system}":
                systems = _part_systems(part, vocab)
                pairs = [(w, t) for w, t in sorted(targets.items())
                         if systems is None or w in systems]
            else:
                pairs = [(g.get("witness"), g.get("principle"))]
            target_filter = (part.get("data") or {}).get("target_filter")
            wanted = []
            for f in fams:
                for w, t in pairs:
                    # the filter reads the witness's target (engine: the
                    # target of the row's system), not the gate's principle
                    if not _target_kept(protos.get(f), targets.get(w), target_filter):
                        continue
                    p = t if g.get("principle") == "{target}" else g.get("principle")
                    wanted.append((f, p, w, g.get("kind")))
            if not wanted:
                continue
            if g.get("per_cell"):
                for w in wanted:
                    if w in seen and w not in have:
                        missing.append(f"{part['id']}: {w}")
            elif not any(w in have for w in wanted):
                missing.append(f"{part['id']}: {wanted[0]}")
    return missing


def _constants(data: DevData) -> Optional[dict]:
    path = data.root / "constants" / "constants" / DC.CONSTANTS_JSON
    if path.exists():
        return {"file": str(path.relative_to(data.root)), "sha256": _file_sha(path)}
    return None


def decisions_record(dec: Decisions, suggestions: Mapping) -> Tuple[dict, List[str]]:
    """The decisions as applied and what blocks the freeze: pending
    decisions, a decision that differs from the code that runs it, a decision
    that differs from the rule's suggestion without a deviation note."""
    out, blocking = {}, []
    for spec in DECISIONS:
        val = dec.values[spec.name]
        entry = {"cd": spec.cd, "meaning": spec.meaning, "value": val,
                 "pending": spec.name in dec.pending,
                 "deviation": dec.deviation.get(spec.name),
                 "note": dec.notes.get(spec.name)}
        entry["tier"] = spec.tier
        if spec.name in dec.pending and spec.tier == "A":
            blocking.append(f"decision {spec.name} ({spec.cd}) pending")
        if spec.code is not None:
            where, fn = spec.code
            code_val = fn()
            want, have = _code_view(val, code_val)
            ok = _same(want, have)
            entry["code"] = {"where": where, "value": code_val, "consistent": ok}
            if not ok:
                blocking.append(f"decision {spec.name}: {val!r} differs from the code "
                                f"({where}: {code_val!r}); change the code there")
        sug = suggestions.get(spec.name)
        if sug is not None:
            entry["suggestion"] = sug
            sv = sug.get("value")
            differs = ((sv is not None and not _same(sv, val))
                       or sug.get("consistent") is False)
            entry["matches_rule"] = not differs
            if (differs and spec.name not in dec.pending
                    and not dec.deviation.get(spec.name)):
                blocking.append(f"decision {spec.name}: {val!r} differs from the "
                                "rule's outcome without a deviation note")
        out[spec.name] = entry
    return ({"schema": "mpc-bench-calibration-decisions-applied/1",
             "decisions_status": dec.status, "decisions_sha256": dec.sha256,
             "decisions": out}, blocking)


def _code_view(val, code_val) -> Tuple[object, object]:
    """A decided value and the code's value as they are compared: a mapping
    on the keys the code carries (the rest of the decision is not code); a
    map of switches (the anchor designs' extensions, the forward arms' as a
    nested map) on the switches of either, a switch the other side lacks
    read as off, so a decided extension the code cannot carry is no match."""
    if not (isinstance(code_val, Mapping) and isinstance(val, Mapping)):
        return val, code_val
    want, have = _switches(val), _switches(code_val)
    if want is not None and have is not None:
        keys = sorted(set(want) | set(have))
        return ({k: want.get(k, False) for k in keys},
                {k: have.get(k, False) for k in keys})
    return {k: val.get(k) for k in code_val}, code_val


def _switches(m: Mapping) -> Optional[Dict[str, bool]]:
    """A map of switches (true or false, or a map of them) flattened to
    ``{"key" or "key.sub": bool}``; None if anything else is in it."""
    out = {}
    for k, x in m.items():
        if isinstance(x, bool):
            out[str(k)] = x
        elif isinstance(x, Mapping) and all(isinstance(y, bool) for y in x.values()):
            out.update({f"{k}.{a}": y for a, y in x.items()})
        else:
            return None
    return out


def _same(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b))
        except (TypeError, ValueError):
            return False
    return _canon(a) == _canon(b)


def _downstream(dec: Decisions, mech_entries: Sequence[dict]) -> List[str]:
    """What the freeze needs outside the generated folder."""
    spec = _hypotheses_spec()
    out = []
    pairs = _pairs((spec.get("declared_dependencies") or {}).get("pairs") or [])
    if pairs != dec["declared_dependencies"] or (spec.get("declared_dependencies")
                                                 or {}).get("status") != "final":
        out.append("protocols/v2/hypotheses_v2.json declared_dependencies: copy "
                   f"{DEPENDENCIES} and set its status to final")
    mech = spec.get("mechanism_on") or {}
    if dec["mechanism_on"] == "rule" and (mech.get("status") != "final"
                                          or mech.get("entries") != list(mech_entries)):
        out.append(f"protocols/v2/hypotheses_v2.json mechanism_on: copy {MECHANISM_ON} "
                   "and set its status to final")
    if dec["null_calibration_ram"] == "drop":
        out.append("designs_v2/null_calibration: leave RAM out of the null-calibration "
                   "scorings")
    if dec["null_calibration_ram"] == "require_defined":
        out.append("protocols/v2/hypotheses_v2.json HCv2-1 and HCv2-3: count only "
                   "defined RAM rows")
    if spec.get("status") != "final":
        out.append("protocols/v2/hypotheses_v2.json status: final at the freeze")
    out.append("registry v3: built from the confirmatory forward arms with "
               f"{FORWARD_ANCHORS} (scripts/v2/build_registry_v3.py)")
    return out


# --------------------------------------------------------------------------
# writing and checking
# --------------------------------------------------------------------------
def _dump(payload) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"


def write(result: Mapping, out_dir=GENERATED_DIR) -> Path:
    """Write every file and the manifest. Files of an earlier build in the
    directory (those its manifest lists) that this build does not produce
    are removed; a directory with protocol files that no build wrote is
    refused, and no other file is touched."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    previous: set = set()
    if (out / MANIFEST).is_file():
        try:
            prev = json.loads((out / MANIFEST).read_text(encoding="utf-8"))
        except ValueError:
            raise BuildError(f"{out / MANIFEST}: not a build manifest") from None
        if not isinstance(prev, Mapping) or prev.get("schema") != MANIFEST_SCHEMA:
            raise BuildError(f"{out / MANIFEST}: not a build manifest")
        previous = {str(n) for n in (prev.get("files") or {})
                    if Path(str(n)).name == str(n) and not str(n).startswith(".")}
    foreign = sorted(p.name for p in out.glob(HE.PROTOCOL_FILE_GLOB)
                     if p.name not in previous)
    if foreign:
        raise BuildError(f"{out} holds protocol files that no build wrote "
                         f"({foreign[:3]}); write into an empty directory or a "
                         "previous build")
    for name in sorted(previous - set(result["files"])):
        if (out / name).is_file():
            (out / name).unlink()
    for name, payload in result["files"].items():
        (out / name).write_text(_dump(payload), encoding="utf-8")
    (out / MANIFEST).write_text(_dump(result["manifest"]), encoding="utf-8")
    return out


def check(out_dir=GENERATED_DIR) -> dict:
    """Re-read a build: every file's hash equals the manifest's, every
    protocol loads under its key with the manifest's hash, and no protocol
    file is outside the manifest."""
    out = Path(out_dir)
    man = json.loads((out / MANIFEST).read_text(encoding="utf-8"))
    problems = []
    for name, sha in sorted(man["files"].items()):
        p = out / name
        if not p.exists():
            problems.append(f"{name}: missing")
            continue
        if sha256_text(p.read_text(encoding="utf-8")) != sha:
            problems.append(f"{name}: changed since the build")
    protos = {}
    for p in sorted(out.glob(HE.PROTOCOL_FILE_GLOB)):
        try:
            protos.update(HE.load_protocol_files([p]))
        except (ValueError, TypeError) as exc:
            problems.append(f"{p.name}: not loadable ({exc})")
    for key, info in man["protocols"].items():
        if key not in protos:
            problems.append(f"protocol {key}: not loadable")
        elif protos[key].hash != info["hash"]:
            problems.append(f"protocol {key}: hash differs from the manifest")
    extra = sorted(set(protos) - set(man["protocols"]))
    problems.extend(f"protocol {k}: not in the manifest" for k in extra)
    return {"ok": not problems, "problems": problems,
            "freeze_ready": man.get("freeze_ready"), "n_protocols": len(protos)}


def template(dev_root=None) -> dict:
    """A decisions file with every decision pending (value null), the code's
    provisional value and, where development outputs exist, the rule's
    suggestion."""
    sugg = {}
    if dev_root is not None and Path(dev_root).exists():
        try:
            sugg = build(None, dev_root)["evidence"]["suggestions"]
        except BuildError:
            raise
    out = {"schema": DECISIONS_SCHEMA, "status": STATUS_DRAFT, "decisions": {}}
    for spec in DECISIONS:
        out["decisions"][spec.name] = {
            "value": None, "cd": spec.cd, "meaning": spec.meaning,
            "provisional": spec.provisional(),
            "suggestion": (sugg.get(spec.name) or {}).get("value"),
            "deviation": None, "note": None}
    return _clean(out)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="build_protocols_v2.py",
                                 description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    tp = sub.add_parser("template", help="a decisions file with every decision pending")
    tp.add_argument("--out", required=True)
    tp.add_argument("--dev-root", default=None)
    bd = sub.add_parser("build", help="build the protocols")
    bd.add_argument("--decisions", default=None)
    bd.add_argument("--dev-root", default=str(DC.DEFAULT_ROOT))
    bd.add_argument("--out", default=str(GENERATED_DIR))
    bd.add_argument("--require-freeze-ready", action="store_true")
    ck = sub.add_parser("check", help="verify a build against its manifest")
    ck.add_argument("--dir", default=str(GENERATED_DIR))
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        if args.command == "template":
            Path(args.out).write_text(_dump(template(args.dev_root)), encoding="utf-8")
            print(f"decisions template written to {args.out}")
            return 0
        if args.command == "check":
            res = check(args.dir)
            print(json.dumps(res, indent=1))
            return 0 if res["ok"] else 1
        res = build(args.decisions, args.dev_root)
        man = res["manifest"]
        if args.require_freeze_ready and not man["freeze_ready"]:
            print("not ready for the freeze; nothing written:", file=sys.stderr)
            for b in man["blocking"]:
                print(f"  - {b}", file=sys.stderr)
            return 1
        out = write(res, args.out)
        n_tables = len(res["files"]) - len(res["protocols"])
        print(f"{len(res['protocols'])} protocols and {n_tables} tables written to "
              f"{out}; freeze_ready={man['freeze_ready']}")
        for b in man["blocking"]:
            print(f"  blocking: {b}")
        return 0
    except BuildError as exc:
        print(f"build_protocols_v2: refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
