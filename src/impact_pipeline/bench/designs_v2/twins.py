"""
Twin sessions of MPC-Bench v2 (design 2.0.5, 3.2; SE calibration against
white-box twins, integrity audit IA-10).

A twin is the same network in a new session: replicate ``r >= 1`` keeps the
structural stream ``SeedSequence(seed)`` and draws the task schedule,
process noise and rest run from ``SeedSequence([seed, r])``; ``r = 0`` is
the run itself, bit for bit.

=============  ============================================  ============  ===========
design         classes                                       confirmatory  development
=============  ============================================  ============  ===========
A_twins        the catalogue witnesses with family-A twins:  networks      networks
               PC_nominal, PC_half, W_PDI_single_attractor,  20000-20009,  820-824,
               W_NAS_no_workspace, W_IIM_feedforward         r = 1..6      r = 0..6
               (R and H; IIM also bidirectional)             (+ 10 r = 0
                                                             identity
                                                             checks)
C1_twins       PC_nominal, W_NAS_no_workspace,               r = 1..6      r = 0..6
               W_IIM_feedforward (R and H)
RAM160_twins   RAM-only eta in {0, 0.1, 0.3} (RAM-PE)        r = 1..7      r = 0..7
=============  ============================================  ============  ===========

On the confirmatory split ``r = 0`` of every twin set is the witness or
RAM-only run of the same seed, so the twins add ``r >= 1`` only; ten
simulation-only tasks re-run PC_nominal at ``r = 0`` on each family-A
network and are compared with the witness run (``tags.identity_check_of``:
the same ``ts`` hash expected, an intended duplicate). On the development
networks (no witness runs) the sets include ``r = 0``. Task counts:
5 x 10 x 6 + 10 = 310, 3 x 10 x 6 = 180, 3 x 10 x 7 = 210 (confirmatory);
175, 105 and 120 (development).
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Sequence

from impact_pipeline.bench.designs_v2 import (
    CONFIRMATORY,
    DEVELOPMENT,
    PRINCIPLES,
    Design,
    DesignError,
    TaskSpec,
    make_scorings,
)
from impact_pipeline.bench.designs_v2 import family_a as FA
from impact_pipeline.bench.designs_v2 import family_c1 as FC
from impact_pipeline.bench.designs_v2 import ram_only as RO

MODULE = "twins"
NETWORKS = {CONFIRMATORY: range(20000, 20010), DEVELOPMENT: range(820, 825)}
REPLICATES = {
    "A": {CONFIRMATORY: range(1, 7), DEVELOPMENT: range(0, 7)},
    "C1": {CONFIRMATORY: range(1, 7), DEVELOPMENT: range(0, 7)},
    "RAM160": {CONFIRMATORY: range(1, 8), DEVELOPMENT: range(0, 8)},
}
RAM_TWIN_CLASSES = ("eta_0", "eta_0.1", "eta_0.3")
TWIN_FORMS = ("iim_bidirectional",)
IDENTITY_CHECK_SYSTEM = "PC_nominal"


def twin_classes(catalogue_family: str) -> tuple:
    """Catalogue witnesses whose twin set belongs to the SE calibration of a
    family (``twin_families`` of the catalogue)."""
    from impact_pipeline.bench import adversarial_v2 as A2

    return tuple(sid for sid in FA.catalogue_systems("witness", catalogue_family)
                 if catalogue_family in (A2.get_entry(sid).get("twin_families") or ()))


def _networks(split, seeds):
    return FA._seeds(NETWORKS, split, seeds)


def _replicates(key, split, replicates):
    base = tuple(REPLICATES[key][split])
    if replicates is None:
        return base
    out = tuple(int(r) for r in replicates)
    bad = sorted(set(out) - set(base))
    if bad:
        raise DesignError(f"replicates {bad} are not part of the {key} twin set "
                          f"({split}: {base})")
    return out


def a_twins(split: str, *, seeds: Optional[Sequence[int]] = None,
            systems: Optional[Sequence[str]] = None,
            replicates: Optional[Sequence[int]] = None,
            identity_checks: Optional[bool] = None) -> List[TaskSpec]:
    """``A_twins`` (plus the ``r = 0`` identity checks on the confirmatory
    networks)."""
    ids = FA._select(twin_classes(FA.CATALOGUE_FAMILY), systems, "twin classes")
    sc = make_scorings(FA.PROTOCOLS, PRINCIPLES, TWIN_FORMS)
    out = []
    nets = _networks(split, seeds)
    for sid in ids:
        for net in nets:
            for r in _replicates("A", split, replicates):
                out.append(FA.catalogue_task(
                    "A_twins", FA.FAMILY, FA.CATALOGUE_FAMILY, sid, net, sc,
                    replicate=r, module=MODULE,
                    tags={"twin_network": int(net), "twin_class": sid}))
    if identity_checks is None:
        identity_checks = split == CONFIRMATORY
    if identity_checks:
        for net in nets:
            out.append(FA.catalogue_task(
                "A_twins", FA.FAMILY, FA.CATALOGUE_FAMILY, IDENTITY_CHECK_SYSTEM, net,
                (), replicate=0, module=MODULE, suffix="r0check",
                tags={"twin_network": int(net), "twin_class": IDENTITY_CHECK_SYSTEM,
                      "identity_check_of":
                          f"A_witnesses-{IDENTITY_CHECK_SYSTEM}-s{int(net):05d}"}))
    return out


def c1_twins(split: str, *, seeds: Optional[Sequence[int]] = None,
             systems: Optional[Sequence[str]] = None,
             replicates: Optional[Sequence[int]] = None) -> List[TaskSpec]:
    """``C1_twins`` (IIM only on the confirmatory split: held out on C1)."""
    ids = FA._select(twin_classes(FC.CATALOGUE_FAMILY), systems, "twin classes")
    sc = make_scorings(FC.PROTOCOLS, FC.principles(split), TWIN_FORMS)
    out = []
    for sid in ids:
        for net in _networks(split, seeds):
            for r in _replicates("C1", split, replicates):
                out.append(FA.catalogue_task(
                    "C1_twins", FC.FAMILY, FC.CATALOGUE_FAMILY, sid, net, sc,
                    replicate=r, module=MODULE,
                    tags={"twin_network": int(net), "twin_class": sid}))
    return out


def ram_twins(split: str, *, seeds: Optional[Sequence[int]] = None,
              systems: Optional[Sequence[str]] = None,
              replicates: Optional[Sequence[int]] = None) -> List[TaskSpec]:
    """``RAM160_twins``: eta in {0, 0.1, 0.3} at 160 trials."""
    out = []
    for cls in FA._select(RAM_TWIN_CLASSES, systems, "RAM twin classes"):
        for net in _networks(split, seeds):
            for r in _replicates("RAM160", split, replicates):
                t = RO.class_task("RAM160_twins", cls, net, replicate=r, module=MODULE)
                tags = dict(t.tags)
                tags.update({"twin_network": int(net), "twin_class": cls})
                out.append(dataclasses.replace(t, tags=tags))
    return out


_TWIN_PERFORMANCE = (
    "kappa = pooled within-network SD of c / RMS(se_c) with its 90 % "
    "chi-square interval, and the q_A tail rates against the network's twin "
    "mean (HCv2-4; development twins: the SE-method choices of calibration)")

ADEMP = {
    "A_twins": {
        "aims": "calibration of every v2 SE method against white-box twins (the "
                "same network in new sessions), and the r = 0 identity check",
        "data": "family-A networks with twin sessions r >= 1 (task schedule, "
                "process noise and rest run from SeedSequence([seed, r])) of "
                "PC_nominal, PC_half, W_NAS_no_workspace, W_IIM_feedforward and "
                "W_PDI_single_attractor; R and H (one cluster); confirmatory "
                "r = 0 is the witness run, re-run once per network as a "
                "simulation-only identity check; the development networks "
                "include r = 0",
        "estimands": "per estimator, SE method and class: the within-network "
                     "SD of c across sessions and the reported se_c",
        "methods": FA.JOINT_BENCH_METHODS + "; IIM also with bidirectional cuts",
        "performance": (_TWIN_PERFORMANCE + "; identity: equal ts hash with the "
                        "witness run, equal structural and different schedule "
                        "hashes across sessions (integrity audit)"),
    },
    "C1_twins": {
        "aims": "calibration of the NAS and IIM SE methods on family C1",
        "data": "family-C1 networks with twin sessions of PC_nominal, "
                "W_NAS_no_workspace and W_IIM_feedforward; R and H (IIM on the "
                "confirmatory split only); confirmatory r = 0 is the witness "
                "run",
        "estimands": "per estimator, SE method and class: the within-network "
                     "SD of c and the reported se_c",
        "methods": FC.C1_METHODS,
        "performance": _TWIN_PERFORMANCE,
    },
    "RAM160_twins": {
        "aims": "calibration of the RAM-PE SE method (shift-null SD, fallback "
                "trial jackknife)",
        "data": "RAM-only agents (160 trials) at eta in {0, 0.1, 0.3} with twin "
                "sessions; confirmatory r = 0 is the RAM-only arm's run",
        "estimands": "the within-network SD of RAM-PE c and the reported se_c "
                     "per eta",
        "methods": "RAM-PE (ram-v3-2026.10) judged by tost-v2 on the A-RAM160 "
                   "anchor; component statuses only",
        "performance": _TWIN_PERFORMANCE,
    },
}

DESIGNS = (
    Design("A_twins", FA.FAMILY,
           "family-A twin sessions (SE calibration) and the r = 0 identity checks",
           a_twins, {CONFIRMATORY: 5 * 10 * 6 + 10, DEVELOPMENT: 5 * 5 * 7},
           ademp=ADEMP["A_twins"]),
    Design("C1_twins", FC.FAMILY, "family-C1 twin sessions (SE calibration)",
           c1_twins, {CONFIRMATORY: 3 * 10 * 6, DEVELOPMENT: 3 * 5 * 7},
           ademp=ADEMP["C1_twins"]),
    Design("RAM160_twins", RO.FAMILY,
           "RAM-only twin sessions at 160 trials (RAM-PE SE calibration)",
           ram_twins, {CONFIRMATORY: 3 * 10 * 7, DEVELOPMENT: 3 * 5 * 8},
           ademp=ADEMP["RAM160_twins"]),
)

__all__ = [
    "ADEMP",
    "DESIGNS",
    "IDENTITY_CHECK_SYSTEM",
    "NETWORKS",
    "RAM_TWIN_CLASSES",
    "REPLICATES",
    "TWIN_FORMS",
    "a_twins",
    "c1_twins",
    "ram_twins",
    "twin_classes",
]
