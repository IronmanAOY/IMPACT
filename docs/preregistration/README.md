# Preregistration of the MPC-Bench computational hypotheses (paper 1)

This folder holds the preregistration of the computational hypotheses of
paper 1 (the MPC-Bench validation of the IMPaCT measurement framework):

- [`MPC_BENCH_PREREGISTRATION.md`](MPC_BENCH_PREREGISTRATION.md): the
  hypotheses HC1-HC10, their decision rules and thresholds, the frozen
  protocols and their hashes, the calibration decisions made on development
  data, and the confirmatory run plan. Its last section,
  ["Errata"](MPC_BENCH_PREREGISTRATION.md#errata-documentation-only-no-change-to-hypotheses-or-decision-rules),
  was added after the freeze (2026-09-29, documentation only): it corrects
  two descriptive numbers and changes no hypothesis or decision rule; the
  text above it is the frozen text.

- [`MPC_BENCH_PREREGISTRATION_V2.md`](MPC_BENCH_PREREGISTRATION_V2.md): the
  MPC-Bench v2 revision round, written after the v1 results: the hypotheses
  HCv2-0 to HCv2-24 (Tier A) with every rule, parameter and data selection of
  `protocols/v2/hypotheses_v2.json`, the estimators and the status rule
  `tost-v2`, the systems, designs and seed map, the calibration on development
  data, the held-out elements and their release, integrity and deviations, the
  confirmatory run plan and cost, the evaluation rules and the frozen files with
  their SHA-256. Its companions in [`v2/`](v2/):
  [`testability_table.md`](v2/testability_table.md),
  [`development_expectations.md`](v2/development_expectations.md) and
  [`operating_characteristics.md`](v2/operating_characteristics.md).
  `tests/v2/test_prereg_v2_consistency.py` keeps it consistent with the
  hypotheses file, the generated protocols, the seed map and the release log.

Both documents are frozen: they are not edited after their tags.
Clarifications and corrections are recorded in this README, dated, or (for
v1) in the errata section.

## How a freeze is registered

A freeze is an annotated git tag on the commit that contains the
preregistration, its frozen protocols and the code that produces and
evaluates the confirmatory results. The tag message quotes the SHA-256 of the
frozen files. The confirmatory steps refuse any checkout whose `src/` and
`scripts/` trees differ from the tag, so the confirmatory runs, the
integrity checks and the evaluators run from a checkout of the tag;
documentation-only commits after the tag do not change those trees. Pushing
the tag to the public repository makes the commit and the tag message
publicly reachable.

A registration on a public registry (for example OSF) adds an independent,
time-stamped copy. It attaches the preregistration as it is at the tagged
commit (for v1 together with the later errata section, either as the
current file or as a separate document), and it records the tag name, the
full commit hash (`git rev-list -n 1 <tag>`), the note on commit
identifiers below (v1) and the protocol hashes the preregistration lists.
They can be recomputed, for example
`python -c "from impact_pipeline.evidence import Protocol;
print(Protocol.from_json('protocols/mpc_bench_v1.json').hash)"`.

## Status of v1

| Item | Value |
|---|---|
| Freeze tag | `mpcbench-freeze-v1` (annotated; on the public repository) |
| Frozen commit | `f2cf2492a6cb4a1b0b1f5b8d680802fded31c1d1` (originally `b908ee3`, see below) |
| Frozen protocols | `protocols/mpc_bench_v1.json` and `protocols/mpc_bench_v1_anchored.json`, with the SHA-256 the tag message quotes |
| Errata | section "Errata" of the preregistration, added 2026-09-29 after the freeze (documentation only) |
| Confirmatory runs | 2026-09-28 to 2026-09-30, from a checkout of the tag; outcomes in the main [README](../../README.md#status-and-results) |
| Public registration | *to be added by the author* |

**Commit identifiers.** The commit messages and author metadata of the
history up to and including the freeze commit were edited on 2026-09-29; no
file was changed. The freeze commit is now `f2cf249`
(`f2cf2492a6cb4a1b0b1f5b8d680802fded31c1d1`); before the edit it was
`b908ee3` (`b908ee3230d8b93a1983960a9621e49bdd5752b2`). Both commits have the
same tree, `72fcc7031de5cc6f660af7f54dc5e0a8b3bd6a6d`, so their content is
identical. The tag points at `f2cf249` and its message states the same; the
confirmatory runs record `b908ee3` as their code commit, because they started
before the edit. `b908ee3` exists only in the author's local history; the tree
can be checked with `git rev-parse f2cf249^{tree}`.

## Status of v2

| Item | Value |
|---|---|
| Freeze tag | `mpcbench-freeze-v2` (annotated; pushed to the public repository on 2026-10-09) |
| Frozen commit | `acb5428bf05c8d494ae938857b471256c542e7dd`, the commit that contains `MPC_BENCH_PREREGISTRATION_V2.md`, its companions, `protocols/v2/` and the code (a document cannot quote the commit that contains it, so the preregistration names it by the tag) |
| Held-out predictions | `protocols/v2/held_out_predictions_v2.json`, committed in `21597fe`; release `5af6e074475de23a` logged 2026-10-08 at head `a16f84c` |
| Confirmatory run | 2026-10-09, from a checkout of the tag; outcomes in the main [README](../../README.md#status-and-results) |
| Public registration | *to be added by the author* |

The v1 preregistration, the v1 tag and the v1 outcomes are not changed by v2.

**Note of 2026-10-10 on the author-level decisions.** Section 10.4 of the v2
preregistration says that the message of the annotated tag
`mpcbench-freeze-v2` names and confirms the two author-level decisions of the
round: the 46-seed null witnesses of HCv2-1 and the paper-2 regime (CD-12).
The tag message does not name them; it lists the frozen protocols, the
hypotheses file, the preregistration, the seed map and the held-out
predictions with their SHA-256 (`git cat-file -p mpcbench-freeze-v2`). The
author confirmed both decisions by pushing the tag on 2026-10-09, before the
confirmatory run started. Neither decision was changed. The frozen
preregistration is not edited; this note records the difference.

## Terms used in the preregistrations

- **read-only v1 files**: the 23 files of the v1 round (the v1 estimators,
  evidence layer and bench modules, the v1 scripts, the frozen protocols,
  `docs/metrics.md` and the v1 preregistration) whose SHA-256 the v1
  regression gate pins (`READ_ONLY_V1_SHA256` in
  `scripts/v2/regression_gate.py`).
- **`1abff4e`** (v1 preregistration, section 1, and the `source` fields of the
  frozen bench protocols): the pre-edit name of the development commit
  `87654ec`; both have the tree `5396cfd177926f5b2b8a107b48d10022d9bc86e6`.
- **"spec V2-5"**, **"novelty synthesis"** (v1 preregistration): working
  documents of the 1.1.0 development (a design note and the planning document
  in which the hypotheses were first formulated); they are not versioned and
  are not needed to read the preregistration.
- **"v2" inside the v1 preregistration** (for example "the v2 verdict
  semantics"): the second evidence-layer revision of release 1.1.0 (verdicts
  `EXCLUDED`, `MPC_CONSISTENT`, `UNDETERMINED`; protocol schema
  `impact-mpc-protocol/2`), not the MPC-Bench v2 round.
- **HN4, H-IIM-n, H-RAM-n, H-PDI-n, HR1, HR2** (titles of v2 hypotheses):
  numbered hypotheses of the component designs of the v2 round (NAS, IIM,
  RAM, PDI and the status rule) from which the HCv2 hypothesis was derived.
  **HC4v2-R, HC5v2** and similar: the v2 successor of the v1 hypothesis with
  that number.
- **D1 to D13** (v2, section 4.7): the descriptive analyses; D10 is not used.
- **"design x.y"** (v2): sections of the design document of the round, a
  working document that is not versioned (v2 preregistration, section 0).
  The terms that appear in the frozen v2 files are explained in
  [`protocols/v2/README.md`](../../protocols/v2/README.md#terms-in-the-frozen-files).

## Scope

The paper-2 (empirical) hypotheses are a different document: the executable
registry `predictions/registry.yaml`, which stays a draft until its own
freeze. Development results quoted in the preregistration are labelled as
development findings; no number in it is a confirmatory result.
