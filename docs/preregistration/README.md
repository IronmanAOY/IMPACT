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

## Status: frozen locally, not registered

**Nothing has been registered publicly.** The preregistration is frozen in
this repository by the local annotated git tag `mpcbench-freeze-v1`, which
points at the commit that contains this folder, the frozen protocols
(`protocols/mpc_bench_v1.json`, `protocols/mpc_bench_v1_anchored.json`) and
the code that will produce and evaluate the confirmatory results. The tag has
not been pushed.

The commit messages of the history up to and including the freeze commit
were edited on 2026-09-29 (wording and author identity only; no file was
changed). The freeze commit is now `f2cf249`
(`f2cf2492a6cb4a1b0b1f5b8d680802fded31c1d1`); before the edit it was
`b908ee3` (`b908ee3230d8b93a1983960a9621e49bdd5752b2`). Both commits have the
same tree, `72fcc7031de5cc6f660af7f54dc5e0a8b3bd6a6d`, so their content is
identical (check with `git rev-parse <commit>^{tree}`). The tag points at
`f2cf249`; outputs produced before the edit record `b908ee3` as their code
commit.

A local tag is not a preregistration: its date and content can be changed by
whoever controls the repository. Before any confirmatory run, the author
must register it on OSF:

1. Push the branch and the tag (`git push origin mpcbench-freeze-v1`), so the
   commit is publicly reachable.
2. Create an OSF registration (for example the "OSF Preregistration" or
   "Open-Ended Registration" template) and attach
   `MPC_BENCH_PREREGISTRATION.md` as it is at the tagged commit. Attach the
   errata section added after the freeze as well (the current version of the
   file, or its "Errata" section as a separate file), or mention it in the
   registration alongside the frozen document, so that readers see both the
   frozen text and the dated corrections.
3. Record in the registration: the full commit hash of the tag
   (`git rev-list -n 1 mpcbench-freeze-v1`, `f2cf2492a6cb4a1b0b1f5b8d680802fded31c1d1`),
   the tag name, the note that this commit was `b908ee3230d8b93a1983960a9621e49bdd5752b2`
   before the commit messages were edited on 2026-09-29 without file changes
   (same tree `72fcc7031de5cc6f660af7f54dc5e0a8b3bd6a6d`), and the two
   protocol hashes listed in section 3 of the preregistration (they can be
   recomputed with `python -c "from impact_pipeline.evidence import Protocol;
   print(Protocol.from_json('protocols/mpc_bench_v1.json').hash)"`).
4. Only then run `scripts/mpcbench_confirmatory.sh` (the confirmatory steps
   refuse to run on anything but a clean checkout whose `src/` and `scripts/`
   equal the tagged commit).

Record the OSF registration identifier and date in this README in a later
commit (documentation-only commits after the tag do not affect the
confirmatory guard, which compares only `src/` and `scripts/`).

| Item | Value |
|---|---|
| Freeze tag | `mpcbench-freeze-v1` (local, annotated) |
| Frozen commit | the commit the tag points to (`git rev-list -n 1 mpcbench-freeze-v1`): `f2cf249`, originally `b908ee3` (commit messages edited 2026-09-29, same tree `72fcc70`) |
| Errata | section "Errata" of the preregistration, added 2026-09-29 after the freeze (documentation only) |
| OSF registration | not yet registered |

## Scope

The paper-2 (empirical) hypotheses are a different document: the executable
registry `predictions/registry.yaml`, which stays a draft until its own
freeze. Development results quoted in the preregistration are labelled as
development findings; no number in it is a confirmatory result.
