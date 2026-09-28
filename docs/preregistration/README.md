# Preregistration of the MPC-Bench computational hypotheses (paper 1)

This folder holds the preregistration of the computational hypotheses of
paper 1 (the MPC-Bench validation of the IMPaCT measurement framework):

- [`MPC_BENCH_PREREGISTRATION.md`](MPC_BENCH_PREREGISTRATION.md): the
  hypotheses HC1-HC10, their decision rules and thresholds, the frozen
  protocols and their hashes, the calibration decisions made on development
  data, and the confirmatory run plan.

## Status: frozen locally, not registered

**Nothing has been registered publicly.** The preregistration is frozen in
this repository by the local annotated git tag `mpcbench-freeze-v1`, which
points at the commit that contains this folder, the frozen protocols
(`protocols/mpc_bench_v1.json`, `protocols/mpc_bench_v1_validated.json`) and
the code that will produce and evaluate the confirmatory results. The tag has
not been pushed.

A local tag is not a preregistration: its date and content can be changed by
whoever controls the repository. Before any confirmatory run, the author
must register it on OSF:

1. Push the branch and the tag (`git push origin mpcbench-freeze-v1`), so the
   commit is publicly reachable.
2. Create an OSF registration (for example the "OSF Preregistration" or
   "Open-Ended Registration" template) and attach
   `MPC_BENCH_PREREGISTRATION.md` as it is at the tagged commit.
3. Record in the registration: the full commit hash of the tag
   (`git rev-list -n 1 mpcbench-freeze-v1`), the tag name, and the two
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
| Frozen commit | the commit the tag points to (`git rev-list -n 1 mpcbench-freeze-v1`) |
| OSF registration | not yet registered |

## Scope

The paper-2 (empirical) hypotheses are a different document: the executable
registry `predictions/registry.yaml`, which stays a draft until its own
freeze. Development results quoted in the preregistration are labelled as
development findings; no number in it is a confirmatory result.
