# MPC-Bench v2 protocols

This folder holds the protocol files of MPC-Bench v2. The v1 protocols one
level up (`mpc_bench_v1.json`, `mpc_bench_v1_anchored.json`,
`mpc_default_v1.json`, `mpc_bench_v1_reference_summary.json`,
`applicability_registry_v1.json`) are frozen. v2 never edits them, and v1
records are still judged under them, byte for byte.

## Files

Every file in this folder except this README is frozen at the tag
`mpcbench-freeze-v2` and listed with its SHA-256 in section 13 of the
[v2 preregistration](../../docs/preregistration/MPC_BENCH_PREREGISTRATION_V2.md#13-frozen-files).

| File | Content |
|---|---|
| `seed_map_v2.json` | The v2 seed map. It records the use of every development and confirmatory seed block, the seeds used before v2, and the reserved random-stream keys. |
| `mpc_bench_v2_template.json`, `mpc_default_v2.json` | Bench template and paper-2 default protocol, schema `impact-mpc-protocol/3` (status rule `tost-v2`). |
| `hypotheses_v2.json` | Machine-readable v2 hypotheses (decision rules, seeds, cells), read by `impact_pipeline.v2.hypothesis_engine`; status `final`. It carries the calibration decisions: the declared dependencies (CD-7) and the mechanism-on table (CD-8) copied from the decided build, and the expectations restated from development. |
| `held_out_predictions_v2.json` | The held-out elements (HO-1 to HO-7, preregistration section 9) and the predicted admission table (section 4.5), transcribed with the one documented correction (HCv2-21, CD-5). The table adds the admission rows that no hypothesis states as a decision. It is the committed copy that the logged release of the held-out predictions and forward anchors names. The tests (`tests/v2/test_hypothesis_engine.py`) check it against `hypotheses_v2.json`, whose held-out predictions they check against the commit of the held-out predictions. It has not changed since the release. |
| `generated/` | Written by the protocol builder (`scripts/v2/build_protocols_v2.py`) from the development calibration and the decisions file, never by hand: the family protocols `mpc_bench_v2_<key>.json` (anchors, `N_anch`, precision blocks, concordance routes, hashes), `forward_anchors.json`, `testability_table.json`, `mechanism_on.json`, `declared_dependencies.json`, `calibration_decisions.json`, `calibration_evidence.json` and `build_manifest.json`. The manifest lists the inputs of the build with their SHA-256 and the steps that followed it outside the folder. |

A protocol without the `status_rule` block is judged by the v1 rule, byte for
byte (schema `impact-mpc-protocol/2`). A protocol with the block has schema
`impact-mpc-protocol/3`.

## Seed policy

Code: `impact_pipeline.v2.seeds`. The map is checked by
`load_seed_map()` / `validate_seed_map()`.

* Development seeds are 0-999. Confirmatory seeds are 20000 and above. The v1
  confirmatory block 10000-19999 is never reused, and 1000-9999 is not
  assigned. Any other seed is refused.
* A run uses one split: development and confirmatory seeds are never mixed.
  A confirmatory run refuses every seed below 20000.
* A record's split label is derived from its seed. It is never written by
  hand.
* The policy classifies task seeds and seed bases. Seeds that a task derives
  from them (null seeds, hashed replicate seeds) are not classified.
* Seeds 980-984 are for smoke tests of held-out conditions. The harness
  discards their outputs unread.

Random streams:

* The structural stream and every v1 stream stay `SeedSequence(seed)`.
* A twin with replicate `r` in 1-30 draws its task schedule, process noise
  and rest from `SeedSequence([seed, r])`. The largest twin set has 8
  sessions per network. Keys 31-40 are reserved for further named streams
  and are not used in this round.
* Label errors, cue-onset jitter and the staggered driver use the reserved
  keys 41, 42 and 43 (`SeedSequence([seed, key])`). A twin index can never
  reach them.

## Run hygiene

* Records follow the result schema `mpc-bench-result/3`
  (`impact_pipeline.v2.records`): one simulation and several scorings per
  task. Each component carries its family-protocol id and hash.
* A component whose estimator raises is `UNDEFINED(ESTIMATOR_ERROR:<type>)`.
  The other components of the task are not touched, and the task status is
  then `ok_with_component_errors`.
* Every UNDEFINED reason comes from the central vocabulary
  `impact_pipeline.v2.reasons`. Every reason maps to UNDEFINED, never to
  ABSENT.
* The v2 confirmatory guard (`impact_pipeline.v2.provenance.confirmatory_guard`)
  refuses a dirty tree (modified tracked files, or untracked files under
  `src/` or `scripts/`), a missing tag `mpcbench-freeze-v2` (a branch of
  that name does not count), `src/` or `scripts/` trees that differ from
  the tag, and seeds below 20000. It checks git trees rather than commits,
  so a rewrite of commit messages does not break the guard.
* Every run records the environment. `scripts/v2/env_lock.py` checks the
  environment against the lock (numpy, scipy, numba, pandas; v2 adds no
  package). It also checks that v2 code imports neither mne nor scikit-learn.
* `scripts/v2/regression_gate.py` (integrity check IA-1) checks that the v1
  path is unchanged:
  * read-only v1 files and frozen protocol hashes are unchanged;
  * every stored record and null-calibration row of the v1 confirmatory
    outputs (folders `c1/` to `c3/`) is re-judged under its v1 protocol with
    no difference;
  * the frozen evaluator's tables and the null-calibration rate and verdict
    tables are reproduced byte for byte;
  * the 14 verification re-runs and about 30 stratified stored records are
    reproduced bit for bit;
  * a v1 BOLD task still ends in the v1 NAS band error.

  `--quick` runs the static checks and the re-judging in seconds. The full
  gate re-runs tasks from scratch; with 6 workers it takes about 20 minutes,
  most of it for the two system-bearer patchworks. The stored v1 outputs
  (`outputs/paper1_mpcbench`) are not versioned, and the gate fails with
  exit status 2 where they are missing; for this reason the gate is not part
  of CI. The tests run the full gate only with `MPCBENCH_RUN_SLOW=1`.

## Terms in the frozen files

Some strings in the frozen files and in the stored records come from the
development of the round. They are kept as they are, because the files are
hashed and the records carry them; this section says what they mean.

* **"design x.y", "design x.y item n", `design_BENCH`, `design_IIM`,
  `design_RAM_PDI`** (in `hypotheses_v2.json`, `calibration_decisions.json`
  and `calibration_evidence.json`): sections and numbered items of the design
  document of the round and of its component designs. These are working
  documents of the author and are not versioned (preregistration v2,
  section 0); everything needed to run and evaluate the hypotheses is in the
  preregistration, its companions and the frozen files.
* **"design S6"** (in the reason `SRPI declared not applicable (design S6)`
  of family-C1 records): item 6 of the SRPI component design, which declares
  SRPI not applicable on the v1 family-C carrier recording.
* **`PRE_DATA_COMMITMENTS`**: the commitments written before any calibration
  output; they are restated in section 7.6 of the preregistration.
* **CD-1 to CD-14, HO-1 to HO-7, IA-1 to IA-10**: the calibration decisions,
  held-out elements and integrity checks, as defined in section 0 of the
  preregistration.
* **`CALIBRATION_PENDING`** (in the run manifests and in the `where` fields of
  `calibration_decisions.json`) is the name of the table of decided
  calibration values in `impact_pipeline.bench.run_bench_v2` and
  `impact_pipeline.bench.designs_v2.anchors`. The values are final; the name
  and the phrase "development feasibility work" in its `meaning` text are
  kept because every run manifest records them.
* **"decider", "critic", "upheld", "dispute"** (in the `note` texts of
  `calibration_decisions.json`): each calibration decision was drafted and
  then reviewed in two separate passes before the freeze. The *decider* pass
  drafted the value and its note from the development evidence and the
  written rules; the *critic* pass checked the draft against the same
  evidence and rules. "Upheld" means the critic accepted the draft; a
  *dispute* is a disagreement between the two passes, and the note says how
  it was resolved. Both passes were part of preparing the decisions file,
  not separate decisions.
* **"BLOCKING: the code constant impact_pipeline.v2.iim_v5.BOOTSTRAP_SE_DF is
  still 12.0 ... (DECISION_LOG follow-up 1)"** (note of CD-3): a follow-up
  recorded when the decision was written, in a working log that is not
  versioned. It was done before the freeze: `BOOTSTRAP_SE_DF` is 9.0 at the
  tag, as the decision requires.
* **"to be confirmed by the author with the freeze push"** (note of the
  paper-2 regime, CD-12): one of the two author-level decisions of section
  10.4 of the preregistration. How the author confirmed them is recorded in
  [`docs/preregistration/README.md`](../../docs/preregistration/README.md#status-of-v2).
