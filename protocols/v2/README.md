# MPC-Bench v2 protocols

This folder holds the protocol files of MPC-Bench v2. The v1 protocols one
level up (`mpc_bench_v1.json`, `mpc_bench_v1_anchored.json`,
`mpc_default_v1.json`, `mpc_bench_v1_reference_summary.json`,
`applicability_registry_v1.json`) are frozen. v2 never edits them, and v1
records are still judged under them, byte for byte.

## Files

| File | Content | Status |
|---|---|---|
| `seed_map_v2.json` | The v2 seed map. It records the use of every development and confirmatory seed block, the seeds used before v2, and the reserved random-stream keys. | present |
| `mpc_bench_v2_template.json`, `mpc_default_v2.json` | Bench template and paper-2 default protocol, schema `impact-mpc-protocol/3` (status rule `tost-v2`). | added with the v2 status rule |
| `hypotheses_v2.json` | Machine-readable v2 hypotheses (decision rules, seeds, cells). | added with the v2 evaluator |
| `generated/` | Family protocols with anchors, `N_anch`, precision blocks and hashes. They are written by the protocol builder from the development calibration, not by hand. | added before the freeze |

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
  sessions per network. Keys 31-40 stay free for further named streams
  (the benchmark design names 31 and 32 for its substrate nulls).
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
* `scripts/v2/regression_gate.py` runs on every merge. It checks that the v1
  path is unchanged:
  * read-only v1 files and frozen protocol hashes are unchanged;
  * every stored C1-C3 record and null-calibration row is re-judged under
    its v1 protocol with no difference;
  * the frozen evaluator's tables and the null-calibration rate and verdict
    tables are reproduced byte for byte;
  * the 14 verification re-runs and about 30 stratified stored records are
    reproduced bit for bit;
  * a v1 BOLD task still ends in the v1 NAS band error.

  `--quick` runs the static checks and the re-judging in seconds. The full
  gate re-runs tasks from scratch; with 6 workers it takes about 20 minutes,
  most of it for the two system-bearer patchworks. The stored v1 outputs
  (`outputs/paper1_mpcbench`) are not versioned, and the gate fails with
  exit status 2 where they are missing. The tests run the full gate only
  with `MPCBENCH_RUN_SLOW=1`.
