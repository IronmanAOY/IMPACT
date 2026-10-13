# MPC protocols

A protocol declares everything an MPC verdict depends on besides the data:
the necessity set `N`, the evidence channels, the construct-scale cutoffs
`(z, delta)` and `alpha`, the null families, the reference anchor, the source
rule, the estimator modes and the bearer nodes. It is an
`impact_pipeline.evidence.Protocol` stored as JSON (schema
`impact-mpc-protocol/1`). Its hash is the SHA-256 of the canonical JSON of
`Protocol.to_dict()` (sorted keys, compact separators, every cutoff written
out) and is recorded with every verdict (`MPC_protocol_hash`, the bench
manifests and records, the null-calibration summary). The
[metrics reference](../docs/metrics.md#10-protocols) describes the fields.

A protocol without a `status_rule` block is judged by the interval rule (the
empirical pipeline); a protocol with the block is judged by the TOST rule
(`status_rule.version` `tost`: ABSENT as a two-sided equivalence test at
`alpha_A = alpha / 5`), which also reads the fields of the benchmark
protocols (directions, rank gates, estimator versions, SE methods, anchors,
precision block, concordance cells; see
[the TOST rule](../docs/metrics.md#9-tost-status-rule)). The empirical
pipeline refuses a protocol with a `status_rule` block.

| File | Use | SHA-256 |
|---|---|---|
| `mpc_default.json` | default protocol for empirical data (`run_pipeline.py` without `--protocol`, `--data-origin real`) | `0fbfe90dc00e5f703f2fac60207f1399066b59c4317fb30be5a337cf051c842b` |
| `mpc_behavioural_ram.json` | **opt-in only**, a substantive choice to fix before use (`--protocol protocols/mpc_behavioural_ram.json`): RAM declared on its behavioural channel only; results under it are behavioural-RAM results | `328101d46c46646ea2075f6daa6af75ac484cca40fed8e60fb377de0d339d9e0` |
| [`examples/`](examples/README.md) | **examples** of protocols derived from `mpc_default.json` that declare a NAS hub (Schaefer-400 / 7 networks, 64-channel EEG) | see `examples/*.derivation.json` |
| `bench/template.json` | template of the benchmark protocols (TOST rule): the status rule, NAS directions, IIM rank gate, estimator versions, candidate SE methods and estimator options; reference `pending` | `c19f73790e965c10aa3809bd86933b4021418aee9c261b9c5a21fc8ecf62929b` |
| `bench/<key>.json` | the benchmark protocols, one per family protocol key | see [below](#bench-protocols-bench) |

`tests/test_protocols.py`, `tests/test_protocol_examples.py` and
`tests/test_docs_consistency.py` check that the hashes in this table are the
hashes of the files.

## Cutoffs and alpha (all protocols)

`(z, delta) = (0.25, 0.10)` for every principle and `alpha = 0.05`
(one-sided). `z` is the smallest effect of interest for presence (a quarter
of the nominal positive control's excess over its null); `delta < z` keeps
"absent" (negligible, a tenth of it) distinct from "weak". ABSENT therefore
needs an estimate close to its null with a small SE; the SE each rule needs
is in the [metrics reference](../docs/metrics.md#92-decision).

## `mpc_default.json` (default for empirical data)

`run_pipeline.py` uses it when `--protocol` is not given, the data are
empirical (`--data-origin real`, the default) and no `--necessity-set` is
given (`--protocol none` builds the protocol from the flags; dummy/synthetic
data always do). `tests/test_protocol_examples.py` checks its hash and the
file's SHA-256.

Two consequences matter for every empirical run (the declarations behind
them are listed [below](#mpc_defaultjson-declarations)):

- RAM can be PRESENT but never ABSENT under the default protocol, because
  RAM also declares two channels that have no estimator yet (see
  **Channels** there). Behavioural non-response alone never excludes.
- The default protocol declares NAS `mode='capacity'` without a hub, because
  the hub depends on the grain (atlas or montage). Without a declared hub NAS
  is UNDEFINED in every run (`UNDEFINED:NAS:NO_DECLARED_WORKSPACE`), so a run
  that is meant to measure NAS (e.g. a Hunter campaign) needs a protocol
  derived from the default that declares the hub:
  [`examples/`](examples/README.md) has two worked examples (to be fixed
  before confirmatory use).

## `mpc_behavioural_ram.json` (opt-in)

`mpc_default.json` with one change: RAM declares only its implemented
behavioural channel, `channels.RAM = ["default"]` (name
`mpc-behavioural-ram`). Everything else (necessity set, cutoffs, alpha,
null families, reference, source rule, estimator modes, bearers) is
identical; `tests/test_protocol_examples.py` checks this. It is never
selected automatically: `run_pipeline.py --protocol
protocols/mpc_behavioural_ram.json` only.

Under it RAM can be ABSENT when the behavioural channel is credibly at its
null, so RAM can exclude. Selecting it is a substantive decision that has to
be fixed (with its hash) before use, and results under it are reported as
**behavioural-RAM results**: an ABSENT RAM means "no
responsiveness-and-adaptation above the null in the recorded behaviour", not
absence of responsiveness (covert, perturbational and endogenous
responsiveness are not measured by this channel). A dataset that measures
covert or perturbational responsiveness declares those typed channels in a
derived protocol instead (for example `["behavioural_feedback",
"covert_neural"]` for events with an `impact_channel` column), so that RAM is
ABSENT only when every measured channel is. `scripts/build_example_protocols.py
--base protocols/mpc_behavioural_ram.json --out-dir <dir>` derives the
example hub protocols from it (their sidecars carry this caveat).

## `mpc_default.json`: declarations

- **Necessity set**: all five principles (RAM, PDI, NAS, IIM, SRPI).
- **Channels**: one untyped `default` channel for PDI, NAS, IIM and SRPI. RAM
  declares `default` (the run's events without a channel label) **and** the
  two channels that have no estimator yet, `perturbational` and `endogenous`.
  A declared channel that is not implemented is UNDEFINED
  (`NOT_IMPLEMENTED:RAM:<channel>`), and a principle is ABSENT only when every
  declared channel is ABSENT (strong-Kleene OR), so under this protocol RAM
  can be PRESENT (through `default`) but never ABSENT. Behavioural
  non-response does not establish the absence of responsiveness and
  adaptation (covert command following, dreaming, paralysis), so a
  necessity-only rule must not exclude on it; undefined is not absent. The
  price is that RAM cannot contribute an exclusion on empirical data, and a
  necessity test of RAM cannot be falsified by RAM evidence, until those
  channels are measured. Dropping these channels (as
  `mpc_behavioural_ram.json` does) is a substantive decision that has to be
  justified and fixed before use.
- **Cutoffs**: `(z, delta) = (0.25, 0.10)` for every principle, `alpha =
  0.05`.
- **Null families**: RAM `onset_jitter` (rigid event-train shift), PDI
  `circular_shift` (the repertoire estimator's state-count null), NAS
  `block_circular_shift` (capacity mode), IIM `circular_shift`. SRPI is not
  declared: its family depends on the mode each run uses
  (`yoked_label_permutation` for agency, `label_permutation` for the legacy
  fallback) and is recorded per row.
- **Reference**: `cohort_high_state`, session `awake`: the cohort's high-state
  mean excess over the null per principle and channel. The high-state runs
  are part of their own reference (their `c` averages 1), so their verdicts
  are not independent tests of necessity; necessity tests among
  report-positive episodes need an external reference
  (`scripts/compute_empirical_reference.py`, from a declared held-out subset
  of participants: keys `P` for the single-channel principles and
  `RAM:default` for RAM's behavioural channel; RAM's unimplemented channels
  get no reference, `NOT_IMPLEMENTED`, and stay UNDEFINED).
- **Source rule**: `single_source` (one bearer, joint dependence above null
  when components come from different node sets).
- **Estimator modes**:
  - RAM: strict contract (`require_explicit_feedback`,
    `require_explicit_goals`) and `update='prediction_error'` where the run
    logs choices and rewards; runs without that log use
    `update_fallback='feedback_magnitude'` (declared here, recorded per row
    in `RAM_estimator` and `RAM_mode_reason`);
  - PDI: `mode='repertoire'` (unlabelled repertoire of distinguishable
    states; the window is in samples, so modality-specific windows need a
    derived protocol; scalp EEG needs the forward-model validation of the
    applicability registry first);
  - NAS: `mode='capacity'` (declare `workspace_nodes` in a derived protocol
    when the hub set is known; without it NAS is UNDEFINED,
    `NO_DECLARED_WORKSPACE`; see [`examples/`](examples/README.md));
  - IIM: calibrated Delta_Psi with the `node_shrinkage` TPM and
    `cut_mode='bidirectional'` (`'directional'` is available);
  - SRPI: `mode='agency'` where the run has self_caused/other_caused events,
    otherwise `mode_fallback='legacy'` (recorded in `SRPI_mode_reason`).
- **Bearer nodes**: all nodes of the run.
- Recommended sampling settings for empirical runs: `--null-surrogates 19`
  and `--bootstrap-se 100` (moving-block bootstrap; `se_df = 99`, so the
  Student-t quantile is within 1% of the normal one; the Monte-Carlo
  relative error of the SE is about 7%); on Hunter
  `--hunter-iim-bootstrap-se 100` (IIM cost grows with `1 + B`).

## Bench protocols (`bench/`)

The benchmark runner (`scripts/run_bench.py`) reads one protocol per
family protocol key from `bench/<key>.json` (name `mpc-bench-<key>`): the
families `A-R`, `A-H`, `A-P`, `A-Q10`, `A-Q25`, `A-J`, `A-RAM160` and
`A-none` (family A under the input declarations), `C-R` and `C-H` (family
C), the estimator forms `<family protocol>+<form>` (for example
`A-R+nas_secondary`, `hopf-eeg64+iim_quadrants`) and one protocol per forward
view (`hopf-*`, `fwdA-*`, `fwdA_bold-*`). Each carries the `status_rule`
block, per-direction NAS anchors, the external reference (the positive
control's mean excess over its null on the development reference block,
seeds 900-939), an anchors block with the validity and specificity of every
principle and the anchored necessity set `N_anch`, a hash-covered
`precision` block (the testability rows) and the admitted PDI concordance
cells. Without a file for a key, development runs derive a draft from
`bench/template.json` (reference `pending`, so every component is
UNDEFINED(INVALID_ANCHORS)); confirmatory runs use the files only. The
family-B protocols are built in code
(`impact_pipeline.bench.designs.family_b.family_b_protocol`).

## Use

```bash
# default protocol (NAS UNDEFINED without a hub)
python run_pipeline.py --dataset-id ds003171 \
    --bids-root /data/openneuro/ds003171 --out-dir outputs/ds003171 \
    --null-surrogates 19 --bootstrap-se 100
# with a declared NAS hub (an example: fix the hub before confirmatory use)
python run_pipeline.py --dataset-id ds003171 \
    --bids-root /data/openneuro/ds003171 --out-dir outputs/ds003171 \
    --protocol protocols/examples/mpc_default_schaefer400_7networks_hub.json \
    --null-surrogates 19 --bootstrap-se 100
# external reference from held-out participants (per channel: RAM:default)
python scripts/compute_empirical_reference.py \
    --data-dir outputs/ds003171/preprocessed --atlas schaefer400 \
    --bids-root /data/openneuro/ds003171 --dataset-id ds003171 \
    --protocol protocols/examples/mpc_default_schaefer400_7networks_hub.json \
    --reference-subjects @heldout.txt --evaluation-subjects @evaluation.txt \
    --out outputs/ds003171_reference/empirical_reference.json \
    --write-protocol outputs/ds003171_reference/mpc_default_hub_external.json
```
