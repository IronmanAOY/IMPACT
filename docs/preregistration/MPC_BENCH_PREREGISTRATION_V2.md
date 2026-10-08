# MPC-Bench v2 preregistration: computational hypotheses of paper 1, revision round

Status: frozen locally by the annotated tag `mpcbench-freeze-v2`, which points
to the commit that contains this document; **not yet registered publicly**
(see [README](README.md)). Written 2026-10-08, after the development
calibration (development seeds 0-999 only) and after the logged release of the
held-out predictions, before any confirmatory run. The v1 preregistration
([`MPC_BENCH_PREREGISTRATION.md`](MPC_BENCH_PREREGISTRATION.md), tag
`mpcbench-freeze-v1`) and the v1 outcomes are not changed by this document.

Companion documents, frozen with this one:

- [`v2/testability_table.md`](v2/testability_table.md): the testability table
  (development precision and the gate of every gated part);
- [`v2/development_expectations.md`](v2/development_expectations.md): the
  development expectation of every hypothesis and part, with the development
  dry run of the evaluator;
- [`v2/operating_characteristics.md`](v2/operating_characteristics.md): the
  operating characteristics of every decisive part at the confirmatory sizes,
  the seed resizes and the confirmatory cost.

## Contents

0. [Reading guide](#0-reading-guide)
1. [Purpose and scope](#1-purpose-and-scope)
2. [What changed from v1, and why](#2-what-changed-from-v1-and-why)
3. [Estimators and the status rule](#3-estimators-and-the-status-rule)
4. [Systems and designs](#4-systems-and-designs)
5. [Seed map](#5-seed-map)
6. [Hypotheses](#6-hypotheses)
7. [Calibration on development data](#7-calibration-on-development-data)
8. [Operating characteristics](#8-operating-characteristics)
9. [Held-out elements and their release](#9-held-out-elements-and-their-release)
10. [Integrity, prior contact and deviations](#10-integrity-prior-contact-and-deviations)
11. [Confirmatory run plan and cost](#11-confirmatory-run-plan-and-cost)
12. [Evaluation and reporting rules](#12-evaluation-and-reporting-rules)
13. [Frozen files](#13-frozen-files)

## 0. Reading guide

- **What decides.** The decisive specification of every hypothesis is
  `protocols/v2/hypotheses_v2.json` (schema `mpc-bench-hypotheses/2`, status
  `final`) as it is in the tagged commit. The evaluator
  (`scripts/bench_hypotheses_v2.py`, engine
  `impact_pipeline.v2.hypothesis_engine`) reads only that file and the frozen
  protocols. Section 6 reproduces every hypothesis and part from the file
  verbatim: statement, text, rule, parameters, data selection, label,
  prediction, development expectation and notes.
  `tests/v2/test_prereg_v2_consistency.py` checks that the ids, rule names and
  parameters quoted here equal the file, that the protocol hashes of section 13
  equal the generated protocols and that the seed map of section 5 equals
  `protocols/v2/seed_map_v2.json`. Should this text and the file ever differ,
  the file is what was evaluated, and the difference is reported as a
  deviation.
- **Design document of the round.** The revision was designed in
  `docs/manuscript/research/v2_design/MPC_BENCH_V2_DESIGN.md` and its component
  designs (NAS, IIM, RAM and PDI, SRPI, rule, benchmark), with the commitments
  written before any calibration output (`PRE_DATA_COMMITMENTS.md`, same
  folder). These are working documents of the author's research folder and
  are not versioned in this repository. References such as "design 3.6" point
  to them for provenance only; everything needed to run and evaluate the
  hypotheses is in this document, its companions and the frozen files of
  section 13.
- **Labels of hypothesis parts.** **C** (construct-derived): the threshold or
  the predicted direction follows from an exact target, the construct
  statement or a theorem, not from a development number. **R** (replication):
  the threshold was set with knowledge of a development result on the same
  generator; a SUPPORTED outcome is a replication on new seeds, not a new
  test. **HO** (held out): no v2 estimator output on this condition existed
  before the freeze (section 9); the prediction was written from mechanism.
- **Outcomes.** `SUPPORTED`, `FALSIFIED`, `INDETERMINATE` (the rule does not
  decide), `NOT_EVALUABLE` (missing input, anchor or manipulation check) and
  `NOT_TESTABLE_BY_DESIGN` (declared before the freeze from the attainable
  precision, section 3.7). A part is **decisive** if its outcome enters the
  hypothesis outcome and **reported** otherwise. A predicted failure is a
  decisive part in its own right. Tier-B hypotheses are `NOT_RUN`.
- **Codes.** CD-1 to CD-14 are the calibration decisions (section 7), HO-1 to
  HO-7 the held-out elements (section 9), IA-1 to IA-10 the integrity checks
  (section 4.6), D1 to D13 the descriptive analyses (section 4.7). M and
  HC1-HC10 are the v1 hypotheses.
- **Development numbers.** Every number called "development" comes from
  development seeds 0-999 and is flagged `DEVELOPMENT - NOT A RESULT` in its
  source. It justifies a frozen choice or states what development showed; none
  may be quoted as a confirmatory result.

## 1. Purpose and scope

Paper 1 presents a measurement framework for the minimal preconditions of
consciousness (MPC): five null-anchored component estimators (RAM, PDI, NAS,
IIM, SRPI), a three-valued exclusion rule (`EXCLUDED` / `MPC_CONSISTENT` /
`UNDETERMINED`; the principles are treated as necessary conditions, which
license exclusion only, and `MPC_CONSISTENT` is not an attribution of
consciousness) and MPC-Bench, a white-box benchmark of simulated systems whose
mechanisms are switched on and off. MPC-Bench v1 tested the v1 instruments
(prerequisite M and HC1-HC10). This document preregisters the hypotheses
HCv2-0 to HCv2-24 of the v2 round, which tests revised instruments.

The hypotheses test instruments, not the theory: the null calibration and the
sampling-error calibration of the estimators, what each estimator identifies
and where identification fails, whether mechanism-bearing systems are falsely
excluded, the behaviour of the status rule and of the verdict rule, and the
admission of estimators to forward-modelled EEG and BOLD observations. The
systems' lack of consciousness is a stipulated premise, and no outcome bears on
whether the five principles are necessary for consciousness.

Scope of the round:

- **Tier A only.** The Tier-B items of the design (HCv2-B1 to HCv2-B8: SRPI v3,
  NAS hub identity, the RAM-PE persistence gate, the single-source test by
  joint dependence, the rule audit, the zero-mean workspace broadcast, the PDI
  slow-drift continuum and the forward arms for RAM-PE and SRPI) were not
  admitted. They are listed as not run, and where a v1 predecessor exists its
  outcome stands (HC7 and HC9 have no v2 successor in this round).
- **No held-out generator.** Family A and the null-calibration generator were
  development substrates in v1; family C and the Hopf model were run
  confirmatorily in v1 and were developed on for v2; family B is white-box by
  construction. Most v2 parts are therefore replications on new seeds of seen
  generators, which is weaker than v1's held-out family C. What is held out is
  a set of conditions (section 9).
- **Paper 2.** Its empirical hypotheses are registered separately
  (`predictions/registry.yaml`, still a draft). Section 7.5 records the paper-2
  regime that the forward admission of this round is computed at.

Paper 1 states the following whatever the outcomes (design 1.5):

1. There is no held-out generator; most v2 parts are replications on new seeds.
2. RAM-PE and PDI v3 are tested only on family A and on forward-modelled
   family-A agents; no contingency or goal-directedness test exists in v2
   (`scrambled_feedback` is mechanism-on for RAM-PE).
3. NAS identification is licensed only under a complete input declaration; on
   human data this will rarely hold, and NAS PRESENT then means hub-periphery
   dependence only.
4. IIM PRESENT on human data is not evidence of recurrent integration; IIM
   contributes to exclusion mainly through ABSENT, which is not reachable at
   bench precision.
5. SRPI in the joint bench is the v1 separability instrument; its ABSENT is
   unreachable at 30 reafference pairs.
6. The squared (information-like) statistics of NAS and IIM make
   `delta = 0.10` correspond to about 0.32 of the reference effect amplitude
   (section 3.8).
7. The declared workspace of the Hopf model is six subcortical regions (a v1
   modelling choice kept in this round).
8. The v1 outcomes are final and are not re-read under v2.

What stays as reported from v1: the 11 v1 outcomes (M and HC1 SUPPORTED;
HC2-HC10 FALSIFIED) with their part-level outcomes and NOT_EVALUABLE parts,
every v1 number and deviation, the frozen v1 artefacts (tag
`mpcbench-freeze-v1`, commit `f2cf249`; protocols `mpc_bench_v1.json`,
`mpc_bench_v1_anchored.json`, `mpc_default_v1.json`,
`applicability_registry_v1.json`; estimator versions `ram-v2`, `pdi-v2`,
`nas-v2`, `iim-v4`, `srpi-v2` of 2026.09; generators 1.1.0). The v1 code path
stays selectable and byte-identical in the v2 tree (IA-1). Re-judging v1
records under the v2 rule, and development re-readings of v1 parts, are design
data and never results. A v2 successor does not change its v1 predecessor's
outcome.

## 2. What changed from v1, and why

### 2.1 The v1 outcomes

Of the 11 preregistered v1 outcomes, prerequisite M and HC1 (null calibration)
were SUPPORTED and HC2-HC10 FALSIFIED; none was INDETERMINATE. Every family-A
outcome reproduced the development expectation recorded in the v1
preregistration, so the confirmatory run replicated the development failures
rather than discovering them. The v1 results and their independent analysis
are kept with the v1 outputs (`outputs/paper1_mpcbench/RESULTS_SUMMARY.md` and
`outputs/paper1_mpcbench/analysis_v1_fable/ANALYSIS_V1.md`; not versioned).

### 2.2 Causes of the v1 failures and the v2 response

The independent v1 analysis ranked six causes by how much of the falsification
pattern they explain. v2 responds to each with a change of an instrument, of
the status rule or of the test, never by re-reading a v1 result.

| v1 cause | What it explained in v1 | v2 response |
|---|---|---|
| 1. Common endogenous drive in family A: the context patterns reach every module including the hub at zero lag, and the slow rhythm reaches every unit | NAS and IIM failures in HC3, HC4, HC5, HC6 and HC9(c) (for example NAS kept 90 % of its anchor with the workspace removed) | NAS v3 and IIM v5 condition on declared exogenous inputs through one input basis. Family A keeps its v1 dynamics and gains a recording device; every family-A system is scored under the complete declaration R and the task-only declaration H from the same simulation, so identification and its failure are tested on identical data (HCv2-7, -8, -14, -24). Under a partial declaration PRESENT certifies dependence only. |
| 2. Observation-domain mismatch in family C: the mechanisms live in the envelope of the oscillators, the recording is the carrier | the missing SRPI anchor, PDI = 0 and RAM blindness in family C | Family C is kept as C1 for NAS and IIM only; RAM-PE, PDI and SRPI are declared `NOT_APPLICABLE_OBSERVATION_MODEL` on C1 (UNDEFINED, never ABSENT; IA-4). |
| 3. A fixed absence margin (`delta = 0.10`) below the attainable precision, tested one-sided | HC4 (SRPI part), HC8(a), much of the UNDETERMINED mass | ABSENT is a two-sided equivalence test at `alpha_A = 0.01` with named reasons (`ABSENT_NOT_REACHABLE`, `NULL_MODEL_VIOLATED`); ABSENT parts whose target cannot reach ABSENT at the development precision are declared `NOT_TESTABLE_BY_DESIGN` before the freeze (section 3.7). The cutoffs are unchanged. |
| 4. Test specification and comparator degeneracy | HC2(b, c, d), HC7, HC9(a), HC10 | IIM uses a rank p-value with a rank gate; rate claims use three-zone rules and demonstrated bounds whose operating characteristics are stated (section 8); predicted failures are decisive parts; deterministic checks moved to the integrity audit; the v1 verdict-stability test HC10 is replaced by the SE calibration on white-box twins (HCv2-4). The rule audit (HC9 successor) is Tier B and not run. |
| 5. Composite and geometry defects of RAM and PDI: factors without a referent in the generator, an unsigned pooled R2, a valley test under the ignition common mode | no RAM or PDI anchor in either family | RAM-PE v3 measures the plasticity channel only, with a signed cross-validated readout; the other RAM facets are declared not applicable. PDI v3 counts states on a content bearer that excludes the declared workspace, with sphered components and a Fisher-axis valley; a concordant count is never exact and is admitted only through a validated battery. |
| 6. Precision and construct of IIM (`se_c` about 0.45 on the positive control; feed-forward propagation and common drive read as integration) | HC2(d), HC3 and HC4 for IIM | IIM v5: directional cuts primary, rank gate, occupancy gate, conditioning on recorded drivers, block-bootstrap SE; at sensor level, rank-safe electrode clusters with symmetric orthogonalisation. |

Two further v1 findings changed the rule: the v1 NAS anchor was dominated by a
nuisance (common drive), which motivates the anchor specificity gate and a
necessity set per protocol (section 3.6); and v1 SRPI produced ABSENTs from
estimates below the null, which the v2 rule no longer allows (IA-9).

### 2.3 Changes by component

| Component | v1 (frozen) | v2 (this round) |
|---|---|---|
| Status rule | one-sided bounds; ABSENT iff the upper bound is below `delta` | `tost-v2`: PRESENT iff `c - q_P se_c > z`; ABSENT iff both one-sided tests at `alpha_A = alpha / 5 = 0.01` pass (TOST); otherwise UNDEFINED with the first applicable reason; exact path only for deterministic known-TPM values |
| Verdict rule | strong-Kleene AND over the necessity set | unchanged, over `N_anch(f)` of each family protocol; every new reason is UNDEFINED-kind |
| Anchors | validity only (lower bound of the positive control's mean excess > 0), development seeds 900-919 | validity (at least 36 of 40 finite, lower bound > 0) and a specificity gate (paired contrast with the own lesion at least 0.5 x the anchor, lower bound > 0) on 900-939 per protocol; replicated confirmatorily (HCv2-6) |
| NAS | `nas-v2-2026.09`, capacity, argmin-z gate, one anchor | `nas-v3-2026.10`: conditional Geweke causalities between declared hub and periphery blocks given a declared input basis; per-direction anchors; PRESENT iff both directions, ABSENT iff either direction at `alpha_A / 2`; resolvability gate (`SAMPLING_UNRESOLVED` on BOLD) |
| IIM | `iim-v4-2026.09`, bidirectional cuts, jackknife | `iim-v5-2026.10`: directional cuts primary, rank gate `p_ind <= 0.05`, occupancy gate (`N_min = 25`), conditioning on recorded drivers, circular block bootstrap (`se_df = 9`) |
| RAM | `ram-v2-2026.09`, three-factor composite | `ram-v3-2026.10` (RAM-PE): plasticity channel, signed ridge readout of the response-pattern change against the choice-signed prediction error, exact shift null, shift-null SD as SE |
| PDI | `pdi-v2-2026.09`, unlabelled repertoire | `pdi-v3-2026.10`: content bearer, Fisher-axis valley, four split offsets, concordance route admitted per cell by a battery |
| SRPI | `srpi-v2-2026.09` | the same instrument under the v2 rule (Tier A); SRPI v3 is Tier B and not run |
| Benchmark | families A, B, C; whole brain; adversaries | family A with a recording device (declarations R, H and the held-out P, Q10, Q25, J); C1; family B cells for IIM v5; twins; staggered hidden-driver adversaries; new witnesses; a RAM-only arm; the Hopf arm and forward-modelled family A with the admission procedure and registry v3; null-calibration generator on the v1 data with a declared hub partition |
| Hypotheses | HC1-HC10 | HCv2-0 to HCv2-24 (Tier A), declarative (`hypotheses_v2.json`), with labels C/R/HO, testability gates, predicted failures and operating characteristics; integrity audit IA-1 to IA-10 outside the tally |
| Seeds | development 0-999, confirmatory >= 10000 | development 0-999, confirmatory >= 20000; 10000-19999 never reused |
| Guards | freeze tag and clean tree | the v1 regression gate on every merge; the v2 confirmatory guard compares the git trees of `src/` and `scripts/` with the tag `mpcbench-freeze-v2` |

### 2.4 v1 hypotheses and their v2 successors

| v1 | v1 outcome | v2 successor |
|---|---|---|
| M | SUPPORTED | HCv2-0 |
| HC1 | SUPPORTED | HCv2-1, HCv2-3 |
| HC2 | FALSIFIED | HCv2-2, HCv2-11, HCv2-12, HCv2-13 |
| HC3 | FALSIFIED | HCv2-7, HCv2-14(f), HCv2-17, HCv2-19 (selectivity and dose parts) |
| HC4 | FALSIFIED | HCv2-22 |
| HC5 | FALSIFIED | HCv2-23 |
| HC6 | FALSIFIED | HCv2-24 |
| HC7 | FALSIFIED | none in this round (HCv2-B4, Tier B, not run) |
| HC8 | FALSIFIED | HCv2-22 (SRPI part); HCv2-B1 not run |
| HC9 | FALSIFIED | none in this round (HCv2-B5, Tier B, not run) |
| HC10 | FALSIFIED | HCv2-4 (SE calibration); D1 (descriptive) |

## 3. Estimators and the status rule

### 3.1 Versions and settings

| Principle | Estimator | Version | SE method (decided) | `se_df` |
|---|---|---|---|---|
| NAS | NAS v3, conditional capacity | `nas-v3-2026.10` | `jackknife_contiguous_10` (CD-2) | 9 |
| IIM | IIM v5, directional cuts | `iim-v5-2026.10` | `circular_block_bootstrap_10pct_B50` (CD-3) | 9 |
| RAM | RAM-PE v3, prediction-error readout | `ram-v3-2026.10` | `shift_null_sd` (CD-4) | `n_null - 1` |
| PDI | PDI v3, repertoire on the content bearer | `pdi-v3-2026.10` | `jackknife_contiguous_10`; admitted concordant components through the concordance route (CD-5) | 9 |
| SRPI | v1 SRPI, agency | `srpi-v2-2026.09` | `jackknife_pairs_10` | 9 |

Other versions: generators `mpc-bench-generators/2.0.0` (v2 systems; the v1
generator version string is unchanged), adversaries
`mpc-bench-adversarial/2.0.0`, forward layer `mpc-bench-forward/2.0.0`,
manipulation checks `mpc-bench-manipulation/1.0.0` (switch checks, frozen in
v1) and `mpc-bench-manipulation/1.1.0` (realisation checks of the new
systems), runner `mpc-bench-runner/2.0.0`, protocol schema
`impact-mpc-protocol/3`, record schema `mpc-bench-result/3`, registry schema
`impact-mpc-registry/3`, protocol builder `mpc-bench-protocol-builder/1.0.0`,
evaluator `mpc-bench-hypotheses-v2/1.0.0` (report
`mpc-bench-hypotheses-evaluator/2.0.0`). Run settings recorded in every
record: IIM macro grain cap 4 (`iim_max_macro_nodes`), PDI k-means seed 0
(`pdi_kmeans_seed`). Every v2 estimator lives in a new module and is
dispatched by the protocol's estimator version; the v1 estimator bodies are not
edited.

### 3.2 Shared infrastructure

**Recording device.** A v2 system carries a table of what an experimenter could
log: the task events (unchanged, so v1 estimators see exactly the v1 table),
one `context_cue` event per context switch with its new label, and the
slow-rhythm phase as sine and cosine channels. Exporting it does not touch the
time series or the events (every v1 estimator returns bit-identical output with
and without it). Estimators never read the oracle.

**Declarations** (hash-covered protocol field `shared_inputs_declaration`):

| Declaration | Content | Shared inputs |
|---|---|---|
| R (complete) | exogenous task events + `context_cue` + slow phase | complete |
| H (task only; the v1 information set) | exogenous task events | partial |
| P (held out) | R without the slow phase | partial |
| Q10, Q25 (held out) | R with each `context_cue` label replaced with probability q = 0.1 or 0.25 by another label (stream `SeedSequence([seed, 41])`) | partial |
| J (held out) | R with each `context_cue` onset shifted by U(-0.25, 0.25) s (stream `SeedSequence([seed, 42])`) | partial |
| none | no inputs (null-calibration generator, Hopf model, paper-2 default) | none |

Only inputs set independently of the system's state enter the basis (bench:
`goal_cue`, `stimulus`, `other_caused`, `context_cue`, slow phase); `response`,
`feedback`, `action` and `self_caused` are excluded; an all-events sensitivity
is reported (D13).

**One input basis for NAS and IIM.** For every declared channel `u`:
`U(t) = [u(t - l), u~_tau(t - l)]` for `l` in `{0} U L` and `tau` in
`{tau_c, 3 tau_c, 10 tau_c}`, where `u~_tau` is the causal first-order low-pass
and `L` the estimator's lag set. Every conditioned null shifts first and then
projects (the series is circularly shifted relative to the others and to `U`,
then all conditioning regressions are fitted); the other order is computed on
the same data in HCv2-2 and reported.

**Coupling time scale and lags (CD-1).** `tau_c = 0.1 s` on every bench
substrate (family A rate units, family C amplitude rate, Hopf model), read from
the generator constants. Lag sets: 20 Hz (families A and C1, null-calibration
generator) NAS `{1, 2}`, IIM `{1, 2}`; 250 Hz (Hopf source view, human EEG) NAS
`{1, 3, 9, 25}`, IIM 1-25; 500 Hz NAS `{1, 4, 14, 50}`. BOLD at TR 2 s is not
resolved, so NAS is `UNDEFINED(SAMPLING_UNRESOLVED)` there (IA-3). NAS at
`tau_c` 0.05 s and 0.2 s is reported (D7), never chosen after the fact.

**UNDEFINED vocabulary.** One module defines every reason; each maps to
UNDEFINED, never to ABSENT (IA-4): `INVALID_ANCHORS`, `INCONCLUSIVE`,
`NO_SAMPLING_SE`, `NULL_FAMILY_MISMATCH`, `MISSING_CHANNEL` (v1);
`ABSENT_NOT_REACHABLE` (`q_A se_c >= delta`), `NULL_MODEL_VIOLATED`
(`c + q_P se_c < -delta`), `SAMPLING_UNRESOLVED` (`dt > tau_c / 2`),
`OBSERVATION_MIXED_NOT_ADMITTED`, `INSUFFICIENT_OCCUPANCY`,
`MACRO_RANK_DEFICIENT`, `NOT_APPLICABLE_OBSERVATION_MODEL`,
`INSUFFICIENT_TIMEPOINTS`, `NO_NULL_CALIBRATION`, `INSUFFICIENT_UPDATES`,
`ESTIMATOR_ERROR:<type>`, `SE_NOT_CALIBRATED` (HCv2-4 reversion rule),
`ESTIMATOR_NOT_VALIDATED:not_admitted` and `:not_observable` (registry v3).
Flags, never statuses: `PRESENT_NOT_REACHABLE`, `ANCHOR_NOT_REPLICATED`,
`ABSENT_BY_CONCORDANCE`, `PRESENT_BY_CONCORDANCE`. The full table is in
[`docs/metrics_v2.md`](../metrics_v2.md).

**SE contract.** Every component record carries `se`, `se_df` and `se_method`;
the status rule reads `se_df` from the record, and the protocol lists the
admitted methods (section 3.1).

**Twins.** A generator option `replicate = r`: `r = 0` reproduces the v1 streams
bit for bit; for `r >= 1` the network and oscillator streams stay
`SeedSequence(seed)` and the task schedule, process noise and rest come from
`SeedSequence([seed, r])`. Twins of one network share its structure and differ
in their sessions; they calibrate every SE method (HCv2-4).

**Runner.** Result schema `mpc-bench-result/3`: per task one simulation and
several scorings (declarations, views, estimator forms). Each estimator runs in
its own try block; a component that raises is `UNDEFINED(ESTIMATOR_ERROR)` and
no other component is touched. Every record carries the family-protocol id and
hash, the code identity by git tree, timing and load. Linear algebra goes
through a fallback: where numpy's divide-and-conquer SVD does not converge on
an exactly rank-deficient input basis, the computation is repeated with SciPy's
`gesvd` (`gelsy` for least squares), so every result numpy computes is
unchanged (section 10.3).

### 3.3 The estimators

**NAS v3 (`nas-v3-2026.10`).** Blocks are the declared hub `H` and the declared
periphery blocks `j` (block means of z-scored nodes, primary; all nodes up to
eight principal components per block, secondary). `receive_j` and `return_j`
are conditional Geweke causalities from periphery to hub and back, given the
other blocks' pasts and the declared input basis; `R` and `B` are their means
over blocks. Null: circular shift of the hub block relative to the periphery
and `U` (shift, then project), K = 19, minimum shift `ceil(0.1 T)`. Each
direction has its own anchor and SE: `c_R`, `c_B`; PRESENT iff both directions
are PRESENT at `alpha` (intersection-union test); ABSENT iff at least one
direction passes the equivalence test at `alpha_A / 2 = 0.005`; reported value
`c_NAS = min(c_R, c_B)`. On the v1 geometry the code reproduces the frozen v1
statistic to 1e-12 (IA-8). Sensor or source-estimate observations without an
admitting registry entry are `UNDEFINED(OBSERVATION_MIXED_NOT_ADMITTED)`.
*Construct:* bidirectional directed linear dependence between a declared hub
block and each declared periphery block beyond a declared exogenous-input
basis. It is read as workspace receive-and-return only under a complete
declaration, resolved sampling and an unmixed or admitted observation;
otherwise it is hub-periphery dependence. Under a partial declaration PRESENT
certifies dependence only, while ABSENT keeps its meaning. NAS does not tell a
loop from a feed-forward relay, does not see zero-mean pattern broadcast, does
not certify which block is the workspace and does not measure nonlinear or
interventional effects.

**IIM v5 (`iim-v5-2026.10`).** Rank p-value `p_ind = (1 + #{m*_b >= m}) /
(K + 1)` against the independence null (K = 39 on the bench, 19 in family-B
cells); PRESENT iff `c - q_P se_c > z` and `p_ind <= 0.05` (a failed rank gate
is `UNDEFINED(INCONCLUSIVE)`, never ABSENT). Conditioning on declared recorded
drivers: continuous or multi-level drivers by residualisation on the common
basis with the shift-then-project null; binary drivers with at most four joint
states by stratification with a stratified pair-rotation null. Occupancy gate:
all `2^n` macro states visited and the rarest visited row with at least
`N_min = 25` transition pairs, else `UNDEFINED(INSUFFICIENT_OCCUPANCY)`.
Directional cuts (14 ordered bipartitions) are primary; the bidirectional form
is a reported secondary with its own anchor. SE: circular block bootstrap
(blocks of 10 % of the run, B = 50), `se_df = 9` (CD-3). Sensor level: disjoint
electrode clusters (the `min(8, floor(n_sensors / 8))` electrodes nearest each
quadrant centroid), symmetric (ZCA) orthogonalisation before median
binarisation; a rank-deficient grain is `MACRO_RANK_DEFICIENT`. The family-B
construct-scale anchor is ring coupling 0.45 (a second anchor, all-to-all 0.4,
is reported). Every run computes IIM on at most 4 macro nodes, the declared
grain of every Tier-A substrate. *Construct:* observational transition-level
integration of a declared macro process, conditional on the declared recorded
drivers, in excess of an independence null; identified as loop integration
only under causal sufficiency relative to the recorded variables and the other
stated assumptions. Under an unrecorded common driver it measures observational
dependence (HCv2-13(b) and HCv2-14(e) test this limit). It is not IIT's Phi.

**RAM-PE v3 (`ram-v3-2026.10`).** Facets G (goal alignment), F (feedback
magnitude) and speed are declared not applicable (no referent in the
generator). The node responses in a 0.4-s window after each stimulus, the
trial-to-trial change of that pattern and the choice-signed prediction error
of a frozen Rescorla-Wagner fit enter a ridge readout (5 contiguous folds,
purged, inner cross-validation of the penalty); the statistic is the Pearson
correlation of the out-of-fold predictions with the signed prediction error.
Null: every cyclic shift of the targets; SE: the SD of that null
(`se_df = n_null - 1`). Fewer than 30 updates or two options missing:
`UNDEFINED(INSUFFICIENT_UPDATES)`. At 79-159 updates ABSENT is unreachable, so
RAM-PE contributes no exclusion in v2. `scrambled_feedback` has its plasticity
on and is RAM-PE-mechanism-on. The RAM hypotheses run in a RAM-only arm at 160
trials (protocol `A-RAM160`); the joint bench stays at 80 trials. *Construct:*
the change of the evoked response pattern carries the outcome's prediction
error signed by the chosen option; the plasticity channel of RAM only.

**PDI v3 (`pdi-v3-2026.10`).** Window patterns (5 samples, z-scored), up to 10
sphered principal components, k-means++ with 8 starts on one half; held-out
recurrence and dwell criteria as v1; a Fisher-axis valley test with a
Ledoit-Wolf within-state covariance (valley ratio <= 0.4); the count `B` is the
mean of `log2 K_hat` over four split offsets; 19 circular-shift surrogates;
delete-a-group jackknife over 10 contiguous blocks. The protocol declares the
content bearer (`pdi_bearer = non_workspace`: every node except the declared
workspace); the full-bearer count is a reported upper bound. A component is
concordant if all 44 counts (11 data sets x 4 offsets) agree; it is then never
treated as exact: it is `UNDEFINED(NO_SAMPLING_SE)` unless its (substrate,
observation stage, view, bearer) cell is admitted (section 7.2), and in an
admitted cell it is ABSENT iff `|c| < delta` and PRESENT iff `c > z`. *Construct:*
the number of recurring, distinguishable multi-node states of a declared
content bearer; ignition and other global access states are not content states.

**SRPI.** The v1 instrument `srpi-v2-2026.09` (decodable self/other response
separability of stimulus-identical, phase-matched events) runs unchanged under
the v2 rule. Its v1-style ABSENTs from estimates below the null become
UNDEFINED by arithmetic (IA-9); its lesion ABSENT parts are
`NOT_TESTABLE_BY_DESIGN`; on C1 it is `NOT_APPLICABLE_OBSERVATION_MODEL`.

### 3.4 Status rule `tost-v2` and fixed cutoffs

For a component with construct value `c = (m - nu) / (rho - nu)` (estimate `m`,
null mean `nu`, reference anchor `rho`), sampling SE `se_c` and degrees of
freedom `df` (Welch-Satterthwaite combination of the estimate's `se_df` with
the null and reference parts, unchanged from v1):

- cutoffs **`z = 0.25`**, **`delta = 0.10`** for every principle; **`alpha =
  0.05`**; **`alpha_A = alpha / |N_decl| = 0.01`** in every protocol
  (`|N_decl| = 5`); `q_P = t(df, 1 - alpha)`, `q_A = t(df, 1 - alpha_A)`;
- **PRESENT** iff `c - q_P se_c > z` (IIM also needs `p_ind <= 0.05`; NAS needs
  both directions);
- **ABSENT** iff `c - q_A se_c > -delta` and `c + q_A se_c < delta` (two
  one-sided tests; NAS: either direction at `alpha_A / 2`);
- otherwise **UNDEFINED** with the first applicable reason:
  `NULL_MODEL_VIOLATED` iff `c + q_P se_c < -delta`; `ABSENT_NOT_REACHABLE` iff
  `q_A se_c >= delta`; else `INCONCLUSIVE`. Flag `PRESENT_NOT_REACHABLE` iff
  UNDEFINED and `q_P se_c >= 1 - z`;
- exact computations (deterministic known-TPM values only): PRESENT iff
  `c > z`, ABSENT iff `|c| < delta`. A data-derived `se_c = 0` is never exact;
  it is `UNDEFINED(NO_SAMPLING_SE)` unless the PDI concordance route is
  admitted for its cell;
- verdict: the unchanged strong-Kleene AND over `N_anch(f)`;
- protocol block `status_rule = {version: tost-v2, alpha: 0.05, alpha_absent:
  0.01, absent_test: tost, null_violation: true, present_reachability_flag:
  true}`, hash-covered. A protocol without the block is judged by the v1 rule
  byte-identically (schema `impact-mpc-protocol/2`).

Implied precision: `s_A = delta / q_A` is 0.035 at df 9, 0.039 at df 19 and
0.043 at df 199; NAS per direction 0.031 at df 9.

### 3.5 Error-control argument and what the rule does not control

Assume (A-SE) that `(c_p - theta_p) / se_c` is approximately `t_df` with the
declared df (tested by HCv2-3 at the null and by HCv2-4 within twin networks,
including the `q_A` tail).

- **E1, component false PRESENT.** If `theta_p <= z`,
  `P(PRESENT) <= alpha`; for NAS through the intersection-union test; the IIM
  rank gate only removes PRESENT decisions.
- **E2, component false ABSENT.** If `|theta_p| >= delta`,
  `P(ABSENT) <= alpha_A`; for NAS `2 x alpha_A / 2 = alpha_A`.
- **E3, verdict false MPC_CONSISTENT.** If some `theta_p <= z` with `p` in
  `N_anch`, `P(MPC_CONSISTENT) <= alpha`.
- **E4, verdict false EXCLUDED.** If every `theta_p >= delta`,
  `P(EXCLUDED) <= |N_anch| x alpha / |N_decl| <= alpha`, under any dependence
  among components and for any anchored subset.

Conditions: the anchor is frozen from development seeds disjoint from the
evaluation seeds; SE validity (A-SE); each principle feeds exactly one ABSENT
decision into the verdict; UNDEFINED is never counted as ABSENT; a valid null
for `nu`. For the PDI concordance route E2 holds only empirically over the
admission battery (section 7.2), not as a sampling-theory guarantee.

The rule does not control estimator bias (a precise estimate near 0 with the
mechanism on; HCv2-5 tests for it), anchor misattribution (addressed by the
specificity gate and the anchor-attributability report D2), construct
misidentification under hidden inputs (addressed by the construct statements)
or the abstention trade-off (coverage and selective risk are reported, D11).

**Reversion rule (HCv2-4).** An SE method FALSIFIED on the anti-conservative
side re-classifies every ABSENT of that estimator in every verdict-level
analysis as `UNDEFINED(SE_NOT_CALIBRATED)`; the hypotheses that used those
ABSENTs are reported both ways. A conservative failure changes nothing.

### 3.6 Anchors and necessity sets

Reference blocks: development seeds 900-939 for every family protocol and view
(PC_nominal and, on the same seeds, each principle's own-lesion witness). No
anchor is computed on confirmatory data. An anchor is **valid** if at least
36 of 40 excesses are finite and the one-sided 95 % Student-t lower bound of
the mean excess is above 0; it is **specific** if the mean paired contrast
(PC minus own lesion) is at least 0.5 x the anchor with a lower bound above 0
(the 0.5 is the v1 HC3 contrast criterion, not tuned). NAS has one anchor per
direction. `N_anch(f)` = the principles valid and specific on protocol `f`.
Fallbacks: F0 all five anchored (verdict hypotheses on all five); F1 two to
four (verdict parts that need an unanchored principle are NOT_EVALUABLE); F2
one (verdict parts NOT_EVALUABLE); F3 none (raw-excess descriptives). A
principle removed by the gate is still evaluated at component level on its own
anchor, scored in HCv2-6 with its development status as the prediction, and
verdict-level results are also reported on `N_decl` (descriptive). Forward
views have validity-only anchors (FM0) and no necessity set; they are used for
admission (section 4.5).

Frozen anchor outcomes (generated protocols; anchor value = mean development
excess of PC_nominal; for NAS receive / return; DEVELOPMENT - NOT A RESULT):

| Protocol | Fallback | `N_anch` | RAM | PDI | NAS (receive / return) | IIM | SRPI |
|---|---|---|---|---|---|---|---|
| `A-H` | F1 | {RAM, PDI, NAS, SRPI} | valid_specific (0.388) | valid_specific (2.429) | valid_specific (0.00705) / valid_specific (0.0146) | valid_nonspecific (0.00337, 36/40 finite) | valid_specific (0.171) |
| `A-H+iim_bidirectional` | F3 | {} | invalid (-, 0/40 finite) | invalid (-, 0/40 finite) | invalid (-) / invalid (-) | valid_nonspecific (0.0102, 36/40 finite) | invalid (-, 0/40 finite) |
| `A-H+nas_secondary` | F3 | {} | invalid (-, 0/40 finite) | invalid (-, 0/40 finite) | valid_nonspecific (0.017) / valid_nonspecific (0.0259) | invalid (-, 0/40 finite) | invalid (-, 0/40 finite) |
| `A-J` | F0 | {RAM, PDI, NAS, IIM, SRPI} | valid_specific (0.388) | valid_specific (2.429) | valid_specific (0.00241) / valid_specific (0.0222) | valid_specific (0.00679) | valid_specific (0.171) |
| `A-P` | F0 | {RAM, PDI, NAS, IIM, SRPI} | valid_specific (0.388) | valid_specific (2.429) | valid_specific (0.00241) / valid_specific (0.0222) | valid_specific (0.00679) | valid_specific (0.171) |
| `A-Q10` | F0 | {RAM, PDI, NAS, IIM, SRPI} | valid_specific (0.388) | valid_specific (2.429) | valid_specific (0.00241) / valid_specific (0.0222) | valid_specific (0.00679) | valid_specific (0.171) |
| `A-Q25` | F0 | {RAM, PDI, NAS, IIM, SRPI} | valid_specific (0.388) | valid_specific (2.429) | valid_specific (0.00241) / valid_specific (0.0222) | valid_specific (0.00679) | valid_specific (0.171) |
| `A-R` | F0 | {RAM, PDI, NAS, IIM, SRPI} | valid_specific (0.388) | valid_specific (2.429) | valid_specific (0.00241) / valid_specific (0.0222) | valid_specific (0.00679) | valid_specific (0.171) |
| `A-R+iim_bidirectional` | F2 | {IIM} | invalid (-, 0/40 finite) | invalid (-, 0/40 finite) | invalid (-) / invalid (-) | valid_specific (0.0121) | invalid (-, 0/40 finite) |
| `A-R+nas_secondary` | F2 | {NAS} | invalid (-, 0/40 finite) | invalid (-, 0/40 finite) | valid_specific (0.0034) / valid_specific (0.0243) | invalid (-, 0/40 finite) | invalid (-, 0/40 finite) |
| `A-R+nas_tau_0.05` | F2 | {NAS} | invalid (-, 0/40 finite) | invalid (-, 0/40 finite) | valid_specific (0.000228) / valid_specific (0.0217) | invalid (-, 0/40 finite) | invalid (-, 0/40 finite) |
| `A-R+nas_tau_0.2` | F2 | {NAS} | invalid (-, 0/40 finite) | invalid (-, 0/40 finite) | valid_specific (0.0023) / valid_specific (0.0226) | invalid (-, 0/40 finite) | invalid (-, 0/40 finite) |
| `A-R+pdi_misdeclared_access` | F0 | {RAM, PDI, NAS, IIM, SRPI} | valid_specific (0.388) | valid_specific (2.429) | valid_specific (0.00241) / valid_specific (0.0222) | valid_specific (0.00679) | valid_specific (0.171) |
| `A-RAM160` | F2 | {RAM} | valid_specific (0.545) | invalid (-, 0/40 finite) | invalid (-) / invalid (-) | invalid (-, 0/40 finite) | invalid (-, 0/40 finite) |
| `A-none` | F0 | {RAM, PDI, NAS, IIM, SRPI} | valid_specific (0.388) | valid_specific (2.429) | valid_specific (0.00241) / valid_specific (0.0222) | valid_specific (0.00679) | valid_specific (0.171) |
| `C1-H` | F1 | {NAS, IIM} | invalid (-, 0/40 finite) | invalid (-, 0/40 finite) | valid_specific (0.0113) / valid_specific (0.0092) | valid_specific (0.00113) | invalid (-, 0/40 finite) |
| `C1-H+iim_bidirectional` | F2 | {IIM} | invalid (-, 0/40 finite) | invalid (-, 0/40 finite) | invalid (-) / invalid (-) | valid_specific (0.00181) | invalid (-, 0/40 finite) |
| `C1-H+nas_secondary` | F2 | {NAS} | invalid (-, 0/40 finite) | invalid (-, 0/40 finite) | valid_specific (0.0122) / valid_specific (0.00997) | invalid (-, 0/40 finite) | invalid (-, 0/40 finite) |
| `C1-R` | F1 | {NAS, IIM} | invalid (-, 0/40 finite) | invalid (-, 0/40 finite) | valid_specific (0.0113) / valid_specific (0.00922) | valid_specific (0.00115) | invalid (-, 0/40 finite) |
| `C1-R+iim_bidirectional` | F2 | {IIM} | invalid (-, 0/40 finite) | invalid (-, 0/40 finite) | invalid (-) / invalid (-) | valid_specific (0.00185) | invalid (-, 0/40 finite) |
| `C1-R+nas_secondary` | F2 | {NAS} | invalid (-, 0/40 finite) | invalid (-, 0/40 finite) | valid_specific (0.0122) / valid_specific (0.01) | invalid (-, 0/40 finite) | invalid (-, 0/40 finite) |
| `C1-R+nas_tau_0.05` | F2 | {NAS} | invalid (-, 0/40 finite) | invalid (-, 0/40 finite) | valid_specific (0.002) / valid_specific (0.00131) | invalid (-, 0/40 finite) | invalid (-, 0/40 finite) |
| `C1-R+nas_tau_0.2` | F2 | {NAS} | invalid (-, 0/40 finite) | invalid (-, 0/40 finite) | valid_specific (0.00596) / valid_specific (0.00499) | invalid (-, 0/40 finite) | invalid (-, 0/40 finite) |
| `fwdA-eeg64` | F3 | {} | invalid (-, 0/40 finite) | valid_nonspecific (2.423) | invalid (-) / invalid (-) | valid_nonspecific (0.000297) | invalid (-, 0/40 finite) |
| `fwdA-eeglow` | F3 | {} | invalid (-, 0/40 finite) | valid_nonspecific (1.987) | invalid (-) / invalid (-) | valid_nonspecific (0.000113) | invalid (-, 0/40 finite) |
| `fwdA-source` | F3 | {} | invalid (-, 0/40 finite) | valid_nonspecific (2.429) | invalid (-) / invalid (-) | valid_nonspecific (0.00679) | invalid (-, 0/40 finite) |
| `fwdA_bold-bold` | F3 | {} | invalid (-, 0/40 finite) | valid_nonspecific (1.303) | invalid (-) / invalid (-) | invalid (-0.000254, 32/40 finite) | invalid (-, 0/40 finite) |
| `fwdA_bold-source` | F3 | {} | invalid (-, 0/40 finite) | valid_nonspecific (2.129) | invalid (-) / invalid (-) | valid_nonspecific (0.00689) | invalid (-, 0/40 finite) |
| `hopf-bold` | F3 | {} | invalid (-, 0/40 finite) | invalid (-, 0/40 finite) | invalid (-) / invalid (-) | invalid (-, 0/40 finite) | invalid (-, 0/40 finite) |
| `hopf-eeg64` | F3 | {} | invalid (-, 0/40 finite) | invalid (-, 0/40 finite) | valid_nonspecific (0.0252) / valid_nonspecific (0.000645) | invalid (1.09e-05) | invalid (-, 0/40 finite) |
| `hopf-eeg64+iim_v1_quadrants` | F3 | {} | invalid (-, 0/40 finite) | invalid (-, 0/40 finite) | invalid (-) / invalid (-) | invalid (-, 0/40 finite) | invalid (-, 0/40 finite) |
| `hopf-eeg64_noref` | F3 | {} | invalid (-, 0/40 finite) | invalid (-, 0/40 finite) | invalid (-) / invalid (-) | valid_nonspecific (2.67e-05) | invalid (-, 0/40 finite) |
| `hopf-eeg64_noref+iim_v1_quadrants` | F3 | {} | invalid (-, 0/40 finite) | invalid (-, 0/40 finite) | invalid (-) / invalid (-) | valid_nonspecific (0.000898) | invalid (-, 0/40 finite) |
| `hopf-eeglow` | F3 | {} | invalid (-, 0/40 finite) | invalid (-, 0/40 finite) | valid_nonspecific (0.0126) / valid_nonspecific (0.00101) | valid_nonspecific (5.16e-05) | invalid (-, 0/40 finite) |
| `hopf-mne_template` | F3 | {} | invalid (-, 0/40 finite) | invalid (-, 0/40 finite) | valid_nonspecific (0.00962) / valid_nonspecific (0.014) | valid_nonspecific (3.88e-05) | invalid (-, 0/40 finite) |
| `hopf-source` | F3 | {} | invalid (-, 0/40 finite) | invalid (-, 0/40 finite) | valid_nonspecific (0.00257) / valid_nonspecific (0.000588) | valid_nonspecific (0.000203) | invalid (-, 0/40 finite) |

The family-B protocols carry the exact anchors of the known transition
matrices (CD-1) and no reference block. Two outcomes differ from the design's
pre-development prediction, and in both the anchor rule decided: NAS v3 is
specific under H, although it was predicted non-specific, and IIM is anchored
under C1-H (HO-4 permits the C1 reference-block anchors). `hopf-bold` and
`hopf-eeg64+iim_v1_quadrants` have no valid anchor on the complete reference
block, so every component under them is UNDEFINED and their verdicts are
UNDETERMINED by construction (section 10.3).

### 3.7 Testability gating

For every ABSENT-type part (target ABSENT on an own-lesion or construct-null
witness) the development rate `pi0(f, p, w)` is the empirical ABSENT rate of
target `p` on witness `w` under the frozen rule, from the dry run and the
reference-block witnesses (at least 40 runs); for every "PRESENT in >= 80 %"
part `pi1` is the empirical PRESENT rate on PC_nominal. A gated cell is
decisive only if `pi >= 0.9`; otherwise it is `NOT_TESTABLE_BY_DESIGN` before
the freeze and its replacement claim decides ("target not PRESENT in >= 80 %",
or a paired contrast with the positive control). The gate reads only precision
on own-lesion and reference systems, never a confirmatory value. The table is
part of each family protocol's hash-covered `precision` block and is rendered
in [`v2/testability_table.md`](v2/testability_table.md). Gating rows of the
decisive parts (development, R):

- A-R IIM PC_nominal: PRESENT 0.906 (58/64), decisive by one row (HCv2-14(d)).
- A-R IIM W_IIM_feedforward: ABSENT 0.23, NOT_TESTABLE_BY_DESIGN (HCv2-14(c)).
- A-R and A-H PDI W_PDI_single_attractor: ABSENT 0.79 (41/52, concordant
  ABSENTs), NOT_TESTABLE_BY_DESIGN (HCv2-22(ii)).
- A-R NAS W_NAS_no_workspace: ABSENT 1.0, decisive (HCv2-22(ii));
  A-R NAS W_NAS_broadcast_only: 0.04, NOT_TESTABLE_BY_DESIGN.
- C1-R and C1-H IIM PC_nominal: PRESENT 0.40 and 0.25, NOT_TESTABLE_BY_DESIGN
  (HCv2-14(d) on C1 replaced by (a)); C1-R and C1-H IIM W_IIM_feedforward:
  ABSENT 0 (0/40), NOT_TESTABLE_BY_DESIGN (HCv2-14(c) on C1). These four rows
  are read from the C1 IIM reference block (prior contact HO-4, section 10.2).
- C1-R and C1-H NAS W_NAS_no_workspace: ABSENT 0.92 each, decisive
  (HCv2-22(ii)).
- B IIM feedforward star: ABSENT 1.0, decisive (HCv2-11(c)).

### 3.8 Scale type and the absence margin

Conditional Geweke causality (nats) and Delta_Psi (bits) are quadratic in a
weak effect amplitude, so `delta = 0.10` on `c` corresponds to an amplitude of
about `sqrt(0.10) = 0.32` of the reference; RAM-PE (a correlation), the v1 SRPI
composite and PDI (bits of a count) are not quadratic. The fixed core keeps
`c` and `delta`; every ABSENT of a quadratic statistic is re-judged on the
amplitude scale `sign(c) sqrt(|c|)` as a preregistered sensitivity analysis
(D6), and a sweep dose counts as mechanism-on only where the development
median `c` at that dose is at least `2 delta` (CD-8).

## 4. Systems and designs

### 4.1 Substrates and family protocols

| Substrate | Generator | Principles scored | Declarations | Family protocols |
|---|---|---|---|---|
| Family A (v1 routing, recording device added) | `simulate_family_a`, v1 dynamics; 80 trials, 30 reafference pairs | all five (SRPI v1) | R and H from one simulation; held-out P, Q10, Q25, J on a subset | `A-R`, `A-H` (and `A-P`, `A-Q10`, `A-Q25`, `A-J`, `A-none`, forms) |
| Family A, RAM-only arm | the same generator, 160 trials | RAM-PE | R (recorded only) | `A-RAM160` |
| Family C1 (v1 family C) | `simulate_family_c`, unchanged | NAS, IIM; RAM, PDI, SRPI `NOT_APPLICABLE_OBSERVATION_MODEL` | R and H | `C1-R`, `C1-H` |
| Family B (4 binary units, exact TPMs) | family-B networks and sampled trajectories | IIM (both cut modes) | recorded, hidden or label-error drivers | `B` and its cell protocols (exact anchors) |
| Null-calibration generator | the v1 null generators, imported read-only; hub = first quarter of the nodes, three equal periphery blocks | all five | none | judged on the A-R anchors (`A-none`) |
| Hopf model (76 regions, 250 Hz, 60 s) | `simulate_whole_brain`, `G_nom = 1.142857` | NAS, IIM | none | one protocol per view: `hopf-source`, `hopf-eeg64`, `hopf-eeg64_noref`, `hopf-eeglow`, `hopf-mne_template`, `hopf-bold`, and the v1 quadrant comparators |
| Forward-modelled family A | family-A agents through the v2 EEG and BOLD forward models at the held-out regime | PDI (content bearer at source; full-bearer upper bound on sensors), IIM | R | `fwdA-source`, `fwdA-eeg64`, `fwdA-eeglow`; slow-context BOLD arm `fwdA_bold-source`, `fwdA_bold-bold` |

Estimator forms with their own protocols: `+nas_secondary` (all-node blocks),
`+iim_bidirectional`, `+nas_tau_0.05` and `+nas_tau_0.2` (D7),
`+pdi_misdeclared_access` (module S declared as the workspace; HCv2-20(d)),
`+iim_v1_quadrants` (the v1 quadrant montage, HCv2-15(b)).

### 4.2 Confirmatory designs

Task counts are those of the generated run plan (`scripts/run_bench_v2.py plan
--split confirmatory` and `scripts/v2/iim_validation_v2.py --list --split
confirmatory`; IA-7 checks the records against them).

| Design | Systems | Seeds | Tasks |
|---|---|---|---|
| `A_witnesses` | PC_nominal, PC_half (all five switches at 0.5 x nominal), the six single deficits (W_RAM_no_plasticity, W_PDI_single_attractor, W_NAS_no_workspace, W_NAS_broadcast_only, W_IIM_feedforward, W_SRPI_no_efference), W_PDI_no_multistability, W_NAS_common_input_control, N_modules_disconnected, N_independent_noise, N_ar1, O_hypersynchronous, O_inert, PW_patchwork (principle-bearer mode); scored under R and H, held-out declarations P, Q10, Q25, J re-scored on a subset | 20000-20019; PC_nominal and the six single deficits 20000-20044; N_independent_noise and N_ar1 20000-20045 | 547 |
| `A_sweeps` | g_b (10 levels), c_int (10 levels), K ({2, 3, 4, 6, 8, 12}) | g_b and K 20000-20009; c_int 20000-20064 | 810 |
| `A_factorial` | 2^5 cells over (eta, K, g_b, c_int, e) | 20000-20009 | 320 |
| `A_adversaries` | adversarial_common_driver, reflex_arc, random_label_self_other, scrambled_feedback, ADV_NAS_staggered_driver (hierarchical, uniform, reversed) | 20000-20019 | 140 |
| `A_heldout` | ADV_NAS_staggered_tau10, ADV_NAS_staggered_sat | 20000-20019 | 40 |
| `C1_witnesses` | 13 family-C1 witness kinds (N_uncoupled replaces the duplicated family-A nulls; N_modules_disconnected added) | 20000-20019; PC_nominal 20000-20044; W_NAS_no_workspace, W_NAS_broadcast_only, W_IIM_feedforward 20000-20128 | 612 |
| `C1_sweeps` | g_b (10 levels), c_int (10 levels) | g_b 20000-20009; c_int 20000-20064 | 750 |
| `C1_factorial` | 2^5 cells | 20000-20009 | 320 |
| `RAM160` | PC_nominal, W_RAM_no_plasticity, eta in {0, 0.1, 0.2, 0.3, 0.6}, reflex_arc, scrambled_feedback, K = 1, g_b = 0 (160 trials) | 20000-20039 | 440 |
| `A_twins` | PC_nominal, PC_half, W_NAS_no_workspace, W_IIM_feedforward, W_PDI_single_attractor; r = 1..6, plus an r = 0 check of PC_nominal | networks 20000-20009 | 310 |
| `C1_twins` | PC_nominal, W_NAS_no_workspace, W_IIM_feedforward; r = 1..6 | networks 20000-20009 | 180 |
| `RAM160_twins` | eta in {0, 0.1, 0.3}; r = 1..7 | networks 20000-20009 | 210 |
| `A_anchors` | PC_nominal and the six single deficits (replication block) | 20900-20939 | 280 |
| `C1_anchors` | PC_nominal and three deficits | 20900-20919 | 80 |
| `RAM160_anchors` | PC_nominal, W_RAM_no_plasticity | 20900-20919 | 40 |
| `forward_anchor_replication` | the reference conditions of the forward arms: Hopf `G_nom` (40 seeds), forward family A EEG and BOLD PC_nominal (20 each) | Hopf 20900-20939; family A 20900-20919 | 80 |
| `whole_brain` (Hopf arm) | G sweep (8 levels 0-4) x 20; hub lesion and size-matched random lesion at `G_nom` x 20; G = 0 extended to 61 seeds, `G_nom` to 150 | 20000-20019, 20000-20060, 20000-20149 | 371 |
| `forward_family_a` (EEG) | PC_nominal 150, N_ar1 61, W_PDI_no_multistability 61, K in {1, 2, 3} x 20 | 20000-20149 | 332 |
| `forward_family_a_bold` (slow contexts: dwell 30-60 s, TR 2 s, window 5 TR) | PC_nominal up to 150 (curtailed), W_PDI_no_multistability 61, K in {1, 2, 3} x 20 | 20000-20149 | at most 271 |
| `null_calibration` | ar1, pink, surrogate_iid, surrogate_linear x T {1200, 2400} x {8, 16} nodes, 50 replicates per cell | seed base 20000 | 800 |
| Family B | the cells of HCv2-2 (1800 tasks), HCv2-11 (480), HCv2-12 (1680) and HCv2-13 (300) | 20000-20499, blocks per hypothesis (section 5) | 4260 |
| Prerequisite M | oracle switch and realisation checks, families A and C1 | 20000-20039 | 40 seeds |

In total 6933 runner tasks, 4260 family-B tasks and the oracle checks on 40
seeds. Removed from the v1 set: `adversarial_parity_grid` (inert in v1) and
`adversarial_hypersynchrony` (duplicates O_hypersynchronous); the whole-brain
interhemispheric and long-range lesions (descriptive in v1) are not run.

### 4.3 Hard cases from v1 kept as explicit conditions

| v1 failure | v2 condition | Tested in |
|---|---|---|
| NAS reads the shared context drive; the workspace receives the common input | family A scored under H; W_NAS_no_workspace under H and R | HCv2-7, HCv2-8(c), HCv2-24(b) |
| both transfer directions inflated, min-z gate defeated | N_modules_disconnected; staggered driver (no edges) under H | HCv2-8(a, b) |
| hidden common driver read as dependence or integration | family-B hidden driver undeclared; adversarial_common_driver; staggered driver | HCv2-13(b), HCv2-8(b) |
| IIM reads feed-forward propagation and common drive | W_IIM_feedforward and O_inert under H and R | HCv2-14 |
| volume conduction at EEG level | Hopf G = 0 at every EEG view; v1 quadrant montage | HCv2-10, HCv2-15(a, b) |
| source estimates keep the mixing floor | MNE-template view | HCv2-10 |
| BOLD sampling does not resolve coupling | every BOLD view | IA-3, HCv2-15(d), HCv2-21 |
| mechanisms hidden in the family-C carrier | C1 with NOT_APPLICABLE declarations | IA-4 |
| PDI masked by the ignition common mode | the v1 ignition operating point; g_b sweep across the v1 masking window 0.67-1.56 | HCv2-19 |
| ignition counted as a repertoire state | W_PDI_single_attractor with ignition on | HCv2-20 |
| wrong-sign IIM ABSENT on hypersynchrony | O_hypersynchronous | HCv2-12(d), IA-4 |
| rarely visited TPM rows, strong coupling, non-monotone target | family B all-to-all 0.6 at T = 1000; ring 0.9 and 1.5 | HCv2-12(b, c) |
| SRPI lesion residual; SRPI ABSENTs below the null | W_SRPI_no_efference; v1 SRPI under the v2 rule | HCv2-22, IA-9, HCv2-3 |
| RAM factors without referents; scrambled feedback mislabelled | RAM-PE; scrambled_feedback relabelled; reflex_arc | HCv2-16, HCv2-17 |
| absence margin below precision | `ABSENT_NOT_REACHABLE` and testability gating | section 3.7, HCv2-22 |

### 4.4 Manipulation and realisation checks (prerequisite M)

On families A and C1, seeds 20000-20039: the frozen
`mpc-bench-manipulation/1.0.0` switch checks (eta, K, g_b, ff_only, c_int, e)
and the `mpc-bench-manipulation/1.1.0` realisation checks of the new systems:
N_modules_disconnected has no directed hub-periphery path; the staggered driver
reaches every module (per-module oracle context information above the v1
floor 0.05) and realises its stated time constants; W_PDI_no_multistability has
ignition occupancy below 0.05; the slow-context preset of the forward BOLD arm
gives complete context runs of at least 30 s. A check is usable in a family if
it passes in at least 36 of 40 seeds (HCv2-0). Parts that need an unusable
switch or system drop the affected rows and, if nothing is left, are
NOT_EVALUABLE with the reason ORACLE; a paired row needs the checks of both
systems; the slow-context check is required only by the rows of the forward
BOLD arm. PC_half between the off and nominal medians and the twin hashes are
reported (HCv2-0(c, d)), never gates.

### 4.5 Forward-model validation layer and the predicted admission table

Registry entries for human EEG or fMRI come only from forward-model arms with
known sources (registry schema `impact-mpc-registry/3`, built by
`scripts/v2/build_registry_v3.py` from the confirmatory forward arms and
`protocols/v2/generated/forward_anchors.json`).

**Held-out admission regime (HO-3).** Lead-field seed 20261001, conduction
width 0.6, 64 electrodes and the paper-2 low-density montage of `n_low = 32`
electrodes (CD-12), average reference (IIM also without reference), band
1-40 Hz, 250 Hz, 60-s windows, no declared inputs; the MNE-template view uses a
template lead field (development seed, electrode positions jittered
N(0, 0.05), width 0.4) with `lambda = tr(L L') / (n SNR^2)`, SNR 3, written in
numpy; BOLD uses the v1 BOLD forward model (TR 2 s, canonical HRF, AR(1) 0.6,
noise 0.5). Registry entries match these regime keys exactly and do not
extrapolate.

**Criteria.** FM0 (anchor): valid on the view's reference block. FMa
(specificity): on every declared null class of the view, false PRESENT with a
one-sided Clopper-Pearson upper bound below 0.07 at level 0.05/m. FMb1 (dose):
Spearman `rho(c, dose) > 0` at one-sided p < 0.05. FMb2 (NAS only): hub lesion
minus size-matched random lesion, paired `Delta c < 0` in at least 80 % of
seeds. FMd (source concordance): the forward contrast between the dose extremes
has the sign of the source contrast and at least half its size in at least
80 % of seeds. FMabs (exclusion safety): false ABSENT among mechanism-on runs
with a Clopper-Pearson upper bound below 0.02 (at least 149 on-runs with 0
events). On-runs are evaluated in seed order and a view stops as soon as an
event makes the demonstration impossible; curtailment only ever stops towards
non-admission. **Outcome** per (principle, estimator version, substrate,
regime): `admitted_for_present` (FM0, FMa, FMb1, FMd and, for NAS, FMb2) and
`admitted_for_absent` (FM0 and FMabs), each yes, no, `vacuous` (`pi0 < 0.1`:
ABSENT admissible but practically unreachable) or `not_observable`. A status
direction that is not admitted becomes
`UNDEFINED(ESTIMATOR_NOT_VALIDATED:not_admitted)`.

**Predicted admission table** (transcribed in
`protocols/v2/held_out_predictions_v2.json`, committed before the release):

| Principle | Views | Pipeline | Predicted | Label | Tested in |
|---|---|---|---|---|---|
| NAS | source | | present yes; absent yes | R (development: null at G = 0; hub lesion 15 % vs random 67 % of unlesioned) | HCv2-10, HCv2-10(absent) |
| NAS | eeg64, eeglow, mne_template | | present no (FMa); absent vacuous | R (development: significant at G = 0 in 5/6-6/6 seeds for every sensor and source-estimate view) | HCv2-10, HCv2-10(absent) |
| NAS | bold | | UNDEFINED (SAMPLING_UNRESOLVED) on every task | C (IA-3) | IA-3 |
| IIM | eeg64, eeg64_noref, eeglow, mne_template | EEG v2 clusters (ZCA) | FMa yes; FMd no (orthogonalisation removes the near-zero-lag Hopf coupling), hence present no; absent yes or vacuous | FMa R; FMd HO | HCv2-15(a), HCv2-15(c), HCv2-15(e-FMd) |
| IIM | eeg64, eeg64_noref | v1 quadrant montage (iim_v1_quadrants) | average reference: MACRO_RANK_DEFICIENT; no reference: PRESENT at G = 0 in >= 80 % | R | HCv2-15(b) |
| IIM | bold | | UNDEFINED (INSUFFICIENT_OCCUPANCY) or not PRESENT at every G; absent vacuous or yes | C | HCv2-15(d-present), HCv2-15(d-absent) |
| PDI | eeg64, eeglow | full-bearer upper bound | present yes (declared non-specific); absent **vacuous** (originally: absent yes) | R (development: PC B 2.55, W_PDI_no_multistability 0/10) | HCv2-21 |
| PDI | bold | slow contexts | absent **vacuous** (originally: absent no, K = 3 read as one state) | R (development 6/10) | HCv2-21 |
| RAM, SRPI | any human modality | | not definable on the paper-2 data (0/2457); no admission in Tier A | - | - |

The two PDI rows were corrected before the release as a logical error (section
10.3).

**What the released anchors already fix.** FM0 reads the frozen forward anchors
(`protocols/v2/generated/forward_anchors.json`), which were computed after the
release at the held-out regime (section 9), so the FM0 outcome of every
admission entry is known before the freeze. FM0 passes for NAS on the Hopf
source, eeg64, eeglow and mne_template views; for IIM on the Hopf source,
eeg64_noref, eeglow and mne_template views and on every forward family-A EEG and
source view; and for PDI on every forward family-A view, BOLD included. FM0
fails for IIM on `hopf-eeg64` (64 electrodes, average reference: 40/40 finite,
mean excess 1.09e-05, one-sided lower bound -9.5e-06, not above 0), on
`hopf-bold` (0/40 finite) and on `fwdA_bold-bold` (32/40 finite), and for NAS on
`hopf-bold` (NAS is `UNDEFINED(SAMPLING_UNRESOLVED)` there by construction).
Admission in either direction needs FM0 (`yes` and `vacuous` alike), so two
predictions of the table cannot hold, and this is known before the freeze: IIM
on eeg64 at average reference ("absent yes or vacuous") will be "absent no", and
IIM on BOLD ("absent vacuous or yes") will be "absent no" unless every BOLD run
is undefined for an observation-model reason (then `not_observable`). The
predictions were not changed after the release. Every IIM component judged under
`hopf-eeg64` has no valid anchor and is UNDEFINED, so in that view IIM is
neither PRESENT nor ABSENT; the parts of HCv2-15 read those statuses as they
fall.

**Consequence for paper 2.** If the predictions hold: paper 2 has no admissible
exclusion on fMRI. On EEG, NAS is not admitted for PRESENT (FMa) and is only
`vacuous` for ABSENT; PDI is `vacuous` for ABSENT, and its PRESENT admission
holds at the forward family-A regime (20 Hz, unfiltered, complete input
declaration), which no paper-2 recording matches (section 7.5). The only
possible admission-dependent EEG exclusion is IIM ABSENT on a view that passes
FM0 (32 electrodes at average reference, 64 electrodes without reference, the
template inverse; not 64 electrodes at average reference) and FMabs at the
confirmatory run (HCv2-15(c)). Paper 2 must therefore plan for having no
admissible EEG exclusion at its registered regime (section 10.4).

### 4.6 Integrity audit (not counted as hypotheses)

`scripts/v2/integrity_audit.py` runs on the confirmatory records before the
evaluation. Any failure is listed, the affected records leave the hypotheses
(`excluded_task_ids`), and the audit outcome is reported beside the tally. A
failure that no exclusion of records repairs, or that shows that the run did not
use the frozen inputs, stops the evaluation instead (`blocking_failures` of the
audit report): every IA-1 failure, every IA-6 failure (a family-protocol hash or
declaration that differs from the frozen protocol), a reason of the vocabulary
that does not map to UNDEFINED (IA-4), an unreadable record line, a seed outside
the seed policy, a mixed or wrong split, a planned task without a record,
curtailment that does not match the plan (IA-7), and the identities of IA-8
(status-rule entry points, NAS v3 against v1). A design whose component error
rate exceeds the IA-2 bound is reported and excludes nothing: its errored
components stay `UNDEFINED(ESTIMATOR_ERROR:<type>)`. A task that still fails
after its one rerun (section 11) is a record-level failure: it is listed and
leaves the hypotheses. If a blocking failure occurs, no outcome is reported
until its cause is removed without reading any outcome (for example by
completing the missing tasks of the plan); the cause and the remedy are reported
as a deviation.

| Id | Check |
|---|---|
| IA-1 | v1 regression gate: v1 reruns and stratified stored records bit for bit; stored v1 confirmatory records re-judged under the v1 protocols with 0 differences; frozen v1 protocol hashes unchanged; a v1 BOLD task still ends in the v1 band error |
| IA-2 | no task loses a component because another raised; component error rate <= 1 % per design; every error is `UNDEFINED(ESTIMATOR_ERROR:<type>)` |
| IA-3 | NAS is `UNDEFINED(SAMPLING_UNRESOLVED)` on every BOLD task and never ABSENT; NAS is defined on every family-A, C1 and Hopf-source task |
| IA-4 | every reason maps to UNDEFINED; no ABSENT on degenerate TPMs, on C1 components declared NOT_APPLICABLE (none PRESENT either), under `OBSERVATION_MIXED_NOT_ADMITTED` or an unadmitted registry direction |
| IA-5 | no system is identical across families (`sha256(ts)`, `sha256(raw_ts)`) |
| IA-6 | each record's family-protocol hash equals the frozen protocol; the evaluator refuses mismatches |
| IA-7 | task counts equal the generated plan; seed policy (development <= 999, confirmatory >= 20000, 10000-19999 unused) |
| IA-8 | NAS v3 equals the v1 NAS on the v1 geometry to <= 1e-12; the stored statuses equal the status rule; the three status-rule entry points agree |
| IA-9 | v1 SRPI under the v2 rule: no SRPI ABSENT with a point estimate outside (-delta, delta); every v1-style sub-null outcome UNDEFINED; the number of SRPI ABSENTs is reported |
| IA-10 | twins: identical structural hash, different schedule hash; `r = 0` equals the witness run bit for bit |

### 4.7 Descriptive analyses (no decision)

D1 flip-versus-margin curve of determinate verdicts; D2 anchor attributability
per protocol; D3 potency classes of the adversaries against the v2 estimators
under H (CD-9); D4 the estimator-visible manipulation check; D5 statuses under
`alpha / |N_anch(f)|`; D6 amplitude-scale re-judging of quadratic statistics;
D7 NAS at `tau_c` 0.05 and 0.2 s; D8 EEG leakage null; D9 the v1 NAS as a
comparator on every A and C1 task; D11 coverage and selective risk per
protocol; D12 `conditioning_delta`; D13 the all-events exogeneity sensitivity.

## 5. Seed map

The seed map is quoted from `protocols/v2/seed_map_v2.json` (SHA-256
`c9cf929f24350eca2b9f03a4be208e3844e7e8c7f9a65391c544d550f6cb7626`). Every split label is derived from the seed policy, never
written by hand, and a run uses one split.

<!-- seed-map:begin -->

Policy (`protocols/v2/seed_map_v2.json`, schema `mpc-bench-seed-map/2`): development seeds 0-999; confirmatory seeds >= 20000; never reused: 10000-19999 (v1 confirmatory seeds (mpcbench-freeze-v1)); unassigned: 1000-9999 (neither development nor confirmatory). Task seeds and seed bases; seeds derived inside a task from a task seed or a seed base are not classified. Freeze tag `mpcbench-freeze-v2`.

Development seeds used before v2: 0-39, 100-119, 200-279, 300-319, 440-449, 500-803, 900-919, 985-998.

| Split | Seeds | Use |
|---|---|---|
| development | `900-939` | reference blocks: PC_nominal and the own-lesion witnesses for every family protocol and view (re-use allowed) |
| development | `320-399` | dry run of every Tier-A design at about 15 % scale |
| development | `320-331` | witnesses (part of `320-399`) |
| development | `332-335` | sweeps (part of `320-399`) |
| development | `336-339` | factorial (part of `320-399`) |
| development | `340-351` | adversaries (part of `320-399`) |
| development | `352-371` | RAM-only arm (part of `320-399`) |
| development | `372-383` | family C1 (part of `320-399`) |
| development | `384-399` | forward arms (part of `320-399`) |
| development | `400-439` | family-B dry run and IIM calibration cells (null-calibration seed base 400) |
| development | `804-819` | Hopf and forward-family-A development at the development regime (lead-field seed 20260928, width 0.5) |
| development | `820-824` | development twin networks, r = 0..6 (RAM-only r = 0..7) |
| development | `850-899` | PDI concordance admission battery (9 classes; 850-869 added before any battery output) |
| development | `940-979` | PDI concordance admission battery (added before any battery output; formerly spare) |
| development | `980-984` | smoke tests of held-out conditions (outputs discarded unread) |
| confirmatory | `20000-20039` | manipulation checks; RAM-only arm |
| confirmatory | `20000-20019` | witnesses, adversaries, held-out conditions |
| confirmatory | `20000-20044` | PC_nominal and the single deficits (45 seed clusters) |
| confirmatory | `20000-20045` | family-A null witnesses N_independent_noise and N_ar1 (46 seed clusters each; HCv2-1) (resize: CD-11 / HCv2-1: 92 clusters, the smallest size whose pooled bound tolerates one PRESENT event; 20020-20045 added, no threshold changed; shared with: the forward family-A null class N_ar1 (20000-20060) observes the same N_ar1 simulations, one configuration under two designs as for PC_nominal; no other assignment uses these systems on these seeds) |
| confirmatory | `20000-20128` | family-C1 single deficits W_NAS_no_workspace, W_NAS_broadcast_only and W_IIM_feedforward (129 seed clusters each; HCv2-23) (resize: CD-11 / HCv2-23: the any-event clusters are keyed by the seed alone, so the A and C1 single deficits on 20000-20044 form 45 clusters (P(SUPPORTED \| development) 0.069); 129 clusters tolerate four event clusters (0.884); 20045-20128 added, no threshold changed; shared with: family A keeps its single deficits on 20000-20044; the lowest level of the C1 c_int sweep (c_int = 0) has the knobs of W_IIM_feedforward and also runs on 20000-20064) |
| confirmatory | `20000-20009` | sweeps, factorial, twin networks |
| confirmatory | `20000-20064` | the c_int sweeps of families A and C1 (10 levels, 65 seeds per level; HCv2-14(f)) (resize: CD-11 / HCv2-14(f): with whole seed profiles resampled, P(SUPPORTED \| development) is 0.16 at 10 seeds per level and 0.82 at 65 for the family-A cell; the C1 cell is held out (HO-4) and follows family A by symmetry; 20010-20064 added, no threshold changed; shared with: the g_b and K sweeps stay on 20000-20009; the lowest level (c_int = 0) has the knobs of W_IIM_feedforward, which the witness designs also run on these seeds (20000-20044 in family A, 20000-20128 in family C1)) |
| confirmatory | `20000-20060` | Hopf G = 0 extension; forward family-A null classes |
| confirmatory | `20000-20149` | Hopf G_nom extension; forward family-A on classes |
| confirmatory | `20000-20499` | family B |
| confirmatory | `20000-20199` | HCv2-2 (part of `20000-20499`) |
| confirmatory | `20240-20279` | HCv2-12(a) (part of `20000-20499`) |
| confirmatory | `20280-20319` | HCv2-11 (part of `20000-20499`) |
| confirmatory | `20320-20419` | HCv2-13 (part of `20000-20499`) |
| confirmatory | `20420-20459` | HCv2-12(b) (part of `20000-20499`) |
| confirmatory | `20460-20499` | HCv2-12(c, d) (part of `20000-20499`) |
| confirmatory | `20000-20000` | null-calibration generator seed base |
| confirmatory | `20900-20919` | anchor replication blocks (extendable to: 20939; extension rule: 20900-20939 where the operating-characteristics check shows replication power < 0.9 at 20 seeds (anchor designs; forward arms after the held-out release); extended: A_anchors to 20939 (CD-11: A-H IIM replication power 0.565 at 20 seeds); forward_anchor_replication, Hopf arm, to 20939 (CD-11 after the held-out release: Hopf IIM validity-only replication power 0.264 (eeg64_noref), 0.699 (mne_template), 0.862 (eeg64) at 20 seeds)) |

Random streams: structural `SeedSequence(seed): network, oscillator frequencies and every v1 stream (existing draws are never re-ordered)`; twins `SeedSequence([seed, r])` with r = 1-30 (task schedule, process noise and rest; r = 0 reproduces the v1 streams bit for bit); reserved keys label_error 41, cue_jitter 42, staggered_driver 43. Lead-field seeds (regime parameters of the forward model, not task seeds): development regime 20260928, held-out regime 20261001.

<!-- seed-map:end -->

The Hopf arm's forward anchor replication runs on 20900-20939 (CD-11, decided
after the held-out release; `calibration_decisions.json`,
`replication_extended.forward_anchor_replication.hopf = true`, the design code
and the map's `extended` list); the two family-A forward arms keep 20900-20919.
The run plan of section 4.2 has these seeds.

## 6. Hypotheses

### 6.1 Conventions and decision rules

Conventions of `hypotheses_v2.json`: `alpha = 0.05`, `alpha_absent = 0.01`,
`z = 0.25`, `delta = 0.10`, false-PRESENT bound `alpha + 0.02 = 0.07`.

- **Combination.** A hypothesis is FALSIFIED if any evaluable decisive part is
  FALSIFIED, SUPPORTED if all are SUPPORTED, else INDETERMINATE;
  NOT_EVALUABLE and NOT_TESTABLE_BY_DESIGN parts are listed. The same rule
  combines the cells of a part.
- **Clusters.** Every re-scoring of one simulation (R and H, cut modes, views,
  primary and secondary forms) is one cluster in pooled bounds; a cluster
  counts its share of rows with the event.
- **Anchored principles.** Component-level parts use the principles with a
  valid anchor on the row's protocol; verdict-level parts use `N_anch`.
- **Tests stated as tests** ("at one-sided p < 0.05") are SUPPORTED when the
  criterion holds and FALSIFIED otherwise, as in v1.
- **Tally.** Decisive parts per outcome, including parts removed before the
  freeze with their development outcome as the predicted outcome (none was
  removed in this round).
- **Predictions.** A decisive part predicts SUPPORTED unless the part or its
  hypothesis states otherwise; stated predictions are shown with each part.

Decision rules (`m` = number of cells of the part unless the part states `m`;
CP = exact Clopper-Pearson bound):

| Rule | Decision |
|---|---|
| `h0_cell` | FALSIFIED if some cell's one-sided CP lower bound at `alpha/m` exceeds the bound; SUPPORTED if every pool's (per principle) CP upper bound at `alpha/P` is below the bound; else INDETERMINATE |
| `h0_cell_exceeds` | the mirror of `h0_cell` for a predicted exceedance: SUPPORTED where `h0_cell` would be FALSIFIED and vice versa |
| `three_zone` | per cell: SUPPORTED iff the point rate >= x; FALSIFIED iff the CP upper bound at `0.05/m` < x; else INDETERMINATE |
| `reversed_three_zone` | per cell (m = 1 unless stated): SUPPORTED iff the CP upper bound < x; FALSIFIED iff the point rate >= x; else INDETERMINATE |
| `three_zone_at_most` | per cell: SUPPORTED iff the point rate <= x; FALSIFIED iff the CP lower bound at `0.05/m` > x; else INDETERMINATE |
| `count_at_most` | per cell (m = 1): SUPPORTED iff the rate <= `max_rate`; FALSIFIED iff the CP lower bound > `falsify_above`; else INDETERMINATE |
| `rate_lower_bound` | per cell (m = 1): SUPPORTED iff the CP lower bound > x; FALSIFIED iff the CP upper bound < x; else INDETERMINATE |
| `demonstration` | per cell: SUPPORTED iff the CP upper bound at `level` < bound; with curtailment, FALSIFIED once the events rule out the demonstration at the planned n, INDETERMINATE while the planned runs are not reached; otherwise FALSIFIED |
| `any_event_clusters` | FALSIFIED if some class's CP lower bound at `alpha/m` exceeds the bound; SUPPORTED if the any-event seed-cluster CP upper bound (alpha) is below the bound; else INDETERMINATE |
| `false_exclusion` | per cell, clusters are seeds: FALSIFIED if the seed-level CP lower bound at `alpha/m` exceeds `seed_falsify` or the cluster-bootstrap lower bound of the row rate exceeds `row_bound`; SUPPORTED if the row rate <= `row_bound` and the seed-level CP upper bound < `seed_bound`; else INDETERMINATE |
| `usable_share` | per (family, check): usable iff passes >= `ceil(share x seeds)`; an unusable check is a FALSIFIED cell |
| `kappa_null` | per cell: `kappa0 = SD(c) / RMS(se_c)` with its chi-square interval; FALSIFIED if the interval lies outside the point range or a tail's CP lower bound at `0.05/(2P)` exceeds `tail_falsify`; SUPPORTED if the point is in the point range, the interval inside the interval range and both tail rates <= `tail_max`; else INDETERMINATE; fewer than 3 rows with a sampling SE: NOT_EVALUABLE |
| `kappa_twins` | per (class, SE method) with c defined in >= `min_defined` of the class's sessions: pooled within-network SD of c over RMS(se_c), chi-square interval on `sum (n_twin - 1)` df, `q_A` tails at `alpha_A / members`; FALSIFIED if the interval lies outside the point range or a tail's CP lower bound at `0.05/m` exceeds `tail_falsify`; SUPPORTED as `kappa_null` with `tail_max`; else INDETERMINATE |
| `concordance_twins` | the share of twin sessions whose count differs from the network's concordant count: SUPPORTED iff <= `share_max`; FALSIFIED iff its CP lower bound > `share_max`; else INDETERMINATE |
| `anchor_replicates` | per (protocol, principle): the anchor status on the replication block (validity: >= 0.9 finite and lower bound > 0; specificity as section 3.6, unless validity only) equals the predicted status from the frozen protocol |
| `spearman_ols` | per cell: Spearman rho > 0 at one-sided p < alpha (and a positive OLS slope on the dose rescaled to [0, 1] where required) |
| `spearman_monotone` | per cell: Spearman(sampled, exact) > 0 at one-sided p < alpha and level medians non-decreasing in the exact value (decreases below `tie_tolerance` count as ties) |
| `jonckheere_terpstra` | per cell: Jonckheere-Terpstra one-sided p < alpha over the stated order, and the stated median condition |
| `wilcoxon`, `sign_test` | per cell: one-sided Wilcoxon signed-rank or sign test p < alpha |
| `hodges_lehmann` | per cell: the median condition (where stated) and a one-sided Hodges-Lehmann lower bound > `lower_gt` |
| `median_threshold` | per cell: the median compared with x by `op` |
| `newcombe_includes_zero` | the Newcombe 95 % interval of the rate difference includes 0 (SUPPORTED) or excludes it (FALSIFIED) |
| `stepwise_sign` | per step: where the gate defines the estimate in >= `gate_share` of runs, the median paired change has the sign of the exact change and the one-sided sign test is significant; otherwise the step counts as correctly undefined |
| `admission_matches` | the observed registry flags equal the predicted ones per view; NOT_EVALUABLE where the anchor cell is invalid |
| `fmd_concordance` | per view and seed, the event "forward contrast has the source's sign and at least `ratio` of its size", judged by the stated three-zone form |
| `describe` | reported only |

Gated parts (`gate`) are decisive only where the testability table gives
`pi >= 0.9`; elsewhere their cells are NOT_TESTABLE_BY_DESIGN and the part
named in `replaced_by` decides. `requires.oracle` applies prerequisite M per
row; `requires.usable` names switches that must be usable. Record names are
bound in one place (`vocabulary`, `fields` and `derived` of the hypotheses
file): a field such as `@nas_z_receive` is a path into the records, for
example `details.estimator.directions.receive.z`; `@iim_p_ind` reads
`details.p_ind`.

### 6.2 Tier-A hypotheses HCv2-0 to HCv2-24

Each block gives the hypothesis-level fields of `hypotheses_v2.json` and then
every part: its role, label and predicted outcome, text, rule, parameters (as
JSON with sorted keys), the further fields of the part, the outcome of the
development dry run (DEVELOPMENT - NOT A RESULT; section 8 and the companion
`v2/development_expectations.md`), and the data selection.

#### HCv2-0: Prerequisite M (manipulation and realisation checks)

<!-- hypothesis HCv2-0 -->
- Tier A; role prerequisite; labels none; successor of M.
- Statement: A switch or new system is usable in a family iff its check passes in >= 36/40 seeds (seeds 20000-20039). Parts that need an unusable switch or system are NOT_EVALUABLE (reason ORACLE). The realisation check of the slow-context preset (PC_nominal under slow_context_bold, family A) is gated in the same way and is required by the rows of the forward BOLD arm only (oracle_checks.conditions). PC_half between the off and nominal medians and the twin hashes are reported, never gates.
- Development expectation: v1: 12/12 family x switch cells usable (worst 39/40). Development (CD-13, seeds 320-359): 26 of 27 checks pass 40/40, every switch check of families A and C1 among them and the slow-context check (shortest complete context run 30.0-34.5 s against 30 s); ADV_NAS_staggered_tau10 driver_reaches_every_module passes 34/40 (36 needed; the six failures, 0.035-0.048 against the floor 0.05, are in the 8 s modules), so that condition is predicted not usable: P(usable) is 0.26 at a true pass rate of 0.85 and 0.63 at 0.90, about 0.74 not usable. If it is not usable, its HCv2-0(b) cell is FALSIFIED and HCv2-9(d) is NOT_EVALUABLE (reason ORACLE). The condition, its check and its threshold are unchanged. Reported (development): PC_half lies between the off and nominal medians in 10 of 12 family x switch cells; not for K in A and C1, whose context-information signature is higher at PC_half's K = 3 than at the nominal K = 6 (medians 0.66 against 0.49 in A, 0.64 against 0.63 in C1); the twins pass 1920/1920 (8 twin witnesses x 40 seeds x replicates 1-6).
- Development dry run (DEVELOPMENT - NOT A RESULT): FALSIFIED.

<!-- part HCv2-0(a) -->
**Part `HCv2-0(a)`** (decisive; label: none; predicted: SUPPORTED (default))

- Text: frozen mpc-bench-manipulation/1.0.0 switch checks (eta, K, g_b, ff_only, c_int, e) usable in families A and C1
- Rule: `usable_share`
- Parameters: `{"near_band": 0.05, "share": 0.9}`
- Development dry run: SUPPORTED (usable iff passes >= ceil(0.9 x seeds)).

<details><summary>Data selection of HCv2-0(a)</summary>

```json
{"source": "manipulation", "where": {"kind": "switch"}, "event": {"passed": true}, "cells": ["family", "check_id"], "cluster": ["family", "check_id", "seed"]}
```

</details>

<!-- part HCv2-0(b) -->
**Part `HCv2-0(b)`** (decisive; label: none; predicted: SUPPORTED (default))

- Text: realisation checks of the new systems and of the slow-context preset (mpc-bench-manipulation/1.1.0) usable
- Rule: `usable_share`
- Parameters: `{"near_band": 0.05, "share": 0.9}`
- Development dry run: FALSIFIED (usable iff passes >= ceil(0.9 x seeds)).

<details><summary>Data selection of HCv2-0(b)</summary>

```json
{"source": "manipulation", "where": {"kind": "new_system"}, "event": {"passed": true}, "cells": ["family", "check_id"], "cluster": ["family", "check_id", "seed"]}
```

</details>

<!-- part HCv2-0(c) -->
**Part `HCv2-0(c)`** (reported; label: none; predicted: none (reported part))

- Text: reported: PC_half lies between the switched-off and the nominal medians of each 1.0.0 signature, per family and switch (design 3.5; never a gate)
- Rule: `describe`
- Parameters: `{}`
- Development dry run: REPORTED (reported).

<details><summary>Data selection of HCv2-0(c)</summary>

```json
{"source": "manipulation", "where": {"kind": "reported", "system_id": "PC_half", "check": "between_off_and_nominal"}, "event": {"passed": true}, "cells": ["family", "check_id"], "cluster": ["family", "check_id"]}
```

</details>

<!-- part HCv2-0(d) -->
**Part `HCv2-0(d)`** (reported; label: none; predicted: none (reported part))

- Text: reported: twins of the twin witnesses (replicates 1-6, families A and C1) share the structural hash of replicate 0 and differ from it in the schedule hash and the series (design 3.5; the integrity audit IA-10 checks the same on the records)
- Rule: `describe`
- Parameters: `{}`
- Development dry run: REPORTED (reported).

<details><summary>Data selection of HCv2-0(d)</summary>

```json
{"source": "manipulation", "where": {"kind": "reported", "check": "twin_hashes"}, "event": {"passed": true}, "cells": ["family", "system_id"], "cluster": ["family", "system_id", "seed", "replicate"]}
```

</details>

#### HCv2-1: Null calibration of PRESENT, including conditioned nulls

<!-- hypothesis HCv2-1 -->
- Tier A; labels none; successor of HC1.
- Statement: In every (null cell, anchored principle) pair, P(PRESENT) <= alpha + 0.02 = 0.07.
- Development expectation: v1: NAS and IIM 0/1280, SRPI 12/1640. NAS under R (development): disconnected 0/20 PRESENT in A and C; staggered 0/20 per variant. RAM-PE (development): the 128 null-calibration RAM-PE components are UNDEFINED(INSUFFICIENT_UPDATES) and are not counted; the family-A null witnesses give 48 defined rows, 0 PRESENT.
- Notes: Null-calibration components are judged under the A-R protocol (as v1 judged them under the bench protocol); anchored = a valid anchor on the row's protocol (component level). RAM-PE rows count only where defined (a finite estimate and no definedness reason; null_calibration_ram = require_defined): the null-calibration generator has no trial or feedback structure, so its RAM-PE components are UNDEFINED by construction and evidence for neither outcome; a decision-type UNDEFINED (for example ABSENT_NOT_REACHABLE) stays a non-event. The family-A null witnesses N_independent_noise and N_ar1 run on 46 seed clusters each (20000-20045; CD-11, seeds resized, never thresholds): 92 clusters is the smallest size whose pooled RAM-PE bound at 0.05 / 5 stays below 0.07 with one PRESENT event (0 events would need 64). The share of null-calibration rows undefined by a definedness reason is reported per cell and principle in HCv2-1(definedness).
- Development dry run (DEVELOPMENT - NOT A RESULT): INDETERMINATE.

<!-- part HCv2-1 -->
**Part `HCv2-1`** (decisive; label: none; predicted: SUPPORTED (default))

- Text: H0 cell rule over (i) the null-calibration generator, (ii) the bench null witnesses (R and H scorings one cluster; family-A N_independent_noise and N_ar1 on 46 seeds each) and (iii) NAS under R on the conditioned nulls; RAM-PE rows only where defined
- Rule: `h0_cell`
- Parameters: `{"alpha": 0.05, "bound": 0.07}`
- Requires: `{"oracle": true}`
- Development dry run: INDETERMINATE (a pooled upper bound is not below the bound).

<details><summary>Data selection of HCv2-1</summary>

```json
{"union": [{"label": "null_calibration", "where": {"design": "@design.null_calibration"}, "cell_label": "null_calibration:{@null_kind}:T{@n_time}:N{@n_nodes}|{principle}"}, {"label": "bench_nulls", "where": {"any": [{"design": "@design.witnesses", "estimator_form": "@form.primary", "family": "@family.A", "system": "@systems.nulls_A"}, {"design": "@design.witnesses", "estimator_form": "@form.primary", "family": "@family.C1", "system": "N_uncoupled"}]}, "cell_label": "witness:{family}:{system}|{principle}"}, {"label": "conditioned_nulls", "where": {"principle": "NAS", "declaration_id": "R", "estimator_form": "@form.primary", "any": [{"design": "@design.witnesses", "system": "N_modules_disconnected", "family": ["@family.A", "@family.C1"]}, {"design": "@design.adversarial", "system": "ADV_NAS_staggered_driver", "family": "@family.A"}]}, "cell_label": "conditioned:{family}:{system}:{@variant}|NAS"}], "where": {"declaration_id": ["R", "H", "none"], "estimator_form": "@form.primary", "any": [{"principle": {"ne": "RAM"}}, {"defined": true}]}, "principle_filter": "valid_anchor", "event": {"status": "PRESENT"}, "pool_by": ["principle"]}
```

</details>

<!-- part HCv2-1(definedness) -->
**Part `HCv2-1(definedness)`** (reported; label: none; predicted: none (reported part))

- Text: reported: share of null-calibration rows undefined by a definedness reason, per null cell and principle
- Rule: `describe`
- Parameters: `{}`
- Development dry run: REPORTED (reported).

<details><summary>Data selection of HCv2-1(definedness)</summary>

```json
{"where": {"design": "@design.null_calibration", "estimator_form": "@form.primary"}, "event": {"defined": false}, "cell_label": "null_calibration:{@null_kind}:T{@n_time}:N{@n_nodes}|{principle}"}
```

</details>

#### HCv2-2: IIM independence-null rank calibration, including conditioned nulls

<!-- hypothesis HCv2-2 -->
- Tier A; labels C; successor of HC2.
- Statement: The rank exceedance P(p_ind <= 0.05) is <= 0.07 in every cell, for the unconditioned, stratified and residualised estimators.
- Development dry run (DEVELOPMENT - NOT A RESULT): INDETERMINATE.

<!-- part HCv2-2 -->
**Part `HCv2-2`** (decisive; label: C; predicted: SUPPORTED (default))

- Text: H0 cell rule over 18 cells (independent units: 4 T x 2 cuts; stratified irrelevant driver: 3 T x 2 cuts; residualised continuous and switching drivers x 2 cuts); cut-mode pairs share trajectories and are one cluster in the pooled bound
- Rule: `h0_cell`
- Parameters: `{"P": 1, "alpha": 0.05, "bound": 0.07}`
- Development dry run: INDETERMINATE (a pooled upper bound is not below the bound).

<details><summary>Data selection of HCv2-2</summary>

```json
{"where": {"design": "@design.family_b", "family": "@family.B", "principle": "IIM", "seed_block": "family_b_rank_calibration", "estimator_form": ["@form.primary", "@form.iim_bidirectional"], "@b_null_order": "shift_then_project"}, "union": [{"label": "independent", "where": {"@b_condition": "independent"}, "cell_label": "independent:T{@b_T}:{estimator_form}"}, {"label": "stratified", "where": {"@b_condition": "stratified"}, "cell_label": "stratified:T{@b_T}:{estimator_form}"}, {"label": "residualised", "where": {"@b_condition": ["residualised_continuous", "residualised_switching"]}, "cell_label": "{@b_condition}:{estimator_form}"}], "event": {"@iim_p_ind": {"le": 0.05}}, "pool_by": ["principle"]}
```

</details>

<!-- part HCv2-2(reported) -->
**Part `HCv2-2(reported)`** (reported; label: none; predicted: none (reported part))

- Text: residualise-then-shift null order on the same data (predicted anti-conservative)
- Rule: `describe`
- Parameters: `{}`
- Development dry run: REPORTED (reported).

<details><summary>Data selection of HCv2-2(reported)</summary>

```json
{"where": {"design": "@design.family_b", "family": "@family.B", "principle": "IIM", "seed_block": "family_b_rank_calibration", "estimator_form": ["@form.primary", "@form.iim_bidirectional"], "@b_null_order": "project_then_shift"}, "cell_label": "{@b_condition}:T{@b_T}:{estimator_form}", "event": {"@iim_p_ind": {"le": 0.05}}}
```

</details>

#### HCv2-3: Interval calibration at the null (HR1, restated)

<!-- hypothesis HCv2-3 -->
- Tier A; labels C, R; successor of HC1; predicted outcome FALSIFIED.
- Statement: (a) kappa0 = SD(c) / RMS(se_c) has its point in [0.8, 1.25] and its 90 % chi-square interval inside [0.67, 1.5]; (b) both one-sided tail rates P(c - q_P se_c > 0) and P(c + q_P se_c < 0) are <= 0.075 (point).
- Development expectation: Development (null calibration at seed 400 with 32 replicates per null kind, and the family-A null witnesses on seeds 320-331, judged under the decided build): all 152 PDI null rows (128 null-calibration, 24 witness) are admitted concordant components without a sampling SE, so the PDI cell is expected NOT_EVALUABLE; HCv2-3(concordant) reports their count and share (152 of 152). SRPI v1 FALSIFIED on the conservative side (kappa0 0.47, 90 % CI 0.43-0.52), as predicted. NAS conservative in both directions (kappa0 0.69; 90 % CI 0.62-0.77 receive, 0.63-0.78 return; lower-tail rates 0.067 and 0.058): FALSIFIED on the conservative side. IIM (kappa0 0.76, 0.69-0.85) and RAM-PE (0.74, 0.60-0.99; 24 defined rows) conservative and short of the calibration criterion (INDETERMINATE).
- Development dry run (DEVELOPMENT - NOT A RESULT): FALSIFIED.

<!-- part HCv2-3 -->
**Part `HCv2-3`** (decisive; label: C; predicted: FALSIFIED (stated))

- Text: per anchored principle under A-R (NAS per direction, each with its own SE): kappa0 and both tails; FALSIFIED if some kappa0 interval lies outside [0.8, 1.25] or some tail's cluster-robust lower bound at 0.05/(2P) exceeds 0.10 (P counts NAS's two directions); an admitted concordant row (a PDI component with SE 0, which has no sampling SE: its error control is the concordance battery) enters no kappa0 or tail, and its cell keeps its place in P
- Rule: `kappa_null`
- Parameters: `{"exclude_concordant": true, "interval": [0.67, 1.5], "level": 0.9, "point": [0.8, 1.25], "q_alpha": 0.05, "tail_falsify": 0.1, "tail_max": 0.075}`
- Cell predictions: `[{"match": {"principle": "SRPI"}, "prediction": "FALSIFIED"}, {"match": {"principle": ["NAS", "IIM", "PDI", "RAM"]}, "prediction": "SUPPORTED"}]`
- Notes: SRPI v1 predicted FALSIFIED on the conservative side [R]. NAS is calibrated per direction: its component c is the smaller direction, whose spread and tails differ from those of either direction even when each direction's SE is exact. RAM-PE rows count only where defined (null_calibration_ram = require_defined, as in HCv2-1). The family-A null witnesses enter on all their seeds (46 each, 20000-20045; CD-11). Admitted concordant rows leave the kappa0 and the tails as in HCv2-4 (a, b); a cell with fewer than 3 other rows is NOT_EVALUABLE. Their number and share per cell are reported in HCv2-3(concordant).
- Development dry run: FALSIFIED (kappa0 in [0.8, 1.25] with its 90 % interval inside [0.67, 1.5], and both tails <= 0.075).

<details><summary>Data selection of HCv2-3</summary>

```json
{"where": {"declaration_id": ["R", "none"], "estimator_form": "@form.primary", "any": [{"design": "@design.null_calibration"}, {"design": "@design.witnesses", "family": "@family.A", "system": "@systems.nulls_A"}], "all": [{"any": [{"principle": {"ne": "RAM"}}, {"defined": true}]}]}, "union": [{"label": "principle", "where": {"principle": {"ne": "NAS"}}, "cell_label": "{principle}"}, {"label": "NAS_receive", "where": {"principle": "NAS"}, "value": "@nas_c_receive", "se": "@nas_se_c_receive", "df": "@nas_df_c_receive", "cell_label": "NAS:receive"}, {"label": "NAS_return", "where": {"principle": "NAS"}, "value": "@nas_c_return", "se": "@nas_se_c_return", "df": "@nas_df_c_return", "cell_label": "NAS:return"}], "principle_filter": "valid_anchor", "value": "c"}
```

</details>

<!-- part HCv2-3(concordant) -->
**Part `HCv2-3(concordant)`** (reported; label: none; predicted: none (reported part))

- Text: reported: the number and share of admitted concordant rows (no sampling SE; error control by the concordance battery) per anchored principle; they enter no kappa0 or tail of HCv2-3
- Rule: `describe`
- Parameters: `{}`
- Development dry run: REPORTED (reported).

<details><summary>Data selection of HCv2-3(concordant)</summary>

```json
{"where": {"declaration_id": ["R", "none"], "estimator_form": "@form.primary", "any": [{"design": "@design.null_calibration"}, {"design": "@design.witnesses", "family": "@family.A", "system": "@systems.nulls_A"}], "all": [{"any": [{"principle": {"ne": "RAM"}}, {"defined": true}]}]}, "principle_filter": "valid_anchor", "event": {"concordant": true}, "cell_label": "{principle}"}
```

</details>

#### HCv2-4: SE calibration against white-box twins (all v2 SE methods)

<!-- hypothesis HCv2-4 -->
- Tier A; role se_calibration; labels C; successor of HC10.
- Statement: Per (estimator, SE method, class) with c defined in >= 80 % of sessions: (a) kappa = pooled within-network SD of c / RMS(se_c) has its point in [0.8, 1.25] and its 90 % interval inside [0.67, 1.5]; (b) the q_A tails are each <= 0.02 (point); (c) concordant PDI counts agree across twins (share differing <= 0.05).
- Development expectation: Development per-class summary (CD-2 to CD-4; twins 820-824 with 7 sessions each, RAM-only twins with 8): NAS FALSIFIED on the anti-conservative side; no anti-conservative failure for IIM, RAM-PE, PDI or SRPI (their failures are conservative or short of the falsification criterion). NAS (jackknife_contiguous_10, calibrated in 7 of 32 classes): in the A-H return direction kappa 3.04 (90 % CI 2.52-3.87; tails 0.171 below, 0.114 above) on W_PDI_single_attractor and 1.94 (1.61-2.47) on PC_nominal; 0.52-0.91 in most C1 cells. IIM (circular_block_bootstrap_10pct_B50 at se_df 9): calibrated in 8 of 20 classes; kappa 0.47-0.72 in the PC_half classes; 4 classes with one tail exceedance in 35 sessions. RAM-PE (shift_null_sd): calibrated in 6 of 13 classes; on the joint bench PC_half kappa 1.25 (1.04-1.59, not entirely above 1.25), above-tail 0.057. PDI W_PDI_single_attractor under A-R and A-H: a jackknife SE in 2 of 35 sessions (33 admitted concordant), so these cells are not eligible. The reversion rule is therefore expected to re-classify NAS ABSENTs; hypotheses that use NAS ABSENT are reported both ways.
- Notes: Reversion rule: an SE method FALSIFIED on the anti-conservative side re-classifies every ABSENT of that estimator in every verdict-level analysis as UNDEFINED(SE_NOT_CALIBRATED); the hypotheses that used those ABSENTs are reported both ways. A conservative failure changes nothing. IIM SE calibration rests on the 20 family-A classes only: family C1 is held out for IIM (HO-4) and has no development twins.
- Development dry run (DEVELOPMENT - NOT A RESULT): FALSIFIED.

<!-- part HCv2-4(a,b) -->
**Part `HCv2-4(a,b)`** (decisive; label: C; predicted: SUPPORTED (default))

- Text: kappa with its chi-square interval on sum (n_twin - 1) df and the q_A tails, per (estimator, SE method, class) with c and its SE defined in >= 80 % of the class's sessions; NAS per direction with its own SE and the direction's TOST level alpha_A / 2; a session without a sampling SE (an undefined component, or an admitted concordant PDI component, whose error control is the concordance battery) forms no cell and counts against the defined share
- Rule: `kappa_twins`
- Parameters: `{"alpha_absent": 0.01, "by_se_method": true, "interval": [0.67, 1.5], "level": 0.9, "members": {"NAS": 2}, "min_defined": 0.8, "point": [0.8, 1.25], "tail_falsify": 0.03, "tail_max": 0.02}`
- Development dry run: FALSIFIED (kappa in [0.8, 1.25] with its 90 % interval inside [0.67, 1.5]; q_A tails <= 0.02).

<details><summary>Data selection of HCv2-4(a,b)</summary>

```json
{"union": [{"label": "A_twins", "where": {"design": ["@design.twins", "@design.witnesses"], "family": "@family.A", "system": ["PC_nominal", "PC_half", "W_NAS_no_workspace", "W_IIM_feedforward", "W_PDI_single_attractor"], "declaration_id": ["R", "H"], "principle": {"ne": "NAS"}}}, {"label": "A_twins_NAS_receive", "where": {"design": ["@design.twins", "@design.witnesses"], "family": "@family.A", "system": ["PC_nominal", "PC_half", "W_NAS_no_workspace", "W_IIM_feedforward", "W_PDI_single_attractor"], "declaration_id": ["R", "H"], "principle": "NAS"}, "value": "@nas_c_receive", "se": "@nas_se_c_receive", "df": "@nas_df_c_receive", "cell_label": "{family}:{system}:{@sweep_level}:{declaration_id}|NAS:receive|{estimator_form}|{estimator_version}"}, {"label": "A_twins_NAS_return", "where": {"design": ["@design.twins", "@design.witnesses"], "family": "@family.A", "system": ["PC_nominal", "PC_half", "W_NAS_no_workspace", "W_IIM_feedforward", "W_PDI_single_attractor"], "declaration_id": ["R", "H"], "principle": "NAS"}, "value": "@nas_c_return", "se": "@nas_se_c_return", "df": "@nas_df_c_return", "cell_label": "{family}:{system}:{@sweep_level}:{declaration_id}|NAS:return|{estimator_form}|{estimator_version}"}, {"label": "C1_twins", "where": {"design": ["@design.twins", "@design.witnesses"], "family": "@family.C1", "system": ["PC_nominal", "W_NAS_no_workspace", "W_IIM_feedforward"], "declaration_id": ["R", "H"], "principle": {"ne": "NAS"}}}, {"label": "C1_twins_NAS_receive", "where": {"design": ["@design.twins", "@design.witnesses"], "family": "@family.C1", "system": ["PC_nominal", "W_NAS_no_workspace", "W_IIM_feedforward"], "declaration_id": ["R", "H"], "principle": "NAS"}, "value": "@nas_c_receive", "se": "@nas_se_c_receive", "df": "@nas_df_c_receive", "cell_label": "{family}:{system}:{@sweep_level}:{declaration_id}|NAS:receive|{estimator_form}|{estimator_version}"}, {"label": "C1_twins_NAS_return", "where": {"design": ["@design.twins", "@design.witnesses"], "family": "@family.C1", "system": ["PC_nominal", "W_NAS_no_workspace", "W_IIM_feedforward"], "declaration_id": ["R", "H"], "principle": "NAS"}, "value": "@nas_c_return", "se": "@nas_se_c_return", "df": "@nas_df_c_return", "cell_label": "{family}:{system}:{@sweep_level}:{declaration_id}|NAS:return|{estimator_form}|{estimator_version}"}, {"label": "RAM_twins", "where": {"design": ["@design.ram_only_twins", "@design.ram_only"], "principle": "RAM", "@sweep_knob": "eta", "@sweep_level": [0.0, 0.1, 0.3]}}], "where": {"seed_block": "twin_networks"}, "cell_label": "{family}:{system}:{@sweep_level}:{declaration_id}|{principle}|{estimator_form}|{estimator_version}", "network": ["family", "system", "@sweep_level", "seed", "declaration_id", "principle", "estimator_form", "member"], "value": "c"}
```

</details>

<!-- part HCv2-4(c) -->
**Part `HCv2-4(c)`** (decisive; label: C; predicted: SUPPORTED (default))

- Text: PDI concordance: among concordant components, the share of twin sessions whose count differs from the concordant count is <= 0.05
- Rule: `concordance_twins`
- Parameters: `{"alpha": 0.05, "share_max": 0.05}`
- Development dry run: SUPPORTED (share of twin sessions differing from the concordant count <= 0.05).

<details><summary>Data selection of HCv2-4(c)</summary>

```json
{"where": {"design": ["@design.twins", "@design.witnesses"], "family": "@family.A", "principle": "PDI", "declaration_id": "R", "estimator_form": "@form.primary", "seed_block": "twin_networks"}, "network": ["family", "system", "seed", "declaration_id"], "value": "estimate"}
```

</details>

#### HCv2-5: No false exclusion of mechanism-bearing systems (HR2)

<!-- hypothesis HCv2-5 -->
- Tier A; labels C; verdict-level ABSENT: yes (the HCv2-4 reversion rule applies).
- Statement: (a) Per (family, protocol, principle): ABSENT among on-rows <= 0.02 per row. (b) Per (family, protocol): EXCLUDED among runs with every N_anch mechanism on <= 0.02.
- Development expectation: 0 events (dry run with v2 SEs at CD-10).
- Development dry run (DEVELOPMENT - NOT A RESULT): INDETERMINATE.

<!-- part HCv2-5(a) -->
**Part `HCv2-5(a)`** (decisive; label: C; predicted: SUPPORTED (default))

- Text: false ABSENT among mechanism-on rows; PDI concordance ABSENTs in their own cells; clusters are seeds
- Rule: `false_exclusion`
- Parameters: `{"alpha": 0.05, "bootstrap_b": 2000, "bootstrap_seed": 20261005, "row_bound": 0.02, "seed_bound": 0.15, "seed_falsify": 0.05}`
- Development dry run: INDETERMINATE (false exclusion per cell (seed clusters)).

<details><summary>Data selection of HCv2-5(a)</summary>

```json
{"where": {"design": ["@design.witnesses", "@design.sweep", "@design.factorial"], "family": ["@family.A", "@family.C1"], "declaration_id": ["R", "H"], "estimator_form": "@form.primary"}, "mechanism_on": "own", "event": {"status": "ABSENT"}, "cells": ["family", "declaration_id", "principle", "concordant"], "cluster": ["family", "seed"]}
```

</details>

<!-- part HCv2-5(b) -->
**Part `HCv2-5(b)`** (decisive; label: C; predicted: SUPPORTED (default))

- Text: false EXCLUDED among runs with every N_anch mechanism on
- Rule: `false_exclusion`
- Parameters: `{"alpha": 0.05, "bootstrap_b": 2000, "bootstrap_seed": 20261005, "row_bound": 0.02, "seed_bound": 0.15, "seed_falsify": 0.05}`
- Development dry run: INDETERMINATE (false exclusion per cell (seed clusters)).

<details><summary>Data selection of HCv2-5(b)</summary>

```json
{"source": "verdicts", "where": {"design": ["@design.witnesses", "@design.sweep", "@design.factorial"], "family": ["@family.A", "@family.C1"], "declaration_id": ["R", "H"], "estimator_form": "@form.primary"}, "mechanism_on": "all_n_anch", "event": {"verdict_value": "EXCLUDED"}, "cells": ["family", "declaration_id"], "cluster": ["family", "seed"]}
```

</details>

<!-- part HCv2-5(dose-only) -->
**Part `HCv2-5(dose-only)`** (reported; label: none; predicted: none (reported part))

- Text: reported, not decisive: ABSENT among the rows that the dose condition alone labels mechanism-on and the CD-8 rule labels off (estimator blindness stays visible)
- Rule: `describe`
- Parameters: `{}`
- Development dry run: REPORTED (reported).

<details><summary>Data selection of HCv2-5(dose-only)</summary>

```json
{"where": {"design": ["@design.witnesses", "@design.sweep", "@design.factorial"], "family": ["@family.A", "@family.C1"], "declaration_id": ["R", "H"], "estimator_form": "@form.primary"}, "mechanism_on": "dose_only", "event": {"status": "ABSENT"}, "cells": ["family", "declaration_id", "principle"], "cluster": ["family", "seed"]}
```

</details>

#### HCv2-6: Anchor validity, specificity and replication

<!-- hypothesis HCv2-6 -->
- Tier A; labels R; successor of HR3.
- Statement: Every (protocol, principle) pair's development anchor status replicates on the confirmatory replication block (validity: one-sided 95 % t lower bound of the mean excess > 0; specificity: mean paired contrast >= 0.5 x anchor with lower bound > 0). Pairs removed by the gate are included.
- Development dry run (DEVELOPMENT - NOT A RESULT): SUPPORTED.

<!-- part HCv2-6(a) -->
**Part `HCv2-6(a)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: family protocols A-R, A-H, C1-R, C1-H and A-RAM160: the observed anchor status on the replication block (family A 20900-20939, extended at CD-11; C1 and A-RAM160 20900-20919) equals the frozen development status
- Rule: `anchor_replicates`
- Parameters: `{"alpha": 0.05, "block_size": 20, "directions": {"NAS": ["@nas_excess_receive", "@nas_excess_return"]}, "min_finite_share": 0.9, "predictions": [{"principle": "RAM", "protocol": "@protocol.A-R", "status": "valid_specific"}, {"principle": "PDI", "protocol": "@protocol.A-R", "status": "valid_specific"}, {"principle": "NAS", "protocol": "@protocol.A-R", "status": "valid_specific"}, {"principle": "IIM", "protocol": "@protocol.A-R", "status": "valid_specific"}, {"principle": "SRPI", "protocol": "@protocol.A-R", "status": "valid_specific"}, {"principle": "NAS", "protocol": "@protocol.A-H", "status": "valid_specific"}, {"principle": "IIM", "protocol": "@protocol.A-H", "status": "valid_nonspecific"}, {"principle": "RAM", "protocol": "@protocol.A-H", "status": "valid_specific"}, {"principle": "PDI", "protocol": "@protocol.A-H", "status": "valid_specific"}, {"principle": "SRPI", "protocol": "@protocol.A-H", "status": "valid_specific"}, {"principle": "NAS", "protocol": "@protocol.C1-R", "status": "valid_specific"}, {"principle": "IIM", "protocol": "@protocol.C1-R", "status": "valid_specific"}, {"principle": "NAS", "protocol": "@protocol.C1-H", "status": "valid_specific"}, {"principle": "IIM", "protocol": "@protocol.C1-H", "status": "valid_specific"}, {"principle": "RAM", "protocol": "@protocol.A-RAM160", "status": "valid_specific"}], "ratio": 0.5}`
- Requires: `{"oracle": true}`
- Development expectation: INDETERMINATE-capable (design 4.9): A-H|IIM is valid at 36/40 finite, on the 0.9 boundary, so its replication power is 0.565 at 20 seeds and 0.60 at 40, and no resize reaches 0.9; P(SUPPORTED | development) is about 0.60, A-H|IIM being the pair most likely to fail. Every other pair has power >= 0.96 at 20 seeds (CD-11).
- Notes: Predictions are read from the frozen family protocols' anchors blocks (CD-7); the listed draft predictions apply only where no frozen protocol states them. They are the development statuses (CD-7): NAS is specific under H, and IIM is anchored under C1-H (the 4.6 rule decides; HO-4 permits the C1 reference-block anchors).
- Development dry run: SUPPORTED (every (protocol, principle) anchor status replicates).

<details><summary>Data selection of HCv2-6(a)</summary>

```json
{"where": {"design": ["@design.anchor_replication", "@design.ram_only_anchor_replication"], "estimator_form": "@form.primary"}, "union": [{"label": "pc", "where": {"system": "PC_nominal"}}, {"label": "lesion", "where": {"any": [{"principle": "RAM", "system": "W_RAM_no_plasticity"}, {"principle": "PDI", "system": "W_PDI_single_attractor"}, {"principle": "NAS", "system": "W_NAS_no_workspace"}, {"principle": "IIM", "system": "W_IIM_feedforward"}, {"principle": "SRPI", "system": "W_SRPI_no_efference"}]}}], "cell_label": "{protocol_id}|{principle}", "value": "excess"}
```

</details>

<!-- part HCv2-6(b) -->
**Part `HCv2-6(b)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: forward views: the reference condition's anchor validity (FM0) replicates
- Rule: `anchor_replicates`
- Parameters: `{"alpha": 0.05, "block_size": 20, "min_finite_share": 0.9, "validity_only": true}`
- Requires: `{"oracle": true}`
- Notes: The v1 quadrant comparator protocols hopf-eeg64+iim_v1_quadrants and hopf-eeg64_noref+iim_v1_quadrants have their own validity-only anchors, from the quadrant pipeline on the released G_nom reference runs, and replicate here like every forward view; they do not inherit the v2 cluster pipeline's anchor (design_IIM D3). The PRESENT statuses of HCv2-15(b) are judged under these anchors.
- Development dry run: NOT_EVALUABLE (no input rows).

<details><summary>Data selection of HCv2-6(b)</summary>

```json
{"where": {"design": "@design.forward_anchor_replication"}, "union": [{"label": "pc", "where": {}}], "cell_label": "{protocol_id}|{principle}", "value": "excess"}
```

</details>

#### HCv2-7: NAS identification under a complete declaration (families A and C1)

<!-- hypothesis HCv2-7 -->
- Tier A; labels R, C; successor of HC3.
- Development expectation: Block means, v1 rule: A (i) 20/20, median 1.01; (ii) 20/20, 0.90; C (i) 20/20, 0.95; (ii) 20/20; relay c 0.45-0.58.
- Development dry run (DEVELOPMENT - NOT A RESULT): FALSIFIED.

<!-- part HCv2-7(i) -->
**Part `HCv2-7(i)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: paired Delta c_NAS(PC - W_NAS_no_workspace) >= 0.5 in >= 80 % of seeds, each family
- Rule: `three_zone`
- Parameters: `{"x": 0.8}`
- Requires: `{"oracle": true}`
- Development dry run: SUPPORTED (three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8).

<details><summary>Data selection of HCv2-7(i)</summary>

```json
{"where": {"design": "@design.witnesses", "declaration_id": "R", "principle": "NAS", "estimator_form": "@form.primary", "family": ["@family.A", "@family.C1"]}, "pair": {"on": ["family", "seed", "declaration_id", "principle", "estimator_form"], "a": {"system": "PC_nominal"}, "b": {"system": "W_NAS_no_workspace"}, "value": "c"}, "event": {"delta": {"ge": 0.5}}, "cells": ["family"]}
```

</details>

<!-- part HCv2-7(ii) -->
**Part `HCv2-7(ii)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: paired Delta c_NAS(PC - W_NAS_broadcast_only) >= 0.5 in >= 80 %
- Rule: `three_zone`
- Parameters: `{"x": 0.8}`
- Requires: `{"oracle": true}`
- Development dry run: SUPPORTED (three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8).

<details><summary>Data selection of HCv2-7(ii)</summary>

```json
{"where": {"design": "@design.witnesses", "declaration_id": "R", "principle": "NAS", "estimator_form": "@form.primary", "family": ["@family.A", "@family.C1"]}, "pair": {"on": ["family", "seed", "declaration_id", "principle", "estimator_form"], "a": {"system": "PC_nominal"}, "b": {"system": "W_NAS_broadcast_only"}, "value": "c"}, "event": {"delta": {"ge": 0.5}}, "cells": ["family"]}
```

</details>

<!-- part HCv2-7(iii) -->
**Part `HCv2-7(iii)`** (decisive; label: C; predicted: SUPPORTED (default))

- Text: relay: W_IIM_feedforward (c_int = 0) is NAS PRESENT in >= 80 % (NAS does not distinguish a loop from a relay)
- Rule: `three_zone`
- Parameters: `{"x": 0.8}`
- Requires: `{"oracle": true}`
- Development dry run: FALSIFIED (three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8).

<details><summary>Data selection of HCv2-7(iii)</summary>

```json
{"where": {"design": "@design.witnesses", "declaration_id": "R", "principle": "NAS", "estimator_form": "@form.primary", "family": ["@family.A", "@family.C1"], "system": "W_IIM_feedforward"}, "event": {"status": "PRESENT"}, "cells": ["family"]}
```

</details>

<!-- part HCv2-7(iv) -->
**Part `HCv2-7(iv)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: Spearman rho(c_NAS, g_b) > 0 at one-sided p < 0.05 with a positive OLS slope on the dose rescaled to [0, 1]
- Rule: `spearman_ols`
- Parameters: `{"alpha": 0.05, "require_positive_slope": true}`
- Requires: `{"usable": ["g_b"]}`
- Development dry run: SUPPORTED (Spearman one-sided p < 0.05 with a positive OLS slope).

<details><summary>Data selection of HCv2-7(iv)</summary>

```json
{"where": {"design": "@design.sweep", "declaration_id": "R", "principle": "NAS", "estimator_form": "@form.primary", "family": ["@family.A", "@family.C1"], "@sweep_knob": "g_b"}, "dose": "@sweep_level", "value": "c", "cells": ["family"]}
```

</details>

<!-- part HCv2-7(v-A) -->
**Part `HCv2-7(v-A)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: the v1 HC3 off-target criterion, predicted failure in family A: |Delta c_NAS(W_PDI_single_attractor - PC)| < z in < 80 % of seeds (reversed three-zone)
- Rule: `reversed_three_zone`
- Parameters: `{"alpha": 0.05, "x": 0.8}`
- Requires: `{"oracle": true}`
- Development expectation: Development (A-R, CD-7): |Delta c_NAS(W_PDI_single_attractor - PC)| < z in 44/52 = 0.846 of seeds, above 0.8, so the reversed three-zone rule gives FALSIFIED: the v1 failure is not seen with the v2 NAS estimator under a complete declaration (under H the share is 28/52). The prediction is unchanged.
- Development dry run: FALSIFIED (reversed three-zone at 0.8: SUPPORTED iff CP upper < 0.8, FALSIFIED iff rate >= 0.8).

<details><summary>Data selection of HCv2-7(v-A)</summary>

```json
{"where": {"design": "@design.witnesses", "declaration_id": "R", "principle": "NAS", "estimator_form": "@form.primary", "family": "@family.A"}, "pair": {"on": ["family", "seed", "declaration_id", "principle", "estimator_form"], "a": {"system": "W_PDI_single_attractor"}, "b": {"system": "PC_nominal"}}, "event": {"delta": {"abs_lt": 0.25}}, "cells": ["family"]}
```

</details>

<!-- part HCv2-7(v-C1) -->
**Part `HCv2-7(v-C1)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: family C1: |Delta c_NAS| < z for the K, eta and e witnesses in >= 80 % (three-zone)
- Rule: `three_zone`
- Parameters: `{"x": 0.8}`
- Requires: `{"oracle": true}`
- Development dry run: SUPPORTED (three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8).

<details><summary>Data selection of HCv2-7(v-C1)</summary>

```json
{"where": {"design": "@design.witnesses", "declaration_id": "R", "principle": "NAS", "estimator_form": "@form.primary", "family": "@family.C1"}, "union": [{"label": "W_PDI_single_attractor", "pair": {"on": ["family", "seed", "declaration_id", "principle", "estimator_form"], "a": {"system": "W_PDI_single_attractor"}, "b": {"system": "PC_nominal"}}}, {"label": "W_RAM_no_plasticity", "pair": {"on": ["family", "seed", "declaration_id", "principle", "estimator_form"], "a": {"system": "W_RAM_no_plasticity"}, "b": {"system": "PC_nominal"}}}, {"label": "W_SRPI_no_efference", "pair": {"on": ["family", "seed", "declaration_id", "principle", "estimator_form"], "a": {"system": "W_SRPI_no_efference"}, "b": {"system": "PC_nominal"}}}], "event": {"delta": {"abs_lt": 0.25}}, "cell_label": "{family}:{member}"}
```

</details>

<!-- part HCv2-7(vi) -->
**Part `HCv2-7(vi)`** (reported; label: none; predicted: none (reported part))

- Text: reported: W_NAS_common_input_control (an easy null), ABSENT rates
- Rule: `describe`
- Parameters: `{}`
- Development dry run: REPORTED (reported).

<details><summary>Data selection of HCv2-7(vi)</summary>

```json
{"where": {"design": "@design.witnesses", "declaration_id": "R", "principle": "NAS", "estimator_form": "@form.primary", "family": ["@family.A", "@family.C1"], "system": ["W_NAS_common_input_control", "PC_nominal", "W_NAS_no_workspace", "W_NAS_broadcast_only"]}, "event": {"status": "ABSENT"}, "cells": ["family", "system"]}
```

</details>

#### HCv2-8: NAS non-identification under hidden inputs, restored by declaration

<!-- hypothesis HCv2-8 -->
- Tier A; labels R, C.
- Statement: "Significant" = both per-direction z > 1.645. '<= 2/20': SUPPORTED iff <= 2/20, FALSIFIED iff >= 5/20 (CP lower bound > 0.10), else INDETERMINATE; other parts three-zone.
- Development expectation: (a) H 18/20, R 0/20; (b) H 20/20 each (rank 1 20/20), R 0/20; (c) secondary H 20/20, R 0/20; primary H c 0.09, PRESENT 3/20; (d) 20/20.
- Development dry run (DEVELOPMENT - NOT A RESULT): INDETERMINATE.

<!-- part HCv2-8(a-H) -->
**Part `HCv2-8(a-H)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: N_modules_disconnected (A): significant under H in >= 80 %
- Rule: `three_zone`
- Parameters: `{"x": 0.8}`
- Requires: `{"oracle": true}`
- Development dry run: INDETERMINATE (three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8).

<details><summary>Data selection of HCv2-8(a-H)</summary>

```json
{"where": {"design": "@design.witnesses", "family": "@family.A", "principle": "NAS", "estimator_form": "@form.primary", "system": "N_modules_disconnected", "declaration_id": "H", "seed_block": "witnesses_20"}, "event": {"@nas_significant": true}}
```

</details>

<!-- part HCv2-8(a-R) -->
**Part `HCv2-8(a-R)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: N_modules_disconnected (A): significant under R in <= 2/20
- Rule: `count_at_most`
- Parameters: `{"falsify_above": 0.1, "max_rate": 0.1}`
- Requires: `{"oracle": true}`
- Development dry run: SUPPORTED (SUPPORTED iff rate <= 0.1; FALSIFIED iff CP lower > 0.1).

<details><summary>Data selection of HCv2-8(a-R)</summary>

```json
{"where": {"design": "@design.witnesses", "family": "@family.A", "principle": "NAS", "estimator_form": "@form.primary", "system": "N_modules_disconnected", "declaration_id": "R", "seed_block": "witnesses_20"}, "event": {"@nas_significant": true}}
```

</details>

<!-- part HCv2-8(b-H) -->
**Part `HCv2-8(b-H)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: staggered driver, each variant: primary significant under H in >= 80 %
- Rule: `three_zone`
- Parameters: `{"x": 0.8}`
- Requires: `{"oracle": true}`
- Development dry run: SUPPORTED (three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8).

<details><summary>Data selection of HCv2-8(b-H)</summary>

```json
{"where": {"design": "@design.adversarial", "family": "@family.A", "principle": "NAS", "system": "ADV_NAS_staggered_driver", "estimator_form": "@form.primary", "declaration_id": "H", "seed_block": "witnesses_20"}, "event": {"@nas_significant": true}, "cells": ["@variant"]}
```

</details>

<!-- part HCv2-8(b-H-rank1) -->
**Part `HCv2-8(b-H-rank1)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: staggered driver, each variant: the rank-1 descriptor significant under H in >= 80 %
- Rule: `three_zone`
- Parameters: `{"x": 0.8}`
- Requires: `{"oracle": true}`
- Development dry run: INDETERMINATE (three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8).

<details><summary>Data selection of HCv2-8(b-H-rank1)</summary>

```json
{"where": {"design": "@design.adversarial", "family": "@family.A", "principle": "NAS", "system": "ADV_NAS_staggered_driver", "estimator_form": "@form.primary", "declaration_id": "H", "seed_block": "witnesses_20"}, "event": {"@nas_rank1_significant": true}, "cells": ["@variant"]}
```

</details>

<!-- part HCv2-8(b-R) -->
**Part `HCv2-8(b-R)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: staggered driver, each variant: primary significant under R in <= 2/20
- Rule: `count_at_most`
- Parameters: `{"falsify_above": 0.1, "max_rate": 0.1}`
- Requires: `{"oracle": true}`
- Development dry run: SUPPORTED (SUPPORTED iff rate <= 0.1; FALSIFIED iff CP lower > 0.1).

<details><summary>Data selection of HCv2-8(b-R)</summary>

```json
{"where": {"design": "@design.adversarial", "family": "@family.A", "principle": "NAS", "system": "ADV_NAS_staggered_driver", "estimator_form": "@form.primary", "declaration_id": "R", "seed_block": "witnesses_20"}, "event": {"@nas_significant": true}, "cells": ["@variant"]}
```

</details>

<!-- part HCv2-8(c-H-secondary) -->
**Part `HCv2-8(c-H-secondary)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: W_NAS_no_workspace (A): secondary PRESENT under H in >= 80 %
- Rule: `three_zone`
- Parameters: `{"x": 0.8}`
- Requires: `{"oracle": true}`
- Development dry run: SUPPORTED (three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8).

<details><summary>Data selection of HCv2-8(c-H-secondary)</summary>

```json
{"where": {"design": "@design.witnesses", "family": "@family.A", "principle": "NAS", "estimator_form": "@form.nas_secondary", "system": "W_NAS_no_workspace", "declaration_id": "H", "seed_block": "witnesses_20"}, "event": {"status": "PRESENT"}}
```

</details>

<!-- part HCv2-8(c-R-secondary) -->
**Part `HCv2-8(c-R-secondary)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: W_NAS_no_workspace (A): secondary PRESENT under R in <= 2/20
- Rule: `count_at_most`
- Parameters: `{"falsify_above": 0.1, "max_rate": 0.1}`
- Requires: `{"oracle": true}`
- Development dry run: SUPPORTED (SUPPORTED iff rate <= 0.1; FALSIFIED iff CP lower > 0.1).

<details><summary>Data selection of HCv2-8(c-R-secondary)</summary>

```json
{"where": {"design": "@design.witnesses", "family": "@family.A", "principle": "NAS", "estimator_form": "@form.nas_secondary", "system": "W_NAS_no_workspace", "declaration_id": "R", "seed_block": "witnesses_20"}, "event": {"status": "PRESENT"}}
```

</details>

<!-- part HCv2-8(c-H-primary) -->
**Part `HCv2-8(c-H-primary)`** (decisive; label: C; predicted: SUPPORTED (default))

- Text: W_NAS_no_workspace (A): primary under H not PRESENT in >= 80 % (blind by geometry: the declared construct limitation)
- Rule: `three_zone`
- Parameters: `{"x": 0.8}`
- Requires: `{"oracle": true}`
- Development dry run: SUPPORTED (three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8).

<details><summary>Data selection of HCv2-8(c-H-primary)</summary>

```json
{"where": {"design": "@design.witnesses", "family": "@family.A", "principle": "NAS", "estimator_form": "@form.primary", "system": "W_NAS_no_workspace", "declaration_id": "H", "seed_block": "witnesses_20"}, "event": {"status": {"ne": "PRESENT"}}}
```

</details>

<!-- part HCv2-8(d) -->
**Part `HCv2-8(d)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: C1: |c_H - c_R| < z in >= 80 % for W_NAS_no_workspace and N_modules_disconnected
- Rule: `three_zone`
- Parameters: `{"x": 0.8}`
- Requires: `{"oracle": true}`
- Development dry run: SUPPORTED (three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8).

<details><summary>Data selection of HCv2-8(d)</summary>

```json
{"where": {"design": "@design.witnesses", "family": "@family.C1", "principle": "NAS", "estimator_form": "@form.primary", "system": ["W_NAS_no_workspace", "N_modules_disconnected"], "seed_block": "witnesses_20"}, "pair": {"on": ["family", "system", "seed", "principle", "estimator_form"], "a": {"declaration_id": "H"}, "b": {"declaration_id": "R"}}, "event": {"delta": {"abs_lt": 0.25}}, "cells": ["system"]}
```

</details>

#### HCv2-9: Declared-input misspecification, NAS and IIM (held out)

<!-- hypothesis HCv2-9 -->
- Tier A; labels HO, R.
- Statement: Part-wise; SUPPORTED iff all decisive parts hold. A failure of (a)-(c) means the declaration is more robust than predicted; a failure of (d)-(e) that the basis covers more than its stated span.
- Development dry run (DEVELOPMENT - NOT A RESULT): NOT_EVALUABLE.

<!-- part HCv2-9(a) -->
**Part `HCv2-9(a)`** (decisive; label: HO; predicted: SUPPORTED (default))

- Text: label errors: for each (estimator, system) pair the excess increases over q in {0, 0.1, 0.25} (Jonckheere-Terpstra one-sided p < 0.05/4) and the median at q = 0.25 exceeds the median at q = 0
- Rule: `jonckheere_terpstra`
- Parameters: `{"alpha": 0.0125, "medians": "last_gt_first", "order": ["R", "Q10", "Q25"]}`
- Requires: `{"oracle": true}`
- Development dry run: NOT_EVALUABLE (Jonckheere-Terpstra one-sided p < 0.0125 and medians last_gt_first).

<details><summary>Data selection of HCv2-9(a)</summary>

```json
{"where": {"design": ["@design.witnesses", "@design.held_out"], "family": "@family.A", "estimator_form": "@form.primary", "all": [{"any": [{"principle": "NAS", "system": ["W_NAS_no_workspace", "N_modules_disconnected"]}, {"principle": "IIM", "system": ["W_IIM_feedforward", "O_inert"]}]}], "declaration_id": ["R", "Q10", "Q25"], "seed_block": "witnesses_20"}, "group": "declaration_id", "value": "c", "cells": ["system", "principle"]}
```

</details>

<!-- part HCv2-9(b) -->
**Part `HCv2-9(b)`** (decisive; label: HO; predicted: SUPPORTED (default))

- Text: slow phase undeclared (P): paired Delta c(P - R) > 0, one-sided Wilcoxon signed-rank p < 0.05/4, per pair
- Rule: `wilcoxon`
- Parameters: `{"alpha": 0.0125, "alternative": "greater"}`
- Requires: `{"oracle": true}`
- Notes: IIM pairs are labelled R (replication of the IIM design's label-error dose)
- Development dry run: NOT_EVALUABLE (no input rows).

<details><summary>Data selection of HCv2-9(b)</summary>

```json
{"where": {"design": ["@design.witnesses", "@design.held_out"], "family": "@family.A", "estimator_form": "@form.primary", "all": [{"any": [{"principle": "NAS", "system": ["W_NAS_no_workspace", "N_modules_disconnected"]}, {"principle": "IIM", "system": ["W_IIM_feedforward", "O_inert"]}]}], "seed_block": "witnesses_20"}, "pair": {"on": ["family", "system", "seed", "principle", "estimator_form"], "a": {"declaration_id": "P"}, "b": {"declaration_id": "R"}}, "cells": ["system", "principle"]}
```

</details>

<!-- part HCv2-9(c) -->
**Part `HCv2-9(c)`** (decisive; label: HO; predicted: SUPPORTED (default))

- Text: cue-onset jitter (J): paired Delta c(J - R) > 0, one-sided sign test p < 0.05/4, per pair
- Rule: `sign_test`
- Parameters: `{"alpha": 0.0125, "alternative": "greater"}`
- Requires: `{"oracle": true}`
- Development dry run: NOT_EVALUABLE (no input rows).

<details><summary>Data selection of HCv2-9(c)</summary>

```json
{"where": {"design": ["@design.witnesses", "@design.held_out"], "family": "@family.A", "estimator_form": "@form.primary", "all": [{"any": [{"principle": "NAS", "system": ["W_NAS_no_workspace", "N_modules_disconnected"]}, {"principle": "IIM", "system": ["W_IIM_feedforward", "O_inert"]}]}], "seed_block": "witnesses_20"}, "pair": {"on": ["family", "system", "seed", "principle", "estimator_form"], "a": {"declaration_id": "J"}, "b": {"declaration_id": "R"}}, "cells": ["system", "principle"]}
```

</details>

<!-- part HCv2-9(d) -->
**Part `HCv2-9(d)`** (decisive; label: HO; predicted: SUPPORTED (default))

- Text: driver outside the basis span (time constants x 10): NAS primary significant under R in >= 50 % of seeds (three-zone at 0.5)
- Rule: `three_zone`
- Parameters: `{"x": 0.5}`
- Requires: `{"oracle": true}`
- Development expectation: P(NOT_EVALUABLE, reason ORACLE) about 0.74: the realisation check driver_reaches_every_module of ADV_NAS_staggered_tau10 passed 34/40 in development (CD-13, HCv2-0). The construct prediction (NAS primary significant under R in >= 50 % of seeds) is unchanged.
- Development dry run: NOT_EVALUABLE (no input rows).

<details><summary>Data selection of HCv2-9(d)</summary>

```json
{"where": {"design": ["@design.held_out", "@design.adversarial"], "family": "@family.A", "system": "ADV_NAS_staggered_tau10", "principle": "NAS", "declaration_id": "R", "estimator_form": "@form.primary", "seed_block": "witnesses_20"}, "event": {"@nas_significant": true}}
```

</details>

<!-- part HCv2-9(e) -->
**Part `HCv2-9(e)`** (decisive; label: HO; predicted: SUPPORTED (default))

- Text: saturating transform: paired NAS excess (saturating minus hierarchical linear) > 0 under R, one-sided sign test p < 0.05
- Rule: `sign_test`
- Parameters: `{"alpha": 0.05, "alternative": "greater"}`
- Requires: `{"oracle": true}`
- Development dry run: NOT_EVALUABLE (no input rows).

<details><summary>Data selection of HCv2-9(e)</summary>

```json
{"where": {"design": ["@design.held_out", "@design.adversarial"], "family": "@family.A", "principle": "NAS", "declaration_id": "R", "estimator_form": "@form.primary", "seed_block": "witnesses_20"}, "pair": {"on": ["family", "seed", "declaration_id", "principle", "estimator_form"], "a": {"system": "ADV_NAS_staggered_sat"}, "b": {"system": "ADV_NAS_staggered_driver", "@variant": "hierarchical"}, "value": "excess"}}
```

</details>

<!-- part HCv2-9(f) -->
**Part `HCv2-9(f)`** (reported; label: none; predicted: none (reported part))

- Text: reported: PC_nominal under each misspecification
- Rule: `describe`
- Parameters: `{}`
- Development dry run: REPORTED (reported).

<details><summary>Data selection of HCv2-9(f)</summary>

```json
{"where": {"design": ["@design.witnesses", "@design.held_out"], "family": "@family.A", "system": "PC_nominal", "principle": ["NAS", "IIM"], "estimator_form": "@form.primary", "declaration_id": ["R", "P", "Q10", "Q25", "J"], "seed_block": "witnesses_20"}, "event": {"status": "PRESENT"}, "cells": ["principle", "declaration_id"]}
```

</details>

<!-- part HCv2-9(f-conditioning) -->
**Part `HCv2-9(f-conditioning)`** (reported; label: none; predicted: none (reported part))

- Text: reported: NAS conditioning_delta (value with the declared basis minus without, same seeds) per system and declaration
- Rule: `describe`
- Parameters: `{}`
- Development dry run: REPORTED (reported).

<details><summary>Data selection of HCv2-9(f-conditioning)</summary>

```json
{"where": {"design": ["@design.witnesses", "@design.held_out", "@design.adversarial"], "family": "@family.A", "principle": "NAS", "estimator_form": "@form.primary", "declaration_id": ["R", "P", "Q10", "Q25", "J"], "seed_block": "witnesses_20"}, "union": [{"label": "receive", "value": "@nas_conditioning_delta", "cell_label": "{system}:{declaration_id}|receive"}, {"label": "return", "value": "@nas_conditioning_delta_return", "cell_label": "{system}:{declaration_id}|return"}]}
```

</details>

#### HCv2-10: NAS forward-model admission (Hopf arm, held-out regime)

<!-- hypothesis HCv2-10 -->
- Tier A; labels R, C.
- Statement: The admission procedure yields the predicted NAS rows: source admitted for PRESENT and for ABSENT; every EEG and source-estimate view not admitted for PRESENT (fails FMa), with admitted_for_absent = vacuous. BOLD: IA-3.
- Development dry run (DEVELOPMENT - NOT A RESULT): NOT_EVALUABLE.

<!-- part HCv2-10 -->
**Part `HCv2-10`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: decisive cells: source PRESENT admission and each EEG view's PRESENT admission; NOT_EVALUABLE if the source anchor is invalid
- Rule: `admission_matches`
- Parameters: `{"anchor_cell": "@view.source", "anchor_field": "anchor_valid", "predictions": {"@view.eeg64": {"admitted_for_present": "no"}, "@view.eeg_low": {"admitted_for_present": "no"}, "@view.mne_template": {"admitted_for_present": "no"}, "@view.source": {"admitted_for_present": "yes"}}}`
- Development dry run: NOT_EVALUABLE (no input rows).

<details><summary>Data selection of HCv2-10</summary>

```json
{"source": "registry", "where": {"principle": "NAS", "arm": "@arm.hopf"}, "cell_label": "{view}"}
```

</details>

<!-- part HCv2-10(absent) -->
**Part `HCv2-10(absent)`** (reported; label: none; predicted: none (reported part))

- Text: reported: the ABSENT flags against their predictions
- Rule: `admission_matches`
- Parameters: `{"predictions": {"@view.eeg64": {"admitted_for_absent": "vacuous"}, "@view.eeg_low": {"admitted_for_absent": "vacuous"}, "@view.mne_template": {"admitted_for_absent": "vacuous"}, "@view.source": {"admitted_for_absent": "yes"}}}`
- Development dry run: NOT_EVALUABLE (no input rows).

<details><summary>Data selection of HCv2-10(absent)</summary>

```json
{"source": "registry", "where": {"principle": "NAS", "arm": "@arm.hopf"}, "cell_label": "{view}"}
```

</details>

#### HCv2-11: IIM directional construct on family B (H-IIM-4)

<!-- hypothesis HCv2-11 -->
- Tier A; labels C.
- Development expectation: T = 10000, 10 seeds, v1 rule: IIM-dir ABSENT 10/10, PRESENT 0/10 at 0.2, 0.6, 1.0; IIM-bid c 0.04/0.40/1.09 vs exact 0.04/0.41/1.12.
- Development dry run (DEVELOPMENT - NOT A RESULT): SUPPORTED.

<!-- part HCv2-11(a) -->
**Part `HCv2-11(a)`** (decisive; label: C; predicted: SUPPORTED (default))

- Text: IIM-dir PRESENT <= 0.07 (H0 cell rule, 12 cells)
- Rule: `h0_cell`
- Parameters: `{"alpha": 0.05, "bound": 0.07}`
- Development dry run: SUPPORTED (every pooled CP upper bound at 0.05/1 is below 0.07).

<details><summary>Data selection of HCv2-11(a)</summary>

```json
{"where": {"design": "@design.family_b", "family": "@family.B", "principle": "IIM", "@b_network": "feedforward_star", "seed_block": "family_b_feedforward_star", "estimator_form": "@form.primary"}, "event": {"status": "PRESENT"}, "cell_label": "c{@b_coupling}:T{@b_T}"}
```

</details>

<!-- part HCv2-11(b-rho) -->
**Part `HCv2-11(b-rho)`** (decisive; label: C; predicted: SUPPORTED (default))

- Text: IIM-bid Spearman(c, coupling) > 0 (one-sided p < 0.025) at each T
- Rule: `spearman_ols`
- Parameters: `{"alpha": 0.025, "require_positive_slope": false}`
- Development dry run: SUPPORTED (Spearman one-sided p < 0.025).

<details><summary>Data selection of HCv2-11(b-rho)</summary>

```json
{"where": {"design": "@design.family_b", "family": "@family.B", "principle": "IIM", "@b_network": "feedforward_star", "seed_block": "family_b_feedforward_star", "estimator_form": "@form.iim_bidirectional"}, "dose": "@b_coupling", "value": "c", "cell_label": "T{@b_T}"}
```

</details>

<!-- part HCv2-11(b-present) -->
**Part `HCv2-11(b-present)`** (decisive; label: C; predicted: SUPPORTED (default))

- Text: IIM-bid PRESENT >= 80 % at coupling 1.0, T = 30000 (three-zone)
- Rule: `three_zone`
- Parameters: `{"x": 0.8}`
- Development dry run: SUPPORTED (three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8).

<details><summary>Data selection of HCv2-11(b-present)</summary>

```json
{"where": {"design": "@design.family_b", "family": "@family.B", "principle": "IIM", "@b_network": "feedforward_star", "seed_block": "family_b_feedforward_star", "estimator_form": "@form.iim_bidirectional", "@b_coupling": 1.0, "@b_T": 30000}, "event": {"status": "PRESENT"}}
```

</details>

<!-- part HCv2-11(c) -->
**Part `HCv2-11(c)`** (decisive; label: C; predicted: SUPPORTED (default))

- Text: IIM-dir ABSENT >= 80 % at every coupling >= 0.2 and both T (three-zone at 0.05/10); decisive only if pi0 >= 0.9 at CD-7, else NOT_TESTABLE_BY_DESIGN and replaced by (a)
- Rule: `three_zone`
- Parameters: `{"m": 10, "x": 0.8}`
- Gate: `{"family": "@protocol.B", "kind": "absent", "principle": "IIM", "witness": "feedforward_star"}`
- Replaced by: `HCv2-11(a)`
- Development dry run: SUPPORTED (three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8).

<details><summary>Data selection of HCv2-11(c)</summary>

```json
{"where": {"design": "@design.family_b", "family": "@family.B", "principle": "IIM", "@b_network": "feedforward_star", "seed_block": "family_b_feedforward_star", "estimator_form": "@form.primary", "@b_coupling": {"ge": 0.2}}, "event": {"status": "ABSENT"}, "cell_label": "c{@b_coupling}:T{@b_T}"}
```

</details>

#### HCv2-12: IIM family-B ground truth: exact tracking, the non-monotone regime and the occupancy gate

<!-- hypothesis HCv2-12 -->
- Tier A; labels C, HO.
- Development dry run (DEVELOPMENT - NOT A RESULT): INDETERMINATE.

<!-- part HCv2-12(a) -->
**Part `HCv2-12(a)`** (decisive; label: C; predicted: SUPPORTED (default))

- Text: per (system, cut): Spearman(sampled excess, exact value) > 0 at one-sided p < 0.05/4 and level medians non-decreasing in the exact value (decreases < 0.10 of the family-B anchor count as ties)
- Rule: `spearman_monotone`
- Parameters: `{"alpha": 0.0125, "tie_tolerance": 0.1}`
- Development dry run: SUPPORTED (Spearman(sampled, exact) one-sided p < 0.0125 and level medians non-decreasing in the exact value (tolerance 0.1)).

<details><summary>Data selection of HCv2-12(a)</summary>

```json
{"where": {"design": "@design.family_b", "family": "@family.B", "principle": "IIM", "seed_block": "family_b_exact_tracking", "@b_T": 30000, "estimator_form": ["@form.primary", "@form.iim_bidirectional"]}, "union": [{"label": "ring", "where": {"@b_network": "ring"}, "group": "@b_coupling"}, {"label": "xor", "where": {"@b_network": "xor_loop"}, "group": "@b_noise"}], "dose": "@b_exact", "value": "c", "cell_label": "{member}:{estimator_form}"}
```

</details>

<!-- part HCv2-12(a-T10000) -->
**Part `HCv2-12(a-T10000)`** (reported; label: none; predicted: none (reported part))

- Text: reported: the same at T = 10000
- Rule: `describe`
- Parameters: `{}`
- Development dry run: REPORTED (reported).

<details><summary>Data selection of HCv2-12(a-T10000)</summary>

```json
{"where": {"design": "@design.family_b", "family": "@family.B", "principle": "IIM", "seed_block": "family_b_exact_tracking", "@b_T": 10000}, "cells": ["@b_network", "estimator_form"]}
```

</details>

<!-- part HCv2-12(b) -->
**Part `HCv2-12(b)`** (decisive; label: HO; predicted: SUPPORTED (default))

- Text: non-monotone regime (ring 0.45 -> 0.9 -> 1.5, T = 30000): at each step where the gate defines IIM in >= 50 % of runs, the median paired change has the sign of the exact change (one-sided sign test p < 0.025); a step where the gate fails in > 50 % counts as correctly undefined
- Rule: `stepwise_sign`
- Parameters: `{"alpha": 0.025, "gate_share": 0.5, "pair_on": ["seed"], "steps": [[0.45, 0.9], [0.9, 1.5]]}`
- Development expectation: The exact change (CD-1 constants) is negative from 0.45 to 0.9 in both cut modes (bidirectional 0.04513 -> 0.03621; directional 0.02909 -> -0.00973) and, from 0.9 to 1.5, positive for the bidirectional form (-> 0.04014) and negative for the directional form (-> -0.03161). The rule tests the sign of the exact change at each step; the earlier prose 'never a rising estimate' contradicted the exact bidirectional rise and was a logical error. The rule and its thresholds are unchanged. The operating characteristic is synthetic, from the exact steps and the SE of the development ring cells at couplings <= 0.5; no estimator ran at ring 0.9 or 1.5 (HO-5).
- Development dry run: NOT_EVALUABLE (each step: correct sign (sign test) or correctly undefined).

<details><summary>Data selection of HCv2-12(b)</summary>

```json
{"where": {"design": "@design.family_b", "family": "@family.B", "principle": "IIM", "seed_block": "family_b_non_monotone", "@b_network": "ring", "@b_coupling": [0.45, 0.9, 1.5], "@b_T": 30000, "estimator_form": ["@form.primary", "@form.iim_bidirectional"]}, "group": "@b_coupling", "exact": "@b_exact", "value": "c", "cells": ["estimator_form"]}
```

</details>

<!-- part HCv2-12(c-low) -->
**Part `HCv2-12(c-low)`** (decisive; label: C; predicted: SUPPORTED (default))

- Text: cells with T pi_min <= N_min / 2 are UNDEFINED(INSUFFICIENT_OCCUPANCY) in >= 90 %
- Rule: `three_zone`
- Parameters: `{"x": 0.9}`
- Development dry run: SUPPORTED (three-zone at 0.9: SUPPORTED iff rate >= 0.9, FALSIFIED iff CP upper < 0.9).

<details><summary>Data selection of HCv2-12(c-low)</summary>

```json
{"where": {"design": "@design.family_b", "family": "@family.B", "principle": "IIM", "seed_block": "family_b_occupancy_gate", "@b_occupancy_cell": "undefined"}, "event": {"reason_code": "INSUFFICIENT_OCCUPANCY"}, "cell_label": "low:{@b_network}:c{@b_coupling}:T{@b_T}:{estimator_form}"}
```

</details>

<!-- part HCv2-12(c-high) -->
**Part `HCv2-12(c-high)`** (decisive; label: C; predicted: SUPPORTED (default))

- Text: cells with T pi_min >= 4 N_min are defined in >= 90 %
- Rule: `three_zone`
- Parameters: `{"x": 0.9}`
- Development dry run: SUPPORTED (three-zone at 0.9: SUPPORTED iff rate >= 0.9, FALSIFIED iff CP upper < 0.9).

<details><summary>Data selection of HCv2-12(c-high)</summary>

```json
{"where": {"design": "@design.family_b", "family": "@family.B", "principle": "IIM", "seed_block": "family_b_occupancy_gate", "@b_occupancy_cell": "defined"}, "event": {"defined": true}, "cell_label": "high:{@b_network}:c{@b_coupling}:T{@b_T}:{estimator_form}"}
```

</details>

<!-- part HCv2-12(d) -->
**Part `HCv2-12(d)`** (decisive; label: C; predicted: SUPPORTED (default))

- Text: the declared reason is correct (INSUFFICIENT_OCCUPANCY; MACRO_RANK_DEFICIENT for the montage) in >= 90 % (three-zone at 0.05/3)
- Rule: `three_zone`
- Parameters: `{"m": 3, "x": 0.9}`
- Development dry run: INDETERMINATE (three-zone at 0.9: SUPPORTED iff rate >= 0.9, FALSIFIED iff CP upper < 0.9).

<details><summary>Data selection of HCv2-12(d)</summary>

```json
{"union": [{"label": "O_hypersynchronous", "where": {"design": "@design.witnesses", "estimator_form": "@form.primary", "family": "@family.A", "system": "O_hypersynchronous", "principle": "IIM", "declaration_id": "R"}, "event": {"reason_code": "INSUFFICIENT_OCCUPANCY"}}, {"label": "all_to_all_0.6_T1000", "where": {"design": "@design.family_b", "family": "@family.B", "principle": "IIM", "seed_block": "family_b_occupancy_gate", "@b_network": "all_to_all", "@b_coupling": 0.6, "@b_T": 1000}, "event": {"reason_code": "INSUFFICIENT_OCCUPANCY"}}, {"label": "v1_quadrant_average_reference", "where": {"design": "@design.whole_brain", "principle": "IIM", "@hopf_G": 0, "view": "@view.eeg64", "estimator_form": "@form.iim_v1_quadrant"}, "event": {"reason_code": "MACRO_RANK_DEFICIENT"}}], "cell_label": "{member}"}
```

</details>

#### HCv2-13: IIM driver conditioning on family B (H-IIM-5 a-c)

<!-- hypothesis HCv2-13 -->
- Tier A; labels C.
- Development expectation: Recorded c -0.03/-0.01/-0.00, exceedance <= 0.05; undeclared c 0.69/0.91/1.10; label errors c 0 -> 0.27 -> 0.67.
- Development dry run (DEVELOPMENT - NOT A RESULT): INDETERMINATE.

<!-- part HCv2-13(a) -->
**Part `HCv2-13(a)`** (decisive; label: C; predicted: SUPPORTED (default))

- Text: recorded and stratified: rank exceedance and PRESENT <= 0.07 (H0 cell rule over 12 T x cut x statistic cells)
- Rule: `h0_cell`
- Parameters: `{"alpha": 0.05, "bound": 0.07}`
- Development dry run: INDETERMINATE (a pooled upper bound is not below the bound).

<details><summary>Data selection of HCv2-13(a)</summary>

```json
{"where": {"design": "@design.family_b", "family": "@family.B", "principle": "IIM", "seed_block": "family_b_driver_conditioning", "@b_network": "hidden_driver", "estimator_form": ["@form.primary", "@form.iim_bidirectional"], "declaration_id": "recorded"}, "union": [{"label": "exceedance", "event": {"@iim_p_ind": {"le": 0.05}}}, {"label": "present", "event": {"status": "PRESENT"}}], "cell_label": "{member}:T{@b_T}:{estimator_form}", "pool_by": ["member"]}
```

</details>

<!-- part HCv2-13(b-present) -->
**Part `HCv2-13(b-present)`** (decisive; label: C; predicted: SUPPORTED (default))

- Text: undeclared (the limit): PRESENT >= 80 % at T = 30000 in each cut family (three-zone at 0.05/2)
- Rule: `three_zone`
- Parameters: `{"m": 2, "x": 0.8}`
- Development dry run: SUPPORTED (three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8).

<details><summary>Data selection of HCv2-13(b-present)</summary>

```json
{"where": {"design": "@design.family_b", "family": "@family.B", "principle": "IIM", "seed_block": "family_b_driver_conditioning", "@b_network": "hidden_driver", "estimator_form": ["@form.primary", "@form.iim_bidirectional"], "declaration_id": "hidden", "@b_T": 30000}, "event": {"status": "PRESENT"}, "cells": ["estimator_form"]}
```

</details>

<!-- part HCv2-13(b-median) -->
**Part `HCv2-13(b-median)`** (decisive; label: C; predicted: SUPPORTED (default))

- Text: undeclared: median c > z at every T
- Rule: `median_threshold`
- Parameters: `{"op": "gt", "x": 0.25}`
- Development dry run: SUPPORTED (median gt 0.25).

<details><summary>Data selection of HCv2-13(b-median)</summary>

```json
{"where": {"design": "@design.family_b", "family": "@family.B", "principle": "IIM", "seed_block": "family_b_driver_conditioning", "@b_network": "hidden_driver", "estimator_form": ["@form.primary", "@form.iim_bidirectional"], "declaration_id": "hidden"}, "value": "c", "cell_label": "T{@b_T}:{estimator_form}"}
```

</details>

<!-- part HCv2-13(c) -->
**Part `HCv2-13(c)`** (decisive; label: C; predicted: SUPPORTED (default))

- Text: label errors: Jonckheere-Terpstra one-sided p < 0.025 per cut and strictly increasing medians
- Rule: `jonckheere_terpstra`
- Parameters: `{"alpha": 0.025, "medians": "strictly_increasing", "order": ["0", "0.1", "0.25"]}`
- Development dry run: SUPPORTED (Jonckheere-Terpstra one-sided p < 0.025 and medians strictly_increasing).

<details><summary>Data selection of HCv2-13(c)</summary>

```json
{"where": {"design": "@design.family_b", "family": "@family.B", "principle": "IIM", "seed_block": "family_b_driver_conditioning", "@b_network": "hidden_driver", "estimator_form": ["@form.primary", "@form.iim_bidirectional"], "@b_T": 10000}, "union": [{"label": "0", "where": {"declaration_id": "recorded"}}, {"label": "0.1", "where": {"declaration_id": "label_error", "@b_label_q": 0.1}}, {"label": "0.25", "where": {"declaration_id": "label_error", "@b_label_q": 0.25}}], "group": "member", "value": "c", "cells": ["estimator_form"]}
```

</details>

#### HCv2-14: IIM-dir on the agents with recorded drivers, and the hidden-driver limit

<!-- hypothesis HCv2-14 -->
- Tier A; labels R, HO, C; successor of HC3.
- Development expectation: Development (CD-6; family A, reference block 900-939 and witness seeds 320-331 and 340-351; block bootstrap SE at the decided se_df 9, df 12 in brackets). Under R: median paired Delta c(PC - W_IIM_feedforward) 0.90 over 52 pairs, 49 >= 0.5, Hodges-Lehmann one-sided 95 % lower bound 0.91; PC_nominal PRESENT 58/64 = 0.906 (60/64), so (d) is decisive by one row; W_IIM_feedforward PRESENT 0/52 and ABSENT 12/52 (13/52), so (c) stays NOT_TESTABLE_BY_DESIGN (pi0 0.23); O_inert PRESENT 0/12; Spearman rho of c with c_int 0.09 (40 runs, one-sided p 0.28; an inverted U, median c 0.97 at c_int 0.53 and 0.06 at 1.2), so (f) would be expected FALSIFIED at 10 seeds per level; on the resized c_int sweep (65 seeds per level, CD-11) P(SUPPORTED | development) is 0.82, see (f). Under H: median Delta c -0.61 over 47 pairs, consistent with (e); PRESENT PC_nominal 24/64, W_IIM_feedforward 29/52, O_inert 11/12 (HCv2-14(H-present)).
- Notes: Family A parts are R; family C1 parts are held out (HO-4). C1 disclosure: the C1 predictions were committed on 2026-10-05 (ac13c2d; file last changed 2026-10-06, d3bcb09). The C1 reference blocks, which HO-4 permits, were computed on 2026-10-07: median Delta c 0.76 under C1-R (33/40 >= 0.5) and 0.82 under C1-H (32/40), contrary to the frozen limit prediction (e); PC PRESENT 0.40 (C1-R) and 0.25 (C1-H), so (d) on C1 is NOT_TESTABLE_BY_DESIGN and replaced by (a). The C1 predictions and thresholds are unchanged, and no testability or seed-count decision on these parts used these values.
- Development dry run (DEVELOPMENT - NOT A RESULT): FALSIFIED.

<!-- part HCv2-14(a) -->
**Part `HCv2-14(a)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: under R, median paired Delta c(PC - W_IIM_feedforward) >= 0.5 with one-sided 95 % Hodges-Lehmann lower bound > z, each family
- Rule: `hodges_lehmann`
- Parameters: `{"alpha": 0.05, "lower_gt": 0.25, "median_op": "ge", "median_x": 0.5}`
- Requires: `{"oracle": true}`
- Development dry run: SUPPORTED (median ge 0.5 and one-sided 0.95 Hodges-Lehmann lower bound > 0.25).

<details><summary>Data selection of HCv2-14(a)</summary>

```json
{"where": {"design": "@design.witnesses", "principle": "IIM", "estimator_form": "@form.primary", "family": ["@family.A", "@family.C1"], "declaration_id": "R"}, "pair": {"on": ["family", "seed", "declaration_id", "principle", "estimator_form"], "a": {"system": "PC_nominal"}, "b": {"system": "W_IIM_feedforward"}, "value": "c"}, "cells": ["family"]}
```

</details>

<!-- part HCv2-14(b) -->
**Part `HCv2-14(b)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: under R, W_IIM_feedforward and O_inert PRESENT <= 0.07 (H0 cell rule, m = 4)
- Rule: `h0_cell`
- Parameters: `{"alpha": 0.05, "bound": 0.07, "m": 4}`
- Requires: `{"oracle": true}`
- Development dry run: INDETERMINATE (a pooled upper bound is not below the bound).

<details><summary>Data selection of HCv2-14(b)</summary>

```json
{"where": {"design": "@design.witnesses", "principle": "IIM", "estimator_form": "@form.primary", "family": ["@family.A", "@family.C1"], "declaration_id": "R", "system": ["W_IIM_feedforward", "O_inert"]}, "event": {"status": "PRESENT"}, "cell_label": "{family}:{system}"}
```

</details>

<!-- part HCv2-14(c) -->
**Part `HCv2-14(c)`** (decisive; label: R; predicted: NOT_TESTABLE_BY_DESIGN (stated))

- Text: W_IIM_feedforward ABSENT in >= 80 % under R: predicted NOT_TESTABLE_BY_DESIGN (pi0 = 0 at bench precision), replaced by (b)
- Rule: `three_zone`
- Parameters: `{"x": 0.8}`
- Requires: `{"oracle": true}`
- Gate: `{"family": "{protocol_id}", "kind": "absent", "per_cell": true, "principle": "IIM", "witness": "W_IIM_feedforward"}`
- Replaced by: `HCv2-14(b)`
- Development dry run: NOT_TESTABLE_BY_DESIGN (every cell is below the gate (pi < 0.9)).

<details><summary>Data selection of HCv2-14(c)</summary>

```json
{"where": {"design": "@design.witnesses", "principle": "IIM", "estimator_form": "@form.primary", "family": ["@family.A", "@family.C1"], "declaration_id": "R", "system": "W_IIM_feedforward"}, "event": {"status": "ABSENT"}, "cell_label": "{protocol_id}"}
```

</details>

<!-- part HCv2-14(d) -->
**Part `HCv2-14(d)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: PC PRESENT >= 80 % under R where pi1 >= 0.9, else replaced by (a)
- Rule: `three_zone`
- Parameters: `{"x": 0.8}`
- Gate: `{"family": "{protocol_id}", "kind": "present", "per_cell": true, "principle": "IIM", "witness": "PC_nominal"}`
- Replaced by: `HCv2-14(a)`
- Development dry run: SUPPORTED (three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8).

<details><summary>Data selection of HCv2-14(d)</summary>

```json
{"where": {"design": "@design.witnesses", "principle": "IIM", "estimator_form": "@form.primary", "family": ["@family.A", "@family.C1"], "declaration_id": "R", "system": "PC_nominal"}, "event": {"status": "PRESENT"}, "cell_label": "{protocol_id}"}
```

</details>

<!-- part HCv2-14(e) -->
**Part `HCv2-14(e)`** (decisive; label: C; predicted: SUPPORTED (default))

- Text: the limit, decisive: under H, median Delta c(PC - W_IIM_feedforward) < 0.25
- Rule: `median_threshold`
- Parameters: `{"op": "lt", "x": 0.25}`
- Requires: `{"oracle": true}`
- Development dry run: SUPPORTED (median lt 0.25).

<details><summary>Data selection of HCv2-14(e)</summary>

```json
{"where": {"design": "@design.witnesses", "principle": "IIM", "estimator_form": "@form.primary", "family": ["@family.A", "@family.C1"], "declaration_id": "H"}, "pair": {"on": ["family", "seed", "declaration_id", "principle", "estimator_form"], "a": {"system": "PC_nominal"}, "b": {"system": "W_IIM_feedforward"}, "value": "c"}, "cells": ["family"]}
```

</details>

<!-- part HCv2-14(f) -->
**Part `HCv2-14(f)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: under R, Spearman rho(c_IIM-dir, c_int) > 0 at one-sided p < 0.05
- Rule: `spearman_ols`
- Parameters: `{"alpha": 0.05, "require_positive_slope": false}`
- Requires: `{"usable": ["c_int"]}`
- Notes: Seed count (CD-11; seeds resized, never thresholds). The family-A c_int sweep runs its 10 levels on 65 seeds each (20000-20064; 10 before); the g_b and K sweeps stay on 10. The levels share their seeds and the null seeds derived from them, and the development values carry a seed effect, so the operating characteristic resamples whole seed profiles: P(SUPPORTED | development) is 0.16 at 10 seeds per level and 0.82 at 65 (0.64 at 45, 0.85 at 70), and P(FALSIFIED | the development effect) is 0.18. The development effect rests on 4 seeds per level and rises then falls with c_int (rho 0.09 over 40 runs). The C1 cell is held out (HO-4), so its factor cannot be computed and the part's P(SUPPORTED) is at most this value; the C1 c_int sweep runs on the same 65 seeds per level, by symmetry and without any C1 IIM value.
- Development dry run: FALSIFIED (Spearman one-sided p < 0.05).

<details><summary>Data selection of HCv2-14(f)</summary>

```json
{"where": {"design": "@design.sweep", "principle": "IIM", "estimator_form": "@form.primary", "family": ["@family.A", "@family.C1"], "declaration_id": "R", "@sweep_knob": "c_int"}, "dose": "@sweep_level", "value": "c", "cells": ["family"]}
```

</details>

<!-- part HCv2-14(H-present) -->
**Part `HCv2-14(H-present)`** (reported; label: none; predicted: none (reported part))

- Text: reported: IIM PRESENT rate under H on O_inert and W_IIM_feedforward (family A; with the drivers hidden a system with every mechanism off can read as integrated)
- Rule: `describe`
- Parameters: `{}`
- Development dry run: REPORTED (reported).

<details><summary>Data selection of HCv2-14(H-present)</summary>

```json
{"where": {"design": "@design.witnesses", "principle": "IIM", "estimator_form": "@form.primary", "family": "@family.A", "declaration_id": "H", "system": ["O_inert", "W_IIM_feedforward"]}, "event": {"status": "PRESENT"}, "cell_label": "{family}:{system}"}
```

</details>

#### HCv2-15: IIM at sensor and BOLD level: specificity, exclusion safety, sensitivity (Hopf arm, held-out regime)

<!-- hypothesis HCv2-15 -->
- Tier A; labels R, C, HO.
- Development dry run (DEVELOPMENT - NOT A RESULT): FALSIFIED.

<!-- part HCv2-15(a) -->
**Part `HCv2-15(a)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: FMa: v2 pipeline PRESENT and rank exceedance at G = 0 with CP upper bound < 0.07 at 0.05/4 in each view (61 sources)
- Rule: `demonstration`
- Parameters: `{"bound": 0.07, "level": 0.0125}`
- Development dry run: FALSIFIED (CP upper bound at 0.0125 below 0.07).

<details><summary>Data selection of HCv2-15(a)</summary>

```json
{"where": {"design": "@design.whole_brain", "principle": "IIM", "@hopf_G": 0, "view": "@view.sensor_set", "estimator_form": "@form.primary", "all": [{"@unlesioned": true}]}, "union": [{"label": "present", "event": {"status": "PRESENT"}}, {"label": "exceedance", "event": {"@iim_p_ind": {"le": 0.05}}}], "cell_label": "{view}:{member}"}
```

</details>

<!-- part HCv2-15(b) -->
**Part `HCv2-15(b)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: v1 quadrant pipeline: PRESENT at G = 0 in >= 80 % without reference (three-zone)
- Rule: `three_zone`
- Parameters: `{"x": 0.8}`
- Development dry run: FALSIFIED (three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8).

<details><summary>Data selection of HCv2-15(b)</summary>

```json
{"where": {"design": "@design.whole_brain", "principle": "IIM", "@hopf_G": 0, "view": "@view.eeg64_noref", "estimator_form": "@form.iim_v1_quadrant"}, "event": {"status": "PRESENT"}}
```

</details>

<!-- part HCv2-15(c) -->
**Part `HCv2-15(c)`** (decisive; label: C; predicted: SUPPORTED (default))

- Text: FMabs: IIM-dir false ABSENT at G_nom with CP upper bound < 0.02 in every view (150 runs, curtailed)
- Rule: `demonstration`
- Parameters: `{"bound": 0.02, "level": 0.05, "n_planned": 150}`
- Development dry run: NOT_EVALUABLE (no input rows).

<details><summary>Data selection of HCv2-15(c)</summary>

```json
{"where": {"design": "@design.whole_brain", "principle": "IIM", "@hopf_G": "@hopf.G_nom", "view": "@view.sensor_set", "estimator_form": "@form.primary", "all": [{"@unlesioned": true}]}, "event": {"status": "ABSENT"}, "cells": ["view"]}
```

</details>

<!-- part HCv2-15(c-pi0) -->
**Part `HCv2-15(c-pi0)`** (reported; label: none; predicted: none (reported part))

- Text: reported: ABSENT rate at G = 0 per view
- Rule: `describe`
- Parameters: `{}`
- Development dry run: REPORTED (reported).

<details><summary>Data selection of HCv2-15(c-pi0)</summary>

```json
{"where": {"design": "@design.whole_brain", "principle": "IIM", "@hopf_G": 0, "view": "@view.sensor_set", "estimator_form": "@form.primary"}, "event": {"status": "ABSENT"}, "cells": ["view"]}
```

</details>

<!-- part HCv2-15(d-present) -->
**Part `HCv2-15(d-present)`** (decisive; label: C; predicted: SUPPORTED (default))

- Text: BOLD: IIM not PRESENT at every G including G = 0 (PRESENT <= 0.07, H0 rule over the 8 levels)
- Rule: `h0_cell`
- Parameters: `{"alpha": 0.05, "bound": 0.07}`
- Development dry run: SUPPORTED (every pooled CP upper bound at 0.05/1 is below 0.07).

<details><summary>Data selection of HCv2-15(d-present)</summary>

```json
{"where": {"design": "@design.whole_brain", "principle": "IIM", "view": "@view.bold", "estimator_form": "@form.primary", "all": [{"@unlesioned": true}]}, "event": {"status": "PRESENT"}, "cell_label": "G{@hopf_G}"}
```

</details>

<!-- part HCv2-15(d-absent) -->
**Part `HCv2-15(d-absent)`** (decisive; label: C; predicted: SUPPORTED (default))

- Text: BOLD: false ABSENT at G_nom with CP upper bound < 0.02
- Rule: `demonstration`
- Parameters: `{"bound": 0.02, "level": 0.05, "n_planned": 150}`
- Development dry run: INDETERMINATE (CP upper bound at 0.05 below 0.02).

<details><summary>Data selection of HCv2-15(d-absent)</summary>

```json
{"where": {"design": "@design.whole_brain", "principle": "IIM", "view": "@view.bold", "@hopf_G": "@hopf.G_nom", "estimator_form": "@form.primary", "all": [{"@unlesioned": true}]}, "event": {"status": "ABSENT"}}
```

</details>

<!-- part HCv2-15(e-FMb1) -->
**Part `HCv2-15(e-FMb1)`** (reported; label: HO; predicted: none (reported part))

- Text: reported: FMb1 at sensor level (Spearman of c with G)
- Rule: `spearman_ols`
- Parameters: `{"alpha": 0.05, "require_positive_slope": false}`
- Development dry run: NOT_EVALUABLE (Spearman one-sided p < 0.05).

<details><summary>Data selection of HCv2-15(e-FMb1)</summary>

```json
{"where": {"design": "@design.whole_brain", "principle": "IIM", "view": "@view.sensor_set", "estimator_form": "@form.primary", "all": [{"@unlesioned": true}]}, "dose": "@hopf_G", "value": "c", "cells": ["view"]}
```

</details>

<!-- part HCv2-15(e-FMd) -->
**Part `HCv2-15(e-FMd)`** (decisive; label: HO; predicted: SUPPORTED (default))

- Text: FMd at sensor level, predicted to fail: the G_nom vs G = 0 contrast has the source's sign and >= half its size in < 80 % of seeds (reversed three-zone)
- Rule: `fmd_concordance`
- Parameters: `{"levels": [0, "@hopf.G_nom"], "ratio": 0.5, "source": "source", "view": "view", "x": 0.8, "zone": "reversed"}`
- Development dry run: NOT_EVALUABLE (FMd: forward contrast has the source's sign and >= 0.5 of its size; reversed three-zone at 0.8: SUPPORTED iff CP upper < 0.8, FALSIFIED iff rate >= 0.8).

<details><summary>Data selection of HCv2-15(e-FMd)</summary>

```json
{"where": {"design": "@design.whole_brain", "principle": "IIM", "estimator_form": "@form.primary", "@hopf_G": [0, "@hopf.G_nom"], "all": [{"@unlesioned": true}]}, "union": [{"label": "source", "where": {"view": "@view.source"}}, {"label": "view", "where": {"view": "@view.sensor_set"}}], "dose": "@hopf_G", "value": "c", "cells": ["view"]}
```

</details>

#### HCv2-16: RAM-PE specificity: no RAM-PE without plasticity (H-RAM-1)

<!-- hypothesis HCv2-16 -->
- Tier A; labels C.
- Development expectation: 0/40 at eta = 0 (development, v2 readout); reflex arc: dry run.
- Development dry run (DEVELOPMENT - NOT A RESULT): SUPPORTED.

<!-- part HCv2-16 -->
**Part `HCv2-16`** (decisive; label: C; predicted: SUPPORTED (default))

- Text: H0 cell rule with a cluster-robust pooled bound over 19 cells: RAM-only arm (W_RAM_no_plasticity, eta = 0, reflex_arc; 40 seeds each) and the 16 factorial cells with eta = 0 (A-R, 10 seeds each)
- Rule: `h0_cell`
- Parameters: `{"alpha": 0.05, "bound": 0.07, "m": 19}`
- Requires: `{"oracle": true, "usable": ["eta"]}`
- Development dry run: SUPPORTED (every pooled CP upper bound at 0.05/1 is below 0.07).

<details><summary>Data selection of HCv2-16</summary>

```json
{"where": {"principle": "RAM", "estimator_form": "@form.primary"}, "union": [{"label": "W_RAM_no_plasticity", "where": {"design": "@design.ram_only", "system": "W_RAM_no_plasticity"}, "cell_label": "ram_only:{member}"}, {"label": "eta_0", "where": {"design": "@design.ram_only", "@sweep_knob": "eta", "@sweep_level": 0}, "cell_label": "ram_only:{member}"}, {"label": "reflex_arc", "where": {"design": "@design.ram_only", "system": "reflex_arc"}, "cell_label": "ram_only:{member}"}, {"label": "factorial", "where": {"design": "@design.factorial", "family": "@family.A", "declaration_id": "R", "bit.RAM": 0}, "cell_label": "factorial:{@cell_id}"}], "event": {"status": "PRESENT"}}
```

</details>

<!-- part HCv2-16(reported) -->
**Part `HCv2-16(reported)`** (reported; label: none; predicted: none (reported part))

- Text: NULL_MODEL_VIOLATED rates
- Rule: `describe`
- Parameters: `{}`
- Development dry run: REPORTED (reported).

<details><summary>Data selection of HCv2-16(reported)</summary>

```json
{"where": {"principle": "RAM", "design": ["@design.ram_only", "@design.factorial"]}, "event": {"reason_code": "NULL_MODEL_VIOLATED"}, "cells": ["design", "system"]}
```

</details>

#### HCv2-17: RAM-PE sensitivity and its predicted non-selectivity (H-RAM-2 revised)

<!-- hypothesis HCv2-17 -->
- Tier A; labels R, C; successor of HC3.
- Statement: ABSENT parts: none (unreachable at 79-159 updates).
- Development expectation: Paired Delta c mean 1.02 (>= 0.5 in 34/40); excess -0.02/0.13/0.39/0.52 at eta 0/0.1/0.3/0.6.
- Development dry run (DEVELOPMENT - NOT A RESULT): FALSIFIED.

<!-- part HCv2-17(a) -->
**Part `HCv2-17(a)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: Spearman rho(c, eta) > 0 at one-sided p < 0.05
- Rule: `spearman_ols`
- Parameters: `{"alpha": 0.05, "require_positive_slope": false}`
- Requires: `{"usable": ["eta"]}`
- Development dry run: SUPPORTED (Spearman one-sided p < 0.05).

<details><summary>Data selection of HCv2-17(a)</summary>

```json
{"where": {"design": "@design.ram_only", "principle": "RAM", "estimator_form": "@form.primary", "@sweep_knob": "eta"}, "dose": "@sweep_level", "value": "c"}
```

</details>

<!-- part HCv2-17(b) -->
**Part `HCv2-17(b)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: paired Delta c(PC - W_RAM_no_plasticity) >= 0.5 in >= 80 % (three-zone)
- Rule: `three_zone`
- Parameters: `{"x": 0.8}`
- Requires: `{"oracle": true}`
- Development dry run: SUPPORTED (three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8).

<details><summary>Data selection of HCv2-17(b)</summary>

```json
{"where": {"design": "@design.ram_only", "principle": "RAM", "estimator_form": "@form.primary"}, "pair": {"on": ["seed", "principle"], "a": {"all": [{"system": "PC_nominal"}, {"@no_sweep": true}]}, "b": {"system": "W_RAM_no_plasticity"}}, "event": {"delta": {"ge": 0.5}}}
```

</details>

<!-- part HCv2-17(c) -->
**Part `HCv2-17(c)`** (decisive; label: C; predicted: SUPPORTED (default))

- Text: PC PRESENT more often than not: CP lower bound (0.05) > 0.5 (>= 26/40); FALSIFIED if the CP upper bound (0.05) is below 0.5
- Rule: `rate_lower_bound`
- Parameters: `{"alpha": 0.05, "x": 0.5}`
- Development dry run: SUPPORTED (SUPPORTED iff CP lower > 0.5; FALSIFIED iff CP upper < 0.5).

<details><summary>Data selection of HCv2-17(c)</summary>

```json
{"where": {"design": "@design.ram_only", "principle": "RAM", "estimator_form": "@form.primary", "all": [{"all": [{"system": "PC_nominal"}, {"@no_sweep": true}]}]}, "event": {"status": "PRESENT"}}
```

</details>

<!-- part HCv2-17(d) -->
**Part `HCv2-17(d)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: non-selectivity against SNR changes, decisive: median paired Delta c(K = 1 - PC) > z with one-sided 95 % Hodges-Lehmann lower bound > 0
- Rule: `hodges_lehmann`
- Parameters: `{"alpha": 0.05, "lower_gt": 0.0, "median_op": "gt", "median_x": 0.25}`
- Requires: `{"usable": ["K"]}`
- Notes: Predicted failure of selectivity (development +0.40).
- Development dry run: FALSIFIED (median gt 0.25 and one-sided 0.95 Hodges-Lehmann lower bound > 0.0).

<details><summary>Data selection of HCv2-17(d)</summary>

```json
{"where": {"design": "@design.ram_only", "principle": "RAM", "estimator_form": "@form.primary"}, "pair": {"on": ["seed", "principle"], "a": {"@sweep_knob": "K", "@sweep_level": 1}, "b": {"all": [{"system": "PC_nominal"}, {"@no_sweep": true}]}}}
```

</details>

<!-- part HCv2-17(d-g_b) -->
**Part `HCv2-17(d-g_b)`** (reported; label: none; predicted: none (reported part))

- Text: reported: the g_b = 0 contrast (development +0.16)
- Rule: `hodges_lehmann`
- Parameters: `{"alpha": 0.05, "lower_gt": 0.0}`
- Development dry run: SUPPORTED (one-sided 0.95 Hodges-Lehmann lower bound > 0.0).

<details><summary>Data selection of HCv2-17(d-g_b)</summary>

```json
{"where": {"design": "@design.ram_only", "principle": "RAM", "estimator_form": "@form.primary"}, "pair": {"on": ["seed", "principle"], "a": {"@sweep_knob": "g_b", "@sweep_level": 0}, "b": {"all": [{"system": "PC_nominal"}, {"@no_sweep": true}]}}}
```

</details>

<!-- part HCv2-17(e) -->
**Part `HCv2-17(e)`** (decisive; label: C; predicted: SUPPORTED (default))

- Text: construct boundary: the Newcombe 95 % interval of the PRESENT-rate difference (scrambled_feedback minus PC) includes 0; FALSIFIED if it excludes 0
- Rule: `newcombe_includes_zero`
- Parameters: `{"a": "scrambled", "b": "pc", "conf": 0.95}`
- Development dry run: SUPPORTED (Newcombe 95% interval of the rate difference includes 0).

<details><summary>Data selection of HCv2-17(e)</summary>

```json
{"where": {"design": "@design.ram_only", "principle": "RAM", "estimator_form": "@form.primary"}, "union": [{"label": "scrambled", "where": {"system": "scrambled_feedback"}}, {"label": "pc", "where": {"all": [{"system": "PC_nominal"}, {"@no_sweep": true}]}}], "event": {"status": "PRESENT"}}
```

</details>

#### HCv2-18: PDI: no states without contents (H-PDI-1)

<!-- hypothesis HCv2-18 -->
- Tier A; labels R.
- Development expectation: No-content systems B <= 0.25 bits (c <= 0.1); surrogates one state 280/280.
- Development dry run (DEVELOPMENT - NOT A RESULT): SUPPORTED.

<!-- part HCv2-18 -->
**Part `HCv2-18`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: H0 cell rule: family-A no-content witnesses (20 seeds each), the 16 factorial cells with K = 1 (10 seeds) and the null-calibration cells (16 x 50)
- Rule: `h0_cell`
- Parameters: `{"alpha": 0.05, "bound": 0.07}`
- Requires: `{"oracle": true}`
- Development dry run: SUPPORTED (every pooled CP upper bound at 0.05/1 is below 0.07).

<details><summary>Data selection of HCv2-18</summary>

```json
{"where": {"principle": "PDI", "estimator_form": "@form.primary"}, "union": [{"label": "witnesses", "where": {"design": "@design.witnesses", "estimator_form": "@form.primary", "family": "@family.A", "system": ["W_PDI_single_attractor", "W_PDI_no_multistability", "O_hypersynchronous", "N_ar1", "N_independent_noise"], "declaration_id": ["R", "H"], "seed_block": "witnesses_20"}, "cell_label": "witness:{system}"}, {"label": "factorial", "where": {"design": "@design.factorial", "family": "@family.A", "declaration_id": ["R", "H"], "bit.PDI": 0}, "cell_label": "factorial:{@cell_id}"}, {"label": "null_calibration", "where": {"design": "@design.null_calibration"}, "cell_label": "null_calibration:{@null_kind}:T{@n_time}:N{@n_nodes}"}], "event": {"status": "PRESENT"}}
```

</details>

<!-- part HCv2-18(reported) -->
**Part `HCv2-18(reported)`** (reported; label: none; predicted: none (reported part))

- Text: reported: B (bits) per class
- Rule: `describe`
- Parameters: `{}`
- Development dry run: REPORTED (reported).

<details><summary>Data selection of HCv2-18(reported)</summary>

```json
{"where": {"design": "@design.witnesses", "estimator_form": "@form.primary", "family": "@family.A", "principle": "PDI", "declaration_id": "R", "system": ["W_PDI_single_attractor", "W_PDI_no_multistability", "O_hypersynchronous", "N_ar1", "N_independent_noise"]}, "value": "estimate", "cells": ["system"]}
```

</details>

#### HCv2-19: PDI exclusion validity in the v1 masking window, and repertoire dose-response (H-PDI-2 + H-PDI-3)

<!-- hypothesis HCv2-19 -->
- Tier A; labels R, C; successor of HC3.
- Development expectation: Content bearer B 2.38-2.47 bits at g_b 0-2; B 0.90/1.51/2.45/3.50 at K 2/3/6/12.
- Development dry run (DEVELOPMENT - NOT A RESULT): SUPPORTED.

<!-- part HCv2-19(a) -->
**Part `HCv2-19(a)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: PDI ABSENT <= 0.07 in every content-on cell (H0 cell rule)
- Rule: `h0_cell`
- Parameters: `{"alpha": 0.05, "bound": 0.07}`
- Development dry run: SUPPORTED (every pooled CP upper bound at 0.05/1 is below 0.07).

<details><summary>Data selection of HCv2-19(a)</summary>

```json
{"where": {"principle": "PDI", "estimator_form": "@form.primary", "family": "@family.A", "declaration_id": ["R", "H"]}, "union": [{"label": "PC_nominal", "where": {"design": "@design.witnesses", "estimator_form": "@form.primary", "system": "PC_nominal"}, "cell_label": "witness:PC_nominal"}, {"label": "g_b_sweep", "where": {"design": "@design.sweep", "@sweep_knob": "g_b"}, "cell_label": "sweep:g_b:{@sweep_level}"}, {"label": "K_sweep", "where": {"design": "@design.sweep", "@sweep_knob": "K", "@sweep_level": {"ge": 2}}, "cell_label": "sweep:K:{@sweep_level}"}, {"label": "factorial_K_on", "where": {"design": "@design.factorial", "bit.PDI": 1}, "cell_label": "factorial:{@cell_id}"}], "event": {"status": "ABSENT"}}
```

</details>

<!-- part HCv2-19(b) -->
**Part `HCv2-19(b)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: PRESENT >= 80 % on PC_nominal and at each g_b level in [0.67, 1.56] (three-zone)
- Rule: `three_zone`
- Parameters: `{"x": 0.8}`
- Development dry run: SUPPORTED (three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8).

<details><summary>Data selection of HCv2-19(b)</summary>

```json
{"where": {"principle": "PDI", "estimator_form": "@form.primary", "family": "@family.A", "declaration_id": "R"}, "union": [{"label": "PC_nominal", "where": {"design": "@design.witnesses", "estimator_form": "@form.primary", "system": "PC_nominal"}, "cell_label": "witness:PC_nominal"}, {"label": "g_b_window", "where": {"design": "@design.sweep", "@sweep_knob": "g_b", "@sweep_level": {"ge": 0.666, "le": 1.556}}, "cell_label": "sweep:g_b:{@sweep_level}"}], "event": {"status": "PRESENT"}}
```

</details>

<!-- part HCv2-19(c) -->
**Part `HCv2-19(c)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: Spearman rho(B, K) > 0 at one-sided p < 0.05
- Rule: `spearman_ols`
- Parameters: `{"alpha": 0.05, "require_positive_slope": false}`
- Requires: `{"usable": ["K"]}`
- Development dry run: SUPPORTED (Spearman one-sided p < 0.05).

<details><summary>Data selection of HCv2-19(c)</summary>

```json
{"where": {"principle": "PDI", "estimator_form": "@form.primary", "family": "@family.A", "declaration_id": "R", "design": "@design.sweep", "@sweep_knob": "K"}, "dose": "@sweep_level", "value": "estimate"}
```

</details>

<!-- part HCv2-19(d) -->
**Part `HCv2-19(d)`** (decisive; label: C; predicted: SUPPORTED (default))

- Text: B >= log2 5 at K = 6 and B >= log2 10 at K = 12 in >= 80 % (count accuracy with one state of slack)
- Rule: `three_zone`
- Parameters: `{"x": 0.8}`
- Requires: `{"usable": ["K"]}`
- Development dry run: SUPPORTED (three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8).

<details><summary>Data selection of HCv2-19(d)</summary>

```json
{"where": {"principle": "PDI", "estimator_form": "@form.primary", "family": "@family.A", "declaration_id": "R", "design": "@design.sweep", "@sweep_knob": "K"}, "union": [{"label": "K6", "where": {"@sweep_level": 6}, "event": {"estimate": {"ge": 2.321928094887362}}}, {"label": "K12", "where": {"@sweep_level": 12}, "event": {"estimate": {"ge": 3.321928094887362}}}], "cell_label": "{member}"}
```

</details>

<!-- part HCv2-19(e) -->
**Part `HCv2-19(e)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: paired Delta c(PC - W_PDI_single_attractor) >= 0.5 in >= 80 %
- Rule: `three_zone`
- Parameters: `{"x": 0.8}`
- Requires: `{"oracle": true}`
- Development dry run: SUPPORTED (three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8).

<details><summary>Data selection of HCv2-19(e)</summary>

```json
{"where": {"design": "@design.witnesses", "estimator_form": "@form.primary", "family": "@family.A", "principle": "PDI", "declaration_id": "R"}, "pair": {"on": ["family", "seed", "declaration_id", "principle", "estimator_form"], "a": {"system": "PC_nominal"}, "b": {"system": "W_PDI_single_attractor"}, "value": "c"}, "event": {"delta": {"ge": 0.5}}}
```

</details>

#### HCv2-20: PDI construct: ignition, access declaration and the concordance route (H-PDI-4, H-PDI-5 revised)

<!-- hypothesis HCv2-20 -->
- Tier A; labels R, HO, C.
- Development expectation: Content bearer one state 40/40; full bearer two states in 10/20 with AMI 0.79; concordant one state 10/10 (W_PDI_single_attractor), 8/10 (W_PDI_no_multistability), PC 0/10.
- Development dry run (DEVELOPMENT - NOT A RESULT): INDETERMINATE.

<!-- part HCv2-20(a) -->
**Part `HCv2-20(a)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: content bearer on W_PDI_single_attractor ABSENT in >= 80 % where the concordance route is admitted for family A (CD-5); otherwise not PRESENT in >= 95 %
- Rule: `three_zone`
- Parameters: `{"x": 0.8}`
- Requires: `{"oracle": true}`
- Branches: `{"else": {"data": {"event": {"status": {"ne": "PRESENT"}}, "where": {"declaration_id": "R", "design": "@design.witnesses", "estimator_form": "@form.primary", "family": "@family.A", "principle": "PDI", "system": "W_PDI_single_attractor"}}, "params": {"x": 0.95}, "text": "not PRESENT in >= 95 % (route not admitted)"}, "if": {"concordance_admitted": {"bearer": "non_workspace", "direction": "absent", "principle": "PDI", "substrate": "synthetic_rate"}}, "then": {"text": "ABSENT in >= 80 % (route admitted)"}}`
- Development expectation: Development (CD-5): under each of A-R and A-H, 41 of 52 W_PDI_single_attractor runs (0.79) are concordant ABSENTs, just below 0.80; reported, not acted on.
- Notes: The gate names the admitted cell by its substrate, synthetic_rate (the family-A substrate the protocols' concordance route carries); it compared the family id 'A' before and would have chosen the 'else' branch.
- Development dry run: SUPPORTED (three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8).

<details><summary>Data selection of HCv2-20(a)</summary>

```json
{"where": {"design": "@design.witnesses", "estimator_form": "@form.primary", "family": "@family.A", "principle": "PDI", "declaration_id": "R", "system": "W_PDI_single_attractor"}, "event": {"status": "ABSENT"}}
```

</details>

<!-- part HCv2-20(b) -->
**Part `HCv2-20(b)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: full bearer: among seeds with K_hat_full = 2, AMI with the oracle ignition gate >= 0.5 in >= 80 % (>= 10 such seeds, else NOT_EVALUABLE)
- Rule: `three_zone`
- Parameters: `{"min_n": 10, "x": 0.8}`
- Requires: `{"oracle": true}`
- Development dry run: NOT_EVALUABLE (three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8).

<details><summary>Data selection of HCv2-20(b)</summary>

```json
{"where": {"design": "@design.witnesses", "estimator_form": "@form.primary", "family": "@family.A", "principle": "PDI", "declaration_id": "R", "system": "W_PDI_single_attractor", "@pdi_k_full": 2}, "event": {"@pdi_ami_ignition": {"ge": 0.5}}}
```

</details>

<!-- part HCv2-20(c) -->
**Part `HCv2-20(c)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: K_hat_full <= 2 in >= 95 %
- Rule: `three_zone`
- Parameters: `{"x": 0.95}`
- Requires: `{"oracle": true}`
- Development dry run: SUPPORTED (three-zone at 0.95: SUPPORTED iff rate >= 0.95, FALSIFIED iff CP upper < 0.95).

<details><summary>Data selection of HCv2-20(c)</summary>

```json
{"where": {"design": "@design.witnesses", "estimator_form": "@form.primary", "family": "@family.A", "principle": "PDI", "declaration_id": "R", "system": "W_PDI_single_attractor"}, "event": {"@pdi_k_full": {"le": 2}}}
```

</details>

<!-- part HCv2-20(d) -->
**Part `HCv2-20(d)`** (decisive; label: HO; predicted: SUPPORTED (default))

- Text: mis-declared access node (module S declared as the workspace): the content bearer counts >= 2 states on W_PDI_single_attractor in >= 25 % of seeds (three-zone at 0.25)
- Rule: `three_zone`
- Parameters: `{"x": 0.25}`
- Requires: `{"oracle": true}`
- Development dry run: NOT_EVALUABLE (no input rows).

<details><summary>Data selection of HCv2-20(d)</summary>

```json
{"where": {"design": "@design.witnesses", "estimator_form": "@form.pdi_misdeclared_access", "family": "@family.A", "principle": "PDI", "declaration_id": "R", "system": "W_PDI_single_attractor"}, "event": {"@pdi_k_content": {"ge": 2}}}
```

</details>

<!-- part HCv2-20(e) -->
**Part `HCv2-20(e)`** (decisive; label: C; predicted: SUPPORTED (default))

- Text: concordance-route safety: among all content-on rows (>= 365) the concordant-ABSENT rate is < 0.01: SUPPORTED iff the CP upper bound (0.05) is below 0.01, FALSIFIED if the CP lower bound exceeds 0.01
- Rule: `h0_cell`
- Parameters: `{"P": 1, "alpha": 0.05, "bound": 0.01, "m": 1}`
- Development dry run: INDETERMINATE (a pooled upper bound is not below the bound).

<details><summary>Data selection of HCv2-20(e)</summary>

```json
{"where": {"principle": "PDI", "estimator_form": "@form.primary", "family": "@family.A", "declaration_id": ["R", "H"]}, "union": [{"label": "PC_nominal", "where": {"design": "@design.witnesses", "estimator_form": "@form.primary", "system": "PC_nominal"}, "cell_label": "content_on"}, {"label": "g_b_sweep", "where": {"design": "@design.sweep", "@sweep_knob": "g_b"}, "cell_label": "content_on"}, {"label": "K_sweep", "where": {"design": "@design.sweep", "@sweep_knob": "K", "@sweep_level": {"ge": 2}}, "cell_label": "content_on"}, {"label": "factorial_K_on", "where": {"design": "@design.factorial", "bit.PDI": 1}, "cell_label": "content_on"}], "event": {"@concordant_absent": true}, "pool_by": ["principle"]}
```

</details>

<!-- part HCv2-20(f-single) -->
**Part `HCv2-20(f-single)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: concordant-and-ABSENT share >= 80 % on W_PDI_single_attractor
- Rule: `three_zone`
- Parameters: `{"x": 0.8}`
- Requires: `{"oracle": true}`
- Development dry run: SUPPORTED (three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8).

<details><summary>Data selection of HCv2-20(f-single)</summary>

```json
{"where": {"design": "@design.witnesses", "estimator_form": "@form.primary", "family": "@family.A", "principle": "PDI", "declaration_id": "R", "system": "W_PDI_single_attractor"}, "event": {"@concordant_absent": true}}
```

</details>

<!-- part HCv2-20(f-no-multistability) -->
**Part `HCv2-20(f-no-multistability)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: concordant-and-ABSENT share >= 70 % on W_PDI_no_multistability
- Rule: `three_zone`
- Parameters: `{"x": 0.7}`
- Requires: `{"oracle": true}`
- Development dry run: SUPPORTED (three-zone at 0.7: SUPPORTED iff rate >= 0.7, FALSIFIED iff CP upper < 0.7).

<details><summary>Data selection of HCv2-20(f-no-multistability)</summary>

```json
{"where": {"design": "@design.witnesses", "estimator_form": "@form.primary", "family": "@family.A", "principle": "PDI", "declaration_id": "R", "system": "W_PDI_no_multistability"}, "event": {"@concordant_absent": true}}
```

</details>

<!-- part HCv2-20(f-PC) -->
**Part `HCv2-20(f-PC)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: concordant share <= 20 % on PC_nominal
- Rule: `three_zone_at_most`
- Parameters: `{"x": 0.2}`
- Development dry run: SUPPORTED (three-zone at most 0.2: SUPPORTED iff rate <= 0.2, FALSIFIED iff CP lower > 0.2).

<details><summary>Data selection of HCv2-20(f-PC)</summary>

```json
{"where": {"design": "@design.witnesses", "estimator_form": "@form.primary", "family": "@family.A", "principle": "PDI", "declaration_id": "R", "system": "PC_nominal"}, "event": {"concordant": true}}
```

</details>

#### HCv2-21: PDI forward-model admission (forward-modelled family A, held-out regime)

<!-- hypothesis HCv2-21 -->
- Tier A; labels R.
- Statement: SUPPORTED iff the observed admission pattern equals the prediction for the three views; FALSIFIED otherwise.
- Development dry run (DEVELOPMENT - NOT A RESULT): NOT_EVALUABLE.

<!-- part HCv2-21 -->
**Part `HCv2-21`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: EEG-64 and EEG-low admitted for PRESENT (declared non-specific upper bound) and admitted_for_absent = vacuous; BOLD (slow contexts) admitted_for_absent = vacuous (corrected before the release; the original prediction is kept in original_predictions)
- Rule: `admission_matches`
- Parameters: `{"original_predictions": {"@view.bold": {"admitted_for_absent": "no"}, "@view.eeg64": {"admitted_for_absent": "yes", "admitted_for_present": "yes"}, "@view.eeg_low": {"admitted_for_absent": "yes", "admitted_for_present": "yes"}}, "predictions": {"@view.bold": {"admitted_for_absent": "vacuous"}, "@view.eeg64": {"admitted_for_absent": "vacuous", "admitted_for_present": "yes"}, "@view.eeg_low": {"admitted_for_absent": "vacuous", "admitted_for_present": "yes"}}}`
- Requires: `{"oracle": true}`
- Notes: Corrected before the release of the held-out anchors as a logical error (CD-5). The original admitted_for_absent predictions (yes on EEG-64 and EEG-low; no on BOLD, 'K = 3 read as one state'; committed in d3bcb09 and kept in original_predictions) assumed the exact path that design decision 9 had already rejected: a concordant PDI count outside an admitted concordance cell is UNDEFINED(NO_SAMPLING_SE), never ABSENT, and no forward cell is admitted (the CD-5 battery covers the source cell only). The corrected values follow mechanically from the 3.6 rule (absent needs FM0 and FMabs; vacuous iff pi0 < 0.1) on the development-regime forward rows (dry run 384-399 and 804-819, lead-field seed 20260928, width 0.5) judged under a development-regime build: on every view FM0 holds, none of the 22 on-runs per view is ABSENT (FMabs is demonstrable at the 190 confirmatory on-runs) and every null run is UNDEFINED(NO_SAMPLING_SE), so pi0 = 0 and each view is vacuous. The PRESENT predictions are unchanged.
- Development dry run: NOT_EVALUABLE (no input rows).

<details><summary>Data selection of HCv2-21</summary>

```json
{"source": "registry", "where": {"principle": "PDI", "arm": "@arm.forward_family_a"}, "cell_label": "{view}"}
```

</details>

#### HCv2-22: Single-deficit lesion witnesses, rule side (HC4v2-R, HC8v2-R)

<!-- hypothesis HCv2-22 -->
- Tier A; labels R; successor of HC4, HC8; predicted outcome FALSIFIED.
- Statement: Per (protocol, witness w with target j in N_anch): (i) target not PRESENT in >= 80 %; (ii) target ABSENT in >= 80 %, decisive only where pi0(f, j, w) >= 0.9; (iii) every other anchored principle: paired |c_p(w) - c_p(PC)| < z in >= 80 % (declared dependencies excluded).
- Development expectation: A-R: W_NAS_no_workspace (ii) decisive if pi0 >= 0.9 and predicted SUPPORTED; W_NAS_broadcast_only (ii) predicted FALSIFIED where testable; W_IIM_feedforward, W_SRPI_no_efference, W_RAM_no_plasticity (ii) NOT_TESTABLE_BY_DESIGN; W_PDI_single_attractor (ii) NOT_TESTABLE_BY_DESIGN with the admitted concordance route (development pi0 0.79, 41/52; CD-5).
- Development dry run (DEVELOPMENT - NOT A RESULT): FALSIFIED.

<!-- part HCv2-22(i) -->
**Part `HCv2-22(i)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: target not PRESENT in >= 80 % (three-zone)
- Rule: `three_zone`
- Parameters: `{"x": 0.8}`
- Requires: `{"oracle": true}`
- Development dry run: SUPPORTED (three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8).

<details><summary>Data selection of HCv2-22(i)</summary>

```json
{"where": {"design": "@design.witnesses", "family": ["@family.A", "@family.C1"], "declaration_id": ["R", "H"], "estimator_form": "@form.primary", "system": "@systems.single_deficits"}, "target_filter": "in_n_anch", "exclude": ["non_target"], "event": {"status": {"ne": "PRESENT"}}, "cell_label": "{protocol_id}|{system}"}
```

</details>

<!-- part HCv2-22(ii) -->
**Part `HCv2-22(ii)`** (decisive; label: R; predicted: FALSIFIED (stated))

- Text: target ABSENT in >= 80 % (three-zone), decisive only where pi0 >= 0.9 (CD-7), else NOT_TESTABLE_BY_DESIGN with its predicted ABSENT_NOT_REACHABLE rate
- Rule: `three_zone`
- Parameters: `{"x": 0.8}`
- Requires: `{"oracle": true}`
- Gate: `{"family": "{protocol_id}", "kind": "absent", "per_cell": true, "principle": "{target}", "witness": "{system}"}`
- Replaced by: `HCv2-22(i)`
- Cell predictions: `[{"match": {"system": "W_NAS_broadcast_only"}, "prediction": "FALSIFIED"}, {"match": {"system": "W_NAS_no_workspace"}, "prediction": "SUPPORTED"}]`
- Development dry run: SUPPORTED (three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8).

<details><summary>Data selection of HCv2-22(ii)</summary>

```json
{"where": {"design": "@design.witnesses", "family": ["@family.A", "@family.C1"], "declaration_id": ["R", "H"], "estimator_form": "@form.primary", "system": "@systems.single_deficits"}, "target_filter": "in_n_anch", "exclude": ["non_target"], "event": {"status": "ABSENT"}, "cell_label": "{protocol_id}|{system}"}
```

</details>

<!-- part HCv2-22(iii) -->
**Part `HCv2-22(iii)`** (decisive; label: R; predicted: FALSIFIED (stated))

- Text: every other anchored principle p: paired |c_p(w) - c_p(PC)| < z in >= 80 % (declared dependencies excluded)
- Rule: `three_zone`
- Parameters: `{"x": 0.8}`
- Requires: `{"oracle": true}`
- Cell predictions: `[{"match": {"declaration_id": "H", "family": "@family.A", "principle": "NAS", "system": "W_PDI_single_attractor"}, "prediction": "FALSIFIED"}, {"match": {"declaration_id": "R", "family": "@family.A", "principle": "NAS", "system": "W_PDI_single_attractor"}, "prediction": "SUPPORTED"}]`
- Notes: The family-A cell (W_PDI_single_attractor, NAS) is the v1 HC3 K -> NAS off-target criterion (HCv2-7(v-A)); CD-7 does not declare it a dependency. Its prediction is split by declaration: under A-H FALSIFIED (development 28/52), under A-R not FALSIFIED (development 44/52 = 0.846). The part stays predicted FALSIFIED through the A-H cell and the other development cells below 0.8 (for example A-R RAM on W_PDI_single_attractor 14/52, A-R IIM on W_NAS_broadcast_only 30/52, C1-R IIM on W_NAS_broadcast_only 8/40).
- Development dry run: FALSIFIED (three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8).

<details><summary>Data selection of HCv2-22(iii)</summary>

```json
{"where": {"design": "@design.witnesses", "family": ["@family.A", "@family.C1"], "declaration_id": ["R", "H"], "estimator_form": "@form.primary"}, "principle_filter": "in_n_anch", "union": [{"label": "W_RAM_no_plasticity", "pair": {"on": ["family", "seed", "declaration_id", "principle", "estimator_form"], "a": {"system": "W_RAM_no_plasticity"}, "b": {"system": "PC_nominal"}}}, {"label": "W_PDI_single_attractor", "pair": {"on": ["family", "seed", "declaration_id", "principle", "estimator_form"], "a": {"system": "W_PDI_single_attractor"}, "b": {"system": "PC_nominal"}}}, {"label": "W_NAS_no_workspace", "pair": {"on": ["family", "seed", "declaration_id", "principle", "estimator_form"], "a": {"system": "W_NAS_no_workspace"}, "b": {"system": "PC_nominal"}}}, {"label": "W_NAS_broadcast_only", "pair": {"on": ["family", "seed", "declaration_id", "principle", "estimator_form"], "a": {"system": "W_NAS_broadcast_only"}, "b": {"system": "PC_nominal"}}}, {"label": "W_IIM_feedforward", "pair": {"on": ["family", "seed", "declaration_id", "principle", "estimator_form"], "a": {"system": "W_IIM_feedforward"}, "b": {"system": "PC_nominal"}}}, {"label": "W_SRPI_no_efference", "pair": {"on": ["family", "seed", "declaration_id", "principle", "estimator_form"], "a": {"system": "W_SRPI_no_efference"}, "b": {"system": "PC_nominal"}}}], "target_filter": "in_n_anch", "exclude": ["target", "declared_dependencies"], "event": {"delta": {"abs_lt": 0.25}}, "cell_label": "{protocol_id}|{system}|{principle}"}
```

</details>

#### HCv2-23: Verdict specificity (HC5v2)

<!-- hypothesis HCv2-23 -->
- Tier A; labels C; successor of HC5; verdict-level ABSENT: yes (the HCv2-4 reversion rule applies).
- Statement: FALSIFIED if some class's CP lower bound at 0.05/m exceeds 0.07; SUPPORTED if the seed-cluster 'any event' pooled rate (a seed is an event if any of its deficit runs is MPC_CONSISTENT; clusters span families) has CP upper bound (0.05) < 0.07; else INDETERMINATE.
- Notes: Seed counts (CD-11; seeds resized, never thresholds). The any-event clusters are keyed by the seed alone and span families, so the A and C1 single deficits on the shared seeds 20000-20044 form 45 clusters, not 90. On the development rates (family A: 3 any-event clusters in 52; family C1: NAS PRESENT in 0 of 72 rows) P(SUPPORTED | development) is 0.069 at 45 clusters. The three C1 single deficits of the pool, W_NAS_no_workspace, W_NAS_broadcast_only and W_IIM_feedforward, therefore run on 129 seeds (20000-20128) and family A keeps 45: 129 is the smallest count whose pooled CP upper bound at 0.05 stays below 0.07 with four event clusters, and P(SUPPORTED | development) is 0.884 (at least 0.88 at every count up to 400; P(FALSIFIED) about 0). The resize works only through C1 clusters without events, and no C1 verdict could be MPC_CONSISTENT in development, where C1 IIM is held out (HO-4); with a Jeffreys predictive on the family-A rate P(SUPPORTED) is 0.78, on both rates 0.39. The other parts that read these C1 systems (HCv2-5, -7, -14 and -22) read them on all their seeds, except that a part pairing them with PC_nominal keeps its 45 pairs; HCv2-4 and HCv2-8(d) keep their named seed blocks.
- Development dry run (DEVELOPMENT - NOT A RESULT): INDETERMINATE.

<!-- part HCv2-23 -->
**Part `HCv2-23`** (decisive; label: C; predicted: SUPPORTED (default))

- Text: MPC_CONSISTENT among single-deficit runs with the target in N_anch
- Rule: `any_event_clusters`
- Parameters: `{"alpha": 0.05, "bound": 0.07}`
- Requires: `{"oracle": true}`
- Development dry run: INDETERMINATE (the any-event upper bound is not below the bound).

<details><summary>Data selection of HCv2-23</summary>

```json
{"source": "verdicts", "where": {"design": "@design.witnesses", "family": ["@family.A", "@family.C1"], "declaration_id": ["R", "H"], "estimator_form": "@form.primary", "system": "@systems.single_deficits"}, "target_filter": "in_n_anch", "event": {"verdict_value": "MPC_CONSISTENT"}, "cell_label": "{protocol_id}|{system}", "cluster": ["seed"]}
```

</details>

<!-- part HCv2-23(N_decl) -->
**Part `HCv2-23(N_decl)`** (reported; label: none; predicted: none (reported part))

- Text: reported: verdicts on N_decl
- Rule: `describe`
- Parameters: `{}`
- Development dry run: REPORTED (reported).

<details><summary>Data selection of HCv2-23(N_decl)</summary>

```json
{"source": "verdicts", "where": {"design": "@design.witnesses", "family": ["@family.A", "@family.C1"], "declaration_id": ["R", "H"], "estimator_form": "@form.primary", "system": "@systems.single_deficits"}, "event": {"@verdict_n_decl": "MPC_CONSISTENT"}, "cell_label": "{protocol_id}|{system}"}
```

</details>

#### HCv2-24: No PRESENT without mechanism in the factorial (HC6 successor)

<!-- hypothesis HCv2-24 -->
- Tier A; labels R; successor of HC6.
- Development dry run (DEVELOPMENT - NOT A RESULT): INDETERMINATE.

<!-- part HCv2-24(a) -->
**Part `HCv2-24(a)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: under R: P(PRESENT | own mechanism off) <= 0.07 in every off-cell (H0 cell rule)
- Rule: `h0_cell`
- Parameters: `{"alpha": 0.05, "bound": 0.07}`
- Development dry run: SUPPORTED (every pooled CP upper bound at 0.05/5 is below 0.07).

<details><summary>Data selection of HCv2-24(a)</summary>

```json
{"where": {"design": "@design.factorial", "family": ["@family.A", "@family.C1"], "declaration_id": "R", "estimator_form": "@form.primary", "own_bit": 0}, "principle_filter": "valid_anchor", "event": {"status": "PRESENT"}, "cell_label": "{family}:{@cell_id}|{principle}"}
```

</details>

<!-- part HCv2-24(b) -->
**Part `HCv2-24(b)`** (decisive; label: R; predicted: SUPPORTED (default))

- Text: under H, decisive predicted failure: NAS is PRESENT above 0.07 in the family-A cells with K on and g_b off (some cell's CP lower bound at 0.05/m > 0.07)
- Rule: `h0_cell_exceeds`
- Parameters: `{"alpha": 0.05, "bound": 0.07}`
- Development dry run: INDETERMINATE (a pooled upper bound is not below the bound).

<details><summary>Data selection of HCv2-24(b)</summary>

```json
{"where": {"design": "@design.factorial", "family": "@family.A", "declaration_id": "H", "estimator_form": "@form.primary", "principle": "NAS", "bit.PDI": 1, "bit.NAS": 0}, "event": {"status": "PRESENT"}, "cell_label": "{@cell_id}"}
```

</details>

### 6.3 Tier-B hypotheses (not admitted, not run)

The Tier-B hypotheses of the design are listed with their entries in the hypotheses file. None was admitted in this round; the evaluator reports them as NOT_RUN, and where a v1 predecessor exists its outcome stands.

#### HCv2-B1: SRPI v3 in the SRPI-only arm

<!-- hypothesis HCv2-B1 -->
- Tier B; labels none; admitted in this round: no.
- Notes: Not admitted in this round (Tier A only); listed as not run, and the v1 outcome stands where a v1 predecessor exists.
- Development dry run (DEVELOPMENT - NOT A RESULT): NOT_RUN.

#### HCv2-B2: NAS hub identity (HN4 revised)

<!-- hypothesis HCv2-B2 -->
- Tier B; labels none; admitted in this round: no.
- Notes: Not admitted in this round (Tier A only); listed as not run, and the v1 outcome stands where a v1 predecessor exists.
- Development dry run (DEVELOPMENT - NOT A RESULT): NOT_RUN.

#### HCv2-B3: RAM-PE persistence gate and the echo family

<!-- hypothesis HCv2-B3 -->
- Tier B; labels none; admitted in this round: no.
- Notes: Not admitted in this round (Tier A only); listed as not run, and the v1 outcome stands where a v1 predecessor exists.
- Development dry run (DEVELOPMENT - NOT A RESULT): NOT_RUN.

#### HCv2-B4: Single source via joint dependence (HC7v2 part a)

<!-- hypothesis HCv2-B4 -->
- Tier B; labels none; admitted in this round: no.
- Notes: Not admitted in this round (Tier A only); listed as not run, and the v1 outcome stands where a v1 predecessor exists.
- Development dry run (DEVELOPMENT - NOT A RESULT): NOT_RUN.

#### HCv2-B5: Rule audit (HC9v2)

<!-- hypothesis HCv2-B5 -->
- Tier B; labels none; admitted in this round: no.
- Notes: Not admitted in this round (Tier A only); listed as not run, and the v1 outcome stands where a v1 predecessor exists.
- Development dry run (DEVELOPMENT - NOT A RESULT): NOT_RUN.

#### HCv2-B6: Zero-mean workspace broadcast (W_ctx_hub_only)

<!-- hypothesis HCv2-B6 -->
- Tier B; labels none; admitted in this round: no.
- Notes: Not admitted in this round (Tier A only); listed as not run, and the v1 outcome stands where a v1 predecessor exists.
- Development dry run (DEVELOPMENT - NOT A RESULT): NOT_RUN.

#### HCv2-B7: PDI slow-drift continuum

<!-- hypothesis HCv2-B7 -->
- Tier B; labels none; admitted in this round: no.
- Notes: Not admitted in this round (Tier A only); listed as not run, and the v1 outcome stands where a v1 predecessor exists.
- Development dry run (DEVELOPMENT - NOT A RESULT): NOT_RUN.

#### HCv2-B8: Forward arms for RAM-PE and SRPI

<!-- hypothesis HCv2-B8 -->
- Tier B; labels none; admitted in this round: no.
- Notes: Not admitted in this round (Tier A only); listed as not run, and the v1 outcome stands where a v1 predecessor exists.
- Development dry run (DEVELOPMENT - NOT A RESULT): NOT_RUN.

### 6.4 Mechanism-on labels and declared dependencies

The mechanism-on table (CD-8) has 1370 entries (`mechanism_on.entries` of the
hypotheses file, equal to `protocols/v2/generated/mechanism_on.json`) and 310
dose-only entries (`dose_only_entries`, reported beside HCv2-5). The declared
dependencies of HCv2-22(iii) (CD-7) are (W_IIM_feedforward, NAS) and
(W_NAS_no_workspace, IIM) (`declared_dependencies.pairs`, equal to
`protocols/v2/generated/declared_dependencies.json`).

## 7. Calibration on development data

DEVELOPMENT - NOT A RESULT. Every number in this section comes from development
seeds 0-999: the full development calibration at code `38245c3` (started
2026-10-07 at 09:52 CEST, 0 task and 0 component errors), the oracle checks
recomputed at code `1dd9096`, and the held-out-regime forward anchors at code
`a16f84c` after the release. Development values that enter a threshold are
labelled R in the hypotheses. The decisions apply rules written before any
calibration output existed: the design and the commitments of 2026-10-07
(section 7.6). The development data could not reverse these rules. Where a rule
left room for judgement, the option that makes a false ABSENT or a false
exclusion less likely was taken. The cutoffs `z = 0.25`, `delta = 0.10` and
`alpha = 0.05` and every hypothesis threshold are unchanged. Independent
recomputations from the raw development records agreed with every number the
protocol builder printed; their differences were resolved as stated below.

Decisions file: `outputs/mpcbench_v2/decisions/calibration_decisions_v2.json`
(status `final`, SHA-256 `d3d5bd5da0043ab750979853f9c73932032b08a3c3e36dac31f14d8b881f1744`), copied with its evidence into
`protocols/v2/generated/calibration_decisions.json` and
`calibration_evidence.json`.

### 7.1 Standard errors (CD-2, CD-3, CD-4)

**When an SE method counts as calibrated** (rule fixed before any calibration
output): in a class, its kappa point is in [0.8, 1.25], its 90 % chi-square
interval lies inside [0.67, 1.5] and both one-sided `q_A` tail rates are at
most 0.02. Kappa is the pooled within-network SD of `c` divided by
`RMS(se_c)`, with an interval on `sum (n_twin - 1)` df; for NAS the tails are
per direction at `alpha_A / 2`. Classes are the HCv2-4 cells; the data are the
development twin networks 820-824 with 7 sessions each (r = 0..6) and the
RAM-only twins with 8 (eta 0, 0.1, 0.3). The design's literal wording
("interval inside [0.8, 1.25]") was replaced before any data existed: at df 30
the interval spans a factor of about 1.54, so that wording could be met only
for a kappa point of about 0.97-0.98. In every class used, `c` is defined in at
least 89 % of sessions; the two PDI W_PDI_single_attractor twin cells (A-H and
A-R) have a jackknife SE in only 2 of 35 sessions (the other 33 are admitted
concordant components without a sampling SE), so they are below HCv2-4's 80 %
defined share and are not counted as calibration failures.

**CD-2, NAS SE method.** Rule: among {contiguous, interleaved} x G {10, 20}, the
method with the largest df that is calibrated in every class; ties go to
contiguous; if none is calibrated everywhere, contiguous G = 10 with the failure
reported. Calibrated classes out of 32: contiguous G10 7, contiguous G20 11,
interleaved G10 13, interleaved G20 13. Outcome: `jackknife_contiguous_10`
(df 9), failure reported. All four schemes fail anti-conservatively in the A-H
return direction (W_PDI_single_attractor: kappa 3.04, 90 % CI 2.52-3.87, tails
0.171 below and 0.114 above; PC_nominal 1.94, 1.61-2.47) and conservatively in
most C1 cells (kappa 0.52-0.91). No inflation factor. Expected consequence:
HCv2-4 FALSIFIED for NAS on the anti-conservative side, so its reversion rule
re-classifies NAS ABSENTs and the hypotheses that use NAS ABSENT are reported
both ways.

**CD-3, IIM SE method.** Rule: keep the circular block bootstrap (10 % blocks,
B = 50) unless it fails in some class and the contiguous jackknife G = 10 is
calibrated in every class. Development (20 family-A classes; C1 is held out for
IIM under HO-4 and has no development twins, so the IIM SE calibration rests on
family A only): bootstrap calibrated in 6/20 (kappa median 0.85, range
0.47-1.14), jackknife in 4/20 (median 0.78, range 0.40-0.87). Outcome:
`circular_block_bootstrap_10pct_B50`, failure reported. This departs from the
protocol builder's mechanical suggestion (the jackknife), which switched
whenever a bootstrap interval lay entirely outside [0.8, 1.25] and did not
check the fallback; the binding rule does not swap in a method that is not
better. The bootstrap's failures are mostly conservative (kappa 0.47-0.72 in
the PC_half classes); its other failures are single-session tail exceedances.

**CD-3, IIM bootstrap df.** Rule: `se_df = 12`, or 9 if a one-sided `q_A` tail
exceeds 0.02, judged per class. At df 12, 6 of 20 classes exceed 0.02, each by
one session in 35 (0.029). Outcome: **`se_df = 9`**. At this granularity the
trigger fires almost surely even for an exactly calibrated SE, so the lowered
df is the rule's conservative outcome (larger quantile, fewer ABSENT calls),
not evidence of anti-conservative tails: over 692 sessions the pooled rates are
0.0014 below and 0.0072 above, both under 0.01. At df 9 the bootstrap is
calibrated in 8/20 classes. Re-judging the stored family-B development statuses
at df 9 changes none of them.

**CD-3, IIM occupancy gate.** Rule: `N_min = 25` unless a development HCv2-12(c)
cell fails its 90 % criterion. Seeds 430-434, 5 runs per cell: the 6 cells with
`T pi_min <= 12.5` were `UNDEFINED(INSUFFICIENT_OCCUPANCY)` in 5/5 runs and the
4 cells with `T pi_min >= 100` defined in 5/5. Outcome: **`N_min = 25`**.

**CD-4, RAM-PE SE method.** Rule: keep the shift-null SD unless it fails in some
class and the trial jackknife G = 10 is calibrated in every class. The shift-null
SD is calibrated in 6/13 classes (RAM-only arm: eta 0 kappa 0.92, eta 0.1 0.97;
eta 0.3 fails by one below-tail event in 40; joint bench PC_half
anti-conservative, kappa 1.25, 90 % CI 1.04-1.59, above-tail 0.057); the trial
jackknife in 1/3 eta classes (kappa 0.51, 0.60, 0.84). Outcome:
**`shift_null_sd`**, failure reported.

The decided SE methods are the ones every non-fallback development run used, so
nothing was re-run for the method choices.

### 7.2 PDI concordance route (CD-5) and the k-means seed

**Rule** (design 2.4 item 4; commitments section 1). A concordant PDI component
(all 44 counts identical) is `UNDEFINED(NO_SAMPLING_SE)` unless its (substrate,
observation stage, view, bearer) cell is admitted. The ABSENT route needs 0
concordant ABSENTs (`|c| < 0.10`) among at least 300 content-on development
runs; the PRESENT route needs 0 concordant PRESENTs (`c > 0.25`) among at least
300 no-content runs. With 0 events in 300 runs the one-sided 95 %
Clopper-Pearson upper bound is 0.00994 < `alpha_A`.

**Battery as run.** The planned battery (seeds 870-899) gave 150 content-on and
120 no-content runs and could admit neither route by arithmetic alone. It was
enlarged before any battery output to development seeds 850-899 and 940-979:
90 seeds x 9 classes = 810 runs, 0 errors, k-means seed 0, protocol A-R, source
view, content bearer. Content-on (K = 2, 3, 6; g_b = 0.67, 1.0, 1.56 at K = 6):
450 runs, 0 concordant ABSENTs (CP upper bound 0.0066); the one concordant run
counted the correct two states (a concordant PRESENT, c = 0.41). No-content
(W_PDI_single_attractor content bearer, W_PDI_no_multistability,
O_hypersynchronous, N_ar1): 360 runs, 332 concordant, all at one state with
zero excess; 0 concordant PRESENTs (CP upper bound 0.0083). The classes share
each seed's network, session length, surrogate seed and k-means seed, so the
runs form 90 seed clusters; the bound over seeds (0/90) is 0.033.

**Outcome.** Exactly one cell is admitted, in both directions: substrate
`synthetic_rate` (family A), observation stage source, view source, bearer
`non_workspace` (the declared content bearer), at the battery's generator
regime; pooled battery counts of the cell in the generated protocols: 456
content-on and 366 no-content runs, 0 events. It applies to family-A
source-view content-bearer components under A-R, A-H, A-P, A-Q10, A-Q25, A-J
and A-none and their forms, and to the source view of the forward EEG arm's
agent. It does not apply to the mis-declared-access form (a different bearer),
to the source view of the slow-context BOLD arm (a different regime: 5348-5403
windows against 2269-2475 in the battery), to any full-bearer cell, to C1 or to
any forward cell. The confirmatory protocols carry the route only on the
admitted cell. Error control of the route is empirical over the declared
battery; it is re-tested confirmatorily by HCv2-20(e) (at least 365 content-on
rows), HCv2-4(c) and the separate HCv2-5 cells.

**Deviation.** Design 2.4 item 4 also lists the forward views of the cell among
the battery's content-on runs. The battery was run as budgeted (810 runs,
source cell only), so no forward cell reached 300 runs and no forward route
exists, not even provisionally; forward concordant components stay
`UNDEFINED(NO_SAMPLING_SE)`, the exclusion-safe outcome. A forward battery was
not run afterwards, because it could only add exclusions (on BOLD, 7 of 44
development content-on runs were concordant at one state).

**Consequences fixed by the rule.** HCv2-20(a) is tested in its "ABSENT in
>= 80 %" form; its gate names the admitted cell's substrate `synthetic_rate`
(it compared the family id "A" before and would have chosen the wrong branch).
On the development rows 41 of 52 W_PDI_single_attractor runs (0.79) are
concordant ABSENTs under A-R and A-H, just below 0.80. HCv2-22(ii) for that
witness stays NOT_TESTABLE_BY_DESIGN (pi0 0.79 < 0.9). HCv2-21 was corrected
(section 10.3).

**k-means seed.** Rule: the design principle (reproducible, not tuned on data,
the frozen estimator equals the calibrated one). Outcome: **0**. Every PDI v3
count (the run, 19 surrogates, 10 jackknife replicates) seeds k-means for k
states with `RandomState((0 x 1009 + k) mod 2^32)`. The value comes from the v2
development feasibility work, was set before any development output, was never
compared with an alternative, and all 4620 development PDI outputs used it (the
v1 bench seeded k-means with the task's null seed). Limitation: the jackknife
SE and the concordance flag do not include k-means initialisation variability;
k-means++ with 8 starts keeps it small.

### 7.3 Anchors, testability, dependencies and replication (CD-7, CD-11)

**Anchors** (section 3.6). A-R: all five principles valid and specific (F0),
40/40 finite; anchors IIM 0.00679, NAS receive 0.00241 and return 0.02222, PDI
2.429, RAM 0.388, SRPI 0.171; paired contrasts 0.00716, 0.00241, 0.02222,
2.415, 0.384, 0.168. A-H: NAS, PDI, RAM and SRPI valid and specific (NAS
receive contrast 0.00448 against the required 0.00353, return 0.01180 against
0.00732); IIM valid at the boundary (36/40 finite: 4 runs
`INSUFFICIENT_OCCUPANCY`; mean 0.00337, lower bound 0.00192) but non-specific
(contrast -0.00182, lower bound -0.00305); `N_anch(A-H)` = {RAM, PDI, NAS, SRPI}
(F1). C1-R and C1-H: NAS and IIM valid and specific (IIM anchor 0.00115 and
0.00113); PDI, RAM and SRPI invalid (0/40 finite: NOT_APPLICABLE by
declaration); `N_anch` = {NAS, IIM} (F1). A-RAM160: RAM valid and specific
(anchor 0.545, contrast 0.559; F2). Secondary forms: the A-H forms
`+nas_secondary` and `+iim_bidirectional` are non-specific; all other forms are
valid and specific. The HCv2-6(a) predictions are these statuses.

**Forward anchors.** Computed once, after the held-out predictions were
committed and released, on development seeds 900-939 at the held-out regime
(the Hopf `G_nom` runs and the PC_nominal runs of both family-A arms; 120 tasks,
0 errors), validity only (FM0). No G = 0 reference run was made: the "G = 0 x
40" item of the design's development budget is withdrawn as an erratum, because
such a run would be estimator output on a held-out condition (FMa, HO-6). The
v1 quadrant comparator protocols receive their own validity-only anchors from
the quadrant pipeline on the same released `G_nom` runs; they do not inherit the
v2 cluster pipeline's anchor. Outcomes are in the table of section 3.6; the
120 development-regime forward anchors (lead-field seed 20260928, width 0.5)
served development builds only.

**Testability** (section 3.7): computed with the decided SE methods, IIM df 9
and the PDI route; the C1 cells of HCv2-22(ii) for PDI, RAM and SRPI are not
cells of that part (their targets are not in `N_anch(C1)`; the witnesses'
components are `UNDEFINED(NOT_APPLICABLE_OBSERVATION_MODEL)` in 12/12
development runs each).

**Declared dependencies for HCv2-22(iii).** The structural dependencies of the
generator fixed by the design, as in v1: (W_IIM_feedforward, NAS), the c_int
relay, and (W_NAS_no_workspace, IIM), the workspace loop. Development may
confirm a declared pair but may not add one. Share of paired seeds with
`|c_p(w) - c_p(PC)| < 0.25`: first pair 5/52 (A-R), 1/52 (A-H), 0/52 (C1-R),
0/52 (C1-H); second pair 0/52 (A-R), 3/40 (C1-R), 4/40 (C1-H). No pair was
added on the basis of development failures, including W_NAS_broadcast_only
(knob ff_only), which v1 did not declare; adding it would have made HCv2-22(iii)
more likely SUPPORTED. The cells below 0.8 stay in the test (for example A-R
RAM on W_PDI_single_attractor 14/52, A-R IIM on W_NAS_broadcast_only 30/52,
A-H NAS on W_PDI_single_attractor 28/52, C1-R IIM on W_NAS_broadcast_only 8/40),
and HCv2-22(iii) stays predicted FALSIFIED. The cell prediction of
(W_PDI_single_attractor, NAS) in family A was FALSIFIED under both declarations
in the file committed before development (`d3bcb09`); after the development
rates were seen it was changed (`21597fe`) to FALSIFIED under A-H (28/52) and
SUPPORTED under A-R (44/52 = 0.846). A cell prediction is reported beside the
cell and decides nothing (the part's outcome comes from its rule), and the
part's prediction (FALSIFIED) is unchanged, but this is a prediction moved
towards development and is listed as a deviation (section 10.5). The same
development fact (44/52 under A-R) left the prediction of HCv2-7(v-A), which
decides on that cell, unchanged (P(SUPPORTED | development) 0.002).

**Replication blocks (CD-11).** Rule: extend a block to 20900-20939 where the
power that a development anchor status recurs is below 0.9 at 20 seeds; seeds
are resized, never thresholds. Power by resampling the reference-block seeds
(2000 draws, seed 20261006, confirmed with two independent streams):

- Family A: below 0.9 at 20 seeds only for A-H|IIM (0.565; A-H|NAS 0.9625;
  every other pair 1.0). The family-A block is extended to **20900-20939**
  (PC_nominal and the six single deficits, 140 additional tasks). A-H|IIM
  reaches only 0.6045 at 40 seeds: its development finite rate sits exactly on
  the 36/40 criterion, so P(at least 90 % finite) is 0.677, 0.629, 0.593 and
  0.566 at 20, 40, 80 and 160 seeds and no resize can reach the target.
  HCv2-6(a) is therefore labelled INDETERMINATE-capable in the design's
  sense (design 4.9: below 0.8, no resize helps). Its rule
  (`anchor_replicates`) has no INDETERMINATE zone, so the alternative outcome
  is FALSIFIED: P(SUPPORTED | development) about 0.60, P(FALSIFIED |
  development) about 0.4; A-H|IIM is the pair most likely to fail.
- C1 and A-RAM160: power 1.0 at 20 seeds; not extended.
- Forward views (validity only, computed from the held-out reference anchors
  after the release): below 0.9 at 20 seeds only for three Hopf IIM views,
  `hopf-eeg64_noref` 0.264, `hopf-mne_template` 0.699 and `hopf-eeg64` 0.862
  (the latter's development status is invalid); every family-A forward view is
  at 0.9745 or above. By the rule the **Hopf arm's** block is extended to
  **20900-20939**; neither family-A arm is extended. At 40 seeds the power is
  0.5225 (`hopf-eeg64_noref`), 0.9535 (`hopf-mne_template`) and 0.7925
  (`hopf-eeg64`, lower than at 20 because replicating an invalid status gets
  harder with more seeds). These are reported; no threshold changes. HCv2-6(b)
  needs every view to replicate: with the views treated as independent (they
  share seeds) the part's P(SUPPORTED | development) is about 0.38 (Hopf views
  at 40 seeds 0.5225 x 0.9535 x 0.7925 x 0.999, family-A EEG IIM views at 20
  seeds 0.9865 x 0.9745, every other view 1.0), and the alternative outcome is
  FALSIFIED. The hypothesis HCv2-6 needs both parts, so its P(SUPPORTED |
  development) is about 0.6 x 0.38, roughly 0.23.

### 7.4 Mechanism labels, agents, adversaries, oracle checks, null calibration, grain cap

**CD-8, mechanism-on labels** (used by HCv2-5). Rule: a row is mechanism-on for
a principle under a protocol only if its dose is at least 0.5 x nominal and the
development median `c` at that dose, for that principle and protocol, is at
least `2 delta = 0.20`; the first matching entry decides; an unmatched row is
not on. Development (dry-run sweeps and factorial, 4 seeds per level or cell;
witnesses, 12 seeds; families A and C1; R and H): over the 1140 entries of the
eight mechanism witnesses, the sweeps and the factorial, the dose condition
alone labels 810 on and the rule 512; all 298 changes are from on to off, 258
of them C1 RAM-PE, PDI and SRPI entries (NOT_APPLICABLE on C1, never ABSENT).
The 40 substantive changes are cells where the dose is on but the v2 estimate
is near zero (for example family-A IIM under R in the eight factorial cells
with g_b off, median `c` 0.048-0.063; C1 NAS in the eight factorial cells with
c_int off, 0.135-0.150). The rule also applies to the three witness systems
whose doses leave mechanisms on besides their target
(W_PDI_no_multistability, W_NAS_common_input_control, N_modules_disconnected),
which adds 18 family-A on-cells. *Deviation:* development IIM on C1 is held out
(HO-4), so for C1 IIM under R and H the label is the dose condition alone;
where the C1 reference blocks give a median at nominal dose it confirms "on"
(PC_nominal 1.01 / 0.98, W_NAS_broadcast_only 0.50 / 0.51,
W_NAS_no_workspace 0.40 / 0.41 under R / H). Rows that are on by dose but off
under the rule are reported beside HCv2-5 (HCv2-5(dose-only)), so estimator
blindness stays visible.

**CD-6, IIM on the agents with the common basis** (expectation only;
thresholds unchanged). Family A, reference block 900-939 and witness seeds
320-331 and 340-351, bootstrap SE at df 9. Under R: median paired Delta c
(PC_nominal - W_IIM_feedforward) 0.90 over 52 pairs, 49 >= 0.5, Hodges-Lehmann
one-sided 95 % lower bound 0.91; PC_nominal PRESENT 58/64 = 0.906, so (d) is
decisive by one row; W_IIM_feedforward PRESENT 0/52 and ABSENT 12/52, so (c)
stays NOT_TESTABLE_BY_DESIGN (pi0 0.23); O_inert PRESENT 0/12; Spearman rho of
`c` with c_int 0.09 (one-sided p 0.28), an inverted U (median `c` 0.97 at c_int
0.53, 0.06 at 1.2). Under H: median Delta c -0.61 over 47 pairs, consistent
with (e); PRESENT PC_nominal 24/64, W_IIM_feedforward 29/52, O_inert 11/12
(`c` 1.95-4.78): a system with every mechanism off reads as integrated when the
drivers are hidden (reported in HCv2-14(H-present)). C1 disclosure: section
10.2.

**CD-9, adversary potency** (descriptive, D3; never used to select or drop a
condition). Under H a construction is potent against an estimator if, in at
least half of the development seeds 340-351, the paired ratio to PC_nominal is
>= 0.5, or PRESENT >= 50 %. Against IIM under H every hidden-driver
construction is potent (common driver PRESENT 11/12; staggered driver 8, 9 and
8 of 12; reflex arc 12/12); with the drivers declared (R) each falls to 0/12.
Against PDI the three staggered variants are potent (ratio about 1.08 on every
seed), a false reading. random_label_self_other and scrambled_feedback are
potent against every non-target principle (those mechanisms are on in them).
Not potent against their own targets: NAS on the staggered and common drivers
(PRESENT 0-1/12), RAM-PE on the reflex arc (0/12), SRPI on random self/other
labels (0/12). scrambled_feedback is RAM-mechanism-on (RAM-PE PRESENT 11/12).

**CD-13, oracle and realisation checks** (prerequisite M). Recomputed on
development seeds 320-359 after the realisation checks of the design were wired
into prerequisite M: 26 of 27 checks pass 40/40, among them every switch check
of families A and C1, every new-system check and the slow-context check
(shortest complete context run 30.0-34.5 s against 30 s).
ADV_NAS_staggered_tau10 fails "driver reaches every module" at 34/40 against
the 36 needed; the six failures (0.035-0.048 against the floor 0.05) are in the
8-s modules. The condition, its check and its threshold stay as specified: the
10-fold time constants define held-out element HO-2. At the development pass
rate P(usable) is 0.26 at a true rate of 0.85 and 0.63 at 0.90, so the
condition is about 74 % likely to be unusable; then the HCv2-0(b) cell is
FALSIFIED and HCv2-9(d) is NOT_EVALUABLE (reason ORACLE). Reported: PC_half
lies between the off and nominal medians in 10 of 12 family x switch cells
(not for K in A and C1); the twins pass 1920/1920.

**RAM-PE on the null-calibration generator** (`null_calibration_ram =
require_defined`). All 128 development RAM-PE components on the 16
null-calibration cells are `UNDEFINED(INSUFFICIENT_UPDATES)`, because the
generator has no trial or feedback structure; counted as non-events they would
satisfy HCv2-1 for RAM-PE vacuously. HCv2-1 and HCv2-3 therefore count RAM-PE
rows only where defined (a finite estimate and no definedness reason);
decision-type UNDEFINED outcomes stay non-events; RAM-PE stays tested on the
family-A null witnesses (development: 48 defined rows, 0 PRESENT); the
undefined records stay in the integrity audit. Consequence (CD-11): to fall
below 0.07 at level 0.05/5 the RAM-PE pool needs at least 64 seed clusters with
0 events, or 92 to tolerate one event. **The family-A null witnesses
N_independent_noise and N_ar1 are resized to 46 seeds each (20000-20045; 92
clusters)**, an author-level decision under the design's rule "resize seeds,
never thresholds" (section 10.4). Thresholds are unchanged.

**IIM macro-grain cap** (run setting 4). Every run computes IIM v5 on at most
4 macro nodes, the declared grain of every Tier-A substrate. All 4477
non-smoke development records ran with this cap, and none of their 6186 IIM
components was refused by it. A refused grain would be
`UNDEFINED(ESTIMATOR_ERROR)`, never ABSENT.

### 7.5 Constants, paper-2 regime, forward anchor regime, Tier B (CD-1, CD-12, CD-14)

**CD-1, deterministic constants** (no estimator involved). `tau_c = 0.1 s` on
every bench substrate; input basis time constants {0.1, 0.3, 1.0} s; lag sets
as section 3.2. Exact family-B anchors (bidirectional / directional): ring 0.45
0.04513 / 0.02909; all-to-all 0.4 0.06837 / 0.02736. Non-monotone ring at
coupling 0.45 / 0.9 / 1.5: bidirectional 0.04513 / 0.03621 / 0.04014;
directional 0.02909 / -0.00973 / -0.03161, `UNDEFINED(NULL_MODEL_VIOLATED)` at
0.9 and 1.5. The ring sweep 0-0.5 and the xor-loop sweep are strictly monotone
in both cut modes; ring 0.1 is exactly ABSENT (`c` 0.079 and 0.070, below
`delta`). The constants file (SHA-256
`d636eeb35e0f0565a21b33fd0186826f14b00cdb91ad28d97bd21a3d4caeb1ff`) was
recomputed from the code at `38245c3` without difference. The Tier-B2
reciprocal-gain metric is not frozen in this round.

**CD-12, paper-2 regime.** Rule: `n_low` is the smallest EEG channel count among
the paper-2 EEG datasets (from their BIDS metadata, 32 if unavailable); the
reference scheme, TR, duration and input declaration of the paper-2
confirmatory data are declared. *Interpretation (a deviation, fixed before any
held-out anchor existed):* the minimum is taken over the paper-2 confirmatory
EEG datasets (the mediated and open subsets of the DREAM database and the
requested Selte et al. (2026) high-density dataset), because admission licenses
confirmatory exclusions; none of them publishes BIDS EEG metadata. Outcome:
**`n_low = 32`** (the fallback, equal to the code default; every development
forward record used it); **average reference** (IIM also without reference);
**TR not applicable** (no confirmatory fMRI in paper 2); **no declared inputs**
(sleep awakenings without a within-state task); **250 Hz**; **duration 60 s**,
the v2 admission window. This is an author-level decision (section 10.4):
paper 2 registers its admission-dependent EEG exclusions at this regime (a 64-
or 32-channel subset, average reference, 250 Hz, zero-phase 1-40 Hz band, 60-s
pre-awakening windows, no declared inputs, no per-window z-scoring unless the
admitted estimators are shown to be scale-invariant per channel); analyses with
other windows, the 0.5-45 Hz band, the 20-Hz sensitivity band or other montage
sizes are exploratory in paper 2. The benchmark settings were not changed to
fit paper 2. Montages below `n_low` have no forward admission (their components
stay `UNDEFINED(ESTIMATOR_NOT_VALIDATED)`). PDI on EEG is admitted only on the
forward family-A arm (20 Hz, unfiltered, complete input declaration), which no
paper-2 recording matches.

**Forward anchor regime** (design 3.6): held out. The order before the freeze
was: the paper-2 regime declared; the held-out predictions committed with the
documented HCv2-21 correction; the release (19:24 CEST); the anchor run
(19:24-19:50); two code commits made after the anchor run had been seen,
`f7709d3` (the builder records a protocol whose every anchor is invalid on the
complete reference block as a result; no value changes: its components stay
UNDEFINED) and `a17d82d` (the Hopf replication block extended to 20900-20939 by
the CD-11 rule); the decisions file set from `draft` to `final`, differing from
its pre-release state only in that extension; the rebuild of the protocols
(`f3a4812`, 20:11; section 9).

**CD-14, SRPI v3 pair count:** not applicable (`srpi_v3_pairs = null`). Tier B
was not run; the v1 SRPI outcome stands; HCv2-B1 to HCv2-B8 are listed as not
run.

### 7.6 Commitments made before any calibration output, and the decisions

Written on 2026-10-07 before the full development calibration started, when no
battery, twin or dry-run output of the v2 estimators existed (only the one-task
trial runs of the runner test, written to a scratch directory and not used):

1. **The PDI concordance battery has 90 seeds (CD-5).** Zero events in 300 runs
   is the smallest sample whose one-sided 95 % Clopper-Pearson upper bound is
   below `alpha_A = 0.01` (0.00994; 0.0106 at 280). The planned battery (seeds
   870-899) gives 150 content-on and 120 no-content runs and cannot admit
   either route; the battery therefore runs on 850-899 and 940-979 (450 and 360
   runs per cell; 850-869 were unassigned and 940-979 spare). The optional
   extension item (940-979 only) is removed.
2. **When an SE method counts as calibrated (CD-2, CD-3, CD-4).** As section
   7.1: the HCv2-4(a) statement (point in [0.8, 1.25], 90 % interval inside
   [0.67, 1.5]) and both `q_A` tails at most 0.02. CD-2: among the methods
   calibrated in every class, the largest df, ties to contiguous, none:
   contiguous G = 10 with the failure reported. CD-3 and CD-4: keep the default
   unless it fails somewhere and the fallback is calibrated in every class; a
   method that is not better is never swapped in. The `iim_bootstrap_se_df`
   rule is unchanged. The fallback twin items and the other optional items run
   together with the planned calibration, so the rules apply in one pass.

These commitments are binding over the design, and the design over the
protocol builder's mechanical suggestions. A logical error found later may be
corrected only with a written reason (section 10.3).

| Decision | CD | Value |
|---|---|---|
| NAS SE method | CD-2 | `jackknife_contiguous_10` (no method calibrated on the twins; failure reported) |
| IIM SE method | CD-3 | `circular_block_bootstrap_10pct_B50` (the fallback is not better; deviation from the builder's suggestion) |
| IIM bootstrap df | CD-3 | 9 |
| IIM occupancy gate | CD-3 | `N_min = 25` |
| RAM-PE SE method | CD-4 | `shift_null_sd` |
| PDI concordance route | CD-5 | the rule: one cell (synthetic_rate / source / source / non_workspace, both directions) |
| Declared dependencies | CD-7 | (W_IIM_feedforward, NAS), (W_NAS_no_workspace, IIM) |
| Mechanism-on labels | CD-8 | the rule (C1 IIM by dose alone) |
| Replication blocks | CD-11 | family A and the Hopf arm extended to 20900-20939; C1, A-RAM160 and both family-A forward arms not |
| Paper-2 regime | CD-12 | `n_low` 32, average reference, no TR, 60 s, no declared inputs |
| SRPI v3 pairs | CD-14 | null (Tier B not run) |
| PDI k-means seed | open | 0 |
| RAM-PE on the null-calibration generator | open | `require_defined` |
| IIM macro grain cap | run setting | 4 |
| Forward anchor regime | design 3.6 | held out |

## 8. Operating characteristics

DEVELOPMENT - NOT A RESULT. Before the freeze each decisive part was given its
operating characteristics (CD-11): `P(SUPPORTED | development rates)`, with the
target 0.8, and `P(FALSIFIED | correct estimator)`, with the target 0.05, by
resampling the part's development rows by cluster to the confirmatory number of
clusters (the engine's own rule functions; B = 1000 replicates, fewer for the
slow rules) and by exact binomial sums. A part predicted SUPPORTED below 0.8 was
resized (more seeds), never re-thresholded; where no resize reaches 0.8 the part
is labelled here and the outcome is reported as it falls. Parts predicted
FALSIFIED are judged by the probability of their predicted outcome and are never
resized. The full table, the method and the checks are in
[`v2/operating_characteristics.md`](v2/operating_characteristics.md).

**Seed resizes made (seeds only; no threshold, rule or prediction changed):**

| Part | Design and systems | Before | After | P(SUPPORTED \| development) |
|---|---|---|---|---|
| HCv2-1 (RAM-PE pool) | family-A N_independent_noise and N_ar1 | 20 seeds each | 46 each (20000-20045; 92 clusters) | 1.000 (0 development events; the pool tolerates one event at 92 clusters) |
| HCv2-23 | C1 W_NAS_no_workspace, W_NAS_broadcast_only, W_IIM_feedforward | 45 seeds (20000-20044) | 129 (20000-20128) | 0.069 -> 0.884 |
| HCv2-14(f) | c_int sweeps of families A and C1, 10 levels | 10 seeds per level | 65 (20000-20064) | family-A cell 0.16 -> 0.82; the C1 cell is held out (HO-4), so the part's value is at most this |
| HCv2-6(a) | family-A replication block | 20900-20919 | 20900-20939 | about 0.60 (no resize reaches 0.8; see below) |
| HCv2-6(b) | Hopf forward anchor replication | 20900-20919 | 20900-20939 | see section 7.3 |

The HCv2-23 resize works only by diluting a pooled bound. Its any-event
clusters are keyed by the seed, and the family-A event rate in development (3
of 52 clusters, from the NAS SE reversion under H) sits just below the bound
0.07; the resize adds C1 clusters that were free of events in development. That
event-freedom could not include IIM, which is held out on C1 (HO-4), and rests
on NAS never being PRESENT on the three C1 targets (0 of 72 rows). With a
Jeffreys predictive on both rates instead of the plug-in, P(SUPPORTED) at 129
seeds is 0.39. The per-class FALSIFIED clause of the part (a Clopper-Pearson
lower bound at 0.05/m above 0.07) still applies to the family-A classes on
their own.

**Parts predicted SUPPORTED with development operating characteristics below
0.8 that no seed resize helps** (labelled here; thresholds unchanged; outcomes
reported as they fall):

| Part | P(SUPPORTED \| development) | Why no resize helps |
|---|---|---|
| HCv2-0(b) | 0.16 at 40 seeds | ADV_NAS_staggered_tau10 driver-reach check 34/40 < 36/40; more seeds lower it (0.056 at 80); FALSIFIED-capable |
| HCv2-2 | about 0 | development rank exceedance above the bound in two bidirectional cells (T1000 3/15, T10000 4/15; pooled 18/270 = 0.067) |
| HCv2-4(a,b) | about 0 | the SE calibration failures of CD-2 and CD-3; even SEs calibrated to kappa = 1 give at most 0.24 over its cells |
| HCv2-5(a) | 0.18 | the PDI concordant cells receive about one seed cluster (the only development concordant PC_half row, 1 of 12); INDETERMINATE whenever such a row appears |
| HCv2-6(a) | 0.58-0.63 | A-H\|IIM finite share on the 0.9 edge; INDETERMINATE-capable in the design's sense, but the rule has no INDETERMINATE zone: the alternative outcome is FALSIFIED (about 0.4) |
| HCv2-6(b) | about 0.38 (views treated as independent; three Hopf IIM views below 0.9 each, section 7.3) | replication of validity on the edge of the criterion; the alternative outcome is FALSIFIED. Hypothesis HCv2-6 about 0.23 |
| HCv2-7(iii) | about 0 | C1 relay: NAS PRESENT 0/12 in development |
| HCv2-7(v-A) | 0.002 | A-R NAS on W_PDI_single_attractor 44/52 >= 0.8: the predicted v1 failure is not seen under R |
| HCv2-8(a-H) | 0.41 | 9/12 = 0.75 < 0.8 |
| HCv2-8(b-H-rank1) | 0.09 | reversed variant 8/12 = 0.67 < 0.8 |
| HCv2-12(d) | 0.09-0.10 | O_hypersynchronous MACRO_RANK_DEFICIENT 9/12 = 0.75 < 0.9 |
| HCv2-15(a) | about 0 | development-regime exceedance 2/20 > 0.07 in three sensor views (a proxy for the held-out regime) |
| HCv2-15(b) | about 0 | v1 quadrant pipeline PRESENT 0/20 at G = 0 at the development regime |
| HCv2-17(d) | about 0 | development median paired Delta c(K = 1 - PC) 0.13 (Hodges-Lehmann lower bound 0.106; dry run FALSIFIED), below `z`. The part's note in the hypotheses file ("development +0.40") quotes the design-time value and is superseded by this one; the prediction and the threshold are unchanged, and the part is expected to fail |
| HCv2-20(a), HCv2-20(f-single) | 0.51-0.52 | concordant ABSENT 41/52 = 0.79 < 0.80; P falls with more seeds (0.31 at 400) while P(FALSIFIED) rises |

Parts predicted FALSIFIED: HCv2-3 (P(FALSIFIED) 1.0), HCv2-22(iii) (1.0) and
HCv2-22(ii) (0.002: with the PDI cell not testable, the testable cells are
predicted SUPPORTED, so the predicted outcome of the part is unlikely). Parts
without a development rate (held out or without development rows), whose
operating characteristics cannot be computed: HCv2-9(a-e), HCv2-10, HCv2-15(c),
HCv2-15(e-FMd), HCv2-20(d) and HCv2-21. HCv2-12(b) has a synthetic operating
characteristic from the exact steps (0.995 at 40 seeds and the development SD;
0.85, 0.61 and 0.30 if the SD at couplings 0.9 and 1.5 is 1.5, 2 and 3 times
larger). HCv2-14(c) is NOT_TESTABLE_BY_DESIGN at its gate. Parts above 0.8 that
miss the `P(FALSIFIED | correct) <= 0.05` target and cannot be fixed by seeds:
HCv2-8(b-R) (0.125: three variant cells at alpha each), HCv2-0(a) (a binary
usable share at the 0.9 boundary) and the two-outcome value rules, whose
`P(FALSIFIED | correct) = 1 - P(SUPPORTED)`.

**Development dry run** (the planned item `dry_run_evaluation`, CD-10; about
15 % of the confirmatory scale; companion `v2/development_expectations.md`):
hypotheses SUPPORTED 5 (HCv2-6, -11, -16, -18, -19), FALSIFIED 8 (HCv2-0, -3,
-4, -7, -14, -15, -17, -22), INDETERMINATE 9 (HCv2-1, -2, -5, -8, -12, -13, -20,
-23, -24), NOT_EVALUABLE 3 (HCv2-9, -10, -21: held-out conditions and the
held-out-regime registry have no development rows), NOT_RUN 8 (Tier B); decisive
parts SUPPORTED 47, FALSIFIED 10, INDETERMINATE 13, NOT_EVALUABLE 13,
NOT_TESTABLE_BY_DESIGN 1. Many INDETERMINATE outcomes reflect the small
development sizes; the operating characteristics above are computed at the
confirmatory sizes.

## 9. Held-out elements and their release

What remains held out (no v2 estimator output before the freeze; only oracle
checks, reference-block anchors where stated, and smoke tests on seeds 980-984
whose outputs the harness discards unread):

| Id | Element | Held out | Tested in |
|---|---|---|---|
| HO-1 | declaration misspecification | label errors in the declared context (q = 0.1, 0.25), missing slow phase (partial declaration), jittered cue onsets, for NAS and IIM on family A | HCv2-9 |
| HO-2 | out-of-scope drivers | staggered driver with its time constants multiplied by 10 (S = C 0.5 s, hub 3 s, M = V 8 s; mostly outside the basis span 10 tau_c = 1 s) and with a saturating post-filter transform | HCv2-9 |
| HO-3 | forward regime | EEG lead-field seed 20261001, conduction width 0.6, the paper-2 low-density montage, and the template-lead-field inverse at that regime; development used seed 20260928 and width 0.5 | HCv2-10, HCv2-15, HCv2-21 |
| HO-4 | IIM v5 on family C1 | IIM-dir with drivers recorded on C1 (only reference-block anchors are computed before the freeze) | HCv2-14 (C1 parts) |
| HO-5 | IIM non-monotone regime | ring coupling 0.9 and 1.5 with the v5 estimator and occupancy gate | HCv2-12(b) |
| HO-6 | IIM sensitivity to Hopf coupling at sensor level | G sweep of the v2 sensor pipeline | HCv2-15(e) |
| HO-7 | Tier B | echo time constants 6 s and 12 s; zero-mean workspace broadcast witness; mis-declared PDI access node on the content-on witnesses | HCv2-B3, HCv2-B6, HCv2-20(d) |

**Committed predictions.** The held-out predictions are the HO-labelled parts
of `hypotheses_v2.json` and the predicted admission table (section 4.5). They
were written from mechanism in the design, except the corrected HCv2-21 values,
which are development-informed (label R; section 10.3), and were first committed
in the hypotheses file on 2026-10-05 at 02:47 CEST (`6d46f49`). Since then the
HO-labelled parts (HCv2-9(a)-(e), HCv2-12(b), HCv2-15(e-FMb1), HCv2-15(e-FMd),
HCv2-20(d)) and the C1 parts of HCv2-14 (HO-4) have not changed in rule,
parameters, prediction or data selection; expectation texts were added to
HCv2-9(d) and HCv2-12(b) in `21597fe`, before the release. The admission parts
decided at the held-out regime had their data bindings adjusted in `d3bcb09`
(2026-10-06; HCv2-10 and HCv2-21), and the HCv2-21 predictions were corrected
in `21597fe` (section 10.3). The held-out
elements and the admission table were transcribed into
`protocols/v2/held_out_predictions_v2.json` (SHA-256 `a0d291e066406128f5dc67b9e6368f8b33b5df2ee71806008bb2b72951b5f726`),
committed on 2026-10-08 at 16:40:40 CEST in commit `21597fe`, with the one
documented correction (HCv2-21; section 10.3). Tests compare the held-out
predictions of the hypotheses file with their commit and with the
transcription; HCv2-21 is the one listed exception.

**Release.** The release of the held-out forward anchors was logged in
`outputs/mpcbench_v2/dev_calibration/held_out_release_log.jsonl` (release id
`5af6e074475de23a`, 2026-10-08 19:24:11 CEST, repository head `a16f84c`,
naming the predictions file above with its commit and SHA-256, clean and
tracked). The 120 held-out-regime reference tasks (development seeds 900-939)
ran only after it (19:24-19:50 CEST), with 0 errors. Two code commits followed
the anchor run, `f7709d3` and `a17d82d` (section 7.5), the decisions file was
set to `final`, and the protocols were then built (`f3a4812`, 20:11 CEST). What
the released anchors fix about the predicted admission table is stated in
section 4.5 and section 10.2. No released file may change afterwards, and none
has:
`tests/v2/test_prereg_v2_consistency.py` checks that the predictions file was
committed before the logged release and is unchanged since. Edits of the
hypotheses file after the release concern no held-out part (only its top-level
status was set to `final` with this document).

## 10. Integrity, prior contact and deviations

### 10.1 Held-out integrity of the development calibration

An audit of the development calibration outputs at `38245c3` (24 runner result
files, 3 family-B result files and 29 provenance files, 182 files in all, the
superseded start included) found: task seeds 320-979 (271 distinct), no file
with a seed of 1000 or more; runner records with split development; no record
with a held-out declaration (A-P, A-Q10, A-Q25, A-J, mis-declared PDI access);
no record with a held-out staggered driver; no forward record at the held-out
lead field or width; the family-B runs leave out the five held-out ring cells
at couplings 0.9 and 1.5; IIM on the Hopf sensor views appears only at G = 0
(the runner withheld 432 scorings); the smoke tests on seeds 980-984 stored
status lines only, and their outputs were not read. **Violations: 0.** That
audit ran on 2026-10-07 at 15:31 CEST, before the release, so it did not cover
the later items: an oracle-only recomputation (`oracle_checks`, seeds
320-359), the held-out-regime forward anchors after the release (the permitted
reference runs), and the dry-run evaluation (stored development records
re-judged, no estimator run). The same read-only audit was therefore run again
on 2026-10-08 at 21:39 CEST over every runner and family-B result file of the
development root, which are the 28 result files listed as inputs in
`build_manifest.json` (no estimator ran): task seeds 320-979 (271 distinct),
no seed of 1000 or more, no record outside the development split, no
held-out declaration, staggered driver or family-B ring cell at 0.9 or 1.5,
and smoke files with status lines only. Its only flags are the 120 released
held-out-regime reference tasks (lead-field seed 20261001, width 0.6) and
their 240 IIM sensor scorings at `G_nom`, which are the permitted reference
runs made after the release (section 10.2). The protocol builder applies the same
development policy to every input it reads (seeds, smoke files, held-out
outputs) and records the SHA-256 of each input file in
`protocols/v2/generated/build_manifest.json`.

### 10.2 Declared prior contact

- **HO-4.** The C1-R and C1-H IIM reference-block anchors (seeds 900-939) were
  computed on 2026-10-07 between 10:47 and 11:06 CEST, after the HCv2-14
  predictions were committed (`6d46f49`, 2026-10-05; the rule, parameters,
  predictions and data selection of the HCv2-14 parts are unchanged since).
  HO-4 permits them. They show: median paired Delta c 0.76 under
  C1-R (33/40 >= 0.5) and 0.82 under C1-H (32/40), contrary to the frozen limit
  prediction HCv2-14(e) ("median < 0.25"); PC PRESENT 0.40 (C1-R) and 0.25
  (C1-H); the specificity of PC_nominal against W_IIM_feedforward was 0.80 of
  the anchor under C1-H (lower bound 0.71) and 0.785 under C1-R (0.69). These
  are development analogues of HCv2-14(e), the decisive part, and of
  HCv2-14(a), (b) and (d) on C1. The same run scored IIM on the C1
  W_NAS_no_workspace and W_NAS_broadcast_only reference runs. As the design's
  gate rule prescribes for every reference block, the C1 testability gates of
  HCv2-14(c) and (d) read these reference-block rates (PRESENT on PC_nominal
  0.40 and 0.25; ABSENT on W_IIM_feedforward 0/40 under both declarations;
  section 3.7). This made (c) and (d) NOT_TESTABLE_BY_DESIGN on C1, and (a)
  decides in place of (d). No threshold or prediction was changed, and no seed
  count used these values: the C1 c_int sweep was resized by symmetry with
  family A, without any C1 IIM value, and the operating characteristics pool
  the C1 reference blocks nowhere except the HCv2-6(a) anchors. The C1 IIM
  anchors also enter HCv2-6(a) as predictions. The note of HCv2-14 in the
  frozen hypotheses file (quoted in section 6.2) names `ac13c2d` as the commit
  of the C1 predictions and says that no testability decision used these
  values; both statements are imprecise, and this paragraph corrects them (the
  file is not edited).
- **HO-3.** The 32-channel montage was already used at the development lead
  field; only lead-field seed 20261001, width 0.6 and the template-lead-field
  inverse at that regime are new.
- **Released held-out-regime anchors** (release `5af6e074475de23a`, runs
  19:24-19:50 CEST on 2026-10-08, development seeds 900-939). The reference
  runs at the held-out regime were made after the predictions were committed
  and released, as the design prescribes; their outcomes were seen before the
  freeze and are frozen in `forward_anchors.json`. They fix FM0 for every
  admission entry: FM0 holds for NAS on the Hopf source, eeg64, eeglow and
  mne_template views, for IIM on the Hopf source, eeg64_noref, eeglow and
  mne_template views and on the forward family-A EEG and source views, and for
  PDI on every forward family-A view; it fails for IIM on `hopf-eeg64` (64
  electrodes, average reference), `hopf-bold` and `fwdA_bold-bold`. Against the
  predicted admission table this means: NAS and PDI as predicted so far (FM0 is
  necessary, not sufficient); IIM on eeg64 at average reference cannot be
  "absent yes or vacuous" and IIM on BOLD cannot be "absent vacuous or yes"
  (both will be "absent no", or `not_observable` on BOLD; section 4.5). The
  same runs gave the v2 sensor pipeline's IIM values at `G_nom` on these seeds
  (the anchors' means), which are one end of the FMd contrast of
  HCv2-15(e-FMd) and one level of the FMb1 trend of HCv2-15(e-FMb1). Those
  HO-labelled parts are therefore held out only for `G != G_nom` and for the
  confirmatory seeds; no run at another G was made at the held-out regime
  (the G = 0 anchor item was withdrawn, section 7.3). No prediction was
  changed after the release.
- **Confirmatory seed 20000, generator only.** During the integration of the
  code on 2026-10-05/06, four systems were generated at confirmatory seed 20000
  to check the plumbing; only their time-series hashes were compared. No
  estimator ran and nothing was stored; the check was then repeated on
  development seeds 5 and 820. The record of the integration does not name the
  four systems.
- **Superseded start of the calibration.** The first start of the full
  development calibration on 2026-10-07, 08:53-09:30 CEST, at code `3e48f14`,
  covered the constants, the oracle checks and 189 of 280 family-A anchor
  tasks on seeds 900-939 (185 completed, 4 failed: the four seed-923 tasks of
  PC_nominal, W_RAM_no_plasticity, W_NAS_no_workspace and W_NAS_broadcast_only
  with "SVD did not converge"). An earlier note in the superseded folder and
  the decision log gave 172 tasks and three failures; the counts here are those
  of the results file, and the folder's note was corrected. Its outputs are not
  used and contain no held-out condition; they are kept apart
  (`outputs/mpcbench_v2/_superseded/`).
  The calibration restarted from an empty directory at `38245c3`, after the SVD
  fallback was added, so every used output comes from one code version.

### 10.3 Corrections and fixes before the freeze

Corrections of logical errors, made before the release:

- **HCv2-21** (forward PDI admission). The original `admitted_for_absent`
  predictions (yes on EEG-64 and EEG-low; no on BOLD, "K = 3 read as one
  state"; committed in `d3bcb09` and kept in the part's
  `original_predictions`) assumed the exact path that design decision 9 had
  already rejected: a concordant PDI count outside an admitted concordance cell
  is `UNDEFINED(NO_SAMPLING_SE)`, never ABSENT, and the CD-5 battery admits the
  source cell only. The corrected values follow mechanically from the admission
  rule (absent needs FM0 and FMabs; vacuous iff `pi0 < 0.1`) on the
  development-regime forward rows (seeds 384-399 and 804-819, lead-field seed
  20260928, width 0.5) judged under a development-regime build: on every view
  FM0 holds, none of the 22 on-runs per view is ABSENT and every null run is
  `UNDEFINED(NO_SAMPLING_SE)`, so `pi0 = 0` and each view is **vacuous**
  (EEG-64 and EEG-low: present yes, absent vacuous; BOLD: absent vacuous). The
  PRESENT predictions are unchanged. CD-5 lists HCv2-21 among the parts it
  affects. The logical error justifies withdrawing the original values; the
  replacement values rest on development-regime rows and are therefore
  development-informed predictions (label R), not predictions written from
  mechanism alone (section 9).
- **HCv2-12(b)** (wording). The part predicts the sign of the exact change at
  each step: negative from 0.45 to 0.9 in both cut modes; from 0.9 to 1.5
  positive for the bidirectional and negative for the directional form. The
  earlier wording "never a rising estimate" contradicted the exact
  bidirectional rise at the second step. The rule (`stepwise_sign`) and the
  thresholds are unchanged.
- **HCv2-20(a)** (gate). The gate compared the family id "A" with the cell's
  substrate `synthetic_rate` and would always have chosen the "not admitted"
  branch; it now names the substrate.
- **HCv2-19(b)** (window, 2026-10-05). The v1 masking window 0.67-1.56 left out
  the sweep level 2/3, stored as 0.666667; it reads 0.666-1.556 and holds all
  five levels.

Fixes found during the calibration (each with a test; no threshold or rule
changed):

- **SVD fallback.** numpy's divide-and-conquer SVD raised "SVD did not
  converge" on an exactly rank-deficient input basis (seed 923 of the family-A
  reference block). The v2 estimators and the forward layer now call numpy
  first and repeat the computation with SciPy's `gesvd` (`gelsy` for least
  squares) only when numpy raises, so every result numpy computes is
  unchanged. The first calibration start was superseded (section 10.2).
- **The `p_ind` field path (HCv2-2, HCv2-13(a)).** The field map read the IIM
  rank p-value at `details.estimator.p_ind`, where only runner records have
  it; family-B validation records carry it at `details.p_ind`, so HCv2-2 was
  not evaluable and HCv2-13(a) silently lost its exceedance cells. The field
  now reads `details.p_ind`, which runner records fill with the same value
  (`a16f84c`, before the release).
- **HCv2-3, admitted concordant rows.** An admitted concordant PDI component has
  SE 0 and no sampling SE; HCv2-3 took such rows into kappa0 and the tails. The
  null rule now leaves them out (`exclude_concordant`), as the twin rule of
  HCv2-4 does, keeps the cell in P, and reports their number
  (HCv2-3(concordant)). In development all 152 PDI null rows are concordant, so
  the PDI cell is expected NOT_EVALUABLE.
- **Realisation checks wired into prerequisite M.** The realisation checks of
  the design (slow-context preset; PC_half between off and nominal; twin
  hashes) are computed with prerequisite M; the slow-context check gates the
  rows of the forward BOLD arm, the other two are reported. The oracle item was
  recomputed on development seeds 320-359 (section 7.4).
- **Protocols without a valid anchor** (`f7709d3`, after the release and the
  held-out-regime anchor run, which showed the case). A protocol whose every
  anchor is invalid on the complete reference block was reported as "anchors
  pending" and kept the build from being ready for the freeze. Such a protocol
  keeps no anchor, so each of its components is UNDEFINED and its verdicts
  UNDETERMINED by construction: `hopf-bold` and `hopf-eeg64+iim_v1_quadrants`
  at the held-out regime. The builder now records this as a result; no value
  changes.
- **Protocol builder.** Brought in line with the decided calibration before
  the final build: concordant components skipped in the SE calibration with a
  defined-share eligibility, the binding switch rules of CD-2 to CD-4, the
  concordance route dropped on the mis-declared-access form and the
  BOLD-observed forward forms, testability gates that honour the part's target
  filter, mechanism-on labels for every witness, family-B components re-judged
  at the decided bootstrap df, own anchors for the quadrant forms, and the
  forward replication decided per arm.

### 10.4 Author-level decisions confirmed with the freeze

Two decisions of the round go beyond the mechanical application of a written
rule. They were taken before the held-out release and are confirmed by the
author in the message of the annotated freeze tag `mpcbench-freeze-v2`, which
names both; pushing that tag makes the confirmation part of the record:

1. **HCv2-1 resize of the null witnesses to 46 seeds.** The design's rule is to
   resize seeds, never thresholds; the alternative was to state that HCv2-1
   could only be INDETERMINATE or FALSIFIED for RAM-PE (section 7.4).
2. **Paper-2 regime (CD-12):** `n_low` 32, average reference, 60-s windows, no
   declared inputs; paper 2 registers its admission-dependent EEG exclusions at
   this regime, other windows are exploratory (section 7.5). The author
   confirms this knowing its likely consequence (section 4.5): if the
   predictions hold, the only possible admission-dependent EEG exclusion at
   this regime is IIM ABSENT on a view that passes FM0 and FMabs, the 64-channel
   average-reference view already fails FM0, and paper 2 must plan for having
   no admissible EEG exclusion.

If the author changes either decision, the tag is not made: the plan and the
generated protocols are rebuilt where the change requires it, this section and
the sections it affects are revised, and the change is reported.

### 10.5 Deviations from the design

- The SE calibration criterion of design 2.1 item 9 was replaced before any
  data by the HCv2-4(a) reading (section 7.6).
- The PDI battery was enlarged to 90 seeds before any battery output, and no
  forward battery was run (no forward concordance route; section 7.2).
- IIM SE method kept against the builder's suggestion (the rule's outcome;
  section 7.1).
- C1 IIM mechanism-on labels by dose alone (section 7.4).
- CD-12 rule taken over the paper-2 confirmatory datasets (section 7.5).
- The "Hopf G = 0 x 40" development anchor item withdrawn (section 7.3).
- Seed resizes of HCv2-1, HCv2-23, HCv2-14(f) and the replication blocks
  (section 8); the seed map's `extended` list does not name the Hopf extension
  (section 5).
- HCv2-3 keeps its a-priori cell predictions (NAS, IIM, PDI, RAM SUPPORTED;
  SRPI FALSIFIED), while its development expectation differs (NAS
  FALSIFIED on the conservative side, IIM and RAM-PE INDETERMINATE, PDI
  NOT_EVALUABLE). The predictions were not moved towards development.
- **HCv2-22(iii), cell prediction moved towards development.** The cell
  prediction of (W_PDI_single_attractor, NAS) in family A under A-R was changed
  from FALSIFIED to SUPPORTED after the development rate 44/52 had been seen
  (`21597fe`). The cell prediction is reported, not decisive; the A-H cell
  prediction (FALSIFIED) and the part's prediction (FALSIFIED) are unchanged.
  HCv2-7(v-A), which decides on the same cell, kept its prediction although the
  same development fact makes its failure likely (section 7.3).
- Expectation texts in the hypotheses file were restated from development for
  HCv2-0, -1, -3, -4, -6(a), -7(v-A), -9(d), -12(b), -14, -20(a) and -22; no
  threshold, rule or prediction changed with them. The other changes to the
  hypotheses file after development outputs existed are listed in the table
  below.

**Changes to the hypotheses file after `d3bcb09`** (2026-10-06, the last
commit before the development calibration started on 2026-10-07; every change
below was made with development outputs in hand; expectation and note texts
are left out):

| Commit | Part or field | Change | Reason | Can it move an outcome? |
|---|---|---|---|---|
| `21597fe` | HCv2-1 (data, text) | RAM-PE rows enter only where defined; family-A N_independent_noise and N_ar1 on 46 seeds | null-calibration RAM-PE `require_defined` (section 7.4); the author-level resize (section 10.4) | yes: fewer rows in the RAM-PE pool, more clusters |
| `21597fe` | HCv2-3 (data) | RAM-PE rows enter only where defined | as HCv2-1 | yes |
| `1dd9096` | HCv2-3 (params, text) | `exclude_concordant`: admitted concordant PDI rows leave kappa0 and the tails | logical error (section 10.3) | yes: the PDI cell is expected NOT_EVALUABLE |
| `21597fe` | HCv2-4(a,b) (data, params, text) | per SE method (`by_se_method`); a class needs `c` and its SE defined in >= 80 % of its sessions | the SE-method decisions of CD-2 to CD-4 (section 7.1) | yes |
| `21597fe` | HCv2-6(a) (params, text) | predictions set to the frozen development statuses (A-H NAS `valid_specific` instead of `valid_nonspecific`; C1-H IIM `valid_specific` added); family-A block 20900-20939 | the part predicts the development status by its definition (section 3.6); CD-11 extension | yes, by construction |
| `21597fe` | HCv2-20(a) (gate) | the gate names the substrate `synthetic_rate` instead of the family id | logical error (section 10.3) | yes: the branch taken |
| `21597fe` | HCv2-21 (params, text) | admission predictions corrected; originals kept | logical error (section 10.3) | yes |
| `21597fe` | HCv2-22(iii) (cell predictions) | A-R cell FALSIFIED -> SUPPORTED | development rate 44/52 (above) | no (annotation; the reported cell prediction changes) |
| `21597fe`, `b610387`, `1dd9096` | reported parts HCv2-1(definedness), HCv2-5(dose-only), HCv2-14(H-present), HCv2-0(c), HCv2-0(d), HCv2-3(concordant) | added | reporting | no (reported only) |
| `b610387` | HCv2-0 (statement), HCv2-0(b) (text), `oracle_checks`; HCv2-6(b) and HCv2-21 (`requires` oracle) | the design 3.5 realisation checks join prerequisite M; the slow-context check gates the forward BOLD rows | wiring of design 3.5 (section 10.3) | yes: HCv2-0(b) counts more checks; BOLD rows can drop (ORACLE) |
| `21597fe`, `52b7e1c` | `seed_blocks` | resizes of HCv2-1, HCv2-23, HCv2-14(f) and the replication blocks | CD-11 (section 8) | yes: sizes |
| `21597fe` | `declared_dependencies`, `mechanism_on`, `pending_calibration` | the CD-7 and CD-8 tables written in; no calibration item left pending | CD-7, CD-8 (sections 7.3, 7.4) | yes: HCv2-5 and HCv2-22(iii) read them |
| `a16f84c` | `fields` | `p_ind` read at `details.p_ind` | defect (section 10.3) | yes: HCv2-2 and HCv2-13(a) become evaluable |
| this document | `status` | `draft` -> `final` | the freeze | no |

Any deviation during the confirmatory run or the evaluation is reported with its
reason in the paper. Results obtained after a change to `src/` or `scripts/`
are exploratory: the confirmatory guard refuses to run such code under the
freeze tag.

## 11. Confirmatory run plan and cost

All steps: `scripts/v2/mpcbench_confirmatory_v2.sh` from a clean checkout of the
tag `mpcbench-freeze-v2`, with 12 workers, on the machine of the development
runs, free of other runs. Every step goes through the v2 confirmatory guard
(clean tree, the `src/` and `scripts/` trees of the tag, seeds >= 20000) and
uses the generated protocols only (`--no-drafts`). Before the run, from the
clean checkout of the tag:

```bash
export MPCBENCH_V1_OUTPUTS=/path/to/outputs/paper1_mpcbench   # the stored v1 outputs (the folder holding c2/)
python scripts/v2/env_lock.py --check          # darwin-arm64, Python 3.10.19, numpy 2.2.6, scipy 1.15.2, numba 0.61.2, pandas 2.3.3
python scripts/v2/regression_gate.py --quick
git diff --quiet mpcbench-freeze-v2 -- protocols/v2 && echo "protocols/v2 equal to the tag"
python scripts/v2/build_protocols_v2.py check  # every generated file and protocol hash against the manifest
python scripts/run_bench_v2.py plan --split confirmatory           # the runner plan (6933 tasks)
python scripts/v2/iim_validation_v2.py --split confirmatory --list # the family-B plan (4260 tasks)
```

The stored v1 outputs are not versioned; the regression gate (and IA-1 of the
audit) reads them through `MPCBENCH_V1_OUTPUTS` and fails with exit status 2
without them. The confirmatory guard compares only the `src/` and `scripts/`
trees with the tag, so the `protocols/v2` comparison and the build check are
part of the pre-run checks; both must pass.

Order (most decision-relevant first): prerequisite M (seeds 20000-20039); the
anchor replication blocks; family-A witnesses; C1 witnesses; the
null-calibration generator; the Hopf arm and forward family A (EEG, then BOLD
with curtailed sampling); family B (`scripts/v2/iim_validation_v2.py`, behind the
same guard); twins; the RAM-only arm; sweeps and factorial; adversaries and
held-out conditions. Tier B is not run (`TIER_B=0`). Sizes: section 4.2.

**Cost.** By the development cost model (CPU seconds per task and design,
measured at `d3bcb09` on development seeds; contention factor 1.39 at 12
workers): **11193 tasks (6933 runner, 4260 family B) and the oracle checks,
about 130.8 CPU-h, about 15.1 h wall at 12 workers.** The development cost of
the C1 witness, sweep, factorial and twin tasks (3.05-4.21 s) was measured
without IIM, which is held out on C1 in development; the confirmatory C1 tasks
score IIM under R and H and the bidirectional form under both. Priced at the C1
reference-block rate (74.56 s per task) the plan is **about 167.4 CPU-h, about
19.4 h wall**. The confirmatory cost is therefore expected between about 131
and 167 CPU-h (15-19 h wall); the forward BOLD arm is priced uncurtailed (271
tasks). The design projected about 106 CPU-h before the resizes. Per-design
figures: companion `v2/operating_characteristics.md`, section 5.

**Failures.** A step that ends with task errors does not stop the later steps;
a refused step stops the plan, and an unknown step name stops the script before
anything runs. When the script exits with 1, it names the steps with task
errors; the operator then runs the script once more with `STEPS` set to those
steps and the same `OUT`, and records that rerun in the run log. Resuming a
step runs its failed tasks again and skips the completed ones (the runner and
the family-B script drop the error records, superseded records and a line cut
off by an interruption before they continue). A task that ends with status
`error` is thus run again once; a task that still fails is reported and enters
no hypothesis (the integrity audit lists it and excludes it; IA-2 bounds the
component error rate at 1 % per design). A component that raises is
`UNDEFINED(ESTIMATOR_ERROR:<type>)` and stays in the records. No seed is added,
replaced or dropped for any other reason, and there is no optional stopping:
all planned runs are made (the curtailment of the forward BOLD arm and of the
admission on-runs is part of the plan), then evaluated once.

**Where.** The run is planned on the workstation that holds the locked
environment. HLRS Hunter is not needed for the plan. If it were used, it would
have to be a CPU-only run in an environment that matches the lock (the default
Hunter stack does not), with HLRS's agreement to CPU-only work; that would be
reported as a deviation ([runbook](../HLRS_HUNTER_RUNBOOK.md#151-mpc-bench-v2-confirmatory-run)).
By the cost model the whole plan fits within one 24-h job on a single node at
12 or more workers; per-design CPU hours for packing steps into shorter jobs are
in the companion `v2/operating_characteristics.md`, section 5.

## 12. Evaluation and reporting rules

After the runs, from the tag, on the confirmatory outputs:

1. **Registry v3** from the confirmatory forward arms:
   `scripts/v2/build_registry_v3.py --results <OUT>/forward --anchors
   protocols/v2/generated/forward_anchors.json --run-id <run id> --purpose
   confirmatory --out <registry.json> --report <report.json>`.
2. **Integrity audit** (IA-1 to IA-10), with `MPCBENCH_V1_OUTPUTS` set as in
   section 11: `scripts/v2/integrity_audit.py --split confirmatory --records
   <OUT> --protocols protocols/v2/generated --plan <OUT>/*/run_plan.json
   <OUT>/family_b/iim_validation_v2_plan.json --registry <registry.json>
   --run-gate full --out <audit.json>`. The run plans of the runner steps and
   the family-B plan together list every planned task id (IA-7); the registry
   lets IA-4 check that no ABSENT stands under an unadmitted direction; the full
   gate includes the v1 reruns of IA-1 (about 20 minutes).
3. **Evaluator**: `scripts/bench_hypotheses_v2.py --freeze-tag mpcbench-freeze-v2
   --records <OUT> --manipulation <prerequisite-M tables> --registry
   <registry.json> --protocols protocols/v2/generated --audit <audit.json>
   --out <evaluation>`. It writes `hypotheses_v2.json`,
   `hypotheses_v2_parts.csv` and `hypotheses_v2_tally.csv` with every
   hypothesis, part and cell and the hashes of every input. It refuses records
   outside the confirmatory split or without the freeze tag, an evaluator
   checkout that is not the frozen tree, a hypotheses file that differs from its
   version at the tag, records whose family-protocol hash differs from the frozen
   protocol, an integrity audit that is missing, has a blocking failure (section
   4.6), is incomplete or was run on other records, manipulation checks from
   another split, a mechanism-on table given outside the frozen hypotheses
   file, and a confirmatory evaluation without the frozen protocols. A failed
   audit without a blocking failure does not stop the evaluation: the records
   it names leave every hypothesis and the audit outcome is reported beside the
   tally.

Reporting rules:

- **Outcomes as they fall.** Every hypothesis and every part is reported with
  its outcome, whatever it is, beside its prediction and its development
  expectation. A part whose rule mirrors a predicted failure
  (`reversed_three_zone`, `h0_cell_exceeds`: HCv2-7(v-A), HCv2-24(b)) is
  SUPPORTED when the failure occurs. A part or hypothesis whose stated
  prediction is FALSIFIED (HCv2-3, HCv2-22(ii), HCv2-22(iii), HCv2-22) is
  reported FALSIFIED when the failure occurs, beside its prediction
  (`matches_prediction`); the tally counts outcomes as the rules return them,
  not agreement with the prediction. The final tally lists SUPPORTED,
  FALSIFIED, INDETERMINATE, NOT_EVALUABLE and NOT_TESTABLE_BY_DESIGN parts and
  the Tier-B hypotheses as not run.
- **INDETERMINATE** means the rule did not decide at the planned size. It is
  reported as such, never re-read as support or failure, and no seeds are added
  after an outcome is seen.
- **NOT_EVALUABLE** is reported with its reason (ORACLE when prerequisite M
  dropped the rows; no input rows; an invalid anchor; a missing registry
  entry). It is never treated as support, and it enters the tally.
- **NOT_TESTABLE_BY_DESIGN** cells were declared before the freeze; their
  replacement claims decide and are reported.
- **Reversion rule.** If HCv2-4 is FALSIFIED on the anti-conservative side for
  an SE method (expected for NAS), every verdict-level analysis is reported
  both with the ABSENTs of that estimator and with them re-classified as
  `UNDEFINED(SE_NOT_CALIBRATED)`.
- **Descriptive analyses** (D1-D13) and figures are descriptive and change no
  decision.
- **No re-runs after outcomes,** except reruns that reproduce identical
  results. The independent recheck follows v1: from-scratch reruns of a
  stratified sample of about 14 tasks and a check of every quoted number. A
  results summary is built as for v1, every number read from a result file,
  with the SHA-256 of every source.
- **v1 stays final** (section 1).

## 13. Frozen files

| Artefact | Identity |
|---|---|
| Code | the commit of the annotated tag `mpcbench-freeze-v2` on branch `polish/hlrs-handoff-2026-09`; that commit contains this document and its companions (a document cannot quote its own commit). The v2 confirmatory guard compares the git trees of `src/` and `scripts/` with the tag. |
| Hypotheses | `protocols/v2/hypotheses_v2.json` (status `final`), SHA-256 `c9486dac60f4083f20ba5c0ce4e6be1bac9ffbf806249fe4c5f03382134fd935`; spec hash `8e37e2c04ae5cb31828d693787ac094f90fce2374a892d795732714ee5dbdefb` (`spec_sha256`: SHA-256 of the canonical sorted-key JSON, which the evaluator and the audit record in their outputs). The file's `version` (`mpc-bench-hypotheses/2.0.0-draft`) and `design` texts still say draft; they name the drafting round, not the status, which is the `status` field. |
| Held-out predictions | `protocols/v2/held_out_predictions_v2.json`, SHA-256 `a0d291e066406128f5dc67b9e6368f8b33b5df2ee71806008bb2b72951b5f726` (commit `21597fe`; named by release `5af6e074475de23a`) |
| Seed map | `protocols/v2/seed_map_v2.json`, SHA-256 `c9cf929f24350eca2b9f03a4be208e3844e7e8c7f9a65391c544d550f6cb7626` |
| Bench template and paper-2 default protocol | `protocols/v2/mpc_bench_v2_template.json` `8a8ba5b4c05a0a31b3a49897f2988eab2020045a5b174db1767fadddc8559766`; `protocols/v2/mpc_default_v2.json` `98bbd94fb5efcd9c31daa1f1865947636886f92f786b8dd925fa257cdef74c2b` |
| Generated protocols and tables | `protocols/v2/generated/` (builder `mpc-bench-protocol-builder/1.0.0`; `build_manifest.json` with `freeze_ready` true, no blocking entry, decisions status `final`) |
| Calibration decisions | `outputs/mpcbench_v2/decisions/calibration_decisions_v2.json`, SHA-256 `d3d5bd5da0043ab750979853f9c73932032b08a3c3e36dac31f14d8b881f1744` (not versioned; copied with its evidence into `protocols/v2/generated/calibration_decisions.json` and `calibration_evidence.json`) |
| Development inputs of the build | listed with their SHA-256 under `inputs` of `build_manifest.json` (files under `outputs/mpcbench_v2/dev_calibration/`, not versioned) |
| Run plan | `scripts/v2/mpcbench_confirmatory_v2.sh` |
| Evaluator and audit | `scripts/bench_hypotheses_v2.py`, `scripts/v2/integrity_audit.py`, `scripts/v2/build_registry_v3.py` |
| v1 | tag `mpcbench-freeze-v1` and its protocols, unchanged (IA-1) |

Generated protocols (the protocol hash is `ProtocolV3.hash`, the SHA-256 of the
canonical protocol; the file SHA-256 is that of the file as written):

<!-- frozen-protocols:begin -->

| Protocol key | File (`protocols/v2/generated/`) | Protocol hash (`ProtocolV3.hash`) | File SHA-256 |
|---|---|---|---|
| `A-H` | `mpc_bench_v2_A-H.json` | `1616cfc357c69cf1e0a968f89cc972a770657bad4b647c97974fb58abae3fc65` | `7f805b4947bd79453a226fbb9b5e6ed0b938b0b66aaabd052b6608c1513b886f` |
| `A-H+iim_bidirectional` | `mpc_bench_v2_A-H+iim_bidirectional.json` | `9256b71e829d04982a0658c5ad108b4207d8c5f2bcd076f58fb9d8ce2207cd1d` | `e2860c43242517bcdeea06a2a7fe91b0629104a8cc116058d5b9170ccd7a238b` |
| `A-H+nas_secondary` | `mpc_bench_v2_A-H+nas_secondary.json` | `b3d4d2bd8d4e1f79281add5958bd1455bb22eed8ef76a15ac66bab11b7b89073` | `9925d78cd03d1015ab80023bb517414e554a8e05119f43d226e29195352e86a8` |
| `A-J` | `mpc_bench_v2_A-J.json` | `f60d4e9c803c0d4fcef494e89d8e72801837c8b0f5d8fdfefb7557380d95a83e` | `ccf79db43af57f367322e248411293a1accd841b78967ace7a85cd01b598e16b` |
| `A-P` | `mpc_bench_v2_A-P.json` | `5588c08c4e3792d558a936490fe7c11c16ba30338fdba3f61b64d8f94ae306c0` | `20ea75521186f5eaef62dce666da02b66c4a76edab5fc48795204b9b12b106df` |
| `A-Q10` | `mpc_bench_v2_A-Q10.json` | `087653800941b1b82115a9d9d835f28b8dc6a988e61902569c503e2cc243d997` | `da9cd5c2070c2fb1e3084ee707be0ce8a08dc92412bc34c6fa3ac53317055edf` |
| `A-Q25` | `mpc_bench_v2_A-Q25.json` | `704edbc9cb77047bb621646093834fdccb3587caa8589d928c5f4690ca555b63` | `eac5b7c9b31024adeaf43c484a086c5b63abee1fd0b9ce9c58148ee248f038e3` |
| `A-R` | `mpc_bench_v2_A-R.json` | `00f25190fc91593be91d981ff3295c01c254f6322bb2a6bb78472ed9ae5328cc` | `aad23fd128f8e96a03ae3b8c7332d9033b281a80f75e45bf3452ec6d353a967e` |
| `A-R+iim_bidirectional` | `mpc_bench_v2_A-R+iim_bidirectional.json` | `04d9f3206b8157e91078fe4c2646e5aa3f50ad52e93d9e7cfdf8aa98d1d01c25` | `f49ecd5d6f691f71868f574776f2f34b0ad228a18bb7aa01474f4e15915f6e19` |
| `A-R+nas_secondary` | `mpc_bench_v2_A-R+nas_secondary.json` | `5fc8e2d48158619577030b6209ef2f9f37c36154aabf2bcdf58b403c66ab309b` | `346e6e75b255ae31702ff7a7e4f4a2c4639ab7189113b50c0e57f3c4d2a26d6e` |
| `A-R+nas_tau_0.05` | `mpc_bench_v2_A-R+nas_tau_0.05.json` | `2fca6d823c057c1746d4599e15d80b3a824fadf4c8b1105323074074cd4dc553` | `18856cabdde0b2ee4cfafd8d96029f6a70383eb90b29adbb3f5dc261d00fe253` |
| `A-R+nas_tau_0.2` | `mpc_bench_v2_A-R+nas_tau_0.2.json` | `f4ca8da7185bee9b5db996d43f146392fdfefa5b4589190fc8d484379548f48e` | `0f349f73492489b70216555fbb35bad9cc491ce1f617c8f95edd2027c881b1db` |
| `A-R+pdi_misdeclared_access` | `mpc_bench_v2_A-R+pdi_misdeclared_access.json` | `b7064c7ba8eec07277eb2d1b505702ef47b546693612fd4707e2188b1463b312` | `8a537570c7c7923b08dc058b18cb49817fd947b4942c74f5339db0b0ad51b491` |
| `A-RAM160` | `mpc_bench_v2_A-RAM160.json` | `d31fbdd61010a8f287553af683829791737706476ed7fd6c583019442a12ee9e` | `1944e0ea0bb0f0e2bd099409de136fff43943e328bbdf3fbd850f64e7446b638` |
| `A-none` | `mpc_bench_v2_A-none.json` | `e2ccb4293561c407f0ad2921f19560446f9933bb953c22b1331313d013562dc1` | `cc6e780db5b93b7af384ee80847d26a3936cca75a660436aa866dc724c566e04` |
| `B` | `mpc_bench_v2_B.json` | `ee3efaaba08018539cbb3a78b485104615d4ee91f664df323aaed021cbaf79fc` | `997eafa76692adb6d1326d87072e3c09d033997c92a68ba2cae2fcd9811fc9e3` |
| `B-all_to_all_0.4-bidirectional-circular_shift` | `mpc_bench_v2_B-all_to_all_0.4-bidirectional-circular_shift.json` | `7a5ef2679094e0f3d420aa57b4460385ae9a52827de5ebce7f89390972b9786c` | `835134c4bdf2add123b00a046898df2bd0213187880b0a98e802c7ee77a6a5b7` |
| `B-all_to_all_0.4-bidirectional-circular_shift-values` | `mpc_bench_v2_B-all_to_all_0.4-bidirectional-circular_shift-values.json` | `3b11aa9a006186313293cd43e2f811bd762c03fa8a47549b2d1e857181903181` | `bd6429fa61d19a2a993dfbf813447529b10397c42c76976845a554db378fb7ab` |
| `B-all_to_all_0.4-bidirectional-stratified_pair_rotation` | `mpc_bench_v2_B-all_to_all_0.4-bidirectional-stratified_pair_rotation.json` | `1c6c98d5c381eacd6f54b69011f0f6283913f8bf421545487904b5fa40c54683` | `74c133ad3a023e29c5cf065562ff5ffe31d1cbc9ec61e00f47da75da33f48687` |
| `B-all_to_all_0.4-bidirectional-stratified_pair_rotation-values` | `mpc_bench_v2_B-all_to_all_0.4-bidirectional-stratified_pair_rotation-values.json` | `d769c1d7a735f180a2e2009b049ca5d3b4e6a03033a780f6a6cfa1ade6bb4b1c` | `0e95945836f03cfe8b7ef112bb35cf0c93ee959050fe809cceb2cf240a23c51b` |
| `B-all_to_all_0.4-directional-circular_shift` | `mpc_bench_v2_B-all_to_all_0.4-directional-circular_shift.json` | `abe60a58086957d19c424b1acdd801fcfafab4b9c0763d6aad185f037a32ee31` | `f8258fbb0c2fcf5bd85b8965968a6d4d3c4aac2645e55f065e580a195cc74a68` |
| `B-all_to_all_0.4-directional-circular_shift-values` | `mpc_bench_v2_B-all_to_all_0.4-directional-circular_shift-values.json` | `faeb352c514b3970b30c93e34b4d40a313786913274de4cae33e10bf1c11fedf` | `fe7d68e898cbbabd8c426144bf3dfd44dddc361e76929576b1f6850dab5f2bad` |
| `B-all_to_all_0.4-directional-stratified_pair_rotation` | `mpc_bench_v2_B-all_to_all_0.4-directional-stratified_pair_rotation.json` | `e4685fcc31180359ca55e25da9367343e34050a177b3a3d2bdac52e08db3982d` | `084e2754582caffad2fd39c574a216d77b53fb7f48c81d0bc1e4938d34a2d160` |
| `B-all_to_all_0.4-directional-stratified_pair_rotation-values` | `mpc_bench_v2_B-all_to_all_0.4-directional-stratified_pair_rotation-values.json` | `66aa30f792bbe5fce1b717e2bb94d6564e23e75ba0500b9df1e558e9de751e3e` | `82c20bb9aac2ee098caddaa45fbd0abfcb3bb84660b24ba4b2d9a817dbac1fe0` |
| `B-ring_0.45-bidirectional-circular_shift` | `mpc_bench_v2_B-ring_0.45-bidirectional-circular_shift.json` | `8401dee13ee3e6687ae2f4f4ea6854949bcdc83ade8eaba2ab6d2e1d471f4fef` | `247d8e4d249d025cfbdc36173c52e00d9fcf5a0489e4cda115487d7d49c28320` |
| `B-ring_0.45-bidirectional-circular_shift-values` | `mpc_bench_v2_B-ring_0.45-bidirectional-circular_shift-values.json` | `ad83431dc08cab0e9ae6f5abacc378173e957fa9ebf59413965d2fafeb6bf3fb` | `76b37df4704adb08039cbf0eee6f2d587290a968f638109edfba0723556a0163` |
| `B-ring_0.45-bidirectional-stratified_pair_rotation` | `mpc_bench_v2_B-ring_0.45-bidirectional-stratified_pair_rotation.json` | `a700774e6a39ee478c9c1361d79dba28f99a85a771e4c41212972902c0f8d359` | `22f80cb42c1287c10c81c69dc35ef7b78ba40ad3f2020002d2a865527655b700` |
| `B-ring_0.45-bidirectional-stratified_pair_rotation-values` | `mpc_bench_v2_B-ring_0.45-bidirectional-stratified_pair_rotation-values.json` | `ac686f3240b795b09830529805f26ff0a3f3fd8e5a4f952446590df409265a83` | `dd95e94be6181dc321310be951efa198dba7834a2d8b4bb3b797e9f7ee47ca22` |
| `B-ring_0.45-directional-circular_shift` | `mpc_bench_v2_B-ring_0.45-directional-circular_shift.json` | `c0998ddb353c72c226a1460b2ae7e2e3b36c582b3f757c2318d1814fb8693e05` | `bf7f7bdf2dbfaf1a6ef8f95d3331113e995a0ea4bf16b8ea5ce14d9e9cf36b10` |
| `B-ring_0.45-directional-circular_shift-values` | `mpc_bench_v2_B-ring_0.45-directional-circular_shift-values.json` | `743054dbe5becbec6a7fa1c6eac6425330c827c1e1d4151fae0acc213f2bab76` | `d401b1339a49f4da02ed039f6a2c46df33e1af9168af8fc723babd22a04a3c32` |
| `B-ring_0.45-directional-stratified_pair_rotation` | `mpc_bench_v2_B-ring_0.45-directional-stratified_pair_rotation.json` | `522ab31bf6f3ef96efc9b6f8fe1ec071ab130b2ab1bf602624fb9760142cee96` | `4bda7e5a1c27c45583f7bf35511dd9495edb00b1c228716e4c97195be3a6f8c4` |
| `B-ring_0.45-directional-stratified_pair_rotation-values` | `mpc_bench_v2_B-ring_0.45-directional-stratified_pair_rotation-values.json` | `7fa02a05eef7d74a00da12cf1b734c08ed66812714c96a38e461ae742797a167` | `3b099e54fb0ca4a1a4e37bf50153893eb772b07b2d945da83a3defb3e83a47bf` |
| `C1-H` | `mpc_bench_v2_C1-H.json` | `65bf4edea04ecb2f19e62efb4d804ce259a6b8183b14bd52f2cb30da63f7ec5f` | `d3bea6aecd52658b9a8bf8f51d39360eac6937cfa9b546996cc2edd21a8c0b58` |
| `C1-H+iim_bidirectional` | `mpc_bench_v2_C1-H+iim_bidirectional.json` | `c538e83e45c06de59e9abf5fec00c999263cd41f5cd4729a485f766e2fad3ddd` | `b455a0df389164cfc5cd5a704fec3f7f001f552bad6f9bf4b5a90c52c7924813` |
| `C1-H+nas_secondary` | `mpc_bench_v2_C1-H+nas_secondary.json` | `5a4f175dd35bae45ecb0652083e2f825f751eed606ae907518d486661feed14e` | `801e8a3d0e9bc637af74cccb80088d338c30d6881375143369fb4b514c48ba35` |
| `C1-R` | `mpc_bench_v2_C1-R.json` | `c1906594f8558f7f749b97a14c6143aa08ff0db43c0d7fa12bed02a41b452864` | `ea26321ebe3ad57a17b883227762c8db999e4888173f7a1b7e8b7c6c13ecf0f0` |
| `C1-R+iim_bidirectional` | `mpc_bench_v2_C1-R+iim_bidirectional.json` | `544145150e802547a5aedbca4da0953fa8a1babac12b9249bf95cdb97b308ea3` | `42b555baffc026579e667471c832f9976f193c5d5a44b7f0f08388ed299274ed` |
| `C1-R+nas_secondary` | `mpc_bench_v2_C1-R+nas_secondary.json` | `2f769261ad1db80025ba141d58a62d925b7914a34c9ce494071fb5f57c8bf43a` | `21a6ddad7d3d5fc6631cd8a1388a0eda73f4a50adcd39c9c4657d3fcbe1f9d9a` |
| `C1-R+nas_tau_0.05` | `mpc_bench_v2_C1-R+nas_tau_0.05.json` | `bb28e5bd0db1787787ab4229b5f7a7187f971bf8a8ee233554f52d64c4e5fffd` | `166cd9021e3bcc0dc998cd28a27b3efb66211e72809abe2657dfc2abfe0d575c` |
| `C1-R+nas_tau_0.2` | `mpc_bench_v2_C1-R+nas_tau_0.2.json` | `9d903459528769d68997efa1bac4818fa2b12aea17cc5823b92de849e99d5a6a` | `d2de69f700d733b30447ba9df634e66093315d8ea1046662c2ea2b8feec58ab0` |
| `fwdA-eeg64` | `mpc_bench_v2_fwdA-eeg64.json` | `46e65134d04cb128820c3f672935d3ccd7259ab551956332ac74a2505ea36cfb` | `687a37018f004aec7f0561f641fc261fc48c9b928a1c53b285c875d799852c7d` |
| `fwdA-eeglow` | `mpc_bench_v2_fwdA-eeglow.json` | `0ad74bafa80b2870afc7e3b16d0554695ebe5ea28305ca07d64b47ebc20cd643` | `774aae140a5a4f70e2bfe904322802c77685ccaee9e3577776da2684bcd24b93` |
| `fwdA-source` | `mpc_bench_v2_fwdA-source.json` | `877f2f690d7d26bcf09c66ef8799a92761437fddbc3eead5ebca146f0ff1a7de` | `0073c8a6856f1ead09168d67785e40f65b81b048b44cb3034eca82d1dfbc8425` |
| `fwdA_bold-bold` | `mpc_bench_v2_fwdA_bold-bold.json` | `fa3799494fea5e1ca7e8910397d56eade68cbb17b9949713a73a45760ade54fa` | `bd568fb9295c18ae06e0219607b55ec864ff24e8de237824831716f1c2fe79f1` |
| `fwdA_bold-source` | `mpc_bench_v2_fwdA_bold-source.json` | `4839b24fd4935e133e03165e86de2d2d547bec0c49398da0eff236f97cad67bd` | `561e88d19f69969df88a67f180a8b7e6d3264987400963a84bf3bd8f46174825` |
| `hopf-bold` | `mpc_bench_v2_hopf-bold.json` | `d582764c940d267bb22316b105c94fda9e31e5421d79d7c4f047b51eb590a45a` | `fb2ec0a850b2f6e4adfedcb235b278b600f589cef4dbcb37a70e5ca89733e4cd` |
| `hopf-eeg64` | `mpc_bench_v2_hopf-eeg64.json` | `0ad5f858224ad3ac3b609e4d563c2fa838e91a82e5cab49c39d2bd0296f891fb` | `3fc63442d774887ef810f204aefd9e68e459e557884ff7984e7611793b143709` |
| `hopf-eeg64+iim_v1_quadrants` | `mpc_bench_v2_hopf-eeg64+iim_v1_quadrants.json` | `126587ee7053515d656ebfe8a4f9bdb384b710677d4be0b93f8ce61b9579c6aa` | `0b30e66095b9c62b2f174e1c1e820d93b210b45d8b26f5a4bfe96c0857bab963` |
| `hopf-eeg64_noref` | `mpc_bench_v2_hopf-eeg64_noref.json` | `ef35e2ed6333159805900187a20e91e546f3f33280161699b924f44c42b737a6` | `13de361f94a4e15cc6bc842edd7167d2f1c1dcef6f360aff3fb7ab85675e4cc1` |
| `hopf-eeg64_noref+iim_v1_quadrants` | `mpc_bench_v2_hopf-eeg64_noref+iim_v1_quadrants.json` | `2fd3dbbad86d7e463fa13ed78b4f60143cff32e23d74dc27b8b196f6b2404480` | `f2fd54327c5f2c551d87487d5cc06705f220e1467072452f0cf8f85fee3c3d2c` |
| `hopf-eeglow` | `mpc_bench_v2_hopf-eeglow.json` | `48f40e647ba0303196c041f6b18bec0e5d604435d24f477cc4d840b00d32e911` | `2ff141f897f765afcb7246a3e73b59d214970497670d1cfc71fc2ab734073a05` |
| `hopf-mne_template` | `mpc_bench_v2_hopf-mne_template.json` | `973c146d725c1ce817c971e50c4b0fd1b36b76d6ccf7e70fb09eb78d88916675` | `5ca22d983d01bf21bd49d1fbb8d65fe43de2dcc444883cb909cc1ea1593067a3` |
| `hopf-source` | `mpc_bench_v2_hopf-source.json` | `50c4c9793bbf2b4433f32a205cb116a57d326978c677aeb4217a366bf378582c` | `d68a37911d71ba7477a4625231668d79a7ce70cbafaac1d0d56839a035000177` |

<!-- frozen-protocols:end -->

<!-- frozen-tables:begin -->

| File (`protocols/v2/generated/`) | SHA-256 |
|---|---|
| `calibration_decisions.json` | `c419ce60bef9dea93f235481c07fee51f06c9ab998c69b4fd3e72009938e20fe` |
| `calibration_evidence.json` | `d4a5c122b0e4f3bd825beeb4b79dacc82aaa899083e69ca5f960331d810caa6e` |
| `declared_dependencies.json` | `ca85de59a4314bef8f85b5fdf7a544644e20bae2ba2f8c441e51a558c1489c6b` |
| `forward_anchors.json` | `9eb2e03aa209ddd11ce1a314d38dabe3bb5c84baa42ded7eeba0bf0614bcc275` |
| `mechanism_on.json` | `711dfe87a5bf112e89c020398974bb390493988413cfe09e3f91f4340509141d` |
| `testability_table.json` | `90d9c999d52b8ae5f9ff26dbddddece3f9fd00c1f1db5547984b064b8a886717` |
| `build_manifest.json` | `36e92d1e5672d82cba765a5466df8c1d40288c3b4db0c2a7fdb737e7c1b85ec0` |

<!-- frozen-tables:end -->

The hashes can be recomputed with
`python scripts/v2/build_protocols_v2.py check`, which re-reads the build and
compares every file and protocol hash with the manifest. A rebuild from the
same decisions file and development inputs after the hypotheses file was set to
`final` reproduces every generated file byte for byte except
`build_manifest.json`, whose `downstream` list then no longer names the step
"hypotheses_v2.json status: final at the freeze" that this commit carries out;
the frozen manifest is the one of the build, unchanged.
