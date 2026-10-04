# Metric Definitions, v2: status rule `tost-v2` and protocol schema /3

This document describes the evidence layer of MPC-Bench v2: the status rule
`tost-v2`, the protocol schema `impact-mpc-protocol/3`, the reason codes of
the v2 layer and the pre-freeze rules for testability, anchors and necessity
sets. The code is `impact_pipeline.evidence_v2` and
`impact_pipeline.v2.testability`; the central vocabulary of reasons and flags
is `impact_pipeline.v2.reasons`.

The v1 layer is unchanged. [`metrics.md`](metrics.md) remains the definition
of every v1 estimator, of the construct scale and of the v1 status rule, and
every protocol of schema `impact-mpc-protocol/2` is still judged by the v1
code, byte for byte.

## Contents

1. [Scope and dispatch](#1-scope-and-dispatch)
2. [Status rule `tost-v2`](#2-status-rule-tost-v2)
3. [Consumption contract](#3-consumption-contract)
4. [Error control](#4-error-control)
5. [Protocol schema /3](#5-protocol-schema-3)
6. [Reason codes and flags](#6-reason-codes-and-flags)
7. [Testability, anchors and necessity sets](#7-testability-anchors-and-necessity-sets)
8. [Scale type and the absence margin](#8-scale-type-and-the-absence-margin)

## 1. Scope and dispatch

A protocol without a `status_rule` block has schema `impact-mpc-protocol/2`
and is judged by the v1 rule. A protocol with the block has schema
`impact-mpc-protocol/3` and is judged by `tost-v2`. The dispatcher
(`evidence_v2.mpc_verdict`, `evidence_v2.assess_component`,
`evidence_v2.load_protocol`) hands a /2 protocol, or no protocol, to the
frozen v1 functions with every argument unchanged, so v1 results come back
bit for bit. The v1 functions refuse a /3 protocol.

The decision of the rule is implemented once and reached through these entry
points; the test suite checks that they agree:

| Entry point | Input | Use |
|---|---|---|
| `assess_item(ev, protocol)` | one evidence item (`ComponentEvidenceV2`) under a /3 protocol | runner, component records |
| `assess_principle(P, items, protocol)` | the items of one principle (channels, directions) | runner, component records |
| `verdict_v3(evidence, protocol)` | the evidence of a run | verdicts |
| `assess_array(...)` | raw inputs, vectorised (the arguments of v1 `component_status_array`) | bulk re-judging |
| `status_c(c, se, df)` | the construct scale, vectorised | rule audit, simulations |
| `status_c_directional(c, se, df)` | directional members on the last axis | rule audit, simulations |

`c`, `se_c` and the degrees of freedom are computed exactly as in v1
(`evidence.component_assessment`, see [metrics.md, section 8.2](metrics.md#82-component-status)):
`c = (m - nu) / (rho - nu)`; `se_c` by the delta method over the sampling
SE, the Monte-Carlo error of the null mean and the reference SE; `df` by
Welch-Satterthwaite from the record's `se_df`.

## 2. Status rule `tost-v2`

With `q_P = t(df, 1 - alpha)` and `q_A = t(df, 1 - alpha_A)` (the normal
quantile when `df` is infinite or missing), `alpha = 0.05` and
**`alpha_A = alpha / |N_decl| = 0.01`** with the five declared principles:

- **PRESENT** iff `c - q_P se_c > z`;
- **ABSENT** iff `c - q_A se_c > -delta` **and** `c + q_A se_c < delta`: two
  one-sided tests (TOST) of equivalence to the null anchor within the margin;
- otherwise **UNDEFINED** with the first applicable reason:
  1. `NULL_MODEL_VIOLATED` iff `c + q_P se_c < -delta` (credibly below the
     null by more than the margin; the null does not fit this system);
  2. `ABSENT_NOT_REACHABLE` iff `q_A se_c >= delta` (the TOST region is
     empty at this precision);
  3. `INCONCLUSIVE` otherwise.
- Flag `PRESENT_NOT_REACHABLE` iff UNDEFINED and `q_P se_c >= 1 - z` (a
  component at the reference could not be PRESENT at this precision).

The cutoffs are the fixed core, `z = 0.25` and `delta = 0.10`. ABSENT and
PRESENT exclude each other because ABSENT needs `c < delta - q_A se_c <= z`.
The one-sided v1 rule also called a component ABSENT when `c` was credibly
below the null; `tost-v2` calls it `NULL_MODEL_VIOLATED`, UNDEFINED.

**Exact computations** (deterministic known-TPM values, `exact = True`
without a sampling SE, SE method `exact` or none) use the same inequalities
with `se_c = 0`: PRESENT iff `c > z`, ABSENT iff `|c| < delta`,
`NULL_MODEL_VIOLATED` iff `c < -delta`. This holds also when the null mean
is a Monte-Carlo estimate or the anchor has an SE: those parts are reported
(`se_null`, `se_reference`) but do not enter the decision. A data-derived
`se = 0` is never exact; it is `NO_SAMPLING_SE` unless the concordance
route applies (section 3).

**Implied precision.** ABSENT is reachable only when `se_c < s_A = delta / q_A`;
a component at the reference can be PRESENT only when
`se_c < s_P = (1 - z) / q_P` (`testability.absent_precision`,
`testability.present_precision`):

| df | s_A at alpha_A = 0.01 | s_A per NAS direction at alpha_A / 2 | s_P at alpha = 0.05 |
|---|---|---|---|
| 9 | 0.035 | 0.031 | 0.409 |
| 12 | 0.037 | 0.033 | 0.421 |
| 19 | 0.039 | 0.035 | 0.434 |
| 199 | 0.043 | 0.038 | 0.454 |

**Order of the checks** for one evidence item (the first that applies
decides the reason):

1. the estimator version differs from the protocol's
   (`ESTIMATOR_NOT_VALIDATED:not_admitted`);
2. the registry admits neither status direction
   (`ESTIMATOR_NOT_VALIDATED:not_admitted` or `:not_observable`);
3. the null family differs from the protocol's (`NULL_FAMILY_MISMATCH:<declared>/<used>`);
4. the estimator declared the value undefined (its reason when it is in the
   vocabulary, else `NOT_DEFINED:<its reason>`);
5. a non-finite estimate (`NON_FINITE_ESTIMATE`);
6. no null mean (`NO_NULL_CALIBRATION`);
7. a malformed null size or null SD (`DEGENERATE_NULL`);
8. an anchor that is missing, not finite or not above the null
   (`INVALID_ANCHORS`);
9. a negative SE or reference SE, `se_df <= 0`, or a break of the SE-method
   contract (`INVALID_SE`);
10. no sampling SE (`NO_SAMPLING_SE`), unless the admitted concordance route
    decides;
11. the decision above;
12. conditions that can only remove a decision: the IIM rank gate, the
    registry for the decided status direction, the SE reversion.

## 3. Consumption contract

Each principle feeds exactly one ABSENT decision into the verdict at
`alpha_A`. A principle with several members declares how they combine: a
union claim ("absent if either is negligible") is tested at `alpha_A / k` per
member; an intersection claim needs no correction.

**Directions (NAS).** NAS v3 reports a receive and a return direction, each
on its own anchor (`c_R`, `c_B`). The protocol declares
`directions = {"NAS": ["receive", "return"]}`. NAS is **PRESENT** iff both
directions are PRESENT at `alpha` (intersection-union test, no correction)
and **ABSENT** iff at least one direction passes the TOST at
`alpha_A / 2 = 0.005` (union, Bonferroni over the two directions). When
neither holds, the reason is the first validity reason of a direction, else
`NULL_MODEL_VIOLATED` if a direction has it, else `ABSENT_NOT_REACHABLE` if
no direction can pass the TOST, else `INCONCLUSIVE`; the flag
`PRESENT_NOT_REACHABLE` is set when some direction cannot reach PRESENT. The
reported value is `c_NAS = min(c_R, c_B)`; it is undefined when a direction
has no value and NAS is UNDEFINED, and an ABSENT through one direction while
the other has no value reports the value of the direction that passed. A
missing direction is
`MISSING_CHANNEL:<channel>:<direction>`: it blocks PRESENT, and ABSENT
through the other direction stays possible. Every item of a directional
principle names its direction.

**Channels.** The channels of a principle are combined by strong-Kleene OR:
ABSENT needs every channel ABSENT (an intersection, each at `alpha_A`), and
PRESENT through any of `k` channels is a union, so each channel's PRESENT
test runs at `alpha / k`. `k` is the number of declared channels, or of the
channels in the evidence when none are declared. A declared channel without
an item is `MISSING_CHANNEL:<channel>`, so the paper-2 default protocol, which
declares the unimplemented perturbational and endogenous RAM channels, can
never make RAM ABSENT through behavioural data.

**SE-method contract.** Every v2 component record carries `se`, `se_df` and
`se_method`; the rule reads `se_df` from the record, and the record must
agree with its SE method. `se_df` describes the variability of the SE
estimator, not a Monte-Carlo count:

| `se_method` | Principles | `se_df` |
|---|---|---|
| `jackknife_contiguous_10` | NAS, IIM, PDI | 9 (G - 1) |
| `jackknife_contiguous_20` | NAS | 19 |
| `jackknife_interleaved_10` | NAS | 9 |
| `jackknife_interleaved_20` | NAS | 19 |
| `circular_block_bootstrap_10pct_B50` | IIM | 12 (the block-count approximation 12.5 rounded down), or 9 if lowered at calibration; never B - 1 |
| `shift_null_sd` | RAM | `n_null - 1` |
| `jackknife_trials_10` | RAM | 9 |
| `jackknife_pairs_10` | SRPI | 9 (the v1 SRPI) |
| `hoeffding` | SRPI | Welch-Satterthwaite (any positive value) |
| `concordant` | PDI | none; `se = 0` (the concordance route) |
| `exact` | IIM | none; `se = 0`, `exact = True` |

An unknown method, a method not defined for the principle or not listed in
the protocol's `se_methods`, an `se_df` that breaks the method's rule, a
sampling SE on `concordant` or `exact`, or `exact` without the `exact` method
is `INVALID_SE`. Where the protocol's `se_methods` lists the admitted methods
of a principle, an item of that principle without an `se_method` is
`INVALID_SE` as well (otherwise a block bootstrap could report `B - 1`
unchecked); only where the protocol declares no methods does `se_df` keep
its v1 meaning for an item without one.

**Concordance route (PDI).** A PDI component whose resampled counts all
agree has `se = 0` and SE method `concordant`. It is not exact. It is
`NO_SAMPLING_SE` unless the protocol's `concordance_route` admits its cell
(principle, substrate, observation stage, view, content bearer). In an
admitted cell it is ABSENT iff `|c| < delta` (flag `ABSENT_BY_CONCORDANCE`)
and PRESENT iff `c > z` (flag `PRESENT_BY_CONCORDANCE`), each only if that
direction of the cell is admitted (otherwise `NO_SAMPLING_SE`); a value in
between is `INCONCLUSIVE`, one below `-delta` is `NULL_MODEL_VIOLATED`. The
route is chosen by the SE method, so v1 components with `se = 0` stay
`NO_SAMPLING_SE` exactly as recorded. Its error control is empirical over the
admission battery, not a sampling-theory guarantee.

**IIM rank gate.** With `rank_gates = {"IIM": 0.05}`, an estimated IIM
component is PRESENT only if also `p_ind <= 0.05` (the rank p-value against
the independence null). A failed gate is `INCONCLUSIVE:NULL_NOT_EXCEEDED`, a
missing `p_ind` is `NO_NULL_CALIBRATION`; the gate never creates ABSENT and
does not apply to exact values.

**Registry licensing.** A registry v3 licenses each status direction
separately (`admitted_for_present`, `admitted_for_absent`: `yes`, `no`,
`vacuous`, `not_observable`; `vacuous` admits). A PRESENT or ABSENT on a
direction that is not admitted becomes
`ESTIMATOR_NOT_VALIDATED:not_admitted` (or `:not_observable`); an item with
no admitted direction is that reason from the start. A v1 applicability
registry licenses both directions together.

**SE reversion.** An SE method found anti-conservative on the twins
re-classifies the ABSENTs of that (principle, SE method) as
`SE_NOT_CALIBRATED` (argument `uncalibrated_se_methods`); PRESENT is not
changed.

**Estimator versions.** A /3 protocol names the estimator version of each
principle; evidence from another version (or without one) is
`ESTIMATOR_NOT_VALIDATED:not_admitted`.

**UNDEFINED is never counted as ABSENT.** Every reason maps to UNDEFINED; the
verdict is EXCLUDED only through an ABSENT principle of the necessity set.

**Verdict.** The strong-Kleene AND over the protocol's `necessity_set`
(`N_anch`, the anchored principles): T gives `MPC_CONSISTENT`, F gives
`EXCLUDED`, U gives `UNDETERMINED`; a bearer, protocol or source code
(`BEARER_MISMATCH`, `PROTOCOL_MISMATCH`, `SOURCE_INCOHERENT`, as in v1) forces
`UNDETERMINED`. Every principle with evidence or in the declared set is
assessed; the AND over the declared set `N_decl` is reported beside the
verdict as `verdict_declared` (descriptive). With an external reference the
protocol's anchors replace those of the items, so every status is on the
protocol's frozen scale.

## 4. Error control

Let `theta_p` be the true construct value of principle `p` on the frozen
anchor scale, and assume (A-SE) that `(c_p - theta_p) / se_c` is about
Student t with the declared `df`.

- **E1, component false PRESENT.** If `theta_p <= z`,
  `P(PRESENT) = P(c - q_P se_c > z) <= alpha`. For NAS,
  `theta_NAS = min(theta_R, theta_B) <= z` means some direction has
  `theta <= z`, and PRESENT needs that direction PRESENT. The IIM rank gate
  only removes PRESENT decisions.
- **E2, component false ABSENT.** If `|theta_p| >= delta`,
  `P(ABSENT) <= alpha_A`: the TOST is an intersection-union test of two
  one-sided tests at `alpha_A`. For NAS, `theta_NAS >= delta` means both
  directions are at least `delta`, and `P(either passes) <= 2 alpha_A / 2`.
- **E3, verdict false MPC_CONSISTENT.** MPC_CONSISTENT needs every principle
  of `N_anch` PRESENT, so `P <= alpha` if one of them has `theta_p <= z`.
- **E4, verdict false EXCLUDED.** EXCLUDED needs some ABSENT, so if every
  principle of `N_anch` has `|theta_p| >= delta`,
  `P(EXCLUDED) <= |N_anch| alpha / |N_decl| <= alpha` under any dependence
  between the components and for any anchored subset.

Conditions: the anchor is frozen from development seeds disjoint from the
evaluation seeds; SE validity (A-SE); the consumption contract; UNDEFINED
never counted as ABSENT; a valid null for `nu` (`NULL_MODEL_VIOLATED`
catches gross failures only). The test suite checks E1-E4 by simulation
under (A-SE), including the NAS boundary case (both directions at `delta`)
and the excess of an unsplit union over `alpha_A`.

**What the rule does not control.** Estimator bias (a precise estimate near 0
with the mechanism on is a correct TOST decision about a wrong `theta`);
anchor misattribution (an anchor dominated by a nuisance scales every `c`
wrongly; the specificity gate and the anchor-attributability report address
it); construct misidentification under hidden inputs (under a partial or
empty input declaration PRESENT certifies dependence only); and the
abstention trade-off (UNDETERMINED buys selective error control with
coverage, which is reported per protocol).

## 5. Protocol schema /3

`evidence_v2.ProtocolV3` has every field of the v1 protocol
([metrics.md, section 8.3](metrics.md#83-combining-evidence)), normalised by
the v1 code, plus:

| Field | Content |
|---|---|
| `status_rule` | `{"version": "tost-v2", "alpha", "alpha_absent", "absent_test": "tost", "null_violation", "present_reachability_flag"}`; `alpha` equals the protocol's, and `alpha_absent` is `alpha` divided by the size of `declared_necessity_set` |
| `declared_necessity_set` | `N_decl`, always the five principles (fixed core), so `alpha_A = 0.01` in every protocol; `necessity_set` is `N_anch`, a subset |
| `directions` | per directional principle its members, for example `{"NAS": ["receive", "return"]}` (two or more lower-case names that differ from the channel names) |
| `rank_gates` | per principle the level of a rank gate on PRESENT, for example `{"IIM": 0.05}` |
| `estimator_versions` | per principle the estimator version the protocol dispatches |
| `se_methods` | per principle the admitted SE methods |
| `shared_inputs_declaration` | `{"id", "shared_inputs"[, "inputs"]}`: the declaration (R, H, P, Q10, Q25, J, none, recorded, hidden, label_error) and its level (complete, partial, none; R is complete, H, P, Q10, Q25 and J are partial, none is none) |
| `concordance_route` | the admitted PDI cells `{principle, substrate, observation_stage, view, bearer, absent, present, provisional, battery}` |
| `anchors` | the anchor block (section 7) |
| `precision` | the testability table (section 7) |

The status rule block of every /3 protocol in v2:

```json
{"version": "tost-v2", "alpha": 0.05, "alpha_absent": 0.01, "absent_test": "tost", "null_violation": true, "present_reachability_flag": true}
```

**Reference.** The v1 kinds (`cohort_high_state`, `external`) and `pending`
(no anchors yet: every component is `INVALID_ANCHORS`; templates only).
External keys are `P`, `P:<channel>`, and for directions `P:<direction>` or
`P:<channel>:<direction>`; a directional principle takes only per-direction
keys (a pooled or a channel-only key is refused, since it would never be
read).

**Hash.** The SHA-256 of the canonical JSON of `to_dict()` (sorted keys,
compact separators, every field written out); evidence records it as
`protocol_id = "sha256:<hash>"`. Every field of the schema is hash-covered.

Shipped /3 protocols (`protocols/v2/`):

| File | Role | Hash |
|---|---|---|
| `mpc_bench_v2_template.json` | bench template: the status rule, directions, rank gate, estimator versions, candidate SE methods and estimator options; reference `pending`; the family protocols are generated from it with their anchors, `N_anch`, declaration, admitted SE methods, concordance cells and precision block | `00a6dd9cbdf91219aa96f078d7411028d4c8131777ec34e372236e5cb081f1ad` |
| `mpc_default_v2.json` | paper-2 default: the v1 default's necessity set, channels, cutoffs, cohort reference and source rule under `tost-v2`, with the v2 estimators and the declaration `none` | `c72c2c7d3a1922b448a5ef7ff2288c4b4987a7d7f1d5931e17e348b736eb549e` |

The frozen v1 protocols keep their hashes (`mpc_bench_v1.json`
`855f6a44...`, `mpc_bench_v1_anchored.json` `780581f5...`,
`mpc_default_v1.json` `383eb310...`).

## 6. Reason codes and flags

Every reason is defined once in `impact_pipeline.v2.reasons`. A reason is
`CODE` or `CODE:detail`, and every reason maps to UNDEFINED, never to ABSENT.

| Code | Meaning | Emitted by |
|---|---|---|
| `INVALID_ANCHORS` | the anchor is missing, not finite or not above the null mean | evidence layer |
| `INCONCLUSIVE` | neither PRESENT nor ABSENT at this precision; `INCONCLUSIVE:NULL_NOT_EXCEEDED` when the IIM rank gate failed | status rule; rank gate |
| `NO_SAMPLING_SE` | an empirical estimate without a sampling SE (a data-derived `se = 0` is never exact) | evidence layer |
| `NULL_FAMILY_MISMATCH` | the null family differs from the protocol's; detail `<declared>/<used>` | evidence layer |
| `MISSING_CHANNEL` | a declared channel (or a direction, `<channel>:<direction>`) has no item | evidence layer |
| `NO_NULL_CALIBRATION` | no null mean, fewer than 19 finite null draws, or a missing rank p under the IIM rank gate | evidence layer; IIM |
| `DEGENERATE_NULL` | a malformed null size, or a Monte-Carlo null without a finite SD | evidence layer |
| `INVALID_SE` | a negative SE or reference SE, `se_df <= 0`, or a break of the SE-method contract | evidence layer |
| `NON_FINITE_ESTIMATE` | the estimator returned a non-finite value | evidence layer |
| `NOT_DEFINED` | the estimator declared its value undefined; detail: its own reason | evidence layer |
| `NOT_IMPLEMENTED` | a declared channel that is not implemented; detail: the channel | evidence layer |
| `MISSING` | no evidence for a principle | evidence layer |
| `ABSENT_NOT_REACHABLE` | `q_A se_c >= delta`: the TOST cannot pass at this precision | status rule |
| `NULL_MODEL_VIOLATED` | `c + q_P se_c < -delta`: credibly below the null by more than the margin | status rule |
| `SAMPLING_UNRESOLVED` | `dt > tau_c / 2`: the sampling does not resolve the coupling time scale | NAS v3 |
| `OBSERVATION_MIXED_NOT_ADMITTED` | a sensor or source-estimate observation without an admitting registry entry | NAS v3, IIM v5 |
| `INSUFFICIENT_OCCUPANCY` | a macro state unvisited, or the rarest visited row below `N_min` transition pairs | IIM v5 |
| `MACRO_RANK_DEFICIENT` | the zero-lag correlation of the cluster means is rank deficient | IIM v5 |
| `NOT_APPLICABLE_OBSERVATION_MODEL` | declared inapplicable to the observation model (C1 carrier recording) | SRPI, RAM-PE, PDI |
| `INSUFFICIENT_TIMEPOINTS` | too few time points for the parameters at every allowed block dimension | NAS v3 secondary |
| `INSUFFICIENT_UPDATES` | fewer than 30 updates or fewer than two options | RAM-PE |
| `ESTIMATOR_ERROR` | the estimator raised; detail: the exception type; other components are unaffected | v2 runner |
| `SE_NOT_CALIBRATED` | the ABSENT of an SE method found anti-conservative on the twins | SE reversion |
| `ESTIMATOR_NOT_VALIDATED` | `:not_admitted` (no admitting registry entry for the status direction, or another estimator version than the protocol's) or `:not_observable` | registry v3; protocol |

Flags are markers next to a status, never a status and never a reason:

| Flag | Meaning |
|---|---|
| `PRESENT_NOT_REACHABLE` | UNDEFINED and `q_P se_c >= 1 - z` |
| `ANCHOR_NOT_REPLICATED` | the anchor status did not replicate on the confirmatory replication block; the frozen protocol is kept |
| `HUB_NOT_PRIVILEGED` | the declared NAS hub is not privileged over every pseudo-hub; restricts the construct wording only |
| `ABSENT_BY_CONCORDANCE` | ABSENT through the admitted concordance route |
| `PRESENT_BY_CONCORDANCE` | PRESENT through the admitted concordance route |
| `NONSPECIFIC_ANCHOR` | the anchor failed the specificity gate; the principle leaves `N_anch` and keeps its component-level evaluation |

**Verdict reason codes** are `KIND:<P>[:detail]` with the component reason's
code and detail, for example `ABSENT_NOT_REACHABLE:IIM`,
`INCONCLUSIVE:IIM:NULL_NOT_EXCEEDED`, `ESTIMATOR_NOT_VALIDATED:NAS:not_admitted`,
`MISSING_CHANNEL:NAS:default:return`, plus `ABSENT:<P>` for an absent
principle, `MISSING:<P>` and the global codes of v1. The verdict is
recoverable from its reasons by the v1 decomposition rule
(`evidence.verdict_from_reasons`).

Component records (`mpc-bench-result/3`) take `status`, `reason`, `flags`,
`c` (NAS: `min(c_R, c_B)`), `c_R`, `c_B`, `se_c` and `df_c` from
`PrincipleAssessment.record_fields()`.

## 7. Testability, anchors and necessity sets

These rules read development data only (seeds 0-999) and never the construct
value of a confirmatory target (`impact_pipeline.v2.testability`).

**Testability gating.** For every ABSENT-type hypothesis part, `pi0` is the
empirical ABSENT rate of the target on its witness under the frozen v2 rule
(at least 40 development runs); the analytic
`mean max(0, 2 Phi(delta / se_c - q_A) - 1)` is reported beside it. For
every "PRESENT in >= 80 %" part, `pi1` is the empirical PRESENT rate on the
positive control (analytic `mean Phi((1 - z) / se_c - q_P)` beside it). A
part is decisive only if its rate is at least 0.9; otherwise it is
`NOT_TESTABLE_BY_DESIGN` before the freeze and replaced by "target not
PRESENT in >= 80 % of seeds" (ABSENT parts) or a paired contrast with the
positive control (PRESENT parts), and its predicted `ABSENT_NOT_REACHABLE`
rate is recorded. The rows (family, principle, witness, kind, `n`, `pi`, the
analytic rate, `s_A`, `s_P`, the attainable `se_c`, the rates of every
status and reason, the outcome) form the protocol's hash-covered `precision`
block (schema `impact-mpc-precision/1`).

**Anchors.** On the development reference block (seeds 900-939) a
principle's anchor is valid iff at least 36 of the 40 PC_nominal excesses are
finite and the one-sided 95 % Student-t lower bound of their mean is above 0.
It is specific iff the mean paired contrast PC_nominal minus the principle's
own lesion on the same seeds is at least 0.5 times the anchor and its
one-sided 95 % lower bound is above 0; otherwise the principle is flagged
`NONSPECIFIC_ANCHOR`. NAS is anchored iff both directions are valid and
specific.

**Necessity sets.** `N_anch` is the set of declared principles whose anchor
is valid and specific; it is the protocol's `necessity_set`, and `alpha_A`
does not depend on it. Fallbacks: F0 every declared principle anchored, F1
two or more, F2 one, F3 none (the protocol then keeps the declared set and
the verdict parts are not evaluable). A principle outside `N_anch` keeps its
component-level evaluation. The protocol's hash-covered `anchors` block
(schema `impact-mpc-anchors/1`) holds the rule, the reference seeds, one
entry per declared principle, `N_anch` and the fallback.

## 8. Scale type and the absence margin

Conditional Granger causality (NAS v3), Delta Psi (IIM) and the SRPI v3 LDC
are quadratic in a weak effect amplitude, so `delta = 0.10` on `c`
corresponds to an amplitude of about `sqrt(0.10) = 0.32` of the reference.
RAM-PE (a correlation), the v1 SRPI composite and PDI (bits of a count) are
not quadratic. Every ABSENT of a quadratic statistic is re-judged on the
amplitude scale `a = sign(c) sqrt(|c|)` as a sensitivity analysis
(`scale="amplitude"`). The scale type is read from the estimator version the
protocol names for the principle (`estimator_versions`,
`evidence_v2.SCALE_TYPES`); without one the cutoffs are not changed.
Because `a` is monotone in `c`, judging the interval of `a` against
`(z, delta)` is the same as judging the interval of `c` against
`(z**2, delta**2)` (`evidence_v2.amplitude_cutoff`); the cutoffs of
non-quadratic estimators are unchanged.

A sweep dose counts as "mechanism on" for a principle and protocol (the
rows of the false-exclusion checks and of the rule audit) only if the dose is
at least 0.5 times the nominal dose and the development median `c` at that
dose is at least `2 delta` (`testability.mechanism_on`, with
`testability.development_median_c` over development runs).
