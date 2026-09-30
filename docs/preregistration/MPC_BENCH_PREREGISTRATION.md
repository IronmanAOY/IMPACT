# MPC-Bench preregistration: computational hypotheses of paper 1

Status: frozen locally by the annotated tag `mpcbench-freeze-v1`; **not yet
registered publicly** (see [README](README.md)). Written 2026-09-28, before
any confirmatory run.

## 1. Scope

Paper 1 presents a measurement framework for the minimal preconditions of
consciousness (MPC): five null-anchored component estimators (RAM, PDI, NAS,
IIM, SRPI), a three-valued exclusion rule (`EXCLUDED` / `MPC_CONSISTENT` /
`UNDETERMINED`; necessary conditions license only exclusion, and
`MPC_CONSISTENT` is not an attribution of consciousness), and MPC-Bench, a
white-box benchmark of simulated systems whose mechanisms are switched on and
off. This document preregisters the computational hypotheses HC1-HC10 that
the confirmatory MPC-Bench runs test. The systems' lack of consciousness is
a stipulated premise; the computations test estimator validity, the
irredundancy of the principles on the declared model class and the
behaviour of the rule, not consciousness.

The empirical hypotheses of paper 2 are registered separately
(`predictions/registry.yaml`, still a draft).

## 2. Development and confirmatory data are disjoint

- **Development** (used for everything in this document up to the freeze):
  seeds 0-999 of families A (modular rate-network agent) and B (binary
  networks with exact transition matrices); the null-calibration generator
  with seed base 0. Outputs: `outputs/paper1_mpcbench/dev/` (not
  version-controlled). The development runs were made from commit `1abff4e`
  and from the working tree that became the frozen commit; the changes made
  meanwhile only added recorded fields (jackknife replicates, the IIM grain
  and cut mode) and the plumbing of protocol-declared estimator options, and
  left the estimators and generators unchanged. Every record carries its git
  SHA and dirty flag, and the analysis code that produced every development
  number is in the frozen commit.
- **Confirmatory** (run only after the freeze, from a clean checkout of the
  tag; the runners refuse anything else): seeds >= 10000 for every design,
  family C (Stuart-Landau oscillator agent, held out), the whole-brain Hopf
  model on the empirical connectome, the adversarial constructions, the
  family-C reference block 19000-19019 and the null-calibration seed base
  10000.
- The development and confirmatory seed sets are disjoint (0-999 vs
  >= 10000), and no confirmatory seed, no family-C estimator output, no
  whole-brain and no adversarial estimator output was used to make any
  choice in this document.
- Declared prior contact with held-out generators: family C was redesigned
  on development seeds with oracle-only manipulation checks (no estimator),
  and a few estimator smoke calls were made on the held-out generators with
  development seeds while they were built; no numbers from those calls were
  kept or used. The family-A manipulation checks and all calibration runs
  below are development runs.

## 3. Frozen artefacts

| Artefact | Identity |
|---|---|
| Code | the commit of the local annotated tag `mpcbench-freeze-v1` (branch `polish/hlrs-handoff-2026-09`) |
| Bench protocol, all five principles | `protocols/mpc_bench_v1.json`, SHA-256 (`evidence.Protocol.hash`) `855f6a444b77030d33faa68fcb45e8576b931d2d681cf215d4dacdb57a6b2520` |
| Bench protocol, anchored necessity set | `protocols/mpc_bench_v1_anchored.json`, SHA-256 `780581f57d24251fc8ed8c39565f0a89a96740908f2ada3f65f8961cf1543f4c` |
| Empirical default protocol (paper 2, not tested here) | `protocols/mpc_default_v1.json`, SHA-256 `383eb310cf4479d3260bc5f43f8972bd0b12104a221579fea17c40e410357267` (unchanged at the freeze) |
| Reference summary | `protocols/mpc_bench_v1_reference_summary.json` |
| Evaluation script | `scripts/bench_hypotheses.py` (checks both bench protocol hashes) |
| Run plan | `scripts/mpcbench_confirmatory.sh` |
| Estimator versions | RAM `ram-v2-2026.09`, PDI `pdi-v2-2026.09`, NAS `nas-v2-2026.09`, IIM `iim-v4-2026.09`, SRPI `srpi-v2-2026.09`; bench 2.0.0, generators `mpc-bench-generators/1.1.0` |

## 4. Calibration decisions (made on development data)

Development evidence: `outputs/paper1_mpcbench/dev/` (runs listed in
section 5), analysed by `scripts/calibrate_bench.py` (output
`dev/calibration/`) and, for the expected behaviour of the hypotheses, by
`scripts/bench_hypotheses.py --development` (output
`dev/hypotheses_dryrun/`, flagged "DEVELOPMENT - NOT A RESULT"). Every
number in sections 4 and 5 is a development number: it justifies a frozen
choice or states what the development runs showed, and none may be quoted
as a confirmatory result.

Decision rules fixed before the development runs were analysed: keep the
spec defaults `(z, delta, alpha) = (0.25, 0.10, 0.05)` unless (i) an
anchored principle's false-PRESENT rate on a development null class exceeds
`alpha + 0.02` at `z` (then raise that principle's `z` to the smallest grid
value that meets it), or (ii) the nominal positive control is ever ABSENT
(then lower `delta`); keep the bidirectional IIM cut unless the directional
cut has a valid anchor and a better dose-response to the recurrent loops.

| Setting | Frozen value | Development justification |
|---|---|---|
| Construct-scale cutoffs | `z = 0.25`, `delta = 0.10` for every principle | (i) no anchored principle was PRESENT on a development null system (bench null witnesses: 0 of 10 per principle; null-calibration generator: see section 5); (ii) the positive control was never ABSENT (0 of 30 runs), and of 900 anchored sweep statuses one was ABSENT (IIM at K = 2). `z` is a construct constant (a quarter of the positive control's excess), `delta < z` keeps "absent" (negligible) distinct from "weak". |
| `alpha` | 0.05, one-sided, Student-t bounds | conventional level; the null rates above did not require a smaller one. |
| Null surrogates `K` | 19 per run | the Monte-Carlo error of the null mean is a small part of `var(c)` on the development positive control (median share: NAS 5%, IIM 0.2%, SRPI 0.5%; the reference SE contributes 7-14%). |
| Sampling SE | delete-a-group jackknife, G = 10 (`se_df = 9`) | G = 10 instead of 5: the one-sided t quantile falls from 2.13 to 1.83 and the SE estimate is more stable; on the positive control the jackknife SE is 0.82 (NAS), 0.63 (IIM) and 0.91 (SRPI) times the between-seed SD of the estimate on the reference seeds (0.25, 0.77 and 1.0 on the witness seeds); the between-seed SD also contains the network variability between seeds, so these ratios do not show an underestimated SE. |
| Bootstrap replicates (empirical pipeline) | `--bootstrap-se 100`, `--hunter-iim-bootstrap-se 100` | not tested on the bench (the bench uses the jackknife); t quantile with 99 df within 1% of the normal one, Monte-Carlo relative error of the SE about 7%. |
| Reference anchor | family A `PC_nominal`, development seeds 900-919, mean excess over its null; anchor rule: one-sided 95% t lower bound > 0 | NAS 0.0706 (SE 0.0023), IIM 0.0094 (SE 0.0016), SRPI 0.164 (SE 0.012) anchored; RAM 0.0050 (SE 0.0041, lower bound -0.0021) and PDI 0 (one state on every seed) not anchored. |
| IIM grain | 4 macro nodes (periphery module means; hub excluded), 2 bins, lag 2 samples, `node_shrinkage` TPM | 16 states and 256 transitions for about 11,900 samples per run; the hub is excluded so that IIM does not measure the workspace loop; 3 bins (81 states) would be undersampled. |
| IIM cut mode | bidirectional (bench and default protocols) | the directional cut had a valid anchor (0.0028, SE 0.0006) but no dose-response to the inter-module loops (Spearman rho(c, c_int) = -0.01, p = 0.92; bidirectional: 0.33, p = 0.011) and 1.5-4 times larger `se_c`; the exact-TPM property that motivates it (a feedforward star has directional Delta_Psi = 0) holds and is tested in HC2. |
| Estimator modes | RAM `prediction_error`, PDI `repertoire` (unlabelled, defaults), NAS `capacity`, SRPI `agency` | unchanged: the modes the constructs call for. RAM `feedback_magnitude` is undefined for +-1 feedback; the PDI valley criterion (0.4) was not relaxed although the planted contexts miss it (valley ratios 0.5-0.6 at the true number of states), because relaxing it to 0.6 would admit cuts of a continuum. |
| Necessity sets | `P*`: all five; `P*_anch`: {NAS, IIM, SRPI} | `P*_anch` = the principles with an anchor (a construct scale) on the bench, fixed by the anchor rule, not by the entry criteria below. |
| Default protocol (`mpc_default_v1.json`) | unchanged (hash as in section 3) | same cutoffs; RAM keeps its unimplemented `perturbational` and `endogenous` channels, so behavioural non-response alone can never exclude (a stance confirmed at the freeze: covert responsiveness is not measured). |
| Sizes and seeds | section 7 | 20 seeds per witness and sweep cell (the 80%-of-seeds criteria), 10 per factorial and patchwork cell, 100 replicates per null-calibration cell (pooled n per principle >= 1200). |

Entry criteria of the applicability registry (spec V2-5), applied to the
development runs (`dev/calibration/entry_criteria.csv`): (0) valid anchor;
(a) false-PRESENT rate <= 0.07 on every development null class; (b1)
Spearman rho(c, own dose) > 0 at p < 0.05; (b2) PRESENT rate <= 0.07 on the
systems without the mechanism (own sweep at dose 0, own single-deficit
witness, the all-off agent `O_inert`); (c) the own switch changes `c` more
than any other switch (paired witness contrasts; declared dependencies
excluded). Only **SRPI-agency** meets all of them. NAS-capacity and IIM meet
(0), (a) and (b1) but fail (b2) and (c); RAM and PDI fail (0). These
estimators are therefore not validated for the `synthetic_rate` substrate;
they stay in `P*_anch` so that the confirmatory runs test them (section 5
states what development predicts).

## 5. Development findings (to be reported as development results)

Runs (all family A unless stated; K = 19, G = 10): oracle manipulation
checks, seeds 0-39; positive-control reference, seeds 900-919; sweeps (5
knobs x 10 levels), seeds 0-5; witnesses (13 per seed), seeds 100-109; IIM
with directional cuts (c_int sweep, reference, 7 witnesses), same seeds;
null-calibration generator, seed base 0, 25 replicates per cell; family B
IIM validation, seeds 0-4. The dev factorial and graded-patchwork designs
were started and stopped unfinished to save time; they contributed nothing.

1. **Manipulation checks** pass for every switch in 40 of 40 seeds (eta, K,
   g_b, ff_only, c_int, e): the generator realises its mechanisms.
2. **RAM (prediction-error update) and PDI (unlabelled repertoire) have no
   valid anchor.** On the positive control the RAM adaptive-update term is near
   zero (the RAM value is exactly 0 on 17 of the 20 reference seeds), and
   the unlabelled repertoire counts one state on every seed although the six
   planted contexts are decoded almost perfectly by an oracle-labelled decoder
   (about 2.58 bits of log2 6 = 2.58; development seeds 0 and 1, diagnostic
   only). The planted contexts are decodable but not separated by density
   valleys at the declared criterion. Consequence: on the bench RAM and PDI
   are always UNDEFINED, and with all five principles in `N` no bench system
   can be MPC_CONSISTENT.
3. **SRPI-agency tracks its mechanism.** Over the efference-copy gain e,
   `c` rises from -0.19 (e = 0) to 1.56 (e = 2) (Spearman rho = 0.93);
   PRESENT in 0% of runs for e <= 0.44 and 100% for e >= 1.33; the lesion
   witness has median `c` 0.06 and is never PRESENT. Off-target: the
   repertoire switch raises SRPI (c = 1.50 at K = 1).
4. **NAS-capacity does not detect the removal of the workspace.** At
   g_b = 0 its `c` is 0.93 and it is PRESENT in 100% of runs (the broadcast-
   only witness: `c` 1.31); it responds more to the repertoire switch (K = 1:
   `c` 0.49) and to the inter-module loops (c_int = 0: 0.74) than to its own
   switch, and it is PRESENT in 40% of the hypersynchronous systems. This is
   the common-driver inflation of observational transfer measures (the
   context patterns and the slow rhythm reach every node).
5. **IIM (bidirectional, 4 macro nodes) reads common drive as integration.**
   The feedforward witness has median `c` 0.96 (PRESENT in 50%), the all-off
   agent `c` 1.78 (PRESENT in 100%), the positive control is PRESENT in only
   30% (`se_c` about 0.4-0.5); dose-response to c_int is weak (rho = 0.33).
6. **ABSENT is reached almost only at the null.** Of 900 anchored sweep
   statuses one was ABSENT (IIM at K = 2, mechanism on), and no single-deficit
   witness was ABSENT for any principle; the null systems
   (independent noise, AR(1)) are ABSENT for NAS and IIM in 10 of 10 seeds
   (SRPI never: its `se_c` of about 0.3 keeps the upper bound above 0.10).
   With all five principles the verdict was EXCLUDED for every null system,
   for 4 of 10 all-off agents, 1 of 10 hypersynchronous systems, 1 of 10
   system-bearer patchworks and 1 of 300 sweep runs, never for a
   single-deficit witness, and never MPC_CONSISTENT.
7. **With `N_anch`** (dry run of the evaluation script on the development
   witnesses) the positive control is MPC_CONSISTENT in 3 of 10 seeds and
   the NAS and IIM single-deficit witnesses in 3-4 of 10 (the rule is only
   as specific as its estimators), the SRPI lesion in 0 of 10; the
   disconnected patchwork (per-principle bearers) is UNDETERMINED in 10 of
   10 (BEARER_MISMATCH) and would be MPC_CONSISTENT in 1 of 10 without the
   source rule; 15 witness runs are "clear" (HC10) and their mean
   verdict-flip rate under rescaled jackknife replicates is 0.20.
8. **Null calibration** (generator of `scripts/null_calibration.py`):
   25 replicates per cell (ar1, pink,
   surrogate_iid and surrogate_linear x T in {1200, 2400} samples x 8 or 16
   nodes). NAS and IIM were never PRESENT (0 of 300 each); SRPI was PRESENT
   in 2 of 400 (1 of 25 in each pink cell with T = 2400, the largest cell rate
   0.04 <= 0.07). On these systems NAS was ABSENT in 50-83% and IIM in 6-54%
   of the replicates per null family, SRPI (whose `se_c` is about 1.1 with the
   generator's few self/other pairs) in 0-4%.
9. **Family B** (exact TPMs, 4 units): exact Delta_Psi (bits), bidirectional /
   directional: independent 0 / 0, feedforward star 0.0185 / 0, ring 0.0447 /
   0.0327, all-to-all 0.0934 / 0.0259, XOR loop 0.482 / 0.116, hidden driver
   0.0120 / undefined (its interventional TPM is not conditionally
   independent). The bidirectional ordering independent < ring < all-to-all
   holds; under directional cuts ring > all-to-all at these parameters.
   Sampled estimates (5 seeds; T = 1000, 3000, 10000; 19 surrogates):
   independent units have median calibrated margins between -1.0 and 0.03
   (margin > 1.645 in 0-20% of runs); under directional cuts the feedforward
   star's estimate shrinks towards 0 with T (median 0.0014, 0.0007, 0.0001
   bits) but still exceeds its circular-shift null (margin > 1.645 in 60-80%,
   because that null's SD is tiny); the median absolute error does not
   decrease monotonically with T for all-to-all (both cut modes) and ring; the
   hidden driver reads as strongly integrated (median margins > 60); the ring
   coupling sweep is monotone (Spearman rho 0.98 bidirectional, 0.70
   directional).

Expected confirmatory outcomes, stated so that the confirmatory results can be
compared with them (the development dry run of the evaluation script on family
A, `dev/hypotheses_dryrun/` and `dev/audit_dev/`; development expectation, not
a prediction added to HC1-HC10): HC1 supported; HC2 falsified (parts b and d);
HC3 supported for the SRPI switch and falsified for the NAS and IIM switches;
HC4 falsified (no target ABSENT); HC5 falsified by the NAS and IIM deficit
classes; HC6 not run in development; HC7(a) supported and HC7(b) falsified;
HC8(a) falsified (SRPI never PRESENT but never ABSENT) and HC8(b) supported
for NAS; HC9(a) falsified (the DCM-like rule was never MPC_CONSISTENT on
positives with RAM and SRPI missing), (b) supported, (c) falsified by the NAS
and IIM deficit classes; HC10 indeterminate (15 clear runs, flip rate 0.20).
Family C, the adversarial and whole-brain sets have never been run with
estimators. The hypotheses are kept as formulated before development (novelty
synthesis, adapted to the v2 verdict semantics) rather than revised towards
what the development runs showed; the paper reports every outcome.

## 6. Hypotheses and decision rules

Notation. `P*` is `protocols/mpc_bench_v1.json` (necessity set: all five
principles); `P*_anch` is `protocols/mpc_bench_v1_anchored.json` (necessity
set `N_anch` = {NAS, IIM, SRPI}, everything else identical). Family C uses
`P*` and `P*_anch` with its own reference (the mean excess of its `PC_nominal`
on seeds 19000-19019 under the same anchor rule), everything else unchanged;
adversarial systems use the family-A anchor. Statuses and verdicts are those
of the evidence layer (`evidence.mpc_verdict`, via
`bench.export.evidence_verdict`), with `alpha = 0.05` one-sided and Student-t
bounds (`G - 1 = 9` degrees of freedom of the jackknife SE,
Welch-Satterthwaite combination with the null and reference parts). "Anchored"
principles are those with a reference value in the protocol (NAS, IIM and SRPI
in family A; family C's own anchors follow the same rule); a principle without
an anchor is UNDEFINED (`INVALID_ANCHORS`) everywhere, so every part of a
hypothesis that needs its status is `NOT_EVALUABLE`, and this is reported, not
treated as support. Paired comparisons use witness runs with the same seed
(same network, same noise; only the switched mechanism differs). CP = exact
one-sided Clopper-Pearson bound at level 0.05 unless a family-wise level is
stated. Outcomes: `SUPPORTED`, `FALSIFIED`, `INDETERMINATE` (the rule does not
decide), `NOT_EVALUABLE` (missing input, anchor or manipulation check). A
hypothesis with several parts is `FALSIFIED` if any evaluable part is,
`SUPPORTED` if all evaluable parts are, else `INDETERMINATE`. Every part is
reported whatever its outcome.

**Prerequisite M (manipulation checks).** The oracle manipulation checks of
`bench.manipulation` (version `mpc-bench-manipulation/1.0.0`, thresholds as
frozen) on confirmatory seeds 10000-10039, families A and C: a switch is
usable in a family if it passes in >= 90% of seeds. Parts of HC3, HC4 and
HC8 that involve an unusable switch are `NOT_EVALUABLE`.

**HC1 - null calibration (component level).** Data: `null_calibration.py`
(families ar1, pink, surrogate_iid, surrogate_linear; T in {1200, 2400}
samples; 8 and 16 nodes; 100 replicates per cell; seed base 10000;
surrogate_linear only for RAM, PDI, SRPI) and the bench null witnesses
`N_independent_noise`, `N_ar1` of families A and C (20 seeds each), all
classified under `P*` (the family-C witnesses under family C's protocol). Prediction: every anchored principle's false-PRESENT
rate is at most `alpha + 0.02 = 0.07`. Rule (the registry's H0 rule):
`FALSIFIED` if some (null class, principle) cell has a CP lower bound at level
`0.05/m` (m cells) above 0.07; else `SUPPORTED` if every anchored principle's
pooled CP upper bound at level `0.05/P` is below 0.07; else `INDETERMINATE`.

**HC2 - IIM ground truth (family B).** Data: `iim_validation.py`, all six
network kinds with 4 units, T in {1000, 3000, 10000, 30000}, seeds
10000-10019, both cut modes, 19 circular-shift surrogates; ring coupling
sweep {0, 0.15, 0.3, 0.45, 0.6, 0.9} at T = 30000. (a) Deterministic,
verified before the freeze and reported: exact Delta_Psi of independent units
is 0 in both cut modes; the feedforward star is 0 with directional and > 0
with bidirectional cuts; with bidirectional cuts independent < ring <
all-to-all. Confirmatory parts: (b) for every system except the hidden
driver and each cut mode, the median |sampled - exact| Delta_Psi decreases
strictly from each T to the next; (c) independent units: at every T and cut
mode, the fraction of runs with calibrated margin `(Delta_Psi -
null mean) / null SD > 1.645` is <= 0.07 and |median margin| < 0.5; (d)
feedforward star, T >= 3000: directional margin > 1.645 in <= 7% of runs,
bidirectional in >= 80%; (e) ring sweep: Spearman correlation between
coupling and calibrated excess > 0 with p < 0.05 in each cut mode. The hidden
driver is reported only (the observed process is not generated by the
interventional TPM).

**HC3 - selectivity (families A and C).** For each switch s with target
principle j(s) (eta -> RAM, K -> PDI, g_b -> NAS, c_int -> IIM, e -> SRPI),
paired by seed: `Delta c_p = c_p(PC_nominal) - c_p(single-deficit witness of
s)` (20 seeds per family). Evaluable for anchored targets. Prediction: (i)
`Delta c_target >= 0.5` in >= 80% of seeds; (ii) for every other anchored
principle, `|Delta c_p| < z_p` in >= 80% of seeds, except the declared
structural dependencies (c_int -> NAS: removing every backward loop also
removes part of the workspace receive-and-return; g_b -> IIM: removing the
workspace removes the loops through the hub), which are reported; (iii)
dose-response (sweeps, 10 levels, 20 seeds): the OLS slope of `c` on the dose
rescaled to [0, 1] is positive for the target and every non-dependent
off-target slope is below 0.2 times it in absolute value. A switch is
`SUPPORTED` if (i)-(iii) hold, `FALSIFIED` if one fails; HC3 combines the
switches of both families.

**HC4 - irredundancy (witnesses, families A and C).** For each
single-deficit witness whose target is in `N_anch`: target ABSENT and every
other principle of `N_anch` PRESENT in >= 80% of seeds (20 per family). A
principle is irredundant on the declared model class if its witness passes
in both families. The number of such principles is reported, whatever it
is; HC4 is `SUPPORTED` if every evaluable witness passes.

**HC5 - specificity bound.** For each single-deficit witness class (family,
witness) whose target is in `N_anch`: `P(MPC_CONSISTENT)` under `P*_anch`.
`FALSIFIED` if some class's CP lower bound at level `0.05/m` exceeds 0.07;
`SUPPORTED` if the pooled CP upper bound is below 0.07; else
`INDETERMINATE`. (A deficit class can be `MPC_CONSISTENT` only through a
false PRESENT of its target, so the component-level bound carries over.)

**HC6 - no presence without the mechanism (factorial).** In the 2^5
factorial (families A and C, 10 seeds per cell), for each anchored principle,
the rate of PRESENT in the cells where its mechanism is off. Rule as HC1
(cells = (family, cell, principle)). Cells whose point rate exceeds 0.07 are
listed as revisions of the implication structure; cells where a switched-on
mechanism is ABSENT in > 20% of seeds are reported. No factorial cell is
structurally unrealisable in the generators (all 32 are realised by
construction), so the original "structural zero" claim (e.g. SRPI-agency
PRESENT with RAM absent is unrealisable in organisms) is not testable on the
bench and is not claimed.

**HC7 - single-source constraint (patchwork).** The disconnected patchwork
(`PW_patchwork`, per-principle bearers; 20 seeds per family): (a) under
`P*_anch` (source rule `single_source`) no run is `MPC_CONSISTENT` (0 of n;
expected UNDETERMINED with `BEARER_MISMATCH`, or EXCLUDED); (b) the same
evidence judged without the source rule (`source_rule = none`) is
`MPC_CONSISTENT` in >= 80% of runs. The joint-dependence test of the
integrated positive control is not computed by the bench and is not part of
HC7. The graded patchwork sweep is reported descriptively.

**HC8 - SRPI agency.** `W_SRPI_no_efference` (families A and C): (a) SRPI
ABSENT in >= 80% of seeds (the rate of "not PRESENT" is reported alongside);
(b) for anchored RAM and NAS, `|c(lesion) - c(PC_nominal)| < z` in >= 80% of
paired seeds. The comparison with the legacy SRPI mode is not run by the
bench and is not part of the confirmatory test.

**HC9 - rule audit.** `benchmark_attribution_rules.py` on all confirmatory
bench records (families A and C, adversarial, whole-brain) with necessity set
`N_anch` (primary) and all five (secondary), cross-fitted anchors by seed
parity, label noise 0: (a) scenario "RAM+SRPI missing": the DCM-like
skip-missing rule returns MPC_CONSISTENT for positive controls
(`all_present`, `PC_nominal`) at a rate > 0.5, the IMPaCT rule (`impact_c`)
never; (b) scenario "none": union, count_1-count_4, arithmetic mean and
uncapped geometric mean each return MPC_CONSISTENT for some single-deficit
class (target in `N_anch`) at a rate > 0.05; (c) `impact_c` returns
MPC_CONSISTENT for every single-deficit class at a rate <= 0.07; (d) the
sensitivity cost (`impact_c` MPC_CONSISTENT rate on positive controls) is
reported.

**HC10 - verdict stability.** Witness and factorial runs of families A and C
under `P*_anch`. A run is clear if, for every principle of `N_anch`, `c` is
more than 2 `se_c` from both cutoffs. Each run's G = 10 delete-a-group
jackknife replicates are rescaled to bootstrap-like replicates
`m*_g = m + sqrt(G - 1) (m_(g) - mean_g m_(g))` and re-judged (same null,
SE and reference). Prediction: the mean verdict-flip rate among clear runs
is < 0.10. `INDETERMINATE` with fewer than 30 clear runs.

## 7. Confirmatory run plan

All steps: `scripts/mpcbench_confirmatory.sh` (resumable; `OUT` = an
output root; each step re-checks the freeze tag and a clean tree). The bench
is CPU-bound: the plan is sized for a local multi-core machine (about 4000
bench systems at about 2-3 CPU-minutes each, plus the null calibration and
family B: roughly 15-25 hours with 14 worker processes; the whole-brain
systems, 209 regions, have not been timed with the estimators); HLRS Hunter is used
only if HLRS accepts CPU-only work there (runbook, section 15). Sizes:

| Step | Design | Seeds | Size |
|---|---|---|---|
| manipulation | oracle checks, families A and C | 10000-10039 | 40 seeds x 6 switches x 2 families |
| bench_A / bench_C | witnesses (13 per family, patchwork with both bearer modes) | 10000-10019 | 280 per family |
| | sweeps (5 knobs x 10 levels) | 10000-10019 | 1000 per family |
| | factorial (32 cells) | 10000-10009 | 320 per family |
| | graded patchwork (6 levels x 2 bearer modes) | 10000-10009 | 120 per family |
| reference_C | `PC_nominal`, family C | 19000-19019 | 20 |
| adversarial | 6 constructions | 10000-10019 | 120 |
| whole_brain | G sweep (8) + 3 lesions + 3 matched controls x source/EEG-like/BOLD-like | 10000-10009 | 420 (descriptive) |
| null_calibration | 4 null families x 2 T x 2 node counts | base 10000 | 100 per cell |
| iim_validation | family B, 6 kinds x 4 T x 2 cut modes + ring sweep | 10000-10019 | 1200 |

Every bench run: K = 19 null surrogates, G = 10 jackknife groups, the frozen
protocol (whose declared estimator options, e.g. the IIM cut mode, the
runner applies). Then the audit and `scripts/bench_hypotheses.py`.

Handling of failures: a task that ends with `status = error` is re-run once
with the same seed (resume); a task that still fails is reported and
excluded from the denominators of the hypotheses it would enter. No seed is
added, replaced or dropped for any other reason, and there is no optional
stopping: all planned runs are made, then evaluated once.

## 8. Analysis

`python scripts/bench_hypotheses.py --freeze-tag mpcbench-freeze-v1 ...`
(the last step of the run plan) writes `hypotheses.json` and
`hypotheses_table.csv` with every part of HC1-HC10 and the prerequisite M.
The script refuses records outside the confirmatory split or without the
freeze tag, protocol files whose hashes differ from the frozen ones and
records whose IIM cut mode differs from the protocol's. Figures
(`scripts/figures/`) are descriptive and do not change any decision.

## 9. Deviations

Any deviation from this plan (a bug fix, a changed step, a failed run) is
reported with its reason in the paper, and results obtained after a code
change to `src/` or `scripts/` are exploratory: the confirmatory guard
refuses to run such code under the freeze tag.

## Errata (documentation only; no change to hypotheses or decision rules)

Added 2026-09-29, after the freeze tag `mpcbench-freeze-v1`, in a
documentation-only commit. The frozen text above is left as it was frozen.
These errata correct two descriptive numbers in section 7. They change no
hypothesis, decision rule, threshold, protocol, seed or task: the
confirmatory runs execute the tasks that the frozen code enumerates, and
those already have the corrected sizes. Both values were checked against the
frozen code (freeze commit `f2cf249`, which was `b908ee3` before the commit
messages were edited on 2026-09-29; both have the same tree).

1. **Whole-brain grain (section 7, run-plan note).** The note says "the
   whole-brain systems, 209 regions". The whole-brain systems have **76
   regions and 265 region-level edges** (grain `region`, the default used by
   the run plan: the fine edge list is aggregated to its 76 parent regions).
   209 is not a region count: it is the minimum edge confidence of the
   connectome export in the file name
   (`data/managed/structural/budapest_connectome_3.0_209_0_median.csv`; every
   edge has confidence >= 209). Checked with
   `impact_pipeline.bench.whole_brain.load_connectome()` (its `provenance`:
   `n_nodes` = 76, `n_edges` = 265, `edge_confidence_min` = 209).
2. **Witness tasks per family (section 7, table row `bench_A / bench_C`,
   witnesses).** The size column says "280 per family". The correct number is
   **260 per family**: 13 witness tasks per seed (the 12 catalogue witnesses,
   with the patchwork witness scored in both bearer modes) x 20 seeds
   (10000-10019). Checked with `impact_pipeline.bench.run_bench.witness_tasks`
   for families A and C (260 tasks each). The design column ("13 per family")
   was already correct.

Note on bracketed codes (added 2026-09-30): codes of the form `V2-x` in
the frozen text (section 4: "spec V2-5") refer to design notes of the
1.1.0 development and are not needed to read this document.
