# MPC-Bench v2: operating characteristics, seed sizes and cost

DEVELOPMENT - NOT A RESULT. Companion of
[`MPC_BENCH_PREREGISTRATION_V2.md`](../MPC_BENCH_PREREGISTRATION_V2.md),
sections 8 and 11. Every probability here is computed from development seeds
(0-999) only. No held-out condition was scored for it, no confirmatory seed was
run, and no threshold, cutoff (`z = 0.25`, `delta = 0.10`, `alpha = 0.05`),
decision rule, prediction or hypothesis id was changed. Only seeds were resized
(CD-11).

## 1. Sources

1. **The operating-characteristics run** (2026-10-08, code `21597fe`, clean
   tree): the decided protocol build before the held-out release, the
   hypotheses file at that commit and the development records of the planned
   item `dry_run_evaluation`, run into a separate root holding a copy of the
   development items it reads (2412 records re-judged, 0 failed). An in-process
   replica of the evaluator reproduces every part outcome of the official
   development evaluation. Output: `outputs/mpcbench_v2/decisions/oc/OC_TABLE.md`
   with its tools and raw outputs (not versioned).
2. **An independent check** of that run from the raw development records
   without the engine, with its own exact binomial sums, Clopper-Pearson bounds
   and Monte Carlo (`outputs/mpcbench_v2/decisions/oc/verify/`, not
   versioned). It corrected the values listed in section 3 and proposed the
   resizes that were made.
3. **The resizes** of HCv2-23 and HCv2-14(f) as made (commit `52b7e1c`) and the
   forward anchor replication power computed from the held-out reference
   anchors after the release (`protocols/v2/generated/calibration_evidence.json`,
   `replication_power`).
4. **The confirmatory plan** listed by `scripts/run_bench_v2.py plan --split
   confirmatory` and `scripts/v2/iim_validation_v2.py --list --split
   confirmatory` at the frozen code (lists only; nothing was run), priced with
   the development cost model (section 5).

## 2. Method

For every Tier-A decisive part, the engine's own rule function is applied to
the part's prepared development rows, resampled by cluster to the confirmatory
number of clusters in each stratum, with B = 1000 replicates (300 for the twin
kappa rule, 150 for the false-exclusion rule). A stratum is one (cell, design
and system). Confirmatory clusters per stratum: single runs, the confirmatory
tasks of that design and system; pairs and anchor seeds, the seeds both
systems share; seed clusters, the seeds any system of the family has; twin
networks, the networks. Where the family-A reference blocks (900-939) run the
same witness under the same protocol, the dry run is pooled with them (52 or 64
runs instead of 12 or 24), as the testability table does. The C1 reference
blocks are pooled nowhere, except that HCv2-6(a) uses the C1 IIM anchor
excesses, which HO-4 permits.

Columns: **P(SUP | dev)** is the cluster bootstrap at the confirmatory size
(the CD-11 criterion, target >= 0.8). **binom / Jeffreys**, for rate rules,
gives the binomial plug-in at each cell's development rate and a Jeffreys
predictive (one Beta(k + 1/2, n - k + 1/2) per cell or pool), a sensitivity
that is very pessimistic where a cell has 4-5 development runs. **P(FAL |
correct)** redraws every event at the boundary of the claim (three-zone x;
count rule `max_rate`; lower-bound rule x; demonstration 0; H0 cell rule 0.05;
any-event and false-exclusion rules 0.05 per seed cluster; kappa rules: the
development values rescaled within each network to kappa = 1); for two-outcome
value rules (Spearman, Hodges-Lehmann, median, Jonckheere-Terpstra, Newcombe,
anchor replication) the development effect is taken as the correct one, so
`P(FAL | correct) = P(FAL | dev)`. Parts predicted FALSIFIED are judged by the
probability of their predicted outcome and never resized. A resize searches a
uniform seed extension of the part's confirmatory systems and confirms the
answer with B = 3000-5000 on the sizes around it, because Clopper-Pearson rules
are saw-toothed in n.

## 3. Every Tier-A decisive part

Values of the operating-characteristics run, with the corrections of the
independent check applied and marked **(checked)**, and the values after the
resizes marked **(resized)**. "n dev -> n conf" counts development clusters and
the confirmatory clusters they stand for.

| Part | Rule | Label | Prediction | Dev source | n dev -> n conf | P(SUP \| dev) | binom / Jeffreys | P(FAL \| dev) | P(FAL \| correct) | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| HCv2-0(a) | usable_share |  | SUPPORTED | dry run | 492 -> 492 | **0.964** | 0.965 / 0.485 | 0.036 | 0.999 | binary usable-share rule; P(FAL) at the 0.9 boundary is ~0.5 per cell by construction (not a sizing criterion) |
| HCv2-0(b) | usable_share |  | SUPPORTED | dry run | 574 -> 574 | **0.16** (checked: 40 seeds) | 0.139 / 0.100 (n = 41 run) | about 0.84 (checked: 40 seeds; 0.849 in the n = 41 run) | 0.999 | ADV_NAS_staggered_tau10 driver check 34/40 < 36/40 (checked: prerequisite M runs 40 seeds; 0.056 at 80). No resize helps: predicted not usable |
| HCv2-1 | h0_cell |  | SUPPORTED | dry run | 224 -> 1012 | **1.000** | 1.000 / 0.636 | 0.000 | 0.002 | after the 46-seed resize (RAM pool 92 clusters, tolerates 1 event); 0 development events in every pool. Pool-level Jeffreys predictive 0.636 (RAM pool needs ~300 clusters for 0.8). See section 4 |
| HCv2-2 | h0_cell | C | SUPPORTED | family-B summary | 270 -> 3600 (18 cells x 200) | **0.000** | 0.000 / - | 1.000 | 0.000 | the rank exceedance of the development records is above the bound in two bidirectional cells (T1000 3/15, T10000 4/15; pooled 18/270 = 0.067): plug-in P(SUP) 0.000, P(FAL) 1.000; at a correct 0.05: 0.996. No resize helps. The run found the part NOT_EVALUABLE because the field map read p_ind where family-B records do not carry it; fixed before the release (a16f84c) |
| HCv2-3 | kappa_null | C | FALSIFIED | dry run | 152 -> 892 | **0.000** | / | 1.000 | 1.000 | predicted FALSIFIED: P(predicted) = 1.000 |
| HCv2-4(a,b) | kappa_twins | C | SUPPORTED | dry run | 515 -> 1030 | **0.000** | / | 1.000 | 0.230 | SE calibration failures (CD-2/CD-3); no resize helps. With SEs rescaled to kappa = 1: P(SUP) 0.000, P(FAL) 0.230 (103 cells) |
| HCv2-4(c) | concordance_twins | C | SUPPORTED | dry run | 25 -> 50 | **1.000** | / | 0.000 | 0.000 |  |
| HCv2-5(a) | false_exclusion | C | SUPPORTED | dry run | 44 -> 90 | **0.18** (checked) | / | 0.000 | 0.000 | the only development concordant row is PC_half seed 328 (1 of 12 PC_half runs); PC_half runs on 20 confirmatory seeds: (11/12)^20 = 0.18 (checked). INDETERMINATE whenever such a row appears; no resize helps |
| HCv2-5(b) | false_exclusion | C | SUPPORTED | dry run | 44 -> 90 | **1.000** | / | 0.000 | 0.007 |  |
| HCv2-6(a) | anchor_replicates | R | SUPPORTED | dry run | 840 -> 620 | **0.625** | / | 0.375 | 0.375 | A-H IIM finite 36/40 on the 0.9 edge: 0.58-0.63 (MC) at 40, 0.60 at 80, 0.58 at 160, 0.55 at 400. No resize helps (already labelled). INDETERMINATE-capable in the design's sense only: the rule has no INDETERMINATE zone, so the alternative outcome is FALSIFIED |
| HCv2-6(b) | anchor_replicates | R | SUPPORTED | held-out reference anchors | 40 -> 40 (Hopf), 20 -> 20 (family A) | about 0.38 | - | about 0.62 | about 0.62 | computed after the release from the held-out reference anchors (section 4.1); product of the per-view powers, views treated as independent (they share seeds). Hypothesis HCv2-6 about 0.6 x 0.38 = 0.23 |
| HCv2-7(i) | three_zone | R | SUPPORTED | dry run + ref. blocks | 64 -> 90 | **1.000** | 1.000 / 0.979 | 0.000 | 0.049 |  |
| HCv2-7(ii) | three_zone | R | SUPPORTED | dry run + ref. blocks | 64 -> 90 | **1.000** | 1.000 / 0.979 | 0.000 | 0.049 |  |
| HCv2-7(iii) | three_zone | C | SUPPORTED | dry run + ref. blocks | 64 -> 90 | **0.000** | 0.000 / 0.000 | 1.000 | 0.049 | C1 relay 0/12 NAS PRESENT: no resize helps |
| HCv2-7(iv) | spearman_ols | R | SUPPORTED | dry run | 80 -> 200 | **1.000** | / | 0.000 | 0.000 |  |
| HCv2-7(v-A) | reversed_three_zone | R | SUPPORTED | dry run + ref. blocks | 52 -> 45 | **0.002** | 0.002 / 0.025 | 0.859 | 0.000 | A-R NAS on W_PDI_single_attractor 44/52 = 0.846 >= 0.8: predicted failure not reached; no resize helps |
| HCv2-7(v-C1) | three_zone | R | SUPPORTED | dry run | 12 -> 20 | **0.986** | 0.980 / 0.805 | 0.000 | 0.030 |  |
| HCv2-8(a-H) | three_zone | R | SUPPORTED | dry run | 12 -> 20 | **0.414** | 0.418 / 0.433 | 0.094 | 0.032 | 9/12 = 0.75 < 0.8: 0.31 at 40, 0.20 at 80, 0.05 at 200 seeds. No resize helps |
| HCv2-8(a-R) | count_at_most | R | SUPPORTED | dry run | 12 -> 20 | **1.000** | 1.000 / 0.903 | 0.000 | 0.043 |  |
| HCv2-8(b-H) | three_zone | R | SUPPORTED | dry run | 36 -> 60 | **1.000** | 1.000 / 0.925 | 0.000 | 0.030 |  |
| HCv2-8(b-H-rank1) | three_zone | R | SUPPORTED | dry run | 36 -> 60 | **0.093** | 0.086 / 0.100 | 0.189 | 0.030 | reversed variant 8/12 = 0.67 < 0.8: falls with n. No resize helps |
| HCv2-8(b-R) | count_at_most | R | SUPPORTED | dry run | 36 -> 60 | **1.000** | 1.000 / 0.740 | 0.000 | 0.125 | P(FAL \| correct at 0.10) = 0.125 > 0.05: three cells at alpha each (m = 1); not fixable by seeds |
| HCv2-8(c-H-secondary) | three_zone | R | SUPPORTED | dry run + ref. blocks | 52 -> 20 | **1.000** | 1.000 / 0.999 | 0.000 | 0.032 |  |
| HCv2-8(c-R-secondary) | count_at_most | R | SUPPORTED | dry run + ref. blocks | 52 -> 20 | **1.000** | 1.000 / 0.994 | 0.000 | 0.043 |  |
| HCv2-8(c-H-primary) | three_zone | C | SUPPORTED | dry run + ref. blocks | 52 -> 20 | **0.904** | 0.877 / 0.826 | 0.002 | 0.032 |  |
| HCv2-8(d) | three_zone | R | SUPPORTED | dry run | 24 -> 40 | **1.000** | 1.000 / 0.949 | 0.000 | 0.020 |  |
| HCv2-9(a) | jonckheere_terpstra | HO | SUPPORTED | - | - | - | - | - | - | HO-1/HO-2: no development output on the misdeclared declarations (cells NOT_EVALUABLE) |
| HCv2-9(b) | wilcoxon | HO | SUPPORTED | - | - | - | - | - | - | HO-1/HO-2: held out |
| HCv2-9(c) | sign_test | HO | SUPPORTED | - | - | - | - | - | - | HO-1/HO-2: held out |
| HCv2-9(d) | three_zone | HO | SUPPORTED | - | - | - | - | - | - | HO-1/HO-2: held out |
| HCv2-9(e) | sign_test | HO | SUPPORTED | - | - | - | - | - | - | HO-1/HO-2: held out |
| HCv2-10 | admission_matches | R | SUPPORTED | - | - | - | - | - | - | HO-3: admission at the held-out regime (predictions committed) |
| HCv2-11(a) | h0_cell | C | SUPPORTED | dry run | 60 -> 480 | **1.000** | 1.000 / 0.986 | 0.000 | 0.001 |  |
| HCv2-11(b-rho) | spearman_ols | C | SUPPORTED | dry run | 60 -> 480 | **1.000** | / | 0.000 | 0.000 |  |
| HCv2-11(b-present) | three_zone | C | SUPPORTED | dry run | 5 -> 40 | **1.000** | 1.000 / 0.879 | 0.000 | 0.043 |  |
| HCv2-11(c) | three_zone | C | SUPPORTED | dry run | 50 -> 400 | **1.000** | 1.000 / 0.277 | 0.000 | 0.029 |  |
| HCv2-12(a) | spearman_monotone | C | SUPPORTED | dry run | 60 -> 480 | **1.000** | / | 0.000 | 0.000 |  |
| HCv2-12(b) | stepwise_sign | HO | SUPPORTED | synthetic | - -> 40 seeds x 3 levels | **0.995** | - | 0.005 | 0.005 | SPECIAL (synthetic, no estimator at ring 0.9/1.5): P(SUP) 0.995 at 40 seeds and the development SD; x1.5 SD 0.85, x2 0.61 (needs ~70 seeds), x3 0.30 (needs ~160). See section 2 |
| HCv2-12(c-low) | three_zone | C | SUPPORTED | dry run | 30 -> 240 | **1.000** | 1.000 / 0.025 | 0.000 | 0.018 |  |
| HCv2-12(c-high) | three_zone | C | SUPPORTED | dry run | 20 -> 160 | **1.000** | 1.000 / 0.085 | 0.000 | 0.041 |  |
| HCv2-12(d) | three_zone | C | SUPPORTED | dry run | 37 -> 121 | **0.101** | 0.090 / 0.129 | 0.391 | 0.044 | O_hypersynchronous MACRO_RANK_DEFICIENT 9/12 = 0.75 < 0.9: 0.015 at 40 seeds. No resize helps |
| HCv2-13(a) | h0_cell | C | SUPPORTED | dry run | 15 -> 300 | **1.000** | 1.000 / 0.928 | 0.000 | 0.001 | the run saw only the PRESENT cells (field-map defect, fixed before the release in a16f84c); development rank exceedance on the recorded cells 0/5 each |
| HCv2-13(b-present) | three_zone | C | SUPPORTED | dry run | 5 -> 100 | **1.000** | 1.000 / 0.771 | 0.000 | 0.040 |  |
| HCv2-13(b-median) | median_threshold | C | SUPPORTED | dry run | 15 -> 300 | **1.000** | / | 0.000 | 0.000 |  |
| HCv2-13(c) | jonckheere_terpstra | C | SUPPORTED | dry run | 5 -> 100 | **1.000** | / | 0.000 | 0.000 |  |
| HCv2-14(a) | hodges_lehmann | R | SUPPORTED | dry run + ref. blocks | 52 -> 45 | **1.000** | / | 0.000 | 0.000 |  |
| HCv2-14(b) | h0_cell | R | SUPPORTED | dry run + ref. blocks | 64 -> 65 | **1.000** | 1.000 / 0.710 | 0.000 | 0.004 | 0 development events |
| HCv2-14(c) | three_zone | R | NOT_TESTABLE_BY_DESIGN | - | - | - | - | - | - | NOT_TESTABLE_BY_DESIGN (pi < 0.9 at the gate) |
| HCv2-14(d) | three_zone | R | SUPPORTED | dry run + ref. blocks | 64 -> 45 | **0.992** | 0.993 / 0.955 | 0.000 | 0.025 | A-R PC_nominal IIM PRESENT 58/64 = 0.906 at 45 seeds; dry run alone 20/24 (checked), 0.793. The C1-R cell (HO-4) is not covered |
| HCv2-14(e) | median_threshold | C | SUPPORTED | dry run + ref. blocks | 52 -> 45 | **1.000** | / | 0.000 | 0.000 |  |
| HCv2-14(f) | spearman_ols | R | SUPPORTED | dry run | 40 -> 650 per family | **0.818** (resized, checked) | / | about 0.18 (resized, checked; 0.973 at 10 seeds) | about 0.18 (resized, checked) | resized to 65 seeds per c_int level in A and C1 (seed profiles resampled; 0.16 at 10, 0.64 at 45); P(FAL \| development effect) 0.18; the C1 cell is held out, so the part's value is at most this |
| HCv2-15(a) | demonstration | R | SUPPORTED | dry run | 20 -> 61 | **0.000** | 0.000 / 0.000 | 1.000 | 0.000 | development regime (HO-3 proxy): exceedance 2/20 on eeg64, eeglow, mne_template > 0.07. No resize helps |
| HCv2-15(b) | three_zone | R | SUPPORTED | dry run | 20 -> 61 | **0.000** | 0.000 / 0.000 | 1.000 | 0.050 | development regime: v1 quadrant PRESENT 0/20 at G = 0. No resize helps |
| HCv2-15(c) | demonstration | C | SUPPORTED | - | - | - | - | - | - | no development rows (BOLD FMabs at the held-out regime) |
| HCv2-15(d-present) | h0_cell | C | SUPPORTED | dry run | 88 -> 331 | **1.000** | 1.000 / 0.994 | 0.000 | 0.002 |  |
| HCv2-15(d-absent) | demonstration | C | SUPPORTED | dry run | 32 -> 150 | **1.000** | 1.000 / 0.422 | 0.000 | 0.000 |  |
| HCv2-15(e-FMd) | fmd_concordance | HO | SUPPORTED | - | - | - | - | - | - | HO-6: G sweep of the sensor views held out |
| HCv2-16 | h0_cell | C | SUPPORTED | dry run | 124 -> 280 | **1.000** | 1.000 / 0.998 | 0.000 | 0.002 |  |
| HCv2-17(a) | spearman_ols | R | SUPPORTED | dry run | 100 -> 200 | **1.000** | / | 0.000 | 0.000 |  |
| HCv2-17(b) | three_zone | R | SUPPORTED | dry run | 20 -> 40 | **0.999** | 1.000 / 0.958 | 0.000 | 0.043 |  |
| HCv2-17(c) | rate_lower_bound | C | SUPPORTED | dry run | 20 -> 40 | **1.000** | 1.000 / 0.998 | 0.000 | 0.043 |  |
| HCv2-17(d) | hodges_lehmann | R | SUPPORTED | dry run | 20 -> 40 | **0.000** | / | 1.000 | 1.000 | median Delta c <= 0.25 (development): no resize helps |
| HCv2-17(e) | newcombe_includes_zero | C | SUPPORTED | dry run | 40 -> 80 | **0.955** | / | 0.045 | 0.045 | two-outcome rule |
| HCv2-18 | h0_cell | R | SUPPORTED | dry run + ref. blocks | 292 -> 1060 | **1.000** | 1.000 / 1.000 | 0.000 | 0.002 |  |
| HCv2-19(a) | h0_cell | R | SUPPORTED | dry run + ref. blocks | 192 -> 365 | **1.000** | 1.000 / 1.000 | 0.000 | 0.002 |  |
| HCv2-19(b) | three_zone | R | SUPPORTED | dry run + ref. blocks | 84 -> 95 | **1.000** | 1.000 / 0.468 | 0.000 | 0.037 |  |
| HCv2-19(c) | spearman_ols | R | SUPPORTED | dry run | 24 -> 60 | **1.000** | / | 0.000 | 0.000 |  |
| HCv2-19(d) | three_zone | C | SUPPORTED | dry run | 8 -> 20 | **1.000** | 1.000 / 0.739 | 0.000 | 0.013 |  |
| HCv2-19(e) | three_zone | R | SUPPORTED | dry run + ref. blocks | 52 -> 45 | **1.000** | 1.000 / 1.000 | 0.000 | 0.025 |  |
| HCv2-20(a) | three_zone | R | SUPPORTED | dry run + ref. blocks | 52 -> 45 | **0.524** | 0.509 / 0.499 | 0.030 | 0.025 | ABSENT (concordant) 41/52 = 0.79 < 0.80: 0.43 at 90, 0.40 at 180, 0.30 at 400 seeds; P(FAL) 0.04 -> 0.15. No resize helps |
| HCv2-20(b) | three_zone | R | SUPPORTED | dry run | 8 -> 30 | **0.940** | 0.928 / 0.707 | 0.000 | 0.025 | dry run only (the reference runs carry no oracle AMI field); 7/8 seeds with K_hat_full = 2 |
| HCv2-20(c) | three_zone | R | SUPPORTED | dry run + ref. blocks | 52 -> 45 | **1.000** | 1.000 / 0.964 | 0.000 | 0.023 |  |
| HCv2-20(d) | three_zone | HO | SUPPORTED | - | - | - | - | - | - | HO-7 (mis-declared access node): held out |
| HCv2-20(e) | h0_cell | C | SUPPORTED | dry run + ref. blocks | 192 -> 365 | **1.000** | 1.000 / 0.587 | 0.000 | 0.005 |  |
| HCv2-20(f-single) | three_zone | R | SUPPORTED | dry run + ref. blocks | 52 -> 45 | **0.524** | 0.509 / 0.499 | 0.030 | 0.025 | same rows and rate as HCv2-20(a): 41/52 = 0.79. No resize helps |
| HCv2-20(f-no-multistability) | three_zone | R | SUPPORTED | dry run | 12 -> 20 | **0.975** | 0.964 / 0.836 | 0.000 | 0.048 |  |
| HCv2-20(f-PC) | three_zone_at_most | R | SUPPORTED | dry run + ref. blocks | 64 -> 45 | **1.000** | 1.000 / 1.000 | 0.000 | 0.025 |  |
| HCv2-21 | admission_matches | R | SUPPORTED | - | - | - | - | - | - | HO-3: admission at the held-out regime |
| HCv2-22(i) | three_zone | R | SUPPORTED | dry run + ref. blocks | 336 -> 360 | **0.924** | 0.928 / 0.762 | 0.000 | 0.025 |  |
| HCv2-22(ii) | three_zone | R | FALSIFIED | dry run + ref. blocks | 64 -> 90 | **0.792** | 0.792 / 0.524 | 0.002 | 0.033 | predicted FALSIFIED: P(predicted) = 0.002 (checked); the PDI cell is NOT_TESTABLE_BY_DESIGN (pi0 0.79), so the testable cells decide; predictions fixed, not resized |
| HCv2-22(iii) | three_zone | R | FALSIFIED | dry run + ref. blocks | 468 -> 405 | **0.000** | 0.000 / 0.000 | 1.000 | 0.021 | predicted FALSIFIED: P(predicted) = 1.000 |
| HCv2-23 | any_event_clusters | C | SUPPORTED | dry run + ref. blocks | 64 -> 129 seed clusters | **0.884** (resized, checked) | / | 0.001 | 0.000 | C1 single deficits resized to 129 seeds; with the seed-keyed any-event clusters the plan before the resize gave 0.069 (checked; the run's 0.518 counted A and C1 clusters separately) |
| HCv2-24(a) | h0_cell | R | SUPPORTED | dry run | 188 -> 470 | **1.000** | 1.000 / 0.628 | 0.000 | 0.007 |  |
| HCv2-24(b) | h0_cell_exceeds | R | SUPPORTED | dry run | 32 -> 80 | **1.000** | 1.000 / 0.999 | 0.000 | 0.000 |  |

## 4. Resizes and parts that no resize helps

**Resizes made** (seeds only):

| Part | Design and systems | Before | After | P(SUP \| dev) before -> after | Extra tasks |
|---|---|---|---|---|---|
| HCv2-1 (RAM-PE pool) | family-A N_independent_noise, N_ar1 | 20 seeds each | 46 each (20000-20045) | 1.000 with 0 development events; the pool at 0.05/5 tolerates 1 event at 92 clusters (bound 0.06998), 2 at 120, 3 at 160; pool-level Jeffreys predictive 0.64 at 92, 0.81 at 300 | 52 |
| HCv2-23 | C1 W_NAS_no_workspace, W_NAS_broadcast_only, W_IIM_feedforward | 45 (20000-20044) | 129 (20000-20128) | 0.069 -> 0.884 (bootstrap 0.885; >= 0.88 at every size from 129 to 400; 0.956 at 148) | 252 |
| HCv2-14(f) | c_int sweep of family A, 10 levels | 10 per level (20000-20009) | 65 per level (20000-20064) | 0.16 -> 0.818 with whole seed profiles resampled (0.64 at 45, 0.79 at 60, 0.85 at 70, 0.88 at 80; B = 10000); level-wise draws give 0.98 | 550 |
| HCv2-14(f), C1 cell | c_int sweep of family C1 | 10 per level | 65 per level (20000-20064) | not computable (C1 IIM held out, HO-4); extended by symmetry, so the part's P(SUP) is at most the family-A value | 550 |
| HCv2-6(a) | family-A anchor replication | 20900-20919 | 20900-20939 | about 0.60 (no resize reaches 0.8) | 140 |
| HCv2-6(b) | Hopf forward anchor replication | 20900-20919 | 20900-20939 | section 4.1 | 20 |

Why HCv2-23 needed the resize: its any-event clusters are keyed by the seed
alone, and the confirmatory A and C1 single deficits share seeds 20000-20044, so
the pool holds 45 merged clusters, not 90. With merged clusters the family-A
event rate (3 of 52 development any-event clusters, from the NAS SE reversion
under H) sits just below the bound 0.07, so a uniform growth of all single
deficits does not help (0.10 at 200 seeds). The resize works only by diluting
the bound with C1 clusters free of events: in development no C1 verdict could be
MPC_CONSISTENT, because C1 IIM is held out, and the evidence of no events rests
on NAS never being PRESENT (0 of 72 rows over the three C1 targets). With a
Jeffreys predictive on the family-A rate P(SUP) is 0.78 at 129 (0.87 at 148);
on both rates 0.39.

Why HCv2-14(f) is resized on seed profiles: the confirmatory levels share their
seeds and the null seeds derived from them, and the development values carry a
seed effect, so whole seed profiles are resampled. The development effect rests
on 4 seeds per level and rises then falls with c_int (rho 0.09 over 40 runs);
`P(FALSIFIED | the development effect)` is 0.18 at 65 seeds.

**Parts predicted SUPPORTED below 0.8 that no seed resize helps:** see the table
in section 8 of the preregistration (HCv2-0(b), HCv2-2, HCv2-4(a,b), HCv2-5(a),
HCv2-6(a), HCv2-6(b) for three Hopf views, HCv2-7(iii), HCv2-7(v-A),
HCv2-8(a-H), HCv2-8(b-H-rank1), HCv2-12(d), HCv2-15(a), HCv2-15(b), HCv2-17(d),
HCv2-20(a), HCv2-20(f-single)). HCv2-22(ii) is predicted FALSIFIED with
P = 0.002.

**Parts above 0.8 that miss `P(FALSIFIED | correct) <= 0.05`** and cannot be
fixed by seeds: HCv2-8(b-R) at 0.125 (three variant cells at alpha each, count
rule with m = 1); HCv2-0(a) (a binary usable share at the 0.9 boundary); the
two-outcome value rules, whose `P(FALSIFIED | correct) = 1 - P(SUPPORTED)`; the
three-zone parts sit at 0.013-0.050 (the largest HCv2-7(i) and (ii) at 0.049
and HCv2-15(b) at 0.050).

**HCv2-12(b), synthetic.** No estimator ran at ring 0.9 or 1.5 (HO-5). Exact `c`
over the ring-0.45 anchor: bidirectional 1 -> 0.802 -> 0.889; directional 1 ->
-0.334 -> -1.087. The per-run SD of `c` is taken as the largest on the
development ring cells at couplings <= 0.5, T 30000 (0.065 bidirectional, 0.077
directional); levels independent given the seed, every run defined at 0.9 and
1.5 (the conservative case); seed-paired sign test at one-sided p < 0.025. The
weak step is bidirectional 0.9 -> 1.5 (exact change +0.087). P(SUP) is 0.995 at
40 seeds and the development SD; if the SD at 0.9 and 1.5 is larger: x1.5 0.85
(40 seeds), 0.95 (60), 0.99 (80); x2 0.61, 0.75, 0.89, 0.95 (40, 60, 80, 100);
x3 0.30, 0.40, 0.55, 0.65, 0.89 (40, 60, 80, 100, 160). No resize was made.

### 4.1 Forward anchor replication power (HCv2-6(b))

Validity-only power that the development anchor status of a forward view recurs
on the replication block, from the held-out reference anchors (seeds 900-939,
after the release), by resampling the reference seeds (2000 draws). The Hopf
arm was extended to 40 seeds because three of its IIM views are below 0.9 at
20; the family-A arms were not.

| Protocol and principle | Development status | Power at 20 seeds | Power at 40 seeds |
|---|---|---|---|
| `fwdA-eeg64 / IIM` | valid | 0.9865 | 1.0 |
| `fwdA-eeg64 / PDI` | valid | 1.0 | 1.0 |
| `fwdA-eeglow / IIM` | valid | 0.9745 | 1.0 |
| `fwdA-eeglow / PDI` | valid | 1.0 | 1.0 |
| `fwdA-source / IIM` | valid | 1.0 | 1.0 |
| `fwdA-source / PDI` | valid | 1.0 | 1.0 |
| `fwdA_bold-bold / IIM` | invalid | 1.0 | 1.0 |
| `fwdA_bold-bold / PDI` | valid | 1.0 | 1.0 |
| `fwdA_bold-source / IIM` | valid | 1.0 | 1.0 |
| `fwdA_bold-source / PDI` | valid | 1.0 | 1.0 |
| `hopf-bold / IIM` | invalid | 1.0 | 1.0 |
| `hopf-bold / NAS` | invalid | 1.0 | 1.0 |
| `hopf-eeg64+iim_v1_quadrants / IIM` | invalid | 1.0 | 1.0 |
| `hopf-eeg64_noref+iim_v1_quadrants / IIM` | valid | 1.0 | 1.0 |
| `hopf-eeg64_noref / IIM` | valid | 0.264 | 0.5225 |
| `hopf-eeg64 / IIM` | invalid | 0.862 | 0.7925 |
| `hopf-eeg64 / NAS` | valid | 1.0 | 1.0 |
| `hopf-eeglow / IIM` | valid | 0.93 | 0.999 |
| `hopf-eeglow / NAS` | valid | 1.0 | 1.0 |
| `hopf-mne_template / IIM` | valid | 0.6985 | 0.9535 |
| `hopf-mne_template / NAS` | valid | 1.0 | 1.0 |
| `hopf-source / IIM` | valid | 0.9805 | 1.0 |
| `hopf-source / NAS` | valid | 1.0 | 1.0 |

HCv2-6(b) needs every view to replicate. With the views treated as independent
(they share seeds, so this is an approximation), the part's P(SUPPORTED |
development) is the product of the powers at the planned sizes (Hopf views at
40 seeds, family-A views at 20): 0.5225 x 0.9535 x 0.7925 x 0.999 x 0.9865 x
0.9745, about 0.38; the alternative outcome is FALSIFIED. With HCv2-6(a) at
about 0.6, the hypothesis HCv2-6 is SUPPORTED with probability about 0.23.

## 5. Confirmatory cost

Development cost model: `CPU_S_PER_TASK` of `scripts/v2/dev_calibration.py`,
CPU seconds per task by design, measured at `d3bcb09` on development seeds;
wall time = CPU-h x 1.39 (measured 12-worker contention) / 12; prerequisite M
8 s per seed. Task counts are those of the frozen plan.

| Design | Tasks | Seeds | CPU s per task | CPU-h (cost model) | CPU-h (C1 IIM priced) |
|---|---|---|---|---|---|
| `A_witnesses` | 547 | 20000-20045 | 105.83 | 16.08 | 16.08 |
| `A_sweeps` | 810 | 20000-20064 | 113.12 | 25.45 | 25.45 |
| `A_factorial` | 320 | 20000-20009 | 116.28 | 10.34 | 10.34 |
| `A_adversaries` | 140 | 20000-20019 | 110.6 | 4.30 | 4.30 |
| `A_heldout` | 40 | 20000-20019 | 93.77 | 1.04 | 1.04 |
| `C1_witnesses` | 612 | 20000-20128 | 4.21 | 0.72 | 12.68 |
| `C1_sweeps` | 750 | 20000-20064 | 3.76 | 0.78 | 15.53 |
| `C1_factorial` | 320 | 20000-20009 | 3.46 | 0.31 | 6.63 |
| `RAM160` | 440 | 20000-20039 | 0.48 | 0.06 | 0.06 |
| `A_twins` | 310 | 20000-20009 | 119.22 | 10.27 | 10.27 |
| `C1_twins` | 180 | 20000-20009 | 3.05 | 0.15 | 3.73 |
| `RAM160_twins` | 210 | 20000-20009 | 0.31 | 0.02 | 0.02 |
| `A_anchors` | 280 | 20900-20939 | 97.55 | 7.59 | 7.59 |
| `C1_anchors` | 80 | 20900-20919 | 74.56 | 1.66 | 1.66 |
| `RAM160_anchors` | 40 | 20900-20919 | 0.32 | 0.00 | 0.00 |
| `whole_brain` | 371 | 20000-20149 | 107.97 | 11.13 | 11.13 |
| `forward_family_a` | 332 | 20000-20149 | 127.0 | 11.71 | 11.71 |
| `forward_family_a_bold` | 271 | 20000-20149 | 40.77 | 3.07 | 3.07 |
| `forward_anchor_replication` | 80 | 20900-20939 | 118.83 | 2.64 | 2.64 |
| `null_calibration` | 800 | 20000-20000 | 34.83 | 7.74 | 7.74 |
| family B (HCv2-2 1800, HCv2-11 480, HCv2-12 1680, HCv2-13 300) | 4260 | 20000-20499 | 13.19 | 15.61 | 15.61 |
| prerequisite M (oracle checks) | 40 seeds | 20000-20039 | 8 per seed | 0.09 | 0.09 |
| **Total** | **11193 tasks** + 40 seeds | | | **130.75 (15.1 h wall)** | **167.35 (19.4 h wall)** |

The development cost of the C1 witness, sweep, factorial and twin tasks
(3.05-4.21 s per task) was measured without IIM, which is held out on C1 in
development (HO-4); the confirmatory C1 tasks score IIM under R and H and the
bidirectional form under both. The last column prices them at the C1
reference-block rate (74.56 s per task, the same scorings). The forward BOLD arm
is priced uncurtailed.

Per step of `scripts/v2/mpcbench_confirmatory_v2.sh` (for packing steps into
jobs; wall time at `W` workers is about CPU-h x 1.39 / `W` by the cost model):

| Step | Designs | CPU-h (cost model) | CPU-h (C1 IIM priced) |
|---|---|---|---|
| `manipulation` | prerequisite M | 0.09 | 0.09 |
| `anchor_replication` | `A_anchors`, `C1_anchors`, `RAM160_anchors`, `forward_anchor_replication` | 11.89 | 11.89 |
| `A_witnesses` | `A_witnesses` | 16.08 | 16.08 |
| `C1_witnesses` | `C1_witnesses` | 0.72 | 12.68 |
| `null_calibration` | `null_calibration` | 7.74 | 7.74 |
| `forward` | `whole_brain`, `forward_family_a`, `forward_family_a_bold` | 25.91 | 25.91 |
| `family_b` | family B | 15.61 | 15.61 |
| `twins` | `A_twins`, `C1_twins`, `RAM160_twins` | 10.44 | 14.02 |
| `ram_only` | `RAM160` | 0.06 | 0.06 |
| `sweeps_factorial` | `A_sweeps`, `A_factorial`, `C1_sweeps`, `C1_factorial` | 36.88 | 57.95 |
| `adversaries_heldout` | `A_adversaries`, `A_heldout` | 5.34 | 5.34 |

History of the estimate: the plan before the resizes was 9821 tasks, 111.9 CPU-h
(132.8 with C1 IIM priced); the resizes of HCv2-23 and HCv2-14(f) (both
families) brought it to 11173 tasks, 130.1 CPU-h (166.7); the Hopf replication
extension adds 20 tasks (0.66 CPU-h). The design projected about 106 CPU-h for
Tier A before the calibration.
