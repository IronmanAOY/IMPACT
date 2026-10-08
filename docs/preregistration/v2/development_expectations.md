# MPC-Bench v2: development expectations

DEVELOPMENT - NOT A RESULT. Companion of [`MPC_BENCH_PREREGISTRATION_V2.md`](../MPC_BENCH_PREREGISTRATION_V2.md), sections 6 and 8. Every number below comes from development seeds (0-999). None of it may be quoted as a confirmatory result, and none of it changes a threshold, a rule or a prediction.

## Sources

1. The development expectation texts of `protocols/v2/hypotheses_v2.json` (fields `expectation` of the hypotheses and parts), quoted verbatim. They were written from the development calibration (CD-10) and from the component designs.
2. The planned development item `dry_run_evaluation` (CD-10): the v2 evaluator in development mode on the development records of the dry run (seeds 320-439), the family-A, C1 and RAM-only reference blocks (900-939), the twins (820-824) and the oracle checks (320-359), re-judged under the generated protocols. Command (from the repository root): `python scripts/v2/dev_calibration.py run dry_run_evaluation --protocol-dir protocols/v2/generated --workers 12`.

| Item | Value |
|---|---|
| Run | 2026-10-08, code `1.1.0+gf3a481202b1bcfb835c2ad365d18faa164539edd` |
| Records re-judged | 2412 (not re-judged: 0); evaluator records 2807 |
| Evaluator | `mpc-bench-hypotheses-v2/1.0.0`, split `development`, status `DEVELOPMENT - NOT A RESULT` |
| Hypotheses file read | `protocols/v2/hypotheses_v2.json` at commit f3a4812 (status `draft`; file SHA-256 `fd96d2a2c7f20f7bb67e0f3c0ef3bdabda8dfea673e5b64756b067596e474033`; spec hash `spec_sha256`, the SHA-256 of the canonical sorted-key JSON that the evaluation records, `1a79e59671e2fd17a085656338cc1ea3c76ff65fc39d8c44e528d196e5e00a32`). The frozen file differs from it only in the top-level `status`, which the evaluator does not read in development mode. |
| Output files | `hypotheses_v2.json` SHA-256 `46c1372b62c92d49379d794b09c9a1c2d8118417dff5f951cca56826df7e6737`; `hypotheses_v2_parts.csv` `09f4d7e19c5b2a88ad54842a7bea3b48f6d045a1e936a88d7d7f449da9f34ea3`; `hypotheses_v2_tally.csv` `591eb8d2c5f43499e1010d246c24175cc8215a8d3e65ca7d4385a735d4aa3a98` |
| Uncalibrated SE methods (HCv2-4 reversion) | nas-v3-2026.10@jackknife_contiguous_10 |
| Protocol hash mismatches | 0 |
| Evaluator errors | 0 |

The dry run is at about 15 % of the confirmatory scale (for example 12 witness seeds instead of 20 or 45, 4 seeds per sweep level instead of 10 or 65), so many three-zone and H0-cell parts end INDETERMINATE for lack of runs, and the HO parts have no development rows by construction (they are NOT_EVALUABLE here). The operating characteristics (companion `operating_characteristics.md`) resample the same development rows to the confirmatory sizes and are the better guide to the confirmatory outcome.

## Tally of the development dry run

| Kind | SUPPORTED | FALSIFIED | INDETERMINATE | NOT_EVALUABLE | NOT_TESTABLE_BY_DESIGN | NOT_RUN |
|---|---|---|---|---|---|---|
| decisive parts | 47 | 10 | 13 | 13 | 1 | - |
| hypotheses | 5 | 8 | 9 | 3 | 0 | 8 |

Reported (non-decisive) parts: 18. Parts removed before the freeze: 0.

## Summary by hypothesis

| Hypothesis | Prediction | Development dry run | Decisive parts (dry-run outcome) |
|---|---|---|---|
| HCv2-0 | SUPPORTED | FALSIFIED | HCv2-0(a) SUPPORTED; HCv2-0(b) FALSIFIED |
| HCv2-1 | SUPPORTED | INDETERMINATE | HCv2-1 INDETERMINATE |
| HCv2-2 | SUPPORTED | INDETERMINATE | HCv2-2 INDETERMINATE |
| HCv2-3 | FALSIFIED | FALSIFIED | HCv2-3 FALSIFIED |
| HCv2-4 | SUPPORTED | FALSIFIED | HCv2-4(a,b) FALSIFIED; HCv2-4(c) SUPPORTED |
| HCv2-5 | SUPPORTED | INDETERMINATE | HCv2-5(a) INDETERMINATE; HCv2-5(b) INDETERMINATE |
| HCv2-6 | SUPPORTED | SUPPORTED | HCv2-6(a) SUPPORTED; HCv2-6(b) NOT_EVALUABLE |
| HCv2-7 | SUPPORTED | FALSIFIED | HCv2-7(i) SUPPORTED; HCv2-7(ii) SUPPORTED; HCv2-7(iii) FALSIFIED; HCv2-7(iv) SUPPORTED; HCv2-7(v-A) FALSIFIED; HCv2-7(v-C1) SUPPORTED |
| HCv2-8 | SUPPORTED | INDETERMINATE | HCv2-8(a-H) INDETERMINATE; HCv2-8(a-R) SUPPORTED; HCv2-8(b-H) SUPPORTED; HCv2-8(b-H-rank1) INDETERMINATE; HCv2-8(b-R) SUPPORTED; HCv2-8(c-H-secondary) SUPPORTED; HCv2-8(c-R-secondary) SUPPORTED; HCv2-8(c-H-primary) SUPPORTED; HCv2-8(d) SUPPORTED |
| HCv2-9 | SUPPORTED | NOT_EVALUABLE | HCv2-9(a) NOT_EVALUABLE; HCv2-9(b) NOT_EVALUABLE; HCv2-9(c) NOT_EVALUABLE; HCv2-9(d) NOT_EVALUABLE; HCv2-9(e) NOT_EVALUABLE |
| HCv2-10 | SUPPORTED | NOT_EVALUABLE | HCv2-10 NOT_EVALUABLE |
| HCv2-11 | SUPPORTED | SUPPORTED | HCv2-11(a) SUPPORTED; HCv2-11(b-rho) SUPPORTED; HCv2-11(b-present) SUPPORTED; HCv2-11(c) SUPPORTED |
| HCv2-12 | SUPPORTED | INDETERMINATE | HCv2-12(a) SUPPORTED; HCv2-12(b) NOT_EVALUABLE; HCv2-12(c-low) SUPPORTED; HCv2-12(c-high) SUPPORTED; HCv2-12(d) INDETERMINATE |
| HCv2-13 | SUPPORTED | INDETERMINATE | HCv2-13(a) INDETERMINATE; HCv2-13(b-present) SUPPORTED; HCv2-13(b-median) SUPPORTED; HCv2-13(c) SUPPORTED |
| HCv2-14 | SUPPORTED | FALSIFIED | HCv2-14(a) SUPPORTED; HCv2-14(b) INDETERMINATE; HCv2-14(c) NOT_TESTABLE_BY_DESIGN; HCv2-14(d) SUPPORTED; HCv2-14(e) SUPPORTED; HCv2-14(f) FALSIFIED |
| HCv2-15 | SUPPORTED | FALSIFIED | HCv2-15(a) FALSIFIED; HCv2-15(b) FALSIFIED; HCv2-15(c) NOT_EVALUABLE; HCv2-15(d-present) SUPPORTED; HCv2-15(d-absent) INDETERMINATE; HCv2-15(e-FMd) NOT_EVALUABLE |
| HCv2-16 | SUPPORTED | SUPPORTED | HCv2-16 SUPPORTED |
| HCv2-17 | SUPPORTED | FALSIFIED | HCv2-17(a) SUPPORTED; HCv2-17(b) SUPPORTED; HCv2-17(c) SUPPORTED; HCv2-17(d) FALSIFIED; HCv2-17(e) SUPPORTED |
| HCv2-18 | SUPPORTED | SUPPORTED | HCv2-18 SUPPORTED |
| HCv2-19 | SUPPORTED | SUPPORTED | HCv2-19(a) SUPPORTED; HCv2-19(b) SUPPORTED; HCv2-19(c) SUPPORTED; HCv2-19(d) SUPPORTED; HCv2-19(e) SUPPORTED |
| HCv2-20 | SUPPORTED | INDETERMINATE | HCv2-20(a) SUPPORTED; HCv2-20(b) NOT_EVALUABLE; HCv2-20(c) SUPPORTED; HCv2-20(d) NOT_EVALUABLE; HCv2-20(e) INDETERMINATE; HCv2-20(f-single) SUPPORTED; HCv2-20(f-no-multistability) SUPPORTED; HCv2-20(f-PC) SUPPORTED |
| HCv2-21 | SUPPORTED | NOT_EVALUABLE | HCv2-21 NOT_EVALUABLE |
| HCv2-22 | FALSIFIED | FALSIFIED | HCv2-22(i) SUPPORTED; HCv2-22(ii) SUPPORTED; HCv2-22(iii) FALSIFIED |
| HCv2-23 | SUPPORTED | INDETERMINATE | HCv2-23 INDETERMINATE |
| HCv2-24 | SUPPORTED | INDETERMINATE | HCv2-24(a) SUPPORTED; HCv2-24(b) INDETERMINATE |
| HCv2-B1 | - | NOT_RUN | - |
| HCv2-B2 | - | NOT_RUN | - |
| HCv2-B3 | - | NOT_RUN | - |
| HCv2-B4 | - | NOT_RUN | - |
| HCv2-B5 | - | NOT_RUN | - |
| HCv2-B6 | - | NOT_RUN | - |
| HCv2-B7 | - | NOT_RUN | - |
| HCv2-B8 | - | NOT_RUN | - |

## Per hypothesis and part

### HCv2-0: Prerequisite M (manipulation and realisation checks)

Development expectation (hypotheses file): v1: 12/12 family x switch cells usable (worst 39/40). Development (CD-13, seeds 320-359): 26 of 27 checks pass 40/40, every switch check of families A and C1 among them and the slow-context check (shortest complete context run 30.0-34.5 s against 30 s); ADV_NAS_staggered_tau10 driver_reaches_every_module passes 34/40 (36 needed; the six failures, 0.035-0.048 against the floor 0.05, are in the 8 s modules), so that condition is predicted not usable: P(usable) is 0.26 at a true pass rate of 0.85 and 0.63 at 0.90, about 0.74 not usable. If it is not usable, its HCv2-0(b) cell is FALSIFIED and HCv2-9(d) is NOT_EVALUABLE (reason ORACLE). The condition, its check and its threshold are unchanged. Reported (development): PC_half lies between the off and nominal medians in 10 of 12 family x switch cells; not for K in A and C1, whose context-information signature is higher at PC_half's K = 3 than at the nominal K = 6 (medians 0.66 against 0.49 in A, 0.64 against 0.63 in C1); the twins pass 1920/1920 (8 twin witnesses x 40 seeds x replicates 1-6).

Dry-run outcome of the hypothesis: **FALSIFIED**.

- **HCv2-0(a)** (decisive; rule `usable_share`): SUPPORTED - usable iff passes >= ceil(0.9 x seeds)
  - `family=A|check_id=eta`: 40/40 (SUPPORTED)
  - `family=A|check_id=K`: 40/40 (SUPPORTED)
  - `family=A|check_id=g_b`: 40/40 (SUPPORTED)
  - `family=A|check_id=ff_only`: 40/40 (SUPPORTED)
  - `family=A|check_id=c_int`: 40/40 (SUPPORTED)
  - `family=A|check_id=e`: 40/40 (SUPPORTED)
  - `family=C1|check_id=eta`: 40/40 (SUPPORTED)
  - `family=C1|check_id=K`: 40/40 (SUPPORTED)
  - `family=C1|check_id=g_b`: 40/40 (SUPPORTED)
  - `family=C1|check_id=ff_only`: 40/40 (SUPPORTED)
  - `family=C1|check_id=c_int`: 40/40 (SUPPORTED)
  - `family=C1|check_id=e`: 40/40 (SUPPORTED)
- **HCv2-0(b)** (decisive; rule `usable_share`): FALSIFIED - usable iff passes >= ceil(0.9 x seeds)
  - `family=A|check_id=ADV_NAS_staggered_tau10:tau10/driver_reaches_every_module`: 34/40 (FALSIFIED)
  - `family=A|check_id=ADV_NAS_staggered_driver:hierarchical/driver_reaches_every_module`: 40/40 (SUPPORTED)
  - `family=A|check_id=ADV_NAS_staggered_driver:hierarchical/stated_time_constants`: 40/40 (SUPPORTED)
  - `family=A|check_id=ADV_NAS_staggered_driver:reversed/driver_reaches_every_module`: 40/40 (SUPPORTED)
  - `family=A|check_id=ADV_NAS_staggered_driver:reversed/stated_time_constants`: 40/40 (SUPPORTED)
  - `family=A|check_id=ADV_NAS_staggered_driver:uniform/driver_reaches_every_module`: 40/40 (SUPPORTED)
  - `family=A|check_id=ADV_NAS_staggered_driver:uniform/stated_time_constants`: 40/40 (SUPPORTED)
  - `family=A|check_id=ADV_NAS_staggered_sat:saturating/driver_reaches_every_module`: 40/40 (SUPPORTED)
  - `family=A|check_id=ADV_NAS_staggered_sat:saturating/stated_time_constants`: 40/40 (SUPPORTED)
  - `family=A|check_id=ADV_NAS_staggered_tau10:tau10/stated_time_constants`: 40/40 (SUPPORTED)
  - `family=A|check_id=N_modules_disconnected/no_hub_periphery_path`: 40/40 (SUPPORTED)
  - `family=A|check_id=PC_nominal:slow_context_bold/slow_context_dwell`: 40/40 (SUPPORTED)
  - `family=A|check_id=W_PDI_no_multistability/no_ignition`: 40/40 (SUPPORTED)
  - `family=C1|check_id=N_modules_disconnected/no_hub_periphery_path`: 40/40 (SUPPORTED)
  - 1 further cells (1 SUPPORTED)
- **HCv2-0(c)** (reported; rule `describe`): REPORTED - reported
  - `family=A|check_id=PC_half:eta/between_off_and_nominal`: 1/1 (REPORTED)
  - `family=A|check_id=PC_half:K/between_off_and_nominal`: 0/1 (REPORTED)
  - `family=A|check_id=PC_half:g_b/between_off_and_nominal`: 1/1 (REPORTED)
  - `family=A|check_id=PC_half:ff_only/between_off_and_nominal`: 1/1 (REPORTED)
  - `family=A|check_id=PC_half:c_int/between_off_and_nominal`: 1/1 (REPORTED)
  - `family=A|check_id=PC_half:e/between_off_and_nominal`: 1/1 (REPORTED)
  - `family=C1|check_id=PC_half:eta/between_off_and_nominal`: 1/1 (REPORTED)
  - `family=C1|check_id=PC_half:K/between_off_and_nominal`: 0/1 (REPORTED)
  - `family=C1|check_id=PC_half:g_b/between_off_and_nominal`: 1/1 (REPORTED)
  - `family=C1|check_id=PC_half:ff_only/between_off_and_nominal`: 1/1 (REPORTED)
  - `family=C1|check_id=PC_half:c_int/between_off_and_nominal`: 1/1 (REPORTED)
  - `family=C1|check_id=PC_half:e/between_off_and_nominal`: 1/1 (REPORTED)
- **HCv2-0(d)** (reported; rule `describe`): REPORTED - reported
  - `family=A|system_id=PC_nominal`: 240/240 (REPORTED)
  - `family=A|system_id=PC_half`: 240/240 (REPORTED)
  - `family=A|system_id=W_PDI_single_attractor`: 240/240 (REPORTED)
  - `family=A|system_id=W_NAS_no_workspace`: 240/240 (REPORTED)
  - `family=A|system_id=W_IIM_feedforward`: 240/240 (REPORTED)
  - `family=C1|system_id=PC_nominal`: 240/240 (REPORTED)
  - `family=C1|system_id=W_NAS_no_workspace`: 240/240 (REPORTED)
  - `family=C1|system_id=W_IIM_feedforward`: 240/240 (REPORTED)

### HCv2-1: Null calibration of PRESENT, including conditioned nulls

Development expectation (hypotheses file): v1: NAS and IIM 0/1280, SRPI 12/1640. NAS under R (development): disconnected 0/20 PRESENT in A and C; staggered 0/20 per variant. RAM-PE (development): the 128 null-calibration RAM-PE components are UNDEFINED(INSUFFICIENT_UPDATES) and are not counted; the family-A null witnesses give 48 defined rows, 0 PRESENT.

Dry-run outcome of the hypothesis: **INDETERMINATE**.

- **HCv2-1** (decisive; rule `h0_cell`): INDETERMINATE - a pooled upper bound is not below the bound
  - pooled: IIM 0/144 (120 clusters), upper bound 0.0376; NAS 0/228 (192 clusters), upper bound 0.0237; PDI 0/176 (152 clusters), upper bound 0.0298; SRPI 0/176 (152 clusters), upper bound 0.0298; RAM 0/48 (24 clusters), upper bound 0.175; m = 72; P = 5
  - 72 further cells (72 SUPPORTED, 0 events in each)
- **HCv2-1(definedness)** (reported; rule `describe`): REPORTED - reported
  - `null_calibration:ar1:T1200:N16|RAM`: 8/8 (REPORTED)
  - `null_calibration:ar1:T1200:N8|RAM`: 8/8 (REPORTED)
  - `null_calibration:ar1:T2400:N16|RAM`: 8/8 (REPORTED)
  - `null_calibration:ar1:T2400:N8|RAM`: 8/8 (REPORTED)
  - `null_calibration:pink:T1200:N16|RAM`: 8/8 (REPORTED)
  - `null_calibration:pink:T1200:N8|RAM`: 8/8 (REPORTED)
  - `null_calibration:pink:T2400:N16|RAM`: 8/8 (REPORTED)
  - `null_calibration:pink:T2400:N8|RAM`: 8/8 (REPORTED)
  - `null_calibration:surrogate_iid:T1200:N16|RAM`: 8/8 (REPORTED)
  - `null_calibration:surrogate_iid:T1200:N8|RAM`: 8/8 (REPORTED)
  - `null_calibration:surrogate_iid:T2400:N16|RAM`: 8/8 (REPORTED)
  - `null_calibration:surrogate_iid:T2400:N8|RAM`: 8/8 (REPORTED)
  - `null_calibration:surrogate_linear:T1200:N16|RAM`: 8/8 (REPORTED)
  - `null_calibration:surrogate_linear:T1200:N8|RAM`: 8/8 (REPORTED)
  - 58 further cells (58 REPORTED)

### HCv2-2: IIM independence-null rank calibration, including conditioned nulls

Development expectation (hypotheses file): none stated at hypothesis level; the dry run below is the development expectation.

Dry-run outcome of the hypothesis: **INDETERMINATE**.

- **HCv2-2** (decisive; rule `h0_cell`): INDETERMINATE - a pooled upper bound is not below the bound
  - pooled: IIM 18/270 (135 clusters), upper bound 0.113; m = 18; P = 1
  - `independent:T10000:bidirectional`: 4/15; lower 0.0414 (SUPPORTED)
  - `independent:T10000:directional`: 1/15; lower 0.000185 (SUPPORTED)
  - `independent:T1000:bidirectional`: 3/15; lower 0.0194 (SUPPORTED)
  - `independent:T1000:directional`: 1/15; lower 0.000185 (SUPPORTED)
  - `independent:T30000:bidirectional`: 1/15; lower 0.000185 (SUPPORTED)
  - `independent:T3000:directional`: 1/15; lower 0.000185 (SUPPORTED)
  - `residualised_switching:bidirectional`: 1/15; lower 0.000185 (SUPPORTED)
  - `residualised_switching:directional`: 1/15; lower 0.000185 (SUPPORTED)
  - `stratified:T10000:bidirectional`: 1/15; lower 0.000185 (SUPPORTED)
  - `stratified:T30000:bidirectional`: 2/15; lower 0.00526 (SUPPORTED)
  - `stratified:T30000:directional`: 1/15; lower 0.000185 (SUPPORTED)
  - `stratified:T3000:bidirectional`: 1/15; lower 0.000185 (SUPPORTED)
  - 6 further cells (6 SUPPORTED, 0 events in each)
- **HCv2-2(reported)** (reported; rule `describe`): REPORTED - reported
  - `residualised_continuous:T12000:directional`: 0/15 (REPORTED)
  - `residualised_continuous:T12000:bidirectional`: 0/15 (REPORTED)
  - `residualised_switching:T12000:directional`: 1/15 (REPORTED)
  - `residualised_switching:T12000:bidirectional`: 1/15 (REPORTED)

### HCv2-3: Interval calibration at the null (HR1, restated)

Development expectation (hypotheses file): Development (null calibration at seed 400 with 32 replicates per null kind, and the family-A null witnesses on seeds 320-331, judged under the decided build): all 152 PDI null rows (128 null-calibration, 24 witness) are admitted concordant components without a sampling SE, so the PDI cell is expected NOT_EVALUABLE; HCv2-3(concordant) reports their count and share (152 of 152). SRPI v1 FALSIFIED on the conservative side (kappa0 0.47, 90 % CI 0.43-0.52), as predicted. NAS conservative in both directions (kappa0 0.69; 90 % CI 0.62-0.77 receive, 0.63-0.78 return; lower-tail rates 0.067 and 0.058): FALSIFIED on the conservative side. IIM (kappa0 0.76, 0.69-0.85) and RAM-PE (0.74, 0.60-0.99; 24 defined rows) conservative and short of the calibration criterion (INDETERMINATE).

Dry-run outcome of the hypothesis: **FALSIFIED**.

- **HCv2-3** (decisive; rule `kappa_null`): FALSIFIED - kappa0 in [0.8, 1.25] with its 90 % interval inside [0.67, 1.5], and both tails <= 0.075
  - P = 6
  - `IIM`: kappa 0.759 [0.686, 0.85]; tails upper 1/120, lower 1/120; side conservative (INDETERMINATE)
  - `PDI`: concordant rows 152; fewer than 3 rows with a sampling SE: the admitted concordant rows carry none (NOT_EVALUABLE)
  - `RAM`: kappa 0.744 [0.602, 0.986]; tails upper 0/24, lower 1/24; side conservative (INDETERMINATE)
  - `SRPI`: kappa 0.471 [0.431, 0.521]; tails upper 0/152, lower 0/152; side conservative (FALSIFIED)
  - `NAS:receive`: kappa 0.687 [0.621, 0.769]; tails upper 0/120, lower 8/120; side conservative (FALSIFIED)
  - `NAS:return`: kappa 0.693 [0.627, 0.776]; tails upper 1/120, lower 7/120; side conservative (FALSIFIED)
- **HCv2-3(concordant)** (reported; rule `describe`): REPORTED - reported
  - `IIM`: 0/120 (REPORTED)
  - `NAS`: 0/120 (REPORTED)
  - `PDI`: 152/152 (REPORTED)
  - `RAM`: 0/24 (REPORTED)
  - `SRPI`: 0/152 (REPORTED)

### HCv2-4: SE calibration against white-box twins (all v2 SE methods)

Development expectation (hypotheses file): Development per-class summary (CD-2 to CD-4; twins 820-824 with 7 sessions each, RAM-only twins with 8): NAS FALSIFIED on the anti-conservative side; no anti-conservative failure for IIM, RAM-PE, PDI or SRPI (their failures are conservative or short of the falsification criterion). NAS (jackknife_contiguous_10, calibrated in 7 of 32 classes): in the A-H return direction kappa 3.04 (90 % CI 2.52-3.87; tails 0.171 below, 0.114 above) on W_PDI_single_attractor and 1.94 (1.61-2.47) on PC_nominal; 0.52-0.91 in most C1 cells. IIM (circular_block_bootstrap_10pct_B50 at se_df 9): calibrated in 8 of 20 classes; kappa 0.47-0.72 in the PC_half classes; 4 classes with one tail exceedance in 35 sessions. RAM-PE (shift_null_sd): calibrated in 6 of 13 classes; on the joint bench PC_half kappa 1.25 (1.04-1.59, not entirely above 1.25), above-tail 0.057. PDI W_PDI_single_attractor under A-R and A-H: a jackknife SE in 2 of 35 sessions (33 admitted concordant), so these cells are not eligible. The reversion rule is therefore expected to re-classify NAS ABSENTs; hypotheses that use NAS ABSENT are reported both ways.

Dry-run outcome of the hypothesis: **FALSIFIED**.

- **HCv2-4(a,b)** (decisive; rule `kappa_twins`): FALSIFIED - kappa in [0.8, 1.25] with its 90 % interval inside [0.67, 1.5]; q_A tails <= 0.02
  - m = 170
  - `A:PC_half:None:H|IIM|iim_bidirectional|iim-v5-2026.10|circular_block_bootstrap_10pct_B50`: kappa 0.528 [0.432, 0.687]; tails below 0/31, above 0/31; side conservative; defined share 0.886 (FALSIFIED)
  - `A:PC_half:None:H|IIM|primary|iim-v5-2026.10|circular_block_bootstrap_10pct_B50`: kappa 0.469 [0.383, 0.609]; tails below 0/31, above 0/31; side conservative; defined share 0.886 (FALSIFIED)
  - `A:PC_half:None:H|PDI|primary|pdi-v3-2026.10|jackknife_contiguous_10`: kappa 0.235 [0.195, 0.3]; tails below 0/35, above 0/35; side conservative; defined share 1 (FALSIFIED)
  - `A:PC_half:None:H|SRPI|primary|srpi-v2-2026.09|jackknife_pairs_10`: kappa 0.6 [0.496, 0.764]; tails below 0/35, above 0/35; side conservative; defined share 1 (FALSIFIED)
  - `A:PC_half:None:R|IIM|primary|iim-v5-2026.10|circular_block_bootstrap_10pct_B50`: kappa 0.625 [0.518, 0.796]; tails below 0/35, above 0/35; side conservative; defined share 1 (FALSIFIED)
  - `A:PC_half:None:R|PDI|primary|pdi-v3-2026.10|jackknife_contiguous_10`: kappa 0.235 [0.195, 0.3]; tails below 0/35, above 0/35; side conservative; defined share 1 (FALSIFIED)
  - `A:PC_half:None:R|SRPI|primary|srpi-v2-2026.09|jackknife_pairs_10`: kappa 0.6 [0.496, 0.764]; tails below 0/35, above 0/35; side conservative; defined share 1 (FALSIFIED)
  - `A:PC_nominal:None:H|NAS:return|primary|nas-v3-2026.10|jackknife_contiguous_10`: kappa 1.942 [1.608, 2.473]; tails below 4/35, above 2/35; side anti_conservative; defined share 1 (FALSIFIED)
  - `A:PC_nominal:None:H|PDI|primary|pdi-v3-2026.10|jackknife_contiguous_10`: kappa 0.301 [0.249, 0.384]; tails below 0/35, above 0/35; side conservative; defined share 1 (FALSIFIED)
  - `A:PC_nominal:None:R|PDI|primary|pdi-v3-2026.10|jackknife_contiguous_10`: kappa 0.301 [0.249, 0.384]; tails below 0/35, above 0/35; side conservative; defined share 1 (FALSIFIED)
  - `A:W_IIM_feedforward:None:H|PDI|primary|pdi-v3-2026.10|jackknife_contiguous_10`: kappa 0.406 [0.336, 0.517]; tails below 0/35, above 0/35; side conservative; defined share 1 (FALSIFIED)
  - `A:W_IIM_feedforward:None:R|NAS:return|primary|nas-v3-2026.10|jackknife_contiguous_10`: kappa 0.488 [0.404, 0.621]; tails below 0/35, above 0/35; side conservative; defined share 1 (FALSIFIED)
  - `A:W_IIM_feedforward:None:R|PDI|primary|pdi-v3-2026.10|jackknife_contiguous_10`: kappa 0.406 [0.336, 0.517]; tails below 0/35, above 0/35; side conservative; defined share 1 (FALSIFIED)
  - `A:W_NAS_no_workspace:None:H|PDI|primary|pdi-v3-2026.10|jackknife_contiguous_10`: kappa 0.325 [0.269, 0.414]; tails below 0/35, above 0/35; side conservative; defined share 1 (FALSIFIED)
  - 71 further cells (13 FALSIFIED, 35 INDETERMINATE, 2 NOT_EVALUABLE, 21 SUPPORTED)
- **HCv2-4(c)** (decisive; rule `concordance_twins`): SUPPORTED - share of twin sessions differing from the concordant count <= 0.05
  - `all`: 0/30; lower 0 (SUPPORTED)

### HCv2-5: No false exclusion of mechanism-bearing systems (HR2)

Development expectation (hypotheses file): 0 events (dry run with v2 SEs at CD-10).

Dry-run outcome of the hypothesis: **INDETERMINATE**.

- **HCv2-5(a)** (decisive; rule `false_exclusion`): INDETERMINATE - false exclusion per cell (seed clusters)
  - m = 14
  - `family=A|declaration_id=R|principle=IIM|concordant=False`: 0/180; 32 clusters; seed upper 0.161 (INDETERMINATE)
  - `family=A|declaration_id=R|principle=NAS|concordant=False`: 0/240; 32 clusters; seed upper 0.161 (INDETERMINATE)
  - `family=A|declaration_id=R|principle=PDI|concordant=False`: 0/271; 32 clusters; seed upper 0.161 (INDETERMINATE)
  - `family=A|declaration_id=R|principle=RAM|concordant=False`: 0/300; 32 clusters; seed upper 0.161 (INDETERMINATE)
  - `family=A|declaration_id=R|principle=SRPI|concordant=False`: 0/300; 32 clusters; seed upper 0.161 (INDETERMINATE)
  - `family=A|declaration_id=H|principle=IIM|concordant=False`: 0/264; 32 clusters; seed upper 0.161 (INDETERMINATE)
  - `family=A|declaration_id=H|principle=NAS|concordant=False`: 0/252; 32 clusters; seed upper 0.161 (INDETERMINATE)
  - `family=A|declaration_id=H|principle=PDI|concordant=False`: 0/271; 32 clusters; seed upper 0.161 (INDETERMINATE)
  - `family=A|declaration_id=H|principle=RAM|concordant=False`: 0/300; 32 clusters; seed upper 0.161 (INDETERMINATE)
  - `family=A|declaration_id=H|principle=SRPI|concordant=False`: 0/300; 32 clusters; seed upper 0.161 (INDETERMINATE)
  - `family=A|declaration_id=R|principle=PDI|concordant=True`: 0/1; seed upper 0.996 (INDETERMINATE)
  - `family=A|declaration_id=H|principle=PDI|concordant=True`: 0/1; seed upper 0.996 (INDETERMINATE)
  - `family=C1|declaration_id=R|principle=NAS|concordant=False`: 0/144; 12 clusters; seed upper 0.375 (INDETERMINATE)
  - `family=C1|declaration_id=H|principle=NAS|concordant=False`: 0/144; 12 clusters; seed upper 0.375 (INDETERMINATE)
- **HCv2-5(b)** (decisive; rule `false_exclusion`): INDETERMINATE - false exclusion per cell (seed clusters)
  - m = 4
  - `family=A|declaration_id=R`: 0/100; 32 clusters; seed upper 0.128 (SUPPORTED)
  - `family=A|declaration_id=H`: 0/144; 32 clusters; seed upper 0.128 (SUPPORTED)
  - `family=C1|declaration_id=R`: 0/136; 12 clusters; seed upper 0.306 (INDETERMINATE)
  - `family=C1|declaration_id=H`: 0/136; 12 clusters; seed upper 0.306 (INDETERMINATE)
- **HCv2-5(dose-only)** (reported; rule `describe`): REPORTED - reported
  - `family=A|declaration_id=R|principle=IIM`: 2/96; 20 clusters (REPORTED)
  - `family=A|declaration_id=R|principle=NAS`: 0/12 (REPORTED)
  - `family=C1|declaration_id=R|principle=PDI`: 0/240; 12 clusters (REPORTED)
  - `family=C1|declaration_id=R|principle=RAM`: 0/252; 12 clusters (REPORTED)
  - `family=C1|declaration_id=R|principle=SRPI`: 0/252; 12 clusters (REPORTED)
  - `family=C1|declaration_id=H|principle=PDI`: 0/240; 12 clusters (REPORTED)
  - `family=C1|declaration_id=H|principle=RAM`: 0/252; 12 clusters (REPORTED)
  - `family=C1|declaration_id=H|principle=SRPI`: 0/252; 12 clusters (REPORTED)
  - `family=C1|declaration_id=R|principle=NAS`: 0/60; 12 clusters (REPORTED)
  - `family=C1|declaration_id=H|principle=NAS`: 0/60; 12 clusters (REPORTED)
  - `family=A|declaration_id=H|principle=IIM`: 0/12; 8 clusters (REPORTED)

### HCv2-6: Anchor validity, specificity and replication

Development expectation (hypotheses file): none stated at hypothesis level; the dry run below is the development expectation.

Dry-run outcome of the hypothesis: **SUPPORTED**.

- **HCv2-6(a)** (decisive; rule `anchor_replicates`): SUPPORTED - every (protocol, principle) anchor status replicates
  - Expectation (hypotheses file): INDETERMINATE-capable (design 4.9): A-H|IIM is valid at 36/40 finite, on the 0.9 boundary, so its replication power is 0.565 at 20 seeds and 0.60 at 40, and no resize reaches 0.9; P(SUPPORTED | development) is about 0.60, A-H|IIM being the pair most likely to fail. Every other pair has power >= 0.96 at 20 seeds (CD-11).
  - `C1-H|PDI`: predicted None, observed None; no own-lesion replication runs (NOT_EVALUABLE)
  - `C1-H|RAM`: predicted None, observed None; no own-lesion replication runs (NOT_EVALUABLE)
  - `C1-H|SRPI`: predicted None, observed None; no own-lesion replication runs (NOT_EVALUABLE)
  - `C1-R|PDI`: predicted None, observed None; no own-lesion replication runs (NOT_EVALUABLE)
  - `C1-R|RAM`: predicted None, observed None; no own-lesion replication runs (NOT_EVALUABLE)
  - `C1-R|SRPI`: predicted None, observed None; no own-lesion replication runs (NOT_EVALUABLE)
  - 15 further cells (15 SUPPORTED)
- **HCv2-6(b)** (decisive; rule `anchor_replicates`): NOT_EVALUABLE - no input rows

### HCv2-7: NAS identification under a complete declaration (families A and C1)

Development expectation (hypotheses file): Block means, v1 rule: A (i) 20/20, median 1.01; (ii) 20/20, 0.90; C (i) 20/20, 0.95; (ii) 20/20; relay c 0.45-0.58.

Dry-run outcome of the hypothesis: **FALSIFIED**.

- **HCv2-7(i)** (decisive; rule `three_zone`): SUPPORTED - three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8
  - `family=A`: 12/12; upper 1 (SUPPORTED)
  - `family=C1`: 12/12; upper 1 (SUPPORTED)
- **HCv2-7(ii)** (decisive; rule `three_zone`): SUPPORTED - three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8
  - `family=A`: 12/12; upper 1 (SUPPORTED)
  - `family=C1`: 12/12; upper 1 (SUPPORTED)
- **HCv2-7(iii)** (decisive; rule `three_zone`): FALSIFIED - three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8
  - `family=A`: 11/12; upper 0.998 (SUPPORTED)
  - `family=C1`: 0/12; upper 0.265 (FALSIFIED)
- **HCv2-7(iv)** (decisive; rule `spearman_ols`): SUPPORTED - Spearman one-sided p < 0.05 with a positive OLS slope
  - `family=A`: rho 0.955, p 6.99e-22, n 40; slope 1.742 (SUPPORTED)
  - `family=C1`: rho 0.992, p 1.34e-35, n 40; slope 2.911 (SUPPORTED)
- **HCv2-7(v-A)** (decisive; rule `reversed_three_zone`): FALSIFIED - reversed three-zone at 0.8: SUPPORTED iff CP upper < 0.8, FALSIFIED iff rate >= 0.8
  - Expectation (hypotheses file): Development (A-R, CD-7): |Delta c_NAS(W_PDI_single_attractor - PC)| < z in 44/52 = 0.846 of seeds, above 0.8, so the reversed three-zone rule gives FALSIFIED: the v1 failure is not seen with the v2 NAS estimator under a complete declaration (under H the share is 28/52). The prediction is unchanged.
  - `family=A`: 12/12; upper 1 (FALSIFIED)
- **HCv2-7(v-C1)** (decisive; rule `three_zone`): SUPPORTED - three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8
  - `C1:W_PDI_single_attractor`: 11/12; upper 0.999 (SUPPORTED)
  - `C1:W_RAM_no_plasticity`: 12/12; upper 1 (SUPPORTED)
  - `C1:W_SRPI_no_efference`: 12/12; upper 1 (SUPPORTED)
- **HCv2-7(vi)** (reported; rule `describe`): REPORTED - reported
  - `family=A|system=PC_nominal`: 0/24 (REPORTED)
  - `family=A|system=W_NAS_broadcast_only`: 1/12 (REPORTED)
  - `family=A|system=W_NAS_common_input_control`: 12/12 (REPORTED)
  - `family=A|system=W_NAS_no_workspace`: 12/12 (REPORTED)
  - `family=C1|system=PC_nominal`: 0/12 (REPORTED)
  - `family=C1|system=W_NAS_broadcast_only`: 4/12 (REPORTED)
  - `family=C1|system=W_NAS_common_input_control`: 3/12 (REPORTED)
  - `family=C1|system=W_NAS_no_workspace`: 11/12 (REPORTED)

### HCv2-8: NAS non-identification under hidden inputs, restored by declaration

Development expectation (hypotheses file): (a) H 18/20, R 0/20; (b) H 20/20 each (rank 1 20/20), R 0/20; (c) secondary H 20/20, R 0/20; primary H c 0.09, PRESENT 3/20; (d) 20/20.

Dry-run outcome of the hypothesis: **INDETERMINATE**.

- **HCv2-8(a-H)** (decisive; rule `three_zone`): INDETERMINATE - three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8
  - `all`: 9/12; upper 0.928 (INDETERMINATE)
- **HCv2-8(a-R)** (decisive; rule `count_at_most`): SUPPORTED - SUPPORTED iff rate <= 0.1; FALSIFIED iff CP lower > 0.1
  - `all`: 0/12; lower 0 (SUPPORTED)
- **HCv2-8(b-H)** (decisive; rule `three_zone`): SUPPORTED - three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8
  - `variant=hierarchical`: 12/12; upper 1 (SUPPORTED)
  - `variant=reversed`: 12/12; upper 1 (SUPPORTED)
  - `variant=uniform`: 12/12; upper 1 (SUPPORTED)
- **HCv2-8(b-H-rank1)** (decisive; rule `three_zone`): INDETERMINATE - three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8
  - `variant=hierarchical`: 10/12; upper 0.983 (SUPPORTED)
  - `variant=reversed`: 8/12; upper 0.912 (INDETERMINATE)
  - `variant=uniform`: 10/12; upper 0.983 (SUPPORTED)
- **HCv2-8(b-R)** (decisive; rule `count_at_most`): SUPPORTED - SUPPORTED iff rate <= 0.1; FALSIFIED iff CP lower > 0.1
  - `variant=hierarchical`: 0/12; lower 0 (SUPPORTED)
  - `variant=reversed`: 0/12; lower 0 (SUPPORTED)
  - `variant=uniform`: 0/12; lower 0 (SUPPORTED)
- **HCv2-8(c-H-secondary)** (decisive; rule `three_zone`): SUPPORTED - three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8
  - `all`: 12/12; upper 1 (SUPPORTED)
- **HCv2-8(c-R-secondary)** (decisive; rule `count_at_most`): SUPPORTED - SUPPORTED iff rate <= 0.1; FALSIFIED iff CP lower > 0.1
  - `all`: 0/12; lower 0 (SUPPORTED)
- **HCv2-8(c-H-primary)** (decisive; rule `three_zone`): SUPPORTED - three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8
  - `all`: 11/12; upper 0.996 (SUPPORTED)
- **HCv2-8(d)** (decisive; rule `three_zone`): SUPPORTED - three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8
  - `system=N_modules_disconnected`: 12/12; upper 1 (SUPPORTED)
  - `system=W_NAS_no_workspace`: 12/12; upper 1 (SUPPORTED)

### HCv2-9: Declared-input misspecification, NAS and IIM (held out)

Development expectation (hypotheses file): none stated at hypothesis level; the dry run below is the development expectation.

Dry-run outcome of the hypothesis: **NOT_EVALUABLE**.

- **HCv2-9(a)** (decisive; rule `jonckheere_terpstra`): NOT_EVALUABLE - Jonckheere-Terpstra one-sided p < 0.0125 and medians last_gt_first
  - `system=N_modules_disconnected|principle=NAS`: n R 12, Q10 0, Q25 0; a group has no finite values (NOT_EVALUABLE)
  - `system=O_inert|principle=IIM`: n R 12, Q10 0, Q25 0; a group has no finite values (NOT_EVALUABLE)
  - `system=W_IIM_feedforward|principle=IIM`: n R 12, Q10 0, Q25 0; a group has no finite values (NOT_EVALUABLE)
  - `system=W_NAS_no_workspace|principle=NAS`: n R 12, Q10 0, Q25 0; a group has no finite values (NOT_EVALUABLE)
- **HCv2-9(b)** (decisive; rule `wilcoxon`): NOT_EVALUABLE - no input rows
- **HCv2-9(c)** (decisive; rule `sign_test`): NOT_EVALUABLE - no input rows
- **HCv2-9(d)** (decisive; rule `three_zone`): NOT_EVALUABLE - no input rows
  - Expectation (hypotheses file): P(NOT_EVALUABLE, reason ORACLE) about 0.74: the realisation check driver_reaches_every_module of ADV_NAS_staggered_tau10 passed 34/40 in development (CD-13, HCv2-0). The construct prediction (NAS primary significant under R in >= 50 % of seeds) is unchanged.
- **HCv2-9(e)** (decisive; rule `sign_test`): NOT_EVALUABLE - no input rows
- **HCv2-9(f)** (reported; rule `describe`): REPORTED - reported
  - `principle=IIM|declaration_id=R`: 20/24 (REPORTED)
  - `principle=NAS|declaration_id=R`: 24/24 (REPORTED)
- **HCv2-9(f-conditioning)** (reported; rule `describe`): REPORTED - reported
  - 42 further cells (42 REPORTED, 0 events in each)

### HCv2-10: NAS forward-model admission (Hopf arm, held-out regime)

Development expectation (hypotheses file): none stated at hypothesis level; the dry run below is the development expectation.

Dry-run outcome of the hypothesis: **NOT_EVALUABLE**.

- **HCv2-10** (decisive; rule `admission_matches`): NOT_EVALUABLE - no input rows
- **HCv2-10(absent)** (reported; rule `admission_matches`): NOT_EVALUABLE - no input rows

### HCv2-11: IIM directional construct on family B (H-IIM-4)

Development expectation (hypotheses file): T = 10000, 10 seeds, v1 rule: IIM-dir ABSENT 10/10, PRESENT 0/10 at 0.2, 0.6, 1.0; IIM-bid c 0.04/0.40/1.09 vs exact 0.04/0.41/1.12.

Dry-run outcome of the hypothesis: **SUPPORTED**.

- **HCv2-11(a)** (decisive; rule `h0_cell`): SUPPORTED - every pooled CP upper bound at 0.05/1 is below 0.07
  - pooled: IIM 0/60 (60 clusters), upper bound 0.0487; m = 12; P = 1
  - `c0.2:T10000`: 0/5; lower 0 (SUPPORTED)
  - `c0.2:T30000`: 0/5; lower 0 (SUPPORTED)
  - `c0.4:T10000`: 0/5; lower 0 (SUPPORTED)
  - `c0.4:T30000`: 0/5; lower 0 (SUPPORTED)
  - `c0.6:T10000`: 0/5; lower 0 (SUPPORTED)
  - `c0.6:T30000`: 0/5; lower 0 (SUPPORTED)
  - `c0.8:T10000`: 0/5; lower 0 (SUPPORTED)
  - `c0.8:T30000`: 0/5; lower 0 (SUPPORTED)
  - `c0:T10000`: 0/5; lower 0 (SUPPORTED)
  - `c0:T30000`: 0/5; lower 0 (SUPPORTED)
  - `c1:T10000`: 0/5; lower 0 (SUPPORTED)
  - `c1:T30000`: 0/5; lower 0 (SUPPORTED)
- **HCv2-11(b-rho)** (decisive; rule `spearman_ols`): SUPPORTED - Spearman one-sided p < 0.025
  - `T10000`: rho 0.987, p 7.07e-24, n 30; slope 1.149 (SUPPORTED)
  - `T30000`: rho 0.987, p 7.07e-24, n 30; slope 1.163 (SUPPORTED)
- **HCv2-11(b-present)** (decisive; rule `three_zone`): SUPPORTED - three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8
  - `all`: 5/5; upper 1 (SUPPORTED)
- **HCv2-11(c)** (decisive; rule `three_zone`): SUPPORTED - three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8
  - `c0.2:T10000`: 5/5; upper 1 (SUPPORTED)
  - `c0.2:T30000`: 5/5; upper 1 (SUPPORTED)
  - `c0.4:T10000`: 5/5; upper 1 (SUPPORTED)
  - `c0.4:T30000`: 5/5; upper 1 (SUPPORTED)
  - `c0.6:T10000`: 5/5; upper 1 (SUPPORTED)
  - `c0.6:T30000`: 5/5; upper 1 (SUPPORTED)
  - `c0.8:T10000`: 5/5; upper 1 (SUPPORTED)
  - `c0.8:T30000`: 5/5; upper 1 (SUPPORTED)
  - `c1:T10000`: 5/5; upper 1 (SUPPORTED)
  - `c1:T30000`: 5/5; upper 1 (SUPPORTED)

### HCv2-12: IIM family-B ground truth: exact tracking, the non-monotone regime and the occupancy gate

Development expectation (hypotheses file): none stated at hypothesis level; the dry run below is the development expectation.

Dry-run outcome of the hypothesis: **INDETERMINATE**.

- **HCv2-12(a)** (decisive; rule `spearman_monotone`): SUPPORTED - Spearman(sampled, exact) one-sided p < 0.0125 and level medians non-decreasing in the exact value (tolerance 0.1)
  - `ring:directional`: rho 0.987, p 7.07e-24, n 30; monotone medians true (SUPPORTED)
  - `ring:bidirectional`: rho 0.987, p 7.07e-24, n 30; monotone medians true (SUPPORTED)
  - `xor:directional`: rho 0.987, p 7.07e-24, n 30; monotone medians true (SUPPORTED)
  - `xor:bidirectional`: rho 0.987, p 7.07e-24, n 30; monotone medians true (SUPPORTED)
- **HCv2-12(a-T10000)** (reported; rule `describe`): REPORTED - reported
  - `b_network=ring|estimator_form=directional`: 0/0 (REPORTED)
  - `b_network=ring|estimator_form=bidirectional`: 0/0 (REPORTED)
  - `b_network=xor_loop|estimator_form=directional`: 0/0 (REPORTED)
  - `b_network=xor_loop|estimator_form=bidirectional`: 0/0 (REPORTED)
- **HCv2-12(b)** (decisive; rule `stepwise_sign`): NOT_EVALUABLE - each step: correct sign (sign test) or correctly undefined
  - Expectation (hypotheses file): The exact change (CD-1 constants) is negative from 0.45 to 0.9 in both cut modes (bidirectional 0.04513 -> 0.03621; directional 0.02909 -> -0.00973) and, from 0.9 to 1.5, positive for the bidirectional form (-> 0.04014) and negative for the directional form (-> -0.03161). The rule tests the sign of the exact change at each step; the earlier prose 'never a rising estimate' contradicted the exact bidirectional rise and was a logical error. The rule and its thresholds are unchanged. The operating characteristic is synthetic, from the exact steps and the SE of the development ring cells at couplings <= 0.5; no estimator ran at ring 0.9 or 1.5 (HO-5).
  - `estimator_form=directional`: 0.45->0.9: NOT_EVALUABLE; 0.9->1.5: NOT_EVALUABLE (NOT_EVALUABLE)
  - `estimator_form=bidirectional`: 0.45->0.9: NOT_EVALUABLE; 0.9->1.5: NOT_EVALUABLE (NOT_EVALUABLE)
- **HCv2-12(c-low)** (decisive; rule `three_zone`): SUPPORTED - three-zone at 0.9: SUPPORTED iff rate >= 0.9, FALSIFIED iff CP upper < 0.9
  - `low:all_to_all:c0.6:T1000:directional`: 5/5; upper 1 (SUPPORTED)
  - `low:all_to_all:c0.6:T1000:bidirectional`: 5/5; upper 1 (SUPPORTED)
  - `low:all_to_all:c0.4:T1000:directional`: 5/5; upper 1 (SUPPORTED)
  - `low:all_to_all:c0.4:T1000:bidirectional`: 5/5; upper 1 (SUPPORTED)
  - `low:all_to_all:c0.6:T10000:directional`: 5/5; upper 1 (SUPPORTED)
  - `low:all_to_all:c0.6:T10000:bidirectional`: 5/5; upper 1 (SUPPORTED)
  - `low:all_to_all:c0.6:T3000:directional`: 5/5; upper 1 (SUPPORTED)
  - `low:all_to_all:c0.6:T3000:bidirectional`: 5/5; upper 1 (SUPPORTED)
  - `low:ring:c0.45:T1000:directional`: 5/5; upper 1 (SUPPORTED)
  - `low:ring:c0.45:T1000:bidirectional`: 5/5; upper 1 (SUPPORTED)
  - `low:xor_loop:cNone:T1000:directional`: 5/5; upper 1 (SUPPORTED)
  - `low:xor_loop:cNone:T1000:bidirectional`: 5/5; upper 1 (SUPPORTED)
- **HCv2-12(c-high)** (decisive; rule `three_zone`): SUPPORTED - three-zone at 0.9: SUPPORTED iff rate >= 0.9, FALSIFIED iff CP upper < 0.9
  - `high:all_to_all:c0.4:T30000:directional`: 5/5; upper 1 (SUPPORTED)
  - `high:all_to_all:c0.4:T30000:bidirectional`: 5/5; upper 1 (SUPPORTED)
  - `high:ring:c0.45:T10000:directional`: 5/5; upper 1 (SUPPORTED)
  - `high:ring:c0.45:T10000:bidirectional`: 5/5; upper 1 (SUPPORTED)
  - `high:ring:c0.45:T30000:directional`: 5/5; upper 1 (SUPPORTED)
  - `high:ring:c0.45:T30000:bidirectional`: 5/5; upper 1 (SUPPORTED)
  - `high:xor_loop:cNone:T30000:directional`: 5/5; upper 1 (SUPPORTED)
  - `high:xor_loop:cNone:T30000:bidirectional`: 5/5; upper 1 (SUPPORTED)
- **HCv2-12(d)** (decisive; rule `three_zone`): INDETERMINATE - three-zone at 0.9: SUPPORTED iff rate >= 0.9, FALSIFIED iff CP upper < 0.9
  - `O_hypersynchronous`: 9/12; upper 0.953 (INDETERMINATE)
  - `all_to_all_0.6_T1000`: 10/10; 5 clusters; upper 1 (SUPPORTED)
  - `v1_quadrant_average_reference`: 20/20; upper 1 (SUPPORTED)

### HCv2-13: IIM driver conditioning on family B (H-IIM-5 a-c)

Development expectation (hypotheses file): Recorded c -0.03/-0.01/-0.00, exceedance <= 0.05; undeclared c 0.69/0.91/1.10; label errors c 0 -> 0.27 -> 0.67.

Dry-run outcome of the hypothesis: **INDETERMINATE**.

- **HCv2-13(a)** (decisive; rule `h0_cell`): INDETERMINATE - a pooled upper bound is not below the bound
  - pooled: exceedance 0/20 (10 clusters), upper bound 0.308; present 0/30 (15 clusters), upper bound 0.218; m = 10; P = 2
  - `exceedance:T10000:directional`: 0/5; lower 0 (SUPPORTED)
  - `exceedance:T10000:bidirectional`: 0/5; lower 0 (SUPPORTED)
  - `exceedance:T30000:directional`: 0/5; lower 0 (SUPPORTED)
  - `exceedance:T30000:bidirectional`: 0/5; lower 0 (SUPPORTED)
  - `present:T10000:directional`: 0/5; lower 0 (SUPPORTED)
  - `present:T10000:bidirectional`: 0/5; lower 0 (SUPPORTED)
  - `present:T30000:directional`: 0/5; lower 0 (SUPPORTED)
  - `present:T30000:bidirectional`: 0/5; lower 0 (SUPPORTED)
  - `present:T3000:directional`: 0/5; lower 0 (SUPPORTED)
  - `present:T3000:bidirectional`: 0/5; lower 0 (SUPPORTED)
- **HCv2-13(b-present)** (decisive; rule `three_zone`): SUPPORTED - three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8
  - `estimator_form=directional`: 5/5; upper 1 (SUPPORTED)
  - `estimator_form=bidirectional`: 5/5; upper 1 (SUPPORTED)
- **HCv2-13(b-median)** (decisive; rule `median_threshold`): SUPPORTED - median gt 0.25
  - `T10000:directional`: median 0.758, n 5 (SUPPORTED)
  - `T10000:bidirectional`: median 0.959, n 5 (SUPPORTED)
  - `T30000:directional`: median 0.806, n 5 (SUPPORTED)
  - `T30000:bidirectional`: median 1.055, n 5 (SUPPORTED)
  - `T3000:directional`: median -, n 0; too few finite values (NOT_EVALUABLE)
  - `T3000:bidirectional`: median -, n 0; too few finite values (NOT_EVALUABLE)
- **HCv2-13(c)** (decisive; rule `jonckheere_terpstra`): SUPPORTED - Jonckheere-Terpstra one-sided p < 0.025 and medians strictly_increasing
  - `estimator_form=directional`: JT p 1.32e-06; medians 0 -0.00489, 0.1 0.243, 0.25 0.554; n 5, 5, 5 (SUPPORTED)
  - `estimator_form=bidirectional`: JT p 1.32e-06; medians 0 -0.00731, 0.1 0.241, 0.25 0.669; n 5, 5, 5 (SUPPORTED)

### HCv2-14: IIM-dir on the agents with recorded drivers, and the hidden-driver limit

Development expectation (hypotheses file): Development (CD-6; family A, reference block 900-939 and witness seeds 320-331 and 340-351; block bootstrap SE at the decided se_df 9, df 12 in brackets). Under R: median paired Delta c(PC - W_IIM_feedforward) 0.90 over 52 pairs, 49 >= 0.5, Hodges-Lehmann one-sided 95 % lower bound 0.91; PC_nominal PRESENT 58/64 = 0.906 (60/64), so (d) is decisive by one row; W_IIM_feedforward PRESENT 0/52 and ABSENT 12/52 (13/52), so (c) stays NOT_TESTABLE_BY_DESIGN (pi0 0.23); O_inert PRESENT 0/12; Spearman rho of c with c_int 0.09 (40 runs, one-sided p 0.28; an inverted U, median c 0.97 at c_int 0.53 and 0.06 at 1.2), so (f) would be expected FALSIFIED at 10 seeds per level; on the resized c_int sweep (65 seeds per level, CD-11) P(SUPPORTED | development) is 0.82, see (f). Under H: median Delta c -0.61 over 47 pairs, consistent with (e); PRESENT PC_nominal 24/64, W_IIM_feedforward 29/52, O_inert 11/12 (HCv2-14(H-present)).

Dry-run outcome of the hypothesis: **FALSIFIED**.

- **HCv2-14(a)** (decisive; rule `hodges_lehmann`): SUPPORTED - median ge 0.5 and one-sided 0.95 Hodges-Lehmann lower bound > 0.25
  - `family=A`: median 0.801, HL 0.832, lower 0.764, n 12 (SUPPORTED)
- **HCv2-14(b)** (decisive; rule `h0_cell`): INDETERMINATE - a pooled upper bound is not below the bound
  - pooled: IIM 0/24 (24 clusters), upper bound 0.117; m = 4; P = 1
  - `A:O_inert`: 0/12; lower 0 (SUPPORTED)
  - `A:W_IIM_feedforward`: 0/12; lower 0 (SUPPORTED)
- **HCv2-14(c)** (decisive; rule `three_zone`): NOT_TESTABLE_BY_DESIGN - every cell is below the gate (pi < 0.9)
  - `A-R`:  (NOT_TESTABLE_BY_DESIGN)
- **HCv2-14(d)** (decisive; rule `three_zone`): SUPPORTED - three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8
  - `A-R`: 20/24; upper 0.941 (SUPPORTED)
- **HCv2-14(e)** (decisive; rule `median_threshold`): SUPPORTED - median lt 0.25
  - `family=A`: median -0.322, n 11 (SUPPORTED)
- **HCv2-14(f)** (decisive; rule `spearman_ols`): FALSIFIED - Spearman one-sided p < 0.05
  - `family=A`: rho 0.0935, p 0.283, n 40; slope 0.0607 (FALSIFIED)
- **HCv2-14(H-present)** (reported; rule `describe`): REPORTED - reported
  - `A:O_inert`: 11/12 (REPORTED)
  - `A:W_IIM_feedforward`: 9/12 (REPORTED)

### HCv2-15: IIM at sensor and BOLD level: specificity, exclusion safety, sensitivity (Hopf arm, held-out regime)

Development expectation (hypotheses file): none stated at hypothesis level; the dry run below is the development expectation.

Dry-run outcome of the hypothesis: **FALSIFIED**.

- **HCv2-15(a)** (decisive; rule `demonstration`): FALSIFIED - CP upper bound at 0.0125 below 0.07
  - `eeg64:present`: 0/20; upper 0.197 (FALSIFIED)
  - `eeg64_noref:present`: 0/20; upper 0.197 (FALSIFIED)
  - `eeglow:present`: 0/20; upper 0.197 (FALSIFIED)
  - `mne_template:present`: 0/20; upper 0.197 (FALSIFIED)
  - `eeg64:exceedance`: 2/20; upper 0.349 (FALSIFIED)
  - `eeg64_noref:exceedance`: 0/20; upper 0.197 (FALSIFIED)
  - `eeglow:exceedance`: 2/20; upper 0.349 (FALSIFIED)
  - `mne_template:exceedance`: 2/20; upper 0.349 (FALSIFIED)
- **HCv2-15(b)** (decisive; rule `three_zone`): FALSIFIED - three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8
  - `all`: 2/20; upper 0.283 (FALSIFIED)
- **HCv2-15(c)** (decisive; rule `demonstration`): NOT_EVALUABLE - no input rows
- **HCv2-15(c-pi0)** (reported; rule `describe`): REPORTED - reported
  - `view=eeg64`: 0/20 (REPORTED)
  - `view=eeg64_noref`: 0/20 (REPORTED)
  - `view=eeglow`: 0/20 (REPORTED)
  - `view=mne_template`: 0/20 (REPORTED)
- **HCv2-15(d-present)** (decisive; rule `h0_cell`): SUPPORTED - every pooled CP upper bound at 0.05/1 is below 0.07
  - pooled: IIM 0/88 (88 clusters), upper bound 0.0335; m = 8; P = 1
  - `G0`: 0/20; lower 0 (SUPPORTED)
  - `G0.571429`: 0/6; lower 0 (SUPPORTED)
  - `G1.142857`: 0/32; lower 0 (SUPPORTED)
  - `G1.714286`: 0/6; lower 0 (SUPPORTED)
  - `G2.285714`: 0/6; lower 0 (SUPPORTED)
  - `G2.857143`: 0/6; lower 0 (SUPPORTED)
  - `G3.428571`: 0/6; lower 0 (SUPPORTED)
  - `G4`: 0/6; lower 0 (SUPPORTED)
- **HCv2-15(d-absent)** (decisive; rule `demonstration`): INDETERMINATE - CP upper bound at 0.05 below 0.02
  - `all`: 0/32; upper 0.0894 (INDETERMINATE)
- **HCv2-15(e-FMb1)** (reported; rule `spearman_ols`): NOT_EVALUABLE - Spearman one-sided p < 0.05
  - `view=eeg64`: no spread in dose or value (NOT_EVALUABLE)
  - `view=eeg64_noref`: no spread in dose or value (NOT_EVALUABLE)
  - `view=eeglow`: no spread in dose or value (NOT_EVALUABLE)
  - `view=mne_template`: no spread in dose or value (NOT_EVALUABLE)
- **HCv2-15(e-FMd)** (decisive; rule `fmd_concordance`): NOT_EVALUABLE - FMd: forward contrast has the source's sign and >= 0.5 of its size; reversed three-zone at 0.8: SUPPORTED iff CP upper < 0.8, FALSIFIED iff rate >= 0.8
  - `view=eeg64`: 0/0; fewer than 1 clusters (NOT_EVALUABLE)
  - `view=eeg64_noref`: 0/0; fewer than 1 clusters (NOT_EVALUABLE)
  - `view=eeglow`: 0/0; fewer than 1 clusters (NOT_EVALUABLE)
  - `view=mne_template`: 0/0; fewer than 1 clusters (NOT_EVALUABLE)

### HCv2-16: RAM-PE specificity: no RAM-PE without plasticity (H-RAM-1)

Development expectation (hypotheses file): 0/40 at eta = 0 (development, v2 readout); reflex arc: dry run.

Dry-run outcome of the hypothesis: **SUPPORTED**.

- **HCv2-16** (decisive; rule `h0_cell`): SUPPORTED - every pooled CP upper bound at 0.05/1 is below 0.07
  - pooled: RAM 0/124 (124 clusters), upper bound 0.0239; m = 19; P = 1
  - 19 further cells (19 SUPPORTED, 0 events in each)
- **HCv2-16(reported)** (reported; rule `describe`): REPORTED - reported
  - 75 further cells (75 REPORTED, 0 events in each)

### HCv2-17: RAM-PE sensitivity and its predicted non-selectivity (H-RAM-2 revised)

Development expectation (hypotheses file): Paired Delta c mean 1.02 (>= 0.5 in 34/40); excess -0.02/0.13/0.39/0.52 at eta 0/0.1/0.3/0.6.

Dry-run outcome of the hypothesis: **FALSIFIED**.

- **HCv2-17(a)** (decisive; rule `spearman_ols`): SUPPORTED - Spearman one-sided p < 0.05
  - `all`: rho 0.934, p 5.48e-46, n 100; slope 1.094 (SUPPORTED)
- **HCv2-17(b)** (decisive; rule `three_zone`): SUPPORTED - three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8
  - `all`: 19/20; upper 0.997 (SUPPORTED)
- **HCv2-17(c)** (decisive; rule `rate_lower_bound`): SUPPORTED - SUPPORTED iff CP lower > 0.5; FALSIFIED iff CP upper < 0.5
  - `all`: 19/20; lower 0.784; upper 0.997 (SUPPORTED)
- **HCv2-17(d)** (decisive; rule `hodges_lehmann`): FALSIFIED - median gt 0.25 and one-sided 0.95 Hodges-Lehmann lower bound > 0.0
  - `all`: median 0.131, HL 0.143, lower 0.106, n 20 (FALSIFIED)
- **HCv2-17(d-g_b)** (reported; rule `hodges_lehmann`): SUPPORTED - one-sided 0.95 Hodges-Lehmann lower bound > 0.0
  - `all`: median 0.175, HL 0.175, lower 0.129, n 20 (SUPPORTED)
- **HCv2-17(e)** (decisive; rule `newcombe_includes_zero`): SUPPORTED - Newcombe 95% interval of the rate difference includes 0
  - `all`: 20/20 vs 19/20, difference 0.05 [-0.116, 0.236] (SUPPORTED)

### HCv2-18: PDI: no states without contents (H-PDI-1)

Development expectation (hypotheses file): No-content systems B <= 0.25 bits (c <= 0.1); surrogates one state 280/280.

Dry-run outcome of the hypothesis: **SUPPORTED**.

- **HCv2-18** (decisive; rule `h0_cell`): SUPPORTED - every pooled CP upper bound at 0.05/1 is below 0.07
  - pooled: PDI 0/376 (252 clusters), upper bound 0.0118; m = 37; P = 1
  - 37 further cells (37 SUPPORTED, 0 events in each)
- **HCv2-18(reported)** (reported; rule `describe`): REPORTED - reported
  - `system=N_ar1`: 0/0 (REPORTED)
  - `system=N_independent_noise`: 0/0 (REPORTED)
  - `system=O_hypersynchronous`: 0/0 (REPORTED)
  - `system=W_PDI_no_multistability`: 0/0 (REPORTED)
  - `system=W_PDI_single_attractor`: 0/0 (REPORTED)

### HCv2-19: PDI exclusion validity in the v1 masking window, and repertoire dose-response (H-PDI-2 + H-PDI-3)

Development expectation (hypotheses file): Content bearer B 2.38-2.47 bits at g_b 0-2; B 0.90/1.51/2.45/3.50 at K 2/3/6/12.

Dry-run outcome of the hypothesis: **SUPPORTED**.

- **HCv2-19(a)** (decisive; rule `h0_cell`): SUPPORTED - every pooled CP upper bound at 0.05/1 is below 0.07
  - pooled: PDI 0/304 (152 clusters), upper bound 0.0195; m = 33; P = 1
  - 33 further cells (33 SUPPORTED, 0 events in each)
- **HCv2-19(b)** (decisive; rule `three_zone`): SUPPORTED - three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8
  - `witness:PC_nominal`: 24/24; upper 1 (SUPPORTED)
  - `sweep:g_b:0.666667`: 4/4; upper 1 (SUPPORTED)
  - `sweep:g_b:0.888889`: 4/4; upper 1 (SUPPORTED)
  - `sweep:g_b:1.111111`: 4/4; upper 1 (SUPPORTED)
  - `sweep:g_b:1.333333`: 4/4; upper 1 (SUPPORTED)
  - `sweep:g_b:1.555556`: 4/4; upper 1 (SUPPORTED)
- **HCv2-19(c)** (decisive; rule `spearman_ols`): SUPPORTED - Spearman one-sided p < 0.05
  - `all`: rho 0.989, p 5.65e-20, n 24; slope 2.512 (SUPPORTED)
- **HCv2-19(d)** (decisive; rule `three_zone`): SUPPORTED - three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8
  - `K6`: 4/4; upper 1 (SUPPORTED)
  - `K12`: 4/4; upper 1 (SUPPORTED)
- **HCv2-19(e)** (decisive; rule `three_zone`): SUPPORTED - three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8
  - `all`: 12/12; upper 1 (SUPPORTED)

### HCv2-20: PDI construct: ignition, access declaration and the concordance route (H-PDI-4, H-PDI-5 revised)

Development expectation (hypotheses file): Content bearer one state 40/40; full bearer two states in 10/20 with AMI 0.79; concordant one state 10/10 (W_PDI_single_attractor), 8/10 (W_PDI_no_multistability), PC 0/10.

Dry-run outcome of the hypothesis: **INDETERMINATE**.

- **HCv2-20(a)** (decisive; rule `three_zone`): SUPPORTED - three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8
  - Expectation (hypotheses file): Development (CD-5): under each of A-R and A-H, 41 of 52 W_PDI_single_attractor runs (0.79) are concordant ABSENTs, just below 0.80; reported, not acted on.
  - Branch: ABSENT in >= 80 % (route admitted)
  - `all`: 11/12; upper 0.996 (SUPPORTED)
- **HCv2-20(b)** (decisive; rule `three_zone`): NOT_EVALUABLE - three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8
  - `all`: 7/8; fewer than 10 clusters (NOT_EVALUABLE)
- **HCv2-20(c)** (decisive; rule `three_zone`): SUPPORTED - three-zone at 0.95: SUPPORTED iff rate >= 0.95, FALSIFIED iff CP upper < 0.95
  - `all`: 12/12; upper 1 (SUPPORTED)
- **HCv2-20(d)** (decisive; rule `three_zone`): NOT_EVALUABLE - no input rows
- **HCv2-20(e)** (decisive; rule `h0_cell`): INDETERMINATE - a pooled upper bound is not below the bound
  - pooled: PDI 0/304 (152 clusters), upper bound 0.0195; m = 1; P = 1
  - `content_on`: 0/304; 152 clusters; lower 0 (SUPPORTED)
- **HCv2-20(f-single)** (decisive; rule `three_zone`): SUPPORTED - three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8
  - `all`: 11/12; upper 0.996 (SUPPORTED)
- **HCv2-20(f-no-multistability)** (decisive; rule `three_zone`): SUPPORTED - three-zone at 0.7: SUPPORTED iff rate >= 0.7, FALSIFIED iff CP upper < 0.7
  - `all`: 10/12; upper 0.97 (SUPPORTED)
- **HCv2-20(f-PC)** (decisive; rule `three_zone_at_most`): SUPPORTED - three-zone at most 0.2: SUPPORTED iff rate <= 0.2, FALSIFIED iff CP lower > 0.2
  - `all`: 0/24; lower 0 (SUPPORTED)

### HCv2-21: PDI forward-model admission (forward-modelled family A, held-out regime)

Development expectation (hypotheses file): none stated at hypothesis level; the dry run below is the development expectation.

Dry-run outcome of the hypothesis: **NOT_EVALUABLE**.

- **HCv2-21** (decisive; rule `admission_matches`): NOT_EVALUABLE - no input rows

### HCv2-22: Single-deficit lesion witnesses, rule side (HC4v2-R, HC8v2-R)

Development expectation (hypotheses file): A-R: W_NAS_no_workspace (ii) decisive if pi0 >= 0.9 and predicted SUPPORTED; W_NAS_broadcast_only (ii) predicted FALSIFIED where testable; W_IIM_feedforward, W_SRPI_no_efference, W_RAM_no_plasticity (ii) NOT_TESTABLE_BY_DESIGN; W_PDI_single_attractor (ii) NOT_TESTABLE_BY_DESIGN with the admitted concordance route (development pi0 0.79, 41/52; CD-5).

Dry-run outcome of the hypothesis: **FALSIFIED**.

- **HCv2-22(i)** (decisive; rule `three_zone`): SUPPORTED - three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8
  - `A-H|W_NAS_broadcast_only`: 11/12; upper 1 (SUPPORTED)
  - `A-H|W_NAS_no_workspace`: 11/12; upper 1 (SUPPORTED)
  - `A-H|W_PDI_single_attractor`: 12/12; upper 1 (SUPPORTED)
  - `A-H|W_RAM_no_plasticity`: 12/12; upper 1 (SUPPORTED)
  - `A-H|W_SRPI_no_efference`: 11/12; upper 1 (SUPPORTED)
  - `A-R|W_IIM_feedforward`: 12/12; upper 1 (SUPPORTED)
  - `A-R|W_NAS_broadcast_only`: 12/12; upper 1 (SUPPORTED)
  - `A-R|W_NAS_no_workspace`: 12/12; upper 1 (SUPPORTED)
  - `A-R|W_PDI_single_attractor`: 12/12; upper 1 (SUPPORTED)
  - `A-R|W_RAM_no_plasticity`: 12/12; upper 1 (SUPPORTED)
  - `A-R|W_SRPI_no_efference`: 11/12; upper 1 (SUPPORTED)
  - `C1-H|W_NAS_broadcast_only`: 12/12; upper 1 (SUPPORTED)
  - `C1-H|W_NAS_no_workspace`: 12/12; upper 1 (SUPPORTED)
  - `C1-R|W_NAS_broadcast_only`: 12/12; upper 1 (SUPPORTED)
  - 1 further cells (1 SUPPORTED)
- **HCv2-22(ii)** (decisive; rule `three_zone`): SUPPORTED - three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8
  - `A-H|W_NAS_broadcast_only`:  (NOT_TESTABLE_BY_DESIGN)
  - `A-H|W_NAS_no_workspace`:  (NOT_TESTABLE_BY_DESIGN)
  - `A-H|W_PDI_single_attractor`:  (NOT_TESTABLE_BY_DESIGN)
  - `A-H|W_RAM_no_plasticity`:  (NOT_TESTABLE_BY_DESIGN)
  - `A-H|W_SRPI_no_efference`:  (NOT_TESTABLE_BY_DESIGN)
  - `A-R|W_IIM_feedforward`:  (NOT_TESTABLE_BY_DESIGN)
  - `A-R|W_NAS_broadcast_only`:  (NOT_TESTABLE_BY_DESIGN)
  - `A-R|W_PDI_single_attractor`:  (NOT_TESTABLE_BY_DESIGN)
  - `A-R|W_RAM_no_plasticity`:  (NOT_TESTABLE_BY_DESIGN)
  - `A-R|W_SRPI_no_efference`:  (NOT_TESTABLE_BY_DESIGN)
  - `C1-H|W_NAS_broadcast_only`:  (NOT_TESTABLE_BY_DESIGN)
  - `C1-R|W_NAS_broadcast_only`:  (NOT_TESTABLE_BY_DESIGN)
  - `A-R|W_NAS_no_workspace`: 12/12; upper 1 (SUPPORTED)
  - `C1-H|W_NAS_no_workspace`: 10/12; upper 0.983 (SUPPORTED)
  - 1 further cells (1 SUPPORTED)
- **HCv2-22(iii)** (decisive; rule `three_zone`): FALSIFIED - three-zone at 0.8: SUPPORTED iff rate >= 0.8, FALSIFIED iff CP upper < 0.8
  - `A-H|W_PDI_single_attractor|RAM`: 3/12; upper 0.708 (FALSIFIED)
  - `A-H|W_PDI_single_attractor|SRPI`: 4/12; upper 0.775 (FALSIFIED)
  - `A-R|W_PDI_single_attractor|RAM`: 3/12; upper 0.708 (FALSIFIED)
  - `A-R|W_PDI_single_attractor|SRPI`: 4/12; upper 0.775 (FALSIFIED)
  - `A-H|W_NAS_broadcast_only|SRPI`: 9/12; upper 0.981 (INDETERMINATE)
  - `A-H|W_NAS_no_workspace|RAM`: 6/12; upper 0.881 (INDETERMINATE)
  - `A-H|W_NAS_no_workspace|SRPI`: 7/12; upper 0.923 (INDETERMINATE)
  - `A-H|W_PDI_single_attractor|NAS`: 6/12; upper 0.881 (INDETERMINATE)
  - `A-R|W_IIM_feedforward|RAM`: 6/12; upper 0.881 (INDETERMINATE)
  - `A-R|W_NAS_broadcast_only|IIM`: 8/12; upper 0.956 (INDETERMINATE)
  - `A-R|W_NAS_broadcast_only|SRPI`: 9/12; upper 0.981 (INDETERMINATE)
  - `A-R|W_NAS_no_workspace|RAM`: 6/12; upper 0.881 (INDETERMINATE)
  - `A-R|W_NAS_no_workspace|SRPI`: 7/12; upper 0.923 (INDETERMINATE)
  - `A-R|W_PDI_single_attractor|IIM`: 7/12; upper 0.923 (INDETERMINATE)
  - 23 further cells (23 SUPPORTED)

### HCv2-23: Verdict specificity (HC5v2)

Development expectation (hypotheses file): none stated at hypothesis level; the dry run below is the development expectation.

Dry-run outcome of the hypothesis: **INDETERMINATE**.

- **HCv2-23** (decisive; rule `any_event_clusters`): INDETERMINATE - the any-event upper bound is not below the bound
  - any-event clusters 1/24, upper bound 0.183; m = 17
  - `A-H|W_NAS_no_workspace`: 1/12; lower 0.000245 (SUPPORTED)
  - 16 further cells (16 SUPPORTED, 0 events in each)
- **HCv2-23(N_decl)** (reported; rule `describe`): REPORTED - reported
  - `A-H|W_IIM_feedforward`: 5/12 (REPORTED)
  - `A-H|W_NAS_no_workspace`: 1/12 (REPORTED)
  - 22 further cells (22 REPORTED, 0 events in each)

### HCv2-24: No PRESENT without mechanism in the factorial (HC6 successor)

Development expectation (hypotheses file): none stated at hypothesis level; the dry run below is the development expectation.

Dry-run outcome of the hypothesis: **INDETERMINATE**.

- **HCv2-24(a)** (decisive; rule `h0_cell`): SUPPORTED - every pooled CP upper bound at 0.05/5 is below 0.07
  - pooled: IIM 0/64 (64 clusters), upper bound 0.0694; NAS 0/128 (128 clusters), upper bound 0.0353; PDI 0/64 (64 clusters), upper bound 0.0694; RAM 0/64 (64 clusters), upper bound 0.0694; SRPI 0/64 (64 clusters), upper bound 0.0694; m = 96; P = 5
  - 96 further cells (96 SUPPORTED, 0 events in each)
- **HCv2-24(b)** (decisive; rule `h0_cell_exceeds`): INDETERMINATE - a pooled upper bound is not below the bound
  - pooled: NAS 16/32 (32 clusters), upper bound 0.656; m = 8; P = 1
  - `b01000`: 2/4; lower 0.033 (SUPPORTED)
  - `b01001`: 2/4; lower 0.033 (SUPPORTED)
  - `b01010`: 2/4; lower 0.033 (SUPPORTED)
  - `b01011`: 2/4; lower 0.033 (SUPPORTED)
  - `b11000`: 2/4; lower 0.033 (SUPPORTED)
  - `b11001`: 2/4; lower 0.033 (SUPPORTED)
  - `b11010`: 2/4; lower 0.033 (SUPPORTED)
  - `b11011`: 2/4; lower 0.033 (SUPPORTED)

### HCv2-B1: SRPI v3 in the SRPI-only arm

Tier B, not admitted in this round: not run (`Not admitted in this round (Tier A only); listed as not run, and the v1 outcome stands where a v1 predecessor exists.`).

### HCv2-B2: NAS hub identity (HN4 revised)

Tier B, not admitted in this round: not run (`Not admitted in this round (Tier A only); listed as not run, and the v1 outcome stands where a v1 predecessor exists.`).

### HCv2-B3: RAM-PE persistence gate and the echo family

Tier B, not admitted in this round: not run (`Not admitted in this round (Tier A only); listed as not run, and the v1 outcome stands where a v1 predecessor exists.`).

### HCv2-B4: Single source via joint dependence (HC7v2 part a)

Tier B, not admitted in this round: not run (`Not admitted in this round (Tier A only); listed as not run, and the v1 outcome stands where a v1 predecessor exists.`).

### HCv2-B5: Rule audit (HC9v2)

Tier B, not admitted in this round: not run (`Not admitted in this round (Tier A only); listed as not run, and the v1 outcome stands where a v1 predecessor exists.`).

### HCv2-B6: Zero-mean workspace broadcast (W_ctx_hub_only)

Tier B, not admitted in this round: not run (`Not admitted in this round (Tier A only); listed as not run, and the v1 outcome stands where a v1 predecessor exists.`).

### HCv2-B7: PDI slow-drift continuum

Tier B, not admitted in this round: not run (`Not admitted in this round (Tier A only); listed as not run, and the v1 outcome stands where a v1 predecessor exists.`).

### HCv2-B8: Forward arms for RAM-PE and SRPI

Tier B, not admitted in this round: not run (`Not admitted in this round (Tier A only); listed as not run, and the v1 outcome stands where a v1 predecessor exists.`).
