# MPC-Bench v2: testability table

DEVELOPMENT - NOT A RESULT. Companion of [`MPC_BENCH_PREREGISTRATION_V2.md`](../MPC_BENCH_PREREGISTRATION_V2.md), sections 3.7 and 7.3. Rendered from `protocols/v2/generated/testability_table.json` (schema `mpc-bench-testability-table/1`, SHA-256 `90d9c999d52b8ae5f9ff26dbddddece3f9fd00c1f1db5547984b064b8a886717`), which the protocol builder wrote from the development records (seeds 0-999) under the decided calibration; the same rows are in the hash-covered `precision` block of each family protocol. Nothing here is computed anew; numbers are rounded for reading, and the JSON file is authoritative.

## Rule

- A gated part (target ABSENT on an own-lesion or construct-null witness, or PRESENT in at least 80 % of positive-control runs) is decisive in a cell only if the empirical development rate `pi` of the gated status (`pi0` for ABSENT, `pi1` for PRESENT) is at least 0.9, on at least 40 development runs.
- Otherwise the cell is `NOT_TESTABLE_BY_DESIGN` before the freeze and the part's replacement claim decides (`replacement`).
- `pi` is the share of development runs with the gated status under the frozen v2 rule (final `alpha_A`, df and reasons); `pi_analytic` is the normal-approximation value reported beside it: the mean over the row's runs of `max(0, 2 Phi(delta / se_c - q_A) - 1)` for ABSENT (of `Phi((1 - z) / se_c - q_P)` for PRESENT), each run at its own `se_c` and df. `s_A = delta / q_A` is the largest `se_c` at which the equivalence test can pass at `c = 0`, `s_P = (1 - z) / q_P` the largest at which a component at the reference (`c = 1`) can be PRESENT; `attainable se_c` is the median development `se_c` of the row.
- The gate reads only precision on own-lesion and reference systems, never a confirmatory value.

## Rows

| Protocol | Kind | Principle | Witness | n | pi | pi (analytic) | Outcome | s_A | s_P | attainable se_c | Replacement | SE method | Sources |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `A-H` | absent | IIM | `W_IIM_feedforward` | 52 | 0 | 0 | NOT_TESTABLE_BY_DESIGN | 0.0404 | 0.44 | 0.636 | target not PRESENT in >= 80 % of seeds | circular_block_bootstrap_10pct_B50 | A_anchors, A_witnesses |
| `A-H` | absent | NAS | `W_NAS_broadcast_only` | 52 | 0 | 0.0977 | NOT_TESTABLE_BY_DESIGN | 0.0313 | 0.412 | 0.0394 | target not PRESENT in >= 80 % of seeds | jackknife_contiguous_10 | A_anchors, A_witnesses |
| `A-H` | absent | NAS | `W_NAS_no_workspace` | 52 | 0.154 | 0.443 | NOT_TESTABLE_BY_DESIGN | 0.0325 | 0.42 | 0.0285 | target not PRESENT in >= 80 % of seeds | jackknife_contiguous_10 | A_anchors, A_witnesses |
| `A-H` | absent | PDI | `W_PDI_no_multistability` | 12 | - | - | no row (12 development runs; the gate needs 40) | - | - | - | - | concordant, jackknife_contiguous_10 | A_witnesses |
| `A-H` | absent | PDI | `W_PDI_single_attractor` | 52 | 0.788 | 0.788 | NOT_TESTABLE_BY_DESIGN | 0.0354 | 0.409 | 0 | target not PRESENT in >= 80 % of seeds | concordant, jackknife_contiguous_10 | A_anchors, A_witnesses |
| `A-H` | absent | RAM | `W_RAM_no_plasticity` | 52 | 0 | 0 | NOT_TESTABLE_BY_DESIGN | 0.042 | 0.45 | 0.329 | target not PRESENT in >= 80 % of seeds | shift_null_sd | A_anchors, A_witnesses |
| `A-H` | absent | SRPI | `W_SRPI_no_efference` | 52 | 0 | 0 | NOT_TESTABLE_BY_DESIGN | 0.0355 | 0.409 | 0.458 | target not PRESENT in >= 80 % of seeds | jackknife_pairs_10 | A_anchors, A_witnesses |
| `A-H` | present | IIM | `PC_nominal` | 64 | 0.375 | 0.257 | NOT_TESTABLE_BY_DESIGN | 0.0384 | 0.428 | 0.688 | paired contrast with the positive control | circular_block_bootstrap_10pct_B50 | A_anchors, A_witnesses |
| `A-H` | present | NAS | `PC_nominal` | 64 | 1 | 1 | decisive | 0.0379 | 0.425 | 0.0872 | - | jackknife_contiguous_10 | A_anchors, A_witnesses |
| `A-H` | present | PDI | `PC_nominal` | 64 | 1 | 1 | decisive | 0.0355 | 0.409 | 0.0901 | - | jackknife_contiguous_10 | A_anchors, A_witnesses |
| `A-H` | present | RAM | `PC_nominal` | 64 | 0.672 | 0.632 | NOT_TESTABLE_BY_DESIGN | 0.0421 | 0.45 | 0.379 | paired contrast with the positive control | shift_null_sd | A_anchors, A_witnesses |
| `A-H` | present | SRPI | `PC_nominal` | 64 | 0.766 | 0.754 | NOT_TESTABLE_BY_DESIGN | 0.0359 | 0.412 | 0.284 | paired contrast with the positive control | jackknife_pairs_10 | A_anchors, A_witnesses |
| `A-R` | absent | IIM | `W_IIM_feedforward` | 52 | 0.231 | 0.281 | NOT_TESTABLE_BY_DESIGN | 0.0356 | 0.41 | 0.0349 | target not PRESENT in >= 80 % of seeds | circular_block_bootstrap_10pct_B50 | A_anchors, A_witnesses |
| `A-R` | absent | NAS | `W_NAS_broadcast_only` | 52 | 0.0385 | 0.0983 | NOT_TESTABLE_BY_DESIGN | 0.031 | 0.411 | 0.0439 | target not PRESENT in >= 80 % of seeds | jackknife_contiguous_10 | A_anchors, A_witnesses |
| `A-R` | absent | NAS | `W_NAS_no_workspace` | 52 | 1 | 0.919 | decisive | 0.0313 | 0.413 | 0.0134 | - | jackknife_contiguous_10 | A_anchors, A_witnesses |
| `A-R` | absent | PDI | `W_PDI_no_multistability` | 12 | - | - | no row (12 development runs; the gate needs 40) | - | - | - | - | concordant, jackknife_contiguous_10 | A_witnesses |
| `A-R` | absent | PDI | `W_PDI_single_attractor` | 52 | 0.788 | 0.788 | NOT_TESTABLE_BY_DESIGN | 0.0354 | 0.409 | 0 | target not PRESENT in >= 80 % of seeds | concordant, jackknife_contiguous_10 | A_anchors, A_witnesses |
| `A-R` | absent | RAM | `W_RAM_no_plasticity` | 52 | 0 | 0 | NOT_TESTABLE_BY_DESIGN | 0.042 | 0.45 | 0.329 | target not PRESENT in >= 80 % of seeds | shift_null_sd | A_anchors, A_witnesses |
| `A-R` | absent | SRPI | `W_SRPI_no_efference` | 52 | 0 | 0 | NOT_TESTABLE_BY_DESIGN | 0.0355 | 0.409 | 0.458 | target not PRESENT in >= 80 % of seeds | jackknife_pairs_10 | A_anchors, A_witnesses |
| `A-R` | present | IIM | `PC_nominal` | 64 | 0.906 | 0.933 | decisive | 0.0362 | 0.414 | 0.201 | - | circular_block_bootstrap_10pct_B50 | A_anchors, A_witnesses |
| `A-R` | present | NAS | `PC_nominal` | 64 | 1 | 0.999 | decisive | 0.0362 | 0.414 | 0.098 | - | jackknife_contiguous_10 | A_anchors, A_witnesses |
| `A-R` | present | PDI | `PC_nominal` | 64 | 1 | 1 | decisive | 0.0355 | 0.409 | 0.0901 | - | jackknife_contiguous_10 | A_anchors, A_witnesses |
| `A-R` | present | RAM | `PC_nominal` | 64 | 0.672 | 0.632 | NOT_TESTABLE_BY_DESIGN | 0.0421 | 0.45 | 0.379 | paired contrast with the positive control | shift_null_sd | A_anchors, A_witnesses |
| `A-R` | present | SRPI | `PC_nominal` | 64 | 0.766 | 0.754 | NOT_TESTABLE_BY_DESIGN | 0.0359 | 0.412 | 0.284 | paired contrast with the positive control | jackknife_pairs_10 | A_anchors, A_witnesses |
| `A-RAM160` | absent | IIM | `W_IIM_feedforward` | 0 | - | - | no row (0 development runs; the gate needs 40) | - | - | - | - | - |  |
| `A-RAM160` | absent | NAS | `W_NAS_broadcast_only` | 0 | - | - | no row (0 development runs; the gate needs 40) | - | - | - | - | - |  |
| `A-RAM160` | absent | NAS | `W_NAS_no_workspace` | 0 | - | - | no row (0 development runs; the gate needs 40) | - | - | - | - | - |  |
| `A-RAM160` | absent | PDI | `W_PDI_no_multistability` | 0 | - | - | no row (0 development runs; the gate needs 40) | - | - | - | - | - |  |
| `A-RAM160` | absent | PDI | `W_PDI_single_attractor` | 0 | - | - | no row (0 development runs; the gate needs 40) | - | - | - | - | - |  |
| `A-RAM160` | absent | RAM | `W_RAM_no_plasticity` | 60 | 0 | 0 | NOT_TESTABLE_BY_DESIGN | 0.0425 | 0.453 | 0.173 | target not PRESENT in >= 80 % of seeds | shift_null_sd | RAM160, RAM160_anchors |
| `A-RAM160` | absent | SRPI | `W_SRPI_no_efference` | 0 | - | - | no row (0 development runs; the gate needs 40) | - | - | - | - | - |  |
| `A-RAM160` | present | IIM | `PC_nominal` | 0 | - | - | no row (0 development runs; the gate needs 40) | - | - | - | - | - |  |
| `A-RAM160` | present | NAS | `PC_nominal` | 0 | - | - | no row (0 development runs; the gate needs 40) | - | - | - | - | - |  |
| `A-RAM160` | present | PDI | `PC_nominal` | 0 | - | - | no row (0 development runs; the gate needs 40) | - | - | - | - | - |  |
| `A-RAM160` | present | RAM | `PC_nominal` | 60 | 0.983 | 0.984 | decisive | 0.0425 | 0.453 | 0.194 | - | shift_null_sd | RAM160, RAM160_anchors |
| `A-RAM160` | present | SRPI | `PC_nominal` | 0 | - | - | no row (0 development runs; the gate needs 40) | - | - | - | - | - |  |
| `B` | absent | IIM | `feedforward_star` | 50 | 1 | 1 | decisive | 0.0354 | 0.409 | 0.00798 | - | - | family_b |
| `C1-H` | absent | IIM | `W_IIM_feedforward` | 40 | 0 | 0 | NOT_TESTABLE_BY_DESIGN | 0.0355 | 0.409 | 0.22 | target not PRESENT in >= 80 % of seeds | circular_block_bootstrap_10pct_B50 | C1_anchors |
| `C1-H` | absent | NAS | `W_NAS_broadcast_only` | 52 | 0.462 | 0.65 | NOT_TESTABLE_BY_DESIGN | 0.0309 | 0.41 | 0.0208 | target not PRESENT in >= 80 % of seeds | jackknife_contiguous_10 | C1_anchors, C1_witnesses |
| `C1-H` | absent | NAS | `W_NAS_no_workspace` | 52 | 0.923 | 0.857 | decisive | 0.031 | 0.411 | 0.0144 | - | jackknife_contiguous_10 | C1_anchors, C1_witnesses |
| `C1-H` | absent | PDI | `W_PDI_no_multistability` | 0 | - | - | no row (0 development runs; the gate needs 40) | - | - | - | - | - |  |
| `C1-H` | absent | PDI | `W_PDI_single_attractor` | 12 | - | - | no row (12 development runs; the gate needs 40) | - | - | - | - | - | C1_witnesses |
| `C1-H` | absent | RAM | `W_RAM_no_plasticity` | 12 | - | - | no row (12 development runs; the gate needs 40) | - | - | - | - | - | C1_witnesses |
| `C1-H` | absent | SRPI | `W_SRPI_no_efference` | 12 | - | - | no row (12 development runs; the gate needs 40) | - | - | - | - | - | C1_witnesses |
| `C1-H` | present | IIM | `PC_nominal` | 40 | 0.25 | 0.402 | NOT_TESTABLE_BY_DESIGN | 0.0356 | 0.41 | 0.472 | paired contrast with the positive control | circular_block_bootstrap_10pct_B50 | C1_anchors |
| `C1-H` | present | NAS | `PC_nominal` | 52 | 1 | 1 | decisive | 0.0359 | 0.412 | 0.0878 | - | jackknife_contiguous_10 | C1_anchors, C1_witnesses |
| `C1-H` | present | PDI | `PC_nominal` | 52 | 0 | - | NOT_TESTABLE_BY_DESIGN | 0.043 | 0.456 | - | paired contrast with the positive control | - | C1_anchors, C1_witnesses |
| `C1-H` | present | RAM | `PC_nominal` | 52 | 0 | - | NOT_TESTABLE_BY_DESIGN | 0.043 | 0.456 | - | paired contrast with the positive control | - | C1_anchors, C1_witnesses |
| `C1-H` | present | SRPI | `PC_nominal` | 52 | 0 | - | NOT_TESTABLE_BY_DESIGN | 0.043 | 0.456 | - | paired contrast with the positive control | - | C1_anchors, C1_witnesses |
| `C1-R` | absent | IIM | `W_IIM_feedforward` | 40 | 0 | 0 | NOT_TESTABLE_BY_DESIGN | 0.0355 | 0.41 | 0.21 | target not PRESENT in >= 80 % of seeds | circular_block_bootstrap_10pct_B50 | C1_anchors |
| `C1-R` | absent | NAS | `W_NAS_broadcast_only` | 52 | 0.442 | 0.641 | NOT_TESTABLE_BY_DESIGN | 0.0309 | 0.41 | 0.0211 | target not PRESENT in >= 80 % of seeds | jackknife_contiguous_10 | C1_anchors, C1_witnesses |
| `C1-R` | absent | NAS | `W_NAS_no_workspace` | 52 | 0.923 | 0.859 | decisive | 0.031 | 0.411 | 0.0147 | - | jackknife_contiguous_10 | C1_anchors, C1_witnesses |
| `C1-R` | absent | PDI | `W_PDI_no_multistability` | 0 | - | - | no row (0 development runs; the gate needs 40) | - | - | - | - | - |  |
| `C1-R` | absent | PDI | `W_PDI_single_attractor` | 12 | - | - | no row (12 development runs; the gate needs 40) | - | - | - | - | - | C1_witnesses |
| `C1-R` | absent | RAM | `W_RAM_no_plasticity` | 12 | - | - | no row (12 development runs; the gate needs 40) | - | - | - | - | - | C1_witnesses |
| `C1-R` | absent | SRPI | `W_SRPI_no_efference` | 12 | - | - | no row (12 development runs; the gate needs 40) | - | - | - | - | - | C1_witnesses |
| `C1-R` | present | IIM | `PC_nominal` | 40 | 0.4 | 0.418 | NOT_TESTABLE_BY_DESIGN | 0.0356 | 0.41 | 0.502 | paired contrast with the positive control | circular_block_bootstrap_10pct_B50 | C1_anchors |
| `C1-R` | present | NAS | `PC_nominal` | 52 | 1 | 1 | decisive | 0.0359 | 0.412 | 0.0871 | - | jackknife_contiguous_10 | C1_anchors, C1_witnesses |
| `C1-R` | present | PDI | `PC_nominal` | 52 | 0 | - | NOT_TESTABLE_BY_DESIGN | 0.043 | 0.456 | - | paired contrast with the positive control | - | C1_anchors, C1_witnesses |
| `C1-R` | present | RAM | `PC_nominal` | 52 | 0 | - | NOT_TESTABLE_BY_DESIGN | 0.043 | 0.456 | - | paired contrast with the positive control | - | C1_anchors, C1_witnesses |
| `C1-R` | present | SRPI | `PC_nominal` | 52 | 0 | - | NOT_TESTABLE_BY_DESIGN | 0.043 | 0.456 | - | paired contrast with the positive control | - | C1_anchors, C1_witnesses |

61 rows: 12 decisive, 29 NOT_TESTABLE_BY_DESIGN, 20 without a row. A row without a row object has fewer than 40 development runs; the builder lists such a cell as a blocker only when the part needs it, and none does in the frozen build (`build_manifest.json`: no blocking entries). The C1 cells of HCv2-22(ii) for PDI, RAM and SRPI are not cells of that part: their targets are not in `N_anch(C1)`.

## Status rates of the gated rows

Development share of each status or reason among the runs of the row (UNDEFINED is the sum of its reasons; `PRESENT_NOT_REACHABLE` is a flag, not a status).

| Protocol | Kind | Principle | Witness | PRESENT | ABSENT | INCONCLUSIVE | ABSENT_NOT_REACHABLE | NULL_MODEL_VIOLATED | INVALID_ANCHORS | UNDEFINED | PRESENT_NOT_REACHABLE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `A-H` | absent | IIM | `W_IIM_feedforward` | 0.558 | 0 | 0 | 0.423 | 0 | 0 | 0.442 | 0.212 |
| `A-H` | absent | NAS | `W_NAS_broadcast_only` | 0.0385 | 0 | 0.269 | 0.692 | 0 | 0 | 0.962 | 0 |
| `A-H` | absent | NAS | `W_NAS_no_workspace` | 0.135 | 0.154 | 0.538 | 0.173 | 0 | 0 | 0.712 | 0 |
| `A-H` | absent | PDI | `W_PDI_single_attractor` | 0 | 0.788 | 0 | 0.212 | 0 | 0 | 0.212 | 0 |
| `A-H` | absent | RAM | `W_RAM_no_plasticity` | 0 | 0 | 0 | 0.904 | 0.0962 | 0 | 1 | 0 |
| `A-H` | absent | SRPI | `W_SRPI_no_efference` | 0.0192 | 0 | 0 | 0.981 | 0 | 0 | 0.981 | 0.654 |
| `A-H` | present | IIM | `PC_nominal` | 0.375 | 0 | 0 | 0.5 | 0.0312 | 0 | 0.625 | 0.516 |
| `A-H` | present | NAS | `PC_nominal` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `A-H` | present | PDI | `PC_nominal` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `A-H` | present | RAM | `PC_nominal` | 0.672 | 0 | 0 | 0.328 | 0 | 0 | 0.328 | 0.0156 |
| `A-H` | present | SRPI | `PC_nominal` | 0.766 | 0 | 0 | 0.234 | 0 | 0 | 0.234 | 0.141 |
| `A-R` | absent | IIM | `W_IIM_feedforward` | 0 | 0.231 | 0.288 | 0.481 | 0 | 0 | 0.769 | 0 |
| `A-R` | absent | NAS | `W_NAS_broadcast_only` | 0 | 0.0385 | 0.135 | 0.827 | 0 | 0 | 0.962 | 0 |
| `A-R` | absent | NAS | `W_NAS_no_workspace` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `A-R` | absent | PDI | `W_PDI_single_attractor` | 0 | 0.788 | 0 | 0.212 | 0 | 0 | 0.212 | 0 |
| `A-R` | absent | RAM | `W_RAM_no_plasticity` | 0 | 0 | 0 | 0.904 | 0.0962 | 0 | 1 | 0 |
| `A-R` | absent | SRPI | `W_SRPI_no_efference` | 0.0192 | 0 | 0 | 0.981 | 0 | 0 | 0.981 | 0.654 |
| `A-R` | present | IIM | `PC_nominal` | 0.906 | 0 | 0 | 0.0938 | 0 | 0 | 0.0938 | 0 |
| `A-R` | present | NAS | `PC_nominal` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `A-R` | present | PDI | `PC_nominal` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `A-R` | present | RAM | `PC_nominal` | 0.672 | 0 | 0 | 0.328 | 0 | 0 | 0.328 | 0.0156 |
| `A-R` | present | SRPI | `PC_nominal` | 0.766 | 0 | 0 | 0.234 | 0 | 0 | 0.234 | 0.141 |
| `A-RAM160` | absent | RAM | `W_RAM_no_plasticity` | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 |
| `A-RAM160` | present | RAM | `PC_nominal` | 0.983 | 0 | 0 | 0.0167 | 0 | 0 | 0.0167 | 0 |
| `B` | absent | IIM | `feedforward_star` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `C1-H` | absent | IIM | `W_IIM_feedforward` | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0.025 |
| `C1-H` | absent | NAS | `W_NAS_broadcast_only` | 0 | 0.462 | 0.365 | 0.173 | 0 | 0 | 0.538 | 0 |
| `C1-H` | absent | NAS | `W_NAS_no_workspace` | 0 | 0.923 | 0.0769 | 0 | 0 | 0 | 0.0769 | 0 |
| `C1-H` | present | IIM | `PC_nominal` | 0.25 | 0 | 0 | 0.75 | 0 | 0 | 0.75 | 0.675 |
| `C1-H` | present | NAS | `PC_nominal` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `C1-H` | present | PDI | `PC_nominal` | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| `C1-H` | present | RAM | `PC_nominal` | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| `C1-H` | present | SRPI | `PC_nominal` | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| `C1-R` | absent | IIM | `W_IIM_feedforward` | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0.025 |
| `C1-R` | absent | NAS | `W_NAS_broadcast_only` | 0 | 0.442 | 0.385 | 0.173 | 0 | 0 | 0.558 | 0 |
| `C1-R` | absent | NAS | `W_NAS_no_workspace` | 0 | 0.923 | 0.0769 | 0 | 0 | 0 | 0.0769 | 0 |
| `C1-R` | present | IIM | `PC_nominal` | 0.4 | 0 | 0 | 0.6 | 0 | 0 | 0.6 | 0.55 |
| `C1-R` | present | NAS | `PC_nominal` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `C1-R` | present | PDI | `PC_nominal` | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| `C1-R` | present | RAM | `PC_nominal` | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| `C1-R` | present | SRPI | `PC_nominal` | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
