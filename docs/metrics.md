# Metric Definitions

This document defines what the code computes: the five null-anchored MPC
estimators of `impact_pipeline.mpc_metrics` (RAM, PDI, NAS, IIM, SRPI) and
their benchmark variants in `impact_pipeline.estimators`, the null
calibration, the evidence layer with its two status rules (the interval rule
of the empirical pipeline and the TOST rule of the benchmark protocols), the
protocols that declare everything a verdict depends on, the deprecated legacy
CI, the exploratory statistic S, the output columns of the pipeline and the
benchmark MPC-Bench. It is implementation-facing: every statement refers to a
module under `src/impact_pipeline/`. Function docstrings are the
authoritative parameter reference.

MPC stands for the Minimal Principles for Consciousness of the IMPaCT framework.
The principles are measured by:

| Code | Name | Principle measured |
|---|---|---|
| RAM | Responsiveness-Adaptation Metric | goal-directed responsiveness and feedback-driven adaptation |
| PDI | Pattern Differentiation Index | differentiation of the state repertoire |
| NAS | Network Availability Score (broadcast capacity) | global availability: receive-and-return broadcast through a workspace |
| IIM | Integrated-Information Measure | integration, an IIT-inspired proxy computed from an estimated causal TPM (it is **not** IIT's Φ and is never written as Φ) |
| SRPI | Self-Referential Processing Index | self/other (legacy) or self-caused/other-caused (agency) processing |

## Contents

1. [Conventions](#1-conventions)
2. [RAM](#2-ram)
3. [PDI](#3-pdi)
4. [NAS](#4-nas)
5. [IIM](#5-iim)
6. [SRPI](#6-srpi)
7. [Null calibration](#7-null-calibration)
8. [Evidence layer: MPC profile, verdict and degree](#8-evidence-layer-mpc-profile-verdict-and-degree)
9. [TOST status rule](#9-tost-status-rule)
10. [Protocols](#10-protocols)
11. [Legacy CI (deprecated)](#11-legacy-ci-deprecated)
12. [Exploratory statistic S](#12-exploratory-statistic-s)
13. [Output columns](#13-output-columns)
14. [Benchmark (MPC-Bench)](#14-benchmark-mpc-bench)

## 1. Conventions

- **Arrays.** The estimators take `ts` with shape `(n_nodes, n_time)` and the
  sample interval `tr` in seconds (TR for fMRI, `1/sfreq` for EEG). Time series
  on disk (`<subject>_run-<k>_<atlas>_ts.npy`) are stored as time x nodes and
  transposed on load.
- **Measured inputs only.** A quantity that cannot be measured is undefined:
  NaN with a reason string (`undefined_reason`, `<metric>_undefined_reason`,
  `PDI_anchor_reason`, ...), never 0. A measured 0 is a legitimate value.
  There are no silent fallbacks (for example no default TR, no implicit
  stimuli, no `response_time` used as feedback) unless a keyword explicitly
  opts in.
- **Modes.** Each estimator keeps its original construct as the default mode
  (`legacy`), with values pinned by regression tests. The construct revisions
  are new modes (`mode=`/`update=` keywords). `run_pipeline.py` uses the modes
  of its protocol: on empirical data by default
  [`protocols/mpc_default.json`](../protocols/README.md#mpc_defaultjson-default-for-empirical-data),
  otherwise the default modes unless `--protocol` or the params dicts select others
  (section 8.6). MPC-Bench runs the `mpc_metrics` estimators in the modes of
  `impact_pipeline.bench.export.OPTIONAL_MODES` (RAM `prediction_error`, PDI
  `repertoire`, NAS `capacity`, SRPI `agency`).
- **Bearer.** RAM, PDI, NAS and SRPI accept `bearer_nodes` (the declared system
  that bears the evidence; default all nodes); IIM draws its subsystem only from
  bearer nodes.
- **Seeds.** Every random step has an explicit seed (`null_seed`,
  `quality_random_state`, `agency_random_state`, `random_state`, `rng`).
- **Hardware.** `hardware_backend` / `--hardware-target` selects NumPy (CPU) or
  CuPy (GPU/APU) array operations for the matrix-heavy kernels. Results agree up
  to rounding; `python -m impact_pipeline.hardware_selftest` checks this.

**Versions and variants.** Every estimator carries a version string that is
recorded with each component (`<P>_estimator_version` in the pipeline,
`estimator_version` in benchmark records). `evidence.ESTIMATOR_VERSIONS` lists
them per principle, the `mpc_metrics` estimator first, then its benchmark
variant:

| Principle | `mpc_metrics` estimator | Benchmark variant (`impact_pipeline.estimators`) |
|---|---|---|
| RAM | `compute_RAM`, `ram-2026.09` | `ram_pe.compute_ram_pe`, `ram-pe-2026.10` (section 2.4) |
| PDI | `compute_PDI`, `pdi-2026.09` | `pdi_declared.compute_pdi_declared`, `pdi-declared-2026.10` (section 3.4) |
| NAS | `compute_NAS`, `nas-2026.09` | `nas_declared.compute_nas_declared`, `nas-declared-2026.10` (section 4.3) |
| IIM | `compute_IIM`, `iim-2026.09` | `iim_macro.compute_iim_macro`, `iim-macro-2026.10` (section 5.6) |
| SRPI | `compute_SRPI`, `srpi-2026.09` | none (section 6.3) |

The empirical pipeline runs the `mpc_metrics` estimators. The variants read
the declared inputs of benchmark systems (section 14.2), report the SE method
of their sampling SE and are judged by the TOST rule (section 9).

## 2. RAM

### 2.1 Definition (default: `update="feedback_magnitude"`)

$$
\mathrm{RAM} = \frac{M}{T + \epsilon}\,Q,
\qquad
Q = \left(G^{w_G} F^{w_F} U^{w_U}\right)^{1/(w_G + w_F + w_U)},
\qquad G, F, U \in [0, 1].
$$

- $M$: response magnitude, `magnitude_scale` (default 0.5) times the mean over
  nodes of the absolute stimulus beta. Goal and feedback events enter the same
  GLM as nuisance regressors with the same response model, so $M$ is the
  stimulus amplitude only. A rank-deficient design makes RAM undefined
  (`magnitude_design_rank_deficient`).
- $T$: response latency in seconds (`latency_method`):
  - `hrf_peak` (default): the model latency, i.e. the peak of the canonical HRF
    (about 5 s) for `response_model="hrf"`, half the boxcar width for
    `"boxcar"`. It is a model constant, not a measurement.
  - `fir`: time to peak of the baseline-corrected event average (global field
    power across nodes) within `fir_window` after onset.
  - `xcorr`: the non-negative lag (up to `xcorr_maxlag` samples) that maximises
    the mean squared correlation between the stimulus stick function and the
    nodes.

  Measured latencies are refined by parabolic interpolation. A peak in the 5%
  edge band of the search window leaves $T$, and RAM, undefined. $T$ is never
  clipped to 0, so RAM cannot collapse to $|\beta|/\epsilon$.
- $\epsilon$: stabiliser equal to the temporal resolution (`epsilon=None` means
  `epsilon = tr`).
- The **canonical HRF** is the Glover double-gamma (delay 6 s, undershoot 12 s,
  dispersion 0.9, undershoot ratio 0.48, truncated at 32 s), evaluated
  analytically on the true time axis at the exact onsets and peak-normalised.
- The speed term $M/(T+\epsilon)$ is in s$^{-1}$; $Q$ is unitless.

Quality terms. Trial-wise response patterns come from a least-squares-all
canonical-HRF GLM (one regressor per goal, stimulus and feedback event) for
`response_model="hrf"` (`quality_response_estimate="auto"` or `"glm"`), and from
event-locked window means otherwise. Every term is chance-corrected and
multiplied by the event-count reliability $r_n = 1 - e^{-n/4}$:

- **G, goal alignment.** Each stimulus response is paired with the pattern of its
  nearest preceding goal event. The association is the held-out correlation of
  the first ridge-CCA pair, fitted on time-blocked training folds
  (`quality_cv_folds`, default 5; ridge `quality_ridge` relative to the mean
  non-zero training eigenvalue, default 1.0). The chance level $m_0$ is the mean
  of the same statistic on multivariate phase-randomised surrogates of the data,
  analysed through the identical event design (`quality_null_samples`, default
  200). $G = [(r - m_0)/(1 - m_0)]_+ \, r_n$. $G \approx 0$ on noise and on
  shuffled goal labels; the in-sample canonical correlation (about 1 for noise
  when nodes outnumber events) is not used.
- **F, feedback integration.** Chance-corrected $|\mathrm{corr}|$ between the
  feedback-locked response strength and the explicit feedback values of the same
  events (aligned 1:1); the null permutes the feedback values.
- **U, adaptive update.** For consecutive stimulus responses $k \to k+1$, the
  update size $\lVert r_{k+1} - r_k \rVert$ is paired with the mean
  $|\text{feedback}|$ of the feedback events in $[s_k, s_{k+1})$; chance-corrected
  $|\mathrm{corr}|$ (permutation null).

Strict contract (defaults `require_explicit_goals=True`,
`require_explicit_feedback=True`): RAM is undefined without goal events, without
explicit feedback events and values, with fewer than 6 goal-response pairs
(`insufficient_goal_response_pairs`), fewer than 3 feedback events or fewer than
3 adaptive updates. A weighted quality term that cannot be measured makes RAM
undefined; a measured 0 gives RAM = 0.

### 2.2 Modality presets (`run_synergy_ci.RAM_PARAM_PRESETS`)

| Parameter | fMRI | EEG |
|---|---|---|
| `response_model` | `hrf` | `boxcar` (0.30 s) |
| `latency_method` | `hrf_peak` | `fir` (0.80 s window) |
| goal-pre / response / goal-objective / feedback windows (s) | 2.0 / 3.0 / 2.0 / 2.0 | 0.20 / 0.40 / 0.20 / 0.20 |
| `quality_lag_sec` | HRF peak minus half the response window | 0.0 |

RAM parameters are modality-specific: without explicit `ram_params` and without
a known modality, `run_synergy_ci.resolve_ram_params` raises.

### 2.3 Typed channels and the prediction-error update

- `impact_channel` (events column) types each event row:
  `behavioural_feedback`, `covert_neural`, `perturbational`, `endogenous`.
  `compute_RAM(..., impact_channel=<channel>)` restricts RAM to that channel's
  events; `compute_RAM_by_channel` returns one result per channel. The first two
  channels are implemented; `perturbational` and `endogenous` return NaN with
  `undefined_reason="NOT_IMPLEMENTED"`. Channels are meant to be combined by the
  evidence layer (Kleene OR), not averaged. The default empirical protocol
  `protocols/mpc_default.json` declares RAM on the untyped `default`
  channel and on `perturbational` and `endogenous`, so RAM can be PRESENT but
  never ABSENT there (behavioural non-response alone never excludes); the
  opt-in `protocols/mpc_behavioural_ram.json` declares only `default`
  (RAM can be ABSENT; a choice to fix before use, and its results are
  behavioural-RAM results: an ABSENT RAM means no
  responsiveness-and-adaptation above the null in the recorded behaviour, not
  absence of responsiveness; see
  [`protocols/README.md`](../protocols/README.md#mpc_behavioural_ramjson-opt-in)).
- `update="prediction_error"`: U regresses the change of the stimulus-response
  mapping $d_k = r_{k+1} - r_k$ on the signed prediction error of a
  Rescorla-Wagner / Q-learning model fitted by maximum likelihood to the logged
  `choice`/`reward` columns. Grid: $\alpha$ in 21 values in $[0, 1]$, $\beta$ in
  $\{0\}$ plus 15 log-spaced values in $[0.25, 32]$; rewards rescaled to
  $[0, 1]$, values start at 0.5, ties go to the smallest $\alpha$ then $\beta$.
  The statistic is the pooled $R^2$ (share of pattern-change variance explained
  by the prediction error), chance-corrected by permuting the prediction errors.
  Undefined without a usable choice/reward log.
- `adaptation_locus` (`state`, `weights`, `undeclared`) is recorded in the
  details; it does not change the estimate.

### 2.4 RAM-PE (`ram-pe-2026.10`)

`estimators.ram_pe.compute_ram_pe` measures the plasticity channel of RAM
only: across consecutive trials of a two-option choice task, the change of
the bearer's stimulus-evoked response pattern carries the intervening
outcome's prediction error, signed by the chosen option, beyond what the
unaligned trial sequence explains. Goal alignment, feedback magnitude and
response speed have no referent in the benchmark generators and are declared
not applicable (`FACETS_NOT_APPLICABLE`); the product composite of
`compute_RAM` and its speed term are not used.

1. The bearer's nodes are z-scored; the response pattern $r_k$ of stimulus $k$
   is the window mean over $[s_k + L, s_k + L + W)$ ($W = 0.4$ s, $L = 0$).
   The window must end before the logged response in at least 95% of the
   trials (`NOT_DEFINED:window_includes_response`).
2. The maximum-likelihood Rescorla-Wagner fit of `compute_RAM` (section 2.3)
   gives the prediction error $\delta_j$ of every logged outcome. The target
   of stimulus $k$ is $y_k = s_k\,\delta_j$ for the outcome $j$ logged before
   the next stimulus, with $s_k = \pm 1$ by the chosen option.
3. Update pairs $d_k = r_{k+1} - r_k$ with target $y_k$. Fewer than 30 updates
   or fewer than two chosen options is `INSUFFICIENT_UPDATES`.
4. The statistic $m$ is the Pearson correlation of the pooled out-of-fold
   ridge predictions of $y$ from $d$ (five contiguous outer folds purged by
   the update lag, features standardised on the training rows, ridge
   $\lambda \in \{0.1, 1, 10, 100, 1000\}$ chosen by an inner purged 4-fold
   contiguous CV).
5. Null `trial_circular_shift` (exact and deterministic): every cyclic shift
   $h \in \{5, \dots, n - 5\}$ of $y$ against the pattern changes, scored by
   the identical CV; $n_{\mathrm{null}} = n - 9$.
6. SE `shift_null_sd`: the SD of that null, `se_df = n_null - 1`. The
   delete-a-group jackknife over 10 contiguous blocks of update rows
   (`jackknife_trials_10`, `se_df = 9`) is always computed and reported.

The statistic is a correlation, first order in the effect. RAM-PE does not
distinguish contingent from non-contingent feedback (an agent that updates on
scrambled feedback is RAM-PE-on), and an outcome-locked transient echo can
produce the observable. On the family-C carrier recording it is declared not
applicable (`NOT_APPLICABLE_OBSERVATION_MODEL`).

## 3. PDI

### 3.1 Default mode (`mode="legacy"`): baseline-referenced composite

$$
\mathrm{PDI}_{\mathrm{raw}} = D_{\mathrm{core}}\,C_{\mathrm{stab}}\,C_{\mathrm{noise}},
\qquad
\mathrm{PDI}^{+} = \max(\mathrm{PDI}_{\mathrm{raw}}, 0).
$$

Four dimensions are measured on the observed run and on each baseline run
(values discretised with shared bin edges, `bins`, default 10):

$$
X_{\Xi} = \mathrm{clip}\!\left(\frac{H_R - h_R}{\log_2 K}, 0, 1\right),\qquad
X_{\Delta} = \frac{2}{R(R-1)}\sum_{n<m}\mathrm{JSD}(p_n, p_m),
$$

$$
X_D = \frac{d_{\mathrm{eff}} - 1}{Q - 1},\quad d_{\mathrm{eff}} = \frac{(\sum_q s_q^2)^2}{\sum_q s_q^4},\qquad
X_C = \frac{1}{|\mathcal A| R}\sum_{a\in\mathcal A}\sum_{n=1}^{R}
\frac{H(\pi_{n,a})}{\log_2(m!)}\cdot\frac{\mathrm{JSD}(\pi_{n,a}, u_m)}{\mathrm{JSD}(\delta_m, u_m)}.
$$

$X_\Xi$ is normalised excess repertoire differentiation (entropy minus one-step
conditional entropy), $X_\Delta$ spatial repertoire divergence, $X_D$ effective
dimensionality and $X_C$ multiscale ordinal complexity (order
`ordinal_order`, scales up to `multiscale_max_scale`). Baseline moments are
weighted by run length ($w_b = N_b/\sum_\ell N_\ell$), and

$$
g_j = \mathrm{clip}\!\left(\frac{X_{j,\mathrm{obs}} - \bar X_{j,\mathrm{base}}}{1 + \sigma_{j,\mathrm{base}}}, 0, 1\right),\qquad
D_{\mathrm{core}} = \sum_j \alpha_j g_j,
$$

$$
C_{\mathrm{stab}} = \mathrm{clip}\!\left(\frac{\Upsilon_{\mathrm{obs}}}{\bar\Upsilon_{\mathrm{base}} + \epsilon_s}, 0, 1\right),\qquad
C_{\mathrm{noise}} = \exp\!\left[-\kappa\max(0, \nu_{\mathrm{obs}} - \bar\nu_{\mathrm{base}})\right],
$$

with weights $(\alpha_\Xi, \alpha_\Delta, \alpha_D, \alpha_C) = (0.35, 0.25, 0.20, 0.20)$.
With `clip_negative=False` the signed contrasts (clipped to $[-1, 1]$) are used.

Baselines in the pipeline (`pdi_require_strict_baseline=True`):

- `PDI_anchor`: baseline = the subject's `<anchor session>/rest` runs (default
  anchor session `deep`); this is the primary endpoint (`pdi_primary_endpoint`)
  and the value in the `PDI` column.
- `PDI_task`: baseline = the same session's `rest` runs.
- The evaluated run is never its own baseline. A missing, identical,
  region-mismatched or unreadable baseline makes the endpoint NaN with a reason
  (for example `missing_deep_rest_baseline`). The shuffled-task fallback of the
  function (`baseline_ts=None`) is not used by the pipeline.

Uncalibrated PDI is positively biased under the null (gains are clipped at 0
before weighting). Cross-condition comparisons should use the calibrated value
(section 7).

### 3.2 `mode="surrogate_excess"` (documented negative result)

Global-state differentiation beyond the linear (spectral) structure. Nodes are
z-scored and reduced to their leading `excess_components` principal components
(default 5, at most the rank and 30); each component is binarised at its
median, so each time point is a $k$-bit global-state word. Three features:

- repertoire entropy of the words (bits per component, in $[0, 1]$);
- LZ76 diversity of the binarised component matrix, normalised by $n/\log_2 n$;
- effective dimensionality of the median-binarised nodes, $(PR - 1)/(N - 1)$
  (the covariance of the continuous data is preserved exactly by the
  surrogates, the binarised one is not).

Each feature is compared with the same feature of multivariate
spectrum-preserving surrogates of the same run (shared random phases across
nodes, `excess_surrogate="fourier"`; `"iaaft"` also keeps each node's marginal).
PDI is the signed excess of the weighted mean of the features
(`excess_weights`, default equal) over its surrogate mean. Nothing is clipped
and no rest baseline is used. `null_surrogates=0` selects 19 surrogates; 1 is
rejected.

Caveat: for fixed auto- and cross-spectra a linear Gaussian process maximises
entropy, so entropy-type features of structured (for example multistable)
dynamics typically fall **below** their surrogates. In the known-answer tests
the aggregate is about 0 for linear Gaussian data but of either sign for
multistable pattern switching. This mode is not a positive "more differentiated
than chance" score: the Gaussian null is the maximum-entropy process for the
given spectra, so an excess over it cannot index differentiation. It is kept
for reference and is not used by the benchmark; `mode="repertoire"`
replaces it.

### 3.3 `mode="repertoire"`: repertoire of distinguishable states

Differentiation as the repertoire of distinguishable multi-node states the
system occupies (Tononi & Edelman 1998; Mensen et al. 2017), in bits, from
short-window patterns: non-overlapping windows of `repertoire_window` samples
(default 5) of the z-scored nodes, summarised by their mean
(`repertoire_features="mean"`) or power. Two variants, selected by the inputs:

- **Labelled** (`state_labels`, or `events` with `tr` and
  `repertoire_label_column`, default `trial_type`, delayed by
  `repertoire_label_delay` seconds): cross-validated multiclass shrinkage-LDA
  decoding of the declared state of each window (time-blocked folds,
  `repertoire_folds`, purged by `repertoire_gap` windows at every test-block
  edge) and the Miller-Madow-corrected mutual information
  $I(\mathrm{state}; \widehat{\mathrm{state}})$ of the pooled CV confusion
  matrix. The null re-runs the whole CV with the label runs block-permuted
  (run order shuffled, run lengths and label counts kept;
  `PDI_null_method = block_label_permutation`). Windows whose samples carry
  different labels, and states with fewer than 3 windows or fewer than 2
  separate visits, are dropped (`dropped_states`).
- **Unlabelled** (no labels): the number $\hat K$ of recurring,
  distinguishable states, from k-means on the windows' leading
  `repertoire_components` (default 10) principal components with a held-out
  criterion: the partition at $k$ is fitted on one half of interleaved time
  blocks and checked on the other half, in both directions, and $\hat K$ is
  the largest $k$ (up to `repertoire_max_states`, default 12) that passes.
  With `repertoire_criterion="separation"` (default) every state must recur
  (at least 3 held-out windows), persist (mean held-out dwell at least
  `repertoire_min_dwell` = 2.5 windows) and be separated from every other
  state by a density valley (ratio at most `repertoire_valley` = 0.4, a
  resolution of about $d' = 3.85$ after window averaging). The null is
  `repertoire_null`: `circular_shift` (default; independent shifts per node),
  `fourier` (multivariate phase randomisation) or `both` (the smaller excess
  of the two, reported with the binding family).

The evidence statistic `raw` is the observed $I$ (labelled) or
$\log_2 \hat K$ (unlabelled) in bits; the value is the signed excess over the
null mean (`PDI_excess`, not clipped; `PDI_calibrated` equals it).
`null_surrogates` is the number of permutations or surrogates (0 selects 19;
1 is rejected). `baseline_ts` and `null_method` are ignored (recorded).
The number of states is an integer, so the null SD can be 0; the excess in
bits, not `PDI_z`, is the construct-scale quantity.

Limitations (see the docstring): states shorter than about 2.5 windows are not
counted; exactly periodic, noise-free oscillations commensurate with the window
grid can form lattice "states"; at most about `n_windows / 6` states are
estimable without labels; the unlabelled count is conservative and falls with
noise; the clustering geometry is Euclidean on z-scored nodes, so volume
conduction, a common reference or a dominant global signal change it, and
scalp-EEG use needs validation on forward-modelled data first (section 8.3,
applicability registry). The pipeline runs the unlabelled variant (a protocol
cannot carry per-run labels; `state_labels`, `events` and `tr` in the PDI
options are rejected).

### 3.4 Declared PDI (`pdi-declared-2026.10`)

`estimators.pdi_declared.compute_pdi_declared` counts the recurring,
distinguishable multi-node states of a declared **content bearer**: every
bearer node except the declared access (workspace) nodes, so that ignition and
other global access-state alternations are not counted as content states.
Content is not identifiable from passive, unlabelled observation alone, so it
is identified by the access-node declaration. Without one (sensor data) the
full-bearer count is the only reading: an exclusion-valid upper bound whose
PRESENT is not content-specific, marked `upper_bound`. The steps follow
`mode="repertoire"` (section 3.3) except where stated:

1. Window patterns (mean of $w = 5$ samples of the z-scored nodes),
   column-centred, reduced to the $r = \min(10, \mathrm{rank})$ leading
   principal components, each scaled to unit variance.
2. Split with offset: the windows are cut into six contiguous blocks after a
   cyclic rotation by $o\,n_w/6$ windows, $o \in \{0, 1/4, 1/2, 3/4\}$; the
   blocks alternate between two halves, and the first window of every block
   is purged.
3. For $k = 2, \dots, k_{\max}$ ($k_{\max} = \min(12, \lfloor\min(n_A, n_B)/3\rfloor)$)
   every k-means solution fitted on one half is checked on the other: every
   state recurs (at least 3 held-out windows) and persists (mean held-out
   dwell at least 2.5 windows), and every pair of states is separated by a
   density valley (ratio at most 0.4) on its **Fisher axis**
   $w = S_w^{-1}(c_j - c_i)$, with $S_w$ the Ledoit-Wolf-shrunk pooled
   within-state covariance. $K_o$ is the largest $k$ that passes in both
   directions (1 if none).
4. $B$ is the mean over the offsets of $\log_2 K_o$ (bits).
5. Null `circular_shift`: 19 surrogates of the bearer, scored by the
   identical $B$. SE: delete-a-group jackknife over 10 contiguous time blocks
   (`jackknife_contiguous_10`, `se_df = 9`).
6. **Concordance.** When all $(10 + 1) \times 4 = 44$ counts (the run and
   its 10 replicates at the four offsets) are identical, `se` is 0 and the SE
   method is `concordant`, without `se_df`. Such a component is not exact:
   the TOST rule decides it only through a concordance route that the
   protocol admits for its cell (section 9.3), and otherwise it is
   `NO_SAMPLING_SE`.

The bearer actually counted is recorded (`counted_bearer`: `non_workspace` or
`full`). The family-C carrier recording is declared not applicable
(`NOT_APPLICABLE_OBSERVATION_MODEL`): its contexts are phase-coded on the
carrier. $B$ is in bits of a count, not a quadratic statistic.

## 4. NAS

### 4.1 Default mode (`mode="legacy"`): synchrony composite

The legacy construct (named "Network Activation Synchrony" in the code) combines
seven terms per frequency band $f$ by a weighted geometric mean and averages the
bands:

$$
\mathrm{NAS}_f = \bar L_f^{\alpha}\,\bar B_f^{\beta}\,\bar H_f^{\gamma}\,\bar D_f^{\delta}\,\bar W_f^{\eta}\,\bar E_f^{\zeta}\,\bar R_f^{\rho},
\qquad
\mathrm{NAS} = \sum_{f}\omega_f\,\mathrm{NAS}_f .
$$

$L$ intra-broadcast synchrony, $B$ broadcast reach, $H$ triadic closure, $D$
temporal stability of the synchrony graphs, $W$ workspace recruitment with
ignition, $E$ directed broadcast efficacy (lagged asymmetry), $R$ reverberatory
persistence. Exponents $(0.20, 0.16, 0.14, 0.12, 0.16, 0.12, 0.10)$, renormalised.
Pairwise synchrony mixes phase locking and envelope coupling
(`lambda_phase`) from band-passed data. With windows $w$ and the broadcast set
$V_{f,w} = \{i : s_i^{f,w} \ge Q_{1-\tau}(s^{f,w})\}$:

$$
G = \mathrm{TopK}_q(s^0),\quad s_i^0 = \frac{1}{R-1}\sum_{j\ne i}|\mathrm{corr}(x_i, x_j)|,
$$

$$
W_{f,w} = r_{f,w}\,\xi_{f,w},\quad r_{f,w} = \frac{|V_{f,w}\cap G|}{|G|},\quad
\xi_{f,w} = \mathrm{clip}\!\left(\frac{L_{f,w} - \bar A^{f,w}_{\bar V\bar V}}{1 - \bar A^{f,w}_{\bar V\bar V} + \epsilon}, 0, 1\right),
$$

$$
E_{f,w} = \mathrm{clip}\!\left(\frac{\langle\Gamma_{ij}\rangle_{i\in V, j\notin V} - \langle\Gamma_{ij}\rangle_{i\notin V, j\in V}}{\langle\Gamma_{ij}\rangle_{i\in V, j\notin V} + \langle\Gamma_{ij}\rangle_{i\notin V, j\in V} + \epsilon}, 0, 1\right),\quad
\Gamma_{ij} = \max(C_{ij} - C_{ji}, 0),
$$

$$
R_{f,w} = r_{f,w}\,\frac{1}{|\mathcal L|}\sum_{\ell\in\mathcal L}\max\!\big(\mathrm{corr}(y_{f,w}(t), y_{f,w}(t+\ell)), 0\big),\quad
y_{f,w}(t) = \frac{1}{|G|}\sum_{i\in G}x_i^{f,w}(t).
$$

- `tr`, `tau`, `bands`, `band_weights`, `window_len` and `step_len` must be
  given explicitly (dataset presets in `run_pipeline.DATASET_CONFIGS`).
- The workspace $G$ is `workspace_nodes` or the top `workspace_quantile` of
  global synchrony strength (at least `workspace_min_size` nodes).
- `normalize=True` (pipeline) clips the value to $[0, 1]$.
- The broadcast set and the inferred workspace are selected from the same data
  they are scored on, so uncalibrated NAS is positive for independent noise.
  Use the calibrated value (section 7). On very small systems the triadic term
  is structurally 0 (the broadcast set needs at least 3 nodes, i.e. about 11
  nodes at `tau=0.2`), which gives a degenerate null.

### 4.2 `mode="capacity"`: Network Availability Score (broadcast capacity)

Requires declared `workspace_nodes` (the hub, integer indices or a boolean mask;
invalid declarations raise). In the pipeline a missing declaration (neither the
protocol's NAS options nor `nas_params` declare a hub) makes NAS UNDEFINED
(`UNDEFINED:NAS:NO_DECLARED_WORKSPACE`) without calling the estimator. The
default empirical protocol `protocols/mpc_default.json` declares no hub,
so a run that measures NAS passes a derived protocol that does
([`protocols/examples/`](../protocols/examples/README.md) has two examples
derived from it, to be fixed before confirmatory use). Every other
bearer node is periphery. Hub and
periphery are reduced to their leading principal components
(`transfer_components`, default 5). Receive (periphery → hub) and return
(hub → periphery) are lagged Gaussian transfer entropies (`transfer_lags`,
default (1, 2) samples; nats), each compared with block circular-shift
surrogates (the hub block is shifted rigidly against the periphery; within-block
structure is kept, hub-periphery alignment is destroyed; minimum shift 10% of
the run; `null_surrogates=0` selects 19). The gated value is the signed excess
of the **weaker** direction (the one with the smaller z), so its z passes a
cutoff only if both directions do (intersection-union). Feedforward-only
broadcast gives about 0.

- Measured drivers can be conditioned on (`confounds`, shape
  `(n_confounds, n_time)`).
- Metastability (SD of the Kuramoto order parameter in `bands`, per-node
  circular-shift null) and the legacy $L$, $B$, $H$ terms (only when `tau`,
  `bands`, `window_len` and `step_len` are given) are profile descriptors
  reported in the details; they do not enter the value. $D$ is not computed.
- Caveat: an unmeasured common driver of hub and periphery inflates the receive
  (and sometimes the return) direction; observational Gaussian transfer cannot
  separate it from a loop. Conditioning on the driver removes it when the driver
  is measured.

### 4.3 Declared NAS (`nas-declared-2026.10`)

`estimators.nas_declared.compute_nas_declared` measures bidirectional,
lagged, linear-Gaussian dependence between a declared hub block and each
declared periphery block, with blocks represented by their mean signals,
beyond a declared exogenous-input basis (section 14.2), at lags that resolve a
declared coupling time scale $\tau_c$. It reads as workspace
receive-and-return only if the shared inputs are declared complete,
$\Delta t \le \tau_c/2$ and the observation is unmixed or admitted; otherwise
it is hub-periphery dependence. Hidden shared inputs inflate NAS and do not in
general deflate it, so under a partial or empty declaration PRESENT certifies
dependence only, while ABSENT keeps its meaning. It does not tell a loop from
a feed-forward relay and does not see zero-mean pattern broadcast.

1. **Gates** (each UNDEFINED, never ABSENT): the declarations
   (`NOT_DEFINED:<reason>`, as `compute_NAS`); resolvability
   (`SAMPLING_UNRESOLVED` iff $\Delta t > \tau_c/2$); the observation
   (`OBSERVATION_MIXED_NOT_ADMITTED` for mixed sensor or source-estimate data
   without an admitting registry entry); the parameter budget and rank
   (`INSUFFICIENT_TIMEPOINTS`: $T - l_{\max} \ge 10\,n_{\mathrm{par}}$ with
   $n_{\mathrm{par}} = 1 + \mathrm{rank}(U) + |L|\sum_B d_B$, and
   $\sum_B d_B$ at most 0.8 times the numerical rank of the observation); a
   degenerate null.
2. **Lags.** $L$ = the sorted unique rounded values of
   geomspace$(1, l_{\max}, \min(4, l_{\max}))$ with
   $l_{\max} = \lceil \tau_l/\Delta t\rceil$. The lag time scale $\tau_l$
   equals $\tau_c$ unless a sensitivity form sets it to 0.05 s or 0.2 s; it
   changes the lags and the input basis, never the resolvability gate.
3. **Blocks.** The primary representation is the mean of the block's z-scored
   nodes (`block_representation="block_mean"`); the secondary form
   `"all_units"` uses every node of a small block or its leading principal
   components.
4. **Statistic** (conditional Geweke causalities, nats). With $Z_j$ the pasts
   of the other blocks and the input basis $U$,
   $$
   \mathrm{receive}_j = \tfrac12\ln\det\Sigma(Y_H \mid Y_H^L, Z_j)
   - \tfrac12\ln\det\Sigma(Y_H \mid Y_H^L, Z_j, Y_j^L),
   $$
   $\mathrm{return}_j$ likewise with hub and block exchanged; $R$ and $B$ are
   the means over the periphery blocks. The basis is rank-deficient by
   construction (lagged and filtered copies are collinear), so it enters
   through its orthonormal span and every regression uses a rank-revealing
   solve.
5. **Null** `block_circular_shift` (as `compute_NAS`): the hub block is
   circularly shifted against the periphery and $U$ by at least 10% of the
   run, $K = 19$; every regression, including the projection on $U$, is
   refitted on the shifted data (shift, then project).
6. **Directions.** `receive` and `return` are separate members of the
   directional principle, each with its own estimate, null, SE and anchor
   (reference keys `NAS:receive`, `NAS:return`); the TOST rule combines them
   (section 9.3).
7. **SE.** Delete-a-group jackknife with $G$ groups, contiguous or
   interleaved (default `jackknife_contiguous_10`; `se_df = G - 1`).

On the capacity geometry (five leading components of the hub and of the
pooled periphery, no inputs) the same code reproduces the statistic and null
of `compute_NAS(mode="capacity")` to 1e-12. Descriptors (the capacity
statistic, the conditioning delta, zero-lag coupling, the BIC lag order,
metastability) are reported and never gate. The statistic is quadratic in a
weak coupling amplitude (section 9.7).

## 5. IIM

### 5.1 Definition

From a node x time series, `compute_IIM`:

1. selects a subsystem (`node_selection`: highest temporal variance, or
   `index`; `node_indices` overrides; only `bearer_nodes` are eligible);
2. discretises each node into `bins` levels (default 3) and estimates the
   transition probability matrix (TPM) at lag `lag_trs` samples;
3. computes the integration mass $\Psi$ of the intact TPM and $\Psi^\kappa$ of
   every system cut $\kappa$.

$$
\Psi = \sum_{M} \frac{1}{|M|}\sum_{m} w(m)\,\min\big(\varphi_e(m), \varphi_c(m)\big),
\qquad
\Delta\Psi = \Psi - \max_{\kappa}\Psi^{\kappa}\ \text{(bits)},
$$

where $\varphi_e/\varphi_c$ are the maximum over purviews of the minimum
Jensen-Shannon divergence (bits) between the effect/cause repertoire and its
partitioned product over mechanism x purview bipartitions, searching both
pairings $(M_1\!\to\!Z_1, M_2\!\to\!Z_2)$ and $(M_1\!\to\!Z_2, M_2\!\to\!Z_1)$.
Mechanisms or purviews of size 1 have no admissible partition and contribute 0.
$w(m)$ are the empirical mechanism-state frequencies. The minimum-information
partition (MIP) is the cut that preserves the most $\Psi$.

Reported values:

- `Delta_Psi`: the integration mass in bits (the null-calibrated statistic).
- `IIM_raw` $= \Delta\Psi / (\Psi + \epsilon)$, the scale-free ratio, and
  `canonical` $= \mathrm{clip}(\mathrm{IIM\_raw}, 0, 1)$ (the uncalibrated value
  of the `IIM` column).
- With null calibration: `canonical_calibrated` $= \max(\Delta\Psi - \mathbb{E}[\Delta\Psi_{\mathrm{null}}], 0)$
  in bits, which becomes the returned `value` (section 7).

Why the ratio is not the calibrated statistic: for independent processes the
estimated $\Psi$ is pure finite-sample bias, and cutting (which averages TPM
rows) removes more of that bias than the intact TPM carries. The ratio then
stays $O(1)$ (0.57 to 0.80 over three seeds for three independent binary units
sampled for $T = 30000$ steps, whose exact $\Delta\Psi$ is 0) and does not
increase with coupling, while $\Delta\Psi \to 0$.

### 5.2 TPM estimators (`tpm_estimator`)

| Estimator | Form |
|---|---|
| `node_shrinkage` (default) | state-by-node TPM (next-step node states conditionally independent given the current state, as in IIT); James-Stein shrinkage of each row towards the node's own transition $p(x_i' \mid x_i)$ |
| `node_laplace` | state-by-node, Laplace smoothing per node (`tpm_alpha`) |
| `per_unit` | state-by-node, unregularised per-unit maximum likelihood; unobserved rows fall back to the unit's own transition |
| `joint_laplace` | legacy joint $K^n \times K^n$ Laplace estimate (the estimate of earlier releases; turns noise into spurious non-factorisable repertoires) |

State budget: `bins ** nodes <= max_state_space` (default 1500) is enforced by
`state_budget_policy` (`reduce_bins_first` default, `reduce_nodes_first`, or
`error`, which makes the result undefined). Every adjustment is logged and
returned (`budget_adjustments`, `bins_requested`, `bins_used`). With the
pipeline's defaults (3 bins, no node cap) a large parcellation ends at 10 nodes
and 2 bins.

### 5.3 Cuts (`cut_mode`)

- `bidirectional` (default, legacy): every unordered bipartition $(A, B)$ with
  both directions replaced by noise: $T_{\mathrm{cut}} = p_A(s'_A \mid s_A)\,p_B(s'_B \mid s_B)$,
  where $p_A$ is the $A$-marginal of the joint next state averaged over $s_B$.
- `directional`: every ordered bipartition $(A, B)$ severs only $A \to B$:
  $T_{\mathrm{cut}}(s' \mid s) = \prod_{i\in A} p_i(s'_i \mid s)\prod_{j\in B}\bar p_j(s'_j \mid s_B)$,
  with $\bar p_j$ = $p_j$ averaged uniformly over the current state of $A$ (each
  receiving unit gets independent noise, the IIT unidirectional cut). Twice as
  many cuts; requires a state-by-node estimator. Feedforward systems are exactly
  reducible under directional cuts ($\Delta\Psi = 0$), not under bidirectional
  cuts. The converse does not hold (a strongly connected XOR loop also gives 0).
- `n_parts=None` evaluates every cut; an integer samples that many cuts without
  replacement (seed `rng`). `partition_mode="balanced"` restricts to near-equal
  cuts.

### 5.4 Exact IIM of a known TPM

`compute_IIM_from_tpm(tpm, state_weights=None, base=2, cut_mode=...)` runs the
same $\Psi$ machinery on a given state-by-state TPM (rows sum to 1 within
`row_sum_tol`; no renormalisation). State order: row/column $k$ is the
big-endian base-$K$ code of the unit states (unit 0 most significant;
`iim_state_table(n, K)`). Default state weights: the long-run distribution of
the chain from a uniform start (`iim_stationary_distribution`). The value is
$\max(\Delta\Psi, 0)$ in bits, the population counterpart of
`canonical_calibrated`; for independent units it is exactly 0. Directional cuts
require a state-by-node TPM (checked to `state_by_node_tol`). Helpers:
`iim_tpm_from_unit_probabilities`, `iim_state_table`,
`iim_stationary_distribution`.

### 5.5 Kernels and engineering

- `psi_kernel="numba"`: host reference kernel with process pools and a
  disk-backed SQLite kernel cache (removed after use). `psi_kernel="xp"`: the
  array-module kernel `impact_pipeline.iim_xp` (NumPy on the CPU, CuPy on a
  GPU/APU). `"auto"` (default) uses `IMPACT_IIM_PSI_KERNEL` if set, else `xp`
  exactly on accelerator backends. The kernels agree to about 1e-15 (tested to
  1e-10); `hardware_selftest` checks it on the device (`iim_psi_xp_parity`).
- Device memory is bounded by `IMPACT_IIM_XP_MAX_ELEMENTS` (default 2^23 float64
  elements per temporary array) and `IMPACT_IIM_XP_CACHE_ELEMENTS` (2^27).
- Checkpoints (`checkpoint_path`) and cache signatures include every parameter
  that changes the result (estimator, `tpm_alpha`, `max_state_space`, bins,
  lag, selected nodes, budget policy, cut mode when directional) and the
  algorithm version `IIM_ALGORITHM_VERSION = "iim-2026.09"`. Checkpoints of
  another version are not resumed.
- Undefined IIM is explicit (`IIM_defined=False`, `IIM_undefined_reason`).
- On HLRS Hunter the same computation is distributed over PBS array jobs
  (`docs/HLRS_HUNTER_RUNBOOK.md`); its results equal `compute_IIM` to 1e-9,
  including the surrogate calibration.

### 5.6 Macro IIM (`iim-macro-2026.10`)

`estimators.iim_macro.compute_iim_macro` measures the part of the one-step
transition structure of 2 to 6 declared macro variables that is lost under
the least destructive unidirectional cut, from a state-by-node
`node_shrinkage` TPM of median-binarised macro signals, conditional on
declared recorded drivers, in bits, in excess of an independence null. It is
identified as loop integration only under causal sufficiency relative to the
recorded variables, adequate occupancy, sampling that resolves the
interaction time scale and an approximately Markov, stationary macro state;
under an unrecorded common driver it measures observational dependence.
Degenerate TPMs are UNDEFINED, never ABSENT. The $\Psi$ functional, the TPM
estimator and the cut TPMs are those of `compute_IIM` (section 5.1): without
strata, input basis or pre-processing the estimate and every null draw equal
`compute_IIM` at 2 bins with `node_shrinkage`.

1. **Macro signals**: the mean over the nodes of each declared macro node.
   The zero-lag correlation matrix of the macro signals needs
   $\lambda_{\min}/\lambda_{\max} \ge 10^{-6}$ (`MACRO_RANK_DEFICIENT`
   otherwise). Sensor substrates are orthogonalised symmetrically (ZCA).
2. **Conditioning** on declared recorded drivers, never on the hub or any
   other variable caused by the macro nodes. Binary drivers with at most four
   joint states are stratified (one TPM per stratum, weighted by its share of
   transition pairs); every other driver is residualised on the input basis
   (section 14.2) with the IIM lag set $\{0, 1, \dots, l_{\max}\}$.
3. **Occupancy**: in every stratum all $2^n$ macro states are visited and the
   rarest row has at least $N_{\min} = 25$ transition pairs
   (`INSUFFICIENT_OCCUPANCY` otherwise).
4. **Statistic** $\Delta\Psi_K = \Psi(T) - \max_\kappa \Psi(T^\kappa)$ over the
   ordered bipartitions (`directional`, the primary form) or the unordered
   ones (`bidirectional`, reported with its own anchor), computed with the
   `iim_xp` kernel on NumPy.
5. **Null** ($K = 39$ on the benchmark systems, 19 in the family-B cells;
   fewer than 19 finite draws: `NO_NULL_CALIBRATION`): circular shifts of
   every macro signal but the first (minimum shift
   $\max(\mathrm{lag} + 1, \lceil 0.1\,T\rceil)$), with every conditioning
   regression refitted on the surrogate (shift, then project); with strata, a
   stratified pair rotation that keeps each node's own transitions within
   every stratum. The rank p-value is
   $p_{\mathrm{ind}} = (1 + \#\{m^*_b \ge m\})/(K + 1)$; a protocol's IIM rank
   gate uses it (section 9.3).
6. **SE**: circular block bootstrap of the analysed series (blocks of
   $\lceil 0.1\,T\rceil$ samples, $B = 50$,
   `circular_block_bootstrap_10pct_B50`) with a declared `se_df` (12, the
   block-count approximation, or 9; the benchmark protocols declare 9), never
   $B - 1$.
7. **Observation gate**: sensor or source-estimate data are
   `OBSERVATION_MIXED_NOT_ADMITTED` unless an admitting registry entry exists;
   the rank and occupancy gates come first.

Exact path: `exact_delta_psi` evaluates a known TPM with its stationary state
weights (equal to `compute_IIM_from_tpm`, section 5.4). Exact components carry
the SE method `exact` and are decided on $c$ alone. $\Delta\Psi$ is quadratic
in a weak coupling amplitude (section 9.7).

## 6. SRPI

### 6.1 Default mode (`mode="legacy"`): self vs non-self events

For each event $e$ with pre-window $W_{\mathrm{pre}}$, response lag $L$ and
response window $W_{\mathrm{resp}}$ (all in seconds, converted to samples):

$$
p_e = \frac{1}{W_{\mathrm{pre}}}\sum_{\tau=e-W_{\mathrm{pre}}}^{e-1}x(\tau),\quad
r_e = \frac{1}{W_{\mathrm{resp}}}\sum_{\tau=e+L}^{e+L+W_{\mathrm{resp}}-1}x(\tau),\quad
d_e = r_e - p_e .
$$

The pre-window $[e - W_{\mathrm{pre}}, e)$ lies strictly before the onset. Four
components:

$$
R = \left[\frac{\Gamma_s - \Gamma_n}{\Gamma_s + \Gamma_n + \epsilon}\right]_+,\quad
\Gamma_c = \langle\lVert d_e\rVert_2\rangle_{e\in\mathcal E_c},
\qquad
S = \big[2(\mathrm{AUC}_{\mathrm{cv}} - 0.5)\big]_+,
$$

$$
K = \left[\frac{\bar\rho_s - \bar\rho_n}{2}\right]_+,\quad
\bar\rho_c = \big\langle\mathrm{corr}(d_{c,i}, d_{c,j})\big\rangle_{i<j},
\qquad
I = \big[|\mathrm{corr}(u_s, m_s)| - |\mathrm{corr}(u_n, m_n)|\big]_+ .
$$

- $S$ (separability) is out-of-sample: a shrinkage-LDA (Ledoit-Wolf intensity,
  floor `covariance_ridge/(1+covariance_ridge)`) is fitted on time-blocked,
  class-stratified training folds (`separability_cv_folds`, default 5) and
  scored on held-out events; $S = 0$ at chance. (The in-sample Mahalanobis
  separability saturates at 1 when nodes outnumber events.)
- $I$ uses absolute correlations: $u_c = P_c v_1$ projects the pre-event states
  on the first right-singular vector of the pooled pre-event states, whose sign
  is arbitrary, so only the strength of the state-response coupling is
  identifiable. $m_c = (\lVert d_e\rVert_2)_{e\in\mathcal E_c}$.
- `directional=True` maps the signed reactivity contrast from $[-1, 1]$ to
  $[0, 1]$ instead of clipping it.

Every component is attenuated by $r_N = 1 - e^{-N_{\mathrm{eff}}/\tau_N}$ with
$N_{\mathrm{eff}} = \min(|\mathcal E_s|, |\mathcal E_n|)$ (`sample_reliability_tau`,
default 4), and

$$
\mathrm{SRPI} = \left(\tilde R^{w_R}\tilde S^{w_S}\tilde K^{w_K}\tilde I^{w_I}\right)^{1/(w_R+w_S+w_K+w_I)} \in [0, 1],
$$

weights $(0.35, 0.25, 0.20, 0.20)$. A weighted component equal to 0 gives
SRPI = 0. SRPI is undefined (NaN) with fewer than `min_events_per_class` events
per class after windowing; `min_events_per_class` must be at least 3 in every
mode (the cross-validated separability needs 2 training events per class).
Self and non-self labels are parsed from `events.tsv` by whole-token patterns
(`impact_pipeline.event_parsing`, shared by readiness checks and computation).

### 6.2 `mode="agency"`: self-caused vs yoked other-caused events

Self-caused events are contrasted with yoked, stimulus-identical, phase-matched
other-caused replays (`agency_events`: `onset`, `trial_type` in
`self_caused`/`other_caused`, `yoked_to`, `phase_bin`, optional `event_id` and
stimulus column). The contract is checked by
`event_parsing.validate_srpi_agency_contract`; a violation makes SRPI undefined
(`agency_contract_violation:<code>`).

- The linear effect of the pre-event state (`agency_pre_components` principal
  components, default 5) is removed from the response changes by a label-free
  regression refitted inside every CV training fold; the grand-mean response is
  kept, so gain differences such as sensory attenuation stay visible.
- Terms: two-sided reactivity $|\Gamma_s - \Gamma_n|/(\Gamma_s + \Gamma_n)$;
  separability $2\,\mathrm{AUC}_{\mathrm{cv}} - 1$ of a shrinkage-LDA on
  `agency_components` (default 10) PCA-reduced changes; stability
  $|\rho_s - \rho_n|/2$; coupling $|\mathrm{corr}_s - \mathrm{corr}_n|/2$.
- Each term is chance-corrected by the mean of a label-permutation null that
  permutes labels within each yoked cluster (a self-caused event and its
  replays; `agency_null_permutations`, default 200; seed `agency_random_state`).
- SRPI is the weighted arithmetic mean of the chance-corrected terms (signed,
  no hard zero, no reliability attenuation).
- Limitation: a self/other difference that is collinear with a pre-event state
  difference (for example motor preparation that always precedes self-caused
  events) is removed with it, so SRPI-agency is about 0 in that regime.
  `agency_pre_components=0` keeps such effects but then also credits pre-event
  confounds.

### 6.3 SRPI on the benchmark

SRPI has no separate variant. The benchmark scores `compute_SRPI(mode="agency")`
(`srpi-2026.09`) through the in-memory path (`bench.export.run_in_memory`)
with a delete-a-group jackknife over 10 groups of yoked pairs
(`jackknife_pairs_10`, `se_df = 9`) and judges it by the TOST rule.

## 7. Null calibration

Every estimator can be compared with a declared null family: the same estimator,
with the same configuration, on surrogate data. The fields are
`<P>_null_n`, `<P>_null_mean`, `<P>_null_sd`, `<P>_z`, `<P>_null_p` (one-sided,
$(1 + \#\{\mathrm{null} \ge \mathrm{obs}\})/(n + 1)$), `<P>_excess`
(observed minus null mean) and `<P>_calibrated` (excess floored at 0; the signed
excess when `clip_negative=False`). For IIM the null moments are also given as
`Delta_Psi_null_mean`/`Delta_Psi_null_sd` (bits), and `canonical_calibrated` =
`IIM_calibrated` is the floored excess. A null that cannot be built gives NaN
with `<P>_null_undefined_reason`.

There are two sets of surrogate families with different names.

The estimators' own calibration (`null_method` of `compute_PDI`, `compute_NAS`
and `compute_IIM`; `mpc_metrics.SURROGATE_METHODS`):

| Family | Keeps | Destroys |
|---|---|---|
| `circular_shift` | each node's marginal and autocorrelation (node 0 fixed, others shifted independently by at least 10% of the run) | cross-node alignment |
| `phase_randomize` | marginals, auto- and (approximately) cross-spectra; multivariate Fourier surrogate with shared phases, amplitude-adjusted by 10 IAAFT iterations | nonlinear / non-Gaussian structure |
| `phase_randomize_independent` | each node's spectrum and marginal (independent phases, amplitude-adjusted as above) | cross-node coupling |
| `shuffle` | marginals | all temporal and cross-node structure |

The generic driver `nulls.component_null` (`nulls.SURROGATE_KINDS`; the
pipeline uses it for RAM and SRPI):

| Kind | Keeps | Destroys |
|---|---|---|
| `circular_shift` | as above | cross-node alignment |
| `phase` | amplitude spectra and all cross-spectra exactly (shared phases); marginals become Gaussian | nonlinear / non-Gaussian structure |
| `phase_independent` | each node's amplitude spectrum | cross-node structure |
| `iaaft`, `iaaft_independent` | each node's marginal exactly and its spectrum approximately (10 iterations by default, shared or independent initial phases) | nonlinear structure (and, for `iaaft_independent`, cross-node structure) |
| `onset_jitter` (events) | the event design; with `common=True` the whole event train is shifted rigidly | alignment of events with the recording |
| `label_permutation` (events) | class counts (stratified by `phase_bin` when given) | the self/non-self labelling |

`nulls` also accepts the aliases `phase_randomize` (= `phase`, not
amplitude-adjusted) and `phase_randomize_independent` (= `phase_independent`).

Pipeline defaults when `--null-surrogates K > 0`
(`synergy_ci.MPC_NULL_KINDS_DEFAULT`):

| Component | Null family | Implementation |
|---|---|---|
| RAM | rigid circular shift of the whole event train, at least 10% of the run | `nulls.component_null(kind="onset_jitter", common=True)` |
| PDI | multivariate amplitude-adjusted phase randomisation, primary endpoint | `compute_PDI(null_surrogates=K)` |
| NAS | independent circular shift per node | `compute_NAS(null_surrogates=K)` |
| IIM | independent circular shift per node, minimum shift $\max(\mathrm{lag}+1, \lceil 0.1\,T\rceil)$; statistic $\Delta\Psi$ | `compute_IIM(null_surrogates=K)` |
| SRPI | self/non-self label permutation (counts kept) | `nulls.component_null(kind="label_permutation")` |

- Per-run seeds are derived (SHA-256) from `null_seed` (default 0), the
  component name and the run path `<subject>/<session>/<condition>/<file>`, so
  results do not depend on where the data live.
- The RAM null cannot separate event locking from periodicity in strictly
  periodic designs: a shift by a multiple of the period realigns the train with
  itself. It is informative for jittered designs.
- With K > 0 the metric columns `RAM`..`SRPI` hold the calibrated values
  (excess over the null mean, floored at 0; NaN when the null could not be
  computed, never the raw value). The raw estimates are in `<P>_estimate`.
  With K = 0 (default) all metric and CI values are unchanged from the
  uncalibrated estimators.
- The construct-revision modes calibrate internally, whatever K is (K = 0
  selects their default size): PDI `repertoire` (state-count null,
  `circular_shift` by default; labelled: `block_label_permutation`; 19 by
  default), PDI `surrogate_excess` (multivariate Fourier, 19), NAS `capacity`
  (`block_circular_shift`, 19) and SRPI `agency` (`yoked_label_permutation`,
  200 permutations).
- Under a protocol the declared null families are checked against the family
  each component used (`NULL_FAMILY_MISMATCH`, section 8.4).

## 8. Evidence layer: MPC profile, verdict and degree

`impact_pipeline.evidence` turns calibrated component evidence into three
outputs:

- **MPC profile**: the five null-anchored components with their null
  calibration, construct-scale values, intervals and statuses.
- **MPC verdict**: three-valued, `EXCLUDED`, `MPC_CONSISTENT` or
  `UNDETERMINED`, with stable reason codes.
- **MPC degree**: a reference-relative evidence summary, computed only for
  `MPC_CONSISTENT` rows.

Sections 8.2 to 8.6 describe the interval rule, which judges every protocol
without a `status_rule` block and is the rule of the empirical pipeline. A
protocol with the block is judged by the TOST rule of section 9.

### 8.1 Stance: necessity only

The MPC stance treats the principles of a declared necessity set $N$ (default
all five) as **necessary** conditions. Necessary conditions license exclusion
only:

- `EXCLUDED`: some principle in $N$ is credibly **absent**. The inference rests
  on two auxiliary premises: that the principles in $N$ are necessary, and that
  the estimator is a valid measure in its registered domain.
- `MPC_CONSISTENT`: every principle in $N$ is credibly **present**. The MPC
  stance does not exclude consciousness. This is **not** an attribution of
  consciousness: no sufficiency claim is made.
- `UNDETERMINED`: otherwise, with reason codes that say why.

### 8.2 Component status

Each piece of evidence (`ComponentEvidence`: estimate, null moments and size,
sampling SE and its degrees of freedom, reference anchor, channel, bearer,
protocol, substrate, grain, estimator, null family, node set) gets a status
PRESENT, ABSENT or UNDEFINED on a two-anchor **construct scale**
(`evidence.component_assessment`):

$$
c = \frac{m - \nu}{\rho - \nu},
$$

where $m$ is the estimate, $\nu$ the mean of the declared null family and
$\rho$ the reference anchor. $c = 0$ at the null and $c = 1$ at the reference.
The anchor is given on the estimate scale (`reference_scale="estimate"`) or as
the reference excess $\rho - \nu$ directly (`"excess"`; the pipeline's cohort
reference and the bench reference use this scale). If $\rho \le \nu$ or an
anchor is not finite, the component is UNDEFINED (`INVALID_ANCHORS`).

- **SE of $c$.** By the delta method with independent parts,
  $$
  \mathrm{se}_c^2 = \frac{\mathrm{se}_m^2 + d_\nu^2\,\sigma_{\mathrm{null}}^2/K
  + c^2\,\mathrm{se}_\rho^2}{(\rho - \nu)^2},
  $$
  with $\mathrm{se}_m$ the sampling SE of the estimate (moving-block bootstrap
  in the pipeline, delete-a-group jackknife in the bench),
  $\sigma_{\mathrm{null}}/\sqrt K$ the Monte-Carlo error of the null mean from
  $K$ surrogates ($K = 0$ declares an analytic null mean), $d_\nu = c - 1$ on
  the estimate scale ($-1$ on the excess scale) and $\mathrm{se}_\rho$ the SE
  of the anchor when available. An empirical estimate without a positive
  sampling SE is UNDEFINED (`NO_SAMPLING_SE`); exact computations on a known
  TPM (`exact=True`) are exempt. $c$ is reported even then.
- **Quantile.** $q = z_{1-\alpha}$, or, when the sampling SE comes from few
  replicates and declares its degrees of freedom `se_df` (jackknife: groups
  $- 1$; bootstrap: valid replicates $- 1$), the Student quantile
  $t_{1-\alpha,\nu}$ with the Welch-Satterthwaite degrees of freedom
  $\nu = \mathrm{se\_df}\,(\mathrm{se}_c^2/s_m^2)^2$ ($s_m = \mathrm{se}_m/(\rho-\nu)$;
  the null and reference parts count as known), so $\nu = \mathrm{se\_df}$
  when the sampling SE dominates.
- With smallest effects of interest $z_j$ (presence) and $\delta_j$ (absence)
  on the $c$ scale, declared in the protocol, $\delta_j \le z_j$:
  - **PRESENT** if the one-sided $(1-\alpha)$ lower bound exceeds $z_j$:
    $c - q\,\mathrm{se}_c > z_j$;
  - **ABSENT** if the one-sided $(1-\alpha)$ upper bound is below $\delta_j$:
    $c + q\,\mathrm{se}_c < \delta_j$. This includes estimates credibly below
    the null: there is no asymmetry that favours non-falsification;
  - **UNDEFINED** otherwise (`INCONCLUSIVE`).
- Protocol values (all shipped protocols): $z_j = 0.25$, $\delta_j = 0.10$,
  $\alpha = 0.05$.
- UNDEFINED reasons, in order: not defined (the estimator's reason), a
  non-finite estimate, `NO_NULL_CALIBRATION` (no null mean),
  `DEGENERATE_NULL` ($K > 0$ without a finite null SD $\ge 0$),
  `INVALID_ANCHORS`, `INVALID_SE` (negative SE or anchor SE, or
  `se_df` $\le 0$) and `NO_SAMPLING_SE`.
- Reported per component: $c$, $\mathrm{se}_c$ and its parts, the degrees of
  freedom and quantile, the lower and upper bounds and the margins to each
  cutoff (`margin_present = lower - z`, `margin_absent = delta - upper`).

### 8.3 Combining evidence

- **Channels** of one principle (for example RAM `behavioural_feedback` and
  `covert_neural`) are combined by strong-Kleene OR: a principle is PRESENT if
  any channel is PRESENT, ABSENT only if every channel is ABSENT.
- **Principles** of $N$ are combined by strong-Kleene AND: T → `MPC_CONSISTENT`,
  F → `EXCLUDED`, U → `UNDETERMINED`. Any ABSENT principle vetoes (no
  compensation by the others).
- **Protocol** (`evidence.Protocol`; fields and hash in section 10). A
  declared channel without an evidence item counts as UNDEFINED
  (`MISSING_CHANNEL:<P>:<channel>`), so a principle with a declared channel
  that is not measured (or not implemented) is never ABSENT; items of
  undeclared channels are ignored (`ignored_channels`); principles without a
  declaration use the channels of their evidence. Evidence whose null family
  differs from the declared one is UNDEFINED (`NULL_FAMILY_MISMATCH`), and
  evidence carrying another protocol's id makes the verdict UNDETERMINED
  (`PROTOCOL_MISMATCH`). A protocol with a `status_rule` block is judged by
  the TOST rule (section 9) instead; the empirical pipeline refuses it.
- **Single-source constraint** (source rule `single_source`). All gated
  evidence must come from one declared bearer (`BEARER_MISMATCH` otherwise)
  and one protocol (`PROTOCOL_MISMATCH`). When the components come from
  different node sets, `evidence.joint_dependence` must show joint dependence
  above null: the Gaussian total correlation (bits) of the lag-embedded first
  principal components of the node sets, against surrogates that circularly
  shift each set independently (between-set dependence destroyed, within-set
  structure kept); overlapping sets are tested on their disjoint atoms;
  criterion `total` (default) or `each` (every block with the rest,
  Holm-corrected). Otherwise the verdict is UNDETERMINED with
  `SOURCE_INCOHERENT` (`SOURCE_INCOHERENT:UNTESTED` when no test covers
  exactly the node sets of the evidence; `SOURCE_INCOHERENT:INSUFFICIENT_SURROGATES`
  when the test had too few surrogates to reach its level). In the pipeline
  the test uses `--null-surrogates` as its surrogate budget.
  `evidence.bearer_coherence` is the older pairwise lagged-Gaussian-MI
  diagnostic (reason `BEARER_MISMATCH:COHERENCE`).
- **Applicability registry** (`registry.ApplicabilityRegistry`,
  `--applicability-registry`, schema `impact-mpc-registry/1`). Evidence from
  an estimator version that is not
  validated for its principle, substrate, grain and regime is UNDEFINED
  (`ESTIMATOR_NOT_VALIDATED:<P>:<estimator>`):

  ```json
  {"schema": "impact-mpc-registry/1", "version": "2026-09",
   "criteria": {"alpha": 0.05, "false_present_tolerance": 0.02},
   "entries": [
     {"estimator": "compute_NAS:capacity", "version": "nas-2026.09",
      "substrate": "eeg_like_forward", "grain": "*",
      "regime": {"T_min": 2000, "nodes_min": 16, "nodes_max": 128},
      "evidence": {"run_id": "bench-run-id", "null_false_present_rate": 0.01,
                   "recovery_slope": 0.8, "cross_talk": 0.03},
      "status": "validated"}]}
  ```

  Entry criteria for `status: validated`: a pinned `version`, a validation
  substrate that is not empirical (human EEG/fMRI need the forward-modelled
  `eeg_like_forward` / `bold_like_forward`; queries on `eeg` or `fmri` are
  matched against them), and benchmark `evidence` with a null false-PRESENT
  rate at most `alpha + false_present_tolerance` (one-sided by default: a
  calibrated rule has a rate far below alpha; `"two_sided": true` enforces
  `alpha ± tolerance`), a positive recovery slope and, when `cross_talk_max` is
  declared, cross-talk within it. Entries that fail are rejected when the file
  is loaded. String fields match shell wildcards (case-insensitive) and may be
  lists; `regime` values are a scalar (equality), a list (membership),
  `{"min", "max"}` or the named bounds `T_min`, `nodes_min`, `nodes_max`,
  `snr_min`. The pipeline queries with substrate = modality, grain = atlas and
  the regime of the component (bearer size; for IIM the scored subsystem,
  bins, series length and sample interval). Without a registry, determinate
  verdicts are possible for unvalidated estimators (the registry is opt-in;
  no registry file is shipped). The same registry also holds the admission
  entries of the benchmark's forward-model arms, which admit each status
  direction separately (`admitted_for_present` / `admitted_for_absent`);
  for the interval rule such an entry validates an estimator only when it
  admits both directions.

### 8.4 Reason codes

Stable strings, `;`-joined in the `MPC_reason` column:

| Code | Meaning |
|---|---|
| `MISSING:<P>` | no evidence for a principle of $N$ (and no declared channel) |
| `MISSING_CHANNEL:<P>:<channel>` | a declared channel has no evidence item |
| `NO_NULL_CALIBRATION:<P>` | a defined estimate without a null family (e.g. `--null-surrogates 0` in a legacy mode) |
| `NO_SAMPLING_SE:<P>` | an empirical estimate without a sampling SE (e.g. `--bootstrap-se 0`) |
| `INVALID_ANCHORS:<P>` | the reference is not above the null mean (or not finite) |
| `INCONCLUSIVE:<P>` | neither credibly present nor credibly absent |
| `UNDEFINED:<P>:<reason>` | the estimator, the null or the SE is undefined: the estimator's reason, `DEGENERATE_NULL`, `INVALID_SE`, `NULL_FAMILY_MISMATCH:<declared>/<used>`, `null_undefined:<reason>`, `iim_option_mismatch:<option>`, `NO_DECLARED_WORKSPACE` (NAS `mode="capacity"` without a declared hub; the estimator is not called) |
| `NOT_IMPLEMENTED:<P>:<channel>` | the evidence channel is not implemented |
| `ESTIMATOR_NOT_VALIDATED:<P>:<estimator>` | not validated for this substrate, grain or regime |
| `ABSENT:<P>` | every channel of $P$ is credibly absent (veto) |
| `BEARER_MISMATCH`, `PROTOCOL_MISMATCH` | evidence from more than one bearer or protocol |
| `SOURCE_INCOHERENT[:<detail>]`, `BEARER_MISMATCH:COHERENCE` | the single-source constraint failed or was not tested (`UNTESTED`, `INSUFFICIENT_SURROGATES`) |

Decomposition rule: the verdict is recoverable from the reasons alone. No reasons
⇔ `MPC_CONSISTENT`; a bearer, protocol or source code ⇒ `UNDETERMINED`;
otherwise any `ABSENT` ⇒ `EXCLUDED`; else `UNDETERMINED`. The reasons of a
verdict over $N$ are the union of the single-principle reasons over $P \in N$
plus the global codes. The dashboard reports disagreeing runs of one
subject-session as `UNDETERMINED` with `RUN_VERDICTS_DISAGREE:<verdicts>` (and
`RUN_VERDICT_MISSING` when a run has no verdict).

Properties verified by the test suite on seeded random configurations:
missingness safety (making a component undefined never creates a determinate
verdict that differs from the original), resolving an undefined component never
reverses a determinate verdict, the verdict is determinate iff all completions
of the undefined components agree, veto (any ABSENT in $N$ ⇒ `EXCLUDED`),
permutation symmetry over principles, channel disjunction, reason-code
decomposability, a declared channel without an item is never ABSENT, and a
missing SE gives UNDEFINED.

### 8.5 MPC degree

For `MPC_CONSISTENT` rows only (NaN otherwise), the construct-scale values
$c_j$ of $N$ (`<P>_c`, 0 = null mean, 1 = reference anchor of the protocol)
are summarised by a capped weighted power mean:

$$
\mathrm{degree} = \Big(\sum_j w_j \min(\max(c_j, 0), \mathrm{cap})^{p}\Big)^{1/p},
$$

with $p = 0$ (geometric mean; the pipeline default, `MPC_DEGREE_P`),
$p = -\infty$ (weakest link), $p = 1$ (arithmetic mean), cap 1
(`MPC_DEGREE_CAP`) and equal weights by default
(`synergy_ci.assemble_mpc_degree(df, weights, p, cap)`). The reference is the
protocol's (by default the cohort high-state mean excess); `--ci-reference`
affects only the legacy CI. `evidence.degree_interval` gives delta-method or
bootstrap intervals; `evidence.verdict_stability` gives the flip rate and
modal verdict across repeated evaluations (a tie for the top count gives
`UNDETERMINED`).

The MPC degree is a reference-relative evidence summary. It is **not** a level
of consciousness and not an anaesthesia-depth index.

### 8.6 Pipeline wiring

`compute_synergy_ci` (CLI: `--protocol`, `--null-surrogates`, `--bootstrap-se`,
`--bootstrap-block-len`, `--necessity-set`, `--applicability-registry`) builds
one `ComponentEvidence` per run, computed principle and channel, judges all
runs once the reference anchors are known and adds the verdict columns
(section 13).

- **Protocol.** `--protocol file.json` declares $N$, channels, cutoffs,
  $\alpha$, null families, reference, source rule, estimator modes and bearer
  nodes; its hash is `MPC_protocol_hash`. `run_pipeline.py` uses
  `protocols/mpc_default.json` for empirical data when `--protocol` is not
  given (and no `--necessity-set`); it logs the protocol's source and hash.
  Under it RAM keeps its unimplemented `perturbational` and `endogenous` channels (never
  ABSENT) and NAS capacity has no hub (UNDEFINED, `NO_DECLARED_WORKSPACE`)
  unless a derived protocol declares one (`protocols/examples/`). The opt-in
  `protocols/mpc_behavioural_ram.json` (RAM on its behavioural channel
  only; behavioural-RAM results, see section 2.3) is used only with
  `--protocol`. Without a protocol (`--protocol none`, and always for dummy
  data) one is built from `--necessity-set`, the null kinds, the mode keys of
  the params dicts (`synergy_ci.MPC_MODE_KEYS`) and the cohort reference;
  conflicting settings raise an error. External
  reference anchors from a held-out subset of participants:
  `scripts/compute_empirical_reference.py` (keys `P`, or `P:channel` for a
  principle with several declared channels: `RAM:default` under the default, from
  per-channel evidence, which its `--data-dir` mode records; the step-2
  `<P>_*` columns describe only the deciding channel).
- **Modes.** The protocol's `estimators` select the construct revisions (RAM
  `update="prediction_error"`, PDI `mode="repertoire"` with its `repertoire_*`
  options, NAS `mode="capacity"`, IIM `cut_mode`/`tpm_estimator`, SRPI
  `mode="agency"`). A declared fallback (`update_fallback` for RAM,
  `mode_fallback` for SRPI) is used for runs that lack the primary mode's
  inputs (a choice/reward log; self_caused/other_caused events); the mode used
  is in `<P>_estimator` (`compute_<P>:<mode>@<version>`) and the reason in
  `<P>_mode_reason` (e.g. `no_agency_events:agency->legacy`). Without a
  declared fallback the primary mode is kept (and is undefined where its
  inputs are missing). RAM channels declared by the protocol are computed one
  by one (`impact_channel`).
- `--null-surrogates 0` (default): the legacy estimator modes run no null
  family, so their components are UNDEFINED (`NO_NULL_CALIBRATION:<P>`).
  Modes with their own null (section 7) always use it.
- `--bootstrap-se 0` (default): no sampling SE, so every empirical component
  is UNDEFINED (`NO_SAMPLING_SE:<P>`) and every verdict `UNDETERMINED`. With
  $B > 0$ each estimate gets a moving-block bootstrap SE
  (`nulls.component_bootstrap_se`; block `--bootstrap-block-len`, default
  $\lceil\sqrt{T}\rceil$ samples; events move with their blocks, replays without
  their self-caused event are dropped; self-calibrating modes run with a
  minimal null size, which does not change their evidence statistic) with
  $B_{\mathrm{valid}} - 1$ degrees of freedom (`<P>_se_df`). The SE stays
  undefined when fewer than two, or fewer than half
  (`MPC_BOOTSTRAP_MIN_VALID_FRACTION`), of the replicates are valid; failures
  are counted in `<P>_boot_failed`. Runtime is about $(B + 1)\times$ per
  component on top of the null cost.
- IIM evidence is on the integration-mass scale: `IIM_estimate` is $\Delta\Psi$
  and `IIM_null_mean`/`IIM_null_sd` in `step2_df.csv` are the $\Delta\Psi$ null
  moments in bits. (In `compute_IIM`'s details and in the Hunter table
  `hunter_iim_results.csv`, `IIM_null_mean`/`IIM_null_sd` are on the `IIM_raw`
  ratio scale; `Delta_Psi` is the common quantity.) IIM bootstrap replicates
  score the subsystem selected on the original data.
- **Hunter.** The campaign computes IIM with the protocol's IIM options and
  bearer nodes. `--hunter-iim-null-surrogates K` adds K surrogate runs per real
  run (otherwise `NO_NULL_CALIBRATION:IIM`, and at K > 0 locally an IIM column
  of NaN, `null_calibration_unavailable`); `--hunter-iim-bootstrap-se B` adds B
  block-bootstrap replicate runs per real run, drawn exactly as the local
  pipeline draws them, and the reducer reports `Delta_Psi_bootstrap_se`
  (otherwise Hunter IIM evidence is `NO_SAMPLING_SE:IIM`). A result computed
  with other IIM options than the protocol's is UNDEFINED
  (`iim_option_mismatch:<option>`).
- Verdicts are per run. Aggregating runs of a subject is an analysis decision.

## 9. TOST status rule

A protocol with a `status_rule` block (section 10.2) is judged by the TOST
rule (`evidence.tost_verdict`); every benchmark protocol in
`protocols/bench/` carries the block. A protocol without it is judged by the
interval rule of section 8 (`evidence.interval_verdict`).
`evidence.mpc_verdict` judges by the protocol's rule, each rule refuses the
other's protocols, and the empirical pipeline refuses a protocol with a
`status_rule` block. The vocabulary of reasons and flags is
`impact_pipeline.reasons`; testability gating and protocol anchors are
`impact_pipeline.testability`.

### 9.1 Entry points

The decision is implemented once and reached through these entry points; the
test suite checks that they agree:

| Entry point | Input | Use |
|---|---|---|
| `assess_item(ev, protocol)` | one evidence item (`ComponentEvidence`) under a TOST protocol | runner, component records |
| `assess_principle(P, items, protocol)` | the items of one principle (channels, directions) | runner, component records |
| `tost_verdict(evidence, protocol)` | the evidence of a run | verdicts |
| `assess_array(...)` | raw inputs, vectorised (the arguments of `component_status_array`) | bulk re-judging |
| `status_c(c, se, df)` | the construct scale, vectorised | simulations |
| `status_c_directional(c, se, df)` | directional members on the last axis | simulations |

`c`, `se_c` and the degrees of freedom are computed exactly as by the
interval rule ([section 8.2](#82-component-status)):
`c = (m - nu) / (rho - nu)`; `se_c` by the delta method over the sampling
SE, the Monte-Carlo error of the null mean and the reference SE; `df` by
Welch-Satterthwaite from the item's `se_df`.

### 9.2 Decision

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

The cutoffs are fixed, `z = 0.25` and `delta = 0.10`. ABSENT and PRESENT
exclude each other because ABSENT needs `c < delta - q_A se_c <= z`. The
one-sided interval rule also calls a component ABSENT when `c` is credibly
below the null; the TOST rule calls it `NULL_MODEL_VIOLATED`, UNDEFINED.

**Exact computations** (deterministic known-TPM values, `exact = True`
without a sampling SE, SE method `exact` or none) use the same inequalities
with `se_c = 0`: PRESENT iff `c > z`, ABSENT iff `|c| < delta`,
`NULL_MODEL_VIOLATED` iff `c < -delta`. This holds also when the null mean
is a Monte-Carlo estimate or the anchor has an SE: those parts are reported
(`se_null`, `se_reference`) but do not enter the decision. A data-derived
`se = 0` is never exact; it is `NO_SAMPLING_SE` unless the concordance
route applies (section 9.3).

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

### 9.3 Consumption contract

Each principle feeds exactly one ABSENT decision into the verdict at
`alpha_A`. A principle with several members declares how they combine: a
union claim ("absent if either is negligible") is tested at `alpha_A / k` per
member; an intersection claim needs no correction.

**Directions (NAS).** Declared NAS reports a receive and a return direction,
each on its own anchor (`c_R`, `c_B`). The protocol declares
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
missing direction is `MISSING_CHANNEL:<channel>:<direction>`: it blocks
PRESENT, and ABSENT through the other direction stays possible. Every item of
a directional principle names its direction.

**Channels.** The channels of a principle are combined by strong-Kleene OR:
ABSENT needs every channel ABSENT (an intersection, each at `alpha_A`), and
PRESENT through any of `k` channels is a union, so each channel's PRESENT
test runs at `alpha / k`. `k` is the number of declared channels, or of the
channels in the evidence when none are declared. A declared channel without
an item is `MISSING_CHANNEL:<channel>`, so a protocol that declares the
unimplemented perturbational and endogenous RAM channels (as
`mpc_default.json` does) can never make RAM ABSENT through behavioural data.

**SE-method contract.** Every benchmark component record carries `se`,
`se_df` and `se_method`; the rule reads `se_df` from the record, and the
record must agree with its SE method (`evidence.SE_METHODS`). `se_df`
describes the variability of the SE estimator, not a Monte-Carlo count:

| `se_method` | Principles | `se_df` |
|---|---|---|
| `jackknife_contiguous_10` | NAS, IIM, PDI | 9 (G - 1) |
| `jackknife_contiguous_20` | NAS | 19 |
| `jackknife_interleaved_10` | NAS | 9 |
| `jackknife_interleaved_20` | NAS | 19 |
| `circular_block_bootstrap_10pct_B50` | IIM | 12 (the block-count approximation 12.5 rounded down) or 9; never B - 1 |
| `shift_null_sd` | RAM | `n_null - 1` |
| `jackknife_trials_10` | RAM | 9 |
| `jackknife_pairs_10` | SRPI | 9 |
| `concordant` | PDI | none; `se = 0` (the concordance route) |
| `exact` | IIM | none; `se = 0`, `exact = True` |

An unknown method, a method not defined for the principle or not listed in
the protocol's `se_methods`, an `se_df` that breaks the method's rule, a
sampling SE on `concordant` or `exact`, or `exact` without the `exact` method
is `INVALID_SE`. Where the protocol's `se_methods` lists the admitted methods
of a principle, an item of that principle without an `se_method` is
`INVALID_SE` as well (otherwise a block bootstrap could report `B - 1`
unchecked); only where the protocol declares no methods does `se_df` keep
its plain meaning for an item without one.

**Concordance route (PDI).** A PDI component whose resampled counts all
agree has `se = 0` and SE method `concordant` (section 3.4). It is not
exact. It is `NO_SAMPLING_SE` unless the protocol's `concordance_route`
admits its cell (principle, substrate, observation stage, view, content
bearer). In an admitted cell it is ABSENT iff `|c| < delta` (flag
`ABSENT_BY_CONCORDANCE`) and PRESENT iff `c > z` (flag
`PRESENT_BY_CONCORDANCE`), each only if that direction of the cell is
admitted (otherwise `NO_SAMPLING_SE`); a value in between is `INCONCLUSIVE`,
one below `-delta` is `NULL_MODEL_VIOLATED`. The route is chosen by the SE
method, so components without one and `se = 0` stay `NO_SAMPLING_SE` exactly
as recorded. Its error control is empirical over the admission battery, not
a sampling-theory guarantee.

**IIM rank gate.** With `rank_gates = {"IIM": 0.05}`, an estimated IIM
component is PRESENT only if also `p_ind <= 0.05` (the rank p-value against
the independence null, section 5.6). A failed gate is
`INCONCLUSIVE:NULL_NOT_EXCEEDED`, a missing `p_ind` is
`NO_NULL_CALIBRATION`; the gate never creates ABSENT and does not apply to
exact values.

**Registry licensing.** The applicability registry licenses each status
direction separately (`admitted_for_present`, `admitted_for_absent`: `yes`,
`no`, `vacuous`, `not_observable`; `vacuous` admits). A PRESENT or ABSENT on
a direction that is not admitted becomes
`ESTIMATOR_NOT_VALIDATED:not_admitted` (or `:not_observable`); an item with
no admitted direction is that reason from the start. A validation entry of
the registry (section 8.3) licenses both directions together.

**SE reversion.** An SE method found anti-conservative on the twins
re-classifies the ABSENTs of that (principle, SE method) as
`SE_NOT_CALIBRATED` (argument `uncalibrated_se_methods`); PRESENT is not
changed.

**Estimator versions.** A TOST protocol names the estimator version of each
principle; evidence from another version (or without one) is
`ESTIMATOR_NOT_VALIDATED:not_admitted`.

**UNDEFINED is never counted as ABSENT.** Every reason maps to UNDEFINED; the
verdict is EXCLUDED only through an ABSENT principle of the necessity set.

**Verdict.** The strong-Kleene AND over the protocol's `necessity_set`
(`N_anch`, the anchored principles): T gives `MPC_CONSISTENT`, F gives
`EXCLUDED`, U gives `UNDETERMINED`; a bearer, protocol or source code
(`BEARER_MISMATCH`, `PROTOCOL_MISMATCH`, `SOURCE_INCOHERENT`, as for the
interval rule) forces `UNDETERMINED`. Every principle with evidence or in the
declared set is assessed; the AND over the declared set `N_decl` is reported
beside the verdict as `verdict_declared` (descriptive). With an external
reference the protocol's anchors replace those of the items, so every status
is on the protocol's anchor scale.

### 9.4 Error control

Let `theta_p` be the true construct value of principle `p` on the protocol's
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

Conditions: the anchors come from development seeds disjoint from the
evaluation seeds; SE validity (A-SE); the consumption contract; UNDEFINED
never counted as ABSENT; a valid null for `nu` (`NULL_MODEL_VIOLATED`
catches gross failures only). The test suite checks E1-E4 by simulation
under (A-SE), including the NAS boundary case (both directions at `delta`)
and the excess of an unsplit union over `alpha_A`.

**What the rule does not control.** Estimator bias (a precise estimate near 0
with the mechanism on is a correct TOST decision about a wrong `theta`);
anchor misattribution (an anchor dominated by a nuisance scales every `c`
wrongly; the specificity gate of section 9.6 addresses it); construct
misidentification under hidden inputs (under a partial or empty input
declaration PRESENT certifies dependence only); and the abstention
trade-off (UNDETERMINED buys selective error control with coverage, which is
reported per protocol).

### 9.5 Reason codes and flags

Every reason is defined once in `impact_pipeline.reasons`. A reason is
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
| `SAMPLING_UNRESOLVED` | `dt > tau_c / 2`: the sampling does not resolve the coupling time scale | declared NAS |
| `OBSERVATION_MIXED_NOT_ADMITTED` | a sensor or source-estimate observation without an admitting registry entry | declared NAS, macro IIM |
| `INSUFFICIENT_OCCUPANCY` | a macro state unvisited, or the rarest visited row below `N_min` transition pairs | macro IIM |
| `MACRO_RANK_DEFICIENT` | the zero-lag correlation of the cluster means is rank deficient | macro IIM |
| `NOT_APPLICABLE_OBSERVATION_MODEL` | declared inapplicable to the observation model (family-C carrier recording) | SRPI, RAM-PE, PDI |
| `INSUFFICIENT_TIMEPOINTS` | too few time points for the parameters at every allowed block dimension | declared NAS |
| `INSUFFICIENT_UPDATES` | fewer than 30 updates or fewer than two options | RAM-PE |
| `ESTIMATOR_ERROR` | the estimator raised; detail: the exception type; other components are unaffected | benchmark runner |
| `SE_NOT_CALIBRATED` | the ABSENT of an SE method found anti-conservative on the twins | SE reversion |
| `ESTIMATOR_NOT_VALIDATED` | `:not_admitted` (no admitting registry entry for the status direction, or another estimator version than the protocol's) or `:not_observable` | registry; protocol |

Flags are markers next to a status, never a status and never a reason:

| Flag | Meaning |
|---|---|
| `PRESENT_NOT_REACHABLE` | UNDEFINED and `q_P se_c >= 1 - z` |
| `ANCHOR_NOT_REPLICATED` | the anchor status did not replicate on a replication block; the protocol is kept unchanged |
| `HUB_NOT_PRIVILEGED` | the declared NAS hub is not privileged over every pseudo-hub; restricts the construct wording only |
| `ABSENT_BY_CONCORDANCE` | ABSENT through the admitted concordance route |
| `PRESENT_BY_CONCORDANCE` | PRESENT through the admitted concordance route |
| `NONSPECIFIC_ANCHOR` | the anchor failed the specificity gate; the principle leaves `N_anch` and keeps its component-level evaluation |

**Verdict reason codes** are `KIND:<P>[:detail]` with the component reason's
code and detail, for example `ABSENT_NOT_REACHABLE:IIM`,
`INCONCLUSIVE:IIM:NULL_NOT_EXCEEDED`, `ESTIMATOR_NOT_VALIDATED:NAS:not_admitted`,
`MISSING_CHANNEL:NAS:default:return`, plus `ABSENT:<P>` for an absent
principle, `MISSING:<P>` and the global codes. The verdict is recoverable from
its reasons by the decomposition rule of section 8.4
(`evidence.verdict_from_reasons`).

Component records (`mpc-bench-result/1`, section 14.3) take `status`,
`reason`, `flags`, `c` (NAS: `min(c_R, c_B)`), `c_R`, `c_B`, `se_c` and
`df_c` from `PrincipleAssessment.record_fields()`.

### 9.6 Testability, anchors and necessity sets

These rules read development data only (seeds 0-999) and never the construct
value of a confirmatory target (`impact_pipeline.testability`).

**Testability gating.** For every ABSENT-type hypothesis part, `pi0` is the
empirical ABSENT rate of the target on its witness under the TOST rule (at
least 40 development runs); the analytic
`mean max(0, 2 Phi(delta / se_c - q_A) - 1)` is reported beside it. For
every "PRESENT in >= 80 %" part, `pi1` is the empirical PRESENT rate on the
positive control (analytic `mean Phi((1 - z) / se_c - q_P)` beside it). A
part is decisive only if its rate is at least 0.9; otherwise it is
`NOT_TESTABLE_BY_DESIGN`, declared before any confirmatory data, and
replaced by "target not PRESENT in >= 80 % of seeds" (ABSENT parts) or a
paired contrast with the positive control (PRESENT parts), and its predicted
`ABSENT_NOT_REACHABLE` rate is recorded. The rows (family, principle,
witness, kind, `n`, `pi`, the analytic rate, `s_A`, `s_P`, the attainable
`se_c`, the rates of every status and reason, the outcome) form the
protocol's hash-covered `precision` block (schema `impact-mpc-precision/1`).

**Anchors.** On the development reference block (seeds 900-939) a
principle's anchor is valid iff at least 36 of the 40 positive-control
excesses are finite and the one-sided 95 % Student-t lower bound of their
mean is above 0. It is specific iff the mean paired contrast of the positive
control minus the principle's own lesion on the same seeds is at least 0.5
times the anchor and its one-sided 95 % lower bound is above 0; otherwise the
principle is flagged `NONSPECIFIC_ANCHOR`. NAS is anchored iff both
directions are valid and specific.

**Necessity sets.** `N_anch` is the set of declared principles whose anchor
is valid and specific; it is the protocol's `necessity_set`, and `alpha_A`
does not depend on it. Fallbacks: F0 every declared principle anchored, F1
two or more, F2 one, F3 none (the protocol then keeps the declared set and
the verdict parts are not evaluable). A principle outside `N_anch` keeps its
component-level evaluation. The protocol's hash-covered `anchors` block
(schema `impact-mpc-anchors/1`) holds the rule, the reference seeds, one
entry per declared principle, `N_anch` and the fallback.

### 9.7 Scale type and the absence margin

The scale type of each estimator version (`evidence.SCALE_TYPES`):

| Scale type | Estimator versions | Statistic |
|---|---|---|
| `quadratic` | `nas-2026.09`, `nas-declared-2026.10`, `iim-2026.09`, `iim-macro-2026.10` | transfer entropy or conditional Geweke causality (NAS), Delta Psi (IIM) |
| `first_order` | `ram-pe-2026.10`, `srpi-2026.09` | a correlation (RAM-PE), the SRPI composite |
| `count` | `pdi-2026.09`, `pdi-declared-2026.10` | bits of a count of states |

A quadratic statistic grows with the square of a weak effect amplitude, so
`delta = 0.10` on `c` corresponds to an amplitude of about
`sqrt(0.10) = 0.32` of the reference. Every ABSENT of a quadratic statistic
is re-judged on the amplitude scale `a = sign(c) sqrt(|c|)` as a sensitivity
analysis (`scale="amplitude"`). The scale type is read from the estimator
version the protocol names for the principle (`estimator_versions`); without
one the cutoffs are not changed. Because `a` is monotone in `c`, judging the
interval of `a` against `(z, delta)` is the same as judging the interval of
`c` against `(z**2, delta**2)` (`evidence.amplitude_cutoff`); the cutoffs of
the other scale types are unchanged.

A sweep dose counts as "mechanism on" for a principle and protocol (the rows
of the false-exclusion checks) only if the dose is at least 0.5 times the
nominal dose and the development median `c` at that dose is at least
`2 delta` (`testability.mechanism_on`, with
`testability.development_median_c` over development runs).

## 10. Protocols

A protocol (`evidence.Protocol`, JSON schema `impact-mpc-protocol/1`)
declares everything a verdict depends on besides the data. The shipped
protocols, their hashes and the reasons behind each declaration are in
[`protocols/README.md`](../protocols/README.md).

### 10.1 Fields of every protocol

| Field | Content |
|---|---|
| `schema` | `impact-mpc-protocol/1` |
| `name` | the protocol's name |
| `necessity_set` | $N$, the principles the verdict requires |
| `channels` | per principle the declared evidence channels (section 8.3) |
| `cutoffs` | per principle $(z_j, \delta_j)$ on the construct scale |
| `alpha` | the one-sided level of the status tests |
| `null_families` | per principle the declared null family |
| `reference` | `{"kind": "cohort_high_state", "session": ..}` or `{"kind": "external", "values": {P or P:channel: ..}, "se": {..}, "scale": "excess" or "estimate", "source": ..}` |
| `source_rule` | `single_source`, `same_bearer` or `none` (section 8.3) |
| `estimators` | per principle the estimator options, e.g. `{"NAS": {"mode": "capacity"}}` |
| `bearer_nodes` | the declared bearer |
| `status_rule` | absent: the interval rule (section 8); present: the TOST rule (section 9) with the fields of section 10.2 |

### 10.2 Fields of the TOST rule

A protocol with a `status_rule` block also has:

| Field | Content |
|---|---|
| `status_rule` | `{"version": "tost", "alpha", "alpha_absent", "absent_test": "tost", "null_violation", "present_reachability_flag"}`; `alpha` equals the protocol's, and `alpha_absent` is `alpha` divided by the size of `declared_necessity_set` |
| `declared_necessity_set` | `N_decl`, always the five principles, so `alpha_A = 0.01` in every protocol; `necessity_set` is `N_anch`, a subset |
| `directions` | per directional principle its members, for example `{"NAS": ["receive", "return"]}` (two or more lower-case names that differ from the channel names) |
| `rank_gates` | per principle the level of a rank gate on PRESENT, for example `{"IIM": 0.05}` |
| `estimator_versions` | per principle the estimator version the protocol dispatches |
| `se_methods` | per principle the admitted SE methods |
| `shared_inputs_declaration` | `{"id", "shared_inputs"[, "inputs"]}`: the input declaration (section 14.2) and its level (complete, partial, none) |
| `concordance_route` | the admitted PDI cells `{principle, substrate, observation_stage, view, bearer, absent, present, provisional, battery}` |
| `anchors` | the anchor block (section 9.6) |
| `precision` | the testability table (section 9.6) |

The status rule block of every benchmark protocol:

```json
{"version": "tost", "alpha": 0.05, "alpha_absent": 0.01, "absent_test": "tost", "null_violation": true, "present_reachability_flag": true}
```

**Reference.** The kinds of every protocol (`cohort_high_state`, `external`)
and `pending` (no anchors yet: every component is `INVALID_ANCHORS`;
templates only). External keys are `P`, `P:<channel>`, and for directions
`P:<direction>` or `P:<channel>:<direction>`; a directional principle takes
only per-direction keys (a pooled or a channel-only key is refused, since it
would never be read).

### 10.3 Hash

The hash is the SHA-256 of the canonical JSON of `Protocol.to_dict()`
(sorted keys, compact separators, every cutoff written out; a protocol with a
`status_rule` block also writes every field of section 10.2). Evidence
records it as `protocol_id = "sha256:<hash>"`, the pipeline as
`MPC_protocol_hash`, and the benchmark records and manifests with each
scoring. Every field is hash-covered, so a change of any declaration, of the
name or of a reference text gives another hash.

## 11. Legacy CI (deprecated)

`compute_CI` is a deprecated alias: it emits a `DeprecationWarning` once per
process and otherwise behaves as before. The output column `CI` is kept for
backward compatibility and is the **legacy geometric-mean CI (not a gate)**. Use
the MPC verdict for decisions and the MPC degree as the secondary summary.

$$
\mathrm{CI} = (\mathrm{RAM}^*)^{\alpha}(\mathrm{PDI}^{+*})^{\beta}(\mathrm{NAS}^*)^{\gamma}(\mathrm{IIM}^*)^{\delta}(\mathrm{SRPI}^*)^{\rho},
\qquad M^* = \max\!\left(\frac{M}{\mathrm{ref}_M}, 0\right),
$$

with equal weights by default.

- **Reference-normalised.** The default reference is the cohort's high-state
  (awake) session: subject means, then cohort mean, per component, from the
  same call (`CI_reference=cohort_high_state`). An external JSON
  (`--ci-reference file.json`, `{"references": {"RAM": .., ...}}`) replaces it.
  A reference that is missing, non-finite or $\le 0$ makes CI undefined for the
  row (no floor).
- **NAS enters directly.** The HypergraphSynergy multiplier was removed from CI,
  so CI does not depend on $\theta$.
- **Three-valued.** Any weighted component that is undefined makes CI NaN with
  `CI_defined=False` and `CI_missing` listing the component names (or
  `<component>_reference` for unusable references). A measured component
  $\le 0$ with positive weight gives CI = 0. Zero-weight components are ignored.
- Downstream statistics exclude undefined CI rows and report how many were
  excluded (`stats/statistics_summary.json`, `ci_definedness`).
- With `--null-surrogates K > 0` CI is computed from the calibrated metric
  columns.

## 12. Exploratory statistic S

`S` (HypergraphSynergy) is a separately reported, exploratory legacy statistic.
It is computed for every $\theta$ in 0.1-0.9 (one `step2_df` row per run and
$\theta$), tested at every $\theta$ with Holm correction across $\theta$, and
never enters CI or the MPC verdict. No $\theta$ is selected post hoc.

## 13. Output columns

`<out-dir>/cache/step2_df.csv` has one row per run and $\theta$ (the metric,
CI and evidence values repeat across $\theta$). `step2_df_mean.csv` holds
subject x session means of the metric columns (no verdicts).

| Column(s) | Content |
|---|---|
| `dataset_id`, `data_origin`, `dataset_role`, `provenance_label`, `hardware_target`, `hardware_backend`, `hardware_runtime` | provenance of the row |
| `subject`, `session`, `theta` | run identity; $\theta$ of S |
| `S` | exploratory HypergraphSynergy at `theta` |
| `RAM`, `PDI`, `NAS`, `IIM`, `SRPI` | metric values (calibrated when `--null-surrogates K > 0`); NaN when undefined |
| `PDI_anchor`, `PDI_task`, `PDI_*_defined`, `PDI_*_reason`, `PDI_primary_endpoint`, `PDI_primary_source`, `PDI_baseline_policy`, `PDI_*_baseline_n_runs`, `PDI_*_baseline_paths` | PDI endpoints and their baselines |
| `IIM_raw`, `IIM_raw_scaled`, `IIM_defined`, `IIM_undefined_reason` | uncalibrated IIM ratio and definedness |
| `CI`, `CI_defined`, `CI_missing`, `CI_reference`, `<P>_norm` | legacy CI (only when all five metrics are computed and CI is not disabled) |
| `MPC_verdict` | `EXCLUDED`, `MPC_CONSISTENT` or `UNDETERMINED` |
| `MPC_reason` | `;`-joined reason codes (section 8.4) |
| `MPC_degree` | MPC degree (`MPC_CONSISTENT` rows only) |
| `MPC_necessity_set`, `MPC_null_surrogates`, `MPC_null_seed`, `MPC_null_families`, `MPC_bootstrap_se`, `MPC_bootstrap_block_len` | evidence configuration of the row |
| `MPC_protocol_hash` | SHA-256 of the protocol (section 10.3) |
| `MPC_joint_dependence`, `MPC_joint_dependence_p` | single-source test: `dependent`, `untested` or the failure reason (empty with one node set), and its p |
| `<P>_status` | PRESENT, ABSENT or UNDEFINED (Kleene OR over the channels) |
| `<P>_margin`, `<P>_margin_absent` | presence margin `c_lower - z` (> 0 iff PRESENT) and absence margin `delta - c_upper` (> 0 iff ABSENT) of the deciding channel |
| `<P>_estimate` | raw estimator value (IIM: $\Delta\Psi$ in bits; PDI repertoire: bits) |
| `<P>_null_mean`, `<P>_null_sd`, `<P>_null_n` | null moments on the scale of `<P>_estimate` |
| `<P>_se`, `<P>_se_df`, `<P>_boot_n`, `<P>_boot_failed` | bootstrap sampling SE of the estimate, its degrees of freedom (valid replicates - 1), valid and failed replicates |
| `<P>_c`, `<P>_c_se`, `<P>_c_df`, `<P>_c_lower`, `<P>_c_upper` | construct-scale value, its SE, effective degrees of freedom and one-sided bounds |
| `<P>_reference`, `<P>_reference_se` | reference anchor (excess scale for the cohort reference) and its SE |
| `<P>_estimator`, `<P>_mode_reason` | estimator id `compute_<P>:<mode>@<version>`; why a declared fallback mode was used (empty otherwise) |
| `<P>_estimator_version` | the `<version>` of that id, exactly as recorded (for a precomputed Hunter IIM result its `iim_algorithm_version`) |
| `<P>_channels` | per-channel statuses `<channel>:<status>` |

Hunter campaigns add `<out-dir>/cache/hunter_iim_results.csv` (one row per real
run: `value`, `raw`, `canonical`, `Psi_full`, `Psi_mip_preserved`, `Delta_Psi`,
the `IIM_null_*` fields, `IIM_z`, `IIM_null_p`, `IIM_excess`,
`canonical_calibrated`, `Delta_Psi_bootstrap_se`, `Delta_Psi_bootstrap_n`,
`Delta_Psi_bootstrap_failed`, `Delta_Psi_bootstrap_block_len`,
`iim_algorithm_version`, `tpm_estimator`, `cut_mode`, `psi_kernel`, selected
nodes, node-selection rule, bins requested/used, `budget_adjustments`, observed
states, transitions and `code_version`).

## 14. Benchmark (MPC-Bench)

`impact_pipeline.bench` provides white-box systems whose mechanisms are known,
the runner that scores them and the checks around it; the command lines are
in the [README](../README.md#benchmark-mpc-bench). The generators never
import `mpc_metrics`. Each system's hidden ground truth is in its `oracle`,
which is stored apart from the data and never read by an estimator.

### 14.1 Systems

| Module | Systems |
|---|---|
| `bench.generators` | family A: a modular stochastic rate-network agent with five mechanism switches (`eta` RAM, `K` PDI, `g_b` NAS with `ff_only`, `c_int` IIM, `e` SRPI) that plays a reversal two-armed bandit and then a reafference phase; family B: binary / kinetic-Ising networks with exact TPMs (the IIM ground truth); family C: a Stuart-Landau oscillator network with the family-A switch semantics; null and hypersynchronous systems |
| `bench.patchwork` | five mutually disconnected single-principle modules; graded patchworks couple them with `inter_module_coupling` in [0, 1] |
| `bench.whole_brain` | a Hopf whole-brain model on the shipped structural connectome (`data/managed/structural/`), with coupling sweeps and lesions |
| `bench.forward` | EEG-like observations (smooth lead field with volume conduction, EMG and sensor noise, average reference, band-pass) and BOLD-like observations (canonical HRF, AR(1) noise), and the views of one simulation: `source`, `eeg64`, `eeg64_noref`, `eeglow`, `mne_template` (a source estimate) and `bold` |
| `bench.adversarial` | constructions designed to fool one estimator: `common_driver`, `reflex_arc`, `random_label_self_other`, `scrambled_feedback` and the staggered hidden driver |
| `bench.catalogue` | the system catalogue `bench/systems.yaml`: every witness and adversary, defined by mechanism (generator, knobs, configuration), never by metric values |
| `bench.designs.null_calibration` | null systems (`ar1`, `pink`, `surrogate_iid`, `surrogate_linear`) with a synthetic task schedule independent of the data |

`bench.manipulation` checks the generators against their oracle alone. A
switch is effective when switching it off changes its oracle signature (a
statistic of hidden generator channels) by a declared relative amount at the
nominal dose; realisation checks confirm that systems defined by a
configuration have or lack their defining mechanism as stated. Versions:
generators `mpc-bench-generators/1.0.0`, forward models
`mpc-bench-forward/1.0.0`, manipulation checks
`mpc-bench-manipulation/1.0.0`, adversaries `mpc-bench-adversarial/1.0.0`,
catalogue `mpc-bench-systems/1.0.0`.

### 14.2 Declared inputs

The recording device (`estimators.declared_inputs.record_inputs`) builds,
from a system, what an experimenter could log: the task events,
`context_cue` events (one per context switch), the slow-rhythm phase
(`slow_phase_sin`, `slow_phase_cos`) and any recorded driver a generator
exposes. It only reads the system, so every pipeline estimator returns the
same output with and without it. A declaration selects, and possibly
corrupts, what the analyst conditions on; its id is the protocol field
`shared_inputs_declaration`:

| Declaration | Content | Shared inputs |
|---|---|---|
| R | exogenous task events, context cues, slow phase and every recorded driver | complete |
| H | exogenous task events only | partial |
| P | R without the slow phase | partial |
| Q10, Q25 | R with each context-cue label replaced, with probability 0.10 or 0.25, by another label | partial |
| J | R with each context-cue onset shifted by U(-0.25, 0.25) s | partial |
| none | no inputs | none |

The family-B cells use `none`, `recorded` (the driver is declared),
`hidden` (it exists and is not declared) and `label_error`. Only inputs set
independently of the system's state enter the basis (goal cues, stimuli,
other-caused events, context cues, the slow phase, recorded drivers);
responses, feedback, actions and self-caused events do not. The input basis
is
$$
U(t) = \big[u(t - l),\ \tilde u_\tau(t - l)\big],\quad l \in \{0\} \cup L,\ \tau \in \{\tau_c, 3\tau_c, 10\tau_c\},
$$
with $\tilde u_\tau$ a causal first-order low-pass of $u$ and $L$ the
estimator's lag set. $\tau_c$ is the intrinsic time constant of the coupled
units, read from the generator constants (0.1 s on every benchmark
substrate). Every conditioned null shifts the raw series first and fits the
projections afterwards (`shift_then_project`).

### 14.3 Runner and records

The runner (`bench.runner`, command line `scripts/run_bench.py`) runs the
tasks of the designs (`bench.designs`). A task is one simulation (a system,
a seed and a twin replicate) with several scorings: an input declaration, an
observation view, an estimator form and the family protocol that judges
them. Each scoring dispatches every principle to the estimator version its
protocol names:

| Estimator version | Scorer |
|---|---|
| `ram-pe-2026.10`, `pdi-declared-2026.10` | `compute_ram_pe`, `compute_pdi_declared` |
| `nas-declared-2026.10` | `compute_nas_declared` on the view's declared hub and periphery blocks |
| `iim-macro-2026.10` | `compute_iim_macro` on the view's macro nodes (at most 4 per run) |
| `srpi-2026.09`, `ram-2026.09`, `pdi-2026.09`, `nas-2026.09`, `iim-2026.09` | the in-memory path of the `mpc_metrics` estimators (`bench.export.run_in_memory`: 19 null surrogates, a jackknife over 10 groups) |

Each estimator runs in its own try block: a raising estimator gives
`UNDEFINED(ESTIMATOR_ERROR:<type>)` for that component and the task status
`ok_with_component_errors`, and touches nothing else. The TOST rule of the
scoring's protocol judges the components. Records (schema
`mpc-bench-result/1`, `bench.records`) hold the simulation hashes and, for
every component, the estimate, the null moments and family, `se`, `se_df`,
`se_method`, `c` (NAS also `c_R` and `c_B`), status, reason, flags,
declaration, identifiability record, observation stage, estimator version,
protocol id and hash, replicate, time and load average. A run writes
`run_plan.json`, `results.jsonl`, `components.csv` and `run_manifest.json`
(code identity, environment, protocol file hashes); it resumes from partial
results and can run one shard of a plan.

The family protocols are `protocols/bench/<key>.json`. Estimator forms are
reported variants of a scoring with their own protocol `<key>+<form>`: the
NAS secondary representation (`nas_secondary`) and lag time scales
(`nas_tau_0.05`, `nas_tau_0.2`), the bidirectional IIM cut
(`iim_bidirectional`), a misdeclared PDI access node
(`pdi_misdeclared_access`) and the electrode-quadrant IIM grain
(`iim_quadrants`). Without a protocol file, a development run derives a
draft from `protocols/bench/template.json` (reference `pending`, so every
component is `INVALID_ANCHORS`); confirmatory runs use the files only.

Designs (`scripts/run_bench.py designs`): family A (`A_witnesses`,
`A_sweeps`, `A_factorial`, `A_adversaries`, `A_heldout`), family C
(`C_witnesses`, `C_sweeps`, `C_factorial`), the RAM-only arm (`RAM160`), twin
sessions (`A_twins`, `C_twins`, `RAM160_twins`), the anchor blocks
(`A_anchors`, `C_anchors`, `RAM160_anchors`), the forward-model arms
(`whole_brain`, `forward_family_a`, `forward_family_a_bold`,
`forward_anchor_replication`) and `null_calibration`. The family-B cells run
through `scripts/iim_validation.py`.

### 14.4 Seeds and run hygiene

Development seeds are 0-999 (the anchor reference block is 900-939; the
smoke seeds 980-984 build held-out conditions, whose outputs the runner
discards unread), confirmatory seeds are 20000 and above, and the seeds in
between are refused (`bench.seeds`). One run never mixes the two splits.
Twins with replicate $r \ge 1$ draw the task schedule and the noise from
`SeedSequence([seed, r])` ($1 \le r \le 30$); label errors, cue jitter and the
staggered driver use the reserved stream keys 41, 42 and 43.

Every record carries the code identity: the commit, the dirty state and the
git trees of `src/`, `scripts/` and `protocols/` (`bench.run_hygiene`). A
confirmatory run (`--confirmatory --freeze-tag <tag>`) is refused unless the
checkout is clean and its `src/` and `scripts/` trees equal those of the tag
the user names, so the code of a confirmatory run is the code fixed before
its data were generated.

### 14.5 Null calibration and IIM validation

`scripts/null_calibration.py` runs the null-calibration design: the null
systems of section 14.1 in 16 cells (four null kinds, two run lengths, two
node counts) scored with the declaration `none` under the protocol `A-none`
(the A-R protocol with that declaration and without IIM's reported cut).
Its `summarise` step reports the PRESENT and ABSENT rates per cell and
principle with Wilson intervals, and the verdict rates.

`scripts/iim_validation.py` scores macro IIM on the family-B cells
(`bench.designs.family_b`): binary networks with exact TPMs, grouped as
`rank_calibration`, `feedforward_star`, `exact_targets` and
`driver_conditioning`, and judges them with the TOST rule on the family-B
construct scale, whose targets come from the exact TPMs.

### 14.6 Hypotheses evaluator

`bench.hypotheses` evaluates a declarative hypotheses file (schema
`mpc-bench-hypotheses/1`) on benchmark records. A hypothesis has parts; a
part names one rule of `bench.hypotheses.RULES`, its parameters and a data
selection, so the file is the single statement of every decision and a file
fixed before the confirmatory data are run states the confirmatory analysis.
Outcomes are `SUPPORTED`, `FALSIFIED`, `INDETERMINATE`, `NOT_EVALUABLE` and
`NOT_TESTABLE_BY_DESIGN` (the testability gate of section 9.6); a hypothesis
with `admitted: false` is `NOT_RUN`. Rates are counted over clusters (every
re-scoring of one simulation is one cluster) with Clopper-Pearson bounds.
`scripts/evaluate_hypotheses.py --spec <file>` runs the evaluator;
`tests/fixtures/hypotheses_example.json` is a small example file.
