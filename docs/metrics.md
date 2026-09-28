# Metric Definitions

This document describes what the code computes in release 1.1.0: the five
null-anchored MPC estimators (RAM, PDI, NAS, IIM, SRPI), the evidence layer
that turns them into the MPC profile, the three-valued MPC verdict and the MPC
degree, and the deprecated legacy CI. It is implementation-facing: every
statement refers to `src/impact_pipeline/mpc_metrics.py`,
`src/impact_pipeline/evidence.py`, `src/impact_pipeline/synergy_ci.py` or
`src/impact_pipeline/nulls.py`. Function docstrings are the authoritative
parameter reference.

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
9. [Legacy CI (deprecated)](#9-legacy-ci-deprecated)
10. [Exploratory statistic S](#10-exploratory-statistic-s)
11. [Output columns](#11-output-columns)

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
  are new modes (`mode=`/`update=` keywords). `run_pipeline.py` uses the default
  modes unless a protocol (`--protocol`, for example
  [`protocols/mpc_default_v1.json`](../protocols/README.md)) or the params
  dicts select others (section 8.6). MPC-Bench uses
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
  evidence layer (Kleene OR), not averaged.
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
for reference and is not used by the bench or registered for paper 2;
`mode="repertoire"` replaces it.

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
invalid declarations raise). Every other bearer node is periphery. Hub and
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
| `joint_laplace` | legacy joint $K^n \times K^n$ Laplace estimate (reproduces pre-v4 values; turns noise into spurious non-factorisable repertoires) |

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
  algorithm version `IIM_ALGORITHM_VERSION = "iim-v4-2026.09"`. Checkpoints of
  another version are not resumed.
- Undefined IIM is explicit (`IIM_defined=False`, `IIM_undefined_reason`).
- On HLRS Hunter the same computation is distributed over PBS array jobs
  (`docs/HLRS_HUNTER_RUNBOOK.md`); its results equal `compute_IIM` to 1e-9,
  including the surrogate calibration.

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
- Protocol values (frozen at `mpcbench-freeze-v1`, all shipped protocols):
  $z_j = 0.25$, $\delta_j = 0.10$, $\alpha = 0.05$, kept after the MPC-Bench
  development runs (dose-response, null systems); the evidence is in the
  [preregistration](preregistration/MPC_BENCH_PREREGISTRATION.md), section 4.
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
- **Protocol** (`evidence.Protocol`, JSON schema `impact-mpc-protocol/2`; the
  shipped protocols and their rationale are in
  [`protocols/`](../protocols/README.md)). Fields: `necessity_set`;
  `channels` (per principle the declared channels); `cutoffs` (per principle
  $(z_j, \delta_j)$); `alpha`; `null_families` (per principle the declared
  family); `reference` (`{"kind": "cohort_high_state", "session": ..}` or
  `{"kind": "external", "values": {P or P:channel: ..}, "se": {..},
  "scale": "excess"|"estimate", "source": ..}`); `source_rule`
  (`single_source`, `same_bearer` or `none`); `estimators` (per principle the
  estimator options, e.g. `{"NAS": {"mode": "capacity"}}`); `bearer_nodes`;
  `name`. The hash is the SHA-256 of the canonical JSON of `to_dict()` (sorted
  keys, compact separators, every cutoff written out); evidence records it as
  `protocol_id = "sha256:<hash>"`. A declared channel without an evidence item
  counts as UNDEFINED (`MISSING_CHANNEL:<P>:<channel>`), so a principle with a
  declared channel that is not measured (or not implemented) is never ABSENT;
  items of undeclared channels are ignored (`ignored_channels`); principles
  without a declaration use the channels of their evidence. Evidence whose
  null family differs from the declared one is UNDEFINED
  (`NULL_FAMILY_MISMATCH`), and evidence carrying another protocol's id makes
  the verdict UNDETERMINED (`PROTOCOL_MISMATCH`).
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
- **Applicability registry** (`--applicability-registry`, schema
  `impact-mpc-registry/2`). Evidence from an estimator version that is not
  validated for its principle, substrate, grain and regime is UNDEFINED
  (`ESTIMATOR_NOT_VALIDATED:<P>:<estimator>`):

  ```json
  {"schema": "impact-mpc-registry/2", "version": "2026-09",
   "criteria": {"alpha": 0.05, "false_present_tolerance": 0.02},
   "entries": [
     {"estimator": "compute_NAS:capacity", "version": "nas-v2-2026.09",
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
  calibrated v2 rule has a rate far below alpha; `"two_sided": true` enforces
  `alpha ± tolerance`), a positive recovery slope and, when `cross_talk_max` is
  declared, cross-talk within it. Entries that fail are rejected when the file
  is loaded. String fields match shell wildcards (case-insensitive) and may be
  lists; `regime` values are a scalar (equality), a list (membership),
  `{"min", "max"}` or the named bounds `T_min`, `nodes_min`, `nodes_max`,
  `snr_min`. The pipeline queries with substrate = modality, grain = atlas and
  the regime of the component (bearer size; for IIM the scored subsystem,
  bins, series length and sample interval). Without a registry, determinate
  verdicts are possible for unvalidated estimators (the registry is opt-in).

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
| `UNDEFINED:<P>:<reason>` | the estimator, the null or the SE is undefined: the estimator's reason, `DEGENERATE_NULL`, `INVALID_SE`, `NULL_FAMILY_MISMATCH:<declared>/<used>`, `null_undefined:<reason>`, `iim_option_mismatch:<option>` |
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
(section 11).

- **Protocol.** `--protocol file.json` declares $N$, channels, cutoffs,
  $\alpha$, null families, reference, source rule, estimator modes and bearer
  nodes (`protocols/mpc_default_v1.json` is the shipped default for empirical
  data); its hash is `MPC_protocol_hash`. Without it a default protocol is
  built from `--necessity-set`, the null kinds, the mode keys of the params
  dicts (`synergy_ci.MPC_MODE_KEYS`) and the cohort reference; conflicting
  settings raise an error.
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

## 9. Legacy CI (deprecated)

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

## 10. Exploratory statistic S

`S` (HypergraphSynergy) is a separately reported, exploratory legacy statistic.
It is computed for every $\theta$ in 0.1-0.9 (one `step2_df` row per run and
$\theta$), tested at every $\theta$ with Holm correction across $\theta$, and
never enters CI or the MPC verdict. No $\theta$ is selected post hoc.

## 11. Output columns

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
| `MPC_protocol_hash` | SHA-256 of the protocol (section 8.3) |
| `MPC_joint_dependence`, `MPC_joint_dependence_p` | single-source test: `dependent`, `untested` or the failure reason (empty with one node set), and its p |
| `<P>_status` | PRESENT, ABSENT or UNDEFINED (Kleene OR over the channels) |
| `<P>_margin`, `<P>_margin_absent` | presence margin `c_lower - z` (> 0 iff PRESENT) and absence margin `delta - c_upper` (> 0 iff ABSENT) of the deciding channel |
| `<P>_estimate` | raw estimator value (IIM: $\Delta\Psi$ in bits; PDI repertoire: bits) |
| `<P>_null_mean`, `<P>_null_sd`, `<P>_null_n` | null moments on the scale of `<P>_estimate` |
| `<P>_se`, `<P>_se_df`, `<P>_boot_n`, `<P>_boot_failed` | bootstrap sampling SE of the estimate, its degrees of freedom (valid replicates - 1), valid and failed replicates |
| `<P>_c`, `<P>_c_se`, `<P>_c_df`, `<P>_c_lower`, `<P>_c_upper` | construct-scale value, its SE, effective degrees of freedom and one-sided bounds |
| `<P>_reference`, `<P>_reference_se` | reference anchor (excess scale for the cohort reference) and its SE |
| `<P>_estimator`, `<P>_mode_reason` | estimator id `compute_<P>:<mode>@<version>`; why a declared fallback mode was used (empty otherwise) |
| `<P>_channels` | per-channel statuses `<channel>:<status>` |

Hunter campaigns add `<out-dir>/cache/hunter_iim_results.csv` (one row per real
run: `value`, `raw`, `canonical`, `Psi_full`, `Psi_mip_preserved`, `Delta_Psi`,
the `IIM_null_*` fields, `IIM_z`, `IIM_null_p`, `IIM_excess`,
`canonical_calibrated`, `Delta_Psi_bootstrap_se`, `Delta_Psi_bootstrap_n`,
`Delta_Psi_bootstrap_failed`, `Delta_Psi_bootstrap_block_len`,
`iim_algorithm_version`, `tpm_estimator`, `cut_mode`, `psi_kernel`, selected
nodes, node-selection rule, bins requested/used, `budget_adjustments`, observed
states, transitions and `code_version`).
