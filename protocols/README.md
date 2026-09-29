# MPC protocols

A protocol declares everything an MPC verdict depends on besides the data:
the necessity set `N`, the evidence channels, the construct-scale cutoffs
`(z, delta)` and `alpha`, the null families, the reference anchor, the source
rule, the estimator modes and the bearer nodes. It is an
`impact_pipeline.evidence.Protocol` stored as JSON (schema
`impact-mpc-protocol/2`). Its hash is the SHA-256 of the canonical JSON of
`Protocol.to_dict()` (sorted keys, compact separators, every cutoff written
out) and is recorded with every verdict (`MPC_protocol_hash`, the bench
manifests and verdicts, the null-calibration summary). The
[metrics reference](../docs/metrics.md#83-combining-evidence)
describes the fields.

| File | Use | SHA-256 |
|---|---|---|
| `mpc_default_v1.json` | default protocol for empirical data (`run_pipeline.py` without `--protocol`, `--data-origin real`); preregistered and confirmed at the freeze, unchanged since | `383eb310cf4479d3260bc5f43f8972bd0b12104a221579fea17c40e410357267` |
| `mpc_behavioural_ram_v1.json` | **opt-in only**, a substantive choice to preregister before use (`--protocol protocols/mpc_behavioural_ram_v1.json`): v1 with RAM declared on its behavioural channel only; results under it are behavioural-RAM results | `531c15b90ea0338402f3eb2df97ae27ab9a9d9a5ffb22e17c93413231ff220b8` |
| `mpc_bench_v1.json` | MPC-Bench protocol, all five principles (frozen at `mpcbench-freeze-v1`; `run_bench.py`, `null_calibration.py`; their default) | `855f6a444b77030d33faa68fcb45e8576b931d2d681cf215d4dacdb57a6b2520` |
| `mpc_bench_v1_anchored.json` | the same with the necessity set restricted to the anchored principles (NAS, IIM, SRPI); verdict-level hypotheses of the preregistration | `780581f57d24251fc8ed8c39565f0a89a96740908f2ada3f65f8961cf1543f4c` |
| `mpc_bench_v1_reference_summary.json` | how the bench reference anchor was computed (seeds, n, mean, SD, SE and lower bound per principle, the anchor rule, code) | — |
| `applicability_registry_v1.json` | applicability registry (`--applicability-registry`; schema `impact-mpc-registry/2`) derived from the confirmatory MPC-Bench runs after the freeze; SHA-256 of the file | `a300e08f6bd04b385896272af44a63d4a9c8f63c3828547f2e1aa1167f663ca9` |
| [`examples/`](examples/README.md) | **examples** of protocols derived from `mpc_default_v1.json` that declare a NAS hub (Schaefer-400 / 7 networks, 64-channel EEG); not preregistered | see `examples/*.derivation.json` |

`tests/test_protocols.py` checks that the hashes in this table are the hashes
of the files, and `scripts/bench_hypotheses.py` refuses bench protocols whose
hashes differ from the frozen ones. The choices below were made on
development data only (seeds 0-999, family A; family B for IIM); the
evidence and the decision rules are in the
[preregistration](../docs/preregistration/MPC_BENCH_PREREGISTRATION.md),
section 4.

## Cutoffs and alpha (all protocols)

`(z, delta) = (0.25, 0.10)` for every principle and `alpha = 0.05`
(one-sided), unchanged from the initial defaults of spec V2-2 because the
development runs met the preregistered conditions for keeping them: no
false PRESENT of an anchored principle on the development null systems and
no false ABSENT anywhere (see the preregistration for the rates). `z` is the
smallest effect of interest for presence (a quarter of the nominal positive
control's excess over its null); `delta < z` keeps "absent" (negligible, a
tenth of it) distinct from "weak". At the bench's precision ABSENT is
reached only when an estimate sits at its null with a small SE (null
systems), which makes the exclusion rule abstain on most single-deficit
systems (a development finding, not a tuning target).

## `mpc_default_v1.json` (default for empirical data)

`run_pipeline.py` uses it when `--protocol` is not given, the data are
empirical (`--data-origin real`, the default) and no `--necessity-set` is
given (`--protocol none` builds the protocol from the flags, as at the freeze;
dummy/synthetic data always do). It is the empirical default protocol of the
[preregistration](../docs/preregistration/MPC_BENCH_PREREGISTRATION.md)
(section 3, and the "Default protocol" row of section 4) and the one paper 2
cites, byte-for-byte as at tag `mpcbench-freeze-v1` (hash above;
`tests/test_protocol_examples.py` checks the hash and the file's SHA-256).

Why this default. RAM declares its implemented untyped `default` channel
(behavioural events) **and** `perturbational` and `endogenous`, which have no
estimator yet. A declared channel without an estimator is UNDEFINED
(`NOT_IMPLEMENTED:RAM:<channel>`), and a principle is ABSENT only when every
declared channel is ABSENT (strong-Kleene OR), so under v1 RAM can be PRESENT
but never ABSENT. That is the preregistered stance, confirmed at the freeze:
behavioural non-response alone must never exclude, because covert,
perturbational and endogenous responsiveness (covert command following,
dreaming, paralysis) are not measured. Undefined is not absent. The price is
that RAM cannot contribute an exclusion on empirical data, and a necessity
test of RAM cannot be falsified by RAM evidence, until those channels are
measured.

NAS. v1 declares NAS `mode='capacity'` without a hub, because the hub depends
on the grain (atlas or montage). Without a declared hub NAS is UNDEFINED in
every run (`UNDEFINED:NAS:NO_DECLARED_WORKSPACE`, 1.1.0 post-freeze; the
frozen code raised instead), so a run that is meant to measure NAS (e.g. a
Hunter campaign) needs a protocol derived from v1 that declares the hub:
[`examples/`](examples/README.md) has two worked examples (derived from v1,
to be preregistered before confirmatory use).

The declarations of v1 are listed in the next section.

## `mpc_behavioural_ram_v1.json` (opt-in)

`mpc_default_v1.json` with one change: RAM declares only its implemented
behavioural channel, `channels.RAM = ["default"]` (name
`mpc-behavioural-ram-v1`). Everything else (necessity set, cutoffs, alpha,
null families, reference, source rule, estimator modes, bearers) is
identical; `tests/test_protocol_examples.py` checks this. It is never
selected automatically: `run_pipeline.py --protocol
protocols/mpc_behavioural_ram_v1.json` only.

Under it RAM can be ABSENT when the behavioural channel is credibly at its
null, so RAM can exclude. Selecting it is a substantive decision that has to
be preregistered (with its hash) before use, and results under it are
reported as **behavioural-RAM results**: an ABSENT RAM means "no
responsiveness-and-adaptation above the null in the recorded behaviour", not
absence of responsiveness (covert, perturbational and endogenous
responsiveness are not measured by this channel). A dataset that measures
covert or perturbational responsiveness declares those typed channels in a
derived protocol instead (for example `["behavioural_feedback",
"covert_neural"]` for events with an `impact_channel` column), so that RAM is
ABSENT only when every measured channel is. `scripts/build_example_protocols.py
--base protocols/mpc_behavioural_ram_v1.json --out-dir <dir>` derives the
example hub protocols from it (their sidecars carry this caveat).

History: an intermediate draft of the 1.1.0 post-freeze fixes had made this
protocol the command-line default under another name; that reversed the
preregistered stance above and was undone (see the
[changelog](../CHANGELOG.md)).

## `mpc_default_v1.json`: declarations (unchanged since the freeze)

- **Necessity set**: all five principles (RAM, PDI, NAS, IIM, SRPI).
- **Channels**: one untyped `default` channel for PDI, NAS, IIM and SRPI. RAM
  declares `default` (the run's events without a channel label) **and** the
  two channels that have no estimator yet, `perturbational` and `endogenous`.
  A declared channel that is not implemented is UNDEFINED
  (`NOT_IMPLEMENTED:RAM:<channel>`), and a principle is ABSENT only when every
  declared channel is ABSENT (strong-Kleene OR), so under this protocol RAM
  can be PRESENT (through `default`) but never ABSENT: an unresponsive
  behavioural record cannot exclude consciousness while perturbational and
  endogenous responsiveness are unmeasured (spec V2-3). This stance was
  confirmed at the freeze: behavioural non-response does not establish the
  absence of responsiveness and adaptation (covert responsiveness,
  dreaming, paralysis), so a necessity-only rule must not exclude on it.
  Dropping these channels (as `mpc_behavioural_ram_v1.json` does) is a
  substantive decision that has to be justified and preregistered.
- **Cutoffs**: `(z, delta) = (0.25, 0.10)` for every principle, `alpha =
  0.05`, as the bench protocol.
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
    in `RAM_estimator` and `RAM_mode_reason`). The fallback estimator is not
    registered in `predictions/registry.yaml`, so its rows are refused by
    `run_predictions.py` (a decision recorded there);
  - PDI: `mode='repertoire'` (unlabelled repertoire of distinguishable
    states; the window is in samples, so modality-specific windows need a
    derived protocol; scalp EEG needs the forward-model validation of the
    applicability registry first);
  - NAS: `mode='capacity'` (declare `workspace_nodes` in a derived protocol
    when the hub set is known; without it NAS is UNDEFINED,
    `NO_DECLARED_WORKSPACE`; see [`examples/`](examples/README.md));
  - IIM: calibrated Delta_Psi with the `node_shrinkage` TPM and
    `cut_mode='bidirectional'` (kept: on the development bench the
    directional mode showed no dose-response to the recurrent loops, see the
    preregistration; `'directional'` is available);
  - SRPI: `mode='agency'` where the run has self_caused/other_caused events,
    otherwise `mode_fallback='legacy'` (recorded in `SRPI_mode_reason`; not
    registered either).
- **Bearer nodes**: all nodes of the run.
- Recommended sampling settings for empirical runs: `--null-surrogates 19`
  and `--bootstrap-se 100` (moving-block bootstrap; `se_df = 99`, so the
  Student-t quantile is within 1% of the normal one; the Monte-Carlo
  relative error of the SE is about 7%); on Hunter
  `--hunter-iim-bootstrap-se 100` (IIM cost grows with `1 + B`).

## `mpc_bench_v1.json` (frozen)

- **Necessity set**: all five principles; **channels**: `default` only (the
  bench agent's responsiveness mechanism is behavioural by construction).
- **Estimator modes** (what the bench computes; the runner applies the
  protocol's declared options and refuses modes it does not compute): RAM
  `update='prediction_error'`, PDI `mode='repertoire'`, NAS
  `mode='capacity'`, SRPI `mode='agency'`, IIM `cut_mode='bidirectional'`,
  `tpm_estimator='node_shrinkage'`. The IIM grain is the generators'
  declared one: 4 macro nodes (the means of the periphery modules; the
  workspace hub is excluded so IIM does not measure the workspace loop),
  2 bins, a lag of 2 samples (0.1 s, one unit time constant); 16 states and
  256 transitions for about 11,900 samples per run.
- **Null families**: RAM `onset_jitter`, PDI `circular_shift`, NAS
  `block_circular_shift`, IIM `circular_shift`, SRPI
  `yoked_label_permutation`; K = 19 surrogates per run (bench runs), for
  which the Monte-Carlo error of the null mean contributes about 5% (NAS)
  or less of the variance of `c` on the development positive control.
- **Reference**: `external`, scale `excess`: the mean excess over its own
  null of the nominal positive control (`PC_nominal`, family A) on
  development reference seeds 900-919 (n = 20), computed by
  `scripts/bench_reference.py` with the **anchor rule**: a principle gets an
  anchor only if the one-sided 95% Student-t lower bound of that mean excess
  is positive. NAS (0.0706, SE 0.0023), IIM (0.0094, SE 0.0016) and SRPI
  (0.164, SE 0.012) are anchored. RAM (mean 0.0050, SE 0.0041) and PDI (0: the
  unlabelled repertoire counts one state on every reference seed) are not,
  so their evidence is UNDEFINED (`INVALID_ANCHORS`) on the bench and a
  bench verdict with all five principles is never `MPC_CONSISTENT`. These
  are development values, not results.
- **Sampling SE**: the delete-a-group jackknife with G = 10 groups (`se_df =
  9`, Student-t bounds): on the development positive control its SE is 0.6
  to 0.9 times the between-seed SD of the estimate (the latter also
  includes network variability).
- Family C (held out) is judged with this protocol and its own reference
  (the mean excess of its `PC_nominal` on confirmatory seeds 19000-19019,
  same anchor rule), everything else unchanged
  (`bench.analysis.protocol_for_family`).

## `mpc_bench_v1_anchored.json` (frozen)

`mpc_bench_v1.json` with the necessity set `{NAS, IIM, SRPI}`, the principles
with a valid construct scale on the bench (and name
`mpc-bench-v1-anchored`). The verdict-level hypotheses of the
preregistration use it, because with RAM and PDI in `N` every bench verdict
is `UNDETERMINED` or `EXCLUDED` by construction. It is not a claim that RAM
and PDI are unnecessary.

## `applicability_registry_v1.json` (derived after the confirmatory runs)

Built by `scripts/build_applicability_registry.py` from the confirmatory
MPC-Bench results of the frozen code (families A and C, the null-calibration
grid, family B and the whole-brain runs; seeds >= 10000). It applies the
entry criteria of the
[preregistration](../docs/preregistration/MPC_BENCH_PREREGISTRATION.md),
section 4, unchanged: (0) a valid anchor, (a) false-PRESENT rate <= 0.07 on
every statistical null class, (b1) a dose-response of `c` to the own knob,
(b2) PRESENT rate <= 0.07 on the systems without the mechanism, (c) the own
switch changes `c` more than any other switch. It was built after the
preregistered hypotheses had been evaluated and changes none of them.

| Principle | Estimator | Substrate | Regime |
|---|---|---|---|
| SRPI | `compute_SRPI:agency@srpi-v2-2026.09` | `synthetic_rate` (family A) | 30 nodes, >= 11,497 samples at 0.05 s |
| NAS | `compute_NAS:capacity@nas-v2-2026.09` | `stuart_landau` (family C) | 30 nodes, >= 11,497 samples at 0.05 s |
| IIM | `compute_IIM:bidirectional@iim-v4-2026.09` | `stuart_landau` (family C), grain `module_mean_periphery` | 4 macro nodes, 2 bins, >= 11,497 samples at 0.05 s |

Every other combination the confirmatory runs reached is listed under
`excluded` with the criteria it fails: RAM and PDI have no anchor in either
family; NAS and IIM fail (b2) and (c) in family A; SRPI has no anchor in
family C; IIM on the exact-TPM systems (family B) and every estimator on the
EEG-like and BOLD-like forward models cannot meet the criteria (no anchor, no
mechanism switches; all BOLD-like tasks failed in the frozen runner). Each
entry also lists the related preregistered outcomes, which are stricter than
the entry criteria (for example, HC3 is falsified for IIM in family C) and
have to be reported with it.

No entry covers a forward-modelled substrate, so with this registry every
component of human EEG or fMRI data is UNDEFINED
(`ESTIMATOR_NOT_VALIDATED`) and every empirical verdict is UNDETERMINED.

```bash
python scripts/build_applicability_registry.py --results outputs/paper1_mpcbench \
    --out protocols/applicability_registry_v1.json \
    --evidence-dir outputs/paper1_mpcbench/registry
```

## Regenerate the reference

Development seeds only (family C and seeds >= 10000 are refused):

```bash
python scripts/bench_reference.py --run --family A --seeds 900-919 \
    --null-surrogates 19 --se-groups 10 --workers 4 \
    --work-dir outputs/paper1_mpcbench/dev/reference_A/run \
    --template protocols/mpc_bench_v1.json --out protocols/mpc_bench_v1.json \
    --name mpc-bench-v1
```

## Use

```bash
# default protocol (v1; NAS UNDEFINED without a hub)
python run_pipeline.py --dataset-id ds003171 \
    --bids-root /data/openneuro/ds003171 --out-dir outputs/ds003171 \
    --null-surrogates 19 --bootstrap-se 100
# v1 with a declared NAS hub (an example: preregister the hub first)
python run_pipeline.py --dataset-id ds003171 \
    --bids-root /data/openneuro/ds003171 --out-dir outputs/ds003171 \
    --protocol protocols/examples/mpc_default_v1_schaefer400_7networks_hub.json \
    --null-surrogates 19 --bootstrap-se 100
# external reference from held-out participants (per channel: RAM:default)
python scripts/compute_empirical_reference.py \
    --data-dir outputs/ds003171/preprocessed --atlas schaefer400 \
    --bids-root /data/openneuro/ds003171 --dataset-id ds003171 \
    --protocol protocols/examples/mpc_default_v1_schaefer400_7networks_hub.json \
    --reference-subjects @heldout.txt --evaluation-subjects @evaluation.txt \
    --out outputs/ds003171_reference/empirical_reference.json \
    --write-protocol outputs/ds003171_reference/mpc_default_v1_hub_external.json
python scripts/run_bench.py factorial --seeds 0-19 --null-surrogates 19 \
    --se-groups 10 --protocol protocols/mpc_bench_v1.json --out outputs/bench/factorial_A
python scripts/null_calibration.py --protocol protocols/mpc_bench_v1.json --out out/null
```

The paper-2 registry (`predictions/registry.yaml`) stays a draft; its
protocol hash is filled at its own freeze.
