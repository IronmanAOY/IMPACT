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
| `mpc_default_v1.json` | default protocol for empirical data (`run_pipeline.py --protocol`) | `383eb310cf4479d3260bc5f43f8972bd0b12104a221579fea17c40e410357267` |
| `mpc_bench_v1.json` | MPC-Bench protocol (`run_bench.py`, `null_calibration.py`; their default) | `02a841a9db55b2a68ce33f6df60895249fa54bf3292323db236884b855ed2e7b` |
| `mpc_bench_v1_reference_summary.json` | how the bench reference anchor was computed (seeds, n, SD per principle, code) | — |

`tests/test_protocols.py` checks that the hashes in this table are the hashes
of the files.

## `mpc_default_v1.json`

- **Necessity set**: all five principles (RAM, PDI, NAS, IIM, SRPI).
- **Channels**: one untyped `default` channel for PDI, NAS, IIM and SRPI. RAM
  declares `default` (the run's events without a channel label) **and** the
  two channels that have no estimator yet, `perturbational` and `endogenous`.
  A declared channel that is not implemented is UNDEFINED
  (`NOT_IMPLEMENTED:RAM:<channel>`), and a principle is ABSENT only when every
  declared channel is ABSENT (strong-Kleene OR), so under this protocol RAM
  can be PRESENT (through `default`) but never ABSENT: an unresponsive
  behavioural record cannot exclude consciousness while perturbational and
  endogenous responsiveness are unmeasured (spec V2-3). Dropping these
  channels in a derived protocol is a substantive decision that has to be
  justified.
- **Cutoffs**: `(z, delta) = (0.25, 0.10)` for every principle, `alpha =
  0.05`: the initial construct-scale defaults of spec V2-2, to be justified
  by the MPC-Bench dose-response before the freeze.
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
  report-positive episodes need an external reference.
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
    when the hub set is known);
  - IIM: calibrated Delta_Psi with the `node_shrinkage` TPM and
    `cut_mode='bidirectional'` (`'directional'` is available);
  - SRPI: `mode='agency'` where the run has self_caused/other_caused events,
    otherwise `mode_fallback='legacy'` (recorded in `SRPI_mode_reason`).
- **Bearer nodes**: all nodes of the run.

## `mpc_bench_v1.json` (provisional)

The bench protocol mirrors the bench's estimator modes
(`bench.export.OPTIONAL_MODES`) and the null families its runner records.
The bench has no cohort, so its reference is `external`: the mean excess over
its own null of the nominal positive control (witness `PC_nominal`, family A)
on development reference seeds 900-907, computed by
`scripts/bench_reference.py` (see `mpc_bench_v1_reference_summary.json`).
These anchors come from development seeds and are not results.

Provisional parts, to be finalised in the calibration phase:

- the cutoffs `(0.25, 0.10)` are placeholders;
- on the reference seeds the PDI repertoire estimate of the positive control
  equals its null (one distinguishable state at the default window), so the
  PDI anchor is 0 and PDI evidence is UNDEFINED (`INVALID_ANCHORS`); the RAM
  anchor is within about one SE of 0, so RAM evidence is effectively never
  determinate. Both need construct work on development data before the
  bench can produce MPC_CONSISTENT verdicts with PDI or RAM in `N`.

Regenerate the reference (development seeds only; family C and seeds >= 10000
are refused):

```bash
python scripts/bench_reference.py --run --family A --seeds 900-907 \
    --null-surrogates 19 --se-groups 0 --workers 4 \
    --work-dir outputs/paper1_mpcbench/reference_A \
    --template protocols/mpc_bench_v1.json --out protocols/mpc_bench_v1.json
```

## Use

```bash
python run_pipeline.py --dataset-id ds003171 \
    --bids-root /data/openneuro/ds003171 --out-dir outputs/ds003171 \
    --protocol protocols/mpc_default_v1.json --null-surrogates 19 --bootstrap-se 100
python scripts/run_bench.py factorial --seeds 0-19 --null-surrogates 19 \
    --se-groups 5 --protocol protocols/mpc_bench_v1.json --out outputs/bench/factorial_A
python scripts/null_calibration.py --protocol protocols/mpc_bench_v1.json --out out/null
```

Before the freeze: finalise the cutoffs and the bench reference, then record
the hash of the frozen protocol in `predictions/registry.yaml`
(`protocols[].hash`).
