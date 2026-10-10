# Example derived protocols (NAS hub) - EXAMPLES, NOT PREREGISTERED

**These files are examples. Do not use them for a confirmatory analysis as
they are.** A hub is a substantive measurement choice: which nodes count as
the workspace decides what NAS measures. Before any confirmatory use, the
hub (and the rule that derives it) must be preregistered together with the
protocol hash, like every other protocol field. The protocol names start
with `EXAMPLE-` so that a verdict computed with them is recognisable in
every output (`MPC_protocol_hash`, `df.attrs['mpc_evidence']`).

## Why derived protocols

NAS `mode="capacity"` (the Network Availability Score, see the
[metrics reference](../../docs/metrics.md#42-modecapacity-network-availability-score-broadcast-capacity))
measures receive-transform-return transfer between a **declared** hub
(`workspace_nodes`) and the periphery. The default empirical protocol
[`mpc_default_v1.json`](../README.md#mpc_default_v1json-default-for-empirical-data)
(the preregistered protocol, which `run_pipeline.py` uses when `--protocol`
is not given) declares no hub, because the hub depends on the grain (atlas or
montage) of a dataset. Without a hub the pipeline records NAS as UNDEFINED
with the reason `UNDEFINED:NAS:NO_DECLARED_WORKSPACE` in every run (it does
not call the estimator and does not stop). A run that is meant to measure NAS
(for example an HLRS Hunter campaign) therefore passes a derived protocol that
declares the hub for its grain (`--protocol <file>`); everything else stays as
in the base protocol.

| File | Grain | Hub (example choice) | Nodes | Hash |
|---|---|---|---|---|
| `mpc_default_v1_schaefer400_7networks_hub.json` | `schaefer400` (Schaefer-2018, 400 parcels, 7 networks) | parcels of the `Cont`, `DorsAttn` and `SalVentAttn` networks (fronto-parietal control, dorsal attention, salience/ventral attention) | 145 of 400 | `421c6735597869b7a13775743a925516896337a6763f5458d0fc83591a4640ac` |
| `mpc_default_v1_eeg64_hub.json` | 64-channel 10-20 montage (BioSemi-64 labels) | fronto-parietal sensors F3, F1, Fz, F2, F4, P3, P1, Pz, P2, P4 | 10 of 64 | `fb26e17d6be8f52dbe9a0c5665f87f6fce37cc860bf999d5257909336afa078c` |

Both are derived from `mpc_default_v1.json` and differ from it only in
`estimators.NAS.workspace_nodes` and `name` (checked by
`tests/test_protocol_examples.py`), so RAM keeps its unimplemented
`perturbational` and `endogenous` channels and is never ABSENT under them.
Each protocol has a `.derivation.json` sidecar with its hash, the base
protocol, its name, hash and RAM channels, the hub rule, the source file and
its SHA-256, the node order, the hub labels and the resulting
`workspace_nodes`.

## How the indices are derived

`workspace_nodes` are 0-based row indices of the run's node x time matrix, so
they are only valid for the node order the pipeline writes:

- **Schaefer-400**: the network of each parcel is read from
  `atlases/schaefer_2018/Schaefer2018_400Parcels_7Networks_order.txt`
  (label names `7Networks_<hemisphere>_<network>_...`). The pipeline's
  `schaefer400` time series have one column per label value in ascending
  order (`NiftiLabelsMasker` on the label image), so node index = label
  value - 1. This holds only when all 400 labels are present in every run;
  check `n_nodes == 400` before use.
- **64-channel EEG**: the pipeline's EEG preprocessing keeps the channels
  common to a subject's runs **in sorted order**, so node index = position of
  the channel name in the sorted montage. This holds only when the kept
  scalp channels are exactly the 64 BioSemi labels; a recording with other
  labels needs its own derivation. **It does not hold for ds005620**, the
  pipeline's EEG dataset (atlas key `eeg64` as well): its BrainVision 10-10
  layout has TP9/TP10 instead of AF7/AF8/P9/P10, and its VEOG/HEOG channels
  (typed EEG) are excluded, so 62 scalp channels are kept. The example's
  indices are all below 62, so on ds005620 they are in range but name other
  channels (no error is raised). Derive the hub from the dataset's
  `channels.tsv` instead (`--eeg-channels`, below); the builder classifies a
  `channels.tsv` as the EEG preprocessing does (type column, EOG/EMG/ECG
  names and `status=bad` excluded).

An out-of-range or duplicate index is an invalid declaration and raises in
the estimator (it is never repaired silently). An index that is in range but
derived for another node order is not detectable by the pipeline; compare
the derivation's `channel_order` (or `n_nodes`) with the data.

## Caveats

- Scalp EEG: volume conduction mixes hub and periphery. NAS on sensor data
  needs the forward-model validation of the applicability registry
  (`eeg_like_forward` substrate) before its evidence can be accepted.
- The other modality-specific settings of the base protocol (for example the
  PDI repertoire window, which is in samples) are unchanged here; a
  confirmatory derived protocol has to preregister them too.
- A cohort reference (`cohort_high_state`) makes the high-state runs part of
  their own anchor. Necessity tests among report-positive episodes need an
  external reference, e.g. from `scripts/compute_empirical_reference.py`
  on a declared held-out subset of participants (with `--protocol <this
  example>`; RAM's reference is `RAM:default`, computed per channel in its
  `--data-dir` mode, and RAM's unimplemented channels get none).

## Rebuild or derive your own

```bash
python scripts/build_example_protocols.py --check   # the shipped files are a fresh build
python scripts/build_example_protocols.py           # rewrite them
# another montage or hub (never overwrites the shipped examples), e.g. ds005620:
python scripts/build_example_protocols.py \
    --eeg-channels <ds005620>/sub-1071/eeg/sub-1071_task-awake_acq-EC_channels.tsv \
    --out-dir my_protocols
```

`--check` fails when a shipped file differs from a fresh build or when this
directory holds a protocol file that is not part of it. Use a derived protocol
with `run_pipeline.py --protocol <file>`.

### From the opt-in behavioural-RAM protocol

```bash
python scripts/build_example_protocols.py \
    --base protocols/mpc_behavioural_ram_v1.json --out-dir my_protocols
python scripts/build_example_protocols.py \
    --base protocols/mpc_behavioural_ram_v1.json --out-dir my_protocols --check
```

`--base` derives the same hubs from another base protocol; file and protocol
names follow the base protocol's name (`mpc_behavioural_ram_v1_*.json`,
`EXAMPLE-mpc-behavioural-ram-v1-*`), and such a build needs `--out-dir` (this
directory holds the examples derived from the default protocol only). Under
[`mpc_behavioural_ram_v1.json`](../README.md#mpc_behavioural_ram_v1json-opt-in)
an ABSENT RAM means no responsiveness-and-adaptation above the null in the
recorded behaviour, not absence of responsiveness; the derivation sidecars
carry this caveat (`ram_channel_caveat`).
