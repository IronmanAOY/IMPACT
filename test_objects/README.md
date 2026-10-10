# Synthetic Validation Data

This folder is reserved for synthetic validation datasets and outputs derived
from them. In the repository it holds only this README and empty
placeholders; it fills when the objects are generated (see
`docs/synthetic_data.md`). The same layout is used for the
synthetic-data archive and for locally generated synthetic datasets.

Use it for:

- synthetic BIDS-style datasets under `test_objects/datasets/`
- pipeline outputs for synthetic validation runs under `test_objects/runs/`
- reusable validation metric exports under `test_objects/metric_bank/`
- the generator's manifests, source inspections and validation reports under
  `test_objects/real_derived_synth_completed/reports/`

Rules:

- synthetic validation data and derived outputs stay here
- real study data and real-study outputs stay outside this folder
- metric-bank exports stored here are selected explicitly by downstream runs
- analyses that mix sources for the legacy Consciousness Index (`CI`) must
  keep the dataset-origin labels

The dashboard and pipeline enforce this separation when a dataset is marked as
synthetic. See `docs/synthetic_data.md` for archive and generation commands.
