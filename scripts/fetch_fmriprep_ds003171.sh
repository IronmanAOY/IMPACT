#!/usr/bin/env bash
# fMRIPrep for ds003171 (propofol fMRI). Thin wrapper around fetch_fmriprep.sh:
# BIDS root <repo>/data/scratch/ds003171 unless --bids-root is given, and
# derivatives in <bids-root>/derivatives/fmriprep, where run_pipeline.py reads
# them (--fmriprep-dir). All options of fetch_fmriprep.sh are accepted, e.g.
#
#   export FS_LICENSE=/abs/path/to/license.txt
#   bash scripts/fetch_fmriprep_ds003171.sh --skip-reconall
#   bash scripts/fetch_fmriprep_ds003171.sh --participant-label 02CB 04HD
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/fetch_fmriprep.sh" --dataset-id ds003171 "$@"
