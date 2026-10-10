Bootstrap: docker
From: condaforge/miniforge3:26.7.2-0@sha256:eeb947cc87d61d46820b123bd7c26e1cbdc4b182ff7d0331e501a32a936b82e3

# IMPaCT Synergy Pipeline, Apptainer/Singularity image. Same content as the
# Dockerfile: pinned conda environment + code + protocols + atlases, no data.
# MPC-Bench confirmatory runs need a git checkout of the freeze tags and are
# not supported in the image.
#
# Build from the repository root:
#   apptainer build impact-synergy-pipeline_1.1.0.sif Singularity
# or convert a locally built Docker image instead:
#   apptainer build impact-synergy-pipeline_1.1.0.sif docker-daemon://impact-synergy-pipeline:1.1.0
#
# Run (the image is read-only; write outputs to a bound folder):
#   apptainer run -B /abs/path/ds005620:/data:ro -B /abs/path/outputs:/out \
#     impact-synergy-pipeline_1.1.0.sif \
#     --dataset-id ds005620 --bids-root /data --out-dir /out \
#     --run-preprocessing --mpc-metrics PDI NAS IIM --no-ci
# Synthetic ("dummy"-origin) runs: --env IMPACT_SYNTH_ROOT=/synth -B /abs/path/to/synth:/synth

%files
  environment.yml /opt/impact/environment.yml
  pyproject.toml /opt/impact/pyproject.toml
  README.md /opt/impact/README.md
  LICENSE /opt/impact/LICENSE
  CITATION.cff /opt/impact/CITATION.cff
  run_pipeline.py /opt/impact/run_pipeline.py
  src /opt/impact/src
  scripts /opt/impact/scripts
  protocols /opt/impact/protocols
  atlases /opt/impact/atlases
  licenses/MIT_LICENSE /opt/impact/licenses/MIT_LICENSE
  licenses/THIRD_PARTY_NOTICES.md /opt/impact/licenses/THIRD_PARTY_NOTICES.md
  licenses/fs_license.txt.example /opt/impact/licenses/fs_license.txt.example

%post
  set -eu
  # Drop bytecode/numba caches and macOS .DS_Store files copied from the build
  # host (the Dockerfile build context excludes them via .dockerignore).
  find /opt/impact -name __pycache__ -type d -prune -exec rm -rf {} +
  find /opt/impact -name .DS_Store -type f -delete
  /opt/conda/bin/conda env create -f /opt/impact/environment.yml
  /opt/conda/bin/conda clean -afy
  /opt/conda/envs/impact-synergy-clean/bin/python -m pip install \
    --no-deps --no-build-isolation --no-cache-dir --root-user-action=ignore -e /opt/impact
  # Provenance routing expects this scaffold under the repository root.
  mkdir -p /opt/impact/test_objects/datasets /opt/impact/test_objects/runs \
    /opt/impact/test_objects/metric_bank /data /out

%environment
  export LANG=C.UTF-8
  export LC_ALL=C.UTF-8
  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONUNBUFFERED=1
  export MPLBACKEND=Agg
  export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
  export CONDA_DEFAULT_ENV=impact-synergy-clean
  export CONDA_PREFIX=/opt/conda/envs/impact-synergy-clean
  export PATH="/opt/conda/envs/impact-synergy-clean/bin:${PATH}"

%runscript
  # Same working directory as the Docker image (WORKDIR). fMRI preprocessing
  # reads atlases/ next to pyproject.toml (or $IMPACT_ATLAS_DIR) regardless.
  cd /opt/impact
  exec python /opt/impact/run_pipeline.py "$@"

%test
  cd /opt/impact
  python -c "import impact_pipeline, mne, nilearn, numba, bids; print('impact_pipeline import OK')"
  bash scripts/download_atlases.sh --verify

%labels
  org.opencontainers.image.title IMPaCT Synergy Pipeline
  org.opencontainers.image.version 1.1.0
  org.opencontainers.image.source https://github.com/IronmanAOY/IMPACT
  org.opencontainers.image.licenses MIT

%help
  IMPaCT Synergy Pipeline 1.1.0. The runscript calls run_pipeline.py; see the
  header of this definition file for build and run examples.
