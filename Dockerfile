# syntax=docker/dockerfile:1
#
# IMPaCT Synergy Pipeline runtime image: run_pipeline.py with the pinned conda
# environment. It contains no data and does not run fMRIPrep (fMRIPrep is its
# own container, see scripts/fetch_fmriprep.sh). MPC-Bench confirmatory runs
# need a git checkout of the freeze tags and are not supported in the image.
#
# Build from the repository root (.dockerignore keeps data/, outputs/,
# test_objects/, dist/ and manuscripts out of the build context):
#   docker build -t impact-synergy-pipeline:1.1.0 .
#
# Run against a mounted BIDS dataset and output folder:
#   docker run --rm --user "$(id -u):$(id -g)" \
#     -v /abs/path/to/ds005620:/data:ro -v /abs/path/to/outputs:/out \
#     impact-synergy-pipeline:1.1.0 \
#     --dataset-id ds005620 --bids-root /data --out-dir /out \
#     --run-preprocessing --mpc-metrics PDI NAS IIM --no-ci
#
# For fMRI, also mount the fMRIPrep derivatives (canonical location
# <bids-root>/derivatives/fmriprep, i.e. inside the /data mount).
# Synthetic ("dummy"-origin) runs write under $IMPACT_SYNTH_ROOT/test_objects:
# add -e IMPACT_SYNTH_ROOT=/synth -v /abs/path/to/synth:/synth.
# Tests (not copied into the image):
#   docker run --rm -v "$PWD/tests:/opt/impact/tests:ro" --entrypoint python \
#     impact-synergy-pipeline:1.1.0 -m pytest -q -p no:cacheprovider tests

# Base image pinned by tag and multi-arch index digest (linux/amd64 + arm64).
FROM condaforge/miniforge3:26.7.2-0@sha256:eeb947cc87d61d46820b123bd7c26e1cbdc4b182ff7d0331e501a32a936b82e3

LABEL org.opencontainers.image.title="IMPaCT Synergy Pipeline" \
      org.opencontainers.image.version="1.1.0" \
      org.opencontainers.image.source="https://github.com/IronmanAOY/IMPACT" \
      org.opencontainers.image.licenses="MIT"

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba_cache

# Conda environment first, so code changes do not invalidate this layer.
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml \
    && conda clean -afy \
    && rm /tmp/environment.yml

# run_pipeline.py checks that it runs inside the impact-synergy-clean env.
ENV CONDA_DEFAULT_ENV=impact-synergy-clean \
    CONDA_PREFIX=/opt/conda/envs/impact-synergy-clean \
    PATH=/opt/conda/envs/impact-synergy-clean/bin:${PATH}

# fMRI preprocessing reads atlases/ next to pyproject.toml (or $IMPACT_ATLAS_DIR).
WORKDIR /opt/impact
COPY pyproject.toml environment.yml README.md LICENSE CITATION.cff run_pipeline.py ./
COPY licenses/ licenses/
COPY atlases/ atlases/
COPY src/ src/
COPY scripts/ scripts/
# MPC protocols: run_pipeline.py's default for empirical data
# (protocols/mpc_default_v1.json), the example derived protocols and the
# other declared protocols.
COPY protocols/ protocols/

# Install the package (dependencies come from the conda env) and pre-create the
# synthetic-data scaffold that provenance routing expects under the repo root.
RUN python -m pip install --no-deps --no-build-isolation --no-cache-dir --root-user-action=ignore -e . \
    && python -c "import impact_pipeline, mne, nilearn, numba, bids; print('impact_pipeline import OK')" \
    && bash scripts/download_atlases.sh --verify \
    && mkdir -p test_objects/datasets test_objects/runs test_objects/metric_bank /data /out \
    && chown -R 1000:1000 test_objects /out

# Unprivileged default user (the base image's uid 1000); override with --user.
USER 1000:1000

ENTRYPOINT ["python", "/opt/impact/run_pipeline.py"]
CMD ["--help"]
