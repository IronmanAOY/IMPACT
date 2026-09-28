import os
import argparse
import glob
import json
import sys
from pathlib import Path

from bids import BIDSLayout
from impact_pipeline.preprocessing import run_preprocessing
from impact_pipeline.synergy_ci import compute_synergy_ci
from impact_pipeline.analysis_bootstrap import (
    paired_bootstrap_mean_diff,
    paired_session_test,
)

# The Melbourne propofol fMRI dataset used by the replication step has no
# public source any more: its repository returns HTTP 404 (checked 2026-09).
# Replication therefore runs only on a local copy that the user supplies.
MELBOURNE_SOURCE_URL = "https://github.com/MelbourneHci/MelbournePropofolData"
REPLICATION_ROOT_ENV = "IMPACT_REPLICATION_ROOT"
# Legacy local folders (relative to the repository) searched when neither
# --replication-root nor IMPACT_REPLICATION_ROOT is given.
LEGACY_REPLICATION_DIRS = (
    ("data", "scratch", "melbourne"),
    ("data", "scratch", "melbourne_propofol"),
    ("data", "melbourne"),
    ("data", "melbourne_propofol"),
)
# Task labels of the replication data must follow this dataset's
# <condition><state> grammar (e.g. task-audioawake, task-audiodeep).
REPLICATION_TASK_GRAMMAR = "ds003171"
REPLICATION_HOWTO = (
    "The Melbourne propofol replication dataset is no longer publicly "
    f"available ({MELBOURNE_SOURCE_URL} returns HTTP 404), so it cannot be "
    "downloaded automatically. To run the replication (step 7, "
    "--run-replication), supply a local BIDS copy with fMRIPrep derivatives "
    "via --replication-root /path/to/dataset (or the "
    f"{REPLICATION_ROOT_ENV} environment variable); task labels must follow "
    f"the {REPLICATION_TASK_GRAMMAR} grammar (e.g. task-audioawake, "
    "task-audiodeep). Otherwise run without --run-replication."
)


class ReplicationDataUnavailable(FileNotFoundError):
    """The replication dataset is not available locally (no public source)."""


class ReplicationInputsMissing(FileNotFoundError, RuntimeError):
    """The replication dataset exists but lacks the inputs replication needs."""


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_replication_root(repo_root=None) -> Path:
    """
    Replication dataset folder when none is given explicitly:
    $IMPACT_REPLICATION_ROOT if set, else the first existing legacy folder
    (data/scratch/melbourne, ...), else the first legacy candidate (which does
    not exist; check_replication_inputs then explains how to proceed).
    """
    env = os.environ.get(REPLICATION_ROOT_ENV, "").strip()
    if env:
        return Path(env).expanduser()
    base = Path(repo_root) if repo_root is not None else _repository_root()
    candidates = [base.joinpath(*parts) for parts in LEGACY_REPLICATION_DIRS]
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


def resolve_replication_root(replication_root=None, repo_root=None) -> Path:
    """Explicit --replication-root if given, else ``default_replication_root``."""
    if replication_root not in (None, ""):
        return Path(os.fspath(replication_root)).expanduser()
    return default_replication_root(repo_root)


def check_replication_inputs(replication_root):
    """
    Fail fast (before any preprocessing or metric computation) unless the
    replication dataset is present: an existing folder with a BIDS
    dataset_description.json and fMRIPrep desc-preproc BOLD derivatives.

    Returns (fmriprep_derivatives_root, n_preproc_bold_files). Raises
    ReplicationDataUnavailable (a FileNotFoundError) when the dataset is absent
    and ReplicationInputsMissing (FileNotFoundError and RuntimeError) when it
    lacks derivatives; both messages say how to proceed.
    """
    if replication_root in (None, ""):
        raise ReplicationDataUnavailable(
            "Replication dataset not found: no --replication-root given. "
            + REPLICATION_HOWTO
        )
    root = Path(os.fspath(replication_root)).expanduser()
    if not root.is_dir():
        raise ReplicationDataUnavailable(
            f"Replication dataset not found at '{root}'. " + REPLICATION_HOWTO
        )
    if not (root / "dataset_description.json").is_file():
        raise ReplicationInputsMissing(
            f"'{root}' is not a BIDS dataset (no dataset_description.json). "
            + REPLICATION_HOWTO
        )
    deriv_root, n_deriv, searched = _find_fmriprep_derivatives(str(root))
    if deriv_root is None:
        raise ReplicationInputsMissing(
            "No fMRIPrep desc-preproc BOLD outputs found for replication; searched: "
            + ", ".join(searched)
            + ". Run fMRIPrep for the replication dataset first (e.g. "
            "scripts/fetch_fmriprep.sh --bids-root <dataset>) or run without "
            "--run-replication."
        )
    return deriv_root, n_deriv


def _infer_tr_from_bold_sidecar(bids_root):
    sidecars = sorted(
        glob.glob(os.path.join(bids_root, "sub-*", "func", "*_bold.json"))
        + glob.glob(os.path.join(bids_root, "sub-*", "ses-*", "func", "*_bold.json"))
    )
    if not sidecars:
        raise ValueError(
            "No *_bold.json sidecar found; explicit TR is required for NAS computation."
        )
    with open(sidecars[0], "r", encoding="utf-8") as f:
        sidecar = json.load(f)
    tr = sidecar.get("RepetitionTime")
    if tr is None:
        raise ValueError(
            f"Missing RepetitionTime in sidecar '{sidecars[0]}'; explicit TR is required."
        )
    tr = float(tr)
    if tr <= 0:
        raise ValueError(f"Invalid RepetitionTime={tr} in '{sidecars[0]}'.")
    return tr


def _replication_nas_params():
    return {
        "zthr": 1.0,
        "eps": 0.2,
        "tau": 0.2,
        "lambda_phase": 0.5,
        "alpha": 0.20,
        "beta": 0.16,
        "gamma": 0.14,
        "delta": 0.12,
        "eta": 0.16,
        "zeta": 0.12,
        "rho": 0.10,
        "bands": ((0.01, 0.10),),
        "band_weights": (1.0,),
        "window_len": 30,
        "step_len": 15,
        "max_triads": 5000,
        "random_state": 0,
        "workspace_nodes": None,
        "workspace_quantile": 0.2,
        "workspace_min_size": 4,
        "directed_lag": 1,
        "reverberation_lags": (2, 3, 4),
        "baseline_ts": None,
        "boost_against_baseline": False,
        "normalize": True,
    }


REPLICATION_THETA = 0.5  # pre-declared theta for the exploratory S replication


def _find_fmriprep_derivatives(data_root):
    """
    Locate an fMRIPrep derivatives folder with desc-preproc BOLD files.

    Searches <data_root>/derivatives/fmriprep, <data_root>/derivatives and
    <data_root> itself for sub-*/func/*desc-preproc_bold.nii.gz (optionally under
    ses-*). Returns (deriv_root, n_files, searched_roots).
    """
    candidates = [
        os.path.join(data_root, "derivatives", "fmriprep"),
        os.path.join(data_root, "derivatives"),
        data_root,
    ]
    for root in candidates:
        pattern = "*desc-preproc_bold.nii.gz"
        hits = glob.glob(os.path.join(root, "sub-*", "func", pattern))
        hits += glob.glob(os.path.join(root, "sub-*", "ses-*", "func", pattern))
        if hits:
            return root, len(hits), candidates
    return None, 0, candidates


def run_replication(data_root, out_dir, atlas, sessions, n_boot=2000, random_state=0,
                    condition='audio', theta=REPLICATION_THETA,
                    task_grammar=REPLICATION_TASK_GRAMMAR, assume_tr=None):
    """
    Exploratory replication of the legacy S contrast on an independent dataset.

    Requires a local BIDS root with fMRIPrep derivatives (the original public
    source is gone, see REPLICATION_HOWTO); ``check_replication_inputs`` fails
    fast with an actionable message otherwise. Task labels are mapped to
    (state, condition) with the ``task_grammar`` dataset's grammar; only the
    requested ``sessions`` are preprocessed, and a run set without both
    sessions is an error, not an empty result.
    Only S is computed (compute_mpc=False; S is the exploratory legacy statistic
    at the pre-declared ``theta``). Returns the paired subject-level difference
    delta_S = mean(S_sessions[0] − S_sessions[1]), its percentile bootstrap 95% CI
    ('ci', subjects resampled), and Cohen's dz ('cohend' = mean diff / SD of diffs).
    """
    deriv_root, n_deriv = check_replication_inputs(data_root)
    data_root = str(Path(os.fspath(data_root)).expanduser())
    # Validate the BIDS root and the TR before the expensive preprocessing step.
    BIDSLayout(data_root, validate=False)
    if assume_tr is None:
        tr = _infer_tr_from_bold_sidecar(data_root)
    else:
        tr = float(assume_tr)  # explicit --assume-tr, as in preprocessing
    prep = os.path.join(out_dir, 'preprocessed')
    s0, s1 = sessions
    prep_summary = run_preprocessing(
        bids_root=data_root,
        fmriprep_deriv=deriv_root,
        out_root=prep,
        assume_tr=assume_tr,
        dataset_id=task_grammar,
        states=(s0, s1),
    )
    written = (prep_summary or {}).get("written_runs") or []
    states_written = {
        str(r.get("state")) for r in written if str(r.get("condition")) == condition
    }
    if not {s0, s1} <= states_written:
        skipped = sorted(
            {str(r.get("task")) for r in (prep_summary or {}).get("skipped_runs") or []}
        )
        raise ReplicationInputsMissing(
            f"Replication preprocessing produced no '{condition}' runs for "
            f"session(s) {sorted({s0, s1} - states_written)} (tasks skipped as "
            f"outside the {task_grammar} grammar or state list: {skipped}). "
            "Replication needs task labels such as "
            f"task-{condition}{s0} and task-{condition}{s1}."
        )

    df = compute_synergy_ci(
        prep,
        atlas,
        thetas=[float(theta)],
        sessions=sessions,
        condition=condition,
        tr=tr,
        compute_mpc=False,
        compute_ci=False,
    )
    boot = paired_bootstrap_mean_diff(
        df, 'S', sessions=(s0, s1), n_boot=n_boot, random_state=random_state
    )
    test = paired_session_test(df, 'S', sessions=(s0, s1))
    return {
        'dataset': os.path.basename(os.path.normpath(str(data_root))),
        'statistic': 'S (exploratory legacy HypergraphSynergy)',
        'theta': float(theta),
        'delta_S': boot['mean_diff'],
        'ci': (boot['lo'], boot['hi']),
        'ci_kind': 'paired subject bootstrap 95% CI of mean difference',
        'cohend': test['dz'],
        't': test['t'],
        'p': test['p'],
        'n_pairs': test['n_pairs'],
        'fmriprep_derivatives': deriv_root,
        'n_preproc_bold_files': int(n_deriv),
    }


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Exploratory replication of the S contrast. " + REPLICATION_HOWTO
    )
    p.add_argument(
        'data_root',
        nargs='?',
        default=None,
        help="Legacy positional form of --replication-root.",
    )
    p.add_argument(
        '--replication-root',
        default=None,
        help=(
            "Local BIDS copy of the replication dataset with fMRIPrep derivatives "
            f"(default: ${REPLICATION_ROOT_ENV}, then data/scratch/melbourne)."
        ),
    )
    p.add_argument('--out-dir', default='outputs/scratch/melbourne')
    p.add_argument('--atlas', default='schaefer400')
    p.add_argument('--sessions', nargs=2, default=['awake', 'deep'])
    args = p.parse_args(argv)
    root = resolve_replication_root(args.replication_root or args.data_root)
    try:
        # Fail fast, before anything is computed.
        check_replication_inputs(root)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(run_replication(root, args.out_dir, args.atlas, tuple(args.sessions)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
