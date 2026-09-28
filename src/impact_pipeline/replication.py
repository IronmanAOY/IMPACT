import os
import argparse
import glob
import json
from bids import BIDSLayout
from impact_pipeline.preprocessing import run_preprocessing
from impact_pipeline.synergy_ci import compute_synergy_ci
from impact_pipeline.analysis_bootstrap import (
    paired_bootstrap_mean_diff,
    paired_session_test,
)


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
                    condition='audio', theta=REPLICATION_THETA):
    """
    Exploratory replication of the legacy S contrast on an independent dataset.

    Requires a BIDS root with fMRIPrep derivatives (see _find_fmriprep_derivatives).
    Only S is computed (compute_mpc=False; S is the exploratory legacy statistic
    at the pre-declared ``theta``). Returns the paired subject-level difference
    delta_S = mean(S_sessions[0] − S_sessions[1]), its percentile bootstrap 95% CI
    ('ci', subjects resampled), and Cohen's dz ('cohend' = mean diff / SD of diffs).
    """
    if not os.path.isdir(str(data_root)):
        raise FileNotFoundError(f"Replication dataset not found at '{data_root}'.")
    deriv_root, n_deriv, searched = _find_fmriprep_derivatives(str(data_root))
    if deriv_root is None:
        raise RuntimeError(
            "No fMRIPrep desc-preproc BOLD outputs found for replication; searched: "
            + ", ".join(searched)
            + ". Run fMRIPrep (e.g. scripts/fetch_fmriprep.sh) first."
        )
    # Validate the BIDS root is readable before the expensive preprocessing step.
    BIDSLayout(data_root, validate=False)
    prep = os.path.join(out_dir, 'preprocessed')
    run_preprocessing(bids_root=data_root, fmriprep_deriv=deriv_root, out_root=prep)

    s0, s1 = sessions
    tr = _infer_tr_from_bold_sidecar(data_root)
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

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('data_root')
    p.add_argument('--out-dir', default='outputs/scratch/melbourne')
    p.add_argument('--atlas', default='schaefer400')
    p.add_argument('--sessions', nargs=2, default=['awake','deep'])
    args = p.parse_args()
    print(run_replication(args.data_root,
                          args.out_dir,
                          args.atlas,
                          tuple(args.sessions)))
