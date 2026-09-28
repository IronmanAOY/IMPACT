import os
import glob
import json
import logging
import warnings
import re
from pathlib import Path

import numpy as np
import pandas as pd
from bids import BIDSLayout
from nilearn import image
from nilearn.maskers import NiftiLabelsMasker
from nilearn.signal import clean
import nibabel as nib

from impact_pipeline.dataset_catalog import parse_state_task

log = logging.getLogger(__name__)

ATLAS_GLOBS = None
ATLAS_DIR_ENV = "IMPACT_ATLAS_DIR"
# The SPM12 AAL image shipped in atlases/aal_SPM12 has 116 labels (AAL-116).
# 'aal90' was the historical (incorrect) key and is accepted as an alias.
ATLAS_KEY_ALIASES = {"aal90": "aal116"}
# Plausible fMRI repetition times (seconds). Values outside this range are
# treated as unreadable metadata, never silently replaced.
MAX_PLAUSIBLE_TR_SEC = 10.0


def normalize_atlas_key(key: str) -> str:
    raw = str(key)
    return ATLAS_KEY_ALIASES.get(raw, raw)


def _default_atlas_root() -> Path:
    env_root = os.environ.get(ATLAS_DIR_ENV, "").strip()
    if env_root:
        return Path(env_root).expanduser()
    return Path(__file__).resolve().parents[2] / "atlases"


def get_atlas_globs(atlas_root=None):
    """
    Resolve atlas resources from local files only.
    This keeps preprocessing fully offline and deterministic.
    The atlas root defaults to <repo>/atlases (override with IMPACT_ATLAS_DIR),
    independent of the current working directory.
    """
    global ATLAS_GLOBS
    use_default_root = atlas_root is None
    if ATLAS_GLOBS is not None and use_default_root:
        return ATLAS_GLOBS

    atlas_root = _default_atlas_root() if use_default_root else Path(atlas_root)
    atlas_sch = atlas_root / "schaefer_2018" / "Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz"
    atlas_aal = atlas_root / "aal_SPM12" / "aal" / "atlas" / "AAL.nii"
    atlas_shen = atlas_root / "shen_1mm_268_parcellation.nii.gz"

    missing = [str(p) for p in (atlas_sch, atlas_aal, atlas_shen) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required local atlas files. Expected:\n"
            + "\n".join(missing)
            + f"\nPopulate atlases/ (or set {ATLAS_DIR_ENV}) before running "
            "fMRI preprocessing."
        )

    globs = {
        "schaefer400": str(atlas_sch.resolve()),
        "aal116": str(atlas_aal.resolve()),
        "shen268": str(atlas_shen.resolve()),
    }
    if use_default_root:
        ATLAS_GLOBS = globs
    return globs


def find_atlas(key):
    atlas_globs = get_atlas_globs()
    try:
        return atlas_globs[normalize_atlas_key(key)]
    except KeyError:
        raise FileNotFoundError(f"No atlas entry for {key}")


def _valid_tr(value) -> float | None:
    try:
        tr = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(tr) or tr <= 0 or tr > MAX_PLAUSIBLE_TR_SEC:
        return None
    return tr


def _header_tr_seconds(img) -> float | None:
    zooms = img.header.get_zooms()
    if len(zooms) < 4:
        return None
    tr = float(zooms[3])
    try:
        units = img.header.get_xyzt_units()[1]
    except Exception:
        units = "sec"
    if units == "msec":
        tr /= 1000.0
    elif units == "usec":
        tr /= 1e6
    return tr


def _sidecar_tr_seconds(sidecar_paths) -> tuple[float | None, str | None]:
    for path in sidecar_paths:
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except Exception:
            continue
        if "RepetitionTime" in meta:
            return meta.get("RepetitionTime"), str(path)
    return None, None


def resolve_bold_tr(img, sidecar_paths=(), assume_tr=None):
    """
    Determine the repetition time of a BOLD run from measured metadata.

    Priority: BIDS sidecar RepetitionTime, then the NIfTI header. If neither
    yields a plausible TR (0 < TR <= 10 s), raise unless ``assume_tr`` is
    given explicitly (CLI --assume-tr); the assumption is logged and returned
    with source 'assumed'.

    Returns (tr_seconds, source, details).
    """
    header_raw = _header_tr_seconds(img)
    header_tr = _valid_tr(header_raw)
    sidecar_raw, sidecar_path = _sidecar_tr_seconds(sidecar_paths)
    sidecar_tr = _valid_tr(sidecar_raw)
    details = {
        "header_tr": header_raw,
        "sidecar_tr": sidecar_raw,
        "sidecar_path": sidecar_path,
    }
    if sidecar_tr is not None:
        if header_tr is not None and abs(header_tr - sidecar_tr) > 0.01 * sidecar_tr:
            log.warning(
                "NIfTI header TR=%.4gs disagrees with sidecar RepetitionTime=%.4gs "
                "(%s); using the sidecar.",
                header_tr,
                sidecar_tr,
                sidecar_path,
            )
        return sidecar_tr, "sidecar", details
    if header_tr is not None:
        return header_tr, "nifti_header", details
    if assume_tr is not None:
        tr = _valid_tr(assume_tr)
        if tr is None:
            raise ValueError(
                f"--assume-tr={assume_tr!r} is not a plausible TR "
                f"(0 < TR <= {MAX_PLAUSIBLE_TR_SEC} s)."
            )
        log.warning(
            "No plausible TR in metadata (header=%r, sidecar=%r); "
            "using explicitly assumed TR=%.4gs.",
            header_raw,
            sidecar_raw,
            tr,
        )
        return tr, "assumed", details
    raise ValueError(
        "Could not determine a plausible TR from the NIfTI header "
        f"({header_raw!r}) or BIDS sidecar RepetitionTime ({sidecar_raw!r}). "
        "Fix the metadata or pass --assume-tr <seconds> explicitly."
    )


def collect_confounds(bids_root, fmriprep_deriv, subj, run_no, task):
    func_dir = os.path.join(fmriprep_deriv, f"sub-{subj}", "func")
    pattern = os.path.join(
        func_dir,
        f"sub-{subj}_task-{task}_run-*_desc-confounds*timeseries.tsv"
    )
    candidates = glob.glob(pattern)
    if not candidates:
        warnings.warn(f"No confounds files found at {pattern}")
        return None, float("nan")

    # filter to the one whose run-index == run_no
    sel = []
    for f in candidates:
        m = re.search(r"_run-0*(\d+)_desc", os.path.basename(f))
        if m and int(m.group(1)) == run_no:
            sel.append(f)

    if len(sel) > 1:
        warnings.warn(f"Multiple confound files for sub-{subj} run-{run_no}: {sel}\n  Picking the first.")
    if not sel:
        warnings.warn(f"No confounds TSV for sub-{subj} task-{task} run-{run_no}")
        return None, float("nan")

    conf_file = sel[0]
    df = pd.read_csv(conf_file, sep="\t")

    compcor_cols = [c for c in df.columns if "compcor" in c.lower()]
    compcor = df[compcor_cols].values if compcor_cols else None
    # Missing FD is undefined (NaN), not zero motion.
    fd = (
        df["framewise_displacement"].mean()
        if "framewise_displacement" in df.columns
        else float("nan")
    )
    return compcor, fd


def _find_preproc_bold(fmriprep_deriv, subj, run_no, task):
    """
    Resolve the fMRIPrep preprocessed BOLD file for a specific subject/task/run.

    Preference order:
      1) MNI152NLin2009cAsym space desc-preproc file for exact run number
      2) any desc-preproc file for exact run number
    """
    func_dir = os.path.join(fmriprep_deriv, f"sub-{subj}", "func")
    pattern = os.path.join(
        func_dir,
        f"sub-{subj}_task-{task}_run-*_desc-preproc_bold.nii.gz",
    )
    candidates = glob.glob(pattern)
    if not candidates:
        return None

    selected = []
    for f in candidates:
        m = re.search(r"_run-0*(\d+)_", os.path.basename(f))
        if m and int(m.group(1)) == int(run_no):
            selected.append(f)

    if not selected:
        return None

    selected = sorted(
        selected,
        key=lambda p: (
            0 if "space-MNI152NLin2009cAsym" in os.path.basename(p) else 1,
            p,
        ),
    )
    return selected[0]


def _write_condition_mean_fd(out_dir):
    """mean_fd.txt = mean of the per-run FD files in this condition folder."""
    vals = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*_run-*_mean_fd.txt"))):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                vals.append(float(fh.read().strip()))
        except (OSError, ValueError):
            continue
    finite = [v for v in vals if np.isfinite(v)]
    value = float(np.mean(finite)) if finite else float("nan")
    with open(os.path.join(out_dir, "mean_fd.txt"), "w") as fp:
        fp.write("nan" if not np.isfinite(value) else f"{value:.6f}")


def preprocess_subject(bids_root, fmriprep_deriv, subj, bf, out_dir, assume_tr=None):
    """
    Preprocess one bold run (bf is a BIDSLayoutFile object).
    Outputs cleaned NIfTI, per-run and per-condition mean FD, and ROI-TS .npy
    files, and returns a record describing the run (TR source, FD, outputs).
    """
    # 1) Resolve preprocessed BOLD from fMRIPrep derivatives for this run.
    # 2) load confounds
    task   = bf.entities['task']
    run_no = int(bf.entities['run'])
    compcor, fd = collect_confounds(
        bids_root=bids_root,
        fmriprep_deriv=fmriprep_deriv,
        subj=subj,
        run_no=run_no,
        task=task
    )

    # 3) load & clean BOLD
    preproc_bold = _find_preproc_bold(
        fmriprep_deriv=fmriprep_deriv,
        subj=subj,
        run_no=run_no,
        task=task,
    )
    if preproc_bold is None:
        raise FileNotFoundError(
            f"No fMRIPrep desc-preproc BOLD found for sub-{subj} task-{task} run-{run_no} "
            f"under {os.path.join(fmriprep_deriv, f'sub-{subj}', 'func')}"
        )

    img = image.load_img(preproc_bold)
    sidecars = [re.sub(r"\.nii(\.gz)?$", ".json", str(preproc_bold))]
    raw_path = getattr(bf, "path", None)
    if raw_path:
        sidecars.append(re.sub(r"\.nii(\.gz)?$", ".json", str(raw_path)))
    tr, tr_source, tr_details = resolve_bold_tr(
        img, sidecar_paths=sidecars, assume_tr=assume_tr
    )

    data = img.get_fdata().reshape(-1, img.shape[-1]).T
    cleaned = clean(
        signals=data,
        confounds=compcor,
        t_r=tr,
        detrend=True,
        standardize=True,
        low_pass=0.1,
        high_pass=0.01,
    ).T
    cleaned_vol = cleaned.reshape(img.shape)

    # 4) save cleaned NIfTI
    os.makedirs(out_dir, exist_ok=True)
    clean_fn = os.path.join(out_dir, f"{subj}_run-{run_no}_cleaned_bold.nii.gz")
    try:
        image.new_img_like(img, cleaned_vol).to_filename(clean_fn)
    except TypeError:
        nib.Nifti1Image(cleaned_vol, img.affine).to_filename(clean_fn)

    # 5) write per-run mean FD (NaN when confounds are missing) and the
    #    condition-level mean over runs (read by the motion model).
    with open(os.path.join(out_dir, f"{subj}_run-{run_no}_mean_fd.txt"), "w") as fp:
        fp.write("nan" if not np.isfinite(fd) else f"{fd:.6f}")
    _write_condition_mean_fd(out_dir)

    # 6) extract ROI time-series for each atlas
    ts_outputs = {}
    for key, atlas_img in get_atlas_globs().items():
        masker = NiftiLabelsMasker(labels_img=atlas_img,
                                   standardize=True,
                                   t_r=tr)
        ts = masker.fit_transform(clean_fn)

        ts_fn = os.path.join(out_dir, f"{subj}_run-{run_no}_{key}_ts.npy")
        np.save(ts_fn, ts)
        ts_outputs[key] = ts_fn

    return {
        "subject": subj,
        "task": task,
        "run": int(run_no),
        "preproc_bold": str(preproc_bold),
        "tr_seconds": float(tr),
        "tr_source": tr_source,
        "tr_header": tr_details.get("header_tr"),
        "tr_sidecar": tr_details.get("sidecar_tr"),
        "mean_fd": (None if not np.isfinite(fd) else float(fd)),
        "confounds_found": compcor is not None,
        "out_dir": str(out_dir),
        "ts_outputs": ts_outputs,
    }


def run_preprocessing(
    bids_root,
    fmriprep_deriv,
    out_root,
    subjects=None,
    *,
    assume_tr=None,
    dataset_id="ds003171",
    states=None,
):
    """
    Clean fMRIPrep derivatives and extract ROI time series for every run whose
    BIDS task label maps to an explicit ``(state, condition)`` of the dataset
    grammar (see dataset_catalog.parse_state_task). Runs that do not map are
    recorded in the returned summary instead of being dropped silently.

    Output layout:
        {out_root}/{subject}/{state}/{condition}/{subject}_run-{k}_{atlas}_ts.npy
    """
    layout = BIDSLayout(bids_root, validate=False)
    # find all subjects in the dataset
    all_subj = sorted(
        {
            str(f.entities.get("subject"))
            for f in layout.get(suffix='bold', extension='nii.gz')
            if f.entities.get("subject") is not None
        }
    )
    if subjects:
        # only keep the ones specified on the CLI ('sub-' prefix optional)
        wanted = {str(s).strip().replace("sub-", "", 1) for s in subjects}
        subjects = [s for s in all_subj if s in wanted]
    else:
        subjects = all_subj
    state_filter = None if states is None else {str(s) for s in states}
    summary = {
        "written_runs": [],
        "skipped_runs": [],
        "subjects_requested": list(subjects),
    }
    for subj in subjects:
        bold_files = layout.get(
            subject=subj,
            suffix='bold',
            extension='nii.gz',
            return_type='object'
        )
        for bf in bold_files:
            task = bf.entities.get('task')
            parsed = parse_state_task(dataset_id, subj, task)
            if parsed is None:
                log.warning(
                    "Skipping sub-%s task-%s: label not in the %s task grammar.",
                    subj,
                    task,
                    dataset_id,
                )
                summary["skipped_runs"].append(
                    {
                        "subject": subj,
                        "task": task,
                        "reason": "task_not_in_dataset_grammar",
                        "file": str(getattr(bf, "path", "")),
                    }
                )
                continue
            sed, cond = parsed
            if state_filter is not None and sed not in state_filter:
                summary["skipped_runs"].append(
                    {
                        "subject": subj,
                        "task": task,
                        "reason": f"state_not_requested:{sed}",
                        "file": str(getattr(bf, "path", "")),
                    }
                )
                continue

            out_dir = os.path.join(out_root, subj, sed, cond)
            record = preprocess_subject(
                bids_root,
                fmriprep_deriv,
                subj,
                bf,
                out_dir,
                assume_tr=assume_tr,
            )
            record.update({"state": sed, "condition": cond})
            summary["written_runs"].append(record)
    summary["summary"] = {
        "subjects_requested": len(subjects),
        "written_runs": len(summary["written_runs"]),
        "skipped_runs": len(summary["skipped_runs"]),
        "tr_sources": sorted({r["tr_source"] for r in summary["written_runs"]}),
    }
    return summary
