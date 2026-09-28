"""
scripts/run_all.sh decides which subjects still need fMRIPrep with the same
rule as run_pipeline._fmriprep_subject_complete: the report sub-<label>.html
AND at least one sub-<label>/**/func/*desc-preproc_bold.nii.gz. It used to
check the report only, so a run that stopped after writing the report was
skipped by run_all.sh but not by run_pipeline.
"""

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

import run_pipeline

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
needs_bash = pytest.mark.skipif(BASH is None, reason="bash not available")

BOLD = "task-audioawake_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
# label -> files under <derivatives> (relative); whether it is complete.
LAYOUTS = {
    "sub-01": (["sub-01.html", f"sub-01/func/sub-01_{BOLD}"], True),
    "sub-02": (["sub-02.html"], False),  # stopped after the report
    "sub-03": ([], False),  # never processed
    "sub-04": (["sub-04.html", "sub-04/anat/sub-04_desc-preproc_T1w.nii.gz"], False),
    "sub-05": (["sub-05.html", f"sub-05/ses-1/func/sub-05_ses-1_{BOLD}"], True),
    "sub-06": ([f"sub-06/func/sub-06_{BOLD}"], False),  # no report
    "sub-07": (["sub-07.html", f"sub-07/func/extra/sub-07_{BOLD}"], False),
}


def _tree(tmp_path, data_root=None):
    data_root = data_root or (tmp_path / "scratch")
    bids = data_root / "ds003171"
    deriv = bids / "derivatives" / "fmriprep"
    for label, (files, _complete) in LAYOUTS.items():
        (bids / label).mkdir(parents=True)
        for rel in files:
            path = deriv / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
    return data_root, deriv


def test_python_rule_known_answers(tmp_path):
    _data_root, deriv = _tree(tmp_path)
    for label, (_files, complete) in LAYOUTS.items():
        sub = label[len("sub-"):]
        assert run_pipeline._fmriprep_subject_complete(deriv, sub) is complete, label


@needs_bash
@pytest.mark.parametrize(
    "data_dir",
    [
        "scratch",
        # A 'func' folder above the derivatives must not change the verdict
        # (the Python rule globs relative to the subject folder).
        "func/scratch",
    ],
)
def test_run_all_requests_fmriprep_for_exactly_the_incomplete_subjects(
    tmp_path, data_dir
):
    data_root, deriv = _tree(tmp_path, tmp_path / data_dir)
    # An existing clone at the pinned snapshot (git stubbed; nothing downloads).
    bids = data_root / "ds003171"
    (bids / ".git").mkdir()
    (bids / "dataset_description.json").write_text('{"Name": "x"}')
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    git = bin_dir / "git"
    git.write_text(
        '#!/bin/bash\ncase "$*" in *"tag --points-at HEAD"*) echo 2.0.1 ;; esac\n'
        "exit 0\n"
    )
    git.chmod(0o755)
    env = dict(os.environ)
    env.update(
        {
            "PATH": os.pathsep.join([str(bin_dir), env.get("PATH", "")]),
            "IMPACT_DATA_ROOT": str(data_root),
            "IMPACT_OUT_DIR": str(tmp_path / "out"),
            "IMPACT_PYTHON": "python",
        }
    )
    proc = subprocess.run(
        [BASH, str(ROOT / "scripts" / "run_all.sh"), "--dry-run", "ds003171"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    fetch = shlex.split(
        next(ln for ln in proc.stdout.splitlines() if "fetch_fmriprep.sh" in ln)
    )
    start = fetch.index("--participant-label") + 1
    requested = set()
    for tok in fetch[start:]:
        if tok.startswith("-"):
            break
        requested.add(tok)
    expected = {
        label[len("sub-"):]
        for label in LAYOUTS
        if not run_pipeline._fmriprep_subject_complete(deriv, label[len("sub-"):])
    }
    assert requested == expected == {"02", "03", "04", "06", "07"}
