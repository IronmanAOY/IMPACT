from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REAL_DATA_ORIGIN = "real"
DUMMY_DATA_ORIGIN = "dummy"
VALID_DATA_ORIGINS = (REAL_DATA_ORIGIN, DUMMY_DATA_ORIGIN)
PROVENANCE_COLUMNS = (
    "dataset_id",
    "data_origin",
    "dataset_role",
    "provenance_label",
)
TEST_OBJECTS_ROOTNAME = "test_objects"


def normalize_data_origin(value) -> str:
    raw = str(value or REAL_DATA_ORIGIN).strip().lower()
    aliases = {
        "real": REAL_DATA_ORIGIN,
        "study": REAL_DATA_ORIGIN,
        "study_data": REAL_DATA_ORIGIN,
        "production": REAL_DATA_ORIGIN,
        "dummy": DUMMY_DATA_ORIGIN,
        "test": DUMMY_DATA_ORIGIN,
        "test_object": DUMMY_DATA_ORIGIN,
        "test_objects": DUMMY_DATA_ORIGIN,
        "synthetic": DUMMY_DATA_ORIGIN,
        "fake": DUMMY_DATA_ORIGIN,
    }
    origin = aliases.get(raw, raw)
    if origin not in VALID_DATA_ORIGINS:
        raise ValueError(
            f"Unknown data origin '{value}'. Expected one of {VALID_DATA_ORIGINS}."
        )
    return origin


def dataset_role_for_origin(data_origin: str) -> str:
    origin = normalize_data_origin(data_origin)
    if origin == DUMMY_DATA_ORIGIN:
        return "synthetic_validation"
    return "study_data"


def provenance_label_for_origin(data_origin: str) -> str:
    origin = normalize_data_origin(data_origin)
    if origin == DUMMY_DATA_ORIGIN:
        return "synthetic_validation"
    return "real_study_data"


def is_test_object_origin(data_origin: str) -> bool:
    return normalize_data_origin(data_origin) == DUMMY_DATA_ORIGIN


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except Exception:
        return False


def ensure_test_objects_scaffold(repo_root: Path | str) -> dict[str, Path]:
    root = Path(repo_root).resolve() / TEST_OBJECTS_ROOTNAME
    paths = {
        "root": root,
        "datasets": root / "datasets",
        "runs": root / "runs",
        "metric_bank": root / "metric_bank",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


@dataclass(frozen=True)
class DatasetProvenance:
    repo_root: Path
    requested_out_dir: Path
    effective_out_dir: Path
    dataset_id: str
    data_origin: str
    dataset_role: str
    provenance_label: str
    test_objects_root: Path
    metric_bank_dataset_dir: Path | None

    def as_result_metadata(self) -> dict[str, str]:
        return {
            "dataset_id": str(self.dataset_id),
            "data_origin": str(self.data_origin),
            "dataset_role": str(self.dataset_role),
            "provenance_label": str(self.provenance_label),
        }

    def as_manifest_dict(self) -> dict[str, object]:
        out = {
            "dataset_id": str(self.dataset_id),
            "data_origin": str(self.data_origin),
            "dataset_role": str(self.dataset_role),
            "provenance_label": str(self.provenance_label),
            "requested_out_dir": str(self.requested_out_dir),
            "effective_out_dir": str(self.effective_out_dir),
            "test_objects_root": str(self.test_objects_root),
            "created_unix": float(time.time()),
        }
        if self.metric_bank_dataset_dir is not None:
            out["metric_bank_dataset_dir"] = str(self.metric_bank_dataset_dir)
        return out


def resolve_dataset_provenance(
    *,
    repo_root: Path | str,
    out_dir: Path | str,
    dataset_id: str,
    data_origin=REAL_DATA_ORIGIN,
    synthetic_root: Path | str | None = None,
) -> DatasetProvenance:
    root = Path(repo_root).resolve()
    requested = Path(out_dir).expanduser().resolve()
    dataset_id_txt = str(dataset_id).strip()
    origin = normalize_data_origin(data_origin)
    role = dataset_role_for_origin(origin)
    label = provenance_label_for_origin(origin)
    synthetic_base = Path(
        synthetic_root or os.environ.get("IMPACT_SYNTH_ROOT") or root
    ).expanduser().resolve()
    if origin == DUMMY_DATA_ORIGIN:
        scaffold = ensure_test_objects_scaffold(synthetic_base)
    else:
        # Real-data runs must not create synthetic scaffolding as a side effect.
        test_root = synthetic_base / TEST_OBJECTS_ROOTNAME
        scaffold = {
            "root": test_root,
            "runs": test_root / "runs",
            "metric_bank": test_root / "metric_bank",
        }

    if origin == DUMMY_DATA_ORIGIN:
        base = requested
        if not _is_relative_to(base, scaffold["root"]):
            base = scaffold["runs"]
        effective = base
        if effective.name != dataset_id_txt:
            effective = effective / dataset_id_txt
        metric_bank_dataset_dir = scaffold["metric_bank"] / dataset_id_txt
    else:
        effective = requested
        if dataset_id_txt != "ds003171" and effective.name != dataset_id_txt:
            effective = effective / dataset_id_txt
        metric_bank_dataset_dir = None

    return DatasetProvenance(
        repo_root=root,
        requested_out_dir=requested,
        effective_out_dir=effective.resolve(),
        dataset_id=dataset_id_txt,
        data_origin=origin,
        dataset_role=role,
        provenance_label=label,
        test_objects_root=scaffold["root"],
        metric_bank_dataset_dir=metric_bank_dataset_dir,
    )


def write_json(path: Path | str, payload) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


CODE_VERSION_ENV = "IMPACT_CODE_VERSION"
RUNTIME_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "nilearn",
    "nibabel",
    "mne",
    "pybids",
    "statsmodels",
    "numba",
    "networkx",
    "cupy",
    "cupy-rocm-7-0",
    "amd-cupy",
)


def _git_output(repo_root: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            # Many batch tasks may query the same checkout concurrently: never
            # take index.lock for opportunistic index refreshes.
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


REPO_ROOT_ENV = "IMPACT_REPO_ROOT"
# A checkout root is recognised by the pipeline entry point the job scripts run.
REPO_ROOT_MARKER = "run_pipeline.py"


def package_version() -> str:
    """Version of the impact_pipeline package (``impact_pipeline.__version__``)."""
    from impact_pipeline import __version__

    return str(__version__)


def _is_repo_root(path: Path | str | None) -> bool:
    return path is not None and (Path(path) / REPO_ROOT_MARKER).is_file()


def resolve_repo_root(
    explicit: Path | str | None = None,
    env=None,
    fallback: Path | str | None = None,
    required: bool = True,
) -> Path | None:
    """
    Root of the checkout that holds ``run_pipeline.py`` (Hunter job scripts cd
    there and run it). Order: ``explicit`` (e.g. --repo-root) >
    IMPACT_REPO_ROOT > the package-relative root (source checkout or editable
    install) > ``fallback`` (e.g. run_pipeline.py's own directory) > the
    current directory. An explicit or environment value that is not a
    checkout is an error (no silent substitution). With a non-editable install
    the package-relative root lies in site-packages and is skipped.
    """
    env = os.environ if env is None else env
    for label, value in (
        ("--repo-root", explicit),
        (REPO_ROOT_ENV, str(env.get(REPO_ROOT_ENV) or "").strip() or None),
    ):
        if value is None:
            continue
        path = Path(value).expanduser().resolve()
        if not _is_repo_root(path):
            raise FileNotFoundError(
                f"{label}={value!r} does not contain {REPO_ROOT_MARKER}; point it at "
                "the impact-synergy-pipeline checkout."
            )
        return path
    for candidate in (Path(__file__).resolve().parents[2], fallback, Path.cwd()):
        if candidate is not None and _is_repo_root(candidate):
            return Path(candidate).resolve()
    if required:
        raise FileNotFoundError(
            f"Cannot locate the pipeline checkout ({REPO_ROOT_MARKER}): the package "
            "is not in a source checkout (non-editable install?). Pass --repo-root "
            f"or set {REPO_ROOT_ENV}."
        )
    return None


def format_code_version(info: dict) -> str:
    """
    Code version string: package version + git commit, e.g.
    '1.1.0+g<40-hex sha>[.dirty]'; without git metadata '1.1.0+<IMPACT_CODE_VERSION>',
    else '1.1.0+unknown'.
    """
    pkg = str(info.get("package_version") or "0+unknown")
    base = pkg.split("+", 1)[0]
    if info.get("git_sha"):
        text = f"{base}+g{info['git_sha']}"
        if info.get("git_dirty"):
            text += ".dirty"
        return text
    if info.get("declared_version"):
        return f"{base}+{info['declared_version']}"
    return f"{base}+unknown"


def collect_code_version(repo_root: Path | str) -> dict[str, object]:
    """
    Identify the code that produced a result: package version, git commit (if
    the checkout is a git work tree), dirty flag and branch, combined in
    ``code_version`` (see ``format_code_version``). Deployments without git
    metadata (e.g. a tarball copied to an HPC workspace) can set
    IMPACT_CODE_VERSION.
    """
    root = Path(repo_root).resolve()
    out: dict[str, object] = {
        "package_version": package_version(),
        "git_sha": None,
        "git_dirty": None,
        "git_branch": None,
        "declared_version": (os.environ.get(CODE_VERSION_ENV) or None),
        "source": "unavailable",
    }
    toplevel = _git_output(root, "rev-parse", "--show-toplevel")
    # Only the checkout itself counts, not an enclosing repository.
    in_repo = toplevel is not None and Path(toplevel).resolve() == root
    sha = _git_output(root, "rev-parse", "HEAD") if in_repo else None
    if sha:
        out["git_sha"] = sha
        out["source"] = "git"
        status = _git_output(root, "status", "--porcelain", "--untracked-files=no")
        out["git_dirty"] = None if status is None else bool(status)
        out["git_branch"] = _git_output(root, "rev-parse", "--abbrev-ref", "HEAD")
    elif out["declared_version"]:
        out["source"] = CODE_VERSION_ENV
    out["code_version"] = format_code_version(out)
    return out


def collect_runtime_versions(packages=RUNTIME_PACKAGES) -> dict[str, object]:
    """Python/platform and installed package versions, without importing them."""
    try:
        from importlib import metadata as importlib_metadata
    except ImportError:  # pragma: no cover
        importlib_metadata = None
    versions: dict[str, str | None] = {}
    for name in packages:
        ver = None
        if importlib_metadata is not None:
            try:
                ver = importlib_metadata.version(name)
            except Exception:
                ver = None
        if ver is not None:
            versions[name] = ver
    return {
        "impact_pipeline_version": package_version(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "packages": versions,
    }


def read_dataset_description(bids_root: Path | str | None) -> dict:
    if bids_root is None:
        return {}
    path = Path(bids_root) / "dataset_description.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def dataset_declares_synthetic(bids_root: Path | str | None) -> bool:
    """True when dataset_description.json marks the dataset as synthetic."""
    desc = read_dataset_description(bids_root)
    synthetic_flag = desc.get("SyntheticData")
    if isinstance(synthetic_flag, str):
        synthetic_flag = synthetic_flag.strip().lower() in {"true", "1", "yes"}
    dataset_type = str(desc.get("DatasetType") or "").strip().lower()
    return bool(synthetic_flag) or dataset_type == "synthetic"


def assert_origin_matches_dataset(bids_root: Path | str | None, data_origin) -> None:
    """
    Refuse to label a dataset that declares itself synthetic as real study data.
    """
    origin = normalize_data_origin(data_origin)
    if origin == REAL_DATA_ORIGIN and dataset_declares_synthetic(bids_root):
        raise ValueError(
            f"Dataset at '{bids_root}' declares SyntheticData/DatasetType=synthetic in "
            "dataset_description.json, but data_origin='real'. Re-run with "
            "--data-origin dummy so synthetic results stay separated from study data."
        )
