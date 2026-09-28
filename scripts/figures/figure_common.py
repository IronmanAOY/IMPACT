"""
Shared helpers of the paper-1 figure scripts (``scripts/figures/fig*.py``).

Every figure script loads result files (CSV/JSON written by the analysis
scripts or MPC-Bench), draws one figure with matplotlib (Agg) and writes into
``--out``: ``<stem>.pdf``, ``<stem>.png``, ``<stem>_data.csv`` (the plotted
values: the table view of the figure) and ``<stem>.provenance.json``
(``figure_code_sha256`` = SHA-256 over the figure script and this module,
the git code version, and the SHA-256 of every input file). The code hash is
also embedded in the PDF (``Subject``) and PNG (``Description``) metadata.
PDF output is deterministic (no creation date).

Colours: the validated reference palette of the dataviz method (categorical
slots in fixed order, one principle per slot; sequential blue; diverging
blue-red with a gray midpoint); ink and gridlines are recessive hairlines.
Categorical slots 3-5 are below 3:1 contrast on white, so identity is always
also carried by a legend, tick labels or direct labels, and every figure
writes its data table.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

PRINCIPLES = ("RAM", "PDI", "NAS", "IIM", "SRPI")
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
         "#4a3aa7", "#e34948")
PRINCIPLE_COLORS = dict(zip(PRINCIPLES, SLOTS))
SEQ_BLUE = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b")
DIVERGING = ("#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f3b4b3", "#e34948",
             "#8f1f1f")
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb",
       "neutral": "#b9b8b2", "band": "#f0efec"}
VERDICT_COLORS = {"EXCLUDED": SLOTS[0], "MPC_CONSISTENT": SLOTS[1],
                  "UNDETERMINED": INK["neutral"]}


def styled(fn):
    """Run a figure function inside ``plt.rc_context()`` so the figure style
    (``setup_style``) never leaks into the calling process."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with plt.rc_context():
            return fn(*args, **kwargs)

    return wrapper


def sequential_cmap():
    return LinearSegmentedColormap.from_list("impact_seq_blue", SEQ_BLUE)


def diverging_cmap():
    return LinearSegmentedColormap.from_list("impact_div_blue_red", DIVERGING)


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.edgecolor": INK["axis"],
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK["secondary"],
        "axes.titlecolor": INK["primary"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": INK["grid"],
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": INK["muted"],
        "ytick.color": INK["muted"],
        "xtick.labelcolor": INK["secondary"],
        "ytick.labelcolor": INK["secondary"],
        "legend.frameon": False,
        "legend.fontsize": 7,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.markersize": 5,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "svg.hashsalt": "impact",
    })


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def figure_code_hash(script_path) -> str:
    """SHA-256 over the figure script and this shared module."""
    h = hashlib.sha256()
    for p in (Path(script_path), Path(__file__)):
        h.update(p.name.encode())
        h.update(b"\0")
        h.update(p.read_bytes())
    return h.hexdigest()


def provenance(script_path, inputs) -> dict:
    try:
        from impact_pipeline.provenance import collect_code_version

        code = collect_code_version(REPO_ROOT)
    except Exception as exc:  # noqa: BLE001 - provenance must not fail a figure
        code = {"code_version": "unknown", "error": str(exc)}
    files = []
    for p in inputs:
        if p is None:
            continue
        p = Path(p)
        files.append({"path": str(p.resolve()),
                      "sha256": sha256_file(p) if p.is_file() else None})
    return {
        "figure_script": os.path.relpath(Path(script_path).resolve(), REPO_ROOT),
        "figure_code_sha256": figure_code_hash(script_path),
        "code_version": code.get("code_version"),
        "git_sha": code.get("git_sha"),
        "git_dirty": code.get("git_dirty"),
        "inputs": files,
        "matplotlib": matplotlib.__version__,
    }


def save_figure(fig, out_dir, stem, prov, data: pd.DataFrame | None = None) -> dict:
    """Write PDF, PNG, the data table and the provenance record."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    code = prov["figure_code_sha256"]
    pdf = out / f"{stem}.pdf"
    png = out / f"{stem}.png"
    fig.savefig(pdf, metadata={"Subject": f"figure_code_sha256={code}",
                               "Creator": prov["figure_script"],
                               "CreationDate": None})
    fig.savefig(png, dpi=200, metadata={"Description": f"figure_code_sha256={code}",
                                        "Software": prov["figure_script"]})
    plt.close(fig)
    paths = {"pdf": str(pdf), "png": str(png)}
    if data is not None:
        dpath = out / f"{stem}_data.csv"
        data.to_csv(dpath, index=False)
        paths["data"] = str(dpath)
    ppath = out / f"{stem}.provenance.json"
    with open(ppath, "w", encoding="utf-8") as fh:
        json.dump({**prov, "outputs": paths}, fh, indent=2, default=str)
    paths["provenance"] = str(ppath)
    return paths


def read_table(path, required=(), name=None) -> pd.DataFrame:
    """CSV with a clear error for missing columns."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{name or 'input'} not found: {p}")
    df = pd.read_csv(p)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{p.name} lacks required columns {missing}")
    return df


def panel_title(ax, letter, text, fontsize=8):
    """Left-aligned panel title with its bold panel letter."""
    ax.set_title(f"$\\bf{{{letter}}}$  {text}", loc="left", fontsize=fontsize,
                 color=INK["primary"])


def normalize_verdicts(series):
    from impact_pipeline.necessity import normalize_verdict

    return series.map(normalize_verdict)
