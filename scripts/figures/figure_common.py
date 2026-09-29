"""
Shared helpers of the paper-1 figure scripts (``scripts/figures/fig*.py``).

Every figure script loads result files (CSV/JSON written by the analysis
scripts or MPC-Bench), draws one figure with matplotlib (Agg) and writes into
``--out``: ``<stem>.pdf`` (vector, fonts embedded as TrueType),
``<stem>.png`` (300 dpi), ``<stem>_data.csv`` (the plotted values: the table
view of the figure) and ``<stem>.provenance.json`` (``figure_code_sha256`` =
SHA-256 over the figure script and this module, the git code version, and
the SHA-256 of every input file). The code hash is also embedded in the PDF
(``Subject``) and PNG (``Description``) metadata. PDF output is deterministic
(no creation date).

Print conventions: full-width figures are 7.0 in (178 mm) wide; text is 6-8
pt in Arial (DejaVu Sans where Arial is not installed), panel letters bold.

Colours (the validated reference palette of the dataviz method):

- categorical series take the slots in fixed order (``SLOTS``; the first three
  pass the colour-vision-deficiency checks for every pair), and identity is
  always also carried by a legend, tick labels or direct labels;
- the three-valued outcomes use one diverging scheme throughout the paper:
  PRESENT / MPC_CONSISTENT dark blue, ABSENT / EXCLUDED red, UNDEFINED /
  UNDETERMINED neutral gray (a lighter gray marks components without a
  construct scale); the pair is separated in lightness as well as hue, so it
  survives grayscale print;
- sequential magnitudes use the single blue ramp, signed effects the
  blue-red diverging ramp with a gray midpoint;
- ink and gridlines are recessive hairlines.
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
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

PRINCIPLES = ("RAM", "PDI", "NAS", "IIM", "SRPI")
# the principles with an anchor (a construct scale) on the bench
ANCHORED = ("NAS", "IIM", "SRPI")
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
# Three-valued outcomes (component status and verdict) on one diverging scheme.
POSITIVE = "#184f95"   # PRESENT, MPC_CONSISTENT
NEGATIVE = "#e34948"   # ABSENT, EXCLUDED
NEUTRAL = "#c3c2b7"    # UNDEFINED (inconclusive), UNDETERMINED
NO_SCALE = "#ecebe6"   # UNDEFINED without a construct scale / not defined
STATUS_COLORS = {"PRESENT": POSITIVE, "ABSENT": NEGATIVE, "UNDEFINED": NEUTRAL,
                 "NO_SCALE": NO_SCALE}
VERDICT_COLORS = {"EXCLUDED": NEGATIVE, "MPC_CONSISTENT": POSITIVE,
                  "UNDETERMINED": NEUTRAL}
FULL_WIDTH = 7.0   # inches (178 mm)
HALF_WIDTH = 3.4   # inches (86 mm)
PNG_DPI = 300


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


def _font_family():
    names = {f.name for f in font_manager.fontManager.ttflist}
    return "Arial" if "Arial" in names else "DejaVu Sans"


def setup_style():
    family = _font_family()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [family, "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7,
        "axes.edgecolor": INK["axis"],
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK["secondary"],
        "axes.titlecolor": INK["primary"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": INK["grid"],
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "xtick.color": INK["axis"],
        "ytick.color": INK["axis"],
        "xtick.labelcolor": INK["secondary"],
        "ytick.labelcolor": INK["secondary"],
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "legend.fontsize": 6.5,
        "legend.handlelength": 1.4,
        "legend.borderaxespad": 0.3,
        "lines.linewidth": 1.5,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.markersize": 4,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.hashsalt": "impact",
        "mathtext.fontset": "custom" if family == "Arial" else "dejavusans",
        "mathtext.rm": family,
        "mathtext.it": f"{family}:italic",
        "mathtext.bf": f"{family}:bold",
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


def _display_path(p: Path) -> str:
    """Repository-relative path when possible (no local absolute paths in the
    provenance of published figures)."""
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p.resolve())


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
        files.append({"path": _display_path(p),
                      "sha256": sha256_file(p) if p.is_file() else None})
    return {
        "figure_script": os.path.relpath(Path(script_path).resolve(), REPO_ROOT),
        "figure_code_sha256": figure_code_hash(script_path),
        "code_version": code.get("code_version"),
        "git_sha": code.get("git_sha"),
        "git_dirty": code.get("git_dirty"),
        "inputs": files,
        "matplotlib": matplotlib.__version__,
        "font": _font_family(),
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
    fig.savefig(png, dpi=PNG_DPI,
                metadata={"Description": f"figure_code_sha256={code}",
                          "Software": prov["figure_script"]})
    plt.close(fig)
    paths = {"pdf": str(pdf), "png": str(png)}
    if data is not None:
        dpath = out / f"{stem}_data.csv"
        data.to_csv(dpath, index=False)
        paths["data"] = str(dpath)
    ppath = out / f"{stem}.provenance.json"
    with open(ppath, "w", encoding="utf-8") as fh:
        json.dump({**prov, "outputs": {k: Path(v).name for k, v in paths.items()}},
                  fh, indent=2, default=str)
    paths["provenance"] = str(ppath)
    return paths


def read_table(path, required=(), name=None) -> pd.DataFrame:
    """CSV with a clear error for missing columns."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{name or 'input'} not found: {p}")
    df = pd.read_csv(p, low_memory=False)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{p.name} lacks required columns {missing}")
    return df


def panel_title(ax, letter, text, fontsize=7.5):
    """Left-aligned panel title with its bold panel letter."""
    ax.set_title(f"$\\bf{{{letter}}}$   {text}", loc="left", fontsize=fontsize,
                 color=INK["primary"], pad=4)


def cutoff_lines(ax, z=0.25, delta=0.10, orientation="h", label=True):
    """The construct-scale cutoffs of the frozen protocols: PRESENT needs the
    lower bound of ``c`` above ``z``, ABSENT the upper bound below ``delta``."""
    line = ax.axhline if orientation == "h" else ax.axvline
    line(z, color=POSITIVE, lw=0.6, alpha=0.6, zorder=1)
    line(delta, color=NEGATIVE, lw=0.6, alpha=0.6, zorder=1)
    line(0.0, color=INK["axis"], lw=0.6, zorder=1)
    if label and orientation == "h":
        x = ax.get_xlim()[1]
        ax.text(x, z, f" z = {z:g}", color=POSITIVE, fontsize=5.5, va="bottom",
                ha="right")
        ax.text(x, delta, f" δ = {delta:g}", color=NEGATIVE, fontsize=5.5,
                va="top", ha="right")


def clopper_pearson_interval(k, n, alpha=0.05):
    """Two-sided exact (1 - alpha) interval of k / n (``necessity``)."""
    from impact_pipeline.necessity import clopper_pearson

    lo, hi = clopper_pearson(k, n, alpha, "two-sided")
    return float(lo), float(hi)


def normalize_verdicts(series):
    from impact_pipeline.necessity import normalize_verdict

    return series.map(normalize_verdict)
