"""
MPC-Bench external generator: Stuart-Landau (Hopf normal form) whole-brain
model on the empirical structural connectome shipped with the repository.

Unlike families A-C this generator is **not** built around the five
mechanism switches. Its manipulations are external and anatomical: the global
coupling ``G`` (sweep), a hub lesion, an interhemispheric (callosal) or
homotopic lesion, a long-range lesion and a size-matched random-edge lesion
(control). There is no stipulated five-bit ground truth; the oracle records
dynamical order parameters (mean Kuramoto order, metastability) instead.

Connectome (provenance)
-----------------------
``data/managed/structural/budapest_connectome_3.0_209_0_median.csv``: a
consensus connectome exported from the Budapest Reference Connectome Server
v3.0 (Szalkai, Kerepesi, Varga and Grolmusz; human diffusion-MRI
tractography). Observed facts about the shipped file (checked by
:func:`load_connectome`; recorded in every system's ``meta['connectome']``):

* a ``;``-separated **edge list** with 1000 rows and the columns ``id
  node1``, ``id node2``, ``name node1``, ``name node2``, ``parent id
  node1/2``, ``parent name node1/2``, ``edge confidence`` and ``edge
  weight(med nof)`` (median number of fibres) - not a 209 x 209 matrix;
* 480 distinct fine-grained vertices (Lausanne-style sub-parcels such as
  ``rh.precuneus_20``) belonging to 76 parent regions (Desikan-Killiany
  cortical regions, subcortical nuclei, brain stem);
* every ``edge confidence`` is >= 209, consistent with ``209`` in the file
  name being the minimum-confidence parameter of the export (and ``median``
  the weight statistic); 11 rows are self-loops;
* only 9 region-level edges cross the hemispheres (2 of them homotopic:
  caudate-caudate and thalamus-thalamus). The 1000-row list is sparse; treat
  interhemispheric structure as unreliable.

Grain: ``'region'`` (default) aggregates the fine edges to the 76 parent
regions (edges within a parent region are dropped), ``'fine'`` keeps the
largest connected component of the fine graph (467 of 480 vertices). Region
weights are ``log(1 + sum of median fibre counts)``, symmetric, scaled so that
the mean node strength is 1.

Dynamics
--------
Node ``j`` (complex ``z_j``) follows the Hopf normal form with diffusive
coupling (as in Deco-type whole-brain models; deco2021revisiting,
luppi2022whole)::

    dz_j = { i w_j z_j + lam [ (a - |z_j|^2) z_j
             + G sum_k W_jk (z_k - z_j) ] } dt + lam sigma dB_j

with ``w_j = 2 pi f_j``, ``f_j ~ N(f_mean, f_sd)`` (alpha-band carrier),
bifurcation parameter ``a`` (< 0: noise-driven damped oscillations near the
bifurcation), amplitude rate ``lam`` and complex white noise. The rotation is
integrated exactly and the rest by Euler-Maruyama. The recorded source signal
is ``Re z`` sampled at ``fs_out``. EEG-like and BOLD-like observations are
produced by :mod:`impact_pipeline.bench.forward`.

This module must not import ``impact_pipeline.mpc_metrics``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from impact_pipeline.bench.generators import PRINCIPLES, BenchSystem, _streams

WHOLE_BRAIN_VERSION = "mpc-bench-whole-brain/1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[3]
CONNECTOME_RELPATH = "data/managed/structural/budapest_connectome_3.0_209_0_median.csv"
CONNECTOME_SHA256 = "40645a340981a30e5de3c2cddd4220b712e21396be865fd584f2a7920a9dd2b9"
GRAINS = ("region", "fine")
LESIONS = (
    "none",
    "hub",
    "interhemispheric",
    "homotopic",
    "long_range",
    "random",
)

_LOBES = {
    "frontal": (
        "superiorfrontal",
        "rostralmiddlefrontal",
        "caudalmiddlefrontal",
        "parsopercularis",
        "parstriangularis",
        "parsorbitalis",
        "lateralorbitofrontal",
        "medialorbitofrontal",
        "precentral",
        "paracentral",
        "frontalpole",
    ),
    "parietal": (
        "superiorparietal",
        "inferiorparietal",
        "supramarginal",
        "postcentral",
        "precuneus",
    ),
    "temporal": (
        "superiortemporal",
        "middletemporal",
        "inferiortemporal",
        "bankssts",
        "fusiform",
        "transversetemporal",
        "entorhinal",
        "temporalpole",
        "parahippocampal",
    ),
    "occipital": ("lateraloccipital", "lingual", "cuneus", "pericalcarine"),
    "cingulate": (
        "rostralanteriorcingulate",
        "caudalanteriorcingulate",
        "posteriorcingulate",
        "isthmuscingulate",
    ),
    "insula": ("insula",),
    "subcortical": (
        "thalamus-proper",
        "thalamus",
        "caudate",
        "putamen",
        "pallidum",
        "hippocampus",
        "amygdala",
        "accumbens-area",
    ),
    "brainstem": ("brain-stem",),
}
_ANTERIOR_LOBES = ("frontal", "cingulate", "insula")
_POSTERIOR_LOBES = ("parietal", "temporal", "occipital")


def region_base_name(name: str) -> str:
    """DK / aseg base name without hemisphere prefix (lower case)."""
    n = str(name).strip()
    for pre in ("ctx-lh-", "ctx-rh-", "Left-", "Right-", "lh.", "rh."):
        if n.startswith(pre):
            n = n[len(pre) :]
            break
    n = n.split("_")[0] if "." not in str(name)[:3] else n.rsplit("_", 1)[0]
    return n.lower()


def hemisphere(name: str) -> str:
    n = str(name)
    if n.startswith(("ctx-lh-", "Left-", "lh.")):
        return "L"
    if n.startswith(("ctx-rh-", "Right-", "rh.")):
        return "R"
    return "M"


def lobe(name: str) -> str:
    base = region_base_name(name)
    for lb, members in _LOBES.items():
        if base in members:
            return lb
    return "other"


@dataclass
class Connectome:
    """Symmetric, normalised structural coupling and its declared anatomy."""

    W: np.ndarray
    labels: List[str]
    hemisphere: List[str]
    lobe: List[str]
    raw_weight: np.ndarray
    provenance: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.W.shape[0])

    def strength(self) -> np.ndarray:
        return self.W.sum(axis=1)

    def hubs(self, k: int = 6) -> List[int]:
        """The ``k`` regions of largest strength (ties by index)."""
        s = self.strength()
        order = np.lexsort((np.arange(self.n), -s))
        return sorted(int(i) for i in order[: int(k)])

    def macro_nodes(self) -> Dict[str, List[int]]:
        """Four cortical macro nodes: hemisphere x (anterior, posterior)."""
        out = {}
        for h in ("L", "R"):
            for part, lobes in (("ant", _ANTERIOR_LOBES), ("post", _POSTERIOR_LOBES)):
                idx = [
                    i
                    for i in range(self.n)
                    if self.hemisphere[i] == h and self.lobe[i] in lobes
                ]
                if idx:
                    out[f"{h}_{part}"] = idx
        return out

    def modules(self) -> Dict[str, List[int]]:
        """Hemisphere x lobe groups (declared anatomy)."""
        out: Dict[str, List[int]] = {}
        for i in range(self.n):
            out.setdefault(f"{self.hemisphere[i]}_{self.lobe[i]}", []).append(i)
        return out


def _resolve_path(path) -> Path:
    p = Path(path) if path is not None else REPO_ROOT / CONNECTOME_RELPATH
    if not p.exists():
        raise FileNotFoundError(
            f"connectome file not found: {p} (the whole-brain generator reads the "
            f"repository file {CONNECTOME_RELPATH}; pass path= for another copy)"
        )
    return p


def load_connectome(path=None, grain: str = "region", verify_hash: bool = True):
    """
    Load the shipped connectome (see the module docstring for provenance).
    Returns a :class:`Connectome` with ``W`` symmetric, zero diagonal,
    weights ``log1p(summed median fibre count)`` scaled to mean strength 1.
    ``verify_hash`` checks the file's SHA-256 against ``CONNECTOME_SHA256``
    (only for the default file).
    """
    if grain not in GRAINS:
        raise ValueError(f"grain must be one of {GRAINS}")
    p = _resolve_path(path)
    raw = p.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if verify_hash and path is None and sha != CONNECTOME_SHA256:
        raise ValueError(
            f"connectome file hash {sha} differs from the declared {CONNECTOME_SHA256}"
        )
    df = pd.read_csv(p, sep=";")
    need = {
        "id node1",
        "id node2",
        "name node1",
        "name node2",
        "parent id node1",
        "parent id node2",
        "parent name node1",
        "parent name node2",
        "edge confidence",
        "edge weight(med nof)",
    }
    missing = sorted(need - set(df.columns))
    if missing:
        raise ValueError(f"connectome file lacks columns {missing}")
    n_rows = int(len(df))
    self_loops = int((df["id node1"] == df["id node2"]).sum())
    df = df[df["id node1"] != df["id node2"]]
    if grain == "region":
        a, b = "parent id node1", "parent id node2"
        na, nb = "parent name node1", "parent name node2"
        dropped_within = int((df[a] == df[b]).sum())
        df = df[df[a] != df[b]]
    else:
        a, b, na, nb = "id node1", "id node2", "name node1", "name node2"
        dropped_within = 0
    names = {}
    for col_id, col_name in ((a, na), (b, nb)):
        for i, nm in zip(df[col_id], df[col_name]):
            names[int(i)] = str(nm)
    ids = np.array(sorted(names))
    pos = {v: k for k, v in enumerate(ids)}
    n = ids.size
    raw_w = np.zeros((n, n))
    for x, y, w in zip(df[a], df[b], df["edge weight(med nof)"]):
        i, j = pos[int(x)], pos[int(y)]
        raw_w[i, j] += float(w)
        raw_w[j, i] += float(w)
    keep = np.arange(n)
    if grain == "fine":
        from scipy.sparse.csgraph import connected_components

        _, lab = connected_components(raw_w > 0, directed=False)
        big = np.argmax(np.bincount(lab))
        keep = np.flatnonzero(lab == big)
        raw_w = raw_w[np.ix_(keep, keep)]
    W = np.log1p(raw_w)
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    W = W / float(W.sum(axis=1).mean())
    labels = [names[int(ids[k])] for k in keep]
    prov = {
        "file": CONNECTOME_RELPATH if path is None else str(path),
        "sha256": sha,
        "source": "Budapest Reference Connectome Server v3.0 export "
        "(consensus human diffusion-MRI connectome)",
        "file_format": "edge list (';'), 1 row per edge",
        "n_rows": n_rows,
        "n_self_loops_dropped": self_loops,
        "n_within_parent_edges_dropped": dropped_within,
        "edge_confidence_min": float(pd.read_csv(p, sep=";")["edge confidence"].min()),
        "grain": grain,
        "n_nodes": int(len(labels)),
        "n_edges": int(np.count_nonzero(np.triu(W, 1))),
        "weight_transform": "log1p(sum of median fibre counts)",
        "normalisation": "symmetric; mean node strength = 1",
        "filename_parameters": "209 = minimum edge confidence (every row >= 209); "
        "median = weight statistic",
    }
    return Connectome(
        W=W,
        labels=labels,
        hemisphere=[hemisphere(x) for x in labels],
        lobe=[lobe(x) for x in labels],
        raw_weight=raw_w,
        provenance=prov,
    )


# ---------------------------------------------------------------------------
# Lesions (external, anatomical manipulations)
# ---------------------------------------------------------------------------


def lesion_mask(
    conn: Connectome,
    kind: str,
    n_hubs: int = 4,
    n_edges: Optional[int] = None,
    seed: int = 0,
) -> np.ndarray:
    """
    Boolean (n x n, symmetric) mask of the edges removed by a lesion:

    * ``'hub'``: every edge of the ``n_hubs`` strongest regions;
    * ``'interhemispheric'``: every edge between the left and right
      hemispheres (callosal/commissural lesion);
    * ``'homotopic'``: edges between homologous regions of the two
      hemispheres (same base name);
    * ``'long_range'``: edges between different lobes or hemispheres
      (topological proxy for long fibres: the file carries no coordinates);
    * ``'random'``: ``n_edges`` existing edges drawn uniformly (size-matched
      control; seed ``seed``);
    * ``'none'``: nothing.
    """
    if kind not in LESIONS:
        raise ValueError(f"lesion must be one of {LESIONS}")
    n = conn.n
    edge = conn.W > 0
    m = np.zeros((n, n), dtype=bool)
    hemi = np.asarray(conn.hemisphere)
    lob = np.asarray(conn.lobe)
    if kind == "hub":
        hubs = conn.hubs(n_hubs)
        m[hubs, :] = True
        m[:, hubs] = True
    elif kind == "interhemispheric":
        lr = np.isin(hemi, ("L", "R"))
        m = (hemi[:, None] != hemi[None, :]) & lr[:, None] & lr[None, :]
    elif kind == "homotopic":
        base = np.asarray([region_base_name(x) for x in conn.labels])
        m = (
            (base[:, None] == base[None, :])
            & (hemi[:, None] != hemi[None, :])
            & np.isin(hemi, ("L", "R"))[:, None]
        )
    elif kind == "long_range":
        m = (lob[:, None] != lob[None, :]) | (hemi[:, None] != hemi[None, :])
    elif kind == "random":
        if n_edges is None:
            raise ValueError("the random lesion needs n_edges")
        iu = np.transpose(np.nonzero(np.triu(edge, 1)))
        rng = np.random.default_rng(int(seed))
        pick = iu[rng.choice(len(iu), size=min(int(n_edges), len(iu)), replace=False)]
        m[pick[:, 0], pick[:, 1]] = True
    m = (m | m.T) & edge
    np.fill_diagonal(m, False)
    return m


def apply_lesion(conn: Connectome, mask: np.ndarray) -> np.ndarray:
    """Coupling after removing the masked edges (no renormalisation)."""
    W = conn.W.copy()
    W[np.asarray(mask, dtype=bool)] = 0.0
    return W


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WholeBrainConfig:
    """Hopf whole-brain model constants (seconds, Hz)."""

    grain: str = "region"
    G: float = 1.0
    a: float = -0.05
    rate: float = 10.0
    f_mean: float = 10.0
    f_sd: float = 0.5
    sigma: float = 0.05
    dt: float = 0.002
    fs_out: float = 250.0
    duration_sec: float = 60.0
    transient_sec: float = 2.0
    lesion: str = "none"
    n_hubs: int = 4
    n_lesion_edges: Optional[int] = None
    n_workspace_hubs: int = 6

    def __post_init__(self):
        if self.grain not in GRAINS:
            raise ValueError(f"grain must be one of {GRAINS}")
        if self.lesion not in LESIONS:
            raise ValueError(f"lesion must be one of {LESIONS}")
        for name in ("rate", "dt", "fs_out", "duration_sec"):
            if not (math.isfinite(getattr(self, name)) and getattr(self, name) > 0):
                raise ValueError(f"{name} must be > 0")
        if not (self.G >= 0 and self.sigma >= 0 and self.f_sd >= 0):
            raise ValueError("G, sigma and f_sd must be >= 0")
        step = round(1.0 / (self.fs_out * self.dt))
        if step < 1 or not math.isclose(
            step * self.dt * self.fs_out, 1.0, rel_tol=1e-9
        ):
            raise ValueError("1 / (fs_out * dt) must be a positive integer")
        if self.rate * self.dt > 0.1:
            raise ValueError("rate * dt must be <= 0.1 (Euler-Maruyama step)")

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def replace(self, **kw) -> "WholeBrainConfig":
        from dataclasses import replace

        return replace(self, **kw)


def whole_brain_config_from_dict(d: Optional[dict]) -> WholeBrainConfig:
    if not d:
        return WholeBrainConfig()
    unknown = sorted(set(d) - set(WholeBrainConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(f"Unknown whole-brain config field(s): {unknown}")
    return WholeBrainConfig(**d)


def kuramoto_order(z: np.ndarray) -> np.ndarray:
    """Kuramoto order parameter R(t) = |mean_j exp(i arg z_j(t))|."""
    ph = np.exp(1j * np.angle(z))
    return np.abs(ph.mean(axis=0))


def simulate_whole_brain(
    config: Optional[WholeBrainConfig] = None,
    seed: int = 0,
    connectome: Optional[Connectome] = None,
) -> BenchSystem:
    """
    Simulate the Hopf whole-brain model (module docstring). Returns a
    :class:`BenchSystem` with ``ts`` = ``Re z`` (regions x samples at
    ``fs_out``), an empty events table (no task; RAM/SRPI are undefined),
    declared ``meta`` (connectome provenance, G, lesion, anatomical workspace
    hubs of the intact connectome, four cortical macro nodes) and an
    ``oracle`` with the order parameter trace, its mean and SD
    (metastability), the envelope ``|z|``, the lesion mask and the coupling
    actually used. Random streams: structure (frequencies), noise and lesion
    draws are independent children of ``seed``.
    """
    cfg = config if config is not None else WholeBrainConfig()
    conn = connectome if connectome is not None else load_connectome(grain=cfg.grain)
    freq_rng, noise_rng, lesion_rng = _streams(seed, 3)
    n_lesion = cfg.n_lesion_edges
    mask = lesion_mask(
        conn,
        cfg.lesion,
        n_hubs=cfg.n_hubs,
        n_edges=n_lesion,
        seed=int(lesion_rng.integers(0, 2**31 - 1)),
    )
    W = apply_lesion(conn, mask)
    n = conn.n
    freqs = np.maximum(freq_rng.normal(cfg.f_mean, cfg.f_sd, size=n), 0.1)
    omega = 2.0 * np.pi * freqs
    rot = np.exp(1j * omega * cfg.dt)
    lam = float(cfg.rate)
    strength = W.sum(axis=1)
    step = int(round(1.0 / (cfg.fs_out * cfg.dt)))
    n_out = int(round(cfg.duration_sec * cfg.fs_out))
    n_skip = int(round(cfg.transient_sec / cfg.dt))
    n_steps = n_skip + n_out * step
    noise_sd = lam * cfg.sigma * math.sqrt(cfg.dt / 2.0)
    z = 0.1 * (noise_rng.standard_normal(n) + 1j * noise_rng.standard_normal(n))
    out = np.zeros((n, n_out), dtype=complex)
    GW = float(cfg.G) * W
    Gs = float(cfg.G) * strength
    k_out = 0
    for t in range(n_steps):
        drift = (cfg.a - (z.real**2 + z.imag**2) - Gs) * z + GW @ z
        z = z * rot + cfg.dt * lam * drift
        z = z + noise_sd * (
            noise_rng.standard_normal(n) + 1j * noise_rng.standard_normal(n)
        )
        if t >= n_skip and (t - n_skip) % step == step - 1:
            out[:, k_out] = z
            k_out += 1
    R = kuramoto_order(out)
    hubs = conn.hubs(cfg.n_workspace_hubs)
    meta = {
        "generator_version": WHOLE_BRAIN_VERSION,
        "family": "whole_brain",
        "dynamics": "hopf_whole_brain",
        "substrate": "stuart_landau",
        "seed": int(seed),
        "dt": float(1.0 / cfg.fs_out),
        "sampling_frequency": float(cfg.fs_out),
        "n_nodes": int(n),
        "n_time": int(n_out),
        "knobs": None,
        "whole_brain_config": cfg.to_dict(),
        "G": float(cfg.G),
        "lesion": cfg.lesion,
        "principles": list(PRINCIPLES),
        "connectome": dict(conn.provenance),
        "region_labels": list(conn.labels),
        "hemisphere": list(conn.hemisphere),
        "lobe": list(conn.lobe),
        "modules": conn.modules(),
        "module_order": sorted(conn.modules()),
        "workspace_nodes": hubs,
        "workspace_rule": f"top-{cfg.n_workspace_hubs} strength regions of the "
        "intact connectome (declared by anatomy, fixed across manipulations)",
        "bearer_nodes": list(range(n)),
        "iim_macro_nodes": conn.macro_nodes(),
        "iim_grain": "cortical_hemisphere_x_anterior_posterior",
        "iim_lag_samples": 2,
        "iim_bins": 2,
        "has_rest_run": False,
    }
    oracle = {
        "family": "whole_brain",
        "seed": int(seed),
        "intended_bits": None,
        "ground_truth": "none (external generator; no stipulated mechanism pattern)",
        "order_parameter": R,
        "mean_order": float(R.mean()),
        "metastability": float(R.std()),
        "envelope": np.abs(out).astype(np.float32),
        "natural_frequency_hz": freqs,
        "lesion_mask": mask,
        "n_edges_removed": int(np.count_nonzero(np.triu(mask, 1))),
        "coupling": W,
        "G": float(cfg.G),
    }
    events = pd.DataFrame(
        columns=["onset", "duration", "trial_type", "value", "impact_channel"]
    )
    return BenchSystem(ts=out.real.copy(), events=events, meta=meta, oracle=oracle)


def g_sweep_levels(n_levels: int = 8, g_max: float = 4.0) -> List[float]:
    """Global-coupling levels ``0 .. g_max`` (first level: uncoupled)."""
    if int(n_levels) < 2:
        raise ValueError("n_levels must be >= 2")
    return [round(float(v), 6) for v in np.linspace(0.0, float(g_max), int(n_levels))]


def manipulations(
    g_levels: Sequence[float] = tuple(g_sweep_levels()),
    lesions: Sequence[str] = ("hub", "interhemispheric", "long_range"),
    base: Optional[WholeBrainConfig] = None,
) -> List[dict]:
    """The external manipulation set: a G sweep plus lesions at the base G
    (each lesion followed by its size-matched random-edge control)."""
    base = base if base is not None else WholeBrainConfig()
    out = [{"name": f"G{g:g}", "config": base.replace(G=float(g))} for g in g_levels]
    conn = None
    for kind in lesions:
        out.append({"name": f"lesion_{kind}", "config": base.replace(lesion=kind)})
        if conn is None:
            conn = load_connectome(grain=base.grain)
        n_e = int(
            np.count_nonzero(np.triu(lesion_mask(conn, kind, n_hubs=base.n_hubs), 1))
        )
        out.append(
            {
                "name": f"lesion_random_matched_{kind}",
                "config": base.replace(lesion="random", n_lesion_edges=n_e),
            }
        )
    return out
