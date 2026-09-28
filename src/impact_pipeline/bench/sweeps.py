"""
MPC-Bench dose-response sweeps: one knob varied over ``n_levels`` doses (10
by default, the first level switching the mechanism off) with the other
knobs at their nominal doses, x seeds.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from impact_pipeline.bench.factorial import FAMILY_GENERATOR, BenchTask
from impact_pipeline.bench.generators import NOMINAL_KNOBS, Knobs

SWEEP_KNOBS = ("eta", "K", "g_b", "c_int", "e")
SWEEP_RANGES = {
    "eta": (0.0, 0.6),
    "K": (1, 10),
    "g_b": (0.0, 2.0),
    "c_int": (0.0, 1.2),
    "e": (0.0, 2.0),
}


def sweep_levels(knob: str, n_levels: int = 10) -> List[float]:
    """Dose levels of ``knob`` (first level = mechanism off). ``K`` levels are
    integers (1..10 for 10 levels)."""
    if knob not in SWEEP_RANGES:
        raise ValueError(f"knob must be one of {SWEEP_KNOBS}")
    n_levels = int(n_levels)
    if n_levels < 2:
        raise ValueError("n_levels must be >= 2")
    lo, hi = SWEEP_RANGES[knob]
    if knob == "K":
        vals = np.unique(np.round(np.linspace(lo, hi, n_levels)).astype(int))
        return [int(v) for v in vals]
    return [round(float(v), 6) for v in np.linspace(lo, hi, n_levels)]


def sweep_tasks(
    seeds: Iterable[int],
    knobs: Sequence[str] = SWEEP_KNOBS,
    family: str = "A",
    n_levels: int = 10,
    config: Optional[dict] = None,
    nominal: Knobs = None,
) -> List[BenchTask]:
    family = str(family).upper()
    if family not in FAMILY_GENERATOR:
        raise ValueError(f"family must be one of {sorted(FAMILY_GENERATOR)}")
    base = nominal if nominal is not None else NOMINAL_KNOBS
    seeds = [int(s) for s in seeds]
    tasks = []
    for knob in knobs:
        for li, level in enumerate(sweep_levels(knob, n_levels)):
            kn = base.replace(**{knob: level})
            for seed in seeds:
                tasks.append(
                    BenchTask(
                        task_id=f"sweep-{family}-{knob}-l{li:02d}-s{seed:05d}",
                        design="sweep",
                        family=family,
                        generator=FAMILY_GENERATOR[family],
                        cell_id=f"{knob}_l{li:02d}",
                        knobs=kn.to_dict(),
                        seed=seed,
                        config=dict(config or {}),
                        sweep_knob=knob,
                        sweep_level=float(level),
                    )
                )
    return tasks


def dose_response_slopes(
    df: pd.DataFrame,
    value_columns: Sequence[str],
    knob_column: str = "sweep_knob",
    level_column: str = "sweep_level",
) -> pd.DataFrame:
    """
    OLS slope of each value column on the dose level, per swept knob (levels
    rescaled to [0, 1] so slopes are comparable across knobs), with the
    number of finite points. Rows with NaN values are dropped per column.
    """
    rows: List[Dict] = []
    for knob, sub in df.groupby(knob_column):
        lv = sub[level_column].astype(float).to_numpy()
        span = float(np.ptp(lv)) if lv.size else 0.0
        x = (lv - lv.min()) / span if span > 0 else np.zeros_like(lv)
        for col in value_columns:
            y = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(y) & np.isfinite(x)
            if ok.sum() < 3 or np.ptp(x[ok]) == 0:
                slope = np.nan
            else:
                slope = float(np.polyfit(x[ok], y[ok], 1)[0])
            rows.append(
                {"knob": knob, "value": col, "slope": slope, "n": int(ok.sum())}
            )
    return pd.DataFrame(rows)
