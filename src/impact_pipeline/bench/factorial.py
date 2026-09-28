"""
MPC-Bench factorial design: the 2^5 mechanism cells of {0,1}^5 x seeds.

A cell's bit pattern (``PRINCIPLES`` order) switches each mechanism on at its
nominal dose (1) or off (0); ``cell_id`` is ``'b'`` followed by the five bits
(e.g. ``b11011`` = NAS mechanism off). The ``ff_only`` NAS variant is a
witness (see ``witnesses.yaml``), not a factorial cell.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, List, Optional, Sequence

from impact_pipeline.bench.generators import (
    NOMINAL_KNOBS,
    Knobs,
    all_bit_patterns,
)

FAMILY_GENERATOR = {"A": "family_a", "C": "family_c"}


@dataclass(frozen=True)
class BenchTask:
    """One simulated system to generate and score."""

    task_id: str
    design: str
    family: str
    generator: str
    cell_id: str
    knobs: dict
    seed: int
    config: dict = field(default_factory=dict)
    witness_id: Optional[str] = None
    sweep_knob: Optional[str] = None
    sweep_level: Optional[float] = None
    bearer_mode: str = "system"

    def to_dict(self) -> dict:
        return asdict(self)


def cell_id(bits: Sequence[int]) -> str:
    return "b" + "".join(str(int(b)) for b in bits)


def factorial_cells(nominal: Knobs = None) -> List[dict]:
    """The 32 cells: ``cell_id``, ``bits`` and the knob dict realising them."""
    base = nominal if nominal is not None else NOMINAL_KNOBS
    cells = []
    for bits in all_bit_patterns():
        kn = Knobs.from_bits(bits, base)
        if tuple(kn.bits()) != tuple(bits):
            raise AssertionError(f"knobs do not realise {bits}")
        cells.append(
            {"cell_id": cell_id(bits), "bits": list(bits), "knobs": kn.to_dict()}
        )
    return cells


def factorial_tasks(
    seeds: Iterable[int],
    family: str = "A",
    config: Optional[dict] = None,
    nominal: Knobs = None,
    cells: Optional[Sequence[str]] = None,
) -> List[BenchTask]:
    """Tasks for every (cell, seed); ``cells`` restricts to given cell ids."""
    family = str(family).upper()
    if family not in FAMILY_GENERATOR:
        raise ValueError(f"family must be one of {sorted(FAMILY_GENERATOR)}")
    wanted = None if cells is None else set(cells)
    tasks = []
    for cell in factorial_cells(nominal):
        if wanted is not None and cell["cell_id"] not in wanted:
            continue
        for seed in seeds:
            tasks.append(
                BenchTask(
                    task_id=f"factorial-{family}-{cell['cell_id']}-s{int(seed):05d}",
                    design="factorial",
                    family=family,
                    generator=FAMILY_GENERATOR[family],
                    cell_id=cell["cell_id"],
                    knobs=dict(cell["knobs"]),
                    seed=int(seed),
                    config=dict(config or {}),
                )
            )
    return tasks
