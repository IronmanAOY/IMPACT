"""
Seed policy of MPC-Bench.

* Development seeds are 0-999; confirmatory seeds are >= 20000; the seeds
  in between are unassigned. Every other value is refused.
* One run uses one split: development and confirmatory seeds are never mixed.
* The split label of a record is derived from its seed (:func:`split_of`),
  never written by hand.
* The policy classifies task seeds and seed bases. Seeds that a task derives
  from them (null seeds, hashed replicate seeds) are not classified.

Random streams (new draws come only from new ``SeedSequence`` keys; existing
draws are never re-ordered): the structural stream and the streams of a
system's own run stay ``SeedSequence(seed)``; twins with replicate
``r >= 1`` draw the task schedule, process noise and rest from
``SeedSequence([seed, r])`` with ``1 <= r <= 30``; label errors, cue-onset
jitter and the staggered driver use the reserved keys 41, 42 and 43
(``SeedSequence([seed, key])``), which a twin index can therefore never
collide with. Keys 31-40 are reserved for further named streams, so a twin
index stops at 30; the largest twin set has 8 sessions per network.
"""

from __future__ import annotations

import numbers
from typing import Iterable, Optional

import numpy as np

DEVELOPMENT = "development"
CONFIRMATORY = "confirmatory"
SPLITS = (DEVELOPMENT, CONFIRMATORY)

DEV_SEED_MIN = 0
DEV_SEED_MAX = 999
CONFIRMATORY_SEED_MIN = 20000

REFERENCE_BLOCK = range(900, 940)
SMOKE_SEEDS = range(980, 985)

STREAM_KEYS = {"label_error": 41, "cue_jitter": 42, "staggered_driver": 43}
TWIN_REPLICATE_MIN = 1
TWIN_REPLICATE_MAX = 30


class SeedPolicyError(ValueError):
    """A seed or a set of seeds that breaks the seed policy."""


def _as_seed(seed) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral):
        raise SeedPolicyError(f"a seed is an integer, got {seed!r}")
    return int(seed)


def split_of(seed) -> str:
    """``'development'`` (0-999) or ``'confirmatory'`` (>= 20000); raises
    :class:`SeedPolicyError` for negative and unassigned seeds."""
    s = _as_seed(seed)
    if DEV_SEED_MIN <= s <= DEV_SEED_MAX:
        return DEVELOPMENT
    if s >= CONFIRMATORY_SEED_MIN:
        return CONFIRMATORY
    if s < 0:
        raise SeedPolicyError(f"seed {s} is negative")
    raise SeedPolicyError(
        f"seed {s} is unassigned (development {DEV_SEED_MIN}-{DEV_SEED_MAX}, "
        f"confirmatory >= {CONFIRMATORY_SEED_MIN})"
    )


def check_seeds(seeds: Iterable, split: Optional[str] = None) -> str:
    """
    The split of a run's seeds. Every seed must be admissible
    (:func:`split_of`), all seeds must share one split (development and
    confirmatory seeds are never mixed in one run) and, when ``split`` is
    given, it must be that split (so a confirmatory run refuses every seed
    below 20000). Returns the split; raises :class:`SeedPolicyError`.
    """
    if split is not None and split not in SPLITS:
        raise SeedPolicyError(f"split must be one of {SPLITS}, got {split!r}")
    seeds = list(seeds)
    if not seeds:
        raise SeedPolicyError("no seeds")
    by_split = {}
    for s in seeds:
        by_split.setdefault(split_of(s), []).append(int(s))
    if len(by_split) > 1:
        raise SeedPolicyError(
            "development and confirmatory seeds are mixed in one run: "
            f"development {by_split[DEVELOPMENT][:5]}, "
            f"confirmatory {by_split[CONFIRMATORY][:5]}"
        )
    (found,) = by_split
    if split is not None and found != split:
        want = (
            f">= {CONFIRMATORY_SEED_MIN}" if split == CONFIRMATORY
            else f"{DEV_SEED_MIN}-{DEV_SEED_MAX}"
        )
        raise SeedPolicyError(
            f"a {split} run uses seeds {want}; got {sorted(set(by_split[found]))[:5]}"
        )
    return found


def assert_development(seeds: Iterable) -> None:
    check_seeds(seeds, DEVELOPMENT)


def assert_confirmatory(seeds: Iterable) -> None:
    check_seeds(seeds, CONFIRMATORY)


def is_smoke_seed(seed) -> bool:
    """Seeds 980-984: smoke tests of held-out conditions, whose outputs the
    harness discards unread."""
    return _as_seed(seed) in SMOKE_SEEDS


# --------------------------------------------------------------------------
# random streams
# --------------------------------------------------------------------------
def structural_seed_sequence(seed) -> np.random.SeedSequence:
    """``SeedSequence(seed)``: the structural stream (and the streams of a
    system's own run)."""
    return np.random.SeedSequence(_as_seed(seed))


def stream_seed_sequence(seed, stream: str) -> np.random.SeedSequence:
    """``SeedSequence([seed, key])`` of a reserved named stream
    (``label_error`` 41, ``cue_jitter`` 42, ``staggered_driver`` 43)."""
    if stream not in STREAM_KEYS:
        raise SeedPolicyError(
            f"unknown stream {stream!r}; one of {sorted(STREAM_KEYS)}")
    return np.random.SeedSequence([_as_seed(seed), STREAM_KEYS[stream]])


def twin_seed_sequence(seed, replicate) -> np.random.SeedSequence:
    """``SeedSequence([seed, r])`` of twin replicate ``1 <= r <= 30`` (the
    schedule, process-noise and rest streams; ``r = 0`` is the witness run
    itself and uses ``SeedSequence(seed)``)."""
    r = _as_seed(replicate)
    if not TWIN_REPLICATE_MIN <= r <= TWIN_REPLICATE_MAX:
        raise SeedPolicyError(
            f"twin replicate must lie in {TWIN_REPLICATE_MIN}-{TWIN_REPLICATE_MAX} "
            f"(keys {sorted(STREAM_KEYS.values())} are reserved), got {r}"
        )
    return np.random.SeedSequence([_as_seed(seed), r])


__all__ = [
    "CONFIRMATORY",
    "CONFIRMATORY_SEED_MIN",
    "DEVELOPMENT",
    "DEV_SEED_MAX",
    "DEV_SEED_MIN",
    "REFERENCE_BLOCK",
    "SMOKE_SEEDS",
    "SPLITS",
    "STREAM_KEYS",
    "SeedPolicyError",
    "TWIN_REPLICATE_MAX",
    "TWIN_REPLICATE_MIN",
    "assert_confirmatory",
    "assert_development",
    "check_seeds",
    "is_smoke_seed",
    "split_of",
    "stream_seed_sequence",
    "structural_seed_sequence",
    "twin_seed_sequence",
]
