"""
Seed policy of MPC-Bench v2.

* Development seeds are 0-999; confirmatory seeds are >= 20000; the v1
  confirmatory block 10000-19999 is never reused, and 1000-9999 is
  unassigned. Every other value is refused.
* One run uses one split: development and confirmatory seeds are never mixed.
* The split label of a record is derived from its seed (:func:`split_of`),
  never written by hand.
* The policy classifies task seeds and seed bases. Seeds that a task derives
  from them (null seeds, hashed replicate seeds) are not classified.

Random streams (new draws come only from new ``SeedSequence`` keys; existing
draws are never re-ordered): the structural stream and every v1 stream stay
``SeedSequence(seed)``; twins with replicate ``r >= 1`` draw the task
schedule, process noise and rest from ``SeedSequence([seed, r])`` with
``1 <= r <= 30``; label errors, cue-onset jitter and the staggered driver use
the reserved keys 41, 42 and 43 (``SeedSequence([seed, key])``), which a twin
index can therefore never collide with. Keys 31-40 stay free for further
named streams (the benchmark design names 31 and 32 for its substrate
nulls), so a twin index stops at 30; the largest twin set has 8 sessions
per network.

The seed map (``protocols/v2/seed_map_v2.json``) records the use of every
block; :func:`load_seed_map` validates it against this policy.
"""

from __future__ import annotations

import json
import numbers
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

DEVELOPMENT = "development"
CONFIRMATORY = "confirmatory"
SPLITS = (DEVELOPMENT, CONFIRMATORY)

DEV_SEED_MIN = 0
DEV_SEED_MAX = 999
V1_CONFIRMATORY_MIN = 10000
V1_CONFIRMATORY_MAX = 19999
CONFIRMATORY_SEED_MIN = 20000

REFERENCE_BLOCK = range(900, 940)
SMOKE_SEEDS = range(980, 985)

STREAM_KEYS = {"label_error": 41, "cue_jitter": 42, "staggered_driver": 43}
TWIN_REPLICATE_MIN = 1
TWIN_REPLICATE_MAX = 30

SEED_MAP_SCHEMA = "mpc-bench-seed-map/2"
SEED_MAP_PATH = (
    Path(__file__).resolve().parents[3] / "protocols" / "v2" / "seed_map_v2.json"
)


class SeedPolicyError(ValueError):
    """A seed or a set of seeds that breaks the v2 seed policy."""


def _as_seed(seed) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral):
        raise SeedPolicyError(f"a seed is an integer, got {seed!r}")
    return int(seed)


def split_of(seed) -> str:
    """``'development'`` (0-999) or ``'confirmatory'`` (>= 20000); raises
    :class:`SeedPolicyError` for negative seeds, the unassigned block
    1000-9999 and the v1 block 10000-19999."""
    s = _as_seed(seed)
    if DEV_SEED_MIN <= s <= DEV_SEED_MAX:
        return DEVELOPMENT
    if s >= CONFIRMATORY_SEED_MIN:
        return CONFIRMATORY
    if V1_CONFIRMATORY_MIN <= s <= V1_CONFIRMATORY_MAX:
        raise SeedPolicyError(
            f"seed {s} lies in the v1 confirmatory block "
            f"{V1_CONFIRMATORY_MIN}-{V1_CONFIRMATORY_MAX}, which v2 never reuses"
        )
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
    """``SeedSequence(seed)``: the structural stream (and every v1 stream)."""
    return np.random.SeedSequence(_as_seed(seed))


def stream_seed_sequence(seed, stream: str) -> np.random.SeedSequence:
    """``SeedSequence([seed, key])`` of a reserved v2 stream
    (``label_error`` 41, ``cue_jitter`` 42, ``staggered_driver`` 43)."""
    if stream not in STREAM_KEYS:
        raise SeedPolicyError(
            f"unknown stream {stream!r}; one of {sorted(STREAM_KEYS)}")
    return np.random.SeedSequence([_as_seed(seed), STREAM_KEYS[stream]])


def twin_seed_sequence(seed, replicate) -> np.random.SeedSequence:
    """``SeedSequence([seed, r])`` of twin replicate ``1 <= r <= 30`` (the
    schedule, process-noise and rest streams; ``r = 0`` is the witness run
    itself and uses the v1 streams)."""
    r = _as_seed(replicate)
    if not TWIN_REPLICATE_MIN <= r <= TWIN_REPLICATE_MAX:
        raise SeedPolicyError(
            f"twin replicate must lie in {TWIN_REPLICATE_MIN}-{TWIN_REPLICATE_MAX} "
            f"(keys {sorted(STREAM_KEYS.values())} are reserved), got {r}"
        )
    return np.random.SeedSequence([_as_seed(seed), r])


# --------------------------------------------------------------------------
# seed map
# --------------------------------------------------------------------------
def _block(entry, where):
    lo, hi = entry.get("min"), entry.get("max")
    if not isinstance(lo, int) or isinstance(lo, bool):
        raise SeedPolicyError(f"{where}: 'min' must be an integer")
    if hi is None:
        return lo, None
    if not isinstance(hi, int) or isinstance(hi, bool) or hi < lo:
        raise SeedPolicyError(f"{where}: 'max' must be an integer >= min")
    return lo, hi


def _overlaps(a, b) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def validate_seed_map(payload: dict) -> dict:
    """
    Check a seed map against the policy: schema; the policy block equals the
    constants of this module; every development block (and every block of
    seeds used before v2) lies in 0-999 and every confirmatory block at or
    above 20000 (extensions included); the development assignments are
    pairwise disjoint and disjoint from the seeds used before v2, except
    blocks marked ``reuse_allowed``; parts lie inside
    their block and are pairwise disjoint; the smoke block is 980-984 and
    marked as discarded; the stream keys equal :data:`STREAM_KEYS`. Returns
    the payload; raises :class:`SeedPolicyError`.
    """
    if payload.get("schema") != SEED_MAP_SCHEMA:
        raise SeedPolicyError(f"seed map schema must be {SEED_MAP_SCHEMA!r}")
    pol = payload.get("policy") or {}
    if pol.get("development") != {"min": DEV_SEED_MIN, "max": DEV_SEED_MAX}:
        raise SeedPolicyError("policy.development differs from the seed policy")
    if pol.get("confirmatory") != {"min": CONFIRMATORY_SEED_MIN}:
        raise SeedPolicyError("policy.confirmatory differs from the seed policy")
    never = [_block(b, "policy.never_reused") for b in pol.get("never_reused") or []]
    if never != [(V1_CONFIRMATORY_MIN, V1_CONFIRMATORY_MAX)]:
        raise SeedPolicyError("policy.never_reused must be the v1 block 10000-19999")
    if pol.get("one_split_per_run") is not True:
        raise SeedPolicyError("policy.one_split_per_run must be true")
    streams = payload.get("streams") or {}
    if streams.get("keys") != STREAM_KEYS:
        raise SeedPolicyError(f"stream keys must be {STREAM_KEYS}")
    twin = streams.get("twin") or {}
    twin_range = (twin.get("r_min"), twin.get("r_max"))
    if twin_range != (TWIN_REPLICATE_MIN, TWIN_REPLICATE_MAX):
        raise SeedPolicyError("twin replicate range differs from the seed policy")

    def _check_parts(entry, block, where):
        parts = [(_block(p, f"{where}.parts"), p) for p in entry.get("parts") or []]
        for (lo, hi), _p in parts:
            if hi is None or lo < block[0] or hi > block[1]:
                raise SeedPolicyError(f"{where}: part {lo}-{hi} outside its block")
        for i, (a, _pa) in enumerate(parts):
            for b, _pb in parts[i + 1:]:
                if _overlaps(a, b):
                    raise SeedPolicyError(f"{where}: parts {a} and {b} overlap")

    dev = payload.get("development") or {}
    used = [_block(b, "development.used_before_v2")
            for b in dev.get("used_before_v2") or []]
    for lo, hi in used:
        if hi is None or lo < DEV_SEED_MIN or hi > DEV_SEED_MAX:
            raise SeedPolicyError(
                f"development.used_before_v2: {lo}-{hi} is not a development block")
    blocks = []
    for k, entry in enumerate(dev.get("assignments") or []):
        where = f"development.assignments[{k}]"
        lo, hi = _block(entry, where)
        if hi is None or lo < DEV_SEED_MIN or hi > DEV_SEED_MAX:
            raise SeedPolicyError(f"{where}: {lo}-{hi} is not a development block")
        if not entry.get("use"):
            raise SeedPolicyError(f"{where}: no 'use'")
        _check_parts(entry, (lo, hi), where)
        if not entry.get("reuse_allowed"):
            clash = [u for u in used if _overlaps((lo, hi), u)]
            if clash:
                raise SeedPolicyError(
                    f"{where}: {lo}-{hi} reuses seeds used before v2 {clash}")
        blocks.append(((lo, hi), where))
    for i, (a, wa) in enumerate(blocks):
        for b, wb in blocks[i + 1:]:
            if _overlaps(a, b):
                raise SeedPolicyError(f"{wa} and {wb} overlap")
    smoke = [
        e for e in dev.get("assignments") or []
        if (e.get("min"), e.get("max")) == (SMOKE_SEEDS.start, SMOKE_SEEDS.stop - 1)
    ]
    if len(smoke) != 1 or smoke[0].get("outputs_discarded_unread") is not True:
        raise SeedPolicyError(
            "the smoke block 980-984 must be declared with discarded outputs")
    conf = payload.get("confirmatory") or {}
    for k, entry in enumerate(conf.get("assignments") or []):
        where = f"confirmatory.assignments[{k}]"
        lo, hi = _block(entry, where)
        if hi is None or lo < CONFIRMATORY_SEED_MIN:
            raise SeedPolicyError(f"{where}: {lo}-{hi} is not a confirmatory block")
        ext = entry.get("extendable_to")
        if ext is not None and (not isinstance(ext, int) or ext < hi):
            raise SeedPolicyError(f"{where}: extendable_to must be an integer >= max")
        if not entry.get("use"):
            raise SeedPolicyError(f"{where}: no 'use'")
        _check_parts(entry, (lo, hi), where)
    return payload


def load_seed_map(path=None) -> dict:
    """The validated seed map (default ``protocols/v2/seed_map_v2.json``)."""
    p = Path(path) if path is not None else SEED_MAP_PATH
    return validate_seed_map(json.loads(p.read_text(encoding="utf-8")))


def seed_uses(seed, seed_map: Optional[dict] = None) -> List[str]:
    """The declared uses of a seed in the seed map (block and part uses, in
    map order); empty when no block covers it. The seed must be admissible."""
    s = _as_seed(seed)
    split = split_of(s)
    smap = seed_map if seed_map is not None else load_seed_map()
    out = []
    for entry in (smap.get(split) or {}).get("assignments") or []:
        hi = entry["max"] if entry.get("extendable_to") is None else (
            entry["extendable_to"])
        if entry["min"] <= s <= hi:
            out.append(entry["use"])
            for part in entry.get("parts") or []:
                if part["min"] <= s <= part["max"]:
                    out.append(f"{entry['use']}: {part['use']}")
    return out


__all__ = [
    "CONFIRMATORY",
    "CONFIRMATORY_SEED_MIN",
    "DEVELOPMENT",
    "DEV_SEED_MAX",
    "DEV_SEED_MIN",
    "REFERENCE_BLOCK",
    "SEED_MAP_PATH",
    "SMOKE_SEEDS",
    "SPLITS",
    "STREAM_KEYS",
    "SeedPolicyError",
    "TWIN_REPLICATE_MAX",
    "TWIN_REPLICATE_MIN",
    "V1_CONFIRMATORY_MAX",
    "V1_CONFIRMATORY_MIN",
    "assert_confirmatory",
    "assert_development",
    "check_seeds",
    "is_smoke_seed",
    "load_seed_map",
    "seed_uses",
    "split_of",
    "stream_seed_sequence",
    "structural_seed_sequence",
    "twin_seed_sequence",
    "validate_seed_map",
]
