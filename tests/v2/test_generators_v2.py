# -*- coding: utf-8 -*-
"""
Generator additions of MPC-Bench v2 (the twin option ``replicate``):

* ``replicate = 0`` reproduces every v1 generator, adversary and witness bit
  for bit, checked against the generator code of the v1 freeze tag (run in
  this process) and against the v1 witness builder;
* a twin ``replicate = r >= 1`` keeps the structural streams and the
  structural hash and draws a new task schedule, noise and rest run;
* every v2 option and configuration field defaults to its v1 value.
"""
import inspect
import math
import subprocess
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from impact_pipeline.bench import adversarial_v2 as A2
from impact_pipeline.bench import generators as g
from impact_pipeline.bench import manipulation_v2 as MV
from impact_pipeline.bench import witnesses
# Imported before any frozen code runs, so that their own imports of the
# generator module bind to the current one (the frozen code imports them
# lazily).
from impact_pipeline.bench import adversarial, compat, forward  # noqa: F401
from impact_pipeline.bench import patchwork, whole_brain  # noqa: F401
from impact_pipeline.v2 import GENERATOR_VERSION_V2
from impact_pipeline.v2 import seeds as S

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
FREEZE_TAG_V1 = "mpcbench-freeze-v1"
BENCH = "src/impact_pipeline/bench"
SMALL = g.AgentConfig(n_trials=16, n_reafference_pairs=6)
MIXED = g.Knobs(eta=0.2, K=4, g_b=0.7, c_int=0.4, e=0.5)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _same(a, b, where="") -> list:
    """Differences between two nested values (exact; NaN equals NaN)."""
    if isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(a, b, check_exact=True)
        except AssertionError as exc:
            return [f"{where}: {exc}"]
        return []
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.dtype != b.dtype or a.shape != b.shape:
            return [f"{where}: {a.dtype}{a.shape} != {b.dtype}{b.shape}"]
        eq = np.array_equal(a, b, equal_nan=a.dtype.kind in "fc")
        return [] if eq else [f"{where}: arrays differ"]
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f"{where}: keys {sorted(set(a) ^ set(b))}"]
        out = []
        for k in a:
            out += _same(a[k], b[k], f"{where}.{k}")
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            return [f"{where}: sequences differ"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += _same(x, y, f"{where}[{i}]")
        return out
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return []
    if type(a) is not type(b) or a != b:
        return [f"{where}: {a!r} != {b!r}"]
    return []


def assert_same_system(a, b):
    diffs = (_same(a.ts, b.ts, "ts") + _same(a.events, b.events, "events")
             + _same(a.meta, b.meta, "meta") + _same(a.oracle, b.oracle, "oracle")
             + _same(a.rest_ts, b.rest_ts, "rest_ts"))
    assert not diffs, diffs[:10]


def _git_show(ref, rel):
    try:
        proc = subprocess.run(["git", "-C", str(REPO_ROOT), "show", f"{ref}:{rel}"],
                              capture_output=True, text=True, timeout=60)
    except Exception:  # noqa: BLE001
        return None
    return proc.stdout if proc.returncode == 0 else None


def _exec_module(name, source, path):
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod  # dataclasses resolve annotations through sys.modules
    exec(compile(source, str(path), "exec"), mod.__dict__)
    return mod


REF_MODULES = ("generators", "patchwork", "adversarial", "witnesses")


@pytest.fixture(scope="module")
def v1_ref():
    """The v1 generator modules of the freeze tag, executed in this process
    and bound to each other (the frozen ``generators`` everywhere)."""
    sources = {m: _git_show(FREEZE_TAG_V1, f"{BENCH}/{m}.py") for m in REF_MODULES}
    if any(s is None for s in sources.values()):
        pytest.skip(f"git tag {FREEZE_TAG_V1} not available")
    names = {m: f"_mpcbench_v1_reference_{m}" for m in REF_MODULES}
    real = sys.modules["impact_pipeline.bench.generators"]
    mods = {}
    try:
        mods["generators"] = _exec_module(names["generators"], sources["generators"],
                                          SRC / "impact_pipeline/bench/generators.py")
        sys.modules["impact_pipeline.bench.generators"] = mods["generators"]
        for m in REF_MODULES[1:]:
            mods[m] = _exec_module(names[m], sources[m],
                                   SRC / f"impact_pipeline/bench/{m}.py")
    finally:
        sys.modules["impact_pipeline.bench.generators"] = real
    yield types.SimpleNamespace(**mods)
    for n in names.values():
        sys.modules.pop(n, None)


def as_ref(ref, value):
    """A configuration or knobs object of the frozen classes (the frozen
    code checks ``isinstance`` against its own classes)."""
    if value is None:
        return None
    if isinstance(value, g.AgentConfig):
        return ref.generators.config_from_dict(value.to_dict())
    if isinstance(value, g.Knobs):
        return ref.generators.knobs_from_dict(value.to_dict())
    raise TypeError(type(value))


@contextmanager
def frozen(ref):
    """Resolve the lazy imports of the frozen code (patchwork, adversarial)
    to the frozen modules while it runs."""
    keys = {f"impact_pipeline.bench.{m}": getattr(ref, m) for m in REF_MODULES}
    saved = {k: sys.modules.get(k) for k in keys}
    sys.modules.update(keys)
    try:
        yield
    finally:
        for k, v in saved.items():
            sys.modules[k] = v


# --------------------------------------------------------------------------
# replicate = 0 is v1, bit for bit
# --------------------------------------------------------------------------
V1_CASES = [
    ("family_a", None, None, 0, "A"),
    ("family_a", None, SMALL, 5, "A"),
    ("family_a", g.OFF_KNOBS, SMALL, 6, "A"),
    ("family_a", MIXED, SMALL.replace(rest_sec=10.0), 7, "A"),
    ("family_a", None, SMALL.replace(n_modules=6, feedback_mode="scrambled"), 8, "A"),
    ("family_c", None, None, 0, "C"),
    ("family_c", MIXED, SMALL.replace(rest_sec=5.0), 9, "C"),
    ("patchwork", None, SMALL, 10, "A"),
    ("null_independent_noise", None, SMALL, 11, "A"),
    ("null_ar1", None, SMALL, 12, "C"),
    ("hypersynchronous", None, SMALL, 13, "A"),
]


def test_v1_case_list_covers_every_v1_generator(v1_ref):
    assert {c[0] for c in V1_CASES} == set(v1_ref.generators.GENERATOR_NAMES)
    assert v1_ref.generators.GENERATOR_VERSION == g.GENERATOR_VERSION


@pytest.mark.parametrize("gen,knobs,cfg,seed,family", V1_CASES)
def test_replicate_zero_reproduces_every_v1_generator(
    v1_ref, gen, knobs, cfg, seed, family
):
    now = g.make_system(gen, knobs, cfg, seed, template_family=family, replicate=0)
    default = g.make_system(gen, knobs, cfg, seed, template_family=family)
    with frozen(v1_ref):
        then = v1_ref.generators.make_system(gen, as_ref(v1_ref, knobs),
                                             as_ref(v1_ref, cfg), seed,
                                             template_family=family)
    assert_same_system(now, then)
    assert_same_system(default, then)


@pytest.mark.parametrize("simulate", ["simulate_family_a", "simulate_family_c"])
def test_replicate_zero_simulators_match_v1(v1_ref, simulate):
    now = getattr(g, simulate)(MIXED, SMALL, 21, replicate=0)
    then = getattr(v1_ref.generators, simulate)(as_ref(v1_ref, MIXED),
                                                as_ref(v1_ref, SMALL), 21)
    assert_same_system(now, then)


@pytest.mark.parametrize("kind", ["common_driver", "reflex_arc",
                                  "random_label_self_other", "scrambled_feedback",
                                  "parity_grid", "hypersynchrony"])
def test_replicate_zero_reproduces_every_v1_adversary(v1_ref, kind):
    now = g.make_system(f"adversarial_{kind}", None, SMALL, 14, replicate=0)
    with frozen(v1_ref):
        then = v1_ref.adversarial.make_adversarial(kind, as_ref(v1_ref, SMALL), 14)
    assert_same_system(now, then)


def test_replicate_zero_reproduces_the_whole_brain_path(v1_ref):
    for gen, sec in (("whole_brain", 1.0), ("whole_brain_eeg", 1.0),
                     ("whole_brain_bold", 30.0)):
        kw = {"whole_brain_config": {"duration_sec": sec, "transient_sec": 0.5}}
        now = g.make_system(gen, None, None, 3, replicate=0, **dict(kw))
        with frozen(v1_ref):
            then = v1_ref.generators.make_system(gen, None, None, 3, **dict(kw))
        assert_same_system(now, then)


@pytest.mark.parametrize("family", ["A", "C"])
def test_replicate_zero_reproduces_every_v1_witness(v1_ref, family):
    cat_v1 = witnesses.load_witnesses()
    cat_v2 = A2.load_catalogue_v2()
    v2_ids = set(A2.system_ids(cat_v2, kind="witness", family=family))
    for w in cat_v1["witnesses"]:
        if family not in w["families"]:
            continue
        with frozen(v1_ref):
            then = v1_ref.witnesses.build_witness_system(w, 15, family,
                                                         as_ref(v1_ref, SMALL))
        now = witnesses.build_witness_system(w, 15, family, SMALL)
        assert_same_system(now, then)
        if w["id"] in v2_ids:
            built = A2.build_system(w["id"], 15, family, config=SMALL, catalogue=cat_v2)
            assert_same_system(built, then)


def test_replicate_zero_with_the_default_configuration_is_v1(v1_ref):
    """A full-length family-A witness at the default configuration."""
    cat_v1 = witnesses.load_witnesses()
    w = witnesses.get_witness("W_NAS_no_workspace", cat_v1)
    with frozen(v1_ref):
        then = v1_ref.witnesses.build_witness_system(w, 2, "A")
    assert_same_system(A2.build_system("W_NAS_no_workspace", 2, "A", replicate=0), then)


# --------------------------------------------------------------------------
# twins
# --------------------------------------------------------------------------
@pytest.mark.parametrize("family,simulate", [("A", g.simulate_family_a),
                                             ("C", g.simulate_family_c)])
@pytest.mark.parametrize("knobs", [None, g.Knobs(g_b=0.0), g.Knobs(c_int=0.0, K=1)])
def test_twins_keep_the_structure_and_change_the_schedule(family, simulate, knobs):
    base = simulate(knobs, SMALL, 31)
    twins = [simulate(knobs, SMALL, 31, replicate=r) for r in (1, 2)]
    h0 = MV.structural_hash(base), MV.schedule_hash(base)
    sched = {h0[1]}
    for r, tw in zip((1, 2), twins):
        assert MV.structural_hash(tw) == h0[0]
        assert MV.schedule_hash(tw) not in sched
        sched.add(MV.schedule_hash(tw))
        assert np.array_equal(tw.oracle["unit_adjacency"],
                              base.oracle["unit_adjacency"])
        assert tw.meta["coupling_scale"] == base.meta["coupling_scale"]
        if family == "C":
            assert np.array_equal(tw.oracle["oscillator_frequency_hz"],
                                  base.oracle["oscillator_frequency_hz"])
        assert not np.array_equal(tw.ts, base.ts)
        assert not tw.events["onset"].equals(base.events["onset"])
        assert tw.meta["replicate"] == r
        assert tw.meta["generator_version"] == GENERATOR_VERSION_V2
        for key in ("replicate",):
            assert key not in base.meta
        assert base.meta["generator_version"] == g.GENERATOR_VERSION
        # the twin's own determinism
        assert np.array_equal(simulate(knobs, SMALL, 31, replicate=r).ts, tw.ts)


def test_twin_streams_are_the_documented_children():
    """Schedule and noise of a twin come from SeedSequence([seed, r]) at the
    positions of the v1 split; structure from SeedSequence(seed)."""
    seed, r = 41, 3
    tw = g.simulate_family_a(None, SMALL, seed, replicate=r)
    kids = S.twin_seed_sequence(seed, r).spawn(8)
    task_rng = np.random.default_rng(kids[g.TWIN_STREAM_INDEX["task"]])
    sched = g._task_schedule(SMALL, task_rng)
    onsets = tw.events.loc[tw.events.trial_type == "goal_cue", "onset"].to_numpy()
    assert np.allclose(onsets, np.round(sched["goal_idx"] * SMALL.dt, 6))
    assert np.array_equal(tw.oracle["good_arm"], sched["good_arm"])
    assert g.TWIN_STREAM_INDEX == {"task": 1, "noise": 2, "rest_task": 4,
                                   "rest_noise": 5}
    v1 = g._streams(seed, 8)
    twin = g._modular_streams(seed, r)
    for i in range(8):
        same = v1[i].bit_generator.state == twin[i].bit_generator.state
        assert same == (i not in g.TWIN_STREAM_INDEX.values()), i
    assert [s.bit_generator.state for s in g._modular_streams(seed, 0)] == [
        s.bit_generator.state for s in v1]


def test_twin_rest_runs_are_new_sessions_of_the_same_oscillators():
    cfg = SMALL.replace(rest_sec=8.0)
    base = g.simulate_family_c(None, cfg, 43)
    tw = g.simulate_family_c(None, cfg, 43, replicate=1)
    assert base.rest_ts.shape == tw.rest_ts.shape
    assert not np.array_equal(base.rest_ts, tw.rest_ts)
    assert np.array_equal(base.oracle["oscillator_frequency_hz"],
                          tw.oracle["oscillator_frequency_hz"])


def test_twins_are_paired_across_knobs():
    """Common random numbers hold within a replicate: knob contrasts of one
    twin share its schedule and noise."""
    a = g.simulate_family_a(None, SMALL, 44, replicate=2)
    b = g.simulate_family_a(g.Knobs(g_b=0.0), SMALL, 44, replicate=2)
    assert MV.schedule_hash(a) == MV.schedule_hash(b)
    assert MV.structural_hash(a) != MV.structural_hash(b)


@pytest.mark.parametrize("bad", [-1, 31, 1.0, True, "1", None])
def test_replicate_values_are_validated(bad):
    with pytest.raises((ValueError, S.SeedPolicyError)):
        g.simulate_family_a(None, SMALL, 0, replicate=bad)


def test_replicate_is_defined_for_the_agent_families_only():
    for gen in ("patchwork", "null_ar1", "hypersynchronous", "adversarial_reflex_arc",
                "whole_brain", "whole_brain_eeg", "whole_brain_bold"):
        with pytest.raises(ValueError, match="family_a and family_c only"):
            g.make_system(gen, None, SMALL, 0, replicate=1)
    c30 = g.make_system("family_c", None, SMALL, 0, replicate=30)
    assert c30.meta["replicate"] == 30
    assert np.int64(2) == g.make_system("family_a", None, SMALL, 0,
                                        replicate=np.int64(2)).meta["replicate"]


# --------------------------------------------------------------------------
# v2 options and configuration fields default to v1
# --------------------------------------------------------------------------
def test_every_new_option_defaults_to_the_v1_value(v1_ref):
    assert g.V2_OPTION_DEFAULTS == {"replicate": 0}
    for fn in (g.simulate_family_a, g.simulate_family_c, g.make_system,
               g._simulate_modular):
        params = inspect.signature(fn).parameters
        for name, default in g.V2_OPTION_DEFAULTS.items():
            assert params[name].default == default, (fn.__name__, name)
    # no v1 parameter moved or changed its default
    for name in ("simulate_family_a", "simulate_family_c", "make_system"):
        old = inspect.signature(getattr(v1_ref.generators, name)).parameters
        new = inspect.signature(getattr(g, name)).parameters
        old_list = [(p.name, p.kind, p.default) for p in old.values()
                    if p.kind != p.VAR_KEYWORD]
        new_list = [(p.name, p.kind, p.default) for p in new.values()
                    if p.kind != p.VAR_KEYWORD and p.name not in g.V2_OPTION_DEFAULTS]
        assert new_list == old_list, name


def test_every_configuration_field_keeps_its_v1_default(v1_ref):
    """Tier A adds no configuration field: the agent configuration, the
    knobs and their serialisation are the v1 ones (a later field must join
    ``V2_OPTION_DEFAULTS`` with its v1 value and keep this serialisation)."""
    assert g.AgentConfig().to_dict() == v1_ref.generators.AgentConfig().to_dict()
    assert g.Knobs().to_dict() == v1_ref.generators.Knobs().to_dict()
    assert g.OFF_KNOBS.to_dict() == v1_ref.generators.OFF_KNOBS.to_dict()
    new_fields = set(g.AgentConfig.__dataclass_fields__) - set(
        v1_ref.generators.AgentConfig.__dataclass_fields__)
    assert new_fields <= set(g.V2_OPTION_DEFAULTS)
    for f in new_fields:
        assert getattr(g.AgentConfig(), f) == g.V2_OPTION_DEFAULTS[f]
    assert g.GENERATOR_VERSION == "mpc-bench-generators/1.1.0"
    assert GENERATOR_VERSION_V2 == "mpc-bench-generators/2.0.0"


def test_v2_generator_modules_keep_the_import_rules():
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(SRC)!r})\n"
        "from impact_pipeline.bench import adversarial_v2, manipulation_v2\n"
        "cat = adversarial_v2.load_catalogue_v2()\n"
        "adversarial_v2.build_system('ADV_NAS_staggered_driver', 0)\n"
        "adversarial_v2.build_system('PC_nominal', 0, replicate=1)\n"
        "banned = ('impact_pipeline.mpc_metrics', 'impact_pipeline.synergy_ci')\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('mne', 'sklearn')\n"
        "             or m in banned)\n"
        "print('BAD', bad)\n"
        "sys.exit(1 if bad else 0)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
