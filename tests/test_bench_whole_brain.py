"""MPC-Bench external generator: the Hopf whole-brain model on the shipped
connectome (provenance, symmetrisation, normalisation, lesions, G sweep) and
the EEG-like / BOLD-like forward models (known answers)."""

import hashlib
import math

import numpy as np
import pytest

from impact_pipeline.bench import forward as fw
from impact_pipeline.bench import whole_brain as wb


@pytest.fixture(scope="module")
def conn():
    return wb.load_connectome()


@pytest.fixture(scope="module")
def source(conn):
    cfg = wb.WholeBrainConfig(duration_sec=20.0)
    return wb.simulate_whole_brain(cfg, seed=1, connectome=conn)


def test_connectome_provenance_and_structure(conn):
    path = wb.REPO_ROOT / wb.CONNECTOME_RELPATH
    assert hashlib.sha256(path.read_bytes()).hexdigest() == wb.CONNECTOME_SHA256
    prov = conn.provenance
    assert prov["n_rows"] == 1000 and prov["n_self_loops_dropped"] == 11
    assert prov["edge_confidence_min"] >= 209
    assert conn.n == 76 and prov["n_edges"] == 265
    W = conn.W
    assert np.allclose(W, W.T) and np.all(np.diag(W) == 0) and np.all(W >= 0)
    assert W.sum(axis=1).mean() == pytest.approx(1.0)
    from scipy.sparse.csgraph import connected_components

    assert connected_components(W > 0, directed=False)[0] == 1
    assert set(conn.hemisphere) == {"L", "R", "M"}
    assert "other" not in set(conn.lobe)
    macro = conn.macro_nodes()
    assert set(macro) == {"L_ant", "L_post", "R_ant", "R_post"}
    flat = [i for v in macro.values() for i in v]
    assert len(flat) == len(set(flat))
    # log1p weights of summed fibre counts before normalisation.
    i, j = np.unravel_index(np.argmax(conn.raw_weight), conn.raw_weight.shape)
    ratio = conn.W[i, j] / math.log1p(conn.raw_weight[i, j])
    k, m = np.transpose(np.nonzero(conn.raw_weight))[0]
    assert conn.W[k, m] / math.log1p(conn.raw_weight[k, m]) == pytest.approx(ratio)
    fine = wb.load_connectome(grain="fine")
    assert fine.n == 467 and fine.provenance["grain"] == "fine"
    with pytest.raises(ValueError):
        wb.load_connectome(grain="voxel")
    with pytest.raises(FileNotFoundError):
        wb.load_connectome(path="/nonexistent/connectome.csv")


def test_lesions_remove_exactly_the_declared_edges(conn):
    edges = conn.W > 0
    hubs = conn.hubs(4)
    m = wb.lesion_mask(conn, "hub", n_hubs=4)
    assert np.array_equal(m, m.T) and np.all(edges[m])
    assert np.all(m[hubs][:, edges.any(axis=0)] == edges[hubs][:, edges.any(axis=0)])
    Wl = wb.apply_lesion(conn, m)
    assert np.all(Wl[hubs] == 0) and np.all(Wl[:, hubs] == 0)
    hemi = np.asarray(conn.hemisphere)
    inter = wb.lesion_mask(conn, "interhemispheric")
    assert np.count_nonzero(np.triu(inter, 1)) == 9
    Wi = wb.apply_lesion(conn, inter)
    lr = np.isin(hemi, ("L", "R"))
    cross = (hemi[:, None] != hemi[None, :]) & lr[:, None] & lr[None, :]
    assert not np.any(Wi[cross])
    homo = wb.lesion_mask(conn, "homotopic")
    assert np.count_nonzero(np.triu(homo, 1)) == 2 and np.all(inter[homo])
    lr_m = wb.lesion_mask(conn, "long_range")
    lob = np.asarray(conn.lobe)
    assert np.all((lob[:, None] != lob[None, :]) | cross | ~lr_m)
    n_e = int(np.count_nonzero(np.triu(lr_m, 1)))
    rnd = wb.lesion_mask(conn, "random", n_edges=n_e, seed=3)
    assert np.count_nonzero(np.triu(rnd, 1)) == n_e and np.all(edges[rnd])
    assert not wb.lesion_mask(conn, "none").any()
    with pytest.raises(ValueError):
        wb.lesion_mask(conn, "random")
    names = [m_["name"] for m_ in wb.manipulations(g_levels=(0.0, 1.0))]
    assert "lesion_random_matched_hub" in names and "G0" in names


def test_simulation_meta_oracle_and_determinism(conn, source):
    s = source
    assert s.ts.shape == (76, 5000) and s.dt == pytest.approx(0.004)
    assert np.all(np.isfinite(s.ts)) and len(s.events) == 0
    assert s.meta["workspace_nodes"] == conn.hubs(6)
    assert s.meta["connectome"]["sha256"] == wb.CONNECTOME_SHA256
    assert s.meta["substrate"] == "stuart_landau" and s.oracle["intended_bits"] is None
    assert s.oracle["order_parameter"].shape == (5000,)
    from impact_pipeline.bench.generators import summarise_oracle

    summ = summarise_oracle(s)
    assert summ["intended_bits"] == [] and 0 < summ["mean_order"] < 1
    again = wb.simulate_whole_brain(
        wb.WholeBrainConfig(duration_sec=20.0), seed=1, connectome=conn
    )
    assert np.array_equal(again.ts, s.ts)
    # The dominant rhythm is the declared alpha carrier.
    f = np.fft.rfftfreq(s.n_time, s.dt)
    spec = np.abs(np.fft.rfft(s.ts, axis=1)) ** 2
    assert 8.5 < f[np.argmax(spec.mean(axis=0))] < 11.5
    with pytest.raises(ValueError):
        wb.WholeBrainConfig(fs_out=300.0)
    with pytest.raises(ValueError):
        wb.WholeBrainConfig(lesion="corpus")


def test_global_coupling_increases_synchrony(conn):
    """Known qualitative answer: the Kuramoto order parameter grows with G."""
    orders = []
    for G in (0.0, 1.0, 4.0):
        cfg = wb.WholeBrainConfig(G=G, duration_sec=8.0)
        orders.append(
            np.mean(
                [
                    wb.simulate_whole_brain(cfg, s, connectome=conn).oracle[
                        "mean_order"
                    ]
                    for s in (0, 1)
                ]
            )
        )
    assert orders[0] < orders[1] < orders[2]
    assert orders[0] < 0.2 and orders[2] > 1.5 * orders[0]
    assert wb.g_sweep_levels(5, 4.0) == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_eeg_forward_model(source):
    e = fw.eeg_forward(source, n_sensors=64, seed=2)
    assert (
        e.ts.shape == (64, source.n_time) and e.meta["substrate"] == "eeg_like_forward"
    )
    # Average reference: sensors sum to zero at every sample.
    assert np.max(np.abs(e.ts.mean(axis=0))) < 1e-10
    # The lead field is smooth (volume conduction) and fixed across seeds.
    L = e.oracle["lead_field"]
    sens = e.oracle["sensor_positions"]
    d = np.linalg.norm(sens[:, None] - sens[None, :], axis=-1)
    rc = np.corrcoef(np.abs(L))
    near = rc[(d > 0) & (d < 0.3)].mean()
    far = rc[d > 1.5].mean()
    assert near > far + 0.3
    e2 = fw.eeg_forward(source, n_sensors=64, seed=9)
    assert np.array_equal(e2.oracle["lead_field"], L)
    assert not np.array_equal(e2.ts, e.ts)
    # Band-pass 1-40 Hz: little power outside the band, alpha kept.
    f = np.fft.rfftfreq(e.n_time, e.dt)
    P = (np.abs(np.fft.rfft(e.ts, axis=1)) ** 2).mean(axis=0)
    alpha = P[(f > 8) & (f < 12)].mean()
    assert P[(f > 60) & (f < 100)].mean() < 1e-3 * alpha
    assert P[f < 0.3].mean() < 1e-2 * alpha
    # Volume conduction makes independent-looking sources correlate at sensors.
    cs = np.abs(np.corrcoef(source.ts)[np.triu_indices(76, 1)]).mean()
    ce = np.abs(np.corrcoef(e.ts)[np.triu_indices(64, 1)]).mean()
    assert ce > cs
    assert set(e.meta["iim_macro_nodes"]) == {"L_ant", "R_ant", "L_post", "R_post"}
    assert e.meta["workspace_nodes"] and "lobe" not in e.meta
    with pytest.raises(ValueError):
        fw.eeg_forward(source, band=(1.0, 200.0))


def test_canonical_hrf_known_answers():
    h = fw.canonical_hrf(0.01)
    t = np.arange(h.size) * 0.01
    assert h.sum() == pytest.approx(1.0)
    assert t[np.argmax(h)] == pytest.approx(5.0, abs=0.02)  # (a1 - 1) * b
    assert 14.0 < t[np.argmin(h)] < 16.5 and h.min() < 0  # undershoot
    assert t[-1] == pytest.approx(32.0)


def test_bold_forward_impulse_response_and_noise(source):
    # Impulse through the forward model (no noise): the HRF sampled at TR.
    imp = source.oracle["envelope"].astype(float) * 0.0
    imp[:, 250] = 1.0  # t = 1 s
    s = type(source)(
        ts=imp, events=source.events, meta=dict(source.meta), oracle={"envelope": imp}
    )
    b = fw.bold_forward(s, tr=2.0, noise_sd=0.0)
    h = fw.canonical_hrf(source.dt)
    clean = b.oracle["bold_clean"][0]
    x = imp[0] - imp[0].mean()
    ref = np.convolve(x, h)[: x.size][::500]
    ref = (ref - ref.mean()) / ref.std()
    assert np.allclose(clean, ref, atol=1e-8)
    # Impulse at 1 s, HRF peak 5 s later: the largest TR sample is t = 6 s.
    assert int(np.argmax(clean)) == 3
    assert b.dt == 2.0 and b.meta["RepetitionTime"] == 2.0
    assert b.meta["substrate"] == "bold_like_forward"
    # AR(1) physiological noise with the declared coefficient.
    long = type(source)(
        ts=np.zeros((4, 250 * 2000)),
        events=source.events,
        meta=dict(source.meta),
        oracle={"envelope": np.random.default_rng(0).random((4, 250 * 2000))},
    )
    nb = fw.bold_forward(long, tr=2.0, ar_coef=0.6, noise_sd=1.0, seed=3)
    e = nb.oracle["physiological_noise"]
    ac = np.mean([np.corrcoef(r[1:], r[:-1])[0, 1] for r in e])
    assert ac == pytest.approx(0.6, abs=0.05)
    with pytest.raises(ValueError):
        fw.bold_forward(source, tr=0.003)
    with pytest.raises(ValueError):
        fw.bold_forward(source, ar_coef=1.0)


def test_generators_do_not_import_mpc_metrics():
    import subprocess
    import sys
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src"
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(src)!r})\n"
        "from impact_pipeline.bench import whole_brain, forward, adversarial, "
        "manipulation, audit, compat, patchwork, generators\n"
        "cfg = generators.AgentConfig(n_trials=4, n_reafference_pairs=2)\n"
        "for kind in adversarial.ADVERSARIAL_KINDS:\n"
        "    adversarial.make_adversarial(kind, cfg, 0, "
        "**({'n': 4} if kind == 'parity_grid' else {}))\n"
        "s = whole_brain.simulate_whole_brain(whole_brain.WholeBrainConfig("
        "duration_sec=2.0, transient_sec=0.1), 0)\n"
        "forward.eeg_forward(s); forward.bold_forward(s)\n"
        "patchwork.simulate_patchwork(None, cfg, 0, inter_module_coupling=0.5)\n"
        "manipulation.oracle_signatures(generators.simulate_family_c(None, cfg, 0))\n"
        "bad = sorted(m for m in sys.modules if m.startswith('impact_pipeline.') and "
        "m.split('.')[1] in ('mpc_metrics', 'synergy_ci', 'hardware_backend'))\n"
        "print('BAD', bad)\n"
        "sys.exit(1 if bad else 0)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
