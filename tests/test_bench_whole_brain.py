"""MPC-Bench external generator: the Hopf whole-brain model on the shipped
connectome (provenance, symmetrisation, normalisation, lesions, G sweep).
The forward models are tested in test_bench_forward.py."""

import hashlib
import math

import numpy as np
import pytest

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


def test_generators_do_not_import_mpc_metrics():
    import subprocess
    import sys
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src"
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(src)!r})\n"
        "from impact_pipeline.bench import whole_brain, forward, adversarial, "
        "manipulation, patchwork, generators\n"
        "cfg = generators.AgentConfig(n_trials=4, n_reafference_pairs=2)\n"
        "for kind in adversarial.ADVERSARIAL_KINDS:\n"
        "    adversarial.make_adversarial(kind, cfg, 0)\n"
        "adversarial.staggered_driver_system('hierarchical', 0, cfg)\n"
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
