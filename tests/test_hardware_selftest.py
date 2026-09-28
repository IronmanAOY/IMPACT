"""
Accelerator eigh fallback/parity (spec D9) and the hardware self-test.

CuPy documents cupy.linalg.eigh as unsupported on ROCm and HPE warns about
wrong rocBLAS results on ROCm 6.4.0/6.4.1, so the device eigh path must be
checked numerically and fall back to NumPy. No GPU is available here: a
NumPy-backed fake ``cupy`` module that reports a HIP runtime stands in.
"""

import sys
import types

import numpy as np
import pytest

from impact_pipeline import hardware_backend as hb
from impact_pipeline import hardware_selftest


def _fake_cupy(eigh=None, add_at_broken=False):
    cp = types.ModuleType("cupy")
    cp.__version__ = "13.6.0+fake-rocm"
    cp.__getattr__ = lambda name: getattr(np, name)  # PEP 562 fallback to NumPy
    cp.asnumpy = np.asarray
    cp.linalg = types.SimpleNamespace(
        eigh=eigh or np.linalg.eigh,
        svd=np.linalg.svd,
        pinv=np.linalg.pinv,
        solve=np.linalg.solve,
        norm=np.linalg.norm,
    )
    if add_at_broken:
        cp.add = types.SimpleNamespace(at=lambda arr, idx, val: None)
    else:
        cp.add = np.add
    runtime = types.SimpleNamespace(
        is_hip=lambda: True,
        getDeviceCount=lambda: 4,
        getDeviceProperties=lambda i: {
            "name": b"AMD Instinct MI300A",
            "gcnArchName": b"gfx942:sramecc+:xnack-",
        },
        runtimeGetVersion=lambda: 60401,
        driverGetVersion=lambda: 60401,
    )
    cp.cuda = types.SimpleNamespace(runtime=runtime)
    return cp


@pytest.fixture
def fake_rocm(monkeypatch):
    def install(**kwargs):
        monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(**kwargs))
        monkeypatch.setattr(hb, "_EIGH_DEVICE_STATUS", {})
        return hb.resolve_hardware_backend("hunter-apu")

    monkeypatch.setattr(hb, "_THREAD_LIMITS_APPLIED", True)
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.setenv(var, "1")
    monkeypatch.delenv(hb.EIGH_BACKEND_ENV, raising=False)
    return install


def _spd(n=12, seed=3):
    a = np.random.default_rng(seed).standard_normal((n, n))
    return a @ a.T + n * np.eye(n)


def test_device_eigh_error_falls_back_to_cpu(fake_rocm):
    def broken_eigh(_m):
        raise NotImplementedError("eigh is not supported on ROCm")

    backend = fake_rocm(eigh=broken_eigh)
    assert backend.runtime == "rocm" and backend.accelerator
    spd = _spd()
    out = hb.accelerated_psd_invsqrt(spd, backend=backend)
    np.testing.assert_allclose(
        out, hb.accelerated_psd_invsqrt(spd, backend="cpu"), rtol=1e-12, atol=1e-12
    )
    status = hb.device_eigh_status(backend)
    assert status["use_device"] is False
    assert "device_eigh_error" in status["reason"]


def test_device_eigh_with_wrong_numbers_is_not_used(fake_rocm):
    def wrong_eigh(m):
        vals, vecs = np.linalg.eigh(np.asarray(m))
        return vals * 1.01, vecs  # plausible-looking but wrong eigenvalues

    backend = fake_rocm(eigh=wrong_eigh)
    parity = hb.check_eigh_parity(backend)
    assert parity["checked"] and not parity["device_ok"]
    assert parity["eigenvalue_rel_err"] > 1e-3
    spd = _spd()
    # the device path would return a wrong matrix; the parity guard keeps the CPU result
    np.testing.assert_allclose(
        hb.accelerated_psd_invsqrt(spd, backend=backend),
        hb.accelerated_psd_invsqrt(spd, backend="cpu"),
        rtol=1e-12,
        atol=1e-12,
    )


def test_device_eigh_used_after_parity_passes(fake_rocm, monkeypatch):
    calls = {"n": 0}

    def good_eigh(m):
        calls["n"] += 1
        return np.linalg.eigh(np.asarray(m))

    backend = fake_rocm(eigh=good_eigh)
    spd = _spd()
    out = hb.accelerated_psd_invsqrt(spd, backend=backend)
    assert hb.device_eigh_status(backend)["use_device"] is True
    assert calls["n"] == 2  # one parity check + one real call
    np.testing.assert_allclose(
        out, hb.accelerated_psd_invsqrt(spd, backend="cpu"), rtol=1e-10
    )

    monkeypatch.setattr(hb, "_EIGH_DEVICE_STATUS", {})
    monkeypatch.setenv(hb.EIGH_BACKEND_ENV, "cpu")
    hb.accelerated_psd_invsqrt(spd, backend=backend)
    assert calls["n"] == 2  # forced CPU: device eigh not called


def test_selftest_compares_kernels_and_reports_device(fake_rocm, capsys, tmp_path):
    fake_rocm()
    report = hardware_selftest.run_selftest(target="hunter-apu", size=32)
    assert report["all_ok"] is True
    assert report["device_info"]["devices"][0]["name"] == "AMD Instinct MI300A"
    assert {c["name"] for c in report["cases"]} >= {
        "matmul",
        "eigh",
        "ufunc_add_at",
        "iim_tpm_kernels",
    }
    rc = hardware_selftest.main(
        ["--target", "hunter-apu", "--size", "32", "--json", str(tmp_path / "st.json")]
    )
    assert rc == 0
    assert "MI300A" in capsys.readouterr().out
    assert (tmp_path / "st.json").exists()


def test_selftest_detects_broken_scatter_add(fake_rocm):
    fake_rocm(add_at_broken=True)
    report = hardware_selftest.run_selftest(target="hunter-apu", size=32)
    failed = {c["name"] for c in report["cases"] if not c["ok"]}
    assert "ufunc_add_at" in failed and "iim_tpm_kernels" in failed
    assert report["all_ok"] is False
    assert hardware_selftest.main(["--target", "hunter-apu", "--size", "32"]) == 1


def test_selftest_reports_missing_accelerator(monkeypatch):
    import importlib

    real_import = importlib.import_module

    def no_cupy(name, *args, **kwargs):
        if name == "cupy":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", no_cupy)
    assert hardware_selftest.main(["--target", "hunter-apu"]) == 2
    assert hardware_selftest.main(["--target", "cpu", "--size", "16"]) == 0


def test_selftest_runs_the_iim_psi_kernel_on_the_device(fake_rocm, capsys):
    import impact_pipeline

    fake_rocm()
    report = hardware_selftest.run_selftest(target="hunter-apu", size=32)
    case = next(c for c in report["cases"] if c["name"] == "iim_psi_xp_parity")
    assert case["ok"] is True and case["rel_err"] < 1e-10
    assert case["psi_full_xp"] == pytest.approx(case["psi_full_numba"], rel=1e-10)
    # the shards of this target use the array-module (device) Psi kernel
    assert case["psi_kernel_for_target"] == "xp"
    assert report["iim_psi_kernel"] == "xp"
    version = impact_pipeline.__version__.split("+", 1)[0]
    assert report["code_version"].startswith(version + "+")
    assert hardware_selftest.main(["--target", "hunter-apu", "--size", "32"]) == 0
    out = capsys.readouterr().out
    assert "iim_psi_xp_parity" in out and "IIM Psi kernel for this target: xp" in out


def test_selftest_detects_wrong_device_psi_numerics(fake_rocm):
    fake_rocm()
    cp = sys.modules["cupy"]
    cp.log = lambda x: np.log(x) * (1.0 + 1e-6)  # subtly wrong device math
    report = hardware_selftest.run_selftest(target="hunter-apu", size=32)
    failed = {c["name"] for c in report["cases"] if not c["ok"]}
    assert failed == {"iim_psi_xp_parity"}
    assert report["all_ok"] is False
