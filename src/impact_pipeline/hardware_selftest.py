"""
Numerical self-test of the accelerator backend against NumPy.

Run on the target node (e.g. inside an interactive or ``-q test`` PBS job on a
Hunter mi300a node) before submitting a campaign::

    PYTHONPATH=src python3 -m impact_pipeline.hardware_selftest --target hunter-apu

(``--json PATH`` additionally writes the report as JSON.)

The test prints device information and compares CuPy results with NumPy for
the operations the pipeline relies on: matmul, eigh (used by
``accelerated_psd_invsqrt``), ``ufunc.at`` scatter-add and weighted
``bincount`` (IIM transition counting / cut TPMs), SVD, pinv, and the IIM TPM
kernels themselves. The exit code is non-zero when any comparison fails, or
when an accelerator was requested but could not be resolved.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from impact_pipeline.hardware_backend import (
    HardwareBackendError,
    backend_summary,
    check_eigh_parity,
    configure_process_for_hardware,
    device_eigh_status,
    device_info,
    get_array_module,
    to_numpy,
)

DEFAULT_RTOL = 1e-8


def _rel_err(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return float("inf")
    scale = float(np.max(np.abs(b))) if b.size else 0.0
    diff = float(np.max(np.abs(a - b))) if a.size else 0.0
    return diff / scale if scale > 0 else diff


def _case(name: str, fn: Callable[[], dict], rtol: float) -> dict:
    t0 = time.perf_counter()
    try:
        rec = dict(fn())
    except Exception as exc:
        rec = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    rec.setdefault("rel_err", None)
    if "ok" not in rec:
        err = rec.get("rel_err")
        rec["ok"] = bool(
            err is not None and np.isfinite(err) and float(err) <= float(rtol)
        )
    rec["name"] = name
    rec["seconds"] = float(time.perf_counter() - t0)
    return rec


def run_selftest(
    target: str = "auto", size: int = 256, seed: int = 0, rtol: float = DEFAULT_RTOL
) -> dict[str, Any]:
    backend = configure_process_for_hardware(target)
    xp = get_array_module(backend)
    rng = np.random.default_rng(int(seed))
    n = int(size)

    a = rng.standard_normal((n, n))
    b = rng.standard_normal((n, n))

    def _matmul():
        out = to_numpy(xp.asarray(a) @ xp.asarray(b))
        return {"rel_err": _rel_err(out, a @ b)}

    def _eigh():
        parity = check_eigh_parity(backend, n=min(n, 128), seed=int(seed), rtol=rtol)
        if not backend.accelerator:
            return {"rel_err": 0.0, "detail": parity}
        return {
            "ok": bool(parity.get("device_ok")),
            "rel_err": (
                max(
                    float(parity.get("eigenvalue_rel_err", np.inf)),
                    float(parity.get("reconstruction_rel_err", np.inf)),
                    float(parity.get("orthogonality_err", np.inf)),
                )
                if parity.get("checked") and "eigenvalue_rel_err" in parity
                else None
            ),
            "detail": parity,
        }

    def _add_at():
        n_states = 64
        i = rng.integers(0, n_states, size=20_000)
        j = rng.integers(0, n_states, size=20_000)
        ref = np.zeros((n_states, n_states))
        np.add.at(ref, (i, j), 1.0)
        dev = xp.zeros((n_states, n_states), dtype=xp.float64)
        xp.add.at(dev, (xp.asarray(i), xp.asarray(j)), 1.0)
        return {"rel_err": _rel_err(to_numpy(dev), ref)}

    def _bincount():
        keys = rng.integers(0, 97, size=5_000)
        w = rng.random(5_000)
        ref = np.bincount(keys, weights=w, minlength=97)
        dev = xp.bincount(xp.asarray(keys), weights=xp.asarray(w), minlength=97)
        return {"rel_err": _rel_err(to_numpy(dev), ref)}

    def _svd():
        m = rng.standard_normal((n, n // 2))
        ref = np.linalg.svd(m, compute_uv=False)
        dev = xp.linalg.svd(xp.asarray(m), full_matrices=False, compute_uv=False)
        return {"rel_err": _rel_err(to_numpy(dev), ref)}

    def _pinv():
        m = rng.standard_normal((n, n // 4))
        return {
            "rel_err": _rel_err(
                to_numpy(xp.linalg.pinv(xp.asarray(m))), np.linalg.pinv(m)
            )
        }

    def _psd_invsqrt():
        from impact_pipeline.hardware_backend import accelerated_psd_invsqrt

        m = rng.standard_normal((64, 64))
        spd = m @ m.T + 64.0 * np.eye(64)
        ref = accelerated_psd_invsqrt(spd, backend="cpu")
        out = accelerated_psd_invsqrt(spd, backend=backend)
        return {"rel_err": _rel_err(out, ref), "eigh_path": device_eigh_status(backend)}

    def _iim_tpm():
        from impact_pipeline import mpc_metrics as mm

        disc = rng.integers(0, 2, size=(5, 400))
        ref = mm._iim_build_states_and_tpm(disc, 2, 1, 1e-3, hardware_backend="cpu")
        out = mm._iim_build_states_and_tpm(disc, 2, 1, 1e-3, hardware_backend=backend)
        errs = [_rel_err(o, r) for o, r in zip(out, ref)]
        cut_ref = mm._iim_build_cut_tpm(
            ref[1], ref[2], 2, (0, 1), (2, 3, 4), hardware_backend="cpu"
        )
        cut_out = mm._iim_build_cut_tpm(
            out[1], out[2], 2, (0, 1), (2, 3, 4), hardware_backend=backend
        )
        errs.append(_rel_err(cut_out, cut_ref))
        return {"rel_err": float(max(errs))}

    cases = [
        _case("matmul", _matmul, rtol),
        _case("eigh", _eigh, rtol),
        _case("ufunc_add_at", _add_at, rtol),
        _case("bincount_weighted", _bincount, rtol),
        _case("svd_values", _svd, rtol),
        _case("pinv", _pinv, rtol),
        _case("psd_invsqrt", _psd_invsqrt, rtol),
        _case("iim_tpm_kernels", _iim_tpm, rtol),
    ]
    # eigh may legitimately fail on ROCm: the pipeline then uses the CPU
    # fallback, so it does not fail the self-test on its own.
    required = [c for c in cases if c["name"] != "eigh"]
    return {
        "requested_target": str(target),
        "backend_summary": backend_summary(backend),
        "device_info": device_info(backend),
        "rtol": float(rtol),
        "cases": cases,
        "eigh_uses_device": bool(device_eigh_status(backend).get("use_device", False)),
        "all_ok": bool(all(c["ok"] for c in required)),
        "note": (
            "CPU backend: comparisons are NumPy against NumPy."
            if not backend.accelerator
            else "Accelerator results compared against NumPy on the host."
        ),
    }


def _print_report(report: dict) -> None:
    print(f"IMPaCT hardware self-test: {report['backend_summary']}")
    dev = report.get("device_info", {})
    for rec in dev.get("devices", []) or []:
        arch = rec.get("gcnArchName", "")
        print(f"  device[{rec.get('index')}]: {rec.get('name')} {arch}".rstrip())
    if dev.get("visible_device_env"):
        print(f"  visible-device env: {dev['visible_device_env']}")
    for case in report["cases"]:
        err = case.get("rel_err")
        err_txt = "n/a" if err is None else f"{float(err):.2e}"
        status = "OK" if case.get("ok") else "FAIL"
        extra = f" ({case['error']})" if case.get("error") else ""
        timing = f"t={case['seconds']:.3f}s"
        print(f"  {case['name']:<18} {status:<4} rel_err={err_txt} {timing}{extra}")
    print(f"  eigh on device: {report['eigh_uses_device']} (CPU fallback otherwise)")
    print(f"  overall: {'OK' if report['all_ok'] else 'FAIL'}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare accelerator (CuPy) kernels with NumPy."
    )
    parser.add_argument(
        "--target", default="auto", help="cpu, auto, gpu or hunter-apu (default: auto)"
    )
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="Optional path for a JSON report.",
    )
    args = parser.parse_args(argv)
    try:
        report = run_selftest(
            target=args.target, size=args.size, seed=args.seed, rtol=args.rtol
        )
    except HardwareBackendError as exc:
        print(f"Hardware target unavailable: {exc}", file=sys.stderr)
        return 2
    _print_report(report)
    if args.json_path:
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
