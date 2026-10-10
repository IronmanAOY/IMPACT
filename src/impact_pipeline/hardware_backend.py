from __future__ import annotations

import importlib
import logging
import os
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Per-process record of whether the accelerator eigh path is trusted.
# CuPy documents cupy.linalg.eigh as "not yet supported" on ROCm (its HIP
# build routes to hipSOLVER syevj), and HPE warns about wrong rocBLAS results
# on ROCm 6.4.0/6.4.1, so device eigh is only used after a numerical parity
# check against NumPy and falls back to the CPU otherwise.
_EIGH_DEVICE_STATUS: dict[tuple, dict[str, Any]] = {}
EIGH_BACKEND_ENV = "IMPACT_EIGH_BACKEND"
EIGH_PARITY_RTOL = 1e-8
_THREAD_LIMITS_APPLIED = False


class HardwareBackendError(RuntimeError):
    """Raised when a requested hardware backend is unavailable or invalid."""


@dataclass(frozen=True)
class HardwareBackend:
    requested: str
    target: str
    accelerator: bool
    array_module: str
    runtime: str
    device_count: int
    device_name: str
    strict: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CPU_BACKEND = HardwareBackend(
    requested="cpu",
    target="cpu",
    accelerator=False,
    array_module="numpy",
    runtime="cpu",
    device_count=0,
    device_name="CPU",
    strict=False,
)


def normalize_hardware_target(target: str | None) -> str:
    key = str(target or "cpu").strip().lower().replace("_", "-")
    aliases = {
        "local": "cpu",
        "host": "cpu",
        "none": "cpu",
        "accelerator": "gpu",
        "rocm": "hunter-apu",
        "hip": "hunter-apu",
        "apu": "hunter-apu",
        "hunter": "hunter-apu",
        "hunter-gpu": "hunter-apu",
        "mi300a": "hunter-apu",
    }
    key = aliases.get(key, key)
    if key not in {"cpu", "auto", "gpu", "hunter-apu"}:
        raise HardwareBackendError(
            f"Unknown hardware target '{target}'. Expected one of: cpu, auto, gpu, hunter-apu."
        )
    return key


def _prop_text(value):
    """CuPy device properties hold strings as bytes; decode them."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _cupy_runtime_name(cp) -> str:
    is_hip = getattr(cp.cuda.runtime, "is_hip", None)
    try:
        if callable(is_hip):
            is_hip = bool(is_hip())
    except Exception:
        is_hip = None
    if is_hip is True:
        return "rocm"
    if is_hip is False:
        return "cuda"

    # Older/newer CuPy builds may not expose is_hip consistently. Version text
    # is a secondary signal, used only when runtime.is_hip is unavailable.
    version_text = " ".join(
        str(getattr(obj, "__version__", ""))
        for obj in (
            cp,
            getattr(cp, "cuda", None),
            getattr(getattr(cp, "cuda", None), "runtime", None),
        )
    ).lower()
    if "rocm" in version_text or "hip" in version_text:
        return "rocm"
    return "cuda"


def _load_cupy_backend(requested: str, *, require_rocm: bool, strict: bool) -> HardwareBackend:
    try:
        cp = importlib.import_module("cupy")
    except Exception as exc:
        raise HardwareBackendError(
            f"Hardware target '{requested}' requires CuPy, but CuPy could not be imported: {exc}"
        ) from exc

    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise HardwareBackendError(
            f"Hardware target '{requested}' requires a visible GPU/APU device, "
            f"but CuPy could not query devices: {exc}"
        ) from exc
    if device_count < 1:
        raise HardwareBackendError(
            f"Hardware target '{requested}' requires a visible GPU/APU device; CuPy reported 0 devices."
        )

    runtime = _cupy_runtime_name(cp)
    if require_rocm and runtime != "rocm":
        raise HardwareBackendError(
            f"Hardware target '{requested}' requires a ROCm/HIP CuPy runtime for Hunter APU, "
            f"but detected runtime='{runtime}'."
        )

    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        raw_name = props.get("name", b"") if isinstance(props, dict) else b""
        if isinstance(raw_name, bytes):
            device_name = _prop_text(raw_name)
        else:
            device_name = str(raw_name or "GPU/APU")
    except Exception:
        device_name = "GPU/APU"

    return HardwareBackend(
        requested=str(requested),
        target="hunter-apu" if require_rocm else "gpu",
        accelerator=True,
        array_module="cupy",
        runtime=runtime,
        device_count=device_count,
        device_name=device_name,
        strict=bool(strict),
    )


def resolve_hardware_backend(target: str | HardwareBackend | dict | None = None) -> HardwareBackend:
    if isinstance(target, HardwareBackend):
        return target
    if isinstance(target, dict):
        raw = target.get("requested") or target.get("target") or "cpu"
    else:
        raw = target
    normalized = normalize_hardware_target(raw)
    if normalized == "cpu":
        return CPU_BACKEND
    if normalized == "auto":
        try:
            return _load_cupy_backend("auto", require_rocm=False, strict=False)
        except HardwareBackendError:
            return replace(CPU_BACKEND, requested="auto")
    if normalized == "gpu":
        return _load_cupy_backend("gpu", require_rocm=False, strict=True)
    if normalized == "hunter-apu":
        return _load_cupy_backend("hunter-apu", require_rocm=True, strict=True)
    raise HardwareBackendError(f"Unhandled hardware target '{target}'.")


def _apply_runtime_thread_limits() -> None:
    """
    Environment variables only affect BLAS/OpenMP pools that have not been
    initialised yet; NumPy is already imported here, so also apply the limit
    to the loaded pools via threadpoolctl (best effort, once per process).
    """
    global _THREAD_LIMITS_APPLIED
    if _THREAD_LIMITS_APPLIED:
        return
    _THREAD_LIMITS_APPLIED = True
    try:
        limit = int(str(os.environ.get("OMP_NUM_THREADS", "1")).strip() or "1")
    except ValueError:
        return
    try:
        from threadpoolctl import threadpool_limits  # type: ignore
    except Exception:
        return
    try:
        threadpool_limits(limits=max(1, limit))
    except Exception as exc:  # pragma: no cover - platform specific
        log.debug("threadpoolctl could not apply thread limits: %s", exc)


def configure_process_for_hardware(backend: str | HardwareBackend | dict | None) -> HardwareBackend:
    resolved = resolve_hardware_backend(backend)
    if resolved.accelerator:
        # Keep CPU support libraries conservative when the accelerator is doing
        # the matrix-heavy work. This avoids accidental CPU oversubscription in
        # batch array tasks and local multi-worker runs.
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        _apply_runtime_thread_limits()
    return resolved


def get_array_module(backend: str | HardwareBackend | dict | None):
    resolved = resolve_hardware_backend(backend)
    if not resolved.accelerator:
        return np
    return importlib.import_module("cupy")


def backend_summary(backend: str | HardwareBackend | dict | None) -> str:
    resolved = resolve_hardware_backend(backend)
    if not resolved.accelerator:
        return f"{resolved.requested}->cpu"
    return (
        f"{resolved.requested}->{resolved.target} "
        f"runtime={resolved.runtime} devices={resolved.device_count} device0={resolved.device_name}"
    )


def to_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    mod = type(value).__module__.split(".", 1)[0]
    if mod == "cupy":
        cp = importlib.import_module("cupy")
        return cp.asnumpy(value)
    return np.asarray(value)


def _on_backend(backend, op):
    """
    Run ``op(xp)`` with NumPy on the CPU backend, otherwise with CuPy, and
    return a NumPy array in both cases.
    """
    resolved = resolve_hardware_backend(backend)
    if not resolved.accelerator:
        return op(np)
    xp = get_array_module(resolved)
    return xp.asnumpy(op(xp))


def accelerated_dot(a, b, backend=None):
    return _on_backend(backend, lambda xp: xp.asarray(a).dot(xp.asarray(b)))


def accelerated_pinv_dot(a, b, backend=None):
    return _on_backend(
        backend, lambda xp: xp.linalg.pinv(xp.asarray(a)).dot(xp.asarray(b))
    )


def accelerated_corrcoef(a, backend=None):
    return _on_backend(backend, lambda xp: xp.corrcoef(xp.asarray(a)))


def accelerated_svd_values(a, backend=None):
    return _on_backend(
        backend,
        lambda xp: xp.linalg.svd(xp.asarray(a), full_matrices=False, compute_uv=False),
    )


def accelerated_solve(a, b, backend=None):
    return _on_backend(
        backend, lambda xp: xp.linalg.solve(xp.asarray(a), xp.asarray(b))
    )


def accelerated_zscore(a, axis=0, backend=None, eps=1e-12):
    def op(xp):
        x = xp.asarray(a, dtype=xp.float64)
        mean = xp.nanmean(x, axis=axis, keepdims=True)
        std = xp.nanstd(x, axis=axis, ddof=0, keepdims=True) + float(eps)
        return xp.nan_to_num((x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)

    return _on_backend(backend, op)


def accelerated_row_norm(a, axis=1, backend=None):
    return _on_backend(backend, lambda xp: xp.linalg.norm(xp.asarray(a), axis=axis))


def _eigh_status_key(resolved: HardwareBackend) -> tuple:
    return (
        resolved.target,
        resolved.runtime,
        resolved.device_name,
        int(resolved.device_count),
    )


def _eigh_backend_policy() -> str:
    policy = str(os.environ.get(EIGH_BACKEND_ENV, "auto") or "auto").strip().lower()
    if policy not in {"auto", "device", "cpu"}:
        raise HardwareBackendError(
            f"{EIGH_BACKEND_ENV}={policy!r} is invalid; "
            "expected one of: auto, device, cpu."
        )
    return policy


def check_eigh_parity(
    backend=None, n: int = 48, seed: int = 0, rtol: float = EIGH_PARITY_RTOL
) -> dict:
    """
    Compare accelerator ``eigh`` against ``numpy.linalg.eigh`` on a fixed
    symmetric positive-definite matrix. Eigenvectors are sign/rotation
    ambiguous, so the check uses eigenvalues, reconstruction and orthogonality.
    """
    resolved = resolve_hardware_backend(backend)
    rng = np.random.default_rng(int(seed))
    a = rng.standard_normal((int(n), int(n)))
    spd = a @ a.T + float(n) * np.eye(int(n))
    ref_vals = np.linalg.eigh(spd)[0]
    out: dict[str, Any] = {
        "backend": resolved.target,
        "runtime": resolved.runtime,
        "n": int(n),
        "rtol": float(rtol),
    }
    if not resolved.accelerator:
        out.update({"device_ok": False, "reason": "cpu_backend", "checked": False})
        return out
    xp = get_array_module(resolved)
    try:
        vals_d, vecs_d = xp.linalg.eigh(xp.asarray(spd, dtype=xp.float64))
        vals = np.asarray(to_numpy(vals_d), dtype=float)
        vecs = np.asarray(to_numpy(vecs_d), dtype=float)
    except Exception as exc:
        out.update(
            {
                "device_ok": False,
                "checked": True,
                "reason": f"device_eigh_error:{type(exc).__name__}: {exc}",
            }
        )
        return out
    scale = float(np.max(np.abs(ref_vals))) or 1.0
    if vals.shape != ref_vals.shape or vecs.shape != spd.shape:
        out.update(
            {
                "device_ok": False,
                "checked": True,
                "reason": "device_eigh_shape_mismatch",
            }
        )
        return out
    val_err = float(np.max(np.abs(np.sort(vals) - ref_vals)) / scale)
    recon_err = float(
        np.max(np.abs((vecs * vals) @ vecs.T - spd)) / float(np.max(np.abs(spd)))
    )
    orth_err = float(np.max(np.abs(vecs.T @ vecs - np.eye(int(n)))))
    finite = bool(np.all(np.isfinite(vals)) and np.all(np.isfinite(vecs)))
    ok = finite and max(val_err, recon_err, orth_err) <= float(rtol)
    out.update(
        {
            "device_ok": bool(ok),
            "checked": True,
            "eigenvalue_rel_err": val_err,
            "reconstruction_rel_err": recon_err,
            "orthogonality_err": orth_err,
            "reason": None if ok else "device_eigh_parity_failed",
        }
    )
    return out


def device_eigh_status(backend=None) -> dict:
    """Return (and cache per process) whether the accelerator eigh path is used."""
    resolved = resolve_hardware_backend(backend)
    if not resolved.accelerator:
        return {"use_device": False, "reason": "cpu_backend"}
    key = _eigh_status_key(resolved)
    cached = _EIGH_DEVICE_STATUS.get(key)
    if cached is not None:
        return cached
    policy = _eigh_backend_policy()
    if policy == "cpu":
        status = {
            "use_device": False,
            "reason": f"{EIGH_BACKEND_ENV}=cpu",
            "policy": policy,
        }
    elif policy == "device":
        status = {
            "use_device": True,
            "reason": f"{EIGH_BACKEND_ENV}=device (parity check skipped)",
            "policy": policy,
        }
    else:
        parity = check_eigh_parity(resolved)
        status = {
            "use_device": bool(parity.get("device_ok")),
            "reason": parity.get("reason"),
            "policy": policy,
            "parity": parity,
        }
        if not status["use_device"]:
            log.warning(
                "Accelerator eigh disabled for this process (%s); "
                "using NumPy eigh on the CPU.",
                parity.get("reason"),
            )
    _EIGH_DEVICE_STATUS[key] = status
    return status


def _cpu_psd_invsqrt(mat, eps):
    m = np.asarray(to_numpy(mat), dtype=float)
    sym = 0.5 * (m + m.T)
    vals, vecs = np.linalg.eigh(sym)
    vals = np.maximum(vals, float(eps))
    return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T


def accelerated_psd_invsqrt(mat, eps=1e-10, backend=None):
    resolved = resolve_hardware_backend(backend)
    if not resolved.accelerator:
        sym = 0.5 * (np.asarray(mat) + np.asarray(mat).T)
        vals, vecs = np.linalg.eigh(sym)
        vals = np.maximum(vals, float(eps))
        return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T
    if not device_eigh_status(resolved).get("use_device", False):
        return _cpu_psd_invsqrt(mat, eps)
    xp = get_array_module(resolved)
    try:
        m = xp.asarray(mat, dtype=xp.float64)
        sym = 0.5 * (m + m.T)
        vals, vecs = xp.linalg.eigh(sym)
        vals = xp.maximum(vals, float(eps))
        out = (vecs * (1.0 / xp.sqrt(vals))) @ vecs.T
        return xp.asnumpy(out)
    except Exception as exc:
        key = _eigh_status_key(resolved)
        _EIGH_DEVICE_STATUS[key] = {
            "use_device": False,
            "reason": f"device_eigh_error:{type(exc).__name__}: {exc}",
            "policy": _eigh_backend_policy(),
        }
        log.warning(
            "Accelerator eigh failed (%s); falling back to NumPy eigh on the CPU.", exc
        )
        return _cpu_psd_invsqrt(mat, eps)


def device_info(backend=None) -> dict:
    """Describe the resolved backend and visible devices (for logs/provenance)."""
    resolved = resolve_hardware_backend(backend)
    info: dict[str, Any] = {
        "backend": resolved.to_dict(),
        "visible_device_env": {
            name: os.environ.get(name)
            for name in (
                "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
                "CUDA_VISIBLE_DEVICES",
                "PMI_LOCAL_RANK",
                "PBS_JOBID",
                "PBS_ARRAY_INDEX",
            )
            if os.environ.get(name) is not None
        },
        "numpy_version": np.__version__,
    }
    if not resolved.accelerator:
        return info
    cp = importlib.import_module("cupy")
    info["cupy_version"] = str(getattr(cp, "__version__", "unknown"))
    runtime = cp.cuda.runtime
    try:
        info["runtime_version"] = int(runtime.runtimeGetVersion())
    except Exception:
        info["runtime_version"] = None
    try:
        info["driver_version"] = int(runtime.driverGetVersion())
    except Exception:
        info["driver_version"] = None
    devices = []
    for idx in range(int(resolved.device_count)):
        rec: dict[str, Any] = {"index": int(idx)}
        try:
            props = runtime.getDeviceProperties(idx)
            name = props.get("name", b"") if isinstance(props, dict) else b""
            rec["name"] = str(_prop_text(name))
            if isinstance(props, dict):
                for key in (
                    "totalGlobalMem",
                    "multiProcessorCount",
                    "gcnArchName",
                    "major",
                    "minor",
                ):
                    val = _prop_text(props.get(key))
                    if val is not None:
                        rec[key] = val
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
        devices.append(rec)
    info["devices"] = devices
    return info
