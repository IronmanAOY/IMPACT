# -*- coding: utf-8 -*-
"""Small helpers shared by several test modules."""

import types

import numpy as np


def fake_rocm_cupy():
    """A NumPy-backed stand-in for a ROCm CuPy build on an MI300A node."""
    cp = types.ModuleType("cupy")
    cp.__version__ = "13.6.0+fake-rocm"
    cp.__getattr__ = lambda name: getattr(np, name)  # NumPy stands in for HIP
    cp.asnumpy = np.asarray
    cp.linalg = np.linalg
    cp.add = np.add
    cp.cuda = types.SimpleNamespace(
        runtime=types.SimpleNamespace(
            is_hip=lambda: True,
            getDeviceCount=lambda: 4,
            getDeviceProperties=lambda i: {"name": b"AMD Instinct MI300A"},
        )
    )
    return cp


def evoked(ts, tr, onsets, node=0):
    """Known-answer evoked response of ``node``: the mean over onsets of the
    three samples after an onset minus the sample before it; NaN with fewer
    than three usable onsets."""
    idx = np.rint(np.asarray(onsets, dtype=float) / tr).astype(int)
    idx = idx[(idx >= 1) & (idx + 4 < ts.shape[1])]
    if idx.size < 3:
        return float("nan")
    return float(np.mean([ts[node, k + 1:k + 4].mean() - ts[node, k - 1]
                          for k in idx]))
