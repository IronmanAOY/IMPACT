"""
TR resolution in fMRI preprocessing (no silent TR fallback).

Before the fix an implausible header TR (e.g. 20 s) was silently replaced by
2.0 s. The TR now comes from measured metadata (BIDS sidecar, then NIfTI
header); without a plausible value preprocessing raises unless --assume-tr
is given explicitly.
"""

import json
import logging

import numpy as np
import pytest

from impact_pipeline import preprocessing


class DummyHeader:
    def __init__(self, zooms=(1, 1, 20), units=("mm", "sec")):
        self._zooms = zooms
        self._units = units

    def get_zooms(self):
        return self._zooms

    def get_xyzt_units(self):
        return self._units

    def copy(self):
        return self


class DummyImg:
    def __init__(self, header=None):
        self.header = header or DummyHeader()
        self.shape = (2, 2, 2, 4)
        self.affine = np.eye(4)

    def get_fdata(self):
        return np.zeros(self.shape)


class DummyMasker:
    def fit_transform(self, img):
        return np.zeros((4, 1))


def _setup(tmp_path, monkeypatch, img, seen):
    fmriprep_deriv = tmp_path / "fmriprep"
    func = fmriprep_deriv / "sub-01" / "func"
    func.mkdir(parents=True)
    bold = func / "sub-01_task-audioawake_run-1_desc-preproc_bold.nii.gz"
    bold.write_bytes(b"")

    def dummy_clean(signals, **kwargs):
        seen["t_r"] = kwargs.get("t_r")
        return signals.T

    monkeypatch.setattr(preprocessing.image, "load_img", lambda p: img)
    monkeypatch.setattr(preprocessing, "clean", dummy_clean)
    monkeypatch.setattr(preprocessing, "NiftiLabelsMasker", lambda **k: DummyMasker())
    monkeypatch.setattr(
        preprocessing, "get_atlas_globs", lambda: {"schaefer400": "dummy_atlas.nii.gz"}
    )
    bf = type("BF", (), {"entities": {"task": "audioawake", "run": 1}})()
    return fmriprep_deriv, bold, bf


def test_implausible_tr_without_metadata_raises(tmp_path, monkeypatch):
    seen = {}
    deriv, _bold, bf = _setup(tmp_path, monkeypatch, DummyImg(), seen)
    with pytest.raises(ValueError, match="--assume-tr"):
        preprocessing.preprocess_subject(
            str(tmp_path / "bids"), str(deriv), "01", bf, str(tmp_path / "out")
        )
    assert "t_r" not in seen  # nothing was cleaned with an invented TR


def test_assume_tr_is_used_only_when_explicit(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    seen = {}
    deriv, _bold, bf = _setup(tmp_path, monkeypatch, DummyImg(), seen)
    rec = preprocessing.preprocess_subject(
        str(tmp_path / "bids"),
        str(deriv),
        "01",
        bf,
        str(tmp_path / "out"),
        assume_tr=2.0,
    )
    assert seen["t_r"] == 2.0
    assert rec["tr_source"] == "assumed"
    assert "explicitly assumed TR" in caplog.text
    # Missing confounds: FD is undefined (NaN), not zero motion.
    assert rec["mean_fd"] is None
    assert (tmp_path / "out" / "mean_fd.txt").read_text() == "nan"


def test_sidecar_repetition_time_wins_over_header(tmp_path, monkeypatch):
    seen = {}
    img = DummyImg(DummyHeader(zooms=(3.0, 3.0, 3.75, 2.5)))
    deriv, bold, bf = _setup(tmp_path, monkeypatch, img, seen)
    bold.with_name(bold.name.replace(".nii.gz", ".json")).write_text(
        json.dumps({"RepetitionTime": 2.0})
    )
    rec = preprocessing.preprocess_subject(
        str(tmp_path / "bids"), str(deriv), "01", bf, str(tmp_path / "out")
    )
    assert seen["t_r"] == 2.0
    assert rec["tr_source"] == "sidecar"


def test_header_tr_in_milliseconds_is_converted():
    img = DummyImg(DummyHeader(zooms=(3.0, 3.0, 3.75, 2000.0), units=("mm", "msec")))
    tr, source, _details = preprocessing.resolve_bold_tr(img)
    assert tr == pytest.approx(2.0)
    assert source == "nifti_header"


def test_implausible_sidecar_and_header_raise():
    img = DummyImg(DummyHeader(zooms=(3.0, 3.0, 3.75, 0.0)))
    with pytest.raises(ValueError):
        preprocessing.resolve_bold_tr(img)
    with pytest.raises(ValueError):
        preprocessing.resolve_bold_tr(img, assume_tr=45.0)
