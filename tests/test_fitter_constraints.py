"""Tests for the configurable fitter constraints (w_inst, dx, base bounds).

These tests do not depend on private LHD FITS data; they exercise the
fitter, the batch workflow plumbing, and the YAML/CLI propagation path
through ``bh_molecule.cli.main_bh``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bh_molecule.dataio import load_v00_wavelengths
from bh_molecule.fit import (
    BHFitter,
    DEFAULT_BASE_TIGHT_NM,
    DEFAULT_DX_TOL_NM,
)
from bh_molecule.physics import BHModel
from bh_molecule.workflows.calibration import (
    summarize_fit_distribution,
)
import bh_molecule.workflows.batch_fit as batch_fit_mod


# ---------------------------------------------------------------------------
# Helpers / fixtures.
# ---------------------------------------------------------------------------


class _DummyVis:
    """Minimal vis stand-in exposing ``spectrum(frame, channel)``."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self._x = x
        self._y = y

    def spectrum(self, frame: int, channel: int):  # noqa: ARG002
        return self._x, self._y


def _synthetic_spectrum(true_w_inst: float = 0.022, true_dx: float = 0.01):
    rng = np.random.default_rng(42)
    v00 = load_v00_wavelengths()
    model = BHModel(v00)
    x = np.linspace(433.05, 433.90, 500)
    y_true = model.full_fit_model(
        x,
        C=0.6,
        T_rot=2600.0,
        dx=true_dx,
        w_inst=true_w_inst,
        base=0.005,
        I_R7=1e-3,
        I_R8=1e-3,
    )
    y = y_true + rng.normal(0.0, 0.005, size=x.shape)
    return x, y, model


# ---------------------------------------------------------------------------
# w_inst_default / w_inst_bounds / fix_w_inst on BHFitter.
# ---------------------------------------------------------------------------


def test_w_inst_default_sets_initial_guess():
    x, y, model = _synthetic_spectrum()
    fitr = BHFitter(vis=_DummyVis(x, y), model=model, w_inst_default=0.022)
    w_idx = fitr.param_names.index("w_inst")
    assert fitr.p0[w_idx] == pytest.approx(0.022)


def test_w_inst_bounds_propagate():
    x, y, model = _synthetic_spectrum()
    fitr = BHFitter(
        vis=_DummyVis(x, y),
        model=model,
        w_inst_bounds=(0.020, 0.024),
    )
    w_idx = fitr.param_names.index("w_inst")
    assert fitr.bounds[0][w_idx] == pytest.approx(0.020)
    assert fitr.bounds[1][w_idx] == pytest.approx(0.024)
    # p0 is moved inside the new window if it was outside.
    assert 0.020 <= fitr.p0[w_idx] <= 0.024


def test_w_inst_bounds_invalid_raises():
    x, y, model = _synthetic_spectrum()
    with pytest.raises(ValueError):
        BHFitter(vis=_DummyVis(x, y), model=model, w_inst_bounds=(0.03, 0.02))
    with pytest.raises(ValueError):
        BHFitter(vis=_DummyVis(x, y), model=model, w_inst_bounds=(-0.001, 0.05))


def test_fix_w_inst_holds_value_exactly():
    """Fixed ``w_inst`` must equal ``w_inst_default`` after the fit, with
    zero reported error (parameter elimination, not just tight bounds)."""
    target_w = 0.022
    x, y, model = _synthetic_spectrum(true_w_inst=target_w)
    fitr = BHFitter(
        vis=_DummyVis(x, y),
        model=model,
        w_inst_default=target_w,
        fix_w_inst=True,
    )
    res = fitr.fit(0, 0)
    w_idx = fitr.param_names.index("w_inst")
    assert res["params"][w_idx] == pytest.approx(target_w, rel=0, abs=1e-12)
    assert res["errors"][w_idx] == 0.0
    # Covariance row/col for w_inst are zero (fixed parameter).
    assert np.all(res["cov"][w_idx, :] == 0.0)
    assert np.all(res["cov"][:, w_idx] == 0.0)
    # And the rest of the recovered parameters are sensible.
    dx_idx = fitr.param_names.index("dx")
    assert abs(res["params"][dx_idx] - 0.01) < 0.02


def test_fix_w_inst_requires_value():
    """``fix_w_inst=True`` requires a finite non-negative value."""
    x, y, model = _synthetic_spectrum()
    bad_p0 = (1.0, 4000.0, 0.0, np.nan, 0.0, 1e-3, 1e-3)
    with pytest.raises(ValueError):
        BHFitter(vis=_DummyVis(x, y), model=model, p0=bad_p0, fix_w_inst=True)


def test_dx_tol_nm_override():
    x, y, model = _synthetic_spectrum()
    fitr = BHFitter(vis=_DummyVis(x, y), model=model, dx_tol_nm=0.05)
    dx_idx = fitr.param_names.index("dx")
    assert fitr.bounds[0][dx_idx] == pytest.approx(-0.05)
    assert fitr.bounds[1][dx_idx] == pytest.approx(+0.05)


def test_dx_tol_nm_default_is_unchanged():
    x, y, model = _synthetic_spectrum()
    fitr = BHFitter(vis=_DummyVis(x, y), model=model)
    dx_idx = fitr.param_names.index("dx")
    assert fitr.bounds[0][dx_idx] == pytest.approx(-DEFAULT_DX_TOL_NM)
    assert fitr.bounds[1][dx_idx] == pytest.approx(+DEFAULT_DX_TOL_NM)


def test_base_bound_with_base_tight():
    x, y, model = _synthetic_spectrum()
    fitr = BHFitter(
        vis=_DummyVis(x, y),
        model=model,
        base_tight=True,
        base_bound=0.005,
    )
    base_idx = fitr.param_names.index("base")
    assert fitr.bounds[0][base_idx] == pytest.approx(-0.005)
    assert fitr.bounds[1][base_idx] == pytest.approx(+0.005)
    assert fitr.p0[base_idx] == 0.0


def test_describe_constraints_reports_active_values():
    x, y, model = _synthetic_spectrum()
    fitr = BHFitter(
        vis=_DummyVis(x, y),
        model=model,
        w_inst_default=0.022,
        w_inst_bounds=(0.020, 0.024),
        fix_w_inst=True,
        dx_tol_nm=0.1,
        base_tight=True,
        base_bound=0.02,
    )
    desc = fitr.describe_constraints()
    assert desc["w_inst_default"] == pytest.approx(0.022)
    assert desc["w_inst_bounds"] == (pytest.approx(0.022), pytest.approx(0.022))
    assert desc["fix_w_inst"] is True
    assert desc["dx_tol_nm"] == pytest.approx(0.1)
    assert desc["base_bound"] == pytest.approx(0.02)
    assert desc["base_tight"] is True


# ---------------------------------------------------------------------------
# Compatibility with the preprocess callback path.
# ---------------------------------------------------------------------------


def test_fix_w_inst_works_with_preprocess_callback():
    """Parameter elimination must still route through `_window_data` /
    `_preprocess` so the batch path keeps working."""
    target_w = 0.025
    x_full, y_full, model = _synthetic_spectrum(true_w_inst=target_w)

    captured = {}

    def preprocess(_vis, frame, channel):
        captured["called"] = (frame, channel)
        return x_full, y_full

    fitr = BHFitter(
        vis=None,
        model=model,
        preprocess=preprocess,
        w_inst_default=target_w,
        fix_w_inst=True,
    )
    res = fitr.fit(11, 7)
    assert captured["called"] == (11, 7)
    w_idx = fitr.param_names.index("w_inst")
    assert res["params"][w_idx] == pytest.approx(target_w)


# ---------------------------------------------------------------------------
# run_bh_batch forwards the new kwargs into BHFitter.
# ---------------------------------------------------------------------------


def _stub_batch_environment(monkeypatch):
    """Stub Vis133M / fitter dependencies and capture BHFitter kwargs."""
    captured_kwargs: dict = {}

    class _Vis:
        def __init__(self):
            self.cube = np.zeros((4, 4, 50))
            self.wl_nm = np.broadcast_to(
                np.linspace(433.05, 433.90, 50), (4, 50)
            ).copy()

        @classmethod
        def from_files(cls, path, scale=1.0):  # noqa: ARG002
            return cls()

        def apply_cw(self, cw_nm):  # noqa: ARG002
            pass

        def set_scale(self, scale):  # noqa: ARG002
            pass

        def set_baseline_zero(self, on=True):  # noqa: ARG002
            pass

        def set_time_linspace(self, t0, t1):  # noqa: ARG002
            pass

        def spectrum(self, frame, channel):  # noqa: ARG002
            return np.linspace(433.05, 433.90, 50), np.zeros(50)

    class _FakeFitter:
        param_names = ["C", "T_rot", "dx", "w_inst", "base", "I_R7", "I_R8"]

        def __init__(self, vis, model, **kwargs):
            captured_kwargs.update(kwargs)
            self.vis = vis
            self.model = model
            self.p0 = np.array([1.0, 4000.0, 0.0, 0.025, 0.0, 1e-3, 1e-3])
            self.bounds = (
                np.array([0, 0, -0.3, 0, -10, 0, 0], dtype=float),
                np.array([10, 10000, 0.3, 0.1, 10, 1, 1], dtype=float),
            )
            self._base_tight = bool(kwargs.get("base_tight", False))
            self._base_bound = float(kwargs.get("base_bound", 0.03))
            self._fix_w_inst = bool(kwargs.get("fix_w_inst", False))

        def set_bounds(self, lower=None, upper=None):  # noqa: ARG002
            pass

        def describe_constraints(self):
            return {
                "param_names": self.param_names,
                "p0": list(self.p0),
                "bounds_lower": list(self.bounds[0]),
                "bounds_upper": list(self.bounds[1]),
                "w_inst_default": float(self.p0[3]),
                "w_inst_bounds": (float(self.bounds[0][3]), float(self.bounds[1][3])),
                "fix_w_inst": self._fix_w_inst,
                "dx_tol_nm": float(self.bounds[1][2]),
                "base_tight": self._base_tight,
                "base_bound": self._base_bound,
                "nm_window": (433.05, 433.90),
                "preprocess": None,
            }

        def fit(self, frame, channel, return_fit=True):  # noqa: ARG002
            x = np.linspace(433.05, 433.90, 50)
            y = np.ones_like(x)
            return {
                "params": np.zeros(7),
                "errors": np.zeros(7),
                "x": x,
                "y": y,
                "yfit": y,
            }

        def plot_single(self, res, title=None):  # noqa: ARG002
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()
            ax.plot(res["x"], res["y"])
            return ax

    monkeypatch.setattr(batch_fit_mod, "Vis133M", _Vis)
    monkeypatch.setattr(batch_fit_mod, "check_background_flat", lambda *a, **k: None)
    monkeypatch.setattr(
        batch_fit_mod, "scan_signal_frames", lambda *a, **k: ([1], [10], None)
    )
    monkeypatch.setattr(batch_fit_mod, "save_batch_fit_grid", lambda *a, **k: None)
    monkeypatch.setattr(batch_fit_mod, "BHFitter", _FakeFitter)
    return captured_kwargs


def test_run_bh_batch_forwards_w_inst_kwargs(monkeypatch, tmp_path):
    captured = _stub_batch_environment(monkeypatch)
    fake_fits = tmp_path / "999000.fits"
    fake_fits.write_bytes(b"")

    batch_fit_mod.run_bh_batch(
        fake_fits,
        frames=[1],
        channels=[10],
        cw=431.91,
        scale=1.0,
        w_inst_default=0.022,
        w_inst_bounds=(0.020, 0.024),
        fix_w_inst=True,
        dx_tol_nm=0.05,
        base_bound=0.02,
        out_dir=tmp_path / "out",
        save_frames=False,
    )

    assert captured["w_inst_default"] == 0.022
    assert captured["w_inst_bounds"] == (0.020, 0.024)
    assert captured["fix_w_inst"] is True
    assert captured["dx_tol_nm"] == 0.05
    assert captured["base_bound"] == 0.02
    # The shared preprocessor was still injected by default.
    assert "preprocess" in captured


def test_run_bh_batch_none_kwargs_do_not_propagate(monkeypatch, tmp_path):
    """None values must NOT be forwarded as `None` to BHFitter."""
    captured = _stub_batch_environment(monkeypatch)
    fake_fits = tmp_path / "999001.fits"
    fake_fits.write_bytes(b"")

    batch_fit_mod.run_bh_batch(
        fake_fits,
        frames=[1],
        channels=[10],
        cw=431.91,
        scale=1.0,
        out_dir=tmp_path / "out",
        save_frames=False,
    )
    assert "w_inst_default" not in captured
    assert "w_inst_bounds" not in captured
    assert captured.get("fix_w_inst") in (None, False)
    assert "dx_tol_nm" not in captured
    assert "base_bound" not in captured


def test_run_bh_batch_bad_w_inst_bounds_raises(monkeypatch, tmp_path):
    _stub_batch_environment(monkeypatch)
    fake_fits = tmp_path / "999002.fits"
    fake_fits.write_bytes(b"")

    with pytest.raises(ValueError):
        batch_fit_mod.run_bh_batch(
            fake_fits,
            frames=[1],
            channels=[10],
            cw=431.91,
            scale=1.0,
            w_inst_bounds=(0.022,),  # not a (lo, hi) pair
            out_dir=tmp_path / "out",
            save_frames=False,
        )


# ---------------------------------------------------------------------------
# YAML -> CLI -> run_bh_batch propagation.
# ---------------------------------------------------------------------------


def test_cli_yaml_propagates_fitter_constraints(monkeypatch, tmp_path):
    """The `bh batch --config <yaml>` entry point must forward the new keys."""
    from bh_molecule import cli

    captured_call: dict = {}

    def _fake_run_bh_batch(fits_path, frames, channels, **kwargs):
        captured_call["fits_path"] = Path(fits_path)
        captured_call["frames"] = frames
        captured_call["channels"] = channels
        captured_call["kwargs"] = kwargs
        return None, None, None

    monkeypatch.setattr(cli, "run_bh_batch", _fake_run_bh_batch)

    fake_fits = tmp_path / "001234.fits"
    fake_fits.write_bytes(b"")

    config = tmp_path / "cfg.yaml"
    config.write_text(
        "fits_file: \"" + str(fake_fits).replace("\\", "/") + "\"\n"
        "cw: 431.91\n"
        "scale: 1.0\n"
        "w_inst_default: 0.022\n"
        "w_inst_bounds: [0.020, 0.024]\n"
        "fix_w_inst: true\n"
        "dx_tol_nm: 0.05\n"
        "base_bound: 0.02\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["bh", "batch", "--config", str(config)])
    cli.main_bh()

    kw = captured_call["kwargs"]
    assert kw["w_inst_default"] == 0.022
    assert kw["w_inst_bounds"] == [0.020, 0.024]
    assert kw["fix_w_inst"] is True
    assert kw["dx_tol_nm"] == 0.05
    assert kw["base_bound"] == 0.02


# ---------------------------------------------------------------------------
# Calibration helper sanity checks.
# ---------------------------------------------------------------------------


def test_summarize_fit_distribution_basic():
    df = pd.DataFrame(
        {
            "frame": [0, 1, 2, 3, 4],
            "channel": [0, 0, 0, 0, 0],
            "w_inst": [0.021, 0.022, 0.023, 0.022, 0.024],
            "dx": [0.01, -0.005, 0.0, 0.012, 0.008],
            "base": [0.0, 0.001, -0.002, 0.003, 0.0],
            "R2": [0.95, 0.90, 0.85, 0.80, 0.99],
        }
    )
    out = summarize_fit_distribution(df, params=("w_inst", "dx", "base"))
    assert list(out.index) == ["w_inst", "dx", "base"]
    assert out.loc["w_inst", "n"] == 5
    assert out.loc["w_inst", "median"] == pytest.approx(0.022)
    assert out.loc["w_inst", "mad"] > 0


def test_summarize_fit_distribution_r2_filter():
    df = pd.DataFrame(
        {
            "w_inst": [0.022, 1.0, 0.023, 0.024],
            "R2": [0.99, 0.20, 0.95, 0.90],
        }
    )
    out = summarize_fit_distribution(df, params=("w_inst",), r2_min=0.5)
    assert out.loc["w_inst", "n"] == 3
    assert out.loc["w_inst", "max"] < 0.05


def test_default_base_constants():
    assert DEFAULT_BASE_TIGHT_NM == pytest.approx(0.03)
    assert DEFAULT_DX_TOL_NM == pytest.approx(0.3)
