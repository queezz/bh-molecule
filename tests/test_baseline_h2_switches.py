"""Tests for the baseline / H2-line / fit-window switches and the batch
pass-through (NME revision session S5b).

Each switch defaults to the production model: ``BHModel()`` has no extra fit
parameters, ``BHFitter`` keeps its seven, and ``run_bh_batch`` builds
``BHModel()`` unless ``model_options`` says otherwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import bh_molecule.workflows.batch_fit as batch_fit_mod
from bh_molecule.fit import BHFitter
from bh_molecule.physics import (
    BASELINE_REF_NM,
    EXTRA_PARAM_SPECS,
    H2_NUISANCE_LINES_NM,
    BHModel,
)

X_WIN = np.linspace(433.05, 433.90, 356)
P7 = dict(
    C=1.0, T_rot=3500.0, dx=0.012, w_inst=0.018692, base=0.02, I_R7=2e-3, I_R8=1e-3
)
W_FIX = 0.018692


class _DummyVis:
    def __init__(self, x, y):
        self._x, self._y = x, y

    def spectrum(self, frame, channel):  # noqa: ARG002
        return self._x, self._y


def _args(d):
    return [d[k] for k in ("C", "T_rot", "dx", "w_inst", "base", "I_R7", "I_R8")]


# --- defaults -----------------------------------------------------------------


def test_defaults_add_no_parameters_and_fitter_keeps_seven():
    m = BHModel()
    assert (m.baseline, m.h2_lines) == ("constant", False)
    assert m.extra_param_names == () and m.extra_p0 == () and m.extra_bounds == ((), ())
    fitr = BHFitter(vis=None, model=m)
    assert fitr.param_names == ["C", "T_rot", "dx", "w_inst", "base", "I_R7", "I_R8"]
    assert fitr.p0.size == 7 and fitr.bounds[0].size == 7
    explicit = BHModel(baseline="constant", h2_lines=False)
    assert np.array_equal(
        m.full_fit_model(X_WIN, *_args(P7)), explicit.full_fit_model(X_WIN, *_args(P7))
    )


@pytest.mark.parametrize("kwargs", [dict(baseline="quadratic"), dict(h2_lines="yes")])
def test_invalid_switch_values_raise(kwargs):
    with pytest.raises(ValueError):
        BHModel(**kwargs)


def test_wrong_number_of_extra_parameters_raises():
    with pytest.raises(TypeError):
        BHModel(baseline="linear").full_fit_model(X_WIN, *_args(P7))
    with pytest.raises(TypeError):
        BHModel().full_fit_model(X_WIN, *_args(P7), 0.1)


# --- linear baseline -------------------------------------------------------------


def test_linear_baseline_adds_slope_about_fixed_reference_on_unshifted_grid():
    slope = 0.25
    base_model, lin = BHModel(v_max=0), BHModel(v_max=0, baseline="linear")
    assert lin.extra_param_names == ("base_slope",)
    diff = lin.full_fit_model(X_WIN, *_args(P7), slope) - base_model.full_fit_model(
        X_WIN, *_args(P7)
    )
    assert np.allclose(diff, slope * (X_WIN - BASELINE_REF_NM), rtol=0, atol=1e-12)
    # Independent of the wavelength shift dx: the baseline lives on the data grid.
    p_shift = dict(P7, dx=-0.05)
    diff2 = lin.full_fit_model(
        X_WIN, *_args(p_shift), slope
    ) - base_model.full_fit_model(X_WIN, *_args(p_shift))
    assert np.allclose(diff, diff2, rtol=0, atol=1e-12)
    # base is the baseline value at the reference wavelength.
    assert BASELINE_REF_NM == pytest.approx(0.5 * (433.05 + 433.90))


def test_linear_baseline_bounds_are_the_declared_specs():
    m = BHModel(baseline="linear")
    lo, hi = m.extra_bounds
    assert (m.extra_p0[0], lo[0], hi[0]) == EXTRA_PARAM_SPECS["base_slope"]
    assert lo[0] < 0 < hi[0]


# --- H2 nuisance lines -------------------------------------------------------------


def test_h2_lines_are_bh_width_gaussians_on_the_shifted_grid():
    m0, mh = BHModel(v_max=0), BHModel(v_max=0, h2_lines=True)
    assert mh.extra_param_names == ("I_H2_R01", "I_H2_R2")
    assert H2_NUISANCE_LINES_NM == {"I_H2_R01": 433.2619, "I_H2_R2": 433.5522}
    a1, a2 = 4e-3, 2e-3
    x = np.linspace(433.05, 433.90, 20001)
    diff = mh.full_fit_model(x, *_args(P7), a1, a2) - m0.full_fit_model(x, *_args(P7))
    dx = P7["dx"]
    expected = a1 * m0.line_profile(
        x + dx, 433.2619, P7["w_inst"], 0.0
    ) + a2 * m0.line_profile(x + dx, 433.5522, P7["w_inst"], 0.0)
    assert np.allclose(diff, expected, rtol=1e-12, atol=1e-12)
    # Peaks appear at rest wavelength minus dx on the measured grid, as BH lines do.
    near = np.abs(x - (433.5522 - dx)) < 0.03
    assert x[near][np.argmax(diff[near])] == pytest.approx(433.5522 - dx, abs=1e-4)
    # Same FWHM as a BH line with T_tra = 0 (pure instrumental width).
    half = diff[near] >= 0.5 * diff[near].max()
    fwhm = x[near][half][-1] - x[near][half][0]
    assert fwhm == pytest.approx(P7["w_inst"], rel=0.01)


def test_combined_switches_order_slope_then_h2():
    m = BHModel(baseline="linear", h2_lines=True)
    assert m.extra_param_names == ("base_slope", "I_H2_R01", "I_H2_R2")
    assert len(m.extra_p0) == 3 and len(m.extra_bounds[0]) == 3


# --- fitter with extra parameters ----------------------------------------------------


def _synthetic(model, extra, noise=0.004, seed=1):
    """Model spectrum normalised to its peak in 433.08-433.30 nm, plus noise.

    Returns ``(y, scale)``; every amplitude-like parameter of the normalised
    spectrum is the true value divided by ``scale``."""
    rng = np.random.default_rng(seed)
    p = dict(P7, base=0.08)
    y = model.full_fit_model(X_WIN, *_args(p), *extra)
    scale = y[(X_WIN >= 433.08) & (X_WIN <= 433.30)].max()
    return y / scale + rng.normal(0.0, noise, X_WIN.size), scale


def test_fitter_appends_model_parameters_and_keeps_fixed_w_inst():
    m = BHModel(v_max=0, baseline="linear", h2_lines=True)
    fitr = BHFitter(
        vis=None,
        model=m,
        base_tight=True,
        base_bound=0.03,
        w_inst_default=W_FIX,
        fix_w_inst=True,
    )
    assert fitr.param_names[7:] == ["base_slope", "I_H2_R01", "I_H2_R2"]
    assert fitr.param_units[7] == "1/nm"
    assert np.allclose(fitr.p0[7:], m.extra_p0)
    assert np.allclose(fitr.bounds[0][7:], m.extra_bounds[0])
    assert np.allclose(fitr.bounds[1][7:], m.extra_bounds[1])
    # Asymmetric base bounds with a fixed w_inst (lo == hi there) are accepted.
    fitr.set_bounds(lower={"base": -0.05}, upper={"base": 0.5})
    b = fitr.param_names.index("base")
    assert (fitr.bounds[0][b], fitr.bounds[1][b]) == (-0.05, 0.5)
    with pytest.raises(ValueError):
        BHFitter(vis=None, model=m, p0=(1.0, 4000.0))


def test_fit_recovers_temperature_slope_and_h2_line():
    truth_model = BHModel(v_max=0, baseline="linear", h2_lines=True)
    y, scale = _synthetic(truth_model, extra=(0.2, 0.0, 3e-3))
    fitr = BHFitter(
        vis=_DummyVis(X_WIN, y),
        model=truth_model,
        base_tight=True,
        w_inst_default=W_FIX,
        fix_w_inst=True,
    )
    fitr.set_bounds(lower={"base": -0.05}, upper={"base": 0.5})
    res = fitr.fit(0, 0)
    p = dict(zip(fitr.param_names, res["params"]))
    assert res["params"].size == 10 and res["cov"].shape == (10, 10)
    assert res["errors"][fitr.param_names.index("w_inst")] == 0.0
    assert p["T_rot"] == pytest.approx(P7["T_rot"], rel=0.05)
    assert p["base_slope"] == pytest.approx(0.2 / scale, abs=0.02)
    assert p["base"] == pytest.approx(0.08 / scale, abs=0.01)
    assert p["I_H2_R2"] > 5 * res["errors"][fitr.param_names.index("I_H2_R2")]
    assert np.allclose(res["yfit"], fitr._f(res["x"], *res["params"]))


def test_explicit_p0_of_production_length_is_extended():
    m = BHModel(v_max=0, baseline="linear")
    y, _ = _synthetic(m, extra=(0.1,))
    fitr = BHFitter(
        vis=_DummyVis(X_WIN, y), model=m, w_inst_default=W_FIX, fix_w_inst=True
    )
    res = fitr.fit(0, 0, p0=(1.0, 4000.0, 0.0, W_FIX, 0.0, 1e-3, 1e-3))
    assert res["params"].size == 8


# --- fit-window upper limit and the batch pass-through --------------------------------


class _CubeVis:
    """Stand-in for ``Vis133M`` exposing ``cube`` and ``wl_nm`` (F, C, P)."""

    def __init__(self):
        wl = np.linspace(432.9, 434.2, 545)
        self.wl_nm = np.tile(wl, (2, 1))
        sig = BHModel(v_max=0).full_fit_model(wl, *_args(dict(P7, base=0.0)))
        hg = 3.0 / (1.0 + ((wl - 434.047) / 0.05) ** 2)  # an H-gamma-like wing
        rng = np.random.default_rng(3)
        cube = np.full((6, 2, wl.size), 100.0)
        cube[4:, :, :] += 50.0 * (sig + hg + 0.05)
        self.cube = cube + rng.normal(0.0, 0.2, cube.shape)

    @classmethod
    def from_files(cls, path, scale=1.0):  # noqa: ARG003
        return cls()

    def apply_cw(self, cw_nm):  # noqa: ARG002
        pass

    def set_scale(self, scale):  # noqa: ARG002
        pass

    def set_baseline_zero(self, on=True):  # noqa: ARG002
        pass

    def set_time_linspace(self, t0, t1):  # noqa: ARG002
        pass


def _run(monkeypatch, tmp_path, name, **kw):
    monkeypatch.setattr(batch_fit_mod, "Vis133M", _CubeVis)
    monkeypatch.setattr(batch_fit_mod, "check_background_flat", lambda *a, **k: None)
    fits = tmp_path / f"{name}.fits"
    fits.write_bytes(b"")
    return batch_fit_mod.run_bh_batch(
        fits,
        frames=[4, 5],
        channels=[0, 1],
        cw=431.91,
        scale=1.0,
        background_frames=(0, 1, 2, 3),
        w_inst_default=W_FIX,
        fix_w_inst=True,
        out_dir=tmp_path / "out",
        save_frames=False,
        save_grid=False,
        **kw,
    )


def test_fit_window_upper_limit_trims_without_changing_normalisation(
    monkeypatch, tmp_path
):
    _, full, _ = _run(monkeypatch, tmp_path, "100001")
    _, trim, shot_dir = _run(
        monkeypatch, tmp_path, "100002", bh_fit_range=(433.05, 433.80)
    )
    xf, yf, _ = full[(4, 0)]
    xt, yt, yfit_t = trim[(4, 0)]
    assert xf.max() > 433.85 and xt.max() <= 433.80
    common = xf <= 433.80
    assert np.array_equal(xt, xf[common])
    assert np.array_equal(yt, yf[common])  # normalisation is set in 433.08-433.30
    assert yfit_t.size == xt.size
    assert not (shot_dir / "grid.pdf").exists()  # save_grid=False


def test_run_bh_batch_passes_model_options_and_writes_extra_columns(
    monkeypatch, tmp_path
):
    res, _, shot_dir = _run(
        monkeypatch,
        tmp_path,
        "100003",
        model_options=dict(
            upper_energies="fernando_bernath", v_max=0, baseline="linear", h2_lines=True
        ),
        bounds={"lower": {"base": -0.05}, "upper": {"base": 0.5}},
    )
    summary = pd.read_csv(shot_dir / "summary.csv")
    for col in ("base_slope", "base_slope_err", "I_H2_R01", "I_H2_R2_err"):
        assert col in summary.columns
    assert len(summary) == 4 and "error" not in summary.columns
    assert summary["base"].between(-0.05, 0.5).all()
    assert (summary["w_inst"] == W_FIX).all()


def test_run_bh_batch_default_model_is_production(monkeypatch, tmp_path):
    seen = {}
    real = batch_fit_mod.BHModel

    def _spy(**kw):
        seen["kw"] = kw
        return real(**kw)

    monkeypatch.setattr(batch_fit_mod, "BHModel", _spy)
    _, _, shot_dir = _run(monkeypatch, tmp_path, "100004")
    assert seen["kw"] == {}
    cols = pd.read_csv(shot_dir / "summary.csv").columns
    assert "base_slope" not in cols and "I_H2_R01" not in cols
    assert (shot_dir / "grid.pdf").exists() is False  # save_grid=False in _run
