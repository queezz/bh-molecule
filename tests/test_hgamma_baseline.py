"""Tests for the H-gamma-wing continuum (``BHModel(baseline="hgamma")``, session S12).

The continuum is set C's linear one plus ``base_hg`` times the far wing of H-gamma,
a Lorentzian at 434.0472 nm (HWHM 0.03 nm) on the shifted grid, normalised to 1 at
433.80 nm. Set C (``baseline="linear"``) must stay bit-identical.
"""

from __future__ import annotations

import numpy as np
import pytest

from bh_molecule.fit import BHFitter
from bh_molecule.physics import (
    BASELINES,
    EXTRA_PARAM_SPECS,
    HGAMMA_AIR_NM,
    HGAMMA_WING_HWHM_NM,
    HGAMMA_WING_REF_NM,
    BHModel,
)

SET_C = dict(upper_energies="fernando_bernath", v_max=0, baseline="linear", h2_lines=True)
HG = dict(SET_C, baseline="hgamma")
X = np.arange(433.0504, 433.80, 0.00239)
W_FIX = 0.018692
P7 = (1.0, 3200.0, 0.004, W_FIX, 0.08, 2e-3, 1e-3)


class _Vis:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def spectrum(self, frame, channel):  # noqa: ARG002
        return self.x, self.y


def _fit(model, y, x=X):
    fitr = BHFitter(_Vis(x, y), model, nm_window=(x[0], x[-1]), base_tight=True,
                    w_inst_default=W_FIX, fix_w_inst=True)
    fitr.set_bounds(lower={"base": -0.05}, upper={"base": 0.5})
    r = fitr.fit(0, 0)
    return dict(zip(fitr.param_names, r["params"])), r, fitr


def test_switch_listed_and_set_c_unchanged():
    assert "hgamma" in BASELINES
    assert BASELINES[0] == "constant"
    assert BHModel(**SET_C).extra_param_names == ("base_slope", "I_H2_R01", "I_H2_R2")


def test_extra_parameters_and_order():
    m = BHModel(**HG)
    assert m.extra_param_names == ("base_slope", "base_hg", "I_H2_R01", "I_H2_R2")
    p0, lo, hi = EXTRA_PARAM_SPECS["base_hg"]
    assert (lo, hi) == (0.0, 10.0) and lo < p0 < hi
    assert m.extra_bounds[0][1] == 0.0 and m.extra_bounds[1][1] == 10.0
    assert BHModel(**dict(HG, h2_lines=False)).extra_param_names == ("base_slope", "base_hg")


def test_wing_shape():
    w = BHModel.hgamma_wing
    assert w(np.array([HGAMMA_AIR_NM - HGAMMA_WING_REF_NM]))[0] == pytest.approx(1.0)
    assert w(np.array([HGAMMA_AIR_NM + HGAMMA_WING_REF_NM]))[0] == pytest.approx(1.0)  # symmetric
    g = HGAMMA_WING_HWHM_NM
    d1, d2 = 0.25, 0.5
    ratio = w(np.array([HGAMMA_AIR_NM - d2]))[0] / w(np.array([HGAMMA_AIR_NM - d1]))[0]
    assert ratio == pytest.approx((d1**2 + g**2) / (d2**2 + g**2), rel=1e-12)
    vals = w(X)
    assert np.all(np.diff(vals) > 0)  # rises toward H-gamma across the window
    assert np.all(np.diff(vals, 2) > 0)  # convex


def test_zero_wing_reproduces_linear_and_wing_adds_exactly():
    lin = BHModel(**SET_C)
    hg = BHModel(**HG)
    slope, a1, a2 = 0.1, 2e-3, 1e-3
    y_lin = lin.full_fit_model(X, *P7, slope, a1, a2)
    assert np.array_equal(hg.full_fit_model(X, *P7, slope, 0.0, a1, a2), y_lin)
    a_hg = 0.3
    diff = hg.full_fit_model(X, *P7, slope, a_hg, a1, a2) - y_lin
    np.testing.assert_allclose(diff, a_hg * BHModel.hgamma_wing(X + P7[2]), rtol=0, atol=1e-12)


def test_wing_on_shifted_grid():
    hg = BHModel(**dict(HG, h2_lines=False))
    a = 0.4
    for dx in (0.0, 0.02):
        p = (1.0, 3000.0, dx, W_FIX, 0.0, 0.0, 0.0)
        y = hg.full_fit_model(X, *p, 0.0, a) - hg.full_fit_model(X, *p, 0.0, 0.0)
        np.testing.assert_allclose(y, a * BHModel.hgamma_wing(X + dx), atol=1e-12)


def test_fit_recovers_wing_and_temperature():
    hg = BHModel(**HG)
    truth = dict(T=2800.0, slope=-0.05, a_hg=0.35)
    p = (1.2, truth["T"], 0.004, W_FIX, 0.10, 1.5e-3, 1.0e-3)
    y = hg.full_fit_model(X, *p, truth["slope"], truth["a_hg"], 2e-3, 1e-3)
    y = y / y[(X >= 433.08) & (X <= 433.30)].max()
    got, r, fitr = _fit(hg, y)
    assert "base_hg" in fitr.param_names
    assert got["T_rot"] == pytest.approx(truth["T"], rel=1e-4)
    # the linear continuum cannot follow the wing and moves T
    got_lin, _, _ = _fit(BHModel(**SET_C), y)
    assert abs(got_lin["T_rot"] / truth["T"] - 1) > 3 * abs(got["T_rot"] / truth["T"] - 1)


def test_invalid_baseline_still_rejected():
    with pytest.raises(ValueError):
        BHModel(baseline="hgamma2")
