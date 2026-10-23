"""Tests for the held continuum (``BHModel(baseline="fixed")``, ``BHFitter(continuum=...)``,
paper repo session S12b).

A continuum computed before the band fit (from pixels outside the BH window) is handed to
the model per spectrum and held; ``base`` is held at 0 and no continuum parameter is
fitted. Set C (``baseline="linear"``) must stay bit-identical.
"""

from __future__ import annotations

import numpy as np
import pytest

from bh_molecule.fit import BHFitter
from bh_molecule.physics import BASELINE_REF_NM, BASELINES, BHModel

SET_C = dict(upper_energies="fernando_bernath", v_max=0, baseline="linear", h2_lines=True)
FIX = dict(SET_C, baseline="fixed")
X = np.arange(433.0504, 433.80, 0.00239)
XW = np.arange(432.50, 433.95, 0.00239)  # a wider grid, as the floor builder returns
W_FIX = 0.018692
P7 = (1.0, 3200.0, 0.004, W_FIX, 0.0, 2e-3, 1e-3)


class _Vis:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def spectrum(self, frame, channel):  # noqa: ARG002
        return self.x, self.y


def _fitter(model, y, x=X, **kw):
    fitr = BHFitter(_Vis(x, y), model, nm_window=(x[0], x[-1]), base_tight=True,
                    w_inst_default=W_FIX, fix_w_inst=True, **kw)
    fitr.set_bounds(lower={"base": -0.05}, upper={"base": 0.5})
    return fitr


def _curved(x, dx=0.004):
    return 0.08 + 0.03 * (x - BASELINE_REF_NM) + 0.2 * BHModel.hgamma_wing(x + dx)


def test_switch_listed_and_set_c_unchanged():
    assert "fixed" in BASELINES and BASELINES[0] == "constant"
    assert BHModel(**SET_C).extra_param_names == ("base_slope", "I_H2_R01", "I_H2_R2")
    assert BHModel(**SET_C).fixed_params == {}
    m = BHModel(**FIX)
    assert m.extra_param_names == ("I_H2_R01", "I_H2_R2")
    assert m.fixed_params == {"base": 0.0}
    tied = BHModel(**dict(FIX, r11_lines="tied"))
    assert tied.fixed_params == {"base": 0.0, "I_R7": 0.0, "I_R8": 0.0}


def test_continuum_must_be_set_and_cover_the_grid():
    m = BHModel(**FIX)
    with pytest.raises(ValueError, match="set_continuum"):
        m.full_fit_model(X, *P7, 1e-3, 1e-3)
    m.set_continuum(X[5:], np.zeros(X.size - 5))
    with pytest.raises(ValueError, match="leaves the held continuum"):
        m.full_fit_model(X, *P7, 1e-3, 1e-3)
    with pytest.raises(ValueError):
        m.set_continuum(X, np.zeros(X.size - 1))
    with pytest.raises(ValueError):
        m.set_continuum(X, np.full(X.size, np.nan))
    with pytest.raises(ValueError, match='baseline="fixed"'):
        BHModel(**SET_C).set_continuum(X, np.zeros(X.size))


def test_held_linear_continuum_equals_set_c_model():
    lin, fix = BHModel(**SET_C), BHModel(**FIX)
    base, slope, a1, a2 = 0.12, -0.07, 2e-3, 1e-3
    p_lin = (*P7[:4], base, *P7[5:])
    y_lin = lin.full_fit_model(X, *p_lin, slope, a1, a2)
    fix.set_continuum(XW[::-1], (base + slope * (XW - BASELINE_REF_NM))[::-1])  # any order
    np.testing.assert_allclose(fix.full_fit_model(X, *P7, a1, a2), y_lin, rtol=0, atol=1e-12)
    # the continuum is on the measured grid: it does not move with dx
    p_dx = (P7[0], P7[1], 0.02, *P7[3:])
    fix0 = BHModel(**FIX)
    fix0.set_continuum(XW, np.zeros_like(XW))
    diff = fix.full_fit_model(X, *p_dx, 0.0, 0.0) - fix0.full_fit_model(X, *p_dx, 0.0, 0.0)
    np.testing.assert_allclose(diff, base + slope * (X - BASELINE_REF_NM), atol=1e-12)


def test_fit_recovers_temperature_with_the_true_continuum_held():
    fix = BHModel(**FIX)
    T = 2800.0
    fix.set_continuum(XW, _curved(XW))
    y = fix.full_fit_model(X, 1.2, T, 0.004, W_FIX, 0.0, 1.5e-3, 1.0e-3, 2e-3, 1e-3)
    calls = []

    def cont(vis, frame, channel):  # noqa: ARG001
        calls.append((frame, channel))
        return XW, _curved(XW)

    fitr = _fitter(BHModel(**FIX), y, continuum=cont)
    r = fitr.fit(3, 7)
    got = dict(zip(fitr.param_names, r["params"]))
    err = dict(zip(fitr.param_names, r["errors"]))
    assert calls == [(3, 7)]
    assert got["T_rot"] == pytest.approx(T, rel=1e-5)
    assert got["base"] == 0.0 and err["base"] == 0.0
    assert "base_slope" not in fitr.param_names
    assert fitr.describe_constraints()["continuum"] == "callable"
    assert fitr.describe_constraints()["model_fixed"] == {"base": 0.0}


def test_the_held_continuum_moves_the_temperature():
    """A continuum held too low leaves flux the band must take: T rises."""
    fix = BHModel(**FIX)
    fix.set_continuum(XW, _curved(XW))
    y = fix.full_fit_model(X, 1.2, 2800.0, 0.004, W_FIX, 0.0, 1.5e-3, 1.0e-3, 2e-3, 1e-3)
    fitr = _fitter(BHModel(**FIX), y, continuum=lambda v, f, c: (XW, _curved(XW) - 0.02))  # noqa: ARG005
    got = dict(zip(fitr.param_names, fitr.fit(0, 0)["params"]))
    assert got["T_rot"] > 2850.0


def test_continuum_argument_validated():
    y = np.zeros_like(X)
    with pytest.raises(ValueError, match='baseline="fixed"'):
        _fitter(BHModel(**SET_C), y, continuum=lambda v, f, c: (XW, 0 * XW))  # noqa: ARG005
    with pytest.raises(ValueError, match="callable"):
        _fitter(BHModel(**FIX), y, continuum=np.zeros_like(XW))


def test_set_c_fit_path_untouched_by_the_new_argument():
    lin = BHModel(**SET_C)
    y = lin.full_fit_model(X, 1.0, 3000.0, 0.004, W_FIX, 0.1, 2e-3, 1e-3, 0.05, 2e-3, 1e-3)
    a = _fitter(BHModel(**SET_C), y).fit(0, 0)["params"]
    b = _fitter(BHModel(**SET_C), y, continuum=None).fit(0, 0)["params"]
    assert np.array_equal(a, b)
