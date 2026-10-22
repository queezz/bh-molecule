"""Tests for the two switches of the NME revision's last checks (session S5h):
the (1-1) R(7)/R(8) lines tied to T_rot through the Hoenl-London factors
(``r11_lines="tied"``) and a linear wavelength stretch (``stretch=True``),
plus the fitter's elimination of model-declared fixed parameters.

Both switches default to the earlier model; set C (``upper_energies=
"fernando_bernath", v_max=0, baseline="linear", h2_lines=True``) must stay
bit-identical.
"""

from __future__ import annotations

import numpy as np
import pytest

from bh_molecule.dataio import load_band_wavelengths
from bh_molecule.fit import BHFitter
from bh_molecule.physics import (
    EXTRA_PARAM_SPECS,
    R11_MODEL_NM,
    STRETCH_REF_NM,
    BHModel,
    Branch,
)

SET_C = dict(upper_energies="fernando_bernath", v_max=0, baseline="linear", h2_lines=True)
X = np.arange(433.0504, 433.80, 0.00239)
P7 = (1.0, 3200.0, 0.006, 0.018692, 0.05, 2e-3, 1e-3)
H2 = (0.0, 2e-3, 1e-3)  # base_slope, I_H2_R01, I_H2_R2
W_FIX = 0.018692


class _Vis:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def spectrum(self, frame, channel):  # noqa: ARG002
        return self.x, self.y


def _fitter(model, y, x=X):
    fitr = BHFitter(_Vis(x, y), model, nm_window=(x[0], x[-1]), base_tight=True,
                    w_inst_default=W_FIX, fix_w_inst=True)
    fitr.set_bounds(lower={"base": -0.05}, upper={"base": 0.5})
    return fitr


def _fit(model, y, x=X):
    fitr = _fitter(model, y, x)
    r = fitr.fit(0, 0)
    return dict(zip(fitr.param_names, r["params"])), r, fitr


# --- defaults --------------------------------------------------------------------


def test_defaults_are_set_c():
    m = BHModel(**SET_C)
    assert (m.r11_lines, m.stretch, m.fixed_params) == ("free", False, {})
    explicit = BHModel(**SET_C, r11_lines="free", stretch=False)
    assert np.array_equal(m.full_fit_model(X, *P7, *H2), explicit.full_fit_model(X, *P7, *H2))


def test_production_fitter_has_no_model_fixed_parameters():
    m = BHModel(**SET_C)
    fitr = _fitter(m, m.full_fit_model(X, *P7, *H2))
    assert fitr._model_fixed == {}
    assert fitr.describe_constraints()["model_fixed"] == {}


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(r11_lines="fixed"),
        dict(r11_lines="tied", v_max=2, upper_energies="fernando_bernath"),
        dict(r11_lines="tied", rot_components=2, v_max=0),
        dict(stretch=1),
        dict(stretch="yes"),
    ],
)
def test_invalid_switches_raise(kwargs):
    with pytest.raises(ValueError):
        BHModel(**kwargs)


# --- r11_lines="tied" ------------------------------------------------------------


def test_tied_positions_are_fb_table_i_without_the_offset():
    m = BHModel(**SET_C, r11_lines="tied")
    r = load_band_wavelengths(1).set_index("J")["R"]
    assert m._r11_fb91["R7"]["wl"] == pytest.approx(433.644051, abs=1e-6)
    assert m._r11_fb91["R8"]["wl"] == pytest.approx(433.331292, abs=1e-6)
    assert m._r11_fb91["R7"]["wl"] == r.loc[7]
    for k in ("R7", "R8"):
        assert R11_MODEL_NM[k] - m._r11_fb91[k]["wl"] == pytest.approx(0.0037, abs=0.0001)


def test_tied_parameters_and_fixed_amplitudes():
    m = BHModel(**SET_C, r11_lines="tied")
    assert m.extra_param_names == ("base_slope", "I_H2_R01", "I_H2_R2", "S_v1")
    assert m.fixed_params == {"I_R7": 0.0, "I_R8": 0.0}
    assert m.extra_p0[-1] == EXTRA_PARAM_SPECS["S_v1"][0]


def test_tied_model_ignores_the_free_amplitudes():
    m = BHModel(**SET_C, r11_lines="tied")
    a = m.full_fit_model(X, *P7[:5], 0.0, 0.0, *H2, 0.4)
    b = m.full_fit_model(X, *P7[:5], 0.3, 0.7, *H2, 0.4)
    assert np.array_equal(a, b)


def test_tied_areas_follow_the_formula():
    m = BHModel(**SET_C, r11_lines="tied")
    C, T, S = 1.3, 3100.0, 0.45
    a7, a8 = m.r11_tied_areas(C, T, S)
    for J, area in ((7, a7), (8, a8)):
        N2 = J + 1
        E = m.upper_energy(1, N2, Branch.R) - m.upper_energy(1, 1, Branch.R)
        wl = m._r11_fb91[f"R{J}"]["wl"]
        photon = 6.62607015e-34 * (299792458.0 / (wl * 1e-9)) / (4 * np.pi)
        # A_coeff(1, N', N'') (2N'+1) = A_vib(1) (N'+1)/2: the R-branch Hoenl-London factor
        A_vib1 = 7.2356e-6 * 0.13678e13
        ref = 1e8 * C * S * photon * m.g_as * A_vib1 * (N2 + 1) / 2 * np.exp(-E / (T * m.K2wn))
        assert area == pytest.approx(ref, rel=1e-12)
    # linear in C and S_v1
    assert m.r11_tied_areas(2 * C, T, S)[0] == pytest.approx(2 * a7)
    assert m.r11_tied_areas(C, T, 3 * S)[1] == pytest.approx(3 * a8)


def test_tied_r8_over_r7_rises_with_t_rot():
    m = BHModel(**SET_C, r11_lines="tied")
    ratios = [np.divide(*m.r11_tied_areas(1.0, T, 1.0)[::-1]) for T in (1500, 3000, 6000)]
    assert ratios[0] < ratios[1] < ratios[2]
    # N' = 9 lies 197 cm^-1 above N' = 8: at 3 kK the two lines are nearly equal
    assert ratios[1] == pytest.approx(1.01, abs=0.01)


def test_tied_model_is_q_band_plus_two_lines():
    m = BHModel(**SET_C, r11_lines="tied")
    S = 0.4
    xs = X + P7[2]
    a7, a8 = m.r11_tied_areas(P7[0], P7[1], S)
    ref = (
        1e8 * m.spectrum(xs, P7[0], P7[1], P7[3], 0.0, Branch.Q)
        + a7 * m.line_profile(xs, m._r11_fb91["R7"]["wl"], P7[3], 0.0)
        + a8 * m.line_profile(xs, m._r11_fb91["R8"]["wl"], P7[3], 0.0)
        + P7[4]
    )
    ref = ref + H2[1] * m.line_profile(xs, 433.2619, P7[3], 0.0)
    ref = ref + H2[2] * m.line_profile(xs, 433.5522, P7[3], 0.0)
    assert np.allclose(m.full_fit_model(X, *P7, *H2, S), ref, rtol=1e-13, atol=1e-15)


def test_tied_fit_recovers_truth_and_holds_the_amplitudes_at_zero():
    m = BHModel(**SET_C, r11_lines="tied")
    y = m.full_fit_model(X, *P7, *H2, 0.4)
    p, r, fitr = _fit(m, y)
    assert p["T_rot"] == pytest.approx(P7[1], rel=1e-5)
    assert p["S_v1"] == pytest.approx(0.4, rel=1e-4)
    assert (p["I_R7"], p["I_R8"]) == (0.0, 0.0)
    i7, i8 = fitr.param_names.index("I_R7"), fitr.param_names.index("I_R8")
    assert r["errors"][i7] == 0.0 and r["errors"][i8] == 0.0
    assert p["w_inst"] == W_FIX
    assert fitr.describe_constraints()["model_fixed"] == {"I_R7": 0.0, "I_R8": 0.0}


def test_tied_fitter_set_bounds_accepts_the_fixed_amplitudes():
    m = BHModel(**SET_C, r11_lines="tied")
    fitr = _fitter(m, m.full_fit_model(X, *P7, *H2, 0.4))
    fitr.set_bounds(lower={"S_v1": 0.0}, upper={"S_v1": 2.0})
    i = fitr.param_names.index("I_R7")
    assert fitr.bounds[0][i] == fitr.bounds[1][i] == 0.0


def test_tied_works_with_natural_boron():
    m = BHModel(**SET_C, r11_lines="tied", isotopologue="natural")
    y = m.full_fit_model(X, *P7, *H2, 0.4)
    assert np.all(np.isfinite(y))
    p, _, _ = _fit(m, y)
    assert p["T_rot"] == pytest.approx(P7[1], rel=1e-4)


# --- stretch=True ------------------------------------------------------------------


def test_stretch_parameter_and_zero_is_the_unstretched_model():
    ms = BHModel(**SET_C, stretch=True)
    assert ms.extra_param_names == ("base_slope", "I_H2_R01", "I_H2_R2", "stretch")
    lo, hi = EXTRA_PARAM_SPECS["stretch"][1:]
    assert (lo, hi) == (-0.02, 0.02)
    m = BHModel(**SET_C)
    assert np.allclose(ms.full_fit_model(X, *P7, *H2, 0.0), m.full_fit_model(X, *P7, *H2),
                       rtol=0, atol=1e-14)


def test_stretch_moves_positions_and_keeps_the_measured_width():
    eps = 0.006
    ms = BHModel(**SET_C, stretch=True)
    m = BHModel(**SET_C)
    # without the slope, the stretched model on x is the unstretched one on the
    # stretched grid with the width scaled by (1 + eps)
    p = list(P7)
    xs_grid = X + eps * (X - STRETCH_REF_NM)
    p_w = list(P7)
    p_w[3] = P7[3] * (1 + eps)
    a = ms.full_fit_model(X, *p, *H2, eps)
    b = m.full_fit_model(xs_grid, *p_w, *H2)
    assert np.allclose(a, b, rtol=1e-12, atol=1e-14)
    # an isolated line keeps its FWHM on the measured grid
    xf = np.linspace(433.40, 433.70, 30001)
    only_r2 = BHModel(upper_energies="fernando_bernath", v_max=0, h2_lines="R2", stretch=True)
    y = only_r2.full_fit_model(xf, 0.0, 3000.0, 0.0, W_FIX, 0.0, 0.0, 0.0, 1e-3, eps)
    half = xf[y >= 0.5 * y.max()]
    assert half[-1] - half[0] == pytest.approx(W_FIX, abs=2e-5)
    # and its peak sits where x + eps (x - ref) equals the line
    peak = xf[np.argmax(y)]
    assert peak + eps * (peak - STRETCH_REF_NM) == pytest.approx(433.5522, abs=2e-5)


def test_stretch_fit_recovers_eps_where_the_plain_fit_cannot():
    ms = BHModel(**SET_C, stretch=True)
    y = ms.full_fit_model(X, *P7, *H2, 0.005)
    p, _, _ = _fit(ms, y)
    assert p["stretch"] == pytest.approx(0.005, abs=1e-6)
    assert p["T_rot"] == pytest.approx(P7[1], rel=1e-5)
    plain, r_plain, _ = _fit(BHModel(**SET_C), y)
    assert abs(plain["T_rot"] - P7[1]) > 1.0
    assert np.sum((r_plain["y"] - r_plain["yfit"]) ** 2) > 1e-6


def test_tied_and_stretch_together():
    m = BHModel(**SET_C, r11_lines="tied", stretch=True)
    assert m.extra_param_names[-2:] == ("S_v1", "stretch")
    y = m.full_fit_model(X, *P7, *H2, 0.4, -0.004)
    p, _, _ = _fit(m, y)
    assert p["S_v1"] == pytest.approx(0.4, rel=1e-3)
    assert p["stretch"] == pytest.approx(-0.004, abs=1e-6)
