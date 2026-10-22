"""Tests for the model-refinement switches of the NME revision (session S5e):
10BH (``isotopologue``, ``bh10_positions``), free F1/F3 lines
(``extra_lines``), a quadratic continuum, the H2 R2 line alone
(``h2_lines="R2"``), two rotational components (``rot_components``) and the
fitter's excluded wavelength ranges (``exclude_nm``).

Every switch defaults to the earlier model; set C
(``upper_energies="fernando_bernath", v_max=0, baseline="linear",
h2_lines=True``) must stay bit-identical.
"""

from __future__ import annotations

from importlib import resources

import numpy as np
import pandas as pd
import pytest

from bh_molecule import isotopes as iso
from bh_molecule.fit import BHFitter
from bh_molecule.physics import (
    BASELINE_REF_NM,
    EXTRA_LINES_NM,
    EXTRA_PARAM_SPECS,
    H2_NUISANCE_LINES_NM,
    BHModel,
    Branch,
)

SET_C = dict(upper_energies="fernando_bernath", v_max=0, baseline="linear", h2_lines=True)
X = np.arange(433.0504, 433.80, 0.00239)
P7 = (1.0, 3200.0, 0.006, 0.018692, 0.05, 2e-3, 1e-3)
W_FIX = 0.018692


class _Vis:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def spectrum(self, frame, channel):  # noqa: ARG002
        return self.x, self.y


def _q11():
    with resources.files("bh_molecule._resources").joinpath("11BH_v00.csv").open("rb") as f:
        return pd.read_csv(f, comment="#").set_index("J")["Q"]


def _fit(model, y, x=X, **kw):
    fitr = BHFitter(_Vis(x, y), model, nm_window=(x[0], x[-1]), base_tight=True,
                    w_inst_default=W_FIX, fix_w_inst=True, **kw)
    fitr.set_bounds(lower={"base": -0.05}, upper={"base": 0.5})
    r = fitr.fit(0, 0)
    return dict(zip(fitr.param_names, r["params"])), r, fitr


# --- defaults and set C ---------------------------------------------------------


def test_set_c_output_is_the_inline_formula():
    m = BHModel(**SET_C)
    assert (m.isotopologue, m.extra_lines, m.rot_components) == ("11BH", (), 1)
    slope, a1, a2 = 0.12, 2e-3, 1e-3
    xs = X + P7[2]
    ref = (
        1e8 * m.spectrum(xs, P7[0], P7[1], P7[3], 0.0, Branch.Q)
        + P7[5] * m.line_profile(xs, 433.6477624402892, P7[3], 0.0)
        + P7[6] * m.line_profile(xs, 433.3350058444114, P7[3], 0.0)
        + P7[4]
    )
    ref = ref + slope * (X - BASELINE_REF_NM)
    for a, wl in zip((a1, a2), H2_NUISANCE_LINES_NM.values()):
        ref = ref + a * m.line_profile(xs, wl, P7[3], 0.0)
    assert np.array_equal(m.full_fit_model(X, *P7, slope, a1, a2), ref)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(isotopologue="10BH"),
        dict(isotopologue="natural"),  # needs F&B energies and v_max=0
        dict(upper_energies="fernando_bernath", v_max=2, isotopologue="natural"),
        dict(bh10_positions="exomol"),
        dict(extra_lines=("F2",)),
        dict(extra_lines=("F1", "F1")),
        dict(rot_components=3),
        dict(rot_components=True),
        dict(v_max=2, rot_components=2),
        dict(baseline="cubic"),
        dict(h2_lines="R0"),
        dict(h2_lines="R2", h2_r01_ratio=1.0),
    ],
)
def test_invalid_refinement_switches_raise(kwargs):
    with pytest.raises(ValueError):
        BHModel(**kwargs)


def test_parameter_order_of_all_switches_together():
    m = BHModel(**dict(SET_C, baseline="quadratic", isotopologue="natural",
                       extra_lines=("F3", "F1"), rot_components=2))
    assert m.extra_param_names == (
        "base_slope", "base_curv", "I_H2_R01", "I_H2_R2", "I_F1", "I_F3", "T_rot2", "frac2"
    )
    lo, hi = m.extra_bounds
    for name, p0, a, b in zip(m.extra_param_names, m.extra_p0, lo, hi):
        assert (p0, a, b) == EXTRA_PARAM_SPECS[name]
        assert a <= p0 <= b


# --- 10BH line data --------------------------------------------------------------


def test_rho_and_natural_share():
    assert iso.RHO_10BH == pytest.approx(1.004164, abs=2e-6)
    assert iso.BH10_OVER_BH11 == pytest.approx(0.199 / 0.801)


def test_dunham_set_reproduces_the_11bh_q_lines():
    """F&B Table VIII with X-state Y04 read as -5.18e-12 (module note)."""
    q11 = _q11()
    J = np.arange(1, 24)
    calc = iso.dunham_term(iso.DUNHAM_A_FB91, 0, J, 1.0, iso.DUNHAM_A_LAMBDA_FB91, "f") - iso.dunham_term(
        iso.DUNHAM_X_FB91, 0, J
    )
    assert np.max(np.abs(calc - q11.loc[J].to_numpy())) < 0.08


def test_mass_scaled_10bh_matches_every_measured_line_but_high_n_q():
    worst_nm = 0.0
    for (v, branch), lines in iso.FB91_10BH_LINES.items():
        for J, obs in lines.items():
            calc = float(iso.bh10_line_wavenumber(v, branch, J, "mass_scaled"))
            d_nm = abs(float(iso.wavenumber_to_air_nm(calc) - iso.wavenumber_to_air_nm(obs)))
            if (v, branch) == (0, "Q") and J >= 18:
                assert calc - obs < -2.0  # Table V Q(18)-Q(20) lie 2.4-5.5 cm-1 blue of scaling
            else:
                worst_nm = max(worst_nm, d_nm)
    assert worst_nm < 0.003
    # The scaled band origin agrees with F&B Table VII.
    origin = iso.dunham_term(iso.DUNHAM_A_FB91, 0, 0, iso.RHO_10BH) - iso.dunham_term(
        iso.DUNHAM_X_FB91, 0, 0, iso.RHO_10BH
    )
    assert float(origin) == pytest.approx(iso.BH10_A_FB91[0].T_v, abs=0.01)


def test_fb91_scheme_uses_measured_lines_and_table_vii_elsewhere():
    wn, src = iso.bh10_q_wavenumbers(25, "fb91")
    meas = iso.FB91_10BH_LINES[(0, "Q")]
    for J in range(1, 26):
        if J in meas:
            assert wn[J] == meas[J] and src[J] == "measured"
            # Table VII reproduces its own lines (Q(20): o-c 0.306 cm-1 in F&B)
            calc = float(iso.bh10_line_wavenumber(0, "Q", J, "fb91"))
            assert abs(calc - meas[J]) < (0.32 if J == 20 else 0.04)
        else:
            assert src[J] == "fb91_constants" and np.isfinite(wn[J])
    assert np.isnan(wn[0])
    # Low-N lines: the two schemes agree within 0.0015 nm through Q(11).
    ms, _ = iso.bh10_q_wavenumbers(25, "mass_scaled")
    d = np.abs(iso.wavenumber_to_air_nm(wn[1:12]) - iso.wavenumber_to_air_nm(ms[1:12]))
    assert np.max(d) < 0.0015
    # 10BH (1-1) R(7)/R(8) of Table V, 433.655 / 433.341 nm (air)
    r = iso.bh10_r11_wavenumbers("fb91")
    assert iso.wavenumber_to_air_nm(r[7]) == pytest.approx(433.6546, abs=2e-4)
    assert iso.wavenumber_to_air_nm(r[8]) == pytest.approx(433.3405, abs=2e-4)


# --- isotopologue="natural" ----------------------------------------------------------


@pytest.mark.parametrize("scheme", ["fb91", "mass_scaled"])
def test_natural_adds_10bh_at_its_share(scheme):
    m11 = BHModel(**SET_C)
    mn = BHModel(**SET_C, isotopologue="natural", bh10_positions=scheme)
    assert mn.extra_param_names == m11.extra_param_names
    T, w = 3000.0, W_FIX
    x = np.linspace(432.9, 434.2, 40001)
    p = (1.0, T, 0.0, w, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    d = mn.full_fit_model(x, *p) - m11.full_fit_model(x, *p)
    s11 = m11.full_fit_model(x, *p)
    # 10BH/11BH emitted power = 0.199/0.801 within the small photon-energy and
    # high-N differences (the partition sums are each isotopologue's own).
    ratio = np.trapezoid(d, x) / np.trapezoid(s11, x)
    assert ratio == pytest.approx(iso.BH10_OVER_BH11, rel=0.01)
    # The difference is the 10BH Q branch, scaled by 0.248 * Z11/Z10.
    scale = iso.BH10_OVER_BH11 * mn.rot_partition(T) / mn.rot_partition(T, "10BH")
    assert np.allclose(d, 1e8 * scale * mn.bh10_spectrum(x, T, w, 0.0), rtol=1e-10, atol=1e-12)
    assert 0.99 < mn.rot_partition(T) / mn.rot_partition(T, "10BH") < 1.01


def test_natural_adds_10bh_r7_r8_at_the_share_of_the_free_amplitudes():
    m11 = BHModel(**SET_C)
    mn = BHModel(**SET_C, isotopologue="natural")
    x = np.linspace(433.2, 433.8, 20001)
    p = (0.0, 3000.0, 0.0, W_FIX, 0.0, 2e-3, 1e-3, 0.0, 0.0, 0.0)
    d = mn.full_fit_model(x, *p) - m11.full_fit_model(x, *p)
    exp = iso.BH10_OVER_BH11 * (
        2e-3 * mn.line_profile(x, mn._bh10["wl_R7"], W_FIX, 0.0)
        + 1e-3 * mn.line_profile(x, mn._bh10["wl_R8"], W_FIX, 0.0)
    )
    assert np.allclose(d, exp, rtol=1e-10, atol=1e-12)


def test_fit_with_natural_model_recovers_temperature():
    truth = BHModel(**SET_C, isotopologue="natural")
    rng = np.random.default_rng(5)
    y = truth.full_fit_model(X, 1.0, 2800.0, 0.004, W_FIX, 0.0, 2e-3, 1e-3, 0.0, 0.0, 0.0)
    y = y / y.max() + 0.1 + 0.05 * (X - BASELINE_REF_NM) + rng.normal(0, 0.005, X.size)
    p, _, _ = _fit(truth, y)
    assert p["T_rot"] == pytest.approx(2800.0, rel=0.03)
    p11, _, _ = _fit(BHModel(**SET_C), y)
    assert p11["T_rot"] != pytest.approx(p["T_rot"], rel=1e-3)


# --- F1/F3, quadratic continuum, H2 R2 only --------------------------------------------


def test_extra_lines_are_bh_width_gaussians_on_the_shifted_grid():
    m0, mf = BHModel(**SET_C), BHModel(**SET_C, extra_lines=("F3", "F1"))
    assert mf.extra_lines == ("F1", "F3")
    p = (*P7, 0.0, 0.0, 0.0)
    d = mf.full_fit_model(X, *p, 3e-3, 1e-3) - m0.full_fit_model(X, *p)
    xs = X + P7[2]
    exp = 3e-3 * m0.line_profile(xs, EXTRA_LINES_NM["F1"], P7[3], 0.0) + 1e-3 * m0.line_profile(
        xs, EXTRA_LINES_NM["F3"], P7[3], 0.0
    )
    assert np.allclose(d, exp, rtol=1e-12, atol=1e-12)
    assert EXTRA_LINES_NM == {"F1": 433.100, "F3": 433.750}


def test_quadratic_continuum_term_and_fit():
    lin, quad = BHModel(**SET_C), BHModel(**dict(SET_C, baseline="quadratic"))
    d = quad.full_fit_model(X, *P7, 0.1, 0.4, 0, 0) - lin.full_fit_model(X, *P7, 0.1, 0, 0)
    assert np.allclose(d, 0.4 * (X - BASELINE_REF_NM) ** 2, rtol=0, atol=1e-12)
    rng = np.random.default_rng(3)
    y = quad.full_fit_model(X, *P7, 0.0, 0.0, 0.0, 0.0)
    y = y / y.max() + 0.1 + 0.1 * (X - BASELINE_REF_NM) + 0.8 * (X - BASELINE_REF_NM) ** 2
    y = y + rng.normal(0, 0.004, X.size)
    p, _, _ = _fit(quad, y)
    assert p["base_curv"] == pytest.approx(0.8, abs=0.15)
    assert p["T_rot"] == pytest.approx(P7[1], rel=0.03)


def test_h2_r2_only_has_one_line():
    m = BHModel(**dict(SET_C, h2_lines="R2"))
    assert m.extra_param_names == ("base_slope", "I_H2_R2")
    full = BHModel(**SET_C)
    assert np.allclose(m.full_fit_model(X, *P7, 0.1, 2e-3), full.full_fit_model(X, *P7, 0.1, 0.0, 2e-3))


# --- two rotational components -------------------------------------------------------


def test_two_components_nest_the_one_temperature_model():
    m1, m2 = BHModel(**SET_C), BHModel(**SET_C, rot_components=2)
    e = (0.1, 1e-3, 1e-3)
    y1 = m1.full_fit_model(X, *P7, *e)
    assert np.allclose(m2.full_fit_model(X, *P7, *e, 9000.0, 0.0), y1, rtol=1e-12, atol=1e-12)
    assert np.allclose(m2.full_fit_model(X, *P7, *e, P7[1], 0.6), y1, rtol=1e-12, atol=1e-12)
    # frac2 = 1: the T2 population, carrying the same total population as C at T1
    T2 = 7000.0
    p_t2 = (P7[0] * m1.rot_partition(P7[1]) / m1.rot_partition(T2), T2, *P7[2:])
    assert np.allclose(m2.full_fit_model(X, *P7, *e, T2, 1.0), m1.full_fit_model(X, *p_t2, *e))


def test_two_component_fit_finds_a_hot_tail():
    m2 = BHModel(**SET_C, rot_components=2)
    y = m2.full_fit_model(X, 1.0, 2000.0, 0.004, W_FIX, 0.0, 0, 0, 0, 0, 0, 8000.0, 0.3)
    y = y / y.max() + np.random.default_rng(2).normal(0, 0.002, X.size)
    fitr = BHFitter(_Vis(X, y), m2, nm_window=(X[0], X[-1]), base_tight=True,
                    w_inst_default=W_FIX, fix_w_inst=True,
                    p0=(1.0, 2500.0, 0.0, W_FIX, 0.0, 1e-3, 1e-3, 0.0, 1e-3, 1e-3, 7000.0, 0.2))
    fitr.set_bounds(lower={"base": -0.05}, upper={"base": 0.5})
    p = dict(zip(fitr.param_names, fitr.fit(0, 0)["params"]))
    # the labels may swap: compare the cold and hot components
    cold, hot = sorted((p["T_rot"], p["T_rot2"]))
    hot_frac = p["frac2"] if p["T_rot2"] > p["T_rot"] else 1.0 - p["frac2"]
    assert cold == pytest.approx(2000.0, rel=0.1)
    assert hot == pytest.approx(8000.0, rel=0.25)
    assert hot_frac == pytest.approx(0.3, abs=0.1)


# --- excluded ranges --------------------------------------------------------------------


def test_exclude_ranges_validate_and_report():
    with pytest.raises(ValueError):
        BHFitter(None, BHModel(), exclude_nm=[(433.3, 433.2)])
    with pytest.raises(ValueError):
        BHFitter(None, BHModel(), exclude_nm=[(433.2, 433.3)], exclude_frame="data")
    f = BHFitter(None, BHModel(), exclude_nm=[(433.245, 433.28)])
    c = f.describe_constraints()
    assert c["exclude_nm"] == [[433.245, 433.28]] and c["exclude_frame"] == "rest"


@pytest.mark.parametrize("frame", ["rest", "measured"])
def test_excluded_range_is_left_out_of_the_fit(frame):
    model = BHModel(**dict(SET_C, h2_lines="R2"))
    dx = 0.015
    rng = np.random.default_rng(11)
    y = model.full_fit_model(X, 1.0, 3000.0, dx, W_FIX, 0.0, 2e-3, 1e-3, 0.0, 0.0)
    y = y / y.max() + 0.1 + rng.normal(0, 0.003, X.size)
    # an unmodelled line at rest 433.262 nm (measured 433.262 - dx)
    y = y + 0.4 * np.exp(-0.5 * ((X + dx - 433.262) / 0.008) ** 2)
    lo, hi = (433.245, 433.280)
    rng_meas = (lo - dx, hi - dx) if frame == "measured" else (lo, hi)
    p, r, _ = _fit(model, y, exclude_nm=[rng_meas], exclude_frame=frame)
    x_rest = r["x"] + (p["dx"] if frame == "rest" else dx)
    assert not np.any((x_rest >= lo + 1e-9) & (x_rest <= hi - 1e-9))
    assert r["x"].size < X.size and r["yfit"].size == r["x"].size
    assert p["T_rot"] == pytest.approx(3000.0, rel=0.04)
    p_all, _, _ = _fit(model, y)
    assert abs(p_all["T_rot"] / 3000.0 - 1) > abs(p["T_rot"] / 3000.0 - 1)
