"""Tests for the BHModel sensitivity switches (NME revision, Reviewer 1 items 7.2/7.3).

Each switch defaults to the production model; these tests pin the defaults to
an inline copy of the original formula and check what each switch changes.
"""

from importlib import resources

import numpy as np
import pandas as pd
import pytest
import scipy.constants as spc
from scipy.integrate import trapezoid

from bh_molecule.constants import BH_A, BH_A_FB91, BH_X_FB91
from bh_molecule.dataio import load_band_wavelengths, load_v00_wavelengths
from bh_molecule.physics import BHModel, Branch


X_WIN = np.linspace(432.8, 434.2, 3000)
PARAMS = dict(C=1.0, dx=0.01, w_inst=0.0187, base=0.001, I_R7=0.002, I_R8=0.003)


def _reference_q_spectrum(x, C, T_rot, w_inst):
    """Production Q-branch model as first written: Huber-Herzberg energies,
    v' = 0..2 co-added at the 0-0 positions, each band normalised at N = 1."""
    m = BHModel()
    wl = load_v00_wavelengths()["Q"].to_numpy()
    y = np.zeros_like(x)
    for v in range(3):
        E_ref = BHModel.energy(v, 1, BH_A)
        for N2 in range(1, 23):
            dF = BHModel.energy(v, N2, BH_A) - E_ref
            pop = C * m.g_as * (2 * N2 + 1) * np.exp(-dF / (T_rot * m.K2wn))
            photon_E = spc.h * spc.c / (wl[N2] * 1e-9) / (4 * np.pi)
            y += photon_E * pop * BHModel.A_coeff(v, N2, N2) * m.line_profile(
                x, wl[N2], w_inst, 0.0
            )
    return y


def _fb_line(v, J, branch, parity):
    """Line position from F&B Table III constants (A minus X), cm^-1."""
    J_up = {"P": J - 1, "Q": J, "R": J + 1}[branch]
    return BH_A_FB91[v].term(J_up, parity) - BH_X_FB91[v].term(J)


# --- defaults ---------------------------------------------------------------


def test_defaults_reproduce_original_model():
    m = BHModel()
    assert (m.upper_energies, m.v_max, m.band_positions, m.T_vib) == (
        "huber_herzberg",
        2,
        "v00",
        None,
    )
    for T in (2000.0, 4000.0, 6000.0):
        y = m.spectrum(X_WIN, 1.0, T, 0.0187, 0.0, Branch.Q)
        assert np.allclose(y, _reference_q_spectrum(X_WIN, 1.0, T, 0.0187), rtol=1e-12)


def test_explicit_defaults_are_identical():
    y0 = BHModel().full_fit_model(X_WIN, T_rot=4000.0, **PARAMS)
    y1 = BHModel(
        upper_energies="huber_herzberg", v_max=2, band_positions="v00", T_vib=None
    ).full_fit_model(X_WIN, T_rot=4000.0, **PARAMS)
    assert np.array_equal(y0, y1)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(upper_energies="dunham"),
        dict(band_positions="mixed"),
        dict(v_max=3),
        dict(T_vib=0.0),
    ],
)
def test_invalid_switch_values_raise(kwargs):
    with pytest.raises(ValueError):
        BHModel(**kwargs)


# --- switch (i): Fernando-Bernath upper-state energies -----------------------


@pytest.mark.parametrize("v", [0, 1, 2])
def test_fb_constants_reproduce_table_I_q_lines_on_f_levels(v):
    """Transcription check: Table III constants with the f sign give the
    observed Table I Q lines (O-C printed in 1e-3 cm^-1, at most 39 here)."""
    with resources.files("bh_molecule._resources").joinpath(f"11BH_v{v}{v}.csv").open() as f:
        tab = pd.read_csv(f, comment="#")
    rows = tab[np.isfinite(tab["Q"]) & (tab["J"] <= 24)]
    oc_f = np.array([r.Q - _fb_line(v, int(r.J), "Q", "f") for r in rows.itertuples()])
    oc_e = np.array([r.Q - _fb_line(v, int(r.J), "Q", "e") for r in rows.itertuples()])
    assert np.max(np.abs(oc_f)) < 0.045
    assert np.max(np.abs(oc_e)) > 4.0  # the e component is the wrong level


def test_fb_upper_energy_uses_f_for_q_and_e_for_p_r():
    m = BHModel(upper_energies="fernando_bernath")
    for N in (1, 5, 17):
        assert m.upper_energy(0, N, Branch.Q) == pytest.approx(BH_A_FB91[0].term(N, "f"))
        assert m.upper_energy(1, N, Branch.R) == pytest.approx(BH_A_FB91[1].term(N, "e"))
    # Huber-Herzberg path unchanged
    assert BHModel().upper_energy(0, 5) == BHModel.energy(0, 5, BH_A)


def test_fb_energies_change_weights_only_slightly():
    """Population ratio new/old at 4 kK stays within 1 % up to N = 17 (v = 0)."""
    hh, fb = BHModel(), BHModel(upper_energies="fernando_bernath")
    for N in range(2, 18):
        d_new = fb.upper_energy(0, N) - fb.upper_energy(0, 1)
        d_old = hh.upper_energy(0, N) - hh.upper_energy(0, 1)
        ratio = np.exp(-(d_new - d_old) / (4000.0 * hh.K2wn))
        assert abs(ratio - 1.0) < 0.01
    y_hh = hh.spectrum(X_WIN, 1.0, 4000.0, 0.0187, 0.0, Branch.Q)
    y_fb = fb.spectrum(X_WIN, 1.0, 4000.0, 0.0187, 0.0, Branch.Q)
    assert not np.array_equal(y_hh, y_fb)


# --- switch (ii): 0-0 only ---------------------------------------------------


def test_v_max_zero_is_the_single_0_0_band_with_nuisance_lines_kept():
    m0, m = BHModel(v_max=0), BHModel()
    y0 = m0.spectrum(X_WIN, 1.0, 4000.0, 0.0187, 0.0, Branch.Q)
    assert np.array_equal(y0, m.spectrum(X_WIN, 1.0, 4000.0, 0.0187, 0.0, Branch.Q, v_max=0))
    assert y0.max() < 0.5 * m.spectrum(X_WIN, 1.0, 4000.0, 0.0187, 0.0, Branch.Q).max()
    # With C = 0 only the R(7)/R(8) features and the baseline remain, unchanged.
    nuis = dict(PARAMS, C=0.0)
    assert np.array_equal(
        m0.full_fit_model(X_WIN, T_rot=4000.0, **nuis),
        m.full_fit_model(X_WIN, T_rot=4000.0, **nuis),
    )


# --- switch (iii): own band positions and a vibrational temperature ----------


def test_vib_weight_is_boltzmann_in_fb_band_origins():
    m = BHModel(band_positions="own", T_vib=3900.0)
    for v in (1, 2):
        G = BH_A_FB91[v].T_v - BH_A_FB91[0].T_v
        assert m.vib_weight(v) == pytest.approx(np.exp(-G / (3900.0 * m.K2wn)))
    assert BHModel().vib_weight(1) == 1.0


def test_own_positions_put_1_1_q_lines_at_measured_wavelengths():
    T_vib = 10000.0
    m = BHModel(band_positions="own", T_vib=T_vib)
    tab0, tab1 = load_band_wavelengths(0), load_band_wavelengths(1)
    x = np.linspace(433.0, 437.0, 40001)
    y = m.spectrum(x, 1.0, 4000.0, 0.0187, 0.0, Branch.Q, N2_max=1)
    # N2_max = 1: one Q(1) line per band; the 2-2 Q(1) lies beyond the grid.
    l00, l11 = tab0.Q[1], tab1.Q[1]
    near = lambda c: np.abs(x - c) < 0.1  # noqa: E731
    a00 = trapezoid(y[near(l00)], x[near(l00)])
    a11 = trapezoid(y[near(l11)], x[near(l11)])
    expected = (
        m.vib_weight(1)
        * BHModel.A_coeff(1, 1, 1)
        / BHModel.A_coeff(0, 1, 1)
        * (l00 / l11)  # photon energy
    )
    assert a11 / a00 == pytest.approx(expected, rel=1e-6)
    assert x[near(l11)][np.argmax(y[near(l11)])] == pytest.approx(l11, abs=2e-4)


def test_own_positions_skip_unobserved_lines_and_leave_433nm_window_0_0_only():
    m = BHModel(band_positions="own", T_vib=3900.0)
    # 2-2 Q is observed to J = 13: N2_max = 22 adds nothing beyond it.
    x22 = np.linspace(443.0, 447.5, 9000)
    y_all = m.spectrum(x22, 1.0, 4000.0, 0.0187, 0.0, Branch.Q)
    y_13 = m.spectrum(x22, 1.0, 4000.0, 0.0187, 0.0, Branch.Q, N2_max=13)
    assert np.allclose(y_all, y_13, rtol=1e-12, atol=0.0)
    # In the 433 nm fit window the 1-1 and 2-2 Q branches are absent, so the
    # mixture reduces to the 0-0-only model there.
    y_mix = m.full_fit_model(X_WIN, T_rot=4000.0, **PARAMS)
    y_00 = BHModel(v_max=0).full_fit_model(X_WIN, T_rot=4000.0, **PARAMS)
    assert np.allclose(y_mix, y_00, rtol=1e-12, atol=1e-300)
