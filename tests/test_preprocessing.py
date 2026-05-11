"""Synthetic tests for batch preprocessing helpers (no FITS files)."""

import numpy as np
import pytest

from bh_molecule.plotting.batch_fit_plots import normalize_curves_for_grid
from bh_molecule.workflows.preprocessing import (
    mean_background_spectrum,
    subtract_background_from_row,
)


def test_subtract_background_is_signal_minus_background():
    signal = np.array([10.0, 12.0, 8.0])
    background = np.array([1.0, 2.0, 3.0])
    out = subtract_background_from_row(signal, background)
    assert np.allclose(out, [9.0, 10.0, 5.0])
    assert not np.allclose(out, subtract_background_from_row(background, signal))


def test_mean_background_spectrum_averages_frames():
    # F=4, C=2, P=3
    cube = np.zeros((4, 2, 3))
    cube[:, 0, :] = 1.0
    cube[0, 0, :] = 10.0
    cube[1, 0, :] = 4.0
    mean_bg = mean_background_spectrum(cube, background_frames=[0, 1], channel=0)
    assert np.allclose(mean_bg, [7.0, 7.0, 7.0])


def test_mean_background_frame_index_zero_based():
    cube = np.arange(2 * 1 * 2, dtype=float).reshape(2, 1, 2)
    m0 = mean_background_spectrum(cube, [0], channel=0)
    m1 = mean_background_spectrum(cube, [1], channel=0)
    assert np.allclose(m0, [0.0, 1.0])
    assert np.allclose(m1, [2.0, 3.0])


def test_mean_background_rejects_bad_frame():
    cube = np.zeros((2, 1, 2))
    with pytest.raises(IndexError):
        mean_background_spectrum(cube, [5], channel=0)


def test_normalize_curves_for_grid_uses_data_range():
    y = np.array([10.0, 20.0, 15.0])
    yfit = np.array([10.0, 18.0, 16.0])
    y_n, yfit_n = normalize_curves_for_grid(y, yfit)
    assert np.allclose(y_n, [0.0, 1.0, 0.5])
    assert yfit_n is not None
    assert np.allclose(yfit_n, [0.0, 0.8, 0.6])


def test_normalize_curves_low_amplitude_fit_not_squashed():
    """Joint min/max used to squash fit; data-only range keeps overlay visible."""
    y = np.array([100.0, 200.0, 150.0])
    yfit = np.array([1.0, 2.0, 1.5])
    y_n, yfit_n = normalize_curves_for_grid(y, yfit)
    assert yfit_n is not None
    assert float(np.max(yfit_n)) < 0.02
    assert float(np.max(y_n)) == pytest.approx(1.0)


def test_baseline_subtract_then_normalize_toy():
    """Normalization conceptually happens on already baseline-processed data."""
    raw = np.array([5.0, 105.0, 55.0])
    bg = np.array([5.0, 5.0, 5.0])
    corrected = subtract_background_from_row(raw, bg)
    y_n, _ = normalize_curves_for_grid(corrected, None)
    assert np.isclose(np.min(y_n), 0.0)
    assert np.isclose(np.max(y_n), 1.0)
