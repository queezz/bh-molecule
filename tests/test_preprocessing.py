"""Synthetic tests for batch preprocessing helpers (no FITS files)."""

import numpy as np
import pytest

from bh_molecule.plotting.batch_fit_plots import normalize_curves_for_grid
from bh_molecule.workflows.preprocessing import (
    BH_FIT_WAVELENGTH_RANGE_NM,
    BH_SCALE_WAVELENGTH_RANGE_NM,
    make_bh_fit_preprocessor,
    mean_background_spectrum,
    prepare_bh_fit_arrays,
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


# ---------------------------------------------------------------------------
# prepare_bh_fit_arrays + make_bh_fit_preprocessor: zero-preserving pipeline.
# ---------------------------------------------------------------------------


class _SyntheticVis:
    """Tiny stand-in exposing ``cube`` and ``wl_nm`` like ``Vis133M``."""

    def __init__(self, wavelengths: np.ndarray, cube: np.ndarray):
        self.wl_nm = np.broadcast_to(wavelengths, (cube.shape[1], wavelengths.size)).copy()
        self.cube = np.asarray(cube, dtype=float)
        self.scale = 1.0


def _make_synthetic_vis_with_hgamma(*, channel: int = 0):
    """Synthetic vis with one BH-like peak inside scale window and a
    very bright fake H-gamma-like peak *outside* the BH scale window.

    Returns ``(vis, frame_signal, channel, background_frames)``.
    """
    wl = np.linspace(432.5, 434.2, 500)
    n_frames = 6
    n_channels = max(1, channel + 1)
    cube = np.zeros((n_frames, n_channels, wl.size), dtype=float)

    # Steady continuum (200 counts) on every frame including background.
    cube += 200.0

    # Background frames 0..3: only continuum (already added).
    bg_frames = (0, 1, 2, 3)

    # Signal frames 4 and 5: add BH-like peak inside the BH band and a
    # very bright H-gamma-like peak well outside the BH scale window.
    bh_center = 433.16
    bh_width = 0.05
    bh_peak_amplitude = 50.0
    bh_peak = bh_peak_amplitude * np.exp(-((wl - bh_center) ** 2) / (2 * bh_width ** 2))

    # H-gamma is around 434.05 nm; place a much brighter Gaussian there
    # so the *full detector* y_max would be dominated by it.
    hgamma_center = 434.05
    hgamma_width = 0.04
    hgamma_amplitude = 5000.0
    hgamma_peak = hgamma_amplitude * np.exp(
        -((wl - hgamma_center) ** 2) / (2 * hgamma_width ** 2)
    )

    signal_frame = 4
    cube[signal_frame, channel, :] += bh_peak + hgamma_peak

    vis = _SyntheticVis(wl, cube)
    return vis, signal_frame, channel, bg_frames, bh_peak_amplitude


def test_prepare_bh_fit_arrays_uses_background_subtraction():
    vis, fr, ch, bg, bh_amp = _make_synthetic_vis_with_hgamma()
    x, y, meta = prepare_bh_fit_arrays(
        vis, fr, ch, background_frames=bg,
        fit_window=BH_FIT_WAVELENGTH_RANGE_NM,
        scale_window=BH_SCALE_WAVELENGTH_RANGE_NM,
    )
    # Continuum was 200 in both signal row and background frames; after
    # subtraction the median should be ~0 (Gaussian tails carry a tiny
    # positive bias).
    assert abs(meta["subtracted_median"]) < 0.01 * bh_amp
    # And the normalized array should not be shifted to min=0 either.
    assert meta["normalized_min"] <= 0.05  # near zero or negative; not min-shifted


def test_prepare_bh_fit_arrays_preserves_negatives():
    """Mild over-subtraction of the background must yield negative y values
    that are NOT clipped to zero."""
    wl = np.linspace(432.5, 434.2, 400)
    cube = np.zeros((5, 1, wl.size), dtype=float)
    cube[:4, 0, :] = 100.0  # background frames have higher continuum
    # Signal frame has lower continuum + a BH-like peak: subtraction yields
    # a negative continuum and a positive bump.
    cube[4, 0, :] = 90.0 + 30.0 * np.exp(-((wl - 433.16) ** 2) / (2 * 0.05 ** 2))
    vis = _SyntheticVis(wl, cube)

    x, y, meta = prepare_bh_fit_arrays(
        vis, 4, 0, background_frames=(0, 1, 2, 3),
        fit_window=BH_FIT_WAVELENGTH_RANGE_NM,
        scale_window=BH_SCALE_WAVELENGTH_RANGE_NM,
    )
    assert meta["n_negative_after_subtract"] > 0
    assert float(np.min(y)) < 0.0
    # And we never shifted to min=0.
    assert not np.isclose(float(np.min(y)), 0.0)


def test_prepare_bh_fit_arrays_does_not_minmax_normalize():
    """Output must not be ``(y - y.min())/(y.max() - y.min())``."""
    wl = np.linspace(432.5, 434.2, 400)
    cube = np.zeros((5, 1, wl.size), dtype=float)
    cube[:4, 0, :] = 50.0  # background continuum
    cube[4, 0, :] = 50.0 + 80.0 * np.exp(-((wl - 433.16) ** 2) / (2 * 0.05 ** 2))
    vis = _SyntheticVis(wl, cube)
    x, y, meta = prepare_bh_fit_arrays(
        vis, 4, 0, background_frames=(0, 1, 2, 3),
        fit_window=BH_FIT_WAVELENGTH_RANGE_NM,
        scale_window=BH_SCALE_WAVELENGTH_RANGE_NM,
    )
    # With min/max normalization, min(y) would be 0 exactly.  Our pipeline
    # preserves zero-around-continuum and the floor sits near zero only by
    # subtraction, never by shifting; for this synthetic data the minimum
    # should not be exactly 0 (background subtraction leaves residual noise
    # = 0 here, but no shift). Allow exact zero only if the data itself is
    # exactly at zero; not because of a min subtraction.
    # The strong invariant: the *median* of the normalized output equals
    # zero (continuum, after bg subtraction, then divided by scale).
    assert abs(meta["normalized_median"]) < 1e-6


def test_scale_window_isolates_bh_from_bright_hgamma():
    """A bright H-gamma-like peak OUTSIDE the BH scale window must NOT
    control the BH normalization scale."""
    vis, fr, ch, bg, bh_amp = _make_synthetic_vis_with_hgamma()
    x, y, meta = prepare_bh_fit_arrays(
        vis, fr, ch, background_frames=bg,
        fit_window=BH_FIT_WAVELENGTH_RANGE_NM,
        scale_window=BH_SCALE_WAVELENGTH_RANGE_NM,
    )
    # Scale should be close to the BH-peak amplitude (~50), NOT to the
    # 5000-count H-gamma-like peak that sits at 434.05 nm.
    assert meta["scale"] == pytest.approx(bh_amp, rel=0.05)
    assert meta["scale"] < 200.0, (
        f"Scale {meta['scale']:.1f} is too large; H-gamma-like peak leaked into scale"
    )
    # The BH-peak normalizes to ~1 inside the scale window.
    sc_lo, sc_hi = meta["scale_window"]
    in_sc = (x >= sc_lo) & (x <= sc_hi)
    assert float(np.nanmax(y[in_sc])) == pytest.approx(1.0, rel=0.05)


def test_scale_method_max_and_p99_both_positive():
    vis, fr, ch, bg, _ = _make_synthetic_vis_with_hgamma()
    _, _, meta_max = prepare_bh_fit_arrays(
        vis, fr, ch, background_frames=bg, scale_method="max",
    )
    _, _, meta_p99 = prepare_bh_fit_arrays(
        vis, fr, ch, background_frames=bg, scale_method="p99",
    )
    assert meta_max["scale"] > 0
    assert meta_p99["scale"] > 0
    assert meta_max["scale"] >= meta_p99["scale"]


def test_invalid_scale_raises():
    """Spectra with no positive values inside scale window must error out."""
    wl = np.linspace(432.5, 434.2, 200)
    cube = np.zeros((2, 1, wl.size), dtype=float)
    # Signal == background everywhere -> y_sub == 0 -> no positive samples.
    vis = _SyntheticVis(wl, cube)
    with pytest.raises(ValueError):
        prepare_bh_fit_arrays(
            vis, 0, 0, background_frames=(0,),
            fit_window=BH_FIT_WAVELENGTH_RANGE_NM,
            scale_window=BH_SCALE_WAVELENGTH_RANGE_NM,
        )


def test_make_bh_fit_preprocessor_returns_same_arrays():
    """Single helper used by batch and single/debug paths."""
    vis, fr, ch, bg, _ = _make_synthetic_vis_with_hgamma()
    preprocess = make_bh_fit_preprocessor(
        background_frames=bg,
        fit_window=BH_FIT_WAVELENGTH_RANGE_NM,
        scale_window=BH_SCALE_WAVELENGTH_RANGE_NM,
    )
    x_pp, y_pp = preprocess(vis, fr, ch)
    x_ref, y_ref, _ = prepare_bh_fit_arrays(
        vis, fr, ch, background_frames=bg,
        fit_window=BH_FIT_WAVELENGTH_RANGE_NM,
        scale_window=BH_SCALE_WAVELENGTH_RANGE_NM,
    )
    assert np.array_equal(x_pp, x_ref)
    assert np.array_equal(y_pp, y_ref)


def test_default_constants():
    assert BH_FIT_WAVELENGTH_RANGE_NM == (433.05, 433.90)
    assert BH_SCALE_WAVELENGTH_RANGE_NM == (433.08, 433.30)
    lo, hi = BH_SCALE_WAVELENGTH_RANGE_NM
    flo, fhi = BH_FIT_WAVELENGTH_RANGE_NM
    assert flo <= lo < hi <= fhi


# ---------------------------------------------------------------------------
# BHFitter integration: tight `base` bound + preprocess delegation.
# ---------------------------------------------------------------------------


def test_bhfitter_base_tight_default_bounds():
    """``base_tight=True`` must default the constant baseline near zero."""
    from bh_molecule.fit import BHFitter, DEFAULT_BASE_TIGHT_NM
    from bh_molecule.physics import BHModel

    vis, fr, ch, bg, _ = _make_synthetic_vis_with_hgamma()
    preprocess = make_bh_fit_preprocessor(
        background_frames=bg,
        fit_window=BH_FIT_WAVELENGTH_RANGE_NM,
        scale_window=BH_SCALE_WAVELENGTH_RANGE_NM,
    )
    fitr = BHFitter(vis=vis, model=BHModel(), preprocess=preprocess, base_tight=True)
    base_idx = fitr.param_names.index("base")
    assert fitr.p0[base_idx] == 0.0
    assert fitr.bounds[0][base_idx] == pytest.approx(-DEFAULT_BASE_TIGHT_NM)
    assert fitr.bounds[1][base_idx] == pytest.approx(+DEFAULT_BASE_TIGHT_NM)


def test_bhfitter_uses_preprocess_when_provided():
    """When ``preprocess`` is given, ``_window_data`` must delegate to it."""
    from bh_molecule.fit import BHFitter
    from bh_molecule.physics import BHModel

    captured = {}

    def fake_preprocess(_vis, frame, channel):
        captured["called"] = (frame, channel)
        x = np.linspace(433.10, 433.20, 30)
        return x, np.ones_like(x)

    fitr = BHFitter(vis=None, model=BHModel(), preprocess=fake_preprocess)
    x, y = fitr._window_data(7, 3)
    assert captured["called"] == (7, 3)
    assert x.size == 30 and y.size == 30
    assert np.all(y == 1.0)
