"""Wavelength calibration utilities for Vis133M (no pandas, numpy + scipy only)."""

import csv
import numpy as np
from scipy.optimize import curve_fit


def load_wavecal_csv(
    path: str,
    n_channels: int,
    n_pixels: int | None = None,
) -> np.ndarray:
    """Load wavelength calibration from CSV using stdlib csv (no pandas).

    CSV format: first column = channel index (0-based or 1-based), remaining
    columns = wavelength [nm] per pixel. Returns shape (n_channels, n_pixels).

    When n_pixels is None, loads all available wavelength columns from the CSV.
    This allows using a CSV with fewer pixels than the cube (e.g. 1024 vs 2048);
    the formula fit from the CSV can then be extrapolated to the cube's pixel count.

    Parameters
    ----------
    path : str
        Path to CSV file.
    n_channels : int
        Expected number of channels.
    n_pixels : int | None
        Expected number of pixels per channel. If None, use all wavelength
        columns present in the CSV (convenient when cube has more pixels).

    Returns
    -------
    wl_nm : ndarray
        Shape (n_channels, n_pixels), wavelength in nm.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < n_channels:
        raise ValueError(f"CSV has {len(rows)} rows, need at least {n_channels}")

    ch_col = []
    wl_rows = []
    for i, row in enumerate(rows[:n_channels]):
        if not row:
            raise ValueError(f"Empty row {i}")
        try:
            ch = int(float(row[0]))
        except (ValueError, TypeError):
            raise ValueError(f"Row {i}: channel col must be numeric, got {row[0]}")
        ch_col.append(ch)
        vals = []
        max_cols = len(row) if n_pixels is None else min(n_pixels + 1, len(row) + 1)
        for j in range(1, max_cols):
            try:
                v = float(row[j])
            except (ValueError, TypeError, IndexError):
                raise ValueError(f"Row {i} col {j}: invalid wavelength")
            vals.append(v)
        if n_pixels is not None and len(vals) != n_pixels:
            raise ValueError(f"Row {i}: expected {n_pixels} wavelength cols, got {len(vals)}")
        wl_rows.append(vals)

    ch_arr = np.array(ch_col)
    is_zero = np.all(ch_arr == np.arange(n_channels))
    is_one = np.all(ch_arr == np.arange(1, n_channels + 1))
    if not (is_zero or is_one):
        raise ValueError("Wavecal col 0 must be channel indices 0..C-1 (or 1..C)")

    wl = np.array(wl_rows, dtype=float)
    if np.isnan(wl).any():
        raise ValueError("Wavecal contains NaN")
    return wl


def csv_to_linear_formulas(wl_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert wavelength matrix to per-channel linear formulas λ(x) = a*x + b.

    Parameters
    ----------
    wl_nm : ndarray
        Shape (n_channels, n_pixels), wavelength per pixel per channel.

    Returns
    -------
    slopes : ndarray
        Shape (n_channels,), slope a_c for each channel.
    intercepts : ndarray
        Shape (n_channels,), intercept b_c for each channel.
    """
    wl = np.asarray(wl_nm, dtype=float)
    n_channels, n_pixels = wl.shape
    x = np.arange(n_pixels, dtype=float)

    slopes = np.full(n_channels, np.nan)
    intercepts = np.full(n_channels, np.nan)

    for c in range(n_channels):
        y = wl[c]
        mask = np.isfinite(y)
        if mask.sum() < 2:
            continue
        p = np.polyfit(x[mask], y[mask], 1)
        slopes[c] = float(p[0])
        intercepts[c] = float(p[1])

    return slopes, intercepts


def _gaussian_plus_baseline(x: np.ndarray, amp: float, mu: float, sigma: float, base: float) -> np.ndarray:
    """Gaussian + constant baseline model."""
    return base + amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fit_peak_gaussian(
    spectrum: np.ndarray,
    pixel_window: tuple[int, int] | None = None,
    baseline_zero: bool = False,
) -> tuple[float, float, float] | None:
    """Fit strongest peak in 1D spectrum with Gaussian + baseline.

    Parameters
    ----------
    spectrum : ndarray
        1D array of intensities (pixel index = x).
    pixel_window : tuple of int, optional
        (start, stop) pixel indices to search. If None, use full spectrum.
    baseline_zero : bool, optional
        If True, subtract minimum before fitting (default False).

    Returns
    -------
    result : tuple or None
        (peak_pixel, peak_amplitude, sigma) if fit succeeds, else None.
    """
    spec = np.asarray(spectrum, dtype=float).ravel()
    if pixel_window is not None:
        lo, hi = pixel_window
        spec = spec[lo:hi].copy()
        x_offset = float(lo)
    else:
        x_offset = 0.0

    if baseline_zero:
        spec = spec - np.min(spec)

    x = np.arange(spec.size, dtype=float)

    # Initial guess: max at peak, amplitude = max - min, sigma ~ few pixels
    idx_max = int(np.argmax(spec))
    amp0 = float(spec[idx_max] - np.min(spec)) or 1.0
    mu0 = float(idx_max)
    sigma0 = max(1.0, spec.size / 20.0)
    base0 = float(np.min(spec))

    bounds = (
        (0.0, 0.0, 0.1, -np.inf),
        (np.inf, spec.size - 0.5, spec.size, np.inf),
    )

    try:
        popt, _ = curve_fit(
            _gaussian_plus_baseline,
            x,
            spec,
            p0=(amp0, mu0, sigma0, base0),
            bounds=bounds,
            maxfev=2000,
        )
    except (RuntimeError, ValueError):
        return None

    amp, mu, sigma, _ = popt
    # mu is relative to window start
    peak_pixel = float(mu) + x_offset
    return (peak_pixel, float(amp), float(sigma))


def measure_peak_from_cube(
    cube: np.ndarray,
    channel: int,
    pixel_window: tuple[int, int] | None = None,
    average_frames: bool = True,
    baseline_zero: bool = False,
) -> tuple[float, float, float] | None:
    """Extract 1D spectrum from FITS cube and fit strongest peak.

    Parameters
    ----------
    cube : ndarray
        Shape (F, C, P).
    channel : int
        Channel index.
    pixel_window : tuple of int, optional
        (start, stop) pixel indices. If None, use full extent.
    average_frames : bool, optional
        If True, average over frames; else use first frame (default True).
    baseline_zero : bool, optional
        Subtract minimum before fitting (default False).

    Returns
    -------
    result : tuple or None
        (peak_pixel, peak_amplitude, sigma) if fit succeeds.
    """
    stack = cube[:, channel, :]
    if average_frames:
        spec = np.mean(stack, axis=0)
    else:
        spec = stack[0]

    return fit_peak_gaussian(spec, pixel_window=pixel_window, baseline_zero=baseline_zero)


def compute_wavelength_shift(
    peak_pixel_old: float,
    peak_pixel_new: float,
    slope: float,
) -> float:
    """Compute wavelength shift Δλ from pixel shift and dispersion.

    Δλ = a_c * (peak_pixel_new - peak_pixel_old)

    Parameters
    ----------
    peak_pixel_old : float
        Peak pixel in reference (old) dataset.
    peak_pixel_new : float
        Peak pixel in new dataset.
    slope : float
        Dispersion a_c (nm/pixel) for the channel.

    Returns
    -------
    delta_lambda : float
        Wavelength shift in nm.
    """
    delta_pixel = peak_pixel_new - peak_pixel_old
    return slope * delta_pixel
