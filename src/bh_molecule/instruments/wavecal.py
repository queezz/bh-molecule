"""Wavelength calibration utilities for Vis133M (no pandas, numpy + scipy only)."""

from __future__ import annotations

import csv
import json
from importlib import resources
from typing import Mapping, Any

import numpy as np
from astropy.io import fits
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


def _header_cw_candidates(header: Mapping[str, Any]) -> list[float]:
    """Internal helper: extract plausible central wavelength values from a FITS header.

    This searches a small set of commonly used keyword names. All values are
    converted to ``float`` and returned in the order they are found.
    """
    keys_nm = [
        "CWL",  # generic "central wavelength"
        "CENWAVE",
        "WAVELEN",
        "LAM_CEN",
    ]
    keys_wcs = [
        "CRVAL1",  # WCS start wavelength (often nm or Å depending on header)
    ]

    values: list[float] = []
    for k in keys_nm:
        if k in header:
            try:
                values.append(float(header[k]))
            except Exception:
                continue
    for k in keys_wcs:
        if k in header:
            try:
                values.append(float(header[k]))
            except Exception:
                continue
    return values


def get_cw_from_header(header: Mapping[str, Any]) -> float | None:
    """Return central wavelength (CW) in nm from a FITS header, if available.

    The search is heuristic and checks a small set of commonly used keyword
    names (e.g. ``CWL``, ``CENWAVE``, ``CRVAL1``). If no suitable value is
    found, ``None`` is returned.
    """
    vals = _header_cw_candidates(header)
    return vals[0] if vals else None


def compute_calibration_from_reference(
    wavcal_csv: str,
    fits_path: str,
    *,
    channel: int = 0,
    degree: int = 2,
    pixel_reference: int | None = None,
) -> dict:
    """Compute a pixel→wavelength polynomial from a reference CSV + FITS cube.

    This is intended for *offline* use to derive a compact wavelength
    calibration formula that can be reused at runtime without needing the
    original CSV file.

    Parameters
    ----------
    wavcal_csv :
        Path to a *Vis133M instrument* CSV file containing per-channel wavelength
        calibration (dispersion) in the
        format expected by :func:`load_wavecal_csv`.
    fits_path :
        Path to a reference FITS cube acquired with the spectrometer.
    channel :
        Channel index to use when fitting the polynomial (default 0).
    degree :
        Polynomial degree for the pixel→wavelength mapping. The default (2)
        usually provides enough flexibility for mild non-linearity.
    pixel_reference :
        Optional reference pixel index. When ``None`` (default), the centre of
        the calibrated pixel range for the chosen channel is used.

    Returns
    -------
    dict
        A dictionary compatible with ``bh_wavecal.json`` containing the keys
        ``reference_cw_nm``, ``coefficients``, ``formula_type`` and
        ``pixel_reference``.
    """
    # Load FITS header for CW metadata (shape is not strictly required here).
    hdu = fits.open(fits_path)[0]
    header = hdu.header

    # Load CSV calibration: allow n_pixels=None so shorter CSVs are acceptable.
    # The CSV must contain at least (channel+1) rows.
    # We do not require the CSV pixel count to match the cube here because the
    # fitted polynomial can be extrapolated to longer pixel ranges.
    # The number of channels is inferred from the cube.
    data = np.asarray(hdu.data, dtype=float)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D cube in FITS file, got {data.ndim}D")
    _, n_channels, _ = data.shape
    if not (0 <= channel < n_channels):
        raise ValueError(f"channel index {channel} out of range 0..{n_channels-1}")

    wavcal = load_wavecal_csv(wavcal_csv, n_channels=n_channels, n_pixels=None)
    if wavcal.shape[0] <= channel:
        raise ValueError(
            f"CSV has only {wavcal.shape[0]} channels, cannot use channel {channel}"
        )

    wl_ch = np.asarray(wavcal[channel], dtype=float)
    n_pix_csv = wl_ch.size
    if n_pix_csv < degree + 1:
        raise ValueError(
            f"Need at least {degree+1} points to fit degree={degree} polynomial, "
            f"got {n_pix_csv}"
        )

    if pixel_reference is None:
        pixel_reference = n_pix_csv // 2

    x = np.arange(n_pix_csv, dtype=float) - float(pixel_reference)
    # np.polyfit returns coefficients in descending powers; we store them in
    # increasing-power order for convenience: c0 + c1*x + c2*x^2 + ...
    poly_desc = np.polyfit(x, wl_ch, degree)
    coeffs = poly_desc[::-1].astype(float)

    # Reference CW: prefer explicit header CW when present; otherwise use the
    # wavelength at the reference pixel.
    cw_header = get_cw_from_header(header)
    if cw_header is not None:
        reference_cw_nm = float(cw_header)
    else:
        reference_cw_nm = float(np.polyval(poly_desc, 0.0))

    return {
        "reference_cw_nm": reference_cw_nm,
        "coefficients": [float(c) for c in coeffs],
        "formula_type": "polynomial",
        "pixel_reference": int(pixel_reference),
    }


def save_bh_wavecal_json(
    params: Mapping[str, Any],
    path: str | None = None,
) -> None:
    """Save calibration parameters to ``bh_wavecal.json``.

    When *path* is ``None``, the JSON is written into the installed package
    resources directory under ``bh_molecule._resources/bh_wavecal.json``. This
    helper is primarily intended for the offline builder in
    ``bh_molecule.calibration_builder``.
    """
    if path is None:
        res_dir = resources.files("bh_molecule._resources")
        path_obj = res_dir.joinpath("bh_wavecal.json")
        path = str(path_obj)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "reference_cw_nm": float(params["reference_cw_nm"]),
                "coefficients": [float(c) for c in params["coefficients"]],
                "formula_type": str(params.get("formula_type", "polynomial")),
                "pixel_reference": int(params["pixel_reference"]),
            },
            f,
            indent=2,
        )


def load_bh_wavecal_json(path: str | None = None) -> dict:
    """Load ``bh_wavecal.json`` from disk or package resources.

    Parameters
    ----------
    path :
        Optional explicit path to a JSON file. When omitted, the file is loaded
        from the installed package resources directory.

    Returns
    -------
    dict
        Parsed JSON dictionary with at least the keys ``reference_cw_nm``,
        ``coefficients``, ``formula_type`` and ``pixel_reference``.
    """
    if path is None:
        res_dir = resources.files("bh_molecule._resources")
        path_obj = res_dir.joinpath("bh_wavecal.json")
        path = str(path_obj)

    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Basic validation / normalisation
    if cfg.get("formula_type") != "polynomial":
        raise ValueError("Only 'polynomial' formula_type is supported")
    coeffs = cfg.get("coefficients")
    if not isinstance(coeffs, (list, tuple)) or len(coeffs) < 2:
        raise ValueError("coefficients must be a list of at least two floats")

    return {
        "reference_cw_nm": float(cfg["reference_cw_nm"]),
        "coefficients": [float(c) for c in coeffs],
        "formula_type": "polynomial",
        "pixel_reference": int(cfg["pixel_reference"]),
    }


def apply_polynomial_wavecal(
    n_pixels: int,
    *,
    cw_nm: float | None = None,
    wavecal: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Return a wavelength axis using a stored polynomial calibration.

    The calibration is defined by ``bh_wavecal.json`` which stores a reference
    central wavelength, a polynomial in pixel-offset coordinates, and a
    reference pixel index. When *cw_nm* is provided and differs from the
    reference CW, the entire wavelength axis is shifted by the scalar
    difference, preserving dispersion.

    Parameters
    ----------
    n_pixels :
        Number of detector pixels along the dispersion direction.
    cw_nm :
        Central wavelength (CW) in nm for the *current* measurement. When
        omitted, the reference CW from the JSON file is used.
    wavecal :
        Optional pre-loaded JSON dictionary. When omitted, it is loaded via
        :func:`load_bh_wavecal_json`.

    Returns
    -------
    ndarray
        Wavelength axis of shape ``(n_pixels,)`` in nanometres.
    """
    if wavecal is None:
        wavecal = load_bh_wavecal_json()

    coeffs = np.asarray(wavecal["coefficients"], dtype=float)
    pixel_ref = int(wavecal["pixel_reference"])
    ref_cw = float(wavecal["reference_cw_nm"])

    x = np.arange(n_pixels, dtype=float) - float(pixel_ref)
    # Evaluate polynomial with coefficients c0 + c1*x + c2*x^2 + ...
    wl = np.zeros_like(x, dtype=float)
    power = np.ones_like(x, dtype=float)
    for c in coeffs:
        wl += c * power
        power *= x

    if cw_nm is not None:
        wl += float(cw_nm) - ref_cw

    return wl


def estimate_cw_from_features(
    spectrum: np.ndarray,
    *,
    wavecal: Mapping[str, Any] | None = None,
) -> float:
    """Estimate central wavelength (CW) from spectral features.

    This helper provides a simple, CSV-free way to estimate the CW for a
    measurement when header metadata are missing or unreliable. The current
    implementation is intentionally conservative:

    - the spectrum is treated as a 1D array (any extra dimensions are
      averaged);
    - the brightest pixel is taken as a proxy for the dominant feature bundle;
    - CW is inferred from the calibrated wavelength at that pixel using the
      stored polynomial from :func:`load_bh_wavecal_json`.

    More sophisticated logic (e.g. explicitly locating H-γ and the BH
    Q-branch bundle) can be built on top of this function in future.

    Parameters
    ----------
    spectrum :
        1D or ND array of intensities along the dispersion axis in the last
        dimension.
    wavecal :
        Optional pre-loaded calibration dictionary. When omitted, it is loaded
        from ``bh_wavecal.json``.

    Returns
    -------
    float
        Estimated CW in nanometres.
    """
    arr = np.asarray(spectrum, dtype=float)
    if arr.ndim == 0:
        raise ValueError("spectrum must be at least 1D")
    if arr.ndim > 1:
        # Average over all axes except the last (dispersion axis).
        axes = tuple(range(arr.ndim - 1))
        arr = arr.mean(axis=axes)

    if wavecal is None:
        wavecal = load_bh_wavecal_json()

    pixel_ref = int(wavecal["pixel_reference"])
    coeffs = np.asarray(wavecal["coefficients"], dtype=float)

    peak_idx = int(np.argmax(arr))
    x = float(peak_idx - pixel_ref)

    # Evaluate polynomial at the peak pixel to estimate absolute wavelength.
    wl = 0.0
    power = 1.0
    for c in coeffs:
        wl += float(c) * power
        power *= x

    return float(wl)
