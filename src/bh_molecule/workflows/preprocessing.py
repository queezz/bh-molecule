"""Shared VIS preprocessing helpers for BH batch / single-spectrum workflows.

This module provides the canonical *fitting* preprocessing path used by both
:func:`bh_molecule.workflows.batch_fit.run_bh_batch` and the
single-spectrum debug notebook.  The pipeline is:

1. Load the raw row from ``vis.cube[frame, channel]`` (no scale, no min
   subtraction).
2. Optionally subtract a mean background spectrum computed from
   ``BACKGROUND_FRAMES`` for the same channel.
3. Crop to the BH **fit** wavelength window
   (:data:`BH_FIT_WAVELENGTH_RANGE_NM`).
4. Compute a robust positive scale **inside the BH scale window**
   (:data:`BH_SCALE_WAVELENGTH_RANGE_NM`) — by default the maximum positive
   value in that window — so bright lines outside the BH band (e.g. H-γ
   on the full detector) cannot dominate the normalization.
5. Divide by that scale.  **Negative values are preserved**; no
   ``y - y.min()`` shift is ever applied.

Display normalization (``normalize_curves_for_grid``) is intentionally a
separate concept — it lives in :mod:`bh_molecule.plotting.batch_fit_plots`
and may use min/max of the data, but the arrays handed to the fitter must
come from :func:`prepare_bh_fit_arrays`.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import numpy as np


# Default BH fitting and scale windows.  Both are in nanometres.
#
# ``BH_FIT_WAVELENGTH_RANGE_NM`` defines the wavelength range cropped *before*
# fitting.  ``BH_SCALE_WAVELENGTH_RANGE_NM`` is a sub-window that should
# bracket the BH bandhead/peak only; the normalization scale is computed
# strictly inside this range so unrelated bright features (H-γ, scattered
# light) cannot dominate the BH scale.
BH_FIT_WAVELENGTH_RANGE_NM: tuple[float, float] = (433.05, 433.90)
BH_SCALE_WAVELENGTH_RANGE_NM: tuple[float, float] = (433.08, 433.30)


def mean_background_spectrum(
    cube: np.ndarray,
    background_frames: Iterable[int],
    channel: int,
) -> np.ndarray:
    """Average raw counts along selected frames for one channel.

    Parameters
    ----------
    cube : ndarray, shape (F, C, P)
        Data cube **before** multiplicative ``scale`` (same layout as
        ``Vis133M.cube``).
    background_frames : iterable of int
        Frame indices (**0-based**, same as ``Vis133M.spectrum``).
    channel : int
        Channel index (**0-based**).

    Returns
    -------
    ndarray, shape (P,)
        Mean spectrum in raw cube units (multiply by ``vis.scale`` to match
        ``spectrum()`` intensity units).
    """
    frames = np.asarray(list(background_frames), dtype=int)
    if frames.ndim != 1:
        raise ValueError("background_frames must be a 1-D iterable of frame indices")
    F, C, P = cube.shape
    if not (0 <= channel < C):
        raise IndexError(f"channel {channel} out of range for C={C}")
    if np.any((frames < 0) | (frames >= F)):
        raise IndexError(f"background frame index out of range [0, {F - 1}]")
    return np.mean(cube[frames, channel, :].astype(float), axis=0)


def subtract_background_from_row(
    row_signal: np.ndarray,
    row_background: np.ndarray,
) -> np.ndarray:
    """Return ``row_signal - row_background`` (signal minus background)."""
    return np.asarray(row_signal, dtype=float) - np.asarray(
        row_background, dtype=float
    )


def _stats(arr: np.ndarray) -> tuple[float, float, float]:
    """Return ``(min, max, median)`` ignoring NaNs.  Empty -> NaNs."""
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.nanmin(a)), float(np.nanmax(a)), float(np.nanmedian(a))


def prepare_bh_fit_arrays(
    vis,
    frame: int,
    channel: int,
    *,
    background_frames: Optional[Iterable[int]] = None,
    fit_window: tuple[float, float] = BH_FIT_WAVELENGTH_RANGE_NM,
    scale_window: tuple[float, float] = BH_SCALE_WAVELENGTH_RANGE_NM,
    scale_method: str = "max",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build the (x, y) arrays handed to :class:`BHFitter` for one spectrum.

    Pipeline
    --------
    1. ``row = vis.cube[frame, channel]`` (raw counts; ignores
       ``vis.scale`` and ``_baseline_zero``).
    2. If ``background_frames`` is given, subtract
       ``mean(vis.cube[bg, channel])``.
    3. Crop to ``fit_window``.
    4. Compute a positive scale from values **inside** ``scale_window``
       only (``scale_method="max"`` -> ``max(positive values)``;
       ``"p99"`` -> 99th percentile of positive values).
    5. Divide by that scale (no minimum shift; negatives preserved).

    Parameters
    ----------
    vis : Vis133M-like
        Object exposing ``cube`` (shape ``(F, C, P)``) and ``wl_nm``
        (shape ``(C, P)``) attributes.
    frame, channel : int
        0-based indices.
    background_frames : iterable of int or None, optional
        Frame indices averaged to estimate the per-channel background.
        ``None`` disables background subtraction.
    fit_window : (float, float)
        Wavelength range cropped before fitting [nm].
    scale_window : (float, float)
        Sub-window used to compute the normalization scale [nm].
    scale_method : {"max", "p99"}
        How to reduce positive values inside ``scale_window`` to a single
        scale.

    Returns
    -------
    x_fit : ndarray
        Wavelength grid inside ``fit_window`` (sorted).
    y_fit : ndarray
        Background-subtracted, normalized intensity (negatives preserved,
        peak ~1 inside ``scale_window``).
    meta : dict
        Diagnostic metadata — see source for keys.

    Raises
    ------
    ValueError
        When ``fit_window`` is empty, ``scale_window`` does not overlap the
        fit window after cropping, or the computed scale is not finite and
        positive.
    """
    if not hasattr(vis, "cube") or not hasattr(vis, "wl_nm"):
        raise TypeError(
            "vis must expose `cube` (F, C, P) and `wl_nm` (C, P) attributes"
        )
    cube = np.asarray(vis.cube, dtype=float)
    F, C, P = cube.shape
    if not (0 <= frame < F):
        raise IndexError(f"frame {frame} out of range for F={F}")
    if not (0 <= channel < C):
        raise IndexError(f"channel {channel} out of range for C={C}")

    wl = np.asarray(vis.wl_nm[channel], dtype=float)
    row = cube[frame, channel].astype(float).copy()

    bg_used: Optional[tuple[int, ...]]
    if background_frames is None:
        bg_row = np.zeros_like(row)
        y_sub_full = row
        bg_used = None
    else:
        bg_row = mean_background_spectrum(cube, background_frames, channel)
        y_sub_full = subtract_background_from_row(row, bg_row)
        bg_used = tuple(int(i) for i in background_frames)

    fit_lo, fit_hi = float(fit_window[0]), float(fit_window[1])
    if not (fit_hi > fit_lo):
        raise ValueError(f"fit_window must be (lo, hi) with hi > lo, got {fit_window!r}")
    fit_mask = (wl >= fit_lo) & (wl <= fit_hi) & np.isfinite(wl) & np.isfinite(y_sub_full)
    if not fit_mask.any():
        raise ValueError(
            f"fit_window {fit_window} contains no finite samples for "
            f"frame {frame} channel {channel}"
        )
    x_fit = wl[fit_mask]
    y_sub = y_sub_full[fit_mask]
    raw_in_fit = row[fit_mask]
    bg_in_fit = bg_row[fit_mask]

    if x_fit.size >= 2 and np.any(np.diff(x_fit) < 0):
        order = np.argsort(x_fit)
        x_fit = x_fit[order]
        y_sub = y_sub[order]
        raw_in_fit = raw_in_fit[order]
        bg_in_fit = bg_in_fit[order]

    sc_lo, sc_hi = float(scale_window[0]), float(scale_window[1])
    if not (sc_hi > sc_lo):
        raise ValueError(
            f"scale_window must be (lo, hi) with hi > lo, got {scale_window!r}"
        )
    sc_mask = (x_fit >= sc_lo) & (x_fit <= sc_hi)
    if not sc_mask.any():
        raise ValueError(
            f"scale_window {scale_window} does not overlap the cropped fit "
            f"window for frame {frame} channel {channel}"
        )

    sc_values = y_sub[sc_mask]
    positives = sc_values[sc_values > 0]
    if positives.size == 0:
        raise ValueError(
            f"No positive values inside scale_window {scale_window} for "
            f"frame {frame} channel {channel}; cannot normalize"
        )
    if scale_method == "max":
        scale = float(np.nanmax(positives))
    elif scale_method == "p99":
        scale = float(np.nanpercentile(positives, 99.0))
    else:
        raise ValueError(
            f"Unknown scale_method={scale_method!r}; expected 'max' or 'p99'"
        )
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(
            f"Computed scale {scale!r} is not finite/positive for frame "
            f"{frame} channel {channel}"
        )

    y_fit = y_sub / scale

    raw_min, raw_max, raw_med = _stats(raw_in_fit)
    bg_min, bg_max, bg_med = _stats(bg_in_fit)
    sub_min, sub_max, sub_med = _stats(y_sub)
    norm_min, norm_max, norm_med = _stats(y_fit)

    meta = {
        "frame": int(frame),
        "channel": int(channel),
        "fit_window": (fit_lo, fit_hi),
        "scale_window": (sc_lo, sc_hi),
        "scale": scale,
        "scale_method": scale_method,
        "background_frames": bg_used,
        "n_points_fit": int(x_fit.size),
        "n_points_scale_window": int(sc_mask.sum()),
        "n_negative_after_subtract": int(np.sum(y_sub < 0)),
        "raw_min": raw_min,
        "raw_max": raw_max,
        "raw_median": raw_med,
        "background_min": bg_min,
        "background_max": bg_max,
        "background_median": bg_med,
        "subtracted_min": sub_min,
        "subtracted_max": sub_max,
        "subtracted_median": sub_med,
        "normalized_min": norm_min,
        "normalized_max": norm_max,
        "normalized_median": norm_med,
    }
    return x_fit, y_fit, meta


def make_bh_fit_preprocessor(
    *,
    background_frames: Optional[Iterable[int]] = None,
    fit_window: tuple[float, float] = BH_FIT_WAVELENGTH_RANGE_NM,
    scale_window: tuple[float, float] = BH_SCALE_WAVELENGTH_RANGE_NM,
    scale_method: str = "max",
    metadata_sink: Optional[dict] = None,
) -> Callable[[object, int, int], tuple[np.ndarray, np.ndarray]]:
    """Build a ``(vis, frame, channel) -> (x_fit, y_fit)`` callable.

    Intended for :class:`bh_molecule.fit.BHFitter`'s ``preprocess`` argument
    so batch and single-spectrum debug paths share *exactly* the same
    preprocessing (see :func:`prepare_bh_fit_arrays`).

    Parameters
    ----------
    metadata_sink : dict or None
        If provided, the metadata dict for each call is stored under
        ``metadata_sink[(frame, channel)]``.  Useful for diagnostics.
    """

    bg_arg = (
        tuple(int(i) for i in background_frames)
        if background_frames is not None
        else None
    )
    fit_w = (float(fit_window[0]), float(fit_window[1]))
    sc_w = (float(scale_window[0]), float(scale_window[1]))

    def _preprocess(vis, frame: int, channel: int):
        x_fit, y_fit, meta = prepare_bh_fit_arrays(
            vis,
            frame,
            channel,
            background_frames=bg_arg,
            fit_window=fit_w,
            scale_window=sc_w,
            scale_method=scale_method,
        )
        if metadata_sink is not None:
            metadata_sink[(int(frame), int(channel))] = meta
        return x_fit, y_fit

    _preprocess.fit_window = fit_w  # type: ignore[attr-defined]
    _preprocess.scale_window = sc_w  # type: ignore[attr-defined]
    _preprocess.background_frames = bg_arg  # type: ignore[attr-defined]
    _preprocess.scale_method = scale_method  # type: ignore[attr-defined]
    return _preprocess


__all__ = [
    "BH_FIT_WAVELENGTH_RANGE_NM",
    "BH_SCALE_WAVELENGTH_RANGE_NM",
    "mean_background_spectrum",
    "subtract_background_from_row",
    "prepare_bh_fit_arrays",
    "make_bh_fit_preprocessor",
]
