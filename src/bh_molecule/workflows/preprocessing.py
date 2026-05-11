"""Shared VIS preprocessing helpers for BH batch / single-spectrum workflows.

``background_frames`` in YAML / :func:`run_bh_batch` feeds only the signal-scan
and flat-check helpers — it does **not** subtract a background spectrum from
each row.  Use :func:`mean_background_spectrum` if you need that explicitly.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


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
