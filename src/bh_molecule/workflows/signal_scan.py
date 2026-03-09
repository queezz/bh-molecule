from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np


def _normalize_indices(values: Iterable[int] | int) -> List[int]:
    """Return a list of indices given an int or iterable of ints."""
    if isinstance(values, int):
        return [int(values)]
    return [int(v) for v in values]


def _default_channels(s, n: int = 5) -> List[int]:
    """Return a small subset of middle channels for quick checks."""
    _, C, _ = s.shape
    if C <= n:
        return list(range(C))
    mid = C // 2
    start = max(mid - n // 2, 0)
    stop = min(start + n, C)
    return list(range(start, stop))


def check_background_flat(s, frames: Iterable[int] | int, channels: Iterable[int] | int | None = None) -> None:
    """Sanity check that background frames look noise-like.

    For each (frame, channel) pair, compute spread/baseline where baseline is
    the median and spread is (95th - 5th) percentile. Uses percentiles instead
    of max to be robust to hot pixels. If the overall max ratio is >~3, print
    a warning.
    """
    frames_list = _normalize_indices(frames)
    if channels is None:
        channels_list = _default_channels(s)
    else:
        channels_list = _normalize_indices(channels)

    max_ratio = 0.0

    for f in frames_list:
        for ch in channels_list:
            _, spec = s.spectrum(f, ch)
            spec = np.asarray(spec, dtype=float)
            if spec.size == 0:
                continue
            baseline = float(np.median(spec))
            if baseline <= 0:
                continue
            p5 = float(np.percentile(spec, 5))
            p95 = float(np.percentile(spec, 95))
            spread = p95 - p5
            ratio = spread / baseline if baseline != 0 else np.inf
            if np.isfinite(ratio):
                max_ratio = max(max_ratio, ratio)

    if max_ratio == 0.0:
        print("Background check: could not compute a finite spread/baseline ratio.")
        return

    if max_ratio > 3.0:
        print(
            f"WARNING: background frames may contain signal "
            f"(spread/baseline ratio = {max_ratio:.2f})"
        )
    else:
        print(f"Background frames look flat (ratio = {max_ratio:.2f})")


def scan_signal_frames(
    s,
    band: Tuple[float, float] = (433.0, 433.4),
    background_frames: Iterable[int] | Tuple[int, ...] = (0, 1, 2, 3),
    threshold_sigma: float = 5.0,
):
    """Scan a wavelength band image to detect frames/channels with signal.

    Returns (frames, channels, img) where frames/channels are lists of indices
    with significant signal and img is the (F, C) band image.
    """
    # Band image: shape (frames, channels)
    img = s.band(band)

    # Noise estimate from background frames
    bg_frames = _normalize_indices(background_frames)
    noise_region = img[bg_frames]
    baseline = float(np.median(noise_region))
    noise = float(np.std(noise_region))
    if noise <= 0:
        noise = 1e-12

    # Detection mask
    threshold = baseline + threshold_sigma * noise
    mask = img > threshold

    # Collapse into frame/channel activity
    frame_signal = mask.sum(axis=1)
    channel_signal = mask.sum(axis=0)

    frames = np.where(frame_signal > 2)[0]
    channels = np.where(channel_signal > 2)[0]

    frames_list = frames.tolist()
    channels_list = channels.tolist()

    print(f"Detected frames: {frames_list}")
    print(f"Detected channels: {channels_list}")

    return frames_list, channels_list, img


__all__ = ["check_background_flat", "scan_signal_frames"]

