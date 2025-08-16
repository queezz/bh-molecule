# src/bh_molecule/instruments/vis133m.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from shelve import BsdDbShelf
from typing import Literal, Sequence, cast
import numpy as np, pandas as pd
from astropy.io import fits
from astropy.io.fits.hdu.image import PrimaryHDU, ImageHDU
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Vis133MRecord:
    """
    Container for a calibrated VIS-1.33 m spectrometer dataset.

    Attributes
    ----------
    wl_nm : np.ndarray, shape (P,)
        Wavelength axis in nanometers loaded from the wavcal CSV, truncated to the
        pixel count P present in the FITS data.
    signal : np.ndarray
        Calibrated intensity after per-(frame, channel) baseline subtraction and scaling:
        (x - min(x, over pixel axis)) * coeff.
        Shape is (F, C, P) for frame-resolved data, or (C, P) when frames were aggregated.
    header : dict
        Copy of the FITS header for the selected HDU (string keys; values JSON-serializable where possible).
    exptime : float | None
        Exposure time in seconds (from 'EXPTIME' or 'EXPOSURE' if available).
    filename : str
        Path to the source FITS file used to load this record.
    coeff : float | np.ndarray
        Calibration coefficient actually used. May be a scalar, a length-C vector (per-channel),
        or an (F, C) array (per-frame & per-channel). Broadcast along the pixel axis.

    Notes
    -----
    P = number of detector pixels; C = number of channels; F = number of frames.
    """

    wl_nm: np.ndarray  # (P,)
    signal: np.ndarray  # (F, C, P) or (C, P) if aggregated
    header: dict
    exptime: float | None
    filename: str
    coeff: float | np.ndarray  # scalar or array used


def _as_float(v) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _header_coeff(hdr: fits.Header, keys: Sequence[str], default: float = 1.0) -> float:
    for k in keys:
        val = _as_float(hdr.get(k))
        if val is not None:
            return val
    return default


def load_vis133m_rec(
    fits_path: str | Path, wavcal_csv: str | Path, **kwargs
) -> Vis133MRecord:
    wl_nm, signal, hdr, exptime = load_vis133m(fits_path, wavcal_csv, **kwargs)
    coeff = kwargs.get("coeff", 1.0)
    if coeff is None:  # if loader pulled from header, reproduce it for the record
        coeff = _header_coeff(
            hdr, kwargs.get("coeff_header_keys", ("C133M", "CALCOEF", "COEFF")), 1.0
        )
    return Vis133MRecord(
        wl_nm=np.asarray(wl_nm),
        signal=np.asarray(signal),
        header=dict(hdr),
        exptime=exptime,
        filename=str(fits_path),
        coeff=np.asarray(coeff) if hasattr(coeff, "__len__") else float(coeff),
    )


# MARK: Load
def load_vis133m(
    fits_path: str | Path,
    wavcal_csv: str | Path,
    *,
    usecols=range(1, 1025),  # skip serial-number col 0
    frames: Literal["all"] | slice | Sequence[int] = "all",  # frame selection
    channels: slice | Sequence[int] | None = None,  # channel selection
    aggregate: Literal[None, "median", "mean", "sum"] = None,
    coeff: (
        float | np.ndarray | None
    ) = 1.0,  # scalar | (C,) | (F,C) | None -> from header or 1.0
    coeff_header_keys: tuple[str, ...] = ("C133M", "CALCOEF", "COEFF"),
) -> tuple[np.ndarray, np.ndarray, fits.Header, float | None]:
    """
    Load and calibrate a VIS-1.33 m spectrometer FITS cube.

    Parameters
    ----------
    fits_path : str | pathlib.Path
        Path to the FITS file. Data must be shaped (F, C, P) or (C, P).
    wavcal_csv : str | pathlib.Path
        CSV with wavelength axis (nm). By default columns 1..1024 are read
        (column 0 is a serial index) and row 0 is used.
    usecols : iterable[int], optional
        Columns to read from `wavcal_csv`. Default `range(1, 1025)` to skip col 0.
    frames : {"all"} | slice | Sequence[int], optional
        Frames to select. `"all"` keeps every frame. Default `"all"`.
    channels : slice | Sequence[int] | None, optional
        Channels to select. `None` keeps all. Default `None`.
    aggregate : {None, "median", "mean", "sum"}, optional
        If set, reduce across frames **after** calibration, returning (C, P).
        If `None`, return full cube (F, C, P). Default `None`.
    coeff : float | np.ndarray | None, optional
        Calibration coefficient(s) applied after per-(frame,channel) baseline subtraction:
            output = (x - min(x over pixels)) * coeff
        Accepts:
        • scalar → same for all (F, C)
        • (C,) → per-channel
        • (F, C) → per frame & channel
        • None → try FITS header keys in `coeff_header_keys`, else 1.0
        Default 1.0.
    coeff_header_keys : tuple[str, ...], optional
        Header keys to probe when `coeff=None`. Default `("C133M", "CALCOEF", "COEFF")`.

    Returns
    -------
    wl_nm : np.ndarray, shape (P,)
        Wavelength axis in nanometers, truncated to match the pixel count P.
    signal : np.ndarray
        Calibrated data. Shape is (F, C, P) if `aggregate is None`,
        else (C, P) when frames are reduced.
    header : astropy.io.fits.Header
        Copy of the primary HDU header.
    exptime : float | None
        Exposure time in seconds if available (from "EXPTIME" or "EXPOSURE").

    Notes
    -----
    - If the FITS data is (C, P), it is promoted to (1, C, P).
    - Baseline is `np.nanmin` along the pixel axis for each (frame, channel).
    - Wavelengths are taken from `wavcal_csv` row 0; adjust by changing `usecols`.

    Examples
    --------
    >>> wl, cube, hdr, t = load_vis133m("133mVis_169625.fits", "133mVis_wavcal.csv", coeff=0.0001837)
    >>> wl, spec, *_ = load_vis133m("133mVis_169625.fits", "133mVis_wavcal.csv", aggregate="median")
    >>> wl, sub, *_ = load_vis133m("133mVis_169625.fits", "133mVis_wavcal.csv",
    ...                            frames=slice(0, 10), channels=range(16))
    """

    with fits.open(fits_path, memmap=True) as hdul:
        hdu = cast(PrimaryHDU | ImageHDU, hdul[0])
        hdr = hdu.header.copy()
        data = np.asarray(hdu.data, float)

    if data.ndim == 2:
        cube = data[None, ...]  # -> (1,C,P)
    elif data.ndim == 3:
        cube = data  # (F,C,P)
    else:
        raise ValueError(f"Unexpected FITS ndim={data.ndim}")

    if frames != "all":
        cube = cube[frames]
    if channels is not None:
        cube = cube[:, channels]

    F, C, P = cube.shape
    wl_nm = (
        pd.read_csv(wavcal_csv, header=None, usecols=usecols)
        .iloc[0]
        .to_numpy(float)[:P]
    )

    if coeff is None:
        for k in coeff_header_keys:
            if k in hdr:
                coeff = float(hdr[k])  # pyright: ignore
                break
        if coeff is None:
            coeff = 1.0

    c = np.asarray(coeff, float)
    if c.ndim == 0:
        scale = c
    elif c.ndim == 1 and c.size == C:
        scale = c[None, :, None]
    elif c.ndim == 2 and c.shape[:2] == (F, C):
        scale = c[:, :, None]
    else:
        raise ValueError(f"coeff shape {c.shape} incompatible with (F,C)=({F},{C})")

    mins = np.nanmin(cube, axis=2, keepdims=True)
    cube_cal = (cube - mins) * scale

    if aggregate is not None:
        reducer = {"median": np.nanmedian, "mean": np.nanmean, "sum": np.nansum}[
            aggregate
        ]
        signal = reducer(cube_cal, axis=0)  # (C,P)
    else:
        signal = cube_cal  # (F,C,P)

    exptime = hdr.get("EXPTIME") or hdr.get("EXPOSURE")
    return wl_nm, signal, hdr, exptime  # pyright: ignore


# MARK: BH map
def band_heatmap(rec, band_nm=(433.0, 433.9)):
    """
    Heatmap (time × channel) by integrating a wavelength band from a Vis133MRecord.
    """
    sig = rec.signal
    if sig.ndim == 2:
        sig = sig[None, ...]  # (1, C, P)
    F, C, P = sig.shape

    m = (rec.wl_nm >= band_nm[0]) & (rec.wl_nm <= band_nm[1])
    spec = sig[..., m].sum(axis=-1)  # (F, C)
    spec = spec - spec[49, 43]  # background like (49,43)
    spec = np.fliplr(spec)  # match your orientation

    dt = rec.exptime if rec.exptime is not None else 1.0
    x0, x1 = 0.0, dt * (F - 1)
    y = np.arange(1, C + 1)

    fig, ax = plt.subplots()
    im = ax.imshow(
        spec.T,
        cmap="plasma",
        aspect=0.22,
        extent=[x0, x1 if x1 > 0 else 1.0, y.min(), y.max()],  # pyright: ignore
        interpolation="none",
        vmin=0,
        vmax=7,
    )
    cbar = plt.colorbar(im, ticks=range(8))
    cbar.set_label(r"intensity (Wm$^{-2}$sr$^{-1}$)")

    ax.set_xlabel("time (s)" if rec.exptime else "frame")
    ax.set_ylabel("channel")
    ax.set_ylim(1, C)
    yticks = [1, 22, 31, C] if C >= 31 else [1, C]
    ax.set_yticks(yticks)
    if C >= 31:
        ax.hlines(31, x0, x1 if x1 > 0 else 1.0, color="w", linestyles="dashed")
    return fig, ax
