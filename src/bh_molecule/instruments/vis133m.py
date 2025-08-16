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
    df_wl_133m = pd.read_csv(wavcal_csv, header=None, usecols=usecols)
    if channels is None:
        channels = range(C)
    wl_nm = np.array([df_wl_133m.iloc[ch].to_numpy(float)[:P] for ch in channels])

    if coeff is None:
        for k in coeff_header_keys:
            if k in hdr:
                coeff = float(hdr[k])
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

    # background subtraction
    mins = np.nanmin(cube, axis=2, keepdims=True)
    # cube_cal = (cube - mins) * scale
    cube_cal = (cube) * scale

    if aggregate is not None:
        reducer = {"median": np.nanmedian, "mean": np.nanmean, "sum": np.nansum}[
            aggregate
        ]
        signal = reducer(cube_cal, axis=0)  # (C,P)
    else:
        signal = cube_cal  # (F,C,P)

    exptime = hdr.get("EXPTIME") or hdr.get("EXPOSURE")
    return wl_nm, signal, hdr, exptime  # pyright: ignore
