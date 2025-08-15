from __future__ import annotations
import hashlib, json
import numpy as np, pandas as pd
from importlib import resources


def n_air(wl_nm: float | np.ndarray) -> float | np.ndarray:
    """Refractive index of air (Science Chronology / 理科年表 formula).

    Parameters
    ----------
    wl_nm:
        Wavelength in nanometers.

    Returns
    -------
    float | np.ndarray
        Refractive index of air at the given wavelength.
    """
    wl_um = np.asarray(wl_nm) * 1e-3  # convert to micrometers
    tmp = 6432.8 + 2949810 / (146 - 1 / wl_um**2) + 25540 / (41 - 1 / wl_um**2)
    return tmp * 1e-8 + 1


def load_v00_wavelengths() -> pd.DataFrame:
    """Load 11BH_v00.csv from package resources and convert to wavelengths (nm)."""
    with (
        resources.files("bh_spectra._resources")
        .joinpath("11BH_v00.csv")
        .open("rb") as f
    ):
        v00_wn = pd.read_csv(
            f, comment="#"
        )  # expects columns P,Q,R in wavenumbers (cm^-1)
    # Convert only spectral branch columns (P,Q,R) from wavenumber (cm^-1) to air wavelengths (nm).
    branch_cols = ["P", "Q", "R"]
    missing = [c for c in branch_cols if c not in v00_wn.columns]
    if missing:
        raise ValueError(f"Missing expected columns in 11BH_v00.csv: {missing}")

    wl_vac_nm = 1e7 / v00_wn[branch_cols]  # vacuum wavelengths (nm)
    wl_air_nm = wl_vac_nm / n_air(wl_vac_nm)  # convert to air wavelengths

    # Preserve J as-is; return DataFrame with columns [J, P, Q, R]
    if "J" in v00_wn.columns:
        out = v00_wn[["J"]].copy()
        out[branch_cols] = wl_air_nm
        return out
    else:
        wl_air_nm.columns = branch_cols
        return wl_air_nm


def hash_params(d: dict) -> str:
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, default=float).encode()
    ).hexdigest()


def save_npz(path, **arrays):
    np.savez_compressed(path, **arrays)


def load_npz(path):
    return np.load(path, allow_pickle=False)
