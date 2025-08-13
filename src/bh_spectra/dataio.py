from __future__ import annotations
import hashlib, json
import numpy as np, pandas as pd
from importlib import resources

def load_v00_wavelengths() -> pd.DataFrame:
    """Load 11BH_v00.csv from package resources and convert to wavelengths (nm)."""
    with resources.files("bh_spectra._resources").joinpath("11BH_v00.csv").open("rb") as f:
        v00_wn = pd.read_csv(f)  # expects columns P,Q,R in wavenumbers (cm^-1)
    wl_nm = 1e7 / v00_wn  # vacuum nm
    wl_nm.columns = ["P","Q","R"]
    return wl_nm

def hash_params(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=float).encode()).hexdigest()

def save_npz(path, **arrays):
    np.savez_compressed(path, **arrays)

def load_npz(path):
    return np.load(path, allow_pickle=False)
