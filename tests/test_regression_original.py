import importlib.util
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from pathlib import Path
import pytest

from bh_molecule.dataio import load_v00_wavelengths
from bh_molecule.physics import BHModel, Branch


def load_original_module():
    repo_root = Path(__file__).resolve().parents[1]
    mod_path = repo_root / "examples" / "bh_spectrum.py"
    spec = importlib.util.spec_from_file_location("bh_orig", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def test_new_matches_original(tmp_path):
    orig = load_original_module()
    # prepare resource CSV in temp dir as expected by original code
    repo_root = Path(__file__).resolve().parents[1]
    src_csv = repo_root / "src" / "bh_molecule" / "_resources" / "11BH_v00.csv"
    dst_dir = tmp_path / "11BH_wl_Fernando"
    dst_dir.mkdir()
    shutil.copy(src_csv, dst_dir / "11BH_v00.csv")

    x = np.linspace(432.8, 434.2, 1000)
    params = dict(C=1.0, T_rot=2000, w_inst=0.02, T_tra=0.0, branch="Q")

    # original spectrum
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        y_orig = orig.BH_spec(x, **params)
    finally:
        os.chdir(cwd)

    # refactored spectrum
    v00 = load_v00_wavelengths()
    model = BHModel(v00)
    y_new = model.spectrum(
        x, C=1.0, T_rot=2000, w_inst=0.02, T_tra=0.0, branch=Branch.Q
    )

    assert np.allclose(y_new, y_orig, rtol=1e-6, atol=1e-10)
