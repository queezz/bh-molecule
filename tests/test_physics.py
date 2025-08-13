import numpy as np
from bh_spectra.dataio import load_v00_wavelengths
from bh_spectra.physics import BHModel, Branch

def test_line_profile_finite():
    v00 = load_v00_wavelengths()
    model = BHModel(v00)
    x = np.linspace(432.8, 434.2, 1000)
    g = model.line_profile(x, 433.5, 0.02, 0.0)
    assert np.isfinite(g).all()
    assert g.max() > 0

def test_spectrum_runs():
    v00 = load_v00_wavelengths()
    model = BHModel(v00)
    x = np.linspace(432.8, 434.2, 1000)
    y = model.spectrum(x, C=1.0, T_rot=2000, w_inst=0.02, T_tra=0.0, branch=Branch.Q, v_max=1, N2_max=5)
    assert np.isfinite(y).all()
