import numpy as np
import matplotlib.pyplot as plt
from bh_spectra.dataio import load_v00_wavelengths
from bh_spectra.physics import BHModel

v00 = load_v00_wavelengths()
model = BHModel(v00)

x = np.linspace(432.8, 434.2, 4000)  # nm
y = model.full_fit_model(x, C=1.0, T_rot=2000, dx=0.0, w_inst=0.02, base=0.0, I_R7=0.5, I_R8=0.3)

plt.figure()
plt.plot(x, y)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity (a.u.)")
plt.title("BH A–X near 433 nm (demo)")
plt.show()
