# BH Spectra

This project provides tools to model and fit the A–X band spectra of boron hydride (BH). It began life as a set of Jupyter notebooks and has since been refactored into a Python package.

## Quick start

Install the package in editable mode:

```bash
pip install -e .
```

Then run the command line tool:

```bash
bh-spectra --xmin 432.8 --xmax 434.2 --points 4000 --out spectrum.npz
```

Or use the Python API:

```python
import numpy as np
from bh_spectra.dataio import load_v00_wavelengths
from bh_spectra.physics import BHModel

v00 = load_v00_wavelengths()
model = BHModel(v00)
x = np.linspace(432.8, 434.2, 4000)
y = model.full_fit_model(x, C=1.0, T_rot=2000, dx=0.0, w_inst=0.02,
                         base=0.0, I_R7=0.5, I_R8=0.3)
```
