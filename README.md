# bh-spectra

BH (boron hydride) A–X band spectrum model and fitting tools.

## Install (editable)

```bash
pip install -e .
```

## Quick start

```python
import numpy as np
from bh_spectra.dataio import load_v00_wavelengths
from bh_spectra.physics import BHModel
from bh_spectra.fit import FrameDataset, BHFitter

v00 = load_v00_wavelengths()
model = BHModel(v00)

x = np.linspace(432.8, 434.2, 4000)  # nm
y = model.full_fit_model(x, C=1.0, T_rot=2000, dx=0.0, w_inst=0.02, base=0.0, I_R7=0.5, I_R8=0.3)
```

See `examples/01_quickstart.py`.

## Mini User Guide
| Module / Function                | Role                                                              |
| -------------------------------- | ----------------------------------------------------------------- |
| `dataio.n_air`                   | Refractive index of air formula for wavelength conversion         |
| `dataio.load_v00_wavelengths`    | Load CSV of line wavenumbers and convert to air wavelengths       |
| `physics.MolecularConstants`     | Container for vibrational/rotational constants of BH              |
| `physics.BHModel.energy`         | Compute molecular energy for given vibrational & rotational state |
| `physics.BHModel.line_profile`   | Gaussian profile combining Doppler and instrumental FWHM          |
| `physics.BHModel.A_coeff`        | Einstein A coefficient with Hönl–London factors                   |
| `physics.BHModel.spectrum`       | Sum line contributions for a chosen branch (P/Q/R)                |
| `physics.BHModel.full_fit_model` | Adds Q‑branch spectrum + independent R₇/R₈ lines & baseline       |
| `fit.FrameDataset`               | Iterator over frames and selected channels of an image stack      |
| `fit.BHFitter.fit_dataset`       | Channel-wise fitting with optional caching                        |


## Build Documentation

Project documentation is written in Markdown and built with [MkDocs](https://www.mkdocs.org/).
To build the HTML site locally:

```bash
pip install mkdocs
mkdocs build
```

The rendered site can be published automatically with GitHub Pages.
