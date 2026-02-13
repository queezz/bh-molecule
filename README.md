

This project provides tools to model and fit the **A–X band spectra** of boron hydride (BH).
The A–X band arises from electronic transitions between the A ¹Π excited state and the X ¹Σ⁺ ground state, producing a distinct band system near 432–434 nm. These spectra are widely used for determining rotational temperature and species concentration in plasmas, as well as for laboratory and astrophysical molecular spectroscopy.

Originally developed as a set of Jupyter notebooks, this codebase is now a Python package with both an API and CLI tools.

📄[**Full documentation**](https://queezz.github.io/bh-molecule/)

**Python example:**

```python
import numpy as np
from bh_molecule.dataio import load_v00_wavelengths
from bh_molecule.physics import BHModel

model = BHModel(load_v00_wavelengths())
x = np.linspace(432.8, 434.2, 4000)
y = model.full_fit_model(x, C=1.0, T_rot=2000, dx=0.0, w_inst=0.02)
```

**CLI example:**

```bash
# Generate a spectrum and save as CSV
bh-spectra-csv --C 5.0 --T_rot 3500 --out spectrum.csv
```

## Venv
```bash
source /Users/queezz/venvs/bh/bin/activate
```

```powershell
& $env:USERPROFILE\.venvs\bh\Scripts\Activate.ps1
```

## Fits viewer

For fast browsing of FITS files, install [fits-viewer](https://github.com/queezz/fits-viewer) directly from GitHub:

```bash
pip install git+https://github.com/queezz/fits-viewer.git
```

**Usage:**

```bash
fits-viewer path/to/file.fits --lo 1 --hi 99 --win 1200x800
```

**Options:**
- `--lo`: Low percentile for initial contrast (default: 1.0)
- `--hi`: High percentile for initial contrast (default: 99.0)
- `--win WxH`: Initial window size in pixels (e.g., 1200x800)
- `--equal`: Lock aspect ratio to equal

This tool is particularly useful for quickly navigating through spectroscopy FITS files with multiple frames.
