
This project provides tools to model and fit the **A–X band spectra** of boron hydride (BH).
The A–X band arises from electronic transitions between the A ¹Π excited state and the X ¹Σ⁺ ground state, producing a distinct band system near 432–434 nm. These spectra are widely used for determining rotational temperature and species concentration in plasmas, as well as for laboratory and astrophysical molecular spectroscopy.

Originally developed as a set of Jupyter notebooks, this codebase is now a Python package with both an API and CLI tools.

**Recommended:** Use a Python virtual environment when installing and running `bh-molecule`. If this repository is synced via Dropbox or another cloud service, create the environment **outside** the repo (for example in `~/.venvs/bh-molecule` on both Windows and macOS). See the [Venv](#venv) section below for details.

📄[**Full documentation**](https://queezz.github.io/bh-molecule/)

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Installation](#installation)
  - [Install Package](#install-package)
- [Package Overview](#package-overview)
- [Python Example](#python-example)
- [Fits Viewer](#fits-viewer)
- [Venv](#venv)
  - [Create virtual environment](#create-virtual-environment)
  - [Activate virtual environment](#activate-virtual-environment)

## Installation

### Install Package

From the repository root:

```bash
pip install -e .
```

Or install directly from GitHub:

```bash
pip install git+https://github.com/queezz/bh-molecule.git
```

## Package Overview

The `bh_molecule` package provides:

- **`BHModel`**: Core physics model for generating BH A–X band spectra with configurable rotational temperature, concentration, and instrumental broadening
- **`BHFitter`**: Fitting tools for matching model spectra to experimental data
- **`load_v00_wavelengths`**: Data I/O utilities for loading wavelength calibration data
- **Calibration tools**: Functions for analyzing wavelength linearity and fiber line fits
- **`Vis133M`**: Instrument interface for working with Vis133M spectrometer data
- **Plotting utilities**: Theme-aware plotting functions with dark/light mode support
- **CLI commands**: `bh-spectra`, `bh-spectra-csv`, and `bh-spectra-plot` for command-line spectrum generation

## Python Example

```python
import numpy as np
from bh_molecule.dataio import load_v00_wavelengths
from bh_molecule.physics import BHModel

model = BHModel(load_v00_wavelengths())
x = np.linspace(432.8, 434.2, 4000)
y = model.full_fit_model(x, C=1.0, T_rot=2000, dx=0.0, w_inst=0.02)
```

## Batch fitting (VIS-1.33 m)

The usual workflow is **notebook-driven**: validate preprocessing and fitter constraints interactively, then scale up.

1. **Debug / one spectrum** — [`examples/13_single_spectrum_batch_debug.ipynb`](examples/13_single_spectrum_batch_debug.ipynb)
2. **Many shots** — [`examples/12_batch_batch.ipynb`](examples/12_batch_batch.ipynb)
3. **Calibrate `w_inst` (and related bounds)** — [`examples/14_w_inst_calibration.ipynb`](examples/14_w_inst_calibration.ipynb)

See the [online docs — BH batch workflow](https://queezz.github.io/bh-molecule/workflow_batch_notebooks/) for terminology (`bh_fit_range` ↔ notebook `BH_FIT_WAVELENGTH_RANGE_NM`, etc.) and preprocessing rationale.

**Automation / reproducible runs:** after those choices are frozen, mirror them in YAML and call the CLI:

```bash
bh batch --config examples/fit_batch_example.yaml
bh batch --config examples/fit_batch_small_example.yaml --run-fit-limit 6
```

Outputs land under `<out_dir>/<shot_id>/`:

| File | Content |
|------|---------|
| `summary.csv` | one row per `(frame, channel)` with fit parameters and errors |
| `curves.pkl` | `(x, y, yfit)` triples from the **fitting** preprocessing path |
| `grid.pdf` | BH-band grid (display normalization for layout) |
| `frames/f{frame:02d}_ch{channel:02d}.png` | per-fit data + fit overlay (set `save_frames: false` to skip) |

Details: [`docs/batch_fit.md`](docs/batch_fit.md), [`docs/fitter_constraints.md`](docs/fitter_constraints.md), [`docs/preprocessing_bh_fits.md`](docs/preprocessing_bh_fits.md).

## Fits Viewer

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


## Venv
Using a dedicated virtual environment keeps this project isolated from your system Python and other projects. When working from a Dropbox or other cloud-synced folder, it is best to keep the virtual environment **outside** the repository to avoid syncing large or transient files. Throughout these examples we place the environment in `~/.venvs/bh-molecule` on both Windows and macOS.

### Create virtual environment

**Linux / macOS:**

```bash
python3 -m venv ~/.venvs/bh-molecule
```

**Windows PowerShell:**

```powershell
python -m venv "$env:USERPROFILE/.venvs/bh-molecule"
```

### Activate virtual environment

**Linux / macOS:**

```bash
source ~/.venvs/bh-molecule/bin/activate
```

**Windows PowerShell:**

```powershell
& "$env:USERPROFILE/.venvs/bh-molecule/Scripts/Activate.ps1"
```