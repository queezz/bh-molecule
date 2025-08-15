# BH Molecule

This project provides tools to model and fit the A–X band spectra of boron hydride (BH). It began life as a set of Jupyter notebooks and has since been refactored into a Python package.

## Installation

You can install **BH Molecule** locally or directly from GitHub. Both methods will make the CLI commands (`bh-spectra`, `bh-spectra-csv`, `bh-spectra-plot`) available in your terminal.

**From a cloned repository**:

```bash
pip install .
```
Or, fordevelopment:
```bash
pip install -e .
```

See the [Dev Install Guide](dev-install-guide.md) for more details.

**Directly from GitHub**:

```bash
pip install git+https://github.com/queezz/bh-molecule.git
```

After installation, try:

```bash
bh-spectra --xmin 432.8 --xmax 434.2 --points 4000 --out spectrum.npz
```
## Quick start

Use the Python API:

**Python example:**

```python
import numpy as np
from bh_molecule.dataio import load_v00_wavelengths
from bh_molecule.physics import BHModel

model = BHModel(load_v00_wavelengths())
x = np.linspace(432.8, 434.2, 4000)
y = model.full_fit_model(x, C=1.0, T_rot=2000, dx=0.0, w_inst=0.02)
```

