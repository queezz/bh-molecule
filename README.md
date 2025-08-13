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

## Documentation

Project documentation is written in Markdown and built with [MkDocs](https://www.mkdocs.org/).
To build the HTML site locally:

```bash
pip install mkdocs
mkdocs build
```

The rendered site can be published automatically with GitHub Pages.
