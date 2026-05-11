# BH Spectrum Fitting

This guide shows the minimal workflow to fit BH rotational spectra.

## 1 Load spectrometer data

```python
from bh_molecule.instruments import Vis133M

vis = Vis133M.from_files("133mVis_169625.fits")
```

## 2 Create the fitter

```python
from bh_molecule.fit import BHFitter
from bh_molecule.physics import BHModel

model = BHModel()
fitter = BHFitter(vis, model)
```

## 3 Fit a single spectrum

```python
res = fitter.fit(frame=10, channel=15)
```

Inspect parameters:

```python
res["summary"]
```

## 4 Adjust parameter bounds

```python
fitter.set_bounds(
    lower={"T_rot": 2000},
    upper={"T_rot": 8000}
)
```

## 5 Batch fitting

```python
df, curves = fitter.batch(
    frames=[10,11,12],
    channels=range(20),
    return_curves=True
)
```

## 6 Plot results

```python
fitter.plot_single(res)

fitter.plot_overlay(curves, frame=10)
```

For full technical documentation see **API → [fit](../api/fit.md)**.

For end-to-end VIS batch fitting (preprocessing, constraints, notebook order), see [Batch fitting workflow (notebooks)](../workflow_batch_notebooks.md) and [Preprocessing for BH batch fits](../preprocessing_bh_fits.md).
