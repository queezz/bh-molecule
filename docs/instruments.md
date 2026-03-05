# Instruments and Wavelength Calibration

This page describes how wavelength calibration for the **Vis133M spectrometer** is generated and used during runtime.

The calibration workflow is:

```
Reference CSV + reference FITS
        ↓
Polynomial wavelength model
        ↓
Stored in bh_wavecal.json
        ↓
Runtime wavelength reconstruction
```

The original CSV reference is only required once to build the calibration.

---

## 1. Reference calibration (instrument CSV)

The Vis133M spectrometer is initially calibrated using a **dedicated instrument
wavecal CSV** (for example `133mVis_wavcal.csv`), which contains the per-channel
dispersion of the spectrometer.

The wavecal CSV contains:

- **1024 detector pixels** in the dispersion direction  
- **multiple spatial channels**, each containing wavelength values for all pixels

The instrument CSV is loaded using:

```
load_wavecal_csv()
```

which returns a wavelength matrix:

```
(n_channels, n_pixels)
```

For diagnostics or verification, the CSV can be converted to simple linear fits using:

```
csv_to_linear_formulas()
```

---

## 2. Polynomial wavelength model and calibration pixels

To avoid loading the CSV at runtime, the wavelength solution is compressed into
an **instrument dispersion polynomial**.

For a chosen detector channel, in the *calibration* pixel coordinate system:

λ(x_cal) = P(x_cal - pixel_reference)

where

- P is the fitted polynomial
- ``x_cal`` is the pixel coordinate in the calibration image
- ``pixel_reference`` is the reference pixel index in that calibration image

The calibration cube used to build this model has a fixed number of pixels in
the dispersion direction:

```
calibration_pixels = len(x_cal)
```

The helper function:

```
compute_calibration_from_reference(wavcal_csv, fits_path)
```

returns a dictionary including:

```
{
  reference_cw_nm,   # CW of the calibration image
  coefficients,      # polynomial coefficients for P
  formula_type,
  pixel_reference,   # reference pixel in calibration coordinates
  calibration_pixels # wavelength axis length of the calibration image
}
```

---

## 3. Stored calibration

The calibration is written to:

```
bh_molecule/_resources/bh_wavecal.json
```

Example schema:

- ``reference_cw_nm`` – central wavelength of the *calibration* FITS image  
- ``coefficients`` – polynomial coefficients for P  
- ``pixel_reference`` – reference pixel index in the calibration cube  
- ``calibration_pixels`` – number of calibration pixels along dispersion  

The file is normally generated using the package-integrated builder module:

```
python -m bh_molecule.calibration_builder --csv 133mVis_wavcal.csv --fits path/to/reference.fits
```

This builder:

1. reads the **instrument wavecal** CSV (e.g. `133mVis_wavcal.csv`)
2. fits the polynomial
3. writes `bh_wavecal.json`

---

## 4. Runtime wavelength reconstruction

During normal data processing the CSV reference is **not used**.

Instead:

1. Load calibration
```
cfg = load_bh_wavecal_json()
```
2. Obtain the central wavelength (CW_data) for the **current** measurement
   from manual input, FITS metadata, H-γ fitting or BH Q-branch fitting.
3. Generate wavelength axis using the calibration pixels and CW shift:

Let:

- ``N = n_pixels`` – dispersion length in the *data* cube
- ``Nc = cfg["calibration_pixels"]`` – dispersion length of the calibration cube

Define the mapping from data pixels to calibration pixels:

```
scale = Nc / N
x_data = np.arange(N)
x_cal  = x_data * scale
```

Then the wavelength axis is:

```
wl = apply_wavecal(n_pixels=N, cw_nm=CW_data, wavecal=cfg)
```

internally implementing:

```
λ(x_data) = P(x_cal - pixel_reference) + (CW_data - reference_cw_nm)
```

with P defined entirely by the calibration step. The code does **not** infer
CW_data automatically; it must be provided explicitly for each dataset.

---

## 5. Central wavelength estimation in BH experiments

Typical spectra around **433 nm** contain two useful features:

- the **BH Q-branch bundle**
- the **H-γ line**

Two CW estimation strategies are supported.

### From instrument metadata

FITS headers may contain CW information:

```
CWL
CENWAVE
WAVELEN
LAM_CEN
CRVAL1
```

These are scanned by:

```
get_cw_from_header()
```

---

### From spectral features

If metadata is unavailable, CW can be estimated from the measured spectrum.

```
estimate_cw_from_features(spectrum)
```

This method:

1. averages over non-dispersion axes
2. finds the dominant peak
3. converts the pixel index to wavelength using the polynomial calibration

More precise workflows may additionally fit:

- the **H-γ peak**
- the **BH Q-branch spectrum**

to refine the CW estimate.