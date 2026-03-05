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

## 2. Polynomial wavelength model

To avoid loading the CSV at runtime, the wavelength solution is compressed into a **polynomial mapping**.

For a chosen detector channel:

λ(x) = c₀ + c₁ x + c₂ x² + ...

where

```
x = pixel - pixel_reference
```

The coefficients are obtained by least-squares fitting to the CSV reference wavelengths.

The helper function:

```
compute_calibration_from_reference(csv_path, fits_path)
```

returns:

```
{
  reference_cw_nm
  coefficients
  formula_type
  pixel_reference
}
```

---

## 3. Stored calibration

The calibration is written to:

```
bh_molecule/_resources/bh_wavecal.json
```

Example:

```json
{
  "reference_cw_nm": 433.0,
  "coefficients": [430.0, 0.02, 1e-7],
  "formula_type": "polynomial",
  "pixel_reference": 512
}
```

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
2. Determine the central wavelength (CW)

    - from FITS header (`get_cw_from_header`)
    - or from spectral features (`estimate_cw_from_features`)
    
3. Generate wavelength axis
```
apply_polynomial_wavecal(n_pixels, cw_nm=cw, wavecal=cfg)
```
If the measured CW differs from the reference CW, the polynomial is shifted by
```
cw_nm − reference_cw_nm
```
preserving dispersion while adapting to the new centre wavelength.

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