# Instruments and Wavelength Calibration

This page describes how wavelength calibration for the **Vis133M spectrometer** is generated and used during runtime.

**Building the calibration** (one-time):

```
Reference CSV + reference FITS
        ↓
Polynomial wavelength model
        ↓
Stored in bh_wavecal.json
```

**Using calibration on new data** (per dataset):

- Load with **Vis133M.from_files(fits_path)** (no automatic CW).
- Inspect in pixel space: **pick_frame()**, **plot_spectrum_pixels()**, **spectrum(..., channel=None)**.
- Estimate CW from a known line: **estimate_cw_from_line(peak_pixel, line=...)**.
- Apply CW: **apply_cw(cw_nm, ...)**; then use wavelength-calibrated plots and **calibration_report()**.

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

## 4. Loading data: Vis133M.from_files

The main entry point for new data is **Vis133M.from_files()**. It loads the FITS
cube and the polynomial calibration from package resources
(`bh_molecule._resources/bh_wavecal.json`). **CW is never guessed**; you must
set it explicitly.

### API

```python
Vis133M.from_files(
    fits_path: str,
    *,
    cw: float | None = None,
    line: str | float | None = None,
    scale: float = 1.0,
)
```

| ``cw``   | ``line``     | Behaviour |
|----------|---------------|-----------|
| float    | —             | Use that CW; wavelength axis is applied immediately. |
| None     | string        | Look up wavelength in `BALMER_LINES_NM`; store for later (no CW applied). |
| None     | float         | Store as line wavelength (no CW applied). |
| both set | —             | **ValueError** (provide only one). |
| neither  | —             | Load data with dispersion only; call `apply_cw()` later. |

Examples:

```python
s = Vis133M.from_files("193791.fits")                    # dispersion only
s = Vis133M.from_files("193791.fits", cw=434.05)        # use CW directly
s = Vis133M.from_files("193791.fits", line="H_gamma")   # store line, no CW yet
s = Vis133M.from_files("193791.fits", line=434.0462)    # store wavelength
```

---

## 5. Inspecting spectra before calibration

When CW is not set at load time, inspect the cube in **pixel space** to choose
a frame and a peak pixel for CW estimation.

### Pick a frame with signal

```python
frame = s.pick_frame()   # frame with maximum total signal (cube.sum(axis=(1,2)))
```

### Spectrum in pixel space

```python
# Single channel or sum over channels
pixels, intensity = s.spectrum(frame, channel=None, reduce="sum_channels")
# Or one channel:
pixels, intensity = s.spectrum(frame, channel=30)
```

With `channel=None` and `reduce="sum_channels"`, the x-axis is **pixel index**
(not wavelength). Use this to identify the pixel of a known line (e.g. H-γ).

### Plot spectrum (pixel x-axis)

```python
fig = s.plot_spectrum_pixels(frame, show_peaks=True)
fig.show()
```

Uses Plotly; optional `show_peaks=True` marks local maxima with
`scipy.signal.find_peaks`.

---

## 6. Estimating CW from a known line

Once you have chosen the **peak pixel** of a reference line (e.g. H-γ), compute
the central wavelength with **estimate_cw_from_line**. This does **not** change
the instance; it only returns the CW value.

### Balmer line constants

Wavelengths (nm, in air) for common reference lines:

```python
from bh_molecule.constants import BALMER_LINES_NM
# {"H_alpha": 656.279, "H_beta": 486.133, "H_gamma": 434.0462, "H_delta": 410.174}
```

### estimate_cw_from_line

```python
cw_nm = s.estimate_cw_from_line(peak_pixel=876, line="H_gamma")
# or
cw_nm = s.estimate_cw_from_line(peak_pixel=876, wavelength=434.0462)
```

You must pass either **line** (string key in `BALMER_LINES_NM`) or
**wavelength** (float, nm). The calculation:

1. Map data pixel → calibration pixel:  
   `scale = calibration_pixels / data_pixels`,  
   `peak_cal = peak_pixel * scale`,  
   `x_rel = peak_cal - pixel_reference`
2. Evaluate polynomial: `wl_ref = P(x_rel)`
3. CW = `reference_cw_nm + (line_nm - wl_ref)`

---

## 7. Applying CW

To apply the computed (or manual) CW and update the wavelength axis:

```python
s.apply_cw(cw_nm)
```

Optional line metadata (for the calibration report):

```python
s.apply_cw(cw_nm, line_name="H_gamma", peak_pixel=876, frame_used=frame)
```

If you pass **line_name** and omit **line_wavelength**, the wavelength is looked
up in `BALMER_LINES_NM`. **apply_cw** is only available for instances created
with **from_files()**.

---

## 8. Runtime wavelength reconstruction (internal)

The dispersion formula is unchanged. With data pixel count `N` and
`calibration_pixels` Nc:

- Map data → calibration pixels: `x_cal = x_data * (Nc / N)`
- Wavelength:  
  `λ(x_data) = P(x_cal - pixel_reference) + (CW - reference_cw_nm)`

This is implemented by **apply_polynomial_wavecal** (and internally
**apply_wavecal**). **from_files** and **apply_cw** use it to build the
per-channel wavelength axis stored in the Vis133M instance.

---

## 9. Typical workflow (Jupyter)

```python
from bh_molecule.instruments import Vis133M

s = Vis133M.from_files("193791.fits")

frame = s.pick_frame()
fig = s.plot_spectrum_pixels(frame, show_peaks=True)
fig.show()

cw = s.estimate_cw_from_line(peak_pixel=876, line="H_gamma")
s.apply_cw(cw, line_name="H_gamma", peak_pixel=876, frame_used=frame)

s.calibration_report()
fig = s.plot_spectrum_plotly(31, 30)
fig.show()
```

---

## 10. Calibration report and metadata

Instances created with **from_files()** store calibration metadata and can
print a **calibration report**:

```python
s.calibration_report()
```

Reported fields include:

- FITS file, data pixels, calibration pixels, binning factor
- Polynomial order
- If CW was applied: method (manual / reference line), line name and
  wavelength, peak pixel, frame used, CW used, reference CW, ΔCW

Attributes on the instance (when from_files):  
`cw_nm`, `cw_method`, `line_name`, `line_wavelength`, `peak_pixel`,
`frame_used`, `reference_cw_nm`, `delta_cw_nm`, `data_pixels`,
`calibration_pixels`, `binning_factor`.

---

## 11. Legacy: CW from header or features

For scripts that do not use **from_files()**, the package still provides:

- **get_cw_from_header(header)** – scan FITS header for common CW keywords  
  (e.g. CWL, CENWAVE, WAVELEN, LAM_CEN, CRVAL1).
- **estimate_cw_from_features(spectrum, wavecal=..., diagnostic=False)** –  
  estimate CW from the brightest pixel, with optional diagnostic plot.  
  Assumes that peak is a known line (e.g. H-γ); see wavecal module docstrings.

These are **not** used inside **Vis133M.from_files()**; the recommended path
is explicit CW via **estimate_cw_from_line** + **apply_cw** as above.