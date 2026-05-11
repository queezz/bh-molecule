# BH Batch Fitting: Operator Quick-Start Guide

For development and analysis, the usual path is the repository notebooks ([overview](workflow_batch_notebooks.md)): single-spectrum debug, folder batch, then `w_inst` calibration before freezing settings.

This page documents **`bh batch`** for **reproducible** runs from a YAML file—after you have already validated preprocessing and fitter constraints interactively.

## Setup

Activate the virtual environment (created outside the synced folder):

**Windows PowerShell:**

```powershell
& "$env:USERPROFILE/.venvs/bh-molecule/Scripts/Activate.ps1"
```

**Linux / macOS:**

```bash
source ~/.venvs/bh-molecule/bin/activate
```

Install or update the package:

```bash
python -m pip install -e ".[dev]"
```

## Quick Test (Limited Run)

Run a fast pipeline check with `--run-fit-limit`:

```bash
bh batch --config examples/fit_batch_example.yaml --run-fit-limit 5
```

This executes only the first 5 (frame, channel) fits, producing valid outputs (`summary.csv`, `curves.pkl`, `grid.pdf`) with fewer rows. Use this to verify calibration and detection before committing to a full run.

## Full Batch Run

Run all detected fits:

```bash
bh batch --config examples/fit_batch_example.yaml
```

For a folder of FITS files, the workflow processes each `.fits` file sequentially. Shots with existing `summary.csv` in the output directory are skipped (resume behavior).

## How Automatic Signal Detection Works

When `frames` or `channels` are not specified in the config (or set to `null`), the workflow uses `scan_signal_frames()` to automatically find frames and channels with BH emission:

1. **Band image extraction** — A 2D image (frames × channels) is created by integrating each spectrum over the `band` wavelength range (default: 433.0–433.4 nm, the BH A–X Q-branch region).

2. **Noise estimation** — The median and standard deviation are computed from the `background_frames` (default: frames 0, 1, 2, 3), which should contain no plasma signal.

3. **Threshold detection** — Pixels in the band image exceeding `baseline + threshold_sigma × noise` are marked as signal. Default `threshold_sigma` is 5.0.

4. **Frame/channel selection** — Frames with signal in more than 2 channels, and channels with signal in more than 2 frames, are selected for fitting.

If no signal is detected (empty frame/channel lists), the workflow continues without error but produces empty outputs. Check console messages for "Detected frames" and "Detected channels" lists.

## Config File Keys

| Key | Required | Description |
|-----|----------|-------------|
| `folder` | One of these | Directory containing `.fits` files |
| `fits_file` / `fits` | One of these | Path to a single FITS file |
| `cw` | Yes | Center wavelength (nm) for calibration |
| `scale` | Yes | Pixel-to-nm scale factor |
| `out_dir` | No | Output directory (default: `results`) |
| `time_range` | No | `[start, stop]` seconds for frame time axis |
| `dark_frame` | No | Frame index to subtract as dark reference |
| `background_frames` | No | Pre-plasma frames: noise estimate for signal detection **and** per-channel mean subtracted in the fitting preprocessor (default: `[0,1,2,3]`) |
| `band` | No | Wavelength range `[lo, hi]` nm for signal detection (default: `[433.0, 433.4]`) |
| `threshold_sigma` | No | Detection threshold in sigma units (default: `5.0`) |
| `frames` | No | Explicit frame indices (skip auto-detection) |
| `channels` | No | Explicit channel indices (skip auto-detection) |
| `bounds` | No | Fit parameter bounds (`{lower: [...], upper: [...]}`) |
| `bh_fit_range` | No | BH fit wavelength window [nm] (crop before fitting); default matches `BH_FIT_WAVELENGTH_RANGE_NM` in preprocessing |
| `bh_scale_range` | No | Sub-window [nm] used only for normalization scale; default matches `BH_SCALE_WAVELENGTH_RANGE_NM` |
| `w_inst_default` | No | Initial guess and fixed value when `fix_w_inst: true` (instrumental FWHM, nm) |
| `w_inst_bounds` | No | `[lo, hi]` for `w_inst` (nm), `0 <= lo < hi` |
| `fix_w_inst` | No | When `true`, hold `w_inst` at `w_inst_default` |
| `dx_tol_nm` | No | Half-width of allowed `dx` (nm); default fitter value if omitted |
| `base_bound` | No | Half-width of tight `base` bound when `base_tight` applies |
| `fitter_kwargs` | No | Extra args for `BHFitter` (`nm_window`, `weight`, `warm_start`) |

## Output Layout

For each shot (FITS file stem), outputs appear under `out_dir/<shot_id>/`:

| File | Content |
|------|---------|
| `summary.csv` | Fit results: frame, channel, parameters, errors, chi2_red, R2, npts |
| `curves.pkl` | Pickle of `{(frame, channel): (x, y, yfit)}` for further analysis |
| `grid.pdf` | Multi-page PDF with normalized spectra grid (frames × channels) |
| `frames/f<NN>_ch<NN>.png` | Per-fit plots (only if `save_frames: true`) |

## Resume Behavior

When running `run_folder_batch` (i.e., config has `folder`), shots are skipped if `summary.csv` already exists in their output directory. To reprocess a shot, delete its output folder first.

## Troubleshooting

- **No frames/channels detected**: Check that `background_frames` truly contain no signal, and that `band` covers the BH emission peak. Lower `threshold_sigma` if signal is weak.
- **Empty summary.csv**: Signal detection found no qualifying frames/channels. Verify data quality and calibration.
- **Background check warning**: If you see "background frames may contain signal", your `background_frames` indices may overlap with plasma discharge times.
