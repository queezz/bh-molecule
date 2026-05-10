# BH Batch Fitting Workflow

This page gives a brief overview of the batch fitting code: where it lives, what it does, and how the pieces fit together.

## Location

- **Batch workflow:** `bh_molecule.workflows.batch_fit`
- **Signal detection:** `bh_molecule.workflows.signal_scan`

The package exposes the main entry points at top level:

```python
from bh_molecule import run_bh_batch, run_folder_batch
```

## Purpose

The workflow runs BH A–X band fits over many **(frame, channel)** pairs from a VIS-1.33 m FITS cube. It:

1. Loads the cube and applies wavelength calibration (CW, scale, optional dark).
2. Decides which frames and channels to fit (either from your input or via automatic signal detection).
3. Fits the BH model for each selected (frame, channel), collects parameters and fit curves.
4. Writes results (CSV, pickle, grid plots, per-fit figures) under a shot output directory.

So you get a table of fit parameters per (frame, channel) plus diagnostic plots, without scripting the loop yourself.

## Main entry points

| Function | Role |
|----------|------|
| **`run_bh_batch`** | Single FITS file: load, calibrate, select frames/channels, run fits, save under `out_dir/<shot_id>/`. |
| **`run_folder_batch`** | Multiple FITS: for each `.fits` in a folder, call `run_bh_batch`; skip shots that already have `summary.csv` (resume). |

Both accept explicit `frames` and `channels` (or `None` for auto selection), plus options such as `cw`, `scale`, `dark_frame`, `time_range`, `band`, `threshold_sigma`, `bounds`, `out_dir`, and **`run_fit_limit`** (run only the first N fits for testing).

## Pipeline steps (single shot)

Inside `run_bh_batch` the flow is:

1. **Load** — `Vis133M.from_files(fits_path, scale=scale)` and apply CW via `vis.apply_cw(cw_nm)`.
2. **Optional dark** — If `dark_frame` is set, subtract that frame from the cube.
3. **Baseline** — `vis.set_baseline_zero(True)` and set time axis from `time_range`.
4. **Background check** — `check_background_flat(vis, background_frames)` to warn if background frames look non-flat.
5. **Frame/channel selection** — If `frames` or `channels` are `None`, `scan_signal_frames(vis, band=..., threshold_sigma=...)` returns lists of frames and channels with signal above threshold; otherwise the provided lists are used.
6. **Fit loop** — `_batch_with_progress(fitr, frames_list, channels_list, run_fit_limit=run_fit_limit)` builds (frame, channel) pairs (optionally truncated to the first N), runs `fitr.fit(f, ch)` for each, and aggregates rows and curves.
7. **Save** — Write `summary.csv`, `curves.pkl`, grid PDF via `save_batch_fit_grid`, and (when `save_frames=True`, the default) per-fit PNGs in `frames/` named via `frame_plot_filename(frame, channel)` (e.g. `f06_ch28.png`).

Selection logic is unchanged when `run_fit_limit` is set; only the number of fits executed is capped so you can test the pipeline quickly.

### Wavelength-shift tolerance

The fit parameter `dx` (in nm) absorbs small mismatches between the data wavelength axis (set by `cw`) and the model. Defaults live in `bh_molecule.fit.DEFAULT_DX_TOL_NM` (currently **0.3 nm**) and the fitter is initialised with symmetric bounds `[-DEFAULT_DX_TOL_NM, +DEFAULT_DX_TOL_NM]` so the fit can shift the model in either direction. Override via the YAML config's `bounds:` block when an experiment needs a different tolerance:

```yaml
bounds:
  lower: [0, 0, -0.5, 0, -10, 0, 0]
  upper: [10, 10000,  0.5, 0.1, 10, 1, 1]
```

A previous regression set the lower `dx` bound to `0`, which silently locked the fit to one side of the model; symptoms were "compressed" fits with all amplitude pushed onto narrow lines. If you see that pattern, check the `dx` bounds first.

### Per-fit PNGs

`save_frames=True` (default) writes `<out>/<shot>/frames/f{frame:02d}_ch{channel:02d}.png`. Set `save_frames: false` in the YAML config (or the Python kwarg) to skip per-fit plots and keep only the summary CSV / grid PDF. The directory is created automatically; failed plots are skipped without aborting the batch.

## Internal helpers

| Helper | Role |
|--------|------|
| **`_batch_with_progress`** | Iterates over (frame, channel) pairs (optionally limited by `run_fit_limit`), calls `BHFitter.fit`, builds a DataFrame of parameters and a dict of `(frame, channel) -> (x, y, yfit)`. Uses tqdm for a progress bar. |
| **`_plot_normalized_grid_pages`** | Takes the curves dict and frame/channel lists; produces a multi-page PDF and a PNG of normalized spectra in a frames×channels grid. |

Signal selection lives in **`signal_scan`**:

- **`scan_signal_frames`** — Builds a band image over the wavelength `band`, compares to background frames, returns frame and channel indices above `threshold_sigma`.
- **`check_background_flat`** — Quick check that background frames look noise-like (peak/baseline ratio).

## Dependencies

- **Vis133M** — Load FITS, apply CW/scale/dark, expose `spectrum(frame, channel)` and band image.
- **BHModel** — BH A–X spectrum model.
- **BHFitter** — Wraps the model and Vis133M data; `fit(frame, channel)` returns parameters, errors, and fit curve.

## Output layout

For a single shot, output is under `out_dir/<shot_id>/`:

| Path | Content |
|------|---------|
| `summary.csv` | One row per (frame, channel): frame, channel, fit parameters, errors, chi2_red, R2, npts. |
| `curves.pkl` | Pickle of `{(frame, channel): (x, y, yfit)}` for plotting. |
| `grid.pdf` / `grid.png` | Normalized spectra in a frames×channels grid (PDF multi-page by channel groups). |
| `frames/f<frame>_ch<channel>.png` | Single-spectrum fit plot per (frame, channel). |

The same structure is used for limited runs (`run_fit_limit`); only the number of rows and curves is smaller.

## CLI

The **`bh batch`** command runs this workflow from a YAML config: see [Command Line](cli_commands.md#batch-fitting-bh-batch). The config supplies `fits_file` or `folder`, plus `cw`, `scale`, and other options; **`--run-fit-limit N`** is passed through as `run_fit_limit` for quick tests.

### Quick subset run

For developer/manual testing, point at a single FITS and limit the fits:

```bash
bh batch --config examples/fit_batch_small_example.yaml --run-fit-limit 6
```

The example config (`examples/fit_batch_small_example.yaml`) targets shot `193788` from the LHD BH dataset and uses an explicit `frames`/`channels` selection so the full run completes in seconds. To exercise shots `193788`, `193789`, and `193790`, copy the file three times (one per `fits_file`) or place those three FITS files in a folder and use the `folder:` key — `run_folder_batch` skips shots whose `summary.csv` already exists, so the three runs can chain via the same `out_dir`.
