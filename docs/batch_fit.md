# Batch fitting pipeline

Code: `bh_molecule.workflows.batch_fit`. Entry points:

```python
from bh_molecule import run_bh_batch, run_folder_batch
```

For **how this is used in practice** (notebook order, constraint calibration), start with [Batch fitting workflow (notebooks)](workflow_batch_notebooks.md).

## Purpose

Run BH A–X fits over many `(frame, channel)` pairs from a VIS-1.33 m FITS cube: wavelength calibration, optional signal-based selection, fitting with shared preprocessing, then CSV / curves / grid artifacts under each shot directory.

## Entry points

| Function | Role |
|----------|------|
| **`run_bh_batch`** | One FITS file → fits → `<out_dir>/<shot_id>/`. |
| **`run_folder_batch`** | Every `.fits` in a folder; skips shots whose `summary.csv` already exists (resume). |

Shared options include `cw`, `scale`, `dark_frame`, `time_range`, `background_frames`, **`bh_fit_range`**, **`bh_scale_range`**, **`w_inst_default`**, **`w_inst_bounds`**, **`fix_w_inst`**, **`dx_tol_nm`**, **`base_bound`**, `frames`, `channels`, automatic detection (`band`, `threshold_sigma`), `bounds`, `fitter_kwargs`, `out_dir`, **`run_fit_limit`**, `save_frames`.

## Preprocessing for fitting (authoritative)

`run_bh_batch` attaches a **`preprocess`** callable to `BHFitter` built by `make_bh_fit_preprocessor`. Under the hood this is `prepare_bh_fit_arrays`:

- subtract the **mean** of `background_frames` for the same channel from the raw cube row;
- crop to **`bh_fit_range`**;
- compute a **positive** normalization scale from **`bh_scale_range`** only (default: max of positive samples);
- divide; **negatives are not clipped** and there is no per-row `y - y.min()` shift.

Rationale: [Preprocessing for BH batch fits](preprocessing_bh_fits.md).

## Loader vs fitter: `prepare_vis_for_bh_batch`

`prepare_vis_for_bh_batch` loads the cube, applies `cw`, `scale`, optional dark subtraction, sets the time axis, and sets **`set_baseline_zero(False)`** so `vis.spectrum` stays comparable to the raw cube row in inspection notebooks. **Fitting** does not use `spectrum` directly; it uses the preprocessor on `vis.cube` as described above.

## Pipeline steps (single shot)

1. **Load** — `prepare_vis_for_bh_batch` (or equivalent steps inline in `run_bh_batch`).
2. **Background sanity** — `check_background_flat(vis, background_frames)`.
3. **Frame/channel selection** — explicit lists or `scan_signal_frames` (band image vs background noise).
4. **Fitter** — `BHFitter(..., preprocess=make_bh_fit_preprocessor(...), base_tight=True, nm_window=bh_fit_range, ...)` plus optional `bounds` / constraint kwargs; see [Fitter constraints](fitter_constraints.md).
5. **Fit loop** — `_batch_with_progress` (optional `run_fit_limit`).
6. **Save** — `summary.csv`, `curves.pkl`, `grid.pdf` / `grid.png` via `save_batch_fit_grid`, optional `frames/f{frame:02d}_ch{channel:02d}.png`.

Grid PDFs use **display** normalization for layout; the fitted curves in `curves.pkl` follow the preprocessing pipeline above.

## Wavelength shift `dx`

Parameter `dx` (nm) absorbs small CW / dispersion mismatches. Default half-width is `DEFAULT_DX_TOL_NM` (0.3 nm). Override with **`dx_tol_nm`** in Python/YAML or the `bounds` block. If the lower bound for `dx` is accidentally nonnegative, the fit can lock to one side and compress the spectrum—check bounds first.

## Output layout (`<out_dir>/<shot_id>/`)

| Path | Content |
|------|---------|
| `summary.csv` | One row per fit: frame, channel, parameters, errors, `chi2_red`, `R2`, `npts`. |
| `curves.pkl` | `{(frame, channel): (x, y, yfit)}` from the **fit** preprocessing. |
| `grid.pdf` / `grid.png` | Normalized grid for quick review. |
| `frames/*.png` | Per-fit overlays when `save_frames=True`. |

## CLI (secondary workflow)

`bh batch --config ...` mirrors these kwargs from YAML. See [Command Line](cli_commands.md#batch-fitting-bh-batch). Prefer validating once in [workflow notebooks](workflow_batch_notebooks.md) before long headless runs.

## Memory behavior on long runs

Long folder runs are now memory-safe by construction: `save_batch_fit_grid` uses a fixed `subplots_adjust` layout plus a plain `pdf.savefig(fig)` (no `bbox_inches="tight"`, no explicit `dpi`), and `run_folder_batch` drops the per-shot `curves` dict + `plt.close("all")` + `gc.collect()` between shots. Optional `psutil`-based RSS diagnostics print one line at each shot / page boundary; set `BH_BATCH_MEM_LOG=0` to silence them.

Full story (symptom, root cause, A/B numbers, opt-out, recommendations): [Batch fit memory behavior (long runs)](batch_fit_memory.md).

## Internal helpers

| Helper | Role |
|--------|------|
| **`_batch_with_progress`** | `(frame, channel)` iteration, tqdm, DataFrame + curves dict. |
| **`scan_signal_frames`** / **`check_background_flat`** | Selection and background checks (`signal_scan`). |

## See also

- [Batch fit memory behavior (long runs)](batch_fit_memory.md)
- [workflows.preprocessing API](api/preprocessing.md)
- [fit.py API](api/fit.md)
