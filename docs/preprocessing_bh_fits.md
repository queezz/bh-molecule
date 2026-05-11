# Preprocessing for BH batch fits

The arrays passed to `BHFitter` in the batch workflow are **not** raw `Vis133M.spectrum` rows. They are built by `prepare_bh_fit_arrays` (wrapped for the fitter via `make_bh_fit_preprocessor` in `run_bh_batch`). The same path is used in [`examples/13_single_spectrum_batch_debug.ipynb`](https://github.com/queezz/bh-molecule/blob/main/examples/13_single_spectrum_batch_debug.ipynb).

This page is about **scientific provenance**: what the code does and why, not implementation trivia.

## Pipeline (summary)

1. Take the raw cube row `vis.cube[frame, channel]` (no per-row minimum subtraction).
2. Subtract the mean over `background_frames` for that channel (if configured).
3. Crop wavelengths to **`bh_fit_range`** (fit window).
4. Compute a **positive** scale from intensities **only inside `bh_scale_range`** (default: max of positive samples; optional `p99`).
5. Divide the cropped spectrum by that scale. **Negative values remain negative** after division.

API reference: [preprocessing module](api/preprocessing.md).

## Why full-spectrum normalization is dangerous

If the normalization divisor is taken over the **entire** cropped fit window, any bright feature in that window—H-γ, scattered light, a neighbor line—can set the scale. The BH band then sits at an arbitrary fraction of “1”, and the fitter couples amplitude (`C`), baseline (`base`), and broadening (`w_inst`) in misleading ways. The symptom is often a fit that looks tolerable on paper while misrepresenting temperature or instrument width.

Restricting the scale to **`bh_scale_range`**, a sub-window placed on the BH bandhead / main structure, keeps the divisor tied to **BH-dominated** pixels.

## Why background subtraction matters

`background_frames` should be pre-plasma (or otherwise signal-free) slices. Averaging them per channel estimates structured bias (stray light, detector offset patterns) that a single additive `base` parameter cannot fully mimic across the band. Subtracting that mean **before** cropping and scaling stabilizes the BH peak amplitude relative to the local continuum.

Background frames are still used for the **flatness check** and optional **signal detection**; for fitting, the same indices feed the subtraction above.

## Why negative values are preserved

Forcing $y \leftarrow y - \min(y)$ (or clipping at zero) erases information about subtraction noise and baseline curvature. The model is fit to the **real** processed counts (scaled), including small negative excursions after background removal. The fitter’s `base` remains interpretable as a small residual offset **around** a spectrum whose BH peak was normalized near unity by the BH-local scale—not around a nonnegative envelope.

## Preprocessing vs display normalization

Multi-frame grid plots (`normalize_curves_for_grid`, PDF/PNG summaries) may rescale curves for **layout** (e.g. comparable peak heights on a page). That is **only** for visualization. The fit always uses `prepare_bh_fit_arrays`. When debugging, compare the debug notebook’s fit arrays to batch outputs, not to grid-plot scaling.

## Practical recommendations

- Choose `bh_scale_range` so it avoids strong non-BH lines when possible; keep it wide enough to sample the BH peak robustly.
- Widen `bh_fit_range` if the model must see wing structure for $T_\mathrm{rot}$, but keep the **scale** window on the BH-dominated region.
- If step 4 fails (“no positive values in scale window”), the window or background is wrong for that spectrum—fix the physics before expanding bounds.

## See also

- [Batch fitting workflow (notebooks)](workflow_batch_notebooks.md)
- [Batch fitting pipeline](batch_fit.md)
- [Fitter constraints](fitter_constraints.md) — constraints are chosen **after** preprocessing is trusted.
