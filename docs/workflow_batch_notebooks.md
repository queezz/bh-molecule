# Batch fitting workflow (notebooks)

For VIS-1.33 m BH A–X work, the practical path is **Jupyter first**: validate preprocessing and constraints visually, then scale to many shots. The YAML / `bh batch` CLI is a supported, reproducible alternative—useful for frozen configs and automation—not the day-to-day interface described here.

Canonical notebooks in the repository (open from a clone of [queezz/bh-molecule](https://github.com/queezz/bh-molecule)):

| Step | Notebook | Role |
|------|----------|------|
| 1 — Explore | [`examples/13_single_spectrum_batch_debug.ipynb`](https://github.com/queezz/bh-molecule/blob/main/examples/13_single_spectrum_batch_debug.ipynb) | One `(frame, channel)`; same preprocessing as batch; plots for sanity checks. |
| 2 — Batch | [`examples/12_batch_batch.ipynb`](https://github.com/queezz/bh-molecule/blob/main/examples/12_batch_batch.ipynb) | `run_folder_batch` over a shot list; production constraints from step 3. |
| 3 — Calibrate constraints | [`examples/14_w_inst_calibration.ipynb`](https://github.com/queezz/bh-molecule/blob/main/examples/14_w_inst_calibration.ipynb) | Loose fits on a small grid; histograms for `w_inst`, `dx`, `base`; you choose bounds or `fix_w_inst`. |
| 4 — Optional | [`examples/fit_batch_example.yaml`](https://github.com/queezz/bh-molecule/blob/main/examples/fit_batch_example.yaml) + `bh batch` | Same parameters as the notebooks, checked into version control. |

## Why this order

1. **Debug before scale** — Batch output is only trustworthy if the per-spectrum pipeline (background, crop, BH-local normalization) matches what you think you are fitting. The debug notebook makes that comparison explicit; see [Preprocessing for BH batch fits](preprocessing_bh_fits.md).

2. **Constraints before production batch** — Unconstrained or overly wide `w_inst` trades off against temperature, amplitude, and baseline in ways that can look acceptable in $\chi^2$ while being physically wrong. The calibration notebook forces a **reviewable** decision (tight bounds vs fixed `w_inst`) before you commit CPU time; see [Fitter constraints](fitter_constraints.md).

3. **Visual inspection** — Per-fit PNGs and `summary.csv` are not a substitute for looking at a few raw rows: H-γ outside the BH scale window, bad background frames, or wavelength quirks show up immediately in single-spectrum plots.

4. **CLI last** — Once windows, `background_frames`, and fitter constraints are frozen, mirroring them in YAML gives reproducibility and CI-style reruns without changing the scientific workflow above.

## Terminology: notebooks vs API / YAML

The notebooks keep configuration in ALL_CAPS Python variables. They map to batch / fitter arguments as follows:

| Notebook variable | `run_bh_batch` / `run_folder_batch` | YAML (when applicable) |
|-------------------|---------------------------------------|-------------------------|
| `BH_FIT_WAVELENGTH_RANGE_NM` | `bh_fit_range` | `bh_fit_range` |
| `BH_SCALE_WAVELENGTH_RANGE_NM` | `bh_scale_range` | `bh_scale_range` |
| `W_INST_DEFAULT` | `w_inst_default` | `w_inst_default` |
| `W_INST_BOUNDS` | `w_inst_bounds` | `w_inst_bounds` |
| `FIX_W_INST` | `fix_w_inst` | `fix_w_inst` |
| `DX_TOL_NM` | `dx_tol_nm` | `dx_tol_nm` |
| `BASE_BOUND` | `base_bound` | `base_bound` |

See also [Batch fitting pipeline](batch_fit.md) and [Command Line](cli_commands.md#batch-fitting-bh-batch).

## See also

- [Preprocessing for BH batch fits](preprocessing_bh_fits.md) — why BH-local normalization and background subtraction matter.
- [Fitter constraints](fitter_constraints.md) — `w_inst`, `dx`, `base`, and the calibration helper API.
- [Operator Guide](operator_guide.md) — CLI-oriented quick start when you already trust the pipeline.
