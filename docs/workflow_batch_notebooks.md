# Batch fitting workflow

For VIS-1.33 m BH A–X work there are two supported entry points:

* **Notebooks 12–14** — exploratory debugging, visual validation, and constraint calibration before trusting large batches.
* **`bh batch` + YAML** — the same `run_bh_batch` / `run_folder_batch` pipeline with a checked-in config; best once preprocessing windows and fitter constraints are settled, or when you want unattended / reproducible reruns.

The sections below emphasize the **notebook path** because that is where preprocessing and `w_inst` choices are usually vetted; the [Operator Guide](operator_guide.md) and [Command Line](cli_commands.md) document the CLI path end-to-end.

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

4. **YAML / CLI when stable** — The same knobs (`bh_fit_range`, `bh_scale_range`, `w_inst_*`, …) appear in `examples/fit_batch_example.yaml`. Use `bh batch --config ...` for reproducible full-folder runs, automation, or sharing configs with collaborators—after steps 1–3 have justified those values.

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
