# Baseline, H2-line and batch switches for the NME revision (session S5b)

**Date:** 2026-10-22 (JST)
**Goal:** let the paper repo refit the whole BH dataset under three baseline
treatments with the S5a model (`upper_energies="fernando_bernath", v_max=0`):
constant ±0.03, linear, and linear plus two H2 nuisance lines with the window
trimmed at 433.80 nm. Production defaults stay as they are.
**Brief:** the owner's decision of 2026-10-22 (evening) in the paper repo,
`<Dropbox>/10-Research/50-Conferences/2026-05-17-PSI/.agents/directions.md`
("Baseline trials"); results note
`Paper/reviews/2026-10-22-revision-design/s5b-baseline-trials.md` there.

## Changed

- `physics.py`: `BHModel(..., baseline="constant", h2_lines=False)`.
  - `baseline="linear"`: `base + base_slope * (x - 433.475)` on the measured
    (unshifted) grid; `BASELINE_REF_NM = 433.475` is the centre of the
    production window and stays fixed when the window is trimmed.
  - `h2_lines=True`: H2 Y 1 -> 2B 0 R0/R1 (433.2619 nm) and R2 (433.5522 nm),
    Dieke's air wavelengths (Crosswhite 1972, Table T), as free-amplitude
    Gaussians with the BH line width on the shifted grid `x + dx`.
  - `extra_param_names`, `extra_p0`, `extra_bounds` (from `EXTRA_PARAM_SPECS`:
    `base_slope` 0 in [-1, 1] per nm; H2 amplitudes 1e-3 in [0, 1], as R7/R8).
    `full_fit_model(x, <seven>, *extra)` takes them after the production seven
    and raises on a wrong count.
- `fit.py`: `BHFitter` appends the model's extra parameters to `param_names`,
  `p0` and `bounds` (a 7-long `p0`/`bounds` is extended, a full-length one used
  as given); `_f`/`_f_w_fixed` take `*params`, so the fixed-`w_inst` elimination
  works with any count. Bug fix: `set_bounds` rejected every call when
  `fix_w_inst=True` (the fixed slot has lo == hi); that slot is now exempt, so
  `run_bh_batch(..., fix_w_inst=True, bounds=...)` works.
- `workflows/batch_fit.py`: `run_bh_batch(..., model_options=None, save_grid=True)`.
  `model_options` goes to `BHModel(**model_options)`; extra parameters become
  `summary.csv` columns; the log prints the model options and the bounds in force
  for `base` and the extras. `save_grid=False` skips `grid.pdf`. The fit-window
  upper limit was already a parameter (`bh_fit_range`); its docstring now says so.
- `docs/phys.md`: the two switches and the batch pass-through.
- `tests/test_baseline_h2_switches.py`: 14 tests (defaults add nothing; linear term
  and its dx independence; H2 Gaussians' positions and width; fitter extension and
  fixed `w_inst`; synthetic recovery of T, slope and an H2 line; trimmed window keeps
  the normalisation; `model_options` reaches the model and the CSV; default batch
  builds `BHModel()`).

## Verified

- `pytest`: 82 passed, 1 skipped (the pre-existing skip). `ruff check` (ruff 0.15
  from the psi2026 venv; this venv has none): the 9 findings are all pre-existing in
  untouched lines (`dataio.py`, `vis133m.py`, two unused imports at the top of
  `physics.py`, `test_regression_original.py`); the changed and new code is clean.
- Production check in the paper repo: with the defaults, `run_bh_batch` on shot
  193788 reproduces the stored production `summary.csv` and `curves.pkl` exactly
  (all columns, difference 0.0).

## Next

- The paper repo's `analysis/scripts/s5b_run_trials.py` runs the three sets.
- Not changed: production defaults, the P/R position offset and the R(7)/R(8)
  hard-coded positions noted in the S5a log.

No version bump (defaults unchanged); nothing pushed.
