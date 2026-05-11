# Fitter constraints (`w_inst`, `dx`, `base`)

`BHFitter` exposes a small set of **high-level, reviewable** controls for the instrumental width `w_inst`, the wavelength shift `dx`, and the additive baseline `base`. They propagate through `run_bh_batch` / `run_folder_batch` and through YAML for `bh batch`.

In day-to-day work these choices are usually made **after** inspecting preprocessing ([Preprocessing for BH batch fits](preprocessing_bh_fits.md)) and **after** a loose calibration pass in [`examples/14_w_inst_calibration.ipynb`](https://github.com/queezz/bh-molecule/blob/main/examples/14_w_inst_calibration.ipynb); see [Batch fitting workflow (notebooks)](workflow_batch_notebooks.md).

## Why constraints matter in practice

**Parameter degeneracy.** BH band shape is sensitive to `T_rot`, amplitude (`C` / line strengths), `w_inst`, `dx`, and `base`. With wide bounds, the optimiser can trade a slightly wrong `w_inst` for a compensating `T_rot` or baseline while keeping $\chi^2$ acceptable—especially when noise is high or the BH window includes blended structure.

**`w_inst` in particular.** Instrumental FWHM is often nearly constant for a given spectrometer state, while the fitter treats it as a free line-broadening dial. Leaving it **too free** invites the optimiser to absorb model–data mismatch into width instead of wavelength shift or temperature. That is not a moral failure of the fitter; it is a reason to **narrow or fix** `w_inst` once you have evidence from data.

**Tighten vs fix.** Use **tight bounds** when you expect small shot-to-shot drift (focus, etendue) but want to stay near a calibrated interval. Use **`fix_w_inst=True`** (with `w_inst_default` set to your adopted value) when the instrument width is stable and you want the optimiser to spend its degrees of freedom elsewhere. Both are legitimate; the choice should come from **histograms and residuals**, not from defaults alone.

**Explicit and reviewable.** Production runs should carry **`w_inst_default`**, **`w_inst_bounds`**, **`fix_w_inst`**, **`dx_tol_nm`**, **`base_bound`** (and the BH windows) in the notebook cell, YAML, or script—so a colleague can reproduce the physics without reverse-engineering the run.

**No automatic bounds.** The library intentionally avoids auto-setting per-spectrum limits from local noise or heuristics. Automated bounds are hard to audit in a paper trail and easy to mistake for “the software figured it out.” The workflow is: **you** choose bounds after looking at data; the code enforces what you wrote.

## API summary

```python
BHFitter(
    vis,
    model,
    ...,
    w_inst_default=None,    # float | None  -- initial guess (and fixed value when fix_w_inst=True)
    w_inst_bounds=None,     # (lo, hi) | None  -- 0 <= lo < hi
    fix_w_inst=False,       # bool  -- hold w_inst at w_inst_default via parameter elimination
    dx_tol_nm=None,         # float | None  -- overrides bounds[dx] = +/- dx_tol_nm
    base_bound=DEFAULT_BASE_TIGHT_NM,
    base_tight=False,       # apply base_bound when True
    preprocess=None,        # see workflows/preprocessing.make_bh_fit_preprocessor
)
```

The 6 free-parameter parameter-elimination path is used internally when
`fix_w_inst=True`; the returned `params`, `errors`, `cov`, and the batch
CSV columns keep the full 7-parameter shape, with `w_inst_err == 0` and
the corresponding covariance row/column zeroed.

`BHFitter.describe_constraints()` returns a JSON-serializable dict that
is printed in the per-shot batch header (see "Reporting" below).

### Defaults are preserved

Every new constructor kwarg defaults to `None` / `False` and preserves
the previous behaviour exactly. Calling code that does not pass them
sees the same `p0` and `bounds` as before this change.

## Batch workflow (Python API)

`run_bh_batch` (and `run_folder_batch`) accept the same five kwargs and
forward them into `BHFitter`:

```python
from bh_molecule import run_folder_batch

run_folder_batch(
    DATA_DIR,
    frames=None,
    channels=None,
    shots=[193788, 193789, 193790],
    cw=431.91,
    scale=1.0,
    background_frames=(0, 1, 2, 3),
    bh_fit_range=(433.05, 433.90),
    bh_scale_range=(433.08, 433.30),
    w_inst_default=0.022,
    w_inst_bounds=(0.020, 0.024),
    fix_w_inst=False,
    dx_tol_nm=0.3,
    base_bound=0.03,
    out_dir="bh_batch_results",
)
```

`None` for any constraint means "use the fitter default; do not forward
the kwarg" -- this keeps the existing API call sites unchanged.

## YAML schema (`bh batch`)

The same keys are accepted in the YAML config consumed by `bh batch
--config <file>` (reproducible / headless runs):

```yaml
# explicit, reproducible fitter constraints
w_inst_default: 0.022
w_inst_bounds: [0.020, 0.024]
fix_w_inst: false
dx_tol_nm: 0.3
base_bound: 0.03
```

See `examples/fit_batch_example.yaml` and
`examples/fit_batch_small_example.yaml` for full templates.

## Reporting

`run_bh_batch` emits a per-shot header that includes the active
constraints (via `tqdm.write`, so it composes with progress bars):

```
Processing shot 193788: .../193788.fits
Output: bh_batch_results/193788
background_frames: (0, 1, 2, 3)
save_frames: False
BH fit window: [433.05, 433.9] nm
BH scale window: [433.08, 433.3] nm
Using frames [...]
Using channels [...]
Fitter constraints: w_inst_default=0.0220 nm, w_inst_bounds=[0.0200, 0.0240] nm,
  fix_w_inst=False, dx_tol_nm=0.3000, base_bound=0.0300, base_tight=True
```

So a run is reproducible from its log alone: feed the printed values
back into the YAML / Python API and you get an identical batch.

## Recommended calibration workflow

The calibration workflow is **manual**. The helper notebook
[`examples/14_w_inst_calibration.ipynb`](https://github.com/queezz/bh-molecule/blob/main/examples/14_w_inst_calibration.ipynb) is a small, inspectable scaffold;
it does not auto-pick bounds. API: [workflows.calibration](api/workflows_calibration.md).

1. Select a few representative shots (`SHOTS`) and an explicit, small
   set of `FRAMES` / `CHANNELS`.
2. Run `run_folder_batch` (or `run_bh_batch`) with **loose** bounds:

   ```python
   w_inst_bounds=(0.005, 0.06)   # generous; see where the optimizer lands
   fix_w_inst=False
   ```
3. Concatenate the per-shot summary tables, then use:

   ```python
   from bh_molecule.workflows.calibration import (
       summarize_fit_distribution,
       plot_fit_distributions,
   )

   summary = summarize_fit_distribution(
       df, params=("w_inst", "dx", "base"), r2_min=0.5
   )
   ```

   The returned table reports `n`, `median`, `mad`, `p05/p16/p50/p84/p95`,
   `min`, `max` per parameter (filtered by `R2 >= r2_min` when the column
   is present).

4. Overlay candidate bounds on histograms:

   ```python
   fig, axes = plot_fit_distributions(
       df,
       params=("w_inst", "dx", "base"),
       suggested_bounds={
           "w_inst": (0.020, 0.024),
           "dx":     (-0.10, 0.10),
           "base":   (-0.03, 0.03),
       },
       r2_min=0.5,
   )
   ```

5. **Inspect** the histograms. If `w_inst` is tight around a single
   value with small MAD across shots/frames/channels, consider either:

   - tight bounds (`w_inst_bounds = (lo, hi)`), or
   - a fixed value (`fix_w_inst=True`, `w_inst_default=median`).

6. Commit the chosen values to the notebook cell (e.g. `W_INST_DEFAULT`, `W_INST_BOUNDS` in [`examples/12_batch_batch.ipynb`](https://github.com/queezz/bh-molecule/blob/main/examples/12_batch_batch.ipynb)) or to YAML for `bh batch`. **Do not**
   add code that picks them automatically -- the goal is explicit,
   inspectable, reviewable production constraints.

## See also

- [Preprocessing for BH batch fits](preprocessing_bh_fits.md)
- [Batch fitting pipeline](batch_fit.md)

## Implementation notes

- Fixed `w_inst` is implemented via *parameter elimination*: `curve_fit`
  receives a six-parameter model where `w_inst` is captured by closure.
  After the fit, the fixed value is re-inserted at index 3 of `params`
  and `errors[3] = 0.0`; the covariance row/column at index 3 are zero.
  This keeps the batch CSV schema and the formatted summary table
  unchanged.
- `w_inst_bounds`, `dx_tol_nm`, and `base_bound` overwrite the relevant
  slots of `self.bounds` directly. `p0` is moved inside the new window
  if it fell outside.
- Invalid bounds raise `ValueError` at fitter construction time, never
  silently.
- The new keys are forwarded by `bh_molecule.cli.main_bh` only when
  present in the YAML; absent keys leave the fitter defaults untouched.
