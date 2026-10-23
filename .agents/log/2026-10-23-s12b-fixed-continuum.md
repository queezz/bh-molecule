# Held-continuum switch (paper repo session S12b)

**Date:** 2026-10-23 (JST)
**Goal:** the paper repo's S12b builds a continuum floor from the pixels outside the BH window,
before the band fit, and holds it in the fit. Add a switch for that with set C unchanged.

## Changed (9493c8e)
- `physics.py`: `BHModel(baseline="fixed")`; `set_continuum(x, values)` (measured grid, interpolated,
  never extrapolated), `continuum_on(x)`; `fixed_params` holds `base` at 0; no continuum parameter.
- `fit.py`: `BHFitter(continuum=f)`, `f(vis, frame, channel) -> (x, values)` called per spectrum;
  validated against the model; `describe_constraints()["continuum"]`.
- `docs/phys.md`; `tests/test_fixed_continuum.py` (7 tests).

## Verified
- pytest 155 passed, 1 skipped; ruff clean; set C bit-identical (paper repo `test_run_bh_batch.py`).
- In the paper repo (analysis 69ea256) the switch reproduces the S12 engine with the floor held to
  0.01 K, and Figure 8 through bh-molecule gives 2834 / 3311 K as the engine.
- Full-dataset runs `S12b-F20`/`S12b-F30` through `run_bh_batch` completed 13 of 49 shots each
  (0 errors) before the session stopped them for low system memory.

## Not done
- No version bump (as S5a-S12).
