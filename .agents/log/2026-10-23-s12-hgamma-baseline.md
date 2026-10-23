# H-gamma-wing continuum switch (session S12)

**Date:** 2026-10-23 (JST)
**Goal:** the paper repo's session S12 (owner's request: the linear continuum of set C
runs above the data between the tail lines of the weak channel-34 spectrum at Rax 3.6 m,
and the quadratic one rides above the data at the blue edge) tests continuum treatments
under the weak BH band. Its candidate "Hw" (set C plus the far wing of H-gamma, amplitude
free) needs a model switch to run over the whole dataset through `run_bh_batch`. Add it
with a default that keeps set C and production unchanged; the paper repo runs it
(`analysis/scripts/s12_run_variants.py`); results note
`Paper/reviews/2026-10-22-revision-design/s12-continuum-investigation.md` there.

## Changed

- `physics.py`, `BHModel(baseline="hgamma")`: set C's linear continuum plus
  `base_hg * hgamma_wing(x + dx)`. `BHModel.hgamma_wing` is a Lorentzian at
  `HGAMMA_AIR_NM` = 434.0472 nm (NIST, air) with HWHM `HGAMMA_WING_HWHM_NM` = 0.03 nm,
  exponent `HGAMMA_WING_POWER` = 2, normalised to 1 at `HGAMMA_WING_REF_NM` = 0.25 nm
  from the line centre (433.80 nm, the red edge of set C's window), on the shifted grid
  like the lines. Adds `base_slope` and `base_hg` (start 0.05, bounds [0, 10]) after the
  production seven, in that order. In the BH window the wing is the far wing, ~ d^-2;
  the measured H-gamma profiles follow d^-2 from 0.03 to 0.3 nm (paper repo S12).
- `EXTRA_PARAM_SPECS` gains `base_hg`; `BASELINES` gains `"hgamma"`.
- `docs/phys.md`: the switch in the table; the extra-parameter order.
- `tests/test_hgamma_baseline.py`: 7 tests (switch listed, set C's extra parameters
  unchanged; parameter order and bounds; the wing's normalisation, symmetry, Lorentzian
  ratio, monotonic convex rise across the window; zero wing equals the linear model
  bit for bit and a wing adds exactly `base_hg * L`; the wing on the shifted grid;
  recovery of T and the wing from a noise-free spectrum where the linear continuum is
  biased; invalid names rejected).

## Verified

- `pytest`: 148 passed, 1 skipped (the pre-existing skip). `ruff check` (psi2026 venv)
  on the changed and new files: all checks passed.
- Set C bit-identical: the paper repo's `analysis/tests/test_run_bh_batch.py` (193788
  refit against the stored production `summary.csv`) passes on this tree.

## Not done

- The wing exponent and width are module constants, not fit parameters (the far wing
  in the window depends on the amplitude only; S12 checked the exponent on the data).
- No version bump (as S5a-S5h).
