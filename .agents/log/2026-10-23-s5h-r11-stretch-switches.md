# Tied (1-1) R(7)/R(8) and wavelength-stretch switches (session S5h)

**Date:** 2026-10-23 (JST)
**Goal:** the paper repo's second Reviewer 1 persona read
(`Paper/reviews/2026-10-23-critical-reads/reviewer-1-persona-2.md`, demand D6) asks for
two refits as Table S2 rows: the (1-1) R(7) and R(8) amplitudes tied to the Q branch
through T_rot and the Hoenl-London factors (one free v'=1 scale, the 0.0037 nm position
offset removed), and a linear wavelength stretch. Add both as switches whose defaults
keep set C and production unchanged; the paper repo runs them
(`analysis/scripts/s5h_run_variants.py`); results note
`Paper/reviews/2026-10-22-revision-design/s5h-last-checks.md` there.

## Changed (commit `638e138`)

- `physics.py`, `BHModel` switches (defaults unchanged):
  - `r11_lines="tied"` (needs `v_max=0`, `rot_components=1`): the 11BH (1-1) R(7), R(8)
    areas are `1e8 C S_v1 g (2N'+1) exp(-dF_e(v'=1,N')/kT) A(1,N',N'') h nu / 4 pi`
    with N' = N'' + 1, the e-level energies of v' = 1 above N' = 1 and `A_coeff`'s
    R-branch Hoenl-London factor (`BHModel.r11_tied_areas`); one free scale `S_v1` =
    n(v'=1,N'=1)/n(v'=0,N'=1), start 0.4, bounds [0, 5]; positions F&B Table I
    (`load_band_wavelengths(1)`: 433.644051 / 433.331292 nm). `I_R7`, `I_R8` stay in the
    signature, are ignored, and the new `BHModel.fixed_params` = `{"I_R7": 0, "I_R8": 0}`.
    With `isotopologue="natural"` the 10BH (1-1) lines follow the tied areas.
  - `stretch=True`: extra parameter `stretch` (eps, bounds +-0.02): the lines sit on
    `x + dx + eps (x - STRETCH_REF_NM)`, `STRETCH_REF_NM = 433.475`; the width is scaled
    by 1 + eps so the width on the measured grid stays `w_inst`; baseline unchanged.
  - New constants `R11_LINE_MODES`, `R11_MODEL_NM` (the production R(7)/R(8) positions,
    now named; values unchanged), `STRETCH_REF_NM`; `EXTRA_PARAM_SPECS` gains `S_v1`,
    `stretch`.
- `fit.py`: `BHFitter` eliminates the parameters a model lists in `fixed_params` (held
  at their values, error and covariance 0), as it does a fixed `w_inst`; `set_bounds`
  exempts them from the strict-ordering check; `describe_constraints()` reports
  `model_fixed`. The previous code path runs unchanged whenever a model declares no
  fixed parameters (production).
- `docs/phys.md`: thirteen switches; the extra-parameter order; `fixed_params`.
- `tests/test_r11_stretch_switches.py`: 20 tests (defaults; invalid values; the tied
  positions against F&B Table I and the 0.0037 nm offset; the tied areas against the
  formula, linear in C and S_v1, R(8)/R(7) rising with T; the tied model as Q band plus
  two lines; recovery of T and S_v1 with I_R7 = I_R8 = 0 and zero errors; set_bounds;
  natural boron; the stretch as a regridding with the measured width kept, peak
  position and FWHM of an isolated line; recovery of eps where the plain fit cannot;
  both switches together).

## Verified

- `pytest`: 141 passed, 1 skipped (the pre-existing skip). `ruff check` (psi2026 venv's
  ruff) on the changed and new files: all checks passed.
- Set C bit-identical: the paper repo's `analysis/tests/test_run_bh_batch.py` (193788
  refit against the stored production `summary.csv`) passes on this tree.
- Full-dataset runs in the paper repo (8,638 fits each, 0 errors): tied -0.5 % / -0.7 %
  (3.6 / 3.9 m), fitted S_v1 0.33 / 0.40; stretch +0.3 % / +0.2 %, median eps
  +1.9e-3 (0.3 pm at Q(18) and Q(21)).

## Not done

- No switch that moves the R(7)/R(8) positions while keeping the amplitudes free, so the
  shares of the tie and the position change are not separated.
- No version bump (as S5a-S5e).
