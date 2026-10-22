# Model-refinement switches for the NME revision (session S5e)

**Date:** 2026-10-23 (JST)
**Goal:** the paper repo's critical read in the persona of Reviewer 1
(`Paper/reviews/2026-10-23-critical-reads/reviewer-1-persona.md`, findings 1-8)
names model refinements that set C does not test: 10BH, the unidentified window
features F1/F3, the line width, a curved continuum, the H2 R0/R1 line and a
second rotational temperature. Add them as switches whose defaults keep set C
and production unchanged, so the paper repo can run them as sensitivity
variants (`analysis/scripts/s5e_*.py`); results note
`Paper/reviews/2026-10-22-revision-design/s5e-model-refinements.md` there.

## Changed

- New `isotopes.py`: 10BH data from Fernando & Bernath (1991), read from 300 dpi
  renders of the library PDF:
  - Table V measured 10BH lines (0-0 and 1-1, P/Q/R), Table VII 10BH constants
    (`BH10_A_FB91`, `BH10_X_FB91`), Table VIII 11BH Dunham coefficients of both
    states with the Lambda-doubling `Q_kl`;
  - `dunham_term` with mass scaling (`Y_kl rho^(k+2l)`, `Q_kl rho^(k+2l+2)`,
    `RHO_10BH = sqrt(mu11/mu10) = 1.004164` from AME2020 masses);
  - `bh10_q_wavenumbers(scheme)`: `"fb91"` (Table V where measured, Table VII
    elsewhere) or `"mass_scaled"`; `bh10_upper_f_energy`, `bh10_r11_wavenumbers`,
    `wavenumber_to_air_nm` (the 11BH tables' `n_air`).
  - `BH10_FRACTION = 0.199`, `BH10_OVER_BH11 = 0.199/0.801`.
- `physics.py`, `BHModel` switches (defaults unchanged):
  - `isotopologue="natural"` (needs F&B energies, `v_max=0`): 10BH Q branch at
    `0.248 * Z11/Z10` with its own f-level energies and positions, and 10BH
    (1-1) R(7)/R(8) at 0.248 times the free R(7)/R(8) amplitudes;
    `bh10_positions` picks the scheme. New `rot_partition`, `bh10_spectrum`,
    `q_band`.
  - `extra_lines=("F1", "F3")`: free Gaussians at 433.100 / 433.750 nm
    (`EXTRA_LINES_NM`, rest frame, BH width), `I_F1`, `I_F3` in [0, 1].
  - `baseline="quadratic"`: adds `base_curv * (x - 433.475)^2`, bounds +-5 /nm^2
    (as the paper repo's S5d curved-continuum test).
  - `h2_lines="R2"`: the R2 line alone (for fits that exclude the R0/R1 region).
  - `rot_components=2` (needs `v_max=0`): `(1-f) n(T_rot) + f n(T_rot2)`, each
    normalised by its partition sum, nested so that `frac2 = 0` is the
    one-temperature model with the same `C`; `T_rot2` in [200, 30000] K.
  - The two unused imports at the top (`dataclass`, `BH_X`) removed.
- `fit.py`: `BHFitter(exclude_nm=[(lo, hi)], exclude_frame="rest"|"measured")`
  drops points from the fit (zero weight); in the rest frame `x + dx` a first fit
  of all points gives `dx`. `_curve_fit` factored out of `fit`. Units for the new
  extra parameters; `describe_constraints` reports the exclusions.
- `workflows/batch_fit.py`: docstring of `model_options` lists the new switches;
  fitter options go through `fitter_kwargs` (no code change).
- `docs/phys.md`: the switch table (eleven switches) and the exclusion option.
- Tests: new `tests/test_refinement_switches.py` (30 tests); in
  `test_baseline_h2_switches.py` the invalid-baseline example is now `"cubic"`.

## Verified

- Set C is bit-identical: a test pins `full_fit_model` with set C's switches to
  the inline formula, and the paper repo's `tests/test_run_bh_batch.py` (shot
  193788 refit against the stored production `summary.csv`) passes on this tree.
- Transcription: the Table VIII Dunham set reproduces the 11BH Table I Q lines
  within 0.07 cm^-1 for J <= 23, with the X-state Y04 read as -5.18e-12 (as printed
  under the 10^11 header it misses by up to 4.4 cm^-1; Table III's L_0(X) =
  -4.5e-12 agrees with the 10^12 reading). Scaled to 10BH, the band origin is
  23073.690 cm^-1 against Table VII's 23073.6860.
- `pytest`: 121 passed, 1 skipped (the pre-existing skip). `ruff check` (psi2026
  venv's ruff 0.15) on the changed and new files: all checks passed.

## Found, not changed

- **10BH high-N Q lines.** The mass-scaled constants reproduce every measured
  10BH P and R line (0-0 and 1-1) and Q(4)-Q(10) within 0.07 cm^-1 (1-1: 0.14),
  but miss Table V's Q(18), Q(19), Q(20) by -2.4, -3.6, -5.5 cm^-1
  (0.045-0.10 nm): Table V puts them 1.7-4.5 cm^-1 blue of 11BH, scaling puts
  them 0.8-1.0 cm^-1 red. Table VII (no H, L terms) fits them, with Q(20) at
  o-c 0.306 cm^-1. Table V's "Q(20)" 23066.492 equals the scaled Q(19)
  (23066.496), and its "Q(19)" 23070.091 lies 0.04 cm^-1 from the 10BH 1-1 R(8)
  23070.053, so a misassignment is possible; not settled here. Both schemes are
  offered.
- The 10BH head (Q(4)-Q(10)) lies 0.31-0.36 cm^-1 (0.006-0.007 nm) red of the
  11BH lines, not "within 0.002 nm" as the paper repo's S7 note says.
- The hard-coded 11BH R(7)/R(8) positions (+0.0037 nm against F&B, S5a log) are
  unchanged; the 10BH R(7)/R(8) use F&B Table V converted like the Q lines.

No version bump (defaults unchanged); nothing pushed.
