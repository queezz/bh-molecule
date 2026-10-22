# Changelog

Dated record of shipped work, newest first; per-session detail in `log/`.

## 2026-10-23

- S5e refinement switches for the NME revision (Reviewer 1 critical read):
  `BHModel(isotopologue="natural", bh10_positions="fb91"|"mass_scaled")` (10BH at
  0.199 of BH, new `isotopes.py` with F&B Tables V, VII, VIII), `extra_lines=("F1",
  "F3")` (free lines at 433.100/433.750 nm), `baseline="quadratic"`,
  `h2_lines="R2"`, `rot_components=2` (two Boltzmann components), and
  `BHFitter(exclude_nm=..., exclude_frame=...)`. Defaults and set C unchanged.
  Found: F&B Table V's 10BH Q(18)-Q(20) disagree with mass scaling by 2.4-5.5
  cm^-1. See `log/2026-10-23-s5e-refinement-switches.md`.

## 2026-10-22

- S5c switch `BHModel(h2_r01_ratio=...)`: ties the H2 R0/R1 amplitude at 433.2619 nm
  to the R2 amplitude at 433.5522 nm (one free amplitude), with Dieke's ratio
  `DIEKE_H2_R01_OVER_R2 = 1.29/1.20`, for the paper's C-prime systematic. Default
  unchanged. See `log/2026-10-22-s5c-tied-h2-lines.md`.
- S5b switches for the NME revision: `BHModel(baseline="linear", h2_lines=True)`
  (linear baseline about 433.475 nm; H2 Y 1 -> 2B 0 lines at 433.2619 and
  433.5522 nm as free nuisance lines), `BHFitter` support for model-declared extra
  parameters, `run_bh_batch(model_options=..., save_grid=...)`, and a fix that lets
  `set_bounds` work with a fixed `w_inst`. Defaults unchanged. See
  `log/2026-10-22-s5b-baseline-switches.md`.
- S5a model switches (commit `70a0783`): `upper_energies`, `v_max`,
  `band_positions`, `T_vib`, with the Fernando–Bernath 1991 constants. See
  `log/2026-10-22-s5a-model-switches.md`.
