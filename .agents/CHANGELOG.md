# Changelog

Dated record of shipped work, newest first; per-session detail in `log/`.

## 2026-10-22

- S5b switches for the NME revision: `BHModel(baseline="linear", h2_lines=True)`
  (linear baseline about 433.475 nm; H2 Y 1 -> 2B 0 lines at 433.2619 and
  433.5522 nm as free nuisance lines), `BHFitter` support for model-declared extra
  parameters, `run_bh_batch(model_options=..., save_grid=...)`, and a fix that lets
  `set_bounds` work with a fixed `w_inst`. Defaults unchanged. See
  `log/2026-10-22-s5b-baseline-switches.md`.
- S5a model switches (commit `70a0783`): `upper_energies`, `v_max`,
  `band_positions`, `T_vib`, with the Fernando–Bernath 1991 constants. See
  `log/2026-10-22-s5a-model-switches.md`.
