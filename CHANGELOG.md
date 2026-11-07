# Changelog

Dated record of the releases, newest first.

## 0.2.0 - 2026-11-07

- Add `BHModel` keyword switches for the BH sensitivity study, all defaulting
  to the production model (outputs bit-identical): `upper_energies`
  (`"fernando_bernath"`: Fernando–Bernath 1991 term values, f level for Q
  lines), `v_max` (0 = the 0–0 band only), `band_positions` and `T_vib`, with
  the F&B Table III constants `BH_A_FB91`, `BH_X_FB91` and
  `dataio.load_band_wavelengths(v)`.
- Add `baseline="linear"` and `h2_lines=True` (the two H2 Y 1 -> 2B 0 lines at
  433.2619 and 433.5522 nm as free features), the model of the revised
  analysis (set C); `BHFitter` appends a model's extra parameters to
  `param_names`, `p0` and `bounds`, and `run_bh_batch` takes `model_options`
  and `save_grid`.
- Fix `BHFitter.set_bounds`, which rejected every call when `fix_w_inst=True`.
- Add `h2_r01_ratio` to tie the H2 R0/R1 amplitude to the resolved R2 line
  (`DIEKE_H2_R01_OVER_R2` = 1.29/1.20).
- Add the refinement switches `isotopologue="natural"` with `bh10_positions`
  (10BH from the new `isotopes` module, F&B Tables V, VII and VIII),
  `extra_lines=("F1", "F3")`, `baseline="quadratic"`, `h2_lines="R2"` and
  `rot_components=2`, and `BHFitter(exclude_nm=..., exclude_frame=...)`.
- Add `r11_lines="tied"` (the (1–1) R(7)/R(8) areas from `T_rot` and the
  Hönl–London factors with one free scale `S_v1`) and `stretch=True` (a linear
  wavelength stretch about 433.475 nm); `BHFitter` eliminates the parameters a
  model lists in `fixed_params`.
- Add `baseline="hgamma"` (the far wing of H-gamma with free amplitude
  `base_hg`) and `baseline="fixed"` with `BHModel.set_continuum` and
  `BHFitter(continuum=...)` for a continuum computed before the fit and held.
- Add the Zenodo DOI badge to the README.
