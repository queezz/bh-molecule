# Model switches for the NME revision (session S5a)

**Date:** 2026-10-22 (JST)
**Goal:** answer Reviewer 1 items 7.2 and 7.3 of the NME paper (BH A–X rotational
temperature in the LHD divertor) with numbers: add sensitivity variants of the
forward model behind switches whose defaults are the production model, test each,
and refit a representative set from the paper repo.
**Brief:** from the paper repo's revision design,
`<Dropbox>/10-Research/50-Conferences/2026-05-17-PSI/Paper/reviews/2026-10-22-revision-design/REVISION-DESIGN.md`
(R1.6, R1.7, §9). The results note is `s5a-model-sensitivity.md` in the same folder.

This repo had no `AGENTS.md` and no `.agents/`; the brief asked for a dated log
entry here, so this file starts `.agents/log/`.

## Changed

- `constants.py`: `LevelConstants` (Fernando & Bernath 1991 Eq. (1), e/f sign) and
  `BH_A_FB91`, `BH_X_FB91` with uncertainties (`*_UNC`) for v = 0, 1, 2, transcribed
  from their Table III (printed p. 398); the source comment gives the DOI, pages and
  the parity convention (p. 401: "+" e, "−" f).
- `physics.py`: `BHModel(..., upper_energies, v_max, band_positions, T_vib)`.
  Defaults `"huber_herzberg"`, `2`, `"v00"`, `None` are the production model.
  - `upper_energies="fernando_bernath"`: F&B term values, f level for Q, e for P/R.
  - `v_max=0`: 0–0 band only; `full_fit_model` keeps the R(7)/R(8) features.
  - `band_positions="own"`: each band at its Table I positions; unobserved lines skipped.
  - `T_vib=<K>`: band weight `exp(-(T_v - T_0)/(k T_vib))` with F&B band origins.
  - New helpers `upper_energy`, `vib_weight`; `spectrum(v_max=None)` takes the model's.
- `dataio.py`: `load_band_wavelengths(v)`; `load_v00_wavelengths()` now calls it
  (same output).
- `docs/phys.md`: a "Model switches" section.
- `tests/test_model_switches.py`: 15 tests.

## Verified

- Defaults: `full_fit_model` and `spectrum` outputs bit-identical to the pre-change
  code on a 4000-point grid at five temperatures; a test pins the Q spectrum to an
  inline copy of the original formula.
- Transcription: Table III with the f sign reproduces every Table I Q line of the
  0–0, 1–1, 2–2 bands within 0.03 cm⁻¹ (0.10 for 0–0 Q(25)); e sign misses by up
  to 17 cm⁻¹. The packaged `11BH_v11.csv`, `11BH_v22.csv` match Table I (read
  against the library PDF, p. 395).
- `pytest`: 68 passed, 1 skipped (the pre-existing skip of the missing prototype).
- Refits in the paper repo reproduce the production `summary.csv` values exactly
  with the defaults.

## Found, not changed

- `spectrum()` looks up P and R positions by the upper N2, but the tables are indexed
  by lower J, so P/R lines are one J off. Production uses Q only (exact).
- `full_fit_model`'s hard-coded R(7), R(8) positions (433.64776, 433.33501 nm) are
  0.0037 nm above the Table I values converted with `n_air` (433.64405, 433.33129).
- 1-1 R(9) at 433.0407 nm lies 0.009 nm below the 433.05 nm window edge; its wing
  reaches the first pixels and is not modelled.
- `run_bh_batch` builds `BHModel()` itself; a dataset refit with a switch (S5b) needs
  a `model_options` pass-through there.

No version bump (defaults unchanged); nothing pushed.
