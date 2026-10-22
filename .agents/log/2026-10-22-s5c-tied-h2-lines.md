# Tied H2 line amplitudes for the C-prime systematic (session S5c)

**Date:** 2026-10-22 (JST), late evening
**Goal:** the paper repo adopted set C (`BHModel(upper_energies="fernando_bernath",
v_max=0, baseline="linear", h2_lines=True)`, fit window 433.05-433.80 nm) as the NME
revision's fits. Its H2 Y 1 -> 2B 0 R0/R1 line (433.2619 nm) lies 0.002 nm from BH
Q(17) and cannot be separated from the low-N Q lines; S5c runs C-prime, set C with
that amplitude tied to the resolved R2 line (433.5522 nm), over the whole dataset and
quotes the difference as a systematic.
**Brief:** the owner's decision of 2026-10-22 in the paper repo,
`<Dropbox>/10-Research/50-Conferences/2026-05-17-PSI/Paper/reviews/2026-10-22-revision-design/REVISION-DESIGN.md`
section 11 (S5c row); results note `s5c-promote-set-c.md` in the same folder.

## Changed

- `physics.py`: `BHModel(..., h2_r01_ratio=None)`. With `h2_lines=True` and a positive
  ratio r, the R0/R1 amplitude is `r * I_H2_R2` and only `I_H2_R2` is fitted
  (`extra_param_names` becomes `base_slope, I_H2_R2`). Raises without `h2_lines=True`
  or for a non-positive, non-finite or boolean ratio. Default `None` keeps both
  amplitudes free, so set C and the production model are unchanged.
- `DIEKE_H2_R01_OVER_R2 = 1.29 / 1.20` (1.075): Dieke's photoelectric intensities I4 of
  4332.619 A (R0 with R1) and 4335.522 A (R2), Crosswhite (1972) Table T, pp.
  T104-T105, as transcribed by the paper repo's S7b. The book's introduction (p. x)
  calls I4 "an attempt to make the intensity values more quantitative" from
  photoelectric measurements in the 77 K low-pressure discharge, and says the eye
  estimates I1-I3 are on an approximately logarithmic scale; it does not say whether I4
  is linear. Read as linear the ratio is 1.075 (the reading S5b's attribution test
  used); read as log10 it would be 10^0.09 = 1.23. Both are below the median 1.6 that
  the free fit takes, so the linear reading is the larger departure from set C.
- `workflows/batch_fit.py`: the `model_options` docstring names the new switch;
  `run_bh_batch` passes it through unchanged.
- `docs/phys.md`: the switch table gains `h2_r01_ratio` (seven switches now).
- `tests/test_baseline_h2_switches.py`: 9 new tests (the ratio constant; tied model
  equals the free model at `I_R01 = r * I_R2`; one extra amplitude; invalid ratios
  raise; synthetic recovery of T and the shared amplitude; `run_bh_batch` writes
  `I_H2_R2` and no `I_H2_R01` column when tied).

## Verified

- `pytest`: 91 passed, 1 skipped (the pre-existing skip). `ruff check` on the changed
  files: only the two pre-existing unused imports at the top of `physics.py`.
- In the paper repo, `analysis/scripts/run_bh_batch.py` (set C) refits shot 193788
  identically (difference 0.0 in every column) with this tree.

## Next

- The paper repo runs `--variant Cprime` for all shots and tabulates
  T_rot(C')/T_rot(C) per channel and configuration.
- Open: whether Dieke's I4 is a linear or a logarithmic intensity; a calculated
  Y 1 -> 2B 0 line strength would replace the lab ratio.

No version bump (defaults unchanged); nothing pushed.
