# BH Physics Explainer

This page summarizes the physical model used in `bh_molecule.physics.BHModel`. It is organized by method; each section states the mathematical definition and key assumptions.

## `energy`

Rovibronic term value $E(v,N)$ (in cm⁻¹) for a given electronic state, using a Dunham-like expansion truncated to cubic vibrational and quartic (centrifugal distortion) rotational terms:

$$
E(v,N)=T_e + G(v) + F_v(N),
$$

with

$$
\begin{aligned}
G(v) &= \omega_e\,(v+\tfrac12) \;-\; \omega_e x_e\,(v+\tfrac12)^2 \;+\; \omega_e y_e\,(v+\tfrac12)^3, \\
B_v  &= B_e \;-\; \alpha_e\,(v+\tfrac12), \\
D_v  &= D_e \;-\; \beta_e\,(v+\tfrac12), \\
F_v(N) &= B_v\,N(N+1)\;-\; D_v\,[N(N+1)]^2 .
\end{aligned}
$$

**Notes.** $T_e$ is the electronic term origin; $\omega_e,\omega_e x_e,\omega_e y_e$ are vibrational constants; $B_e,\alpha_e,D_e,\beta_e$ are rotational and centrifugal-distortion constants, all state-specific.

---

## `line_profile`

Lines are modeled as Gaussian in wavelength with Doppler and instrumental widths added in quadrature (FWHM):

$$
\Delta\lambda = \sqrt{\Delta\lambda_D^2 + \Delta\lambda_{\rm inst}^2}.
$$

The Doppler FWHM at temperature $T$ (for emitter mass $m$) follows the standard expression

$$
\Delta\lambda_D = \lambda \sqrt{\frac{8\ln 2\,k_B T}{m c^2}} .
$$

The corresponding standard deviation is

$$
\sigma=\frac{\Delta\lambda}{2\sqrt{2\ln 2}} ,
$$

and the normalized profile at wavelength $x$ is

$$
g(x)=\frac{1}{\sqrt{2\pi}\,\sigma}\exp\!\left[-\frac{(x-\lambda)^2}{2\sigma^2}\right].
$$

---

## `A_coeff`
Einstein $A_{ul}$ for a rovibronic line.

For the BH $A\,^1\Pi \rightarrow X\,^1\Sigma^+$ system, line Einstein coefficients are formed from **band** $A_{\rm vib}(v')$ (per upper vibrational level) and Hönl–London rotational factors:

$$
A_{ul}(v', N_2 \to N_1)=
\frac{A_{\rm vib}(v')\, H_{\rm HL}(N_2,\Delta N)}{2N_2+1}
$$

$$
\Delta N = N_2 - N_1 \in \{-1,0,+1\}, \quad
H_{\mathrm{HL}} =
\begin{cases}
    N_2/2, & \Delta N = -1 \quad (\text{P}) \\
    (2N_2+1)/2, & \Delta N = 0 \quad (\text{Q}) \\
    (N_2+1)/2, & \Delta N = +1 \quad (\text{R})
\end{cases}
$$

**Notes.** $H_{\rm HL}$ are the case-(a) factors appropriate to a $^{1}\Pi \to {}^{1}\Sigma^+$ transition and partition intensity among P/Q/R branches ($\Delta N=-1,0,+1$). Electronic degeneracies, Λ-doubling, parity, and nuclear-spin substructure are neglected here and can be incorporated via additional weights if needed.

---

## `spectrum`

For a chosen rotational branch (P/Q/R) on a wavelength grid $x$, the model sums lines whose centers $\lambda_{v'N_2\to v''N_1}$ come from **tabulated wavelengths** (X-state only fixes positions) while **A-state** level energies and populations set intensities.

Per-line contribution near $\lambda_0$ is

$$
I_\ell(x)=\frac{h\nu_0}{4\pi}\; n'(v',N_2)\; A_{ul}(v',N_2\!\to\!N_1)\; g_\lambda(x),
$$

where

$n'(v',N_2)$ : upper-state populations (Boltzmann at $T_{\rm rot}$, scaled by an overall factor $C$)

$A_{ul}$ : as above

$g_\lambda$ : Gaussian line profile with total FWHM $\Delta\lambda$ (Doppler + instrumental).


The total branch spectrum is the sum over included $(v',N_2\to v'',N_1)$ within specified bounds.

---

## `full_fit_model`
Composite model for the 433 nm window.

The forward model used around 433 nm is the sum of:

1. the BH **Q-branch** spectrum (as in `spectrum`, with branch fixed to Q), evaluated on a shifted grid $x+\delta x$ and rescaled numerically by $10^8$ for conditioning, plus
2. two **auxiliary Gaussian** lines at fixed wavelengths $\lambda_{R7}=433.64776244\,\mathrm{nm}$ and $\lambda_{R8}=433.33500584\,\mathrm{nm}$ with amplitudes $I_{R7}, I_{R8}$ and the **same** instrumental width as the BH part (Doppler set to zero for these), and
3. a constant baseline $b$.

Putting it together:

$$ \begin{aligned} y(x) ={}& 
10^{8}\,S_{\mathrm{Q}}\!\left(x+\delta x;\, C, T_{\mathrm{rot}}, w_{\mathrm{inst}}\right) \\ &
{}+ I_{R7}\, g\!\left(x; \lambda_{R7}, w_{\mathrm{inst}}\right) \\ &
     {}+ I_{R8}\, g\!\left(x; \lambda_{R8}, w_{\mathrm{inst}}\right) + b. \end{aligned} $$

**Notes.** The X-state affects only line centers (via tables). If Doppler broadening is also required for the auxiliary features, replace their $g$ with the full Doppler + instrument $\Delta\lambda$.

---

## Model switches (sensitivity studies)

`BHModel(...)` takes thirteen keyword switches. Their defaults are the production model above; any other value is a sensitivity variant, not a replacement.

| Switch | Default | Alternative |
|---|---|---|
| `upper_energies` | `"huber_herzberg"`: `energy` with `BH_A` | `"fernando_bernath"`: Fernando & Bernath (1991) Eq. (1) with the per-level Table III constants `BH_A_FB91`; f level for Q lines, e level for P/R lines |
| `v_max` | `2`: $v'=0,1,2$ co-added | `0`: the 0–0 band only (the R7/R8 features of `full_fit_model` stay) |
| `band_positions` | `"v00"`: every band at the 0–0 line positions | `"own"`: each band at its measured positions (F&B Table I: `11BH_v11.csv`, `11BH_v22.csv`); lines not observed are skipped |
| `T_vib` | `None`: each band normalised at its own $N=1$ | a temperature [K]: band $v$ weighted by $\exp[-(T_v-T_0)/k_B T_{\rm vib}]$ with the F&B band origins |
| `baseline` | `"constant"`: the constant `base` | `"linear"`: `base + base_slope * (x - 433.475)` on the measured (unshifted) grid, so `base` is the value at the centre of the production window; adds the fit parameter `base_slope` [1/nm], start 0, bounds ±1; `"quadratic"`: the same plus `base_curv * (x - 433.475)**2`, adds `base_slope` and `base_curv` [1/nm²], start 0, bounds ±5; `"hgamma"`: the linear form plus `base_hg * L(x + dx)`, the far wing of H$\gamma$ (`BHModel.hgamma_wing`: a Lorentzian at 434.0472 nm with HWHM 0.03 nm, normalised to 1 at 0.25 nm from the line centre, i.e. at 433.80 nm), adds `base_slope` and `base_hg`, start 0.05, bounds [0, 10]; `"fixed"`: a continuum computed before the fit and held, `BHModel.set_continuum(x, values)` on the measured grid (interpolated linearly, never extrapolated), `base` held at 0 (`fixed_params`), no continuum parameter fitted; `BHFitter(continuum=f)` calls `f(vis, frame, channel) -> (x, values)` for each spectrum and sets it |
| `h2_lines` | `False` | `"R2"`: only the R2 line (`I_H2_R2`), for fits that leave the R0/R1 region out of the data (`BHFitter(exclude_nm=...)`); `True`: the H$_2$ lines Y 1 → 2B 0 R0/R1 (433.2619 nm) and R2 (433.5522 nm), Dieke's air wavelengths (Crosswhite 1972, Table T), as free-amplitude Gaussians with the BH line width on the shifted grid `x + dx`; adds `I_H2_R01`, `I_H2_R2`, start 1e-3, bounds [0, 1] like `I_R7`, `I_R8` |
| `h2_r01_ratio` | `None`: the two H$_2$ amplitudes are free | a positive number $r$ (only with `h2_lines=True`): the R0/R1 amplitude is tied to R2, $I_{\rm R0/R1} = r\,I_{\rm R2}$, so only `I_H2_R2` is fitted. R0/R1 lies on BH Q(17) (0.002 nm away) and the low-$N$ Q lines, so a free amplitude there can take BH intensity; the tie sets it from the resolved R2 line. `DIEKE_H2_R01_OVER_R2` $= 1.29/1.20 = 1.075$ is the ratio of Dieke's photoelectric intensities $I_4$ (Crosswhite 1972, Table T, pp. T104–T105), read as linear; the eye estimates give 0.9/0.8 and 1.0/1.0 |
| `isotopologue` | `"11BH"` | `"natural"` (needs `upper_energies="fernando_bernath"`, `v_max=0`): adds $^{10}$BH at its natural share (0.199 of BH, `BH10_OVER_BH11` = 0.248 of $^{11}$BH) with the same $T_{\rm rot}$: the $^{10}$BH 0–0 Q lines with their own f-level energies, each isotopologue normalised by its own partition sum $Z=\sum_{N=1}^{22}(2N+1)e^{-\Delta F_N/kT}$, and the $^{10}$BH 1–1 R(7)/R(8) lines (F&B Table V) at 0.248 times the free $^{11}$BH amplitudes. No new fit parameters |
| `bh10_positions` | `"fb91"`: F&B Table V where measured, their Table VII $^{10}$BH constants elsewhere | `"mass_scaled"`: the $^{11}$BH Dunham set of F&B Table VIII scaled with $\rho=\sqrt{\mu_{11}/\mu_{10}}$ ($Y_{kl}\rho^{k+2l}$, $\Lambda$-doubling $Q_{kl}\rho^{k+2l+2}$). The two agree within 0.0015 nm to Q(11) and for every measured P and R line; Table V's Q(18)–Q(20) lie 2.4–5.5 cm⁻¹ blue of the scaled values (`bh_molecule.isotopes`) |
| `extra_lines` | `()` | `("F1", "F3")`: the unidentified features at 433.100 and 433.750 nm (rest frame, `EXTRA_LINES_NM`) as free Gaussians with the BH width on the shifted grid; adds `I_F1`, `I_F3`, bounds [0, 1] |
| `rot_components` | `1` | `2` (needs `v_max=0`): upper-level populations $(1-f)\,n(T_{\rm rot}) + f\,n(T_{\rm rot,2})$, each normalised by $Z(T)$ and scaled so that $f=0$ is the one-temperature model with the same `C`; adds `T_rot2` [K] (start 6000, bounds 200–30 000) and `frac2` (0.1, [0, 1]) |
| `r11_lines` | `"free"`: the $^{11}$BH 1–1 R(7) and R(8) lines with free amplitudes `I_R7`, `I_R8` at 433.64776 / 433.33501 nm (`R11_MODEL_NM`, 0.0037 nm red of F&B Table I) | `"tied"` (needs `v_max=0`, `rot_components=1`): their areas follow from the common $T_{\rm rot}$, $1\mathrm{e}8\,C\,S_{v1}\,g\,(2N'+1)\,e^{-\Delta F_e(1,N')/kT}\,A(1,N',N'')\,h\nu/4\pi$ with $N'=N''+1$, the R-branch Hönl–London factor $(N'+1)/2$ of `A_coeff`, and the e-level energies of $v'=1$ above $N'=1$ (`BHModel.r11_tied_areas`); one free scale `S_v1` $= n(v'=1,N'=1)/n(v'=0,N'=1)$ (start 0.4, bounds [0, 5]); positions F&B Table I (433.644051 / 433.331292 nm). `I_R7`, `I_R8` stay in the signature, are ignored, and `BHModel.fixed_params` makes `BHFitter` hold them at 0 |
| `stretch` | `False` | `True`: a linear wavelength stretch `stretch` ($\varepsilon$, start 0, bounds ±0.02): the model lines sit on `x + dx + eps * (x - 433.475)` (`STRETCH_REF_NM`), a relative dispersion error of the measured grid, and the Gaussian width is scaled by $1+\varepsilon$ so that the width on the measured grid stays `w_inst`; the baseline stays on the measured grid |

The Fernando–Bernath term value is

$$
F_{e/f}(J) = T_v + B_v x - D_v x^2 + H_v x^3 + L_v x^4 \pm \tfrac12 x\,(q_v + q_{Dv}\,x), \qquad x = J(J+1),
$$

with "+" for e and "−" for f. A Q line ($\Delta J = 0$) joins an f level of A$\,^1\Pi$ to an e level of X$\,^1\Sigma^+$, so the Q-branch populations use $F_f$. With the f sign the tabulated constants reproduce every observed Q line of the 0–0, 1–1 and 2–2 bands to 0.03 cm⁻¹; the e sign misses by up to 17 cm⁻¹.

With `band_positions="own"` the 1–1 and 2–2 Q branches lie at 436.7–439.9 nm and 443.4–446.8 nm (air), outside the 433.05–433.90 nm fit window, so in that window the mixture reduces to the 0–0 model whatever `T_vib` is.

The extra fit parameters follow the seven of `full_fit_model` in the order `base_slope`, `base_curv` or `base_hg`, `I_H2_R01`, `I_H2_R2`, `I_F1`, `I_F3`, `T_rot2`, `frac2`, `S_v1`, `stretch` (`BHModel.extra_param_names`, only those the switches add; `I_H2_R2` alone when `h2_r01_ratio` ties the two lines or `h2_lines="R2"`). `BHFitter` appends them to `param_names` with the model's start values and bounds; `run_bh_batch(..., model_options={...})` builds the model with any of the switches and writes the extra parameters as `summary.csv` columns. The fit-window upper limit is the upper edge of `run_bh_batch`'s `bh_fit_range` (e.g. `(433.05, 433.80)` leaves out the H$\gamma$ wing); the normalisation, taken inside `bh_scale_range`, does not depend on it. The offset `base` keeps the fitter's bound (`base_tight`, `base_bound`) unless `bounds` overrides it, e.g. `{"lower": {"base": -0.05}, "upper": {"base": 0.5}}`. `BHFitter(exclude_nm=[(lo, hi), ...], exclude_frame="rest")` gives zero weight to wavelength ranges: in the rest frame `x + dx` (a first fit of all points gives `dx`) or on the measured grid; the returned `x`, `y`, `yfit` and hence R² cover the points fitted. Parameters a model lists in `fixed_params` are eliminated from the fit like a fixed `w_inst` (value held, error and covariance 0); `describe_constraints()` reports them as `model_fixed`.
