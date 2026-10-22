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

`BHModel(...)` takes four keyword switches. Their defaults are the production model above; any other value is a sensitivity variant, not a replacement.

| Switch | Default | Alternative |
|---|---|---|
| `upper_energies` | `"huber_herzberg"`: `energy` with `BH_A` | `"fernando_bernath"`: Fernando & Bernath (1991) Eq. (1) with the per-level Table III constants `BH_A_FB91`; f level for Q lines, e level for P/R lines |
| `v_max` | `2`: $v'=0,1,2$ co-added | `0`: the 0–0 band only (the R7/R8 features of `full_fit_model` stay) |
| `band_positions` | `"v00"`: every band at the 0–0 line positions | `"own"`: each band at its measured positions (F&B Table I: `11BH_v11.csv`, `11BH_v22.csv`); lines not observed are skipped |
| `T_vib` | `None`: each band normalised at its own $N=1$ | a temperature [K]: band $v$ weighted by $\exp[-(T_v-T_0)/k_B T_{\rm vib}]$ with the F&B band origins |

The Fernando–Bernath term value is

$$
F_{e/f}(J) = T_v + B_v x - D_v x^2 + H_v x^3 + L_v x^4 \pm \tfrac12 x\,(q_v + q_{Dv}\,x), \qquad x = J(J+1),
$$

with "+" for e and "−" for f. A Q line ($\Delta J = 0$) joins an f level of A$\,^1\Pi$ to an e level of X$\,^1\Sigma^+$, so the Q-branch populations use $F_f$. With the f sign the tabulated constants reproduce every observed Q line of the 0–0, 1–1 and 2–2 bands to 0.03 cm⁻¹; the e sign misses by up to 17 cm⁻¹.

With `band_positions="own"` the 1–1 and 2–2 Q branches lie at 436.7–439.9 nm and 443.4–446.8 nm (air), outside the 433.05–433.90 nm fit window, so in that window the mixture reduces to the 0–0 model whatever `T_vib` is.
