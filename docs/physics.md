# Physics Module
Documentation extracted from docstrings in `bh_molecule.physics`.
*(Generated on 2025-08-16 21:57:34)*

## BHModel

Core BH spectrum model; file I/O is done elsewhere (dataio).

### `energy`

Rovibronic term value $E(v,N)$ (in cm⁻¹) for a given electronic state.

The model uses a Dunham-like expansion truncated to
cubic vibrational terms and quartic (centrifugal distortion) in rotation:

$$
E(v,N) \;=\; T_e \;+\; G(v) \;+\; F_v(N),
$$

with

$$
\begin{aligned}
G(v) &= \omega_e\,(v+\tfrac12) - \omega_e x_e\,(v+\tfrac12)^2 + \omega_e y_e\,(v+\tfrac12)^3, \\
B_v  &= B_e - \alpha_e\,(v+\tfrac12), \\
D_v  &= D_e - \beta_e\,(v+\tfrac12), \\
F_v(N) &= B_v\,N(N+1) - D_v\,\big[N(N+1)\big]^2 .
\end{aligned}
$$

#### Parameters
`v` : int
    Vibrational quantum number $v \ge 0$.

`N` : int
    Rotational quantum number (spinless). For singlet states,
    $J \approx N$; fine/Λ-doubling and spin-rotation are neglected here.

`c` : MolecularConstants
    Parameter set for the electronic state (fields in cm⁻¹):
    `T_e, omega_e, omega_e_x_e, omega_e_y_e, B_e, alpha_e, D_e, beta_e`.

#### Returns
float
    Rovibronic term value $E(v,N)$ in **cm⁻¹**.

#### Notes
- Truncation: includes up to $(v+\tfrac12)^3$ in $G(v)$ and
  $[N(N+1)]^2$ in $F_v(N)$. Higher-order terms (e.g. $H_v$
  or additional Dunham coefficients) are omitted.
- Parity/Λ-doubling, spin-rotation, hyperfine, and electronic spin are
  ignored (appropriate for a simplified ^1Π↔^1Σ^+ treatment in this codebase).

#### Examples
```python
from bh_molecule.constants import BH_A
E01 = BHModel.energy(v=0, N=1, c=BH_A)
E11 = BHModel.energy(v=1, N=1, c=BH_A)
assert E11 > E01
```

### `line_profile`

Gaussian line profile with Doppler + instrumental broadening (FWHMs added in quadrature).

#### Parameters
`x` : array_like
    Wavelength axis in **nm**.

`wl` : float
    Line center wavelength in **nm**.

`w_inst` : float
    Instrumental full width at half maximum (FWHM) in **nm**, assumed Gaussian.

`T` : float
    Translational/kinetic temperature in **K** for Doppler broadening.

#### Returns
numpy.ndarray
    Normalized Gaussian profile sampled on `x` (units ≈ nm⁻¹; area ≈ 1 when integrated over `x`).

#### Notes
The Doppler FWHM (in nm) is computed from a compact numerical form
tailored to this model:

$$
\Delta\lambda_D \approx 7.72 \times 10^{-5}\; \lambda\,
\sqrt{\frac{T\,K_{2\mathrm{eV}}}{M_{\mathrm{BH}}}}
$$

where $\lambda$ is in nm, $K_{2\mathrm{eV}} = 8.617\times10^{-5}\,\mathrm{eV\,K^{-1}}$,
and $M_{\mathrm{BH}}$ is the BH molecular mass in amu. This is equivalent to the
standard expression

$$
\Delta\lambda_D = \lambda \sqrt{\frac{8\ln 2\,k_B T}{m c^2}}
$$

after unit conversions (nm, eV, amu). The total Gaussian FWHM is

$$
\Delta\lambda = \sqrt{\Delta\lambda_D^2 + \Delta\lambda_{\mathrm{inst}}^2},
$$

and the standard deviation is $\sigma = \Delta\lambda / (2\sqrt{2\ln 2})$.
The returned profile is

$$
g(x) = \frac{1}{\sqrt{2\pi}\,\sigma}\;\exp\!\left[-\frac{(x-\lambda)^2}{2\sigma^2}\right].
$$

Broadcasting: `wl`, `w_inst`, and `T` may be scalars or arrays
broadcastable to the shape of `x`.

#### Examples
```python
x = np.linspace(433.0, 434.0, 2001)
g = model.line_profile(x, wl=433.5, w_inst=0.02, T=0.0)  # instrument-limited
assert np.isfinite(g).all()
```

### `A_coeff`

Einstein $A_{ul}$ for a single rovibronic line of the BH
$A\,^1\Pi \rightarrow X\,^1\Sigma^+$ system.

This uses band Einstein coefficients (per upper vibrational level) and
Hönl–London factors to apportion intensity among P/Q/R rotational branches:

$$
A_{ul}(v', N_2 \to N_1)
= \frac{A_{\mathrm{vib}}(v') \, H_{\mathrm{HL}}(N_2, \Delta N)}{2N_2 + 1},
$$

with

$$
\Delta N = N_2 - N_1 \in \{-1,0,+1\}, \quad
H_{\mathrm{HL}} =
\begin{cases}
    N_2/2, & \Delta N = -1 \quad (\text{P}) \\
    (2N_2+1)/2, & \Delta N = 0 \quad (\text{Q}) \\
    (N_2+1)/2, & \Delta N = +1 \quad (\text{R})
\end{cases}
$$

#### Parameters
`v` : int
    Upper-state vibrational quantum number $v'$. Supported here: 0, 1, 2.

`N2` : int
    Upper-state rotational quantum number (A-state). For singlets, $J = N$.

`N1` : int
    Lower-state rotational quantum number (X-state).

#### Returns
float
    Line Einstein $A_{ul}$ in s⁻¹.

#### Notes
- `A_vib[v]` are pre-tabulated band Einstein coefficients for
  $A(v') \rightarrow X$ (units s⁻¹), and the Hönl–London factors
  correspond to a $^1\Pi \rightarrow {}^1\Sigma^+$ transition in the
  Hund's case (a) limit.
- This simplified partition neglects Λ-doubling, parity, and nuclear-spin
  substructure; any additional statistical weights should be applied
  elsewhere (e.g. electronic degeneracy).

#### Raises
ValueError
    If $\Delta N \notin \{-1,0,+1\}$ or `v` is out of the supported range.

#### Examples
```python
A = BHModel.A_coeff(v=0, N2=8, N1=7)  # R branch (ΔN=+1)
```

### `spectrum`

Compute the BH band spectrum on wavelength grid `x` for a single branch.

This model uses:
- **Upper (emitting) A-state** rovibrational energies from the **parametric constants** (`BH_A`);
- **Lower X-state** only for **line positions**, read from the **tabulated wavelengths**.

#### Parameters
`x` : ndarray
    Wavelength grid in **nm**.

`C` : float
    Population scale (absorbing other constants, path length, etc.).

`T_rot` : float
    Rotational temperature (K) used in Boltzmann factor for A-state populations.

`w_inst` : float
    Instrumental Gaussian FWHM (nm).

`T_tra` : float
    Translational temperature (K) for Doppler broadening in `line_profile`.

`branch` : Branch
    Which rotational branch to synthesize: `Branch.P`, `Branch.Q`, or `Branch.R`.

`v_max` : int, default 2
    Highest upper-state vibrational level $v'$ to include (inclusive).

`N2_max` : int, default 22
    Highest upper-state rotational quantum number $N_2$ to include (inclusive).

#### Returns
ndarray
    Spectrum on `x` (same shape), in arbitrary units.

#### Notes
- **A-state physics** (energies, populations) is evaluated from `BH_A` via `energy(...)`.
- **X-state** enters only through the **tabulated line centers** for the chosen `branch`.
- Per-line intensity is:
  $(h\nu)/(4\pi)\,n'(v',N_2)\,A(v',N_2\!\to\!N_1)\,g_\lambda(x)$,
  where `g_\lambda` is a Gaussian with Doppler+instrumental width.

### `full_fit_model`

Composite forward model for the 433 nm window:
BH Q-branch (A→X) + two fixed auxiliary lines + constant baseline.

The BH **A-state** populations/energies are computed from parametric constants
via :meth:`spectrum` (branch fixed to Q), while **X-state** enters only through
the tabulated line centers used inside :meth:`spectrum`. Two nearby isolated
features at fixed wavelengths (``R7``, ``R8``) are modeled as Gaussians and
added on top, plus a constant baseline.

#### Parameters
`x` : ndarray
    Wavelength grid in **nm**.

`C` : float
    Overall population/intensity scale for the BH Q-branch.

`T_rot` : float
    Rotational temperature (K) for the A-state Boltzmann factor.

`dx` : float
    Rigid wavelength shift in **nm** applied to `x` (accounts for calibration/tilt).

`w_inst` : float
    Instrumental Gaussian FWHM in **nm** used for line broadening.

`base` : float
    Constant background offset (a.u.).

`I_R7` : float
    Amplitude for the auxiliary Gaussian at $\lambda_{R7}=433.64776244\,\mathrm{nm}$.

`I_R8` : float
    Amplitude for the auxiliary Gaussian at $\lambda_{R8}=433.33500584\,\mathrm{nm}$.

#### Returns
ndarray
    Modeled spectrum sampled on `x` (same shape), in arbitrary units.

#### Notes
- The BH Q-branch contribution is scaled by ``1e8`` internally to bring values
  to a convenient numeric range for fitting; this does not change relative shapes.
- The auxiliary lines use :meth:`line_profile` with the same `w_inst` and
  a translational temperature fixed to ``0.0`` (instrument-limited broadening).
  If Doppler broadening is needed, promote `T_tra` to a parameter.
- Set ``I_R7=I_R8=0`` to exclude the auxiliary features.

#### Examples
```python
y = model.full_fit_model(
    x, C=1.2, T_rot=2100.0, dx=0.005,
    w_inst=0.02, base=0.01, I_R7=0.3, I_R8=0.2,
)
```
