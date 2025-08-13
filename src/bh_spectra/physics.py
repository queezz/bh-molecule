from dataclasses import dataclass
import enum
import numpy as np
import scipy.constants as spc
from .constants import MolecularConstants, BH_X, BH_A


class Branch(enum.Enum):
    P = "P"
    Q = "Q"
    R = "R"


class BHModel:
    """Core BH spectrum model; file I/O is done elsewhere (dataio)."""

    def __init__(self, v00_wl_df):
        # DataFrame with columns P,Q,R (wavelengths in nm)
        self.v00_wl = v00_wl_df
        # Conversion constants
        self.K2wn = spc.Boltzmann / (spc.h * spc.c * 100)  # cm⁻¹ per K
        self.K2eV = spc.Boltzmann / spc.eV  # eV K⁻¹
        self.M_BH = 11.81  # amu
        self.wn2Hz = spc.c * 1e2
        self.g_as = 8

    # MARK: energy
    @staticmethod
    def energy(v: int, N: int, c: MolecularConstants) -> float:
        r"""
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
        B_v  &= B_e - \alpha_e\,(v+\tfrac12), \quad
        D_v  &= D_e - \beta_e\,(v+\tfrac12), \\
        F_v(N) &= B_v\,N(N+1) - D_v\,\big[N(N+1)\big]^2 .
        \end{aligned}
        $$

        #### Parameters
        v : int
            Vibrational quantum number $v \ge 0$.
        N : int
            Rotational quantum number (spinless). For singlet states,
            $J \approx N$; fine/Λ-doubling and spin-rotation are neglected here.
        c : MolecularConstants
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
        from bh_spectra.constants import BH_A
        E01 = BHModel.energy(v=0, N=1, c=BH_A)
        E11 = BHModel.energy(v=1, N=1, c=BH_A)
        assert E11 > E01
        ```
        """
        B_v = c.B_e - c.alpha_e * (v + 0.5)
        D_v = c.D_e - c.beta_e * (v + 0.5)
        G = (
            c.omega_e * (v + 0.5)
            - c.omega_e_x_e * (v + 0.5) ** 2
            + c.omega_e_y_e * (v + 0.5) ** 3
        )
        F = B_v * N * (N + 1) - D_v * N**2 * (N + 1) ** 2
        return c.T_e + G + F

    # MARK: line prof
    def line_profile(self, x, wl, w_inst, T):
        r"""
        Gaussian line profile with Doppler + instrumental broadening (FWHMs added in quadrature).

        #### Parameters
        x : array_like
            Wavelength axis in **nm**.
        wl : float
            Line center wavelength in **nm**.
        w_inst : float
            Instrumental full width at half maximum (FWHM) in **nm**, assumed Gaussian.
        T : float
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
        """
        # Doppler + instrumental Gaussian FWHM (nm)
        w_D = 7.72e-5 * wl * np.sqrt(T * self.K2eV / self.M_BH)
        w = np.sqrt(w_D**2 + w_inst**2)
        sigma = w / (2 * np.sqrt(2 * np.log(2)))
        return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
            -((x - wl) ** 2) / (2 * sigma**2)
        )

    # MARK: A-coeff
    @staticmethod
    def A_coeff(v: int, N2: int, N1: int) -> float:
        r"""
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
        v : int
            Upper-state vibrational quantum number $v'$. Supported here: 0, 1, 2.
        N2 : int
            Upper-state rotational quantum number (A-state). For singlets, $J = N$.
        N1 : int
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
        """
        if v not in (0, 1, 2):
            raise ValueError("v must be 0, 1, or 2 for the available A_vib table.")

        # Band Einstein coefficients for v'=0..2 (s⁻¹)
        A_vib = 7.2356e-6 * np.array([0.16135e13, 0.13678e13, 0.10345e13])

        dN = N2 - N1
        if dN == -1:  # P branch
            HLF = N2 / 2
        elif dN == 0:  # Q branch
            HLF = (2 * N2 + 1) / 2
        elif dN == 1:  # R branch
            HLF = (N2 + 1) / 2
        else:
            raise ValueError("invalid ΔN (must be -1, 0, or +1)")

        return float(A_vib[v] * HLF / (2 * N2 + 1))

    # MARK: spec
    def spectrum(
        self,
        x: np.ndarray,
        C: float,
        T_rot: float,
        w_inst: float,
        T_tra: float,
        branch: Branch,
        v_max: int = 2,
        N2_max: int = 22,
    ) -> np.ndarray:
        r"""
        Compute the BH band spectrum on wavelength grid `x` for a single branch.

        This model uses:
        - **Upper (emitting) A-state** rovibrational energies from the **parametric constants** (`BH_A`);
        - **Lower X-state** only for **line positions**, read from the **tabulated wavelengths**.

        #### Parameters
        x : ndarray
            Wavelength grid in **nm**.
        C : float
            Population scale (absorbing other constants, path length, etc.).
        T_rot : float
            Rotational temperature (K) used in Boltzmann factor for A-state populations.
        w_inst : float
            Instrumental Gaussian FWHM (nm).
        T_tra : float
            Translational temperature (K) for Doppler broadening in `line_profile`.
        branch : Branch
            Which rotational branch to synthesize: `Branch.P`, `Branch.Q`, or `Branch.R`.
        v_max : int, default 2
            Highest upper-state vibrational level $v'$ to include (inclusive).
        N2_max : int, default 22
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

        """
        # Map branch to ΔN = N2 - N1
        DELTA_N = {Branch.P: -1, Branch.Q: 0, Branch.R: +1}
        dN = DELTA_N[branch]

        # Wavelengths (nm) for the chosen branch from the X-state table
        wl_nm = self.v00_wl[branch.value].to_numpy()

        # Output spectrum
        y = np.zeros_like(x, dtype=float)

        # Convenience aliases
        g_as = self.g_as  # electronic degeneracy factor
        K2wn = self.K2wn  # K -> cm^-1
        Aconst = BH_A  # upper-state (A) constants

        for v in range(0, v_max + 1):
            # Reference energy for population normalization in this v'
            E_ref = self.energy(v, 1, Aconst)

            for N2 in range(1, N2_max + 1):
                N1 = N2 - dN  # since dN = N2 - N1

                # Skip invalid lower-state N1 (e.g., P-branch at N2=0 would give N1<0)
                if N1 < 0:
                    continue

                # --- Upper-state A-term energies (parametric) ---
                E_u = self.energy(v, N2, Aconst)  # cm^-1
                ΔF = E_u - E_ref  # cm^-1

                # Boltzmann-weighted A-state population (up to a scale C)
                pop_rot = C * g_as * (2 * N2 + 1) * np.exp(-ΔF / (T_rot * K2wn))

                # --- Lower-state X-term only via tabulated line center (nm) ---
                λ0 = wl_nm[N2]  # line position for this branch and N2

                # Photon energy factor (J)
                ν = spc.c / (λ0 * 1e-9)  # greek nu, not v
                photon_E = spc.h * ν / (4 * np.pi)

                # Transition strength via Einstein A with Hönl–London factor
                A_ul = self.A_coeff(v, N2, N1)

                # Line profile (nm^-1), Doppler+instrumental Gaussian
                profile = self.line_profile(x, λ0, w_inst, T_tra)

                # Accumulate contribution of this line
                y += photon_E * pop_rot * A_ul * profile

        return y

    # MARK: full ift
    def full_fit_model(
        self,
        x: np.ndarray,
        C: float,
        T_rot: float,
        dx: float,
        w_inst: float,
        base: float,
        I_R7: float,
        I_R8: float,
    ) -> np.ndarray:
        r"""
        Composite forward model for the 433 nm window:
        BH Q-branch (A→X) + two fixed auxiliary lines + constant baseline.

        The BH **A-state** populations/energies are computed from parametric constants
        via :meth:`spectrum` (branch fixed to Q), while **X-state** enters only through
        the tabulated line centers used inside :meth:`spectrum`. Two nearby isolated
        features at fixed wavelengths (``R7``, ``R8``) are modeled as Gaussians and
        added on top, plus a constant baseline.

        #### Parameters
        x : ndarray
            Wavelength grid in **nm**.
        C : float
            Overall population/intensity scale for the BH Q-branch.
        T_rot : float
            Rotational temperature (K) for the A-state Boltzmann factor.
        dx : float
            Rigid wavelength shift in **nm** applied to `x` (accounts for calibration/tilt).
        w_inst : float
            Instrumental Gaussian FWHM in **nm** used for line broadening.
        base : float
            Constant background offset (a.u.).
        I_R7 : float
            Amplitude for the auxiliary Gaussian at $\lambda_{R7}=433.64776244\,\mathrm{nm}$.
        I_R8 : float
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
        """
        wl_R7 = 433.6477624402892  # 1-1, R7
        wl_R8 = 433.3350058444114  # 1-1, R8
        T_tra = 0.0
        xs = x + dx
        return (
            1e8 * self.spectrum(xs, C, T_rot, w_inst, T_tra, Branch.Q)
            + I_R7 * self.line_profile(xs, wl_R7, w_inst, T_tra)
            + I_R8 * self.line_profile(xs, wl_R8, w_inst, T_tra)
            + base
        )
