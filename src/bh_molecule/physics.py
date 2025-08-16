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
    """BH spectroscopy model.
    See Also [Physics explainer](../phys.md)
    """

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
        r"""Level term value :math:`E(v,N)` [cm⁻¹].

        Parameters
        ----------
        v : int
            Vibrational quantum number (:math:`v'`).
        N : int
            Rotational quantum number (:math:`N`); for singlets, :math:`J=N`.
        c : MolecularConstants
            State constants (:math:`T_e, \omega_e, \omega_e x_e, \omega_e y_e, B_e, \alpha_e, D_e, \beta_e`).

        Returns
        -------
        float
            Term value :math:`E(v,N)` in cm⁻¹.

        See Also [Physics explainer — energy](../phys.md#energy)
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
        r"""Gaussian line profile in wavelength with Doppler ⊕ instrumental FWHM.

        Parameters
        ----------
        x : array_like
            Wavelength grid [nm].
        wl : float
            Line center wavelength [nm].
        w_inst : float
            Instrumental FWHM [nm] (Gaussian).
        T : float
            Translational/kinetic temperature [K] for Doppler broadening.

        Returns
        -------
        numpy.ndarray
            Normalized profile sampled on ``x`` (area ≈ 1).

        See Also [Physics explainer — line profile](../phys.md#line-profile)
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
        r"""Einstein :math:`A_{ul}` for :math:`A\,^1\Pi \rightarrow X\,^1\Sigma^+` line.

        Parameters
        ----------
        v : int
            Upper vibrational level :math:`v'`.
        N2 : int
            Upper rotational level :math:`N_2` (A-state).
        N1 : int
            Lower rotational level :math:`N_1` (X-state).

        Returns
        -------
        float
            :math:`A_{ul}` in s⁻¹.

        See Also [Physics explainer — A coefficient](../phys.md#a-coeff)
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
        r"""Branch spectrum on a wavelength grid; sums Gaussian lines over P/Q/R.

        Parameters
        ----------
        x : ndarray
            Wavelength grid [nm].
        C : float
            Population scale factor.
        T_rot : float
            Rotational temperature [K].
        w_inst : float
            Instrumental Gaussian FWHM [nm].
        T_tra : float
            Translational temperature [K] for Doppler.
        branch : Branch
            Rotational branch to synthesize.
        v_max : int, default 2
            Highest upper vibrational level :math:`v'` (inclusive).
        N2_max : int, default 22
            Highest upper rotational level :math:`N_2` (inclusive).

        Returns
        -------
        ndarray
            Spectrum on ``x`` (same shape), arbitrary units.

        See Also [Physics explainer — spectrum](../phys.md#spectrum)
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
        r"""Composite forward model near 433 nm: BH Q-branch + two fixed Gaussians + baseline.

        Parameters
        ----------
        x : ndarray
            Wavelength grid [nm].
        C : float
            Overall population/intensity scale for the BH Q-branch.
        T_rot : float
            Rotational temperature [K].
        dx : float
            Grid shift [nm].
        w_inst : float
            Instrumental Gaussian FWHM [nm].
        base : float
            Constant baseline.
        I_R7 : float
            Amplitude of auxiliary line at :math:`\lambda_{R7}`.
        I_R8 : float
            Amplitude of auxiliary line at :math:`\lambda_{R8}`.

        Returns
        -------
        np.ndarray
            Model evaluated on ``x``.

        See Also [Physics explainer — full fit model](../phys.md#full-fit-model)
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
