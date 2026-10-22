from dataclasses import dataclass
import enum
import numpy as np
import scipy.constants as spc
from .constants import MolecularConstants, BH_X, BH_A, BH_A_FB91


class Branch(enum.Enum):
    P = "P"
    Q = "Q"
    R = "R"


# Sensitivity switches of `BHModel`; the first entry is the production default.
UPPER_ENERGIES = ("huber_herzberg", "fernando_bernath")
BAND_POSITIONS = ("v00", "own")


class BHModel:
    """BH spectroscopy model.
    See Also [Physics explainer](../phys.md)

    Parameters
    ----------
    v00_wl_df : DataFrame, optional
        0-0 line positions (columns J, P, Q, R in nm); default from resources.
    upper_energies : {"huber_herzberg", "fernando_bernath"}
        Upper A-state term values used in the rotational population weights.
        ``"huber_herzberg"`` (default) is :meth:`energy` with ``BH_A``.
        ``"fernando_bernath"`` is Fernando & Bernath (1991) Eq. (1) with the
        per-level Table III constants ``BH_A_FB91``, on the f level for Q lines
        and the e level for P/R lines.
    v_max : int, default 2
        Highest upper vibrational level summed by :meth:`spectrum` (when its
        own ``v_max`` is not given) and hence by :meth:`full_fit_model`.
        ``v_max=0`` is the 0-0-only model.
    band_positions : {"v00", "own"}
        ``"v00"`` (default) puts the lines of every band at the 0-0 positions.
        ``"own"`` uses each band's measured positions (F&B Table I, resources
        ``11BH_v11.csv``, ``11BH_v22.csv``) and skips lines not observed there.
    T_vib : float or None
        ``None`` (default): every band is normalised at its own N = 1 level,
        with no vibrational population factor. A value in K weights band v by
        ``exp(-(T_v - T_0) / (k T_vib))`` with the F&B band origins T_v.

    The defaults reproduce the production model exactly.
    """

    def __init__(
        self,
        v00_wl_df=None,
        *,
        upper_energies: str = "huber_herzberg",
        v_max: int = 2,
        band_positions: str = "v00",
        T_vib: float | None = None,
    ):
        # DataFrame with columns P,Q,R (wavelengths in nm). If not provided,
        # load the default v00 table from package resources.
        if v00_wl_df is None:
            from .dataio import load_v00_wavelengths

            v00_wl_df = load_v00_wavelengths()

        if upper_energies not in UPPER_ENERGIES:
            raise ValueError(f"upper_energies must be one of {UPPER_ENERGIES}")
        if band_positions not in BAND_POSITIONS:
            raise ValueError(f"band_positions must be one of {BAND_POSITIONS}")
        if int(v_max) not in (0, 1, 2):
            raise ValueError("v_max must be 0, 1, or 2")
        if T_vib is not None and not float(T_vib) > 0:
            raise ValueError("T_vib must be None or a positive temperature [K]")

        self.v00_wl = v00_wl_df
        self.upper_energies = upper_energies
        self.v_max = int(v_max)
        self.band_positions = band_positions
        self.T_vib = None if T_vib is None else float(T_vib)

        # Per-band line tables for band_positions="own" (v=0 is v00_wl).
        self._band_wl = {0: v00_wl_df}
        if band_positions == "own":
            from .dataio import load_band_wavelengths

            for v in (1, 2):
                self._band_wl[v] = load_band_wavelengths(v)
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

    def upper_energy(self, v: int, N: int, branch: Branch = Branch.Q) -> float:
        r"""Upper A-state term value [cm⁻¹] under the ``upper_energies`` switch.

        ``"huber_herzberg"``: :meth:`energy` with ``BH_A``. ``"fernando_bernath"``:
        F&B (1991) Eq. (1) with ``BH_A_FB91[v]``; the upper level of a Q line is
        the f component, of a P or R line the e component.
        """
        if self.upper_energies == "huber_herzberg":
            return self.energy(v, N, BH_A)
        parity = "f" if branch is Branch.Q else "e"
        return float(BH_A_FB91[v].term(N, parity))

    def vib_weight(self, v: int) -> float:
        r"""Vibrational population factor of band v relative to v = 0.

        1 when ``T_vib`` is None (each band normalised at its own N = 1);
        otherwise :math:`\exp[-(T_v - T_0)/(k_B T_{\rm vib})]` with the F&B
        Table III band origins.
        """
        if self.T_vib is None:
            return 1.0
        G = BH_A_FB91[v].T_v - BH_A_FB91[0].T_v
        return float(np.exp(-G / (self.T_vib * self.K2wn)))

    def _line_positions(self, v: int, branch: Branch) -> np.ndarray:
        """Line centres [nm] indexed by table row J, under ``band_positions``."""
        table = self.v00_wl if self.band_positions == "v00" else self._band_wl[v]
        return table[branch.value].to_numpy()

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
        v_max: int | None = None,
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
        v_max : int, optional
            Highest upper vibrational level :math:`v'` (inclusive); defaults to
            the model's ``v_max`` (2 unless the model was built otherwise).
        N2_max : int, default 22
            Highest upper rotational level :math:`N_2` (inclusive).

        Returns
        -------
        ndarray
            Spectrum on ``x`` (same shape), arbitrary units.

        See Also [Physics explainer — spectrum](../phys.md#spectrum)
        """
        if v_max is None:
            v_max = self.v_max

        # Map branch to ΔN = N2 - N1
        DELTA_N = {Branch.P: -1, Branch.Q: 0, Branch.R: +1}
        dN = DELTA_N[branch]

        # Output spectrum
        y = np.zeros_like(x, dtype=float)

        # Convenience aliases
        g_as = self.g_as  # electronic degeneracy factor
        K2wn = self.K2wn  # K -> cm^-1
        own_positions = self.band_positions == "own"

        for v in range(0, v_max + 1):
            # Wavelengths (nm) for the chosen branch, rows indexed by table J
            # (the lower-state J, so the N2 lookup below is exact for Q only)
            wl_nm = self._line_positions(v, branch)

            # Reference energy for population normalization in this v'
            E_ref = self.upper_energy(v, 1, branch)
            w_vib = self.vib_weight(v)

            for N2 in range(1, N2_max + 1):
                N1 = N2 - dN  # since dN = N2 - N1

                # Skip invalid lower-state N1 (e.g., P-branch at N2=0 would give N1<0)
                if N1 < 0:
                    continue

                # Own band positions: skip lines the band's table does not list
                if own_positions and (N2 >= wl_nm.size or not np.isfinite(wl_nm[N2])):
                    continue

                # --- Upper-state A-term energies (parametric) ---
                E_u = self.upper_energy(v, N2, branch)  # cm^-1
                ΔF = E_u - E_ref  # cm^-1

                # Boltzmann-weighted A-state population (up to a scale C)
                pop_rot = C * g_as * (2 * N2 + 1) * np.exp(-ΔF / (T_rot * K2wn))
                pop_rot = pop_rot * w_vib

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
