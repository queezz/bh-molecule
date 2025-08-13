from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import constants
import enum


@dataclass
class MolecularConstants:
    T_e: float
    omega_e: float
    omega_e_x_e: float
    omega_e_y_e: float
    B_e: float
    alpha_e: float
    D_e: float
    beta_e: float


BH_X_CONSTS = MolecularConstants(
    0, 2366.9, 49.39, 0.364, 12.021, 0.412, 0.001242, -0.000026
)
BH_A_CONSTS = MolecularConstants(
    23135.8, 2250.9, 56.66, -15.83, 12.295, 0.8346, 0.001451, 0
)


class Branch(enum.Enum):
    P = "P"
    Q = "Q"
    R = "R"


@dataclass
class BHConstants:
    v00_wl: pd.DataFrame
    K2wn: float = 0.695
    wn2Hz: float = constants.c * 1e2
    K2eV: float = 8.617e-5
    M_BH: float = 11.81
    g_as: int = 8


class BHModel:
    """
    v00_wl - wavelength data for 0-0 transition
    in ./11BH_wl_Fernando
    11BH_v00.csv
    """

    def __init__(self, v00_wl: pd.DataFrame):
        self.v00_wl = v00_wl
        self.K2wn = 0.695
        self.wn2Hz = constants.c * 1e2
        self.K2eV = 8.617e-5
        self.M_BH = 11.81
        self.g_as = 8

    def energy(self, v, N, consts: MolecularConstants):
        B_v = consts.B_e - consts.alpha_e * (v + 0.5)
        D_v = consts.D_e - consts.beta_e * (v + 0.5)
        G = (
            consts.omega_e * (v + 0.5)
            - consts.omega_e_x_e * (v + 0.5) ** 2
            + consts.omega_e_y_e * (v + 0.5) ** 3
        )
        F = B_v * N * (N + 1) - D_v * N**2 * (N + 1) ** 2
        return consts.T_e + G + F

    def line_profile(self, x, wl, w_inst, T):
        w_D = 7.72e-5 * wl * np.sqrt(T * self.K2eV / self.M_BH)
        w = np.sqrt(w_D**2 + w_inst**2)
        sigma = w / (2 * np.sqrt(2 * np.log(2)))
        return (
            1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-((x - wl) ** 2) / (2 * sigma**2))
        )

    def A_coeff(self, v, N2, N1):
        """
        Calculate the Einstein A coefficient for a given vibrational level v
        and rotational quantum numbers N2 → N1.

        Parameters:
            v (int): vibrational level (0, 1, or 2)
            N2 (int): upper rotational quantum number
            N1 (int): lower rotational quantum number

        Returns:
            float: A coefficient (s⁻¹)
        """
        # Vibrational A coefficients for Δv = 0 transitions (v=0,1,2)
        A_vib = 7.2356e-6 * np.array(
            [0.16135e13, 0.13678e13, 0.10345e13]  # v=0  # v=1  # v=2
        )

        ΔN = N2 - N1
        if ΔN == -1:  # P-branch: N' = N+1 → N
            HLF = N2 / 2
        elif ΔN == 0:  # Q-branch: N' = N → N
            HLF = (2 * N2 + 1) / 2
        elif ΔN == 1:  # R-branch: N' = N-1 → N
            HLF = (N2 + 1) / 2
        else:
            raise ValueError(
                f"Invalid rotational transition: N2={N2}, N1={N1} (ΔN={ΔN})"
            )

        # Total A coefficient with rotational factor
        return A_vib[v] * HLF / (2 * N2 + 1)

    def spectrum(self, x, C, T_rot, w_inst, T_tra, branch: Branch):
        v_min, v_max = 0, 2
        N2_min, N2_max = 1, 22
        intensity = 0

        for v in range(v_min, v_max + 1):
            for N2 in range(N2_min, N2_max + 1):
                if branch == Branch.P:
                    N1 = N2 + 1
                elif branch == Branch.Q:
                    N1 = N2
                elif branch == Branch.R:
                    N1 = N2 - 1
                else:
                    continue
                wl = self.v00_wl[branch.value].to_numpy()
                del_G = self.energy(v, 0, BH_A_CONSTS) - self.energy(0, 0, BH_A_CONSTS)
                del_F = self.energy(v, N2, BH_A_CONSTS) - self.energy(v, 1, BH_A_CONSTS)
                n_prime = (
                    C * self.g_as * (2 * N2 + 1) * np.exp(-del_F / (T_rot * self.K2wn))
                )
                nu = constants.c / (wl[N2] * 1e-9)
                intensity += (
                    constants.h
                    * nu
                    / (4 * np.pi)
                    * n_prime
                    * self.A_coeff(v, N2, N1)
                    * self.line_profile(x, wl[N2], w_inst, T_tra)
                )
        return intensity

    def full_fit(self, x, C, T_rot, dx, w_inst, base, I_R7, I_R8):
        T_tra = 0
        wl_R7 = 433.6477624402892
        wl_R8 = 433.3350058444114
        x_shifted = x + dx
        return (
            1e8 * self.spectrum(x_shifted, C, T_rot, w_inst, T_tra, Branch.Q)
            + I_R7 * self.line_profile(x_shifted, wl_R7, w_inst, T_tra)
            + I_R8 * self.line_profile(x_shifted, wl_R8, w_inst, T_tra)
            + base
        )
