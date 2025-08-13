from dataclasses import dataclass
import enum
import numpy as np
from scipy import constants

@dataclass(frozen=True)
class MolecularConstants:
    T_e: float; omega_e: float; omega_e_x_e: float; omega_e_y_e: float
    B_e: float; alpha_e: float; D_e: float; beta_e: float

# Example constants (placeholders; adjust to your validated values)
BH_X = MolecularConstants(0, 2366.9, 49.39, 0.364, 12.021, 0.412, 0.001242, -0.000026)
BH_A = MolecularConstants(23135.8, 2250.9, 56.66, -15.83, 12.295, 0.8346, 0.001451, 0)

class Branch(enum.Enum):
    P = "P"; Q = "Q"; R = "R"

class BHModel:
    """Core BH spectrum model; file I/O is done elsewhere (dataio)."""
    def __init__(self, v00_wl_df):
        # DataFrame with columns P,Q,R (wavelengths in nm)
        self.v00_wl = v00_wl_df
        # Conversion constants
        self.K2wn = 0.695
        self.K2eV = 8.617e-5
        self.M_BH = 11.81  # amu
        self.wn2Hz = constants.c * 1e2
        self.g_as = 8

    @staticmethod
    def energy(v, N, c: MolecularConstants):
        B_v = c.B_e - c.alpha_e * (v + 0.5)
        D_v = c.D_e - c.beta_e  * (v + 0.5)
        G = c.omega_e*(v+0.5) - c.omega_e_x_e*(v+0.5)**2 + c.omega_e_y_e*(v+0.5)**3
        F = B_v*N*(N+1) - D_v*N**2*(N+1)**2
        return c.T_e + G + F

    def line_profile(self, x, wl, w_inst, T):
        # Doppler + instrumental Gaussian FWHM (nm)
        w_D = 7.72e-5 * wl * np.sqrt(T * self.K2eV / self.M_BH)
        w = np.sqrt(w_D**2 + w_inst**2)
        sigma = w / (2*np.sqrt(2*np.log(2)))
        return (1/(np.sqrt(2*np.pi)*sigma)) * np.exp(-((x-wl)**2)/(2*sigma**2))

    @staticmethod
    def A_coeff(v, N2, N1):
        A_vib = 7.2356e-6 * np.array([0.16135e13, 0.13678e13, 0.10345e13])  # v=0..2
        dN = N2 - N1
        if dN == -1: HLF = N2/2
        elif dN == 0: HLF = (2*N2+1)/2
        elif dN == 1: HLF = (N2+1)/2
        else: raise ValueError("invalid ΔN")
        return A_vib[v] * HLF / (2*N2+1)

    def spectrum(self, x, C, T_rot, w_inst, T_tra, branch: Branch, v_max=2, N2_max=22):
        intensity = 0.0
        wl = self.v00_wl[branch.value].to_numpy()
        for v in range(0, v_max+1):
            for N2 in range(1, N2_max+1):
                N1 = N2 + (1 if branch==Branch.P else 0) - (1 if branch==Branch.R else 0)
                del_F = self.energy(v, N2, BH_A) - self.energy(v, 1, BH_A)
                n_prime = C * self.g_as * (2*N2+1) * np.exp(-del_F/(T_rot*self.K2wn))
                nu = constants.c / (wl[N2]*1e-9)
                intensity += constants.h*nu/(4*np.pi) * n_prime                            * self.A_coeff(v, N2, N1)                            * self.line_profile(x, wl[N2], w_inst, T_tra)
        return intensity

    def full_fit_model(self, x, C, T_rot, dx, w_inst, base, I_R7, I_R8):
        wl_R7 = 433.6477624402892; wl_R8 = 433.3350058444114; T_tra = 0.0
        xs = x + dx
        return (1e8 * self.spectrum(xs, C, T_rot, w_inst, T_tra, Branch.Q)
                + I_R7 * self.line_profile(xs, wl_R7, w_inst, T_tra)
                + I_R8 * self.line_profile(xs, wl_R8, w_inst, T_tra)
                + base)
