"""
Molecular constants for BH A–X system (units: cm^-1).

This module defines a typed container `MolecularConstants` and the preset
parameter sets `BH_X` (X^1Σ^+) and `BH_A` (A^1Π).

The field names are explicit for learners (e.g., `omega_e_x_e`) and come with
spectroscopy-friendly aliases (`we`, `wexe`, `Be`, etc.) so both camps are happy.


All values are in wavenumbers (cm^-1).
These parameters are intended for rovibronic term-value calculations as used by `bh_molecule.physics.BHModel`.
"""

from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Dict, Any


@dataclass(frozen=True)
class MolecularConstants:
    """
    Dunham-like rovibrational constants (units: cm^-1).

    Parameters
    ----------
    T_e : float
        Electronic term origin.
    omega_e : float
        Harmonic vibrational constant (ω_e).
    omega_e_x_e : float
        Anharmonicity (ω_e x_e).
    omega_e_y_e : float
        Higher-order anharmonicity (ω_e y_e).
    B_e : float
        Rotational constant at equilibrium (B_e).
    alpha_e : float
        Rotation–vibration coupling (α_e).
    D_e : float
        Centrifugal distortion (D_e).
    beta_e : float
        Vibration dependence of D_e (β_e).

    Attributes
    ----------
    Te : float
        Alias of T_e.
    we : float
        Alias of omega_e.
    wexe : float
        Alias of omega_e_x_e.
    weye : float
        Alias of omega_e_y_e.
    Be : float
        Alias of B_e.
    De : float
        Alias of D_e.
    alphae : float
        Alias of alpha_e.
    betae : float
        Alias of beta_e.
    """

    T_e: float = field(
        metadata={"unit": "cm^-1", "desc": "Electronic term origin (T_e)"}
    )
    omega_e: float = field(
        metadata={"unit": "cm^-1", "desc": "Harmonic vibrational const (ω_e)"}
    )
    omega_e_x_e: float = field(
        metadata={"unit": "cm^-1", "desc": "Anharmonicity (ω_e x_e)"}
    )
    omega_e_y_e: float = field(
        metadata={"unit": "cm^-1", "desc": "Higher-order anharmonicity (ω_e y_e)"}
    )
    B_e: float = field(
        metadata={"unit": "cm^-1", "desc": "Rotational const at equilibrium (B_e)"}
    )
    alpha_e: float = field(
        metadata={"unit": "cm^-1", "desc": "Rotation–vibration coupling (α_e)"}
    )
    D_e: float = field(
        metadata={"unit": "cm^-1", "desc": "Centrifugal distortion (D_e)"}
    )
    beta_e: float = field(
        metadata={"unit": "cm^-1", "desc": "Vibration dependence of D_e (β_e)"}
    )

    # Spectroscopy-friendly aliases (read-only)
    @property
    def Te(self) -> float:
        return self.T_e

    @property
    def we(self) -> float:
        return self.omega_e

    @property
    def wexe(self) -> float:
        return self.omega_e_x_e

    @property
    def weye(self) -> float:
        return self.omega_e_y_e

    @property
    def Be(self) -> float:
        return self.B_e

    @property
    def De(self) -> float:
        return self.D_e

    @property
    def alphae(self) -> float:
        return self.alpha_e

    @property
    def betae(self) -> float:
        return self.beta_e

    # Utilities
    def as_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def as_table(self) -> str:
        lines = []
        for f in fields(self):
            meta = f.metadata or {}
            unit = meta.get("unit", "")
            desc = meta.get("desc", "")
            val = getattr(self, f.name)
            lines.append(f"{f.name:14s} = {val: .6f} {unit:6s}  # {desc}")
        return "\n".join(lines)


# Ground state X^1Σ^+
BH_X = MolecularConstants(
    T_e=0.0,
    omega_e=2366.9,
    omega_e_x_e=49.39,
    omega_e_y_e=0.364,
    B_e=12.021,
    alpha_e=0.412,
    D_e=0.001242,
    beta_e=-0.000026,
)

# Excited state A^1Π
BH_A = MolecularConstants(
    T_e=23135.8,
    omega_e=2250.9,
    omega_e_x_e=56.66,
    omega_e_y_e=-15.83,
    B_e=12.295,
    alpha_e=0.8346,
    D_e=0.001451,
    beta_e=0.0,
)

# Optional convenience map
BH_CONSTANTS: Dict[str, MolecularConstants] = {"X": BH_X, "A": BH_A}


@dataclass(frozen=True)
class LevelConstants:
    r"""
    Term-value constants of one vibrational level (units: cm^-1).

    Fernando & Bernath (1991) Eq. (1), with :math:`x = J(J+1)`:

    .. math::

        F(J) = T_v + B_v x - D_v x^2 + H_v x^3 + L_v x^4
               \pm \tfrac12 x\,(q_v + q_{Dv} x)

    "+" is the e and "-" the f Λ-doubling component (their p. 401). The
    doubling constants are zero for a Σ state.
    """

    T_v: float
    B_v: float
    D_v: float
    H_v: float = 0.0
    L_v: float = 0.0
    q_v: float = 0.0
    q_Dv: float = 0.0

    def term(self, J, parity: str | None = None):
        """Term value F(J) [cm^-1]; ``parity`` is "e" or "f" (needed when q != 0)."""
        x = J * (J + 1.0)
        F = self.T_v + self.B_v * x - self.D_v * x**2 + self.H_v * x**3 + self.L_v * x**4
        if self.q_v == 0.0 and self.q_Dv == 0.0:
            return F
        if parity not in ("e", "f"):
            raise ValueError(
                f"parity must be 'e' or 'f' for a Λ-doubled level, got {parity!r}"
            )
        sign = 1.0 if parity == "e" else -1.0
        return F + sign * 0.5 * x * (self.q_v + self.q_Dv * x)


# 11BH A¹Π and X¹Σ⁺ constants per vibrational level, cm^-1.
# Source: W. T. M. L. Fernando and P. F. Bernath, "Fourier transform
# spectroscopy of the A¹Π–X¹Σ⁺ transition of BH and BD", J. Mol. Spectrosc.
# 145, 392–402 (1991), doi:10.1016/0022-2852(91)90126-U. Table III (printed
# p. 398); term form Eq. (1) (p. 394); "+" = e, "-" = f (p. 401). Values are
# as printed, rescaled from the table's 10^n headers; the one-sigma
# uncertainty in the printed last digits follows each value and is repeated in
# absolute units in BH_A_FB91_UNC / BH_X_FB91_UNC. Checked against Table I
# (p. 395): with the f sign these constants reproduce every observed Q line of
# the 0-0, 1-1 and 2-2 bands within 0.03 cm^-1 (0.10 for 0-0 Q(25)); the e sign
# misses by up to 17 cm^-1, so Q-branch upper levels are f, P/R upper levels e.
# Observed Q lines reach J = 25, 19, 13 for v = 0, 1, 2 (Table I); F&B call the
# power series a poor representation of the A-state data (p. 399), so terms
# beyond the observed J are extrapolations.
BH_A_FB91: Dict[int, LevelConstants] = {
    0: LevelConstants(
        T_v=23073.9708,  # (22)
        B_v=11.906298,  # (81)
        D_v=1.43853e-3,  # (74)
        H_v=5.00e-8,  # (19)
        L_v=-6.88e-11,  # (16)
        q_v=3.7364e-2,  # (28)
        q_Dv=-1.6846e-5,  # (63)
    ),
    1: LevelConstants(
        T_v=25160.8602,  # (25)
        B_v=11.16801,  # (12)
        D_v=1.5986e-3,  # (15)
        H_v=-6.39e-8,  # (62)
        L_v=-18.71e-11,  # (87)
        q_v=3.4736e-2,  # (39)
        q_Dv=-1.997e-5,  # (13)
    ),
    2: LevelConstants(
        T_v=26992.2261,  # (46)
        B_v=10.22817,  # (38)
        D_v=2.0179e-3,  # (92)
        H_v=-20.0e-8,  # (81)
        L_v=-164e-11,  # (23)
        q_v=3.0301e-2,  # (95)
        q_Dv=-1.817e-5,  # (67)
    ),
}

BH_A_FB91_UNC: Dict[int, LevelConstants] = {
    0: LevelConstants(
        T_v=0.0022, B_v=0.000081, D_v=0.00074e-3, H_v=0.19e-8,
        L_v=0.16e-11, q_v=0.0028e-2, q_Dv=0.0063e-5,
    ),
    1: LevelConstants(
        T_v=0.0025, B_v=0.00012, D_v=0.0015e-3, H_v=0.62e-8,
        L_v=0.87e-11, q_v=0.0039e-2, q_Dv=0.013e-5,
    ),
    2: LevelConstants(
        T_v=0.0046, B_v=0.00038, D_v=0.0092e-3, H_v=8.1e-8,
        L_v=23e-11, q_v=0.0095e-2, q_Dv=0.067e-5,
    ),
}

# X-state L_v is printed with a 10^12 header (A state: 10^11); for v = 1, 2 it
# is fixed to the v = 0 value (Table III footnote a), uncertainty 0 here.
BH_X_FB91: Dict[int, LevelConstants] = {
    0: LevelConstants(
        T_v=0.0,
        B_v=11.815745,  # (43)
        D_v=1.22305e-3,  # (50)
        H_v=9.86e-8,  # (13)
        L_v=-4.5e-12,  # (10)
    ),
    1: LevelConstants(
        T_v=2269.22707,  # (25)
        B_v=11.400844,  # (37)
        D_v=1.20106e-3,  # (36)
        H_v=10.052e-8,  # (16)
        L_v=-4.5e-12,  # fixed
    ),
    2: LevelConstants(
        T_v=4443.03454,  # (71)
        B_v=10.992633,  # (60)
        D_v=1.1800e-3,  # (11)
        H_v=10.55e-8,  # (50)
        L_v=-4.5e-12,  # fixed
    ),
}

BH_X_FB91_UNC: Dict[int, LevelConstants] = {
    0: LevelConstants(T_v=0.0, B_v=0.000043, D_v=0.00050e-3, H_v=0.13e-8, L_v=1.0e-12),
    1: LevelConstants(T_v=0.00025, B_v=0.000037, D_v=0.00036e-3, H_v=0.016e-8, L_v=0.0),
    2: LevelConstants(T_v=0.00071, B_v=0.000060, D_v=0.0011e-3, H_v=0.50e-8, L_v=0.0),
}

# Balmer line wavelengths in air (nm), for wavelength calibration
BALMER_LINES_NM: Dict[str, float] = {
    "H_alpha": 656.279,
    "H_beta": 486.133,
    "H_gamma": 434.0462,
    "H_delta": 410.174,
}

__all__ = [
    "MolecularConstants",
    "BH_X",
    "BH_A",
    "BH_CONSTANTS",
    "LevelConstants",
    "BH_A_FB91",
    "BH_A_FB91_UNC",
    "BH_X_FB91",
    "BH_X_FB91_UNC",
    "BALMER_LINES_NM",
]
