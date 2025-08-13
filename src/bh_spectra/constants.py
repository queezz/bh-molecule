"""
Molecular constants for BH A–X system (units: cm^-1).

This module defines a typed container `MolecularConstants` and the preset
parameter sets `BH_X` (X^1Σ^+) and `BH_A` (A^1Π).

The field names are explicit for learners (e.g., `omega_e_x_e`) and come with
spectroscopy-friendly aliases (`we`, `wexe`, `Be`, etc.) so both camps are happy.

Notes
-----
- All values are in wavenumbers (cm^-1).
- These parameters are intended for rovibronic term-value calculations as used
  by `bh_spectra.physics.BHModel`.
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

    Attributes (aliases)
    --------------------
    Te, we, wexe, weye, Be, De, alphae, betae
        Spectroscopy-friendly read-only aliases to the fields above.

    Methods
    -------
    as_dict()
        Return a plain dict of field -> value.
    as_table()
        Return a human-readable multiline table with units and descriptions.
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

__all__ = ["MolecularConstants", "BH_X", "BH_A", "BH_CONSTANTS"]
