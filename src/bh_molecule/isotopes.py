"""10BH line positions and level energies for ``BHModel(isotopologue="natural")``.

Natural boron is about 19.9 % 10B. The 10BH A-X (0-0) Q branch lies on and
beside the 11BH one; its high-N lines are shifted relative to their 11BH
partners. This module provides the 10BH Q-line positions and A-state f-level
energies under two position schemes:

``"fb91"``
    Fernando & Bernath (1991) Table V, measured 10BH lines, where measured;
    the other Q lines from their Table VII 10BH constants (F&B Eq. (1), f sign).
    This is F&B's own description of 10BH; Table VII was fitted to Table V
    without H and L terms.
``"mass_scaled"``
    The 11BH Dunham coefficients of F&B Table VIII, mass-scaled to 10BH:
    ``Y_kl -> Y_kl * rho**(k + 2 l)`` and, for the Lambda-doubling terms,
    ``Q_kl -> Q_kl * rho**(k + 2 l + 2)`` (q ~ B**2 / dE), with
    ``rho = sqrt(mu(11BH) / mu(10BH))``.

The two schemes agree within 0.0015 nm for Q(1)-Q(11) and for every measured
10BH P and R line, and they disagree for the high-N Q lines: Table V puts
10BH Q(18)-Q(20) 1.7-4.5 cm^-1 *blue* of 11BH, the mass-scaled constants put
them 0.8-1.0 cm^-1 *red* (they miss the three measured lines by 2.4-5.5
cm^-1). Which is right is not settled here; ``BHModel`` offers both.

Source: W. T. M. L. Fernando and P. F. Bernath, J. Mol. Spectrosc. 145,
392-402 (1991), doi:10.1016/0022-2852(91)90126-U. Table V (printed p. 399),
Table VII (p. 400), Table VIII (p. 401), read from 300 dpi renders of the
library PDF. Table VIII prints the X-state Y04 as -5.18(75) under a 10^11
header; read so, the Dunham set misses the 11BH Table I Q lines by up to
4.4 cm^-1 at J = 23. Read as -5.18e-12 it reproduces them within 0.07 cm^-1,
and Table III's level constant L_0(X) = -4.5e-12 (10^12 header) agrees, so
-5.18e-12 is used.
"""

from __future__ import annotations

import numpy as np

from .constants import LevelConstants

# Atomic masses [u], AME2020.
M_11B: float = 11.00930536
M_10B: float = 10.01293695
M_1H: float = 1.00782503207
MU_11BH: float = M_11B * M_1H / (M_11B + M_1H)
MU_10BH: float = M_10B * M_1H / (M_10B + M_1H)
RHO_10BH: float = float(np.sqrt(MU_11BH / MU_10BH))

# Natural boron isotope fractions (F&B p. 399: "20 and 80 %"; IUPAC 19.9 %).
BH10_FRACTION: float = 0.199
BH11_FRACTION: float = 1.0 - BH10_FRACTION
BH10_OVER_BH11: float = BH10_FRACTION / BH11_FRACTION

BH10_POSITIONS = ("fb91", "mass_scaled")

# F&B Table VIII, 11BH Dunham coefficients, cm^-1, key (k, l):
# E = sum Y_kl (v + 1/2)^k x^l  (+-) 1/2 sum Q_kl (v + 1/2)^k x^l, x = J(J+1),
# "+" e, "-" f (A state only).
DUNHAM_X_FB91: dict[tuple[int, int], float] = {
    (0, 0): 0.0,
    (1, 0): 2366.7296,
    (2, 0): -49.33983,
    (3, 0): 0.36227,
    (0, 1): 12.025755,
    (1, 1): -0.421565,
    (2, 1): 0.0033195,
    (0, 2): -1.23493e-3,
    (1, 2): 2.247e-5,
    (0, 3): 9.98e-8,
    (0, 4): -5.18e-12,  # printed -5.18 under 10^11; read as 10^12, see module doc
}
DUNHAM_A_FB91: dict[tuple[int, int], float] = {
    (0, 0): 23135.4376,
    (1, 0): 2251.4577,
    (2, 0): -56.5725,
    (3, 0): -15.82,  # fixed by F&B to Johns et al.
    (0, 1): 12.20035,
    (1, 1): -0.53767,
    (2, 1): -0.10070,
    (0, 2): -1.4577e-3,
    (1, 2): 9.86e-5,
    (2, 2): -1.224e-4,
    (0, 3): 2.32e-8,
    (1, 3): 1.38e-7,
    (2, 3): -1.631e-7,
    (0, 4): -6.96e-11,
}
DUNHAM_A_LAMBDA_FB91: dict[tuple[int, int], float] = {
    (0, 1): 3.8263e-2,
    (1, 1): -1.406e-3,
    (2, 1): -6.38e-4,
    (0, 2): -1.532e-5,
    (1, 2): -3.20e-6,
}

# F&B Table VII, 10BH constants (D and q_D fixed by F&B from isotope relations).
# T_vv is the band origin; the X level is put at T_v = 0.
BH10_A_FB91: dict[int, LevelConstants] = {
    0: LevelConstants(T_v=23073.6860, B_v=12.00484, D_v=1.462e-3, q_v=3.848e-2, q_Dv=-1.685e-5),
    1: LevelConstants(T_v=22890.048, B_v=11.25334, D_v=1.625e-3, q_v=3.605e-2, q_Dv=-2.065e-5),
}
BH10_X_FB91: dict[int, LevelConstants] = {
    0: LevelConstants(T_v=0.0, B_v=11.91502, D_v=1.243e-3),
    1: LevelConstants(T_v=0.0, B_v=11.49288, D_v=1.221e-3),
}

# F&B Table V, measured 10BH lines (vacuum wavenumbers, cm^-1), by lower J.
FB91_10BH_LINES: dict[tuple[int, str], dict[int, float]] = {
    (0, "P"): {3: 23002.976, 4: 22979.923, 5: 22957.241, 6: 22934.884, 7: 22912.752, 8: 22890.997},
    (0, "Q"): {
        4: 23075.012, 5: 23075.627, 6: 23076.310, 7: 23076.979, 8: 23077.681,
        10: 23078.937, 18: 23073.175, 19: 23070.091, 20: 23066.492,
    },
    (0, "R"): {
        0: 23097.676, 1: 23121.965, 2: 23146.307, 3: 23170.741, 4: 23195.248, 5: 23219.802,
        6: 23244.171, 7: 23268.473, 8: 23292.481, 9: 23316.312, 10: 23339.648, 11: 23362.514,
    },
    (1, "P"): {6: 22746.135, 8: 22694.976, 9: 22668.645},
    (1, "Q"): {
        3: 22886.834, 4: 22884.755, 5: 22881.949, 6: 22878.571, 7: 22874.458,
        10: 22856.937, 11: 22849.024,
    },
    (1, "R"): {3: 22977.058, 4: 22997.353, 5: 23016.876, 6: 23035.584, 7: 23053.345, 8: 23070.053},
}


def dunham_term(
    coeffs: dict[tuple[int, int], float],
    v: int,
    J,
    rho: float = 1.0,
    lambda_coeffs: dict[tuple[int, int], float] | None = None,
    parity: str | None = None,
):
    """Term value [cm^-1] from Dunham coefficients, mass-scaled by ``rho``.

    ``Y_kl`` scale as ``rho**(k + 2 l)``; the Lambda-doubling ``Q_kl`` as
    ``rho**(k + 2 l + 2)`` and enter with +1/2 (e) or -1/2 (f).
    """
    J = np.asarray(J, dtype=float)
    x = J * (J + 1.0)
    h = v + 0.5
    e = np.zeros_like(x)
    for (k, n), c in coeffs.items():
        e = e + c * rho ** (k + 2 * n) * h**k * x**n
    if lambda_coeffs:
        if parity not in ("e", "f"):
            raise ValueError("parity must be 'e' or 'f' with Lambda-doubling terms")
        sign = 1.0 if parity == "e" else -1.0
        for (k, n), c in lambda_coeffs.items():
            e = e + sign * 0.5 * c * rho ** (k + 2 * n + 2) * h**k * x**n
    return e


def _scheme(scheme: str) -> str:
    if scheme not in BH10_POSITIONS:
        raise ValueError(f"10BH positions must be one of {BH10_POSITIONS}")
    return scheme


def bh10_upper_f_energy(N, v: int = 0, scheme: str = "fb91"):
    """10BH A-state f-level term value [cm^-1] (origin: X v, J = 0)."""
    _scheme(scheme)
    if scheme == "fb91":
        return BH10_A_FB91[v].term(np.asarray(N, dtype=float), "f")
    return dunham_term(DUNHAM_A_FB91, v, N, RHO_10BH, DUNHAM_A_LAMBDA_FB91, "f") - dunham_term(
        DUNHAM_X_FB91, v, 0, RHO_10BH
    )


def bh10_line_wavenumber(v: int, branch: str, J, scheme: str = "fb91"):
    """Calculated 10BH (v-v) line [cm^-1, vacuum] by lower J; no measured values."""
    _scheme(scheme)
    J = np.asarray(J, dtype=float)
    up = {"P": J - 1, "Q": J, "R": J + 1}[branch]
    par = "f" if branch == "Q" else "e"
    if scheme == "fb91":
        return BH10_A_FB91[v].term(up, par) - BH10_X_FB91[v].term(J)
    return dunham_term(DUNHAM_A_FB91, v, up, RHO_10BH, DUNHAM_A_LAMBDA_FB91, par) - dunham_term(
        DUNHAM_X_FB91, v, J, RHO_10BH
    )


def bh10_q_wavenumbers(N_max: int = 25, scheme: str = "fb91") -> tuple[np.ndarray, np.ndarray]:
    """10BH (0-0) Q-line wavenumbers [cm^-1, vacuum] indexed by N = 0..N_max.

    Returns ``(wn, source)``; ``wn[0]`` is NaN (no Q(0)). ``"fb91"`` takes the
    measured Table V lines and calculates the rest from Table VII;
    ``"mass_scaled"`` calculates every line from the scaled Dunham set.
    ``source`` holds "measured", "fb91_constants" or "mass_scaled".
    """
    N = np.arange(N_max + 1)
    wn = np.asarray(bh10_line_wavenumber(0, "Q", N, scheme), dtype=float)
    src = np.full(N.size, "fb91_constants" if scheme == "fb91" else "mass_scaled", dtype=object)
    if scheme == "fb91":
        for j, w in FB91_10BH_LINES[(0, "Q")].items():
            if j <= N_max:
                wn[j], src[j] = w, "measured"
    wn[0], src[0] = np.nan, ""
    return wn, src


def bh10_r11_wavenumbers(scheme: str = "fb91") -> dict[int, float]:
    """10BH (1-1) R(7) and R(8) [cm^-1, vacuum]: Table V (``"fb91"``) or scaled."""
    _scheme(scheme)
    if scheme == "fb91":
        return {j: FB91_10BH_LINES[(1, "R")][j] for j in (7, 8)}
    return {j: float(bh10_line_wavenumber(1, "R", j, scheme)) for j in (7, 8)}


def wavenumber_to_air_nm(wn):
    """Vacuum wavenumber [cm^-1] -> air wavelength [nm], as the 11BH tables are converted."""
    from .dataio import n_air

    vac = 1e7 / np.asarray(wn, dtype=float)
    return vac / n_air(vac)


__all__ = [
    "BH10_FRACTION",
    "BH10_OVER_BH11",
    "BH10_POSITIONS",
    "RHO_10BH",
    "DUNHAM_X_FB91",
    "DUNHAM_A_FB91",
    "DUNHAM_A_LAMBDA_FB91",
    "BH10_A_FB91",
    "BH10_X_FB91",
    "FB91_10BH_LINES",
    "dunham_term",
    "bh10_upper_f_energy",
    "bh10_line_wavenumber",
    "bh10_q_wavenumbers",
    "bh10_r11_wavenumbers",
    "wavenumber_to_air_nm",
]
