import enum
import numpy as np
import scipy.constants as spc
from .constants import MolecularConstants, BH_A, BH_A_FB91
from .isotopes import (
    BH10_OVER_BH11,
    BH10_POSITIONS,
    bh10_q_wavenumbers,
    bh10_r11_wavenumbers,
    bh10_upper_f_energy,
    wavenumber_to_air_nm,
)


class Branch(enum.Enum):
    P = "P"
    Q = "Q"
    R = "R"


# Sensitivity switches of `BHModel`; the first entry is the production default.
UPPER_ENERGIES = ("huber_herzberg", "fernando_bernath")
BAND_POSITIONS = ("v00", "own")
BASELINES = ("constant", "linear", "quadratic", "hgamma", "fixed")
ISOTOPOLOGUES = ("11BH", "natural")
H2_LINE_SETS = (False, True, "R2")
ROT_COMPONENTS = (1, 2)
R11_LINE_MODES = ("free", "tied")

# Positions of the 11BH (1-1) R(7) and R(8) lines in ``full_fit_model``
# [nm]. ``r11_lines="free"`` (production) keeps the values the model has
# always used; they lie 0.0037 nm red of Fernando & Bernath (1991) Table I
# converted with the package's own ``n_air`` (433.644051 / 433.331292 nm,
# ``load_band_wavelengths(1)``), the positions ``r11_lines="tied"`` uses.
R11_MODEL_NM: dict[str, float] = {
    "R7": 433.6477624402892,
    "R8": 433.3350058444114,
}

# Pivot of the optional linear wavelength stretch (``stretch=True``) [nm]:
# the centre of the production window, as for the linear baseline, so that
# ``dx`` keeps its meaning (the shift at the window centre).
STRETCH_REF_NM: float = 433.475

# H-gamma far wing of ``baseline="hgamma"``: a Lorentzian centred on H-gamma (NIST ASD,
# H I 4340.472 A in air) on the shifted grid x + dx, HWHM ``HGAMMA_WING_HWHM_NM``,
# normalised to 1 at ``HGAMMA_WING_REF_NM`` from the line centre (433.80 nm, the red edge
# of set C's window), so that its amplitude ``base_hg`` is the wing level there. In the
# BH window the line centre is 0.25-1.0 nm away and the shape is the far wing ~ d^-2;
# the measured H-gamma profiles of the 1.33 m spectrometer follow d^-2 from 0.03 to
# 0.3 nm (paper repo S12). ``HGAMMA_WING_POWER`` is the exponent of the far wing.
HGAMMA_AIR_NM: float = 434.0472
HGAMMA_WING_HWHM_NM: float = 0.03
HGAMMA_WING_REF_NM: float = 0.25
HGAMMA_WING_POWER: float = 2.0

# Unidentified features in the fit window used as optional free nuisance
# lines (``extra_lines``), rest-frame air wavelengths [nm]: F1 on the blue
# foot of the Q head and F3 next to Q(22), found in the mean residuals of the
# production and set C fits (paper repo S1, S5b, S7b, S5d: F1 433.09-433.12,
# F3 433.745-433.755 nm in the rest frame).
EXTRA_LINES_NM: dict[str, float] = {
    "F1": 433.100,
    "F3": 433.750,
}

# Reference wavelength of the linear baseline [nm]: the centre of the
# production fit window 433.05-433.90 nm. Fixed (not the centre of whatever
# window is fitted) so that the offset ``base`` keeps one meaning when the
# window is trimmed.
BASELINE_REF_NM: float = 433.475

# H2 lines in the 433 nm window used as optional nuisance features
# (``h2_lines=True``): Dieke's Y 1 -> 2B 0 R0/R1 (4332.619 A, on the BH
# Q(1)-Q(3) lines) and R2 (4335.522 A); air wavelengths from H. M. Crosswhite
# (ed.), The Hydrogen Molecule Wavelength Tables of G. H. Dieke (1972),
# Table T, pp. T104-T105.
H2_NUISANCE_LINES_NM: dict[str, float] = {
    "I_H2_R01": 433.2619,
    "I_H2_R2": 433.5522,
}

# Intensity ratio I(R0/R1) / I(R2) of the two H2 Y 1 -> 2B 0 lines used when
# their amplitudes are tied (``h2_r01_ratio``): Dieke's photoelectric
# intensities I4 = 1.29 (4332.619 A, R0 with R1) and 1.20 (4335.522 A, R2),
# Crosswhite (1972) Table T, pp. T104-T105, taken as a linear ratio. The eye
# estimates of the same table give 0.9/0.8 (77 K discharge) and 1.0/1.0
# (microwave discharge).
DIEKE_H2_R01_OVER_R2: float = 1.29 / 1.20

# Start values and bounds of the optional extra fit parameters, appended in
# this order after (C, T_rot, dx, w_inst, base, I_R7, I_R8). base_slope is in
# normalised intensity per nm (+-1 /nm is +-0.85 across the 0.85 nm window);
# the H2 amplitudes are line areas in the units of I_R7 and I_R8.
EXTRA_PARAM_SPECS: dict[str, tuple[float, float, float]] = {
    # name: (p0, lower, upper)
    "base_slope": (0.0, -1.0, 1.0),
    # base_curv [1/nm^2] of baseline="quadratic": +-5 is +-0.9 at 433.05 nm
    # and +-0.5 at 433.80 nm, far above the 0.2-0.4 the data take.
    "base_curv": (0.0, -5.0, 5.0),
    # base_hg of baseline="hgamma": the H-gamma wing level at 433.80 nm (dx = 0), in
    # normalised units; the measured weak spectra take 0.1-0.5.
    "base_hg": (0.05, 0.0, 10.0),
    "I_H2_R01": (1e-3, 0.0, 1.0),
    "I_H2_R2": (1e-3, 0.0, 1.0),
    "I_F1": (1e-3, 0.0, 1.0),
    "I_F3": (1e-3, 0.0, 1.0),
    # second rotational component of rot_components=2 [K] and its population
    # fraction; the first component is the production T_rot.
    "T_rot2": (6000.0, 200.0, 30000.0),
    "frac2": (0.1, 0.0, 1.0),
    # r11_lines="tied": population of the (1-1) upper levels relative to the
    # (0-0) ones, n(v'=1, N'=1) / n(v'=0, N'=1) at the common T_rot.
    "S_v1": (0.4, 0.0, 5.0),
    # stretch=True: relative dispersion error eps of the wavelength grid,
    # x_rest = x + dx + eps * (x - STRETCH_REF_NM). +-0.02 is +-8.5 pm at the
    # blue end of the 433.05-433.80 nm window and +-3 pm at Q(18) and Q(21),
    # several times the 3-4 pm mismatches seen there.
    "stretch": (0.0, -0.02, 0.02),
}


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
    baseline : {"constant", "linear", "quadratic", "hgamma", "fixed"}
        ``"constant"`` (default): the baseline is the constant ``base``.
        ``"linear"``: ``base + base_slope * (x - BASELINE_REF_NM)`` on the
        measured (unshifted) wavelength grid; adds the fit parameter
        ``base_slope`` [1/nm].
        ``"quadratic"``: the linear form plus
        ``base_curv * (x - BASELINE_REF_NM)**2``; adds ``base_slope`` and
        ``base_curv`` [1/nm^2], the curvature bounded to +-5 /nm^2.
        ``"hgamma"``: the linear form plus ``base_hg * L(x + dx)``, the far wing
        of H-gamma (:meth:`hgamma_wing`: a Lorentzian at ``HGAMMA_AIR_NM`` with
        HWHM ``HGAMMA_WING_HWHM_NM``, 1 at 433.80 nm); adds ``base_slope`` and
        ``base_hg`` (bounds [0, 10]). The wing is convex and rises toward
        H-gamma only, unlike the symmetric quadratic.
        ``"fixed"``: a continuum computed before the fit and held: the model
        adds :meth:`continuum_on` (the array given to :meth:`set_continuum`,
        interpolated onto the measured grid) and holds ``base`` at 0
        (:attr:`fixed_params`); no continuum parameter is fitted. ``BHFitter``
        takes a ``continuum`` callable that sets it per spectrum.
    h2_lines : {False, True, "R2"}, default False
        ``True`` adds the two H2 Y 1 -> 2B 0 lines of ``H2_NUISANCE_LINES_NM``
        as free-amplitude Gaussians with the BH line width, on the same
        shifted grid ``x + dx`` as the BH lines; adds the fit parameters
        ``I_H2_R01`` and ``I_H2_R2``. ``"R2"`` adds the resolved R2 line
        only (``I_H2_R2``), for fits that exclude the R0/R1 region from the
        data instead (``BHFitter(exclude_nm=...)``).
    h2_r01_ratio : float or None, default None
        Only with ``h2_lines=True``. ``None`` (default): the two H2 amplitudes
        are free. A positive value ties them: the R0/R1 line (433.2619 nm)
        gets ``h2_r01_ratio * I_H2_R2``, so only ``I_H2_R2`` is fitted. R0/R1
        sits on BH Q(17) and the low-N Q lines and cannot be separated from
        them by the fit; the tie takes its amplitude from the resolved R2
        line. ``DIEKE_H2_R01_OVER_R2`` (1.29/1.20) is Dieke's ratio.
    isotopologue : {"11BH", "natural"}, default "11BH"
        ``"natural"`` adds 10BH at its natural share (``BH10_FRACTION`` =
        0.199 of BH, i.e. 0.248 of 11BH), at the same T_rot: the 10BH (0-0)
        Q lines with their own f-level energies, each isotopologue
        normalised by its own rotational partition function, and the 10BH
        (1-1) R(7)/R(8) lines at 0.248 times the free 11BH R(7)/R(8)
        amplitudes (F&B Table V positions). No extra fit parameters. Needs
        ``upper_energies="fernando_bernath"`` and ``v_max=0``.
    bh10_positions : {"fb91", "mass_scaled"}, default "fb91"
        10BH Q-line positions and energies with ``isotopologue="natural"``
        (:mod:`bh_molecule.isotopes`): F&B Table V where measured and their
        Table VII constants elsewhere, or the mass-scaled 11BH Dunham set.
        They differ for Q(13) and above (see the module).
    extra_lines : tuple of {"F1", "F3"}, default ()
        Unidentified window features of ``EXTRA_LINES_NM`` added as
        free-amplitude Gaussians with the BH line width on the shifted grid,
        like the H2 lines; adds ``I_F1``, ``I_F3`` (bounds [0, 1]).
    rot_components : {1, 2}, default 1
        ``2`` makes the upper-state population a sum of two Boltzmann
        distributions, ``(1 - frac2) n(T_rot) + frac2 n(T_rot2)``, each
        normalised by its partition function, scaled so that ``frac2 = 0``
        is the one-temperature model with the same ``C``. Adds ``T_rot2``
        [K] and ``frac2``. Needs ``v_max=0``.
    r11_lines : {"free", "tied"}, default "free"
        The 11BH (1-1) R(7) and R(8) lines in the window. ``"free"``
        (default): two free amplitudes ``I_R7``, ``I_R8`` at
        ``R11_MODEL_NM``. ``"tied"``: their areas follow from the common
        ``T_rot`` and the Hoenl-London factors of the (1-1) band
        (:meth:`r11_tied_areas`: upper e levels N' = 8, 9 of v' = 1), with
        one free population scale ``S_v1`` = n(v'=1, N'=1) / n(v'=0, N'=1),
        at F&B Table I positions (433.644051 / 433.331292 nm, the 0.0037 nm
        offset of ``R11_MODEL_NM`` removed). ``I_R7`` and ``I_R8`` stay in
        the signature but are ignored, and :attr:`fixed_params` holds them
        at 0 in the fit. Needs ``v_max=0`` and ``rot_components=1``.
    stretch : bool, default False
        ``True`` adds a linear wavelength stretch ``stretch`` (eps,
        dimensionless): the model lines sit on ``x + dx + eps * (x -
        STRETCH_REF_NM)`` instead of ``x + dx``, i.e. a relative dispersion
        error of the measured grid. The Gaussian width is scaled by
        ``1 + eps`` with it, so the line width on the measured grid stays
        ``w_inst`` (the instrument function is fixed in pixels) and only the
        positions move. The baseline stays on the measured grid.

    Notes
    -----
    Extra fit parameters are appended to the production signature of
    :meth:`full_fit_model` in the order of :attr:`extra_param_names`.
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
        baseline: str = "constant",
        h2_lines: bool | str = False,
        h2_r01_ratio: float | None = None,
        isotopologue: str = "11BH",
        bh10_positions: str = "fb91",
        extra_lines: tuple[str, ...] = (),
        rot_components: int = 1,
        r11_lines: str = "free",
        stretch: bool = False,
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
        if baseline not in BASELINES:
            raise ValueError(f"baseline must be one of {BASELINES}")
        if not (isinstance(h2_lines, bool) or h2_lines == "R2"):
            raise ValueError("h2_lines must be True, False or 'R2'")
        if h2_r01_ratio is not None:
            if h2_lines is not True:
                raise ValueError("h2_r01_ratio needs h2_lines=True")
            if isinstance(h2_r01_ratio, bool) or not (
                np.isfinite(float(h2_r01_ratio)) and float(h2_r01_ratio) > 0
            ):
                raise ValueError("h2_r01_ratio must be None or a positive number")
        if isotopologue not in ISOTOPOLOGUES:
            raise ValueError(f"isotopologue must be one of {ISOTOPOLOGUES}")
        if bh10_positions not in BH10_POSITIONS:
            raise ValueError(f"bh10_positions must be one of {BH10_POSITIONS}")
        if isotopologue == "natural" and (
            upper_energies != "fernando_bernath" or int(v_max) != 0
        ):
            raise ValueError(
                "isotopologue='natural' needs upper_energies='fernando_bernath' and v_max=0"
            )
        if isinstance(extra_lines, str):
            extra_lines = (extra_lines,)
        extra_lines = tuple(extra_lines)
        unknown = [n for n in extra_lines if n not in EXTRA_LINES_NM]
        if unknown or len(set(extra_lines)) != len(extra_lines):
            raise ValueError(
                f"extra_lines must be distinct names from {tuple(EXTRA_LINES_NM)}"
            )
        if isinstance(rot_components, bool) or rot_components not in ROT_COMPONENTS:
            raise ValueError(f"rot_components must be one of {ROT_COMPONENTS}")
        if rot_components == 2 and int(v_max) != 0:
            raise ValueError("rot_components=2 needs v_max=0")
        if r11_lines not in R11_LINE_MODES:
            raise ValueError(f"r11_lines must be one of {R11_LINE_MODES}")
        if r11_lines == "tied" and (int(v_max) != 0 or rot_components != 1):
            raise ValueError("r11_lines='tied' needs v_max=0 and rot_components=1")
        if not isinstance(stretch, bool):
            raise ValueError("stretch must be True or False")

        self.v00_wl = v00_wl_df
        self.upper_energies = upper_energies
        self.v_max = int(v_max)
        self.band_positions = band_positions
        self.T_vib = None if T_vib is None else float(T_vib)
        self.baseline = baseline
        self.h2_lines = h2_lines
        self.h2_r01_ratio = None if h2_r01_ratio is None else float(h2_r01_ratio)
        self.isotopologue = isotopologue
        self.bh10_positions = bh10_positions
        # keep the EXTRA_LINES_NM order whatever order was given
        self.extra_lines = tuple(n for n in EXTRA_LINES_NM if n in extra_lines)
        self.rot_components = int(rot_components)
        self.r11_lines = r11_lines
        self.stretch = stretch
        self._bh10 = self._bh10_tables() if isotopologue == "natural" else None

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
        self._r11_fb91 = self._r11_fb91_tables() if r11_lines == "tied" else None
        # Held continuum of baseline="fixed" (measured grid, values); set per
        # spectrum with set_continuum().
        self._continuum: tuple[np.ndarray, np.ndarray] | None = None

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

    @property
    def extra_param_names(self) -> tuple[str, ...]:
        """Fit parameters added by the switches, appended after
        ``(C, T_rot, dx, w_inst, base, I_R7, I_R8)``; empty for the defaults."""
        names: list[str] = []
        if self.baseline in ("linear", "quadratic", "hgamma"):
            names.append("base_slope")
        if self.baseline == "quadratic":
            names.append("base_curv")
        if self.baseline == "hgamma":
            names.append("base_hg")
        if self.h2_lines == "R2":
            names.append("I_H2_R2")
        elif self.h2_lines:
            if self.h2_r01_ratio is None:
                names.extend(H2_NUISANCE_LINES_NM)
            else:
                names.append("I_H2_R2")  # R0/R1 follows as h2_r01_ratio * I_H2_R2
        names.extend(f"I_{n}" for n in self.extra_lines)
        if self.rot_components == 2:
            names.extend(("T_rot2", "frac2"))
        if self.r11_lines == "tied":
            names.append("S_v1")
        if self.stretch:
            names.append("stretch")
        return tuple(names)

    @property
    def fixed_params(self) -> dict[str, float]:
        """Parameters of the production seven that this model ignores, with
        the value the fitter holds them at (``BHFitter`` eliminates them as it
        does a fixed ``w_inst``); empty for the defaults. ``baseline="fixed"``
        holds ``base`` at 0 (the continuum is the held array)."""
        out: dict[str, float] = {}
        if self.baseline == "fixed":
            out["base"] = 0.0
        if self.r11_lines == "tied":
            out.update({"I_R7": 0.0, "I_R8": 0.0})
        return out

    # MARK: held continuum
    def set_continuum(self, x, values) -> None:
        """Set the held continuum of ``baseline="fixed"``: ``values`` on the
        measured wavelength grid ``x`` [nm] (any range covering the fit window;
        :meth:`continuum_on` interpolates linearly)."""
        if self.baseline != "fixed":
            raise ValueError('set_continuum needs baseline="fixed"')
        x = np.asarray(x, dtype=float).ravel()
        v = np.asarray(values, dtype=float).ravel()
        if x.size != v.size or x.size < 2:
            raise ValueError("continuum x and values must have the same length >= 2")
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(v))):
            raise ValueError("continuum x and values must be finite")
        order = np.argsort(x)
        x, v = x[order], v[order]
        if np.any(np.diff(x) <= 0):
            raise ValueError("continuum x must be strictly increasing once sorted")
        self._continuum = (x, v)

    def continuum_on(self, x) -> np.ndarray:
        """The held continuum on the measured grid ``x`` (``baseline="fixed"``).

        Raises if none is set or ``x`` leaves its range (by more than 1e-9 nm)."""
        if self._continuum is None:
            raise ValueError('baseline="fixed" needs a continuum: call set_continuum(x, values)')
        cx, cv = self._continuum
        x = np.asarray(x, dtype=float)
        if x.size and (x.min() < cx[0] - 1e-9 or x.max() > cx[-1] + 1e-9):
            raise ValueError(
                f"fit grid {x.min():.4f}-{x.max():.4f} nm leaves the held continuum "
                f"{cx[0]:.4f}-{cx[-1]:.4f} nm"
            )
        return np.interp(x, cx, cv)

    @property
    def extra_p0(self) -> tuple[float, ...]:
        """Start values of :attr:`extra_param_names`."""
        return tuple(EXTRA_PARAM_SPECS[n][0] for n in self.extra_param_names)

    @property
    def extra_bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """``(lower, upper)`` bounds of :attr:`extra_param_names`."""
        names = self.extra_param_names
        return (
            tuple(EXTRA_PARAM_SPECS[n][1] for n in names),
            tuple(EXTRA_PARAM_SPECS[n][2] for n in names),
        )

    def _line_positions(self, v: int, branch: Branch) -> np.ndarray:
        """Line centres [nm] indexed by table row J, under ``band_positions``."""
        table = self.v00_wl if self.band_positions == "v00" else self._band_wl[v]
        return table[branch.value].to_numpy()

    # MARK: (1-1) R(7)/R(8)
    def _r11_fb91_tables(self) -> dict:
        """F&B Table I positions [nm, air] and upper e-level energies above
        v' = 1, N' = 1 [cm^-1] of the (1-1) R(7) and R(8) lines."""
        from .dataio import load_band_wavelengths

        r = load_band_wavelengths(1).set_index("J")["R"]
        E1 = self.upper_energy(1, 1, Branch.R)
        out = {}
        for J in (7, 8):
            N2 = J + 1
            out[f"R{J}"] = {
                "wl": float(r.loc[J]),
                "N2": N2,
                "dF": float(self.upper_energy(1, N2, Branch.R) - E1),
            }
        return out

    def r11_tied_areas(self, C: float, T_rot: float, S_v1: float) -> tuple[float, float]:
        r"""Areas of the (1-1) R(7) and R(8) lines for ``r11_lines="tied"``.

        In the units of :meth:`full_fit_model` (the Q branch is ``1e8 *``
        :meth:`spectrum`): ``1e8 C S_v1 g (2N'+1) exp(-dF / k T_rot)
        A(1, N', N'') h nu / 4 pi`` with N' = N'' + 1, dF the upper e-level
        energy above v' = 1, N' = 1 (``upper_energies``), and
        :meth:`A_coeff`'s R-branch Hoenl-London factor (N' + 1)/2 and v' = 1
        band coefficient. ``S_v1`` is therefore n(v'=1, N'=1) / n(v'=0, N'=1).
        """
        if self._r11_fb91 is None:
            raise ValueError("r11_tied_areas needs r11_lines='tied'")
        areas = []
        for J in (7, 8):
            line = self._r11_fb91[f"R{J}"]
            N2 = line["N2"]
            pop = self.g_as * (2 * N2 + 1) * np.exp(-line["dF"] / (T_rot * self.K2wn))
            photon_E = spc.h * (spc.c / (line["wl"] * 1e-9)) / (4 * np.pi)
            areas.append(1e8 * C * S_v1 * photon_E * pop * self.A_coeff(1, N2, J))
        return float(areas[0]), float(areas[1])

    # MARK: 10BH, partition, two-T
    def _bh10_tables(self) -> dict:
        """10BH Q-line positions [nm, air], f-level energies and R(7)/R(8) [nm]."""
        wn, src = bh10_q_wavenumbers(N_max=30, scheme=self.bh10_positions)
        N = np.arange(wn.size)
        r11 = bh10_r11_wavenumbers(self.bh10_positions)
        return {
            "wl_Q": np.asarray(wavenumber_to_air_nm(wn), dtype=float),
            "source_Q": src,
            "E_f": np.asarray(bh10_upper_f_energy(N, 0, self.bh10_positions), dtype=float),
            "wl_R7": float(wavenumber_to_air_nm(r11[7])),
            "wl_R8": float(wavenumber_to_air_nm(r11[8])),
        }

    def _f_level_offsets(self, isotope: str = "11BH", N2_max: int = 22) -> np.ndarray:
        """f-level energies above N = 1 [cm^-1] of the (v=0) Q-branch upper levels, N = 1..N2_max."""
        cache = self.__dict__.setdefault("_f_offsets", {})
        key = (isotope, int(N2_max))
        if key not in cache:
            N = np.arange(1, N2_max + 1)
            if isotope == "10BH":
                E = self._bh10["E_f"][N] - self._bh10["E_f"][1]
            else:
                E = np.array([self.upper_energy(0, int(n), Branch.Q) for n in N])
                E = E - self.upper_energy(0, 1, Branch.Q)
            cache[key] = E
        return cache[key]

    def rot_partition(self, T_rot: float, isotope: str = "11BH", N2_max: int = 22) -> float:
        r"""Rotational partition sum of the Q-branch upper levels (v' = 0, f),
        :math:`Z = \sum_{N=1}^{N_2^{max}} (2N+1)\,e^{-\Delta F_N / k T}`, with
        :math:`\Delta F_N` above N = 1, as :meth:`spectrum` weights them."""
        N = np.arange(1, N2_max + 1)
        dF = self._f_level_offsets(isotope, N2_max)
        return float(np.sum((2 * N + 1) * np.exp(-dF / (T_rot * self.K2wn))))

    def bh10_spectrum(
        self, x: np.ndarray, T_rot: float, w_inst: float, T_tra: float, N2_max: int = 22
    ) -> np.ndarray:
        """10BH (0-0) Q branch with C = 1, normalised like the 11BH :meth:`spectrum`
        (each at its own N = 1); see :meth:`full_fit_model` for the abundance
        and partition-function scaling."""
        if self._bh10 is None:
            raise ValueError("bh10_spectrum needs isotopologue='natural'")
        y = np.zeros_like(x, dtype=float)
        dF = self._f_level_offsets("10BH", N2_max)
        for i, N2 in enumerate(range(1, N2_max + 1)):
            wl = self._bh10["wl_Q"][N2]
            pop = self.g_as * (2 * N2 + 1) * np.exp(-dF[i] / (T_rot * self.K2wn))
            photon_E = spc.h * (spc.c / (wl * 1e-9)) / (4 * np.pi)
            A_ul = self.A_coeff(0, N2, N2)
            y += photon_E * pop * A_ul * self.line_profile(x, wl, w_inst, T_tra)
        return y

    def q_band(self, xs: np.ndarray, T_rot: float, w_inst: float, T_tra: float = 0.0) -> np.ndarray:
        """Q-branch emission with C = 1 on the shifted grid ``xs``: 11BH, plus
        10BH at ``BH10_OVER_BH11 * Z11 / Z10`` for ``isotopologue="natural"``."""
        y = self.spectrum(xs, 1.0, T_rot, w_inst, T_tra, Branch.Q)
        if self._bh10 is not None:
            scale = BH10_OVER_BH11 * self.rot_partition(T_rot) / self.rot_partition(T_rot, "10BH")
            y = y + scale * self.bh10_spectrum(xs, T_rot, w_inst, T_tra)
        return y

    # MARK: H-gamma wing
    @staticmethod
    def hgamma_wing(xs: np.ndarray, power: float = HGAMMA_WING_POWER) -> np.ndarray:
        r"""Far wing of H-gamma for ``baseline="hgamma"`` on the rest-frame grid ``xs``.

        :math:`L(d) = [1 + (d/\gamma)^p]^{-1} / [1 + (d_0/\gamma)^p]^{-1}` with
        :math:`d = |x_s - \lambda_{H\gamma}|`, :math:`\gamma` =
        ``HGAMMA_WING_HWHM_NM``, :math:`d_0` = ``HGAMMA_WING_REF_NM`` and
        :math:`p` = ``power`` (2: Lorentzian); 1 at 0.25 nm from the line centre.
        """
        d = np.abs(np.asarray(xs, dtype=float) - HGAMMA_AIR_NM)
        g = HGAMMA_WING_HWHM_NM
        ref = 1.0 / (1.0 + (HGAMMA_WING_REF_NM / g) ** power)
        return (1.0 / (1.0 + (d / g) ** power)) / ref

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
        *extra: float,
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
            Constant baseline; with ``baseline="linear"``, its value at
            ``BASELINE_REF_NM``.
        I_R7 : float
            Amplitude of auxiliary line at :math:`\lambda_{R7}`.
        I_R8 : float
            Amplitude of auxiliary line at :math:`\lambda_{R8}`.
        *extra : float
            Values of :attr:`extra_param_names`, in that order (none for the
            default model): ``base_slope`` [1/nm] for ``baseline="linear"``
            (and ``base_curv`` [1/nm^2] after it for ``"quadratic"``),
            then ``I_H2_R01`` and ``I_H2_R2`` for ``h2_lines=True`` (only
            ``I_H2_R2`` when ``h2_r01_ratio`` ties the two, or for
            ``h2_lines="R2"``), then ``I_F1``, ``I_F3`` of ``extra_lines``,
            then ``T_rot2``, ``frac2`` for ``rot_components=2``,
            then ``S_v1`` for ``r11_lines="tied"`` (``I_R7`` and ``I_R8``
            are then ignored), then ``stretch`` for ``stretch=True``.

        Returns
        -------
        np.ndarray
            Model evaluated on ``x``.

        See Also [Physics explainer — full fit model](../phys.md#full-fit-model)
        """
        names = self.extra_param_names
        if len(extra) != len(names):
            raise TypeError(
                f"full_fit_model expects {len(names)} extra parameter(s) "
                f"{names} for this model, got {len(extra)}"
            )
        wl_R7 = R11_MODEL_NM["R7"]  # 1-1, R7
        wl_R8 = R11_MODEL_NM["R8"]  # 1-1, R8
        T_tra = 0.0
        values = dict(zip(names, extra))
        if self.stretch:
            # relative dispersion error; the width on the measured grid stays w_inst
            eps = values["stretch"]
            xs = x + dx + eps * (np.asarray(x, dtype=float) - STRETCH_REF_NM)
            w_inst = w_inst * (1.0 + eps)
        else:
            xs = x + dx
        if self._r11_fb91 is not None:
            # (1-1) R(7)/R(8) from T_rot and Hoenl-London, F&B Table I positions
            I_R7, I_R8 = self.r11_tied_areas(C, T_rot, values["S_v1"])
            wl_R7 = self._r11_fb91["R7"]["wl"]
            wl_R8 = self._r11_fb91["R8"]["wl"]
        if self._bh10 is None and self.rot_components == 1:
            # production path, unchanged
            band = 1e8 * self.spectrum(xs, C, T_rot, w_inst, T_tra, Branch.Q)
        else:
            band = self.q_band(xs, T_rot, w_inst, T_tra)
            if self.rot_components == 2:
                T2, f2 = values["T_rot2"], values["frac2"]
                ratio = self.rot_partition(T_rot) / self.rot_partition(T2)
                band = (1.0 - f2) * band + f2 * ratio * self.q_band(xs, T2, w_inst, T_tra)
            band = 1e8 * C * band
        y = (
            band
            + I_R7 * self.line_profile(xs, wl_R7, w_inst, T_tra)
            + I_R8 * self.line_profile(xs, wl_R8, w_inst, T_tra)
            + base
        )
        if self._bh10 is not None:
            # 10BH (1-1) R(7)/R(8) at the natural share of the free 11BH amplitudes
            y = y + BH10_OVER_BH11 * (
                I_R7 * self.line_profile(xs, self._bh10["wl_R7"], w_inst, T_tra)
                + I_R8 * self.line_profile(xs, self._bh10["wl_R8"], w_inst, T_tra)
            )
        if self.baseline == "fixed":
            y = y + self.continuum_on(x)
        if not names:
            return y
        if self.baseline in ("linear", "quadratic", "hgamma"):
            u = np.asarray(x, dtype=float) - BASELINE_REF_NM
            y = y + values["base_slope"] * u
            if self.baseline == "quadratic":
                y = y + values["base_curv"] * u**2
            elif self.baseline == "hgamma":
                y = y + values["base_hg"] * self.hgamma_wing(xs)
        if self.h2_lines == "R2":
            y = y + values["I_H2_R2"] * self.line_profile(
                xs, H2_NUISANCE_LINES_NM["I_H2_R2"], w_inst, T_tra
            )
        elif self.h2_lines:
            amps = dict(values)
            if self.h2_r01_ratio is not None:
                amps["I_H2_R01"] = self.h2_r01_ratio * values["I_H2_R2"]
            for name, wl in H2_NUISANCE_LINES_NM.items():
                y = y + amps[name] * self.line_profile(xs, wl, w_inst, T_tra)
        for name in self.extra_lines:
            y = y + values[f"I_{name}"] * self.line_profile(
                xs, EXTRA_LINES_NM[name], w_inst, T_tra
            )
        return y
