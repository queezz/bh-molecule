"""Instrument loaders for bh_molecule.

This package exposes instrument-specific loaders such as ``vis133m``
and wavelength calibration utilities.
"""

from .vis133m import Vis133M
from .wavecal import (
    load_wavecal_csv,
    csv_to_linear_formulas,
    fit_peak_gaussian,
    measure_peak_from_cube,
    compute_wavelength_shift,
    compute_calibration_from_reference,
    save_bh_wavecal_json,
    load_bh_wavecal_json,
    apply_polynomial_wavecal,
    estimate_cw_from_features,
    get_cw_from_header,
)

__all__ = [
    "vis133m",
    "Vis133M",
    "load_wavecal_csv",
    "csv_to_linear_formulas",
    "fit_peak_gaussian",
    "measure_peak_from_cube",
    "compute_wavelength_shift",
    "compute_calibration_from_reference",
    "save_bh_wavecal_json",
    "load_bh_wavecal_json",
    "apply_polynomial_wavecal",
    "estimate_cw_from_features",
    "get_cw_from_header",
]
