from .physics import BHModel, Branch, MolecularConstants
from .dataio import load_v00_wavelengths
from .fit import BHFitter
from .calibration import (
    analyze_wavelength_linearity,
    linear_fit_residuals,
    plot_linearity_summary,
    plot_fiber_line_fits,
)

from .instruments.vis133m import Vis133M
from .workflows.batch_fit import run_bh_batch, run_folder_batch

from . import plotting
from .plotting import use_dark, reset_light, dark_theme

__all__ = [
    "BHModel",
    "Branch",
    "MolecularConstants",
    "load_v00_wavelengths",
    "BHFitter",
    "analyze_wavelength_linearity",
    "linear_fit_residuals",
    "plot_linearity_summary",
    "plot_fiber_line_fits",
    "Vis133M",
    "run_bh_batch",
    "run_folder_batch",
    "plotting",
    "use_dark",
    "reset_light",
    "dark_theme",
]

__version__ = __import__("importlib.metadata").metadata.version("bh-molecule")
