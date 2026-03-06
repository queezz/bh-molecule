from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

try:
    # Prefer rich notebook progress bars when available.
    from tqdm.notebook import tqdm  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - fallback for non-notebook envs
    from tqdm import tqdm

from bh_molecule.instruments import Vis133M


def export_band_maps_pdf(
    fits_files,
    output_pdf,
    cw_nm,
    nm_range,
    *,
    cmap=None,
    cbar_label="intensity",
    channel_line=None,
    require_time=False,
    subtract_dark=True,
    log_scale=False,
):
    """Export per-file band-map plots to a multi-page PDF.

    Parameters
    ----------
    fits_files : iterable
        Iterable of FITS file paths.
    output_pdf : path-like
        Output PDF path.
    cw_nm : float
        Central wavelength (nm) to apply via ``Vis133M.apply_cw``.
    nm_range : tuple
        ``(lo, hi)`` wavelength range in nanometres for ``Vis133M.plot_band_map``.
    cmap, cbar_label, channel_line, require_time, subtract_dark, log_scale :
        Forwarded to ``Vis133M.plot_band_map``.
    """
    # Materialise list so we can scan and then render.
    paths = [Path(p) for p in fits_files]

    # First pass: determine global intensity scale.
    vmin = np.inf
    vmax = -np.inf

    for path in tqdm(paths, desc="Scanning intensity"):
        s = Vis133M.from_files(path)
        s.apply_cw(cw_nm=cw_nm)

        img = s.band(nm_range, subtract_dark=subtract_dark)

        if log_scale:
            # For logarithmic colour scaling we must use the original
            # linear intensities but ignore non-positive values so that
            # LogNorm can derive sensible limits.
            img = img[img > 0]

        if img.size > 0:
            vmin = min(vmin, float(img.min()))
            vmax = max(vmax, float(img.max()))

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin = None
        vmax = None

    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_pdf) as pdf:
        for path in tqdm(paths, desc="Rendering band maps"):
            s = Vis133M.from_files(path)
            s.apply_cw(cw_nm=cw_nm)

            fig, ax = plt.subplots(figsize=(7.6, 4.6))

            s.plot_band_map(
                nm_range,
                ax=ax,
                cmap=cmap,
                cbar_label=cbar_label,
                channel_line=channel_line,
                require_time=require_time,
                subtract_dark=subtract_dark,
                log_scale=log_scale,
                vmin=vmin,
                vmax=vmax,
            )

            fig.suptitle(path.name, y=0.995)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def export_band_maps_from_folder(
    folder,
    output_pdf,
    cw_nm,
    nm_range,
    pattern="*.fits",
):
    """Export band maps for all FITS files in a folder."""
    folder = Path(folder)
    files = sorted(folder.glob(pattern))
    return export_band_maps_pdf(files, output_pdf, cw_nm, nm_range)


__all__ = ["export_band_maps_pdf", "export_band_maps_from_folder"]
