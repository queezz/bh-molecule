from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from bh_molecule.instruments import Vis133M


def export_band_maps_pdf(
    fits_files,
    output_pdf,
    cw_nm,
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
    """
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_pdf) as pdf:
        for path in fits_files:
            path = Path(path)

            s = Vis133M.from_files(path)
            s.apply_cw(cw_nm=cw_nm)

            fig, ax = plt.subplots(figsize=(7.6, 4.6))

            # Prefer the API described in the batch-export design, but keep a
            # compatibility fallback for the current Vis133M signature.
            try:
                s.plot_band_map(ax=ax)
            except TypeError:
                nm_range = (float(cw_nm) - 0.25, float(cw_nm) + 0.25)
                s.plot_band_map(nm_range, ax=ax)

            fig.suptitle(path.name, y=0.995)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def export_band_maps_from_folder(
    folder,
    output_pdf,
    cw_nm,
    pattern="*.fits",
):
    """Export band maps for all FITS files in a folder."""
    folder = Path(folder)
    files = sorted(folder.glob(pattern))
    return export_band_maps_pdf(files, output_pdf, cw_nm)


__all__ = ["export_band_maps_pdf", "export_band_maps_from_folder"]

