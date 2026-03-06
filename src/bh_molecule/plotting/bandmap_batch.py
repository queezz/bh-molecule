from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FixedLocator, FormatStrFormatter

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
    grid=None,
    grid_figsize=(11, 8.5),
    show_axes=False,
    show_titles=True,
    title_fontsize=8,
    single_colorbar=True,
    show_spines=False,
    annotate=None,
    tight_layout=True,
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
    grid : tuple (rows, cols) | None
        If set, render band maps in a contact-sheet grid; otherwise one per page.
    grid_figsize : tuple
        Figure size for grid pages.
    show_axes : bool
        If False, remove xticks, yticks, xlabel, ylabel on grid subplots.
    show_titles : bool
        If True, show shot number (path.stem) as subplot title.
    title_fontsize : int
        Font size for subplot titles.
    single_colorbar : bool
        If True (default), one colorbar per page is drawn from the last subplot's
        image (all subplots share vmin/vmax). If False, each subplot gets its own
        colorbar.
    show_spines : bool
        If False, hide subplot borders (spines).
    annotate : None | "max" | "snr"
        If "max", show max intensity in the upper-left corner of each subplot.
    tight_layout : bool
        Unused for grid pages; grid uses constrained_layout=True instead.
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
        if grid is None:
            # One band-map per page.
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
        else:
            # Contact-sheet style grid of band-maps per page.
            rows, cols = grid
            plots_per_page = max(int(rows) * int(cols), 1)

            for start in tqdm(
                range(0, len(paths), plots_per_page), desc="Rendering pages"
            ):
                batch = paths[start : start + plots_per_page]

                fig, axes = plt.subplots(
                    rows, cols, figsize=grid_figsize, constrained_layout=True
                )
                # Normalise axes array for both scalar and 2D cases.
                if hasattr(axes, "ravel"):
                    axes_flat = axes.ravel()
                else:  # pragma: no cover - degenerate 1x1 case
                    axes_flat = [axes]

                im_last = None  # last imshow result for page colorbar
                # Draw each band map into the grid.
                for ax, path in zip(axes_flat, batch):
                    s = Vis133M.from_files(path)
                    s.apply_cw(cw_nm=cw_nm)

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
                        colorbar=not single_colorbar,
                    )
                    if single_colorbar:
                        im_last = ax.images[0]
                    if show_titles:
                        ax.set_title(path.stem, fontsize=title_fontsize)
                    else:
                        ax.set_title("")

                    if annotate == "max":
                        img = ax.images[0].get_array()
                        ax.text(
                            0.02,
                            0.95,
                            f"{float(np.max(img)):.1e}",
                            transform=ax.transAxes,
                            fontsize=6,
                            color="white",
                            va="top",
                        )
                    elif annotate == "snr":
                        # SNR annotation: extend as needed (e.g. compute from data).
                        pass

                # Axis and spine control for used axes.
                for ax in axes_flat[: len(batch)]:
                    if not show_axes:
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_xlabel("")
                        ax.set_ylabel("")
                    if not show_spines:
                        for spine in ax.spines.values():
                            spine.set_visible(False)

                # One colorbar per page when single_colorbar (same norm via vmin/vmax).
                if single_colorbar and im_last is not None:
                    cb = fig.colorbar(
                        im_last,
                        ax=axes,
                        location="right",
                        fraction=0.025,
                    )
                    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1e"))
                    if vmin is not None and vmax is not None:
                        cb.ax.yaxis.set_major_locator(
                            FixedLocator(np.linspace(vmin, vmax, 4))
                        )

                # Hide any unused axes on the last page.
                for ax in axes_flat[len(batch) :]:
                    ax.axis("off")

                pdf.savefig(fig)
                plt.close(fig)


def export_band_maps_from_folder(
    folder,
    output_pdf,
    cw_nm,
    nm_range,
    pattern="*.fits",
    **kwargs,
):
    """Export band maps for all FITS files in a folder.

    Example
    -------
    export_band_maps_from_folder(
        folder,
        "bh_maps.pdf",
        cw_nm=431.91,
        nm_range=(431.0, 433.5),
        grid=(2, 3),
    )
    """
    folder = Path(folder)
    files = sorted(folder.glob(pattern))
    return export_band_maps_pdf(files, output_pdf, cw_nm, nm_range, **kwargs)


__all__ = ["export_band_maps_pdf", "export_band_maps_from_folder"]
