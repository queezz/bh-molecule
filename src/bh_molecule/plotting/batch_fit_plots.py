"""Batch fit grid plotting for the BH workflow.

Used by workflows.batch_fit to save normalized (frame × channel) grid PDFs.
Single-fit visualization remains in fit_plots.py.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


# Optional process-memory diagnostics.  ``psutil`` is an optional dependency:
# if it is missing (or RSS lookup fails), ``_log_rss`` becomes a silent no-op
# so batch runs keep working in minimal environments.  Set
# ``BH_BATCH_MEM_LOG=0`` to silence the prints when psutil is available.
_MEM_LOG_ENABLED = os.environ.get("BH_BATCH_MEM_LOG", "1").lower() not in (
    "",
    "0",
    "false",
    "no",
    "off",
)
try:
    import psutil as _psutil  # type: ignore[import]

    _MEM_PROC = _psutil.Process(os.getpid())
except Exception:  # pragma: no cover - psutil truly optional
    _psutil = None
    _MEM_PROC = None


def _log_rss(label: str) -> None:
    """Print a one-line RSS snapshot of the current process if possible."""
    if not _MEM_LOG_ENABLED or _MEM_PROC is None:
        return
    try:
        rss_gb = _MEM_PROC.memory_info().rss / 1024**3
        print(f"[MEM] {label}: RSS = {rss_gb:.2f} GB")
    except Exception:  # pragma: no cover - diagnostics must never crash
        pass


def normalize_curves_for_grid(
    y: np.ndarray,
    yfit: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Linearly rescale data and fit to [0, 1] using **data** ``y`` min/max only.

    The fit curve is drawn in the same normalized vertical system as the data
    (both use ``vmin``, ``vmax`` from ``y``).  This avoids the misleading case
    where joint min/max over ``y`` and ``yfit`` squashes a low-amplitude (but
    shape-correct) fit against the bottom of the panel when the optimizer
    returns a much smaller absolute scale than the data.

    Parameters
    ----------
    y : ndarray
        Observed spectrum in the fit window.
    yfit : ndarray or None
        Model spectrum on the same grid.

    Returns
    -------
    y_n, yfit_n
        Normalized arrays; ``y_n`` spans [0, 1] when ``y`` is non-constant.
    """
    y = np.asarray(y, dtype=float)
    vmin = float(np.min(y))
    vmax = float(np.max(y))
    if vmax <= vmin:
        y_n = np.zeros_like(y)
        yfit_n = (
            np.zeros_like(y, dtype=float)
            if yfit is not None
            else None
        )
        return y_n, yfit_n
    y_n = (y - vmin) / (vmax - vmin)
    if yfit is None:
        return y_n, None
    yfit = np.asarray(yfit, dtype=float)
    yfit_n = (yfit - vmin) / (vmax - vmin)
    return y_n, yfit_n


def save_batch_fit_grid(
    curves: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]],
    frames: list[int],
    channels: list[int],
    *,
    pdf_path: str | Path,
    channels_per_page: int = 6,
    rasterize_data: bool = True,
) -> None:
    """Plot normalized spectra in a (frames × channels) grid and save as multi-page PDF.

    - Rows correspond to frames, columns to channels (grouped into pages).
    - Each panel rescales **data** to [0, 1]; the fit uses the same linear map
      (:func:`normalize_curves_for_grid`) so overlay shape matches the fit window.
    - Data: markers only (no line), optionally rasterized in PDF. Fit: smooth line (vector).
    - Axes are simplified for compact overview.

    Parameters
    ----------
    curves : dict
        Mapping (frame, channel) -> (x, y, yfit) arrays.
    frames : list of int
        Frame indices (order preserved for grid rows).
    channels : list of int
        Channel indices (order preserved for grid columns).
    pdf_path : path-like
        Output PDF path.
    channels_per_page : int, optional
        Number of channel columns per page. Default 6.
    rasterize_data : bool, optional
        If True (default), rasterize raw data points in the PDF to keep file size and
        render time small; fit curves and axes remain vector.
    """
    frames_sorted = sorted(set(frames))
    channels_sorted = sorted(set(channels))
    if not frames_sorted or not channels_sorted:
        return

    _log_rss(f"save_batch_fit_grid start: {Path(pdf_path).name}")

    with PdfPages(str(pdf_path)) as pdf:
        for i_start in range(0, len(channels_sorted), channels_per_page):
            page_channels = channels_sorted[i_start : i_start + channels_per_page]
            if not page_channels:
                continue

            nrows = len(frames_sorted)
            ncols = len(page_channels)
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(2.4 * ncols, 1.8 * nrows),
                squeeze=False,
            )

            fig.suptitle("BH band spectra (normalized)", y=0.995)

            for r, frame in enumerate(frames_sorted):
                for c, ch in enumerate(page_channels):
                    ax = axes[r, c]
                    key = (frame, ch)
                    if key not in curves:
                        ax.set_axis_off()
                        continue

                    x, y, yfit = curves[key]
                    x = np.asarray(x, dtype=float)
                    y = np.asarray(y, dtype=float)
                    yfit = (
                        np.asarray(yfit, dtype=float) if yfit is not None else None
                    )

                    y_n, yfit_n = normalize_curves_for_grid(y, yfit)

                    # Data: markers only (rasterized in PDF). Fit: smooth line (vector).
                    ax.plot(
                        x,
                        y_n,
                        ".",
                        ms=1.5,
                        linestyle="none",
                        color="k",
                        alpha=0.7,
                        rasterized=rasterize_data,
                    )
                    if yfit_n is not None:
                        ax.plot(x, yfit_n, "-", lw=1.5, color="#7397de")

                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    ax.set_ylim(-0.05, 1.05)
                    ax.set_yticks([0.0, 1.0])
                    ax.set_yticklabels(["0", "1"], fontsize=7)

                    if x.size > 1:
                        xticks = np.linspace(x.min(), x.max(), 3)
                        ax.set_xticks(xticks)
                        ax.set_xticklabels([f"{t:.2f}" for t in xticks], fontsize=7)
                    else:
                        ax.set_xticks([])

                    if getattr(ax, "legend_", None):
                        ax.legend_.remove()

                    if r == 0:
                        ax.set_title(f"ch {ch}", fontsize=9)
                    if c == 0:
                        ax.text(
                            -0.1,
                            0.5,
                            f"f {frame}",
                            transform=ax.transAxes,
                            ha="right",
                            va="center",
                            fontsize=9,
                        )

            # NOTE: avoid ``tight_layout`` + ``bbox_inches="tight"`` + explicit
            # ``dpi=200`` here.  In long batch runs that combination forces an
            # extra full-figure RendererAgg allocation per page just to
            # measure ink bounds, which fragments process memory and
            # eventually triggers ``MemoryError: bad allocation`` inside
            # ``RendererAgg(int(width), int(height), dpi)`` after dozens of
            # shots.  A fixed ``subplots_adjust`` layout plus a plain
            # ``pdf.savefig(fig)`` keeps the output visually equivalent (PDFs
            # are vector anyway) while drastically lowering peak RAM.
            fig.subplots_adjust(
                left=0.06,
                right=0.98,
                bottom=0.05,
                top=0.95,
                hspace=0.25,
                wspace=0.15,
            )
            pdf.savefig(fig)
            _log_rss(f"page {1 + i_start // channels_per_page} after pdf.savefig")
            plt.close(fig)
            gc.collect()
            _log_rss(f"page {1 + i_start // channels_per_page} after plt.close+gc")

    _log_rss(f"save_batch_fit_grid done: {Path(pdf_path).name}")


__all__ = ["save_batch_fit_grid", "normalize_curves_for_grid"]
