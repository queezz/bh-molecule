from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Any
import pickle

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from bh_molecule.instruments.vis133m import Vis133M
from bh_molecule.physics import BHModel
from bh_molecule.fit import BHFitter
from .signal_scan import check_background_flat, scan_signal_frames


def _normalize_indices(values: Iterable[int] | int) -> list[int]:
    """Return a list of indices given an int or iterable of ints."""
    if isinstance(values, int):
        return [values]
    return [int(v) for v in values]


def _iter_with_tqdm(seq, desc: str):
    """Wrap an iterable with tqdm when available, else return as-is."""
    try:
        from tqdm.auto import tqdm  # type: ignore[import]

        return tqdm(seq, desc=desc)
    except Exception:
        return seq


def _batch_with_progress(
    fitr: BHFitter,
    frames: list[int],
    channels: list[int],
    run_fit_limit: int | None = None,
):
    """Run fits with a progress bar over (frame, channel) pairs.

    run_fit_limit: if set, run only the first N fits after selection (for testing).
    """
    rows = []
    curves: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    pairs = [(f, ch) for f in frames for ch in channels]
    if run_fit_limit is not None:
        pairs = pairs[:run_fit_limit]
        if len(pairs) < run_fit_limit:
            print(
                f"Run limit {run_fit_limit} requested but only {len(pairs)} (frame, channel) pairs selected."
            )

    for f, ch in _iter_with_tqdm(pairs, desc="Fits"):
        try:
            r = fitr.fit(f, ch, return_fit=True)
            params, errs = r["params"], r["errors"]
            x, y = r["x"], r["y"]
            yfit = r.get("yfit", None)
            dof = max(len(y) - len(params), 1)
            chi2 = (
                float(np.sum((y - yfit) ** 2)) / dof
                if yfit is not None
                else np.nan
            )
            ss_res = (
                float(np.sum((y - yfit) ** 2)) if yfit is not None else np.nan
            )
            ss_tot = (
                float(np.sum((y - np.mean(y)) ** 2))
                if yfit is not None
                else np.nan
            )
            r2 = (
                1.0 - ss_res / ss_tot
                if yfit is not None and ss_tot > 0
                else np.nan
            )
            row = {
                "frame": f,
                "channel": ch,
                **{n: v for n, v in zip(fitr.param_names, params)},
                **{
                    f"{n}_err": e
                    for n, e in zip(fitr.param_names, errs)
                },
                "chi2_red": chi2,
                "R2": r2,
                "npts": len(y),
            }
            rows.append(row)
            if yfit is not None:
                curves[(f, ch)] = (x, y, yfit)
        except Exception as e:  # pragma: no cover - defensive
            rows.append({"frame": f, "channel": ch, "error": repr(e)})

    df = pd.DataFrame(rows).sort_values(["frame", "channel"]).reset_index(drop=True)
    return df, curves


def _plot_normalized_grid_pages(
    curves: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]],
    frames: list[int],
    channels: list[int],
    grid_pdf_path: Path,
    channels_per_page: int = 6,
) -> None:
    """Plot normalized spectra in a structured (frames × channels) grid.

    - Rows correspond to frames.
    - Columns correspond to channels (grouped into pages).
    - Each spectrum is normalized to [0, 1] for visualization.
    - Axes are simplified for compact overview.
    """
    frames_sorted = sorted(set(frames))
    channels_sorted = sorted(set(channels))
    if not frames_sorted or not channels_sorted:
        return

    with PdfPages(str(grid_pdf_path)) as pdf:
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

            # Global title for the page
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
                    yfit = np.asarray(yfit, dtype=float) if yfit is not None else None

                    # Normalize to [0, 1]
                    vals = [y]
                    if yfit is not None:
                        vals.append(yfit)
                    all_vals = np.concatenate(vals)
                    vmin = float(np.min(all_vals)) if all_vals.size else 0.0
                    vmax = float(np.max(all_vals)) if all_vals.size else 1.0
                    if vmax > vmin:
                        y_n = (y - vmin) / (vmax - vmin)
                        yfit_n = (yfit - vmin) / (vmax - vmin) if yfit is not None else None
                    else:
                        y_n = np.zeros_like(y)
                        yfit_n = np.zeros_like(y) if yfit is not None else None

                    ax.plot(x, y_n, lw=1.0, color="k")
                    if yfit_n is not None:
                        ax.plot(x, yfit_n, lw=1.2, color="#7397de")

                    # Simplify axes
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    ax.set_ylim(-0.05, 1.05)
                    ax.set_yticks([0.0, 1.0])
                    ax.set_yticklabels(["0", "1"], fontsize=7)

                    # Keep a few coarse x ticks
                    if x.size > 1:
                        xticks = np.linspace(x.min(), x.max(), 3)
                        ax.set_xticks(xticks)
                        ax.set_xticklabels([f"{t:.2f}" for t in xticks], fontsize=7)
                    else:
                        ax.set_xticks([])

                    ax.legend_.remove() if getattr(ax, "legend_", None) else None

                    # Channel labels at top of each column
                    if r == 0:
                        ax.set_title(f"ch {ch}", fontsize=9)

                    # Frame labels on the left side of each row
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

            fig.tight_layout(rect=[0, 0, 1, 0.97])

            # Save into the multi-page PDF
            pdf.savefig(fig, bbox_inches="tight")

            plt.close(fig)


def run_bh_batch(
    fits_file: str | Path,
    frames: Iterable[int] | int | None,
    channels: Iterable[int] | int | None,
    *,
    cw: float,
    scale: float,
    dark_frame: int | None = None,
    time_range: tuple[float, float] = (0.0, 10.0),
    background_frames: Iterable[int] | tuple[int, ...] = (0, 1, 2, 3),
    band: tuple[float, float] = (433.0, 433.4),
    threshold_sigma: float = 5.0,
    fitter_kwargs: Mapping[str, Any] | None = None,
    bounds: tuple[Any, Any] | Mapping[str, Any] | None = None,
    out_dir: str | Path = "results",
    run_fit_limit: int | None = None,
    save_frames: bool = False,
):
    """Run BH batch fitting for a single VIS-1.33 m FITS file.

    Steps:
    - Load data with Vis133M.from_files
    - Apply CW calibration
    - Apply scale factor
    - Optionally apply dark subtraction using a reference frame
    - Enable baseline_zero
    - Set the time axis
    - Instantiate BHModel and BHFitter
    - Optionally update bounds
    - Run batch fitting and save results/figures under out_dir/<shot_id>/

    run_fit_limit: if set, run only the first N (frame, channel) fits after
        selection (for quick pipeline testing). Saves results, CSV, and plots
        for those fits only.
    save_frames: if True, save per-(frame, channel) fit plots in frames/
        (zero-padded filenames fNN_chNN.png). Default False.
    """
    fits_path = Path(fits_file)
    if not fits_path.is_file():
        raise FileNotFoundError(f"FITS file not found: {fits_path}")

    frames_list = None if frames is None else _normalize_indices(frames)
    channels_list = None if channels is None else _normalize_indices(channels)

    base_out = Path(out_dir)
    shot_id = fits_path.stem
    shot_dir = base_out / shot_id
    shot_dir.mkdir(parents=True, exist_ok=True)

    summary_path = shot_dir / "summary.csv"
    curves_path = shot_dir / "curves.pkl"
    grid_pdf_path = shot_dir / "grid.pdf"

    # Load VIS data
    vis = Vis133M.from_files(str(fits_path), scale=scale)

    # Apply CW calibration (regenerate wavelength axis)
    vis.apply_cw(cw_nm=float(cw))

    # Ensure scale matches the requested value
    vis.set_scale(float(scale))

    # Optional dark subtraction: interpret dark_frame as a frame index whose
    # image is subtracted from all frames.
    if dark_frame is not None:
        f_idx = int(dark_frame)
        F, C, P = vis.cube.shape
        if not (0 <= f_idx < F):
            raise IndexError(f"dark_frame {f_idx} out of range for F={F}")
        dark_img = vis.cube[f_idx]  # (C, P)
        vis.cube = vis.cube - dark_img[None, :, :]

    # Apply baseline_zero=True
    vis.set_baseline_zero(True)

    # Set time axis from time_range over the number of frames
    t_start, t_stop = map(float, time_range)
    vis.set_time_linspace(t_start, t_stop)

    # Step 1: background validation on the requested background frames
    check_background_flat(vis, background_frames)

    # Step 2: automatic signal detection if frames/channels not provided
    if frames_list is None or channels_list is None:
        auto_frames, auto_channels, _ = scan_signal_frames(
            vis,
            band=band,
            background_frames=background_frames,
            threshold_sigma=threshold_sigma,
        )
        if frames_list is None:
            frames_list = auto_frames
        if channels_list is None:
            channels_list = auto_channels

    if frames_list is None or channels_list is None:
        raise ValueError("Could not determine frames/channels for batch fitting.")

    print(f"Using frames {sorted(set(frames_list))}")
    print(f"Using channels {sorted(set(channels_list))}")

    # Model and fitter
    model = BHModel()
    fitter_kwargs = dict(fitter_kwargs or {})
    fitr = BHFitter(vis=vis, model=model, **fitter_kwargs)

    # Apply bounds if provided
    if bounds is not None:
        if isinstance(bounds, Mapping):
            lower = bounds.get("lower")
            upper = bounds.get("upper")
            fitr.set_bounds(lower=lower, upper=upper)
        else:
            lower, upper = bounds
            fitr.set_bounds(lower=lower, upper=upper)

    # Run batch fit with progress bars over frames/channels
    resb, curves = _batch_with_progress(
        fitr, frames_list, channels_list, run_fit_limit=run_fit_limit
    )

    if run_fit_limit is not None:
        print(f"Executed {len(resb)} fits (limited run).")

    # Save summary table
    resb.to_csv(summary_path, index=False)

    # Save curves as pickle
    with curves_path.open("wb") as f:
        pickle.dump(curves, f)

    # Save grid plots (normalized, structured by frames×channels, paged by channel)
    _plot_normalized_grid_pages(
        curves,
        frames_list,
        channels_list,
        grid_pdf_path=grid_pdf_path,
        channels_per_page=6,
    )

    if save_frames:
        frames_set = sorted(set(frames_list))
        channels_set = sorted(set(channels_list))
        per_fit_dir = shot_dir / "frames"
        per_fit_dir.mkdir(exist_ok=True)
        for f in frames_set:
            for ch in channels_set:
                key = (f, ch)
                if key not in curves:
                    continue
                x, y, yfit = curves[key]
                res_single = {"x": x, "y": y, "yfit": yfit}
                ax = fitr.plot_single(res_single, title=f"f{f} ch{ch}")
                fig_single = ax.figure
                out_path = per_fit_dir / f"f{f:02d}_ch{ch:02d}.png"
                fig_single.savefig(out_path, dpi=200)
                plt.close(fig_single)

    return resb, curves, shot_dir


def run_folder_batch(
    folder: str | Path,
    frames: Iterable[int] | int | None,
    channels: Iterable[int] | int | None,
    **kwargs,
):
    """Run BH batch fitting for all .fits files in a folder.

    Respects resume semantics: if ``summary.csv`` already exists for a shot,
    that FITS file is skipped.
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Folder not found: {folder_path}")

    base_out = Path(kwargs.get("out_dir", "results"))
    frames_list = None if frames is None else _normalize_indices(frames)
    channels_list = None if channels is None else _normalize_indices(channels)

    results: dict[str, Any] = {}

    for fits_path in sorted(folder_path.glob("*.fits")):
        shot_id = fits_path.stem
        shot_dir = base_out / shot_id
        summary_path = shot_dir / "summary.csv"
        if summary_path.is_file():
            # Resume: skip already processed shots
            continue

        resb, curves, _ = run_bh_batch(
            fits_path,
            frames_list,
            channels_list,
            **kwargs,
        )
        results[shot_id] = resb

    return results


__all__ = ["run_bh_batch", "run_folder_batch"]

