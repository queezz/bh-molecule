from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Any
import pickle

import matplotlib.pyplot as plt

from bh_molecule.instruments.vis133m import Vis133M
from bh_molecule.physics import BHModel
from bh_molecule.fit import BHFitter
from .signal_scan import check_background_flat, scan_signal_frames


def _normalize_indices(values: Iterable[int] | int) -> list[int]:
    """Return a list of indices given an int or iterable of ints."""
    if isinstance(values, int):
        return [values]
    return [int(v) for v in values]


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
    grid_png_path = shot_dir / "grid.png"
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

    # Run batch fit
    resb, curves = fitr.batch(frames_list, channels_list, return_curves=True)

    # Save summary table
    resb.to_csv(summary_path, index=False)

    # Save curves as pickle
    with curves_path.open("wb") as f:
        pickle.dump(curves, f)

    # Save grid plots
    fig, _ = fitr.plot_grid(curves, frames=frames_list, channels=channels_list)
    fig.savefig(grid_png_path, dpi=200)
    fig.savefig(grid_pdf_path)
    plt.close(fig)

    # Save individual fit figures
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
            out_path = per_fit_dir / f"f{f}_ch{ch}.png"
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

