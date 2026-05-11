from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Any
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bh_molecule.instruments.vis133m import Vis133M
from bh_molecule.physics import BHModel
from bh_molecule.fit import BHFitter
from bh_molecule.plotting.batch_fit_plots import save_batch_fit_grid
from .signal_scan import check_background_flat, scan_signal_frames


def prepare_vis_for_bh_batch(
    fits_file: str | Path,
    *,
    cw: float,
    scale: float,
    dark_frame: int | None = None,
    time_range: tuple[float, float] = (0.0, 10.0),
) -> Vis133M:
    """Load and calibrate VIS data exactly as :func:`run_bh_batch` does before fitting.

    Steps: ``Vis133M.from_files`` → :meth:`~bh_molecule.instruments.vis133m.Vis133M.apply_cw`
    → :meth:`~bh_molecule.instruments.vis133m.Vis133M.set_scale` → optional
    ``dark_frame`` subtraction from the cube →
    :meth:`~bh_molecule.instruments.vis133m.Vis133M.set_baseline_zero` (``True``) →
    :meth:`~bh_molecule.instruments.vis133m.Vis133M.set_time_linspace`.

    Notes
    -----
    Frame and channel indices are **0-based** everywhere (FITS time slice 0 is
    the first frame).

    The ``background_frames`` argument to :func:`run_bh_batch` is **not** used
    here. Those indices only feed :func:`~bh_molecule.workflows.signal_scan.check_background_flat`
    and :func:`~bh_molecule.workflows.signal_scan.scan_signal_frames` (noise level
    for thresholding in the band image). They are **not** subtracted from
    per-channel spectra; the only default spectral preprocessing is
    *per-row minimum subtraction* when baseline-zero mode is on (see
    :meth:`~bh_molecule.instruments.vis133m.Vis133M.spectrum`).
    """
    fits_path = Path(fits_file)
    if not fits_path.is_file():
        raise FileNotFoundError(f"FITS file not found: {fits_path}")

    vis = Vis133M.from_files(str(fits_path), scale=scale)
    vis.apply_cw(cw_nm=float(cw))
    vis.set_scale(float(scale))

    if dark_frame is not None:
        f_idx = int(dark_frame)
        F, C, P = vis.cube.shape
        if not (0 <= f_idx < F):
            raise IndexError(f"dark_frame {f_idx} out of range for F={F}")
        dark_img = vis.cube[f_idx]
        vis.cube = vis.cube - dark_img[None, :, :]

    vis.set_baseline_zero(True)
    t_start, t_stop = map(float, time_range)
    vis.set_time_linspace(t_start, t_stop)
    return vis


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

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["frame", "channel"]).reset_index(drop=True)
    return df, curves


def frame_plot_filename(frame: int, channel: int) -> str:
    """Return the canonical per-fit PNG filename for ``(frame, channel)``.

    Filenames are zero-padded to two digits to keep ``ls`` ordering sensible
    (e.g. ``f06_ch08.png``).  This helper exists so tests and external
    scripts can rely on the same convention.
    """
    return f"f{int(frame):02d}_ch{int(channel):02d}.png"


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
    save_frames: bool = True,
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
    save_frames: if True (default), save per-(frame, channel) fit plots in
        ``<shot_dir>/frames/`` using zero-padded filenames produced by
        :func:`frame_plot_filename` (e.g. ``f06_ch08.png``). Set to False
        from the YAML config or Python API to skip per-fit PNG generation
        when only the summary CSV / grid plots are needed.
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
    print(f"[{shot_id}] starting batch fit (save_frames={save_frames})")

    summary_path = shot_dir / "summary.csv"
    curves_path = shot_dir / "curves.pkl"
    grid_pdf_path = shot_dir / "grid.pdf"

    bg_tuple = tuple(_normalize_indices(background_frames))
    print(
        f"[{shot_id}] background_frames={bg_tuple} "
        "(flat-check + signal detection; not subtracted from spectra)"
    )

    vis = prepare_vis_for_bh_batch(
        fits_path,
        cw=cw,
        scale=scale,
        dark_frame=dark_frame,
        time_range=time_range,
    )

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

    if not frames_list or not channels_list:
        print(f"WARNING: No BH signal detected in {fits_path.name} (frames={frames_list}, channels={channels_list})")

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
    save_batch_fit_grid(
        curves,
        frames_list,
        channels_list,
        pdf_path=grid_pdf_path,
        channels_per_page=6,
    )

    if save_frames:
        frames_set = sorted(set(frames_list))
        channels_set = sorted(set(channels_list))
        per_fit_dir = shot_dir / "frames"
        per_fit_dir.mkdir(exist_ok=True)
        png_pairs = [
            (f, ch)
            for f in frames_set
            for ch in channels_set
            if (f, ch) in curves
        ]
        n_saved = 0
        for f, ch in _iter_with_tqdm(png_pairs, desc=f"[{shot_id}] PNGs"):
            x, y, yfit = curves[(f, ch)]
            res_single = {"x": x, "y": y, "yfit": yfit}
            out_path = per_fit_dir / frame_plot_filename(f, ch)
            try:
                ax = fitr.plot_single(res_single, title=f"f{f} ch{ch}")
                fig_single = ax.figure
                fig_single.savefig(out_path, dpi=200)
                plt.close(fig_single)
                n_saved += 1
            except Exception as exc:  # pragma: no cover - defensive
                print(f"Failed to save per-fit plot {out_path.name}: {exc!r}")
                plt.close("all")
        print(f"[{shot_id}] saved {n_saved} per-fit PNGs to {per_fit_dir}")

    print(f"[{shot_id}] saved results to {shot_dir}")
    return resb, curves, shot_dir


def run_folder_batch(
    folder: str | Path,
    frames: Iterable[int] | int | None,
    channels: Iterable[int] | int | None,
    *,
    shots: Iterable[int | str] | None = None,
    **kwargs,
):
    """Run BH batch fitting for ``.fits`` files in a folder.

    Parameters
    ----------
    folder : path-like
        Directory containing the FITS files.
    frames, channels : int, iterable, or None
        Forwarded to :func:`run_bh_batch` (None enables auto signal detection).
    shots : iterable of int or str, optional
        If given, only process FITS files whose stem (e.g. ``"193788"``) is
        in this list. Useful for running a small subset of a large folder.
        IDs are coerced to ``str``; missing matches print a warning.
    **kwargs
        Forwarded to :func:`run_bh_batch` (``cw``, ``scale``, ``out_dir``,
        ``save_frames``, ``run_fit_limit``, ...).

    Notes
    -----
    Resume semantics: shots whose ``summary.csv`` already exists under
    ``out_dir/<shot_id>/`` are skipped.
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Folder not found: {folder_path}")

    base_out = Path(kwargs.get("out_dir", "results"))
    frames_list = None if frames is None else _normalize_indices(frames)
    channels_list = None if channels is None else _normalize_indices(channels)

    all_fits = sorted(folder_path.glob("*.fits"))
    if shots is not None:
        wanted = {str(s) for s in shots}
        selected = [p for p in all_fits if p.stem in wanted]
        missing = wanted - {p.stem for p in selected}
        if missing:
            print(
                f"WARNING: shots {sorted(missing)} not found under {folder_path}"
            )
        fits_paths = selected
    else:
        fits_paths = all_fits

    print(f"Processing {len(fits_paths)} shot(s) under {folder_path}")

    results: dict[str, Any] = {}

    n = len(fits_paths)
    for i, fits_path in enumerate(fits_paths, start=1):
        shot_id = fits_path.stem
        shot_dir = base_out / shot_id
        summary_path = shot_dir / "summary.csv"
        if summary_path.is_file():
            print(f"[{shot_id}] ({i}/{n}) skipping (summary.csv already exists)")
            continue

        print(f"--- [{shot_id}] ({i}/{n}) ---")
        resb, curves, _ = run_bh_batch(
            fits_path,
            frames_list,
            channels_list,
            **kwargs,
        )
        results[shot_id] = resb

    return results


__all__ = [
    "run_bh_batch",
    "run_folder_batch",
    "frame_plot_filename",
    "prepare_vis_for_bh_batch",
]

