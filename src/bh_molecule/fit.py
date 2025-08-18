import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math


# Optional: your pretty formatter (drop this if you already have it imported)
def _format_fit_parameters(params, cov, names, units=None):
    units = units or [""] * len(names)
    errs = np.sqrt(np.diag(cov))
    out = []
    for n, v, e, u in zip(names, params, errs, units):
        if n.lower() == "t_rot":
            v = int(np.round(v, -2))
            e = int(np.round(e, -2))
            out.append((n, f"{v} ± {e} {u}".strip()))
        elif e < 1e-3:
            out.append((n, f"{v:.6f} ± {e:.1e} {u}".strip()))
        else:
            out.append((n, f"{v:.5g} ± {e:.2g} {u}".strip()))
    return pd.DataFrame(out, columns=["Parameter", "Formatted"])


def fit_single_channel_frame(
    vis,  # Vis133M instance
    model,  # physics.BHModel instance (configured with v00 wavelengths)
    frame: int,
    channel: int,
    nm_window=(433.05, 433.90),
    p0=(
        1.0,
        4000.0,
        0.01,
        0.025,
        0.0,
        1e-3,
        1e-3,
    ),  # (C, T_rot, dx, w_inst, base, I_R7, I_R8)
    bounds=([0, 0, 0, 0, -10, 0, 0], [10, 10000, 1, 0.1, 10, 1, 1]),
    maxfev=40000,
    return_fit=False,
):
    wl, spec = vis.spectrum(frame, channel)

    # crop to the requested wavelength window & keep finite values
    lo, hi = map(float, nm_window)
    m = (wl >= lo) & (wl <= hi) & np.isfinite(spec) & np.isfinite(wl)
    x = wl[m].astype(float)
    y = spec[m].astype(float)

    # ensure x is monotonic (some channels can be reversed)
    if x.size >= 2 and np.any(np.diff(x) < 0):
        idx = np.argsort(x)
        x, y = x[idx], y[idx]

    # wrap BHModel.full_fit_model into a curve_fit-compatible callable
    def _f(x, C, T_rot, dx, w_inst, base, I_R7, I_R8):
        return model.full_fit_model(x, C, T_rot, dx, w_inst, base, I_R7, I_R8)

    params, cov = curve_fit(_f, x, y, p0=p0, bounds=bounds, maxfev=maxfev)

    names = ["C", "T_rot", "dx", "w_inst", "base", "I_R7", "I_R8"]
    units = ["", "K", "nm", "nm", "", "", ""]
    summary = _format_fit_parameters(params, cov, names, units)

    result = {
        "params": params,
        "cov": cov,
        "errors": np.sqrt(np.diag(cov)),
        "summary": summary,
        "data_window": (x, y),
    }

    if return_fit:
        result["fit_y"] = _f(x, *params)

    return result


# --- Quick plotting helper ----------------------------------------------------


def plot_fit(res, title=None, xlim=(433.05, 433.90), ylim=None, ax=None):
    """
    res: dict returned by fit_single_channel_frame(..., return_fit=True)
    """
    x, y = res["data_window"]
    yfit = res["fit_y"]

    ax = ax or plt.subplots(figsize=(6, 5))[1]
    ax.scatter(x, y, s=6, label="data", zorder=2, color="k")
    ax.plot(x, yfit, lw=2.5, label="fit", zorder=1, color="#7397de")
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("intensity [arb]")
    if title:
        ax.set_title(title)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.minorticks_on()
    ax.legend()
    return ax


def batch_fit(
    vis,
    model,
    frames,
    channels,
    nm_window=(433.05, 433.90),
    p0=(1.0, 4000.0, 0.01, 0.025, 0.0, 1e-3, 1e-3),
    bounds=([0, 0, 0, 0, -10, 0, 0], [10, 10000, 1, 0.1, 10, 1, 1]),
    maxfev=40000,
    return_curves=False,
):
    """
    Returns:
      df: tidy DataFrame with params, errors, chi2, r2 per (frame, channel)
      curves (optional): dict[(frame, channel)] -> (x, y, yfit)
    """
    names = ["C", "T_rot", "dx", "w_inst", "base", "I_R7", "I_R8"]
    rows = []
    curves = {} if return_curves else None

    for f in frames:
        for ch in channels:
            try:
                r = fit_single_channel_frame(
                    vis,
                    model,
                    f,
                    ch,
                    nm_window=nm_window,
                    p0=p0,
                    bounds=bounds,
                    maxfev=maxfev,
                    return_fit=return_curves,
                )
                params = r["params"]
                errs = r["errors"]

                # simple metrics
                x, y = r["data_window"]
                yfit = r.get("fit_y", None)
                dof = max(len(y) - len(params), 1)
                chi2 = (
                    float(np.sum((y - yfit) ** 2)) / dof if yfit is not None else np.nan
                )
                ss_res = float(np.sum((y - yfit) ** 2)) if yfit is not None else np.nan
                ss_tot = (
                    float(np.sum((y - np.mean(y)) ** 2)) if yfit is not None else np.nan
                )
                r2 = (
                    1.0 - ss_res / ss_tot if yfit is not None and ss_tot > 0 else np.nan
                )

                row = {
                    "frame": f,
                    "channel": ch,
                    **{n: v for n, v in zip(names, params)},
                    **{f"{n}_err": e for n, e in zip(names, errs)},
                    "chi2_red": chi2,
                    "R2": r2,
                    "npts": len(y),
                }
                rows.append(row)

                if return_curves:
                    curves[(f, ch)] = (x, y, yfit)

            except Exception as e:
                rows.append({"frame": f, "channel": ch, "error": repr(e)})

    df = pd.DataFrame(rows).sort_values(["frame", "channel"]).reset_index(drop=True)
    return (df, curves) if return_curves else df


def plot_batch_grid(
    curves,
    frames=None,
    channels=None,
    ncols=3,
    xlim=(433.05, 433.90),
    ylim=None,
    suptitle=None,
):
    keys = sorted(
        [
            (f, ch)
            for (f, ch) in curves.keys()
            if (frames is None or f in frames) and (channels is None or ch in channels)
        ]
    )
    if not keys:
        raise ValueError("No matching (frame, channel) in curves.")

    n = len(keys)
    ncols = min(max(1, ncols), n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False
    )
    axs = axes.ravel()

    for ax, key in zip(axs, keys):
        x, y, yfit = curves[key]
        ax.scatter(x, y, s=5, label="data", zorder=2, color="k")
        ax.plot(x, yfit, lw=2, label="fit", zorder=1, color="#7397de")
        ax.set_title(f"f{key[0]} ch{key[1]}")
        ax.set_xlabel("wavelength [nm]")
        ax.set_ylabel("intensity [arb]")
        ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)
        ax.minorticks_on()
        ax.legend(fontsize=8)

    for ax in axs[n:]:
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, y=0.995)
    fig.tight_layout()
    return fig, axes


def plot_frame_overlay(
    curves, frame, channels=None, xlim=(433.05, 433.90), ylim=None, title=None
):
    fig, ax = plt.subplots(figsize=(7, 4))
    for (f, ch), (x, y, yfit) in sorted(curves.items()):
        if f != frame or (channels is not None and ch not in channels):
            continue
        ax.plot(x, yfit, lw=2, label=f"ch{ch}")
        ax.scatter(x, y, s=4, alpha=0.6)
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("intensity [arb]")
    ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.minorticks_on()
    ax.legend(ncols=2, fontsize=8)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax
