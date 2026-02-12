"""Wavelength calibration linearity analysis and diagnostics."""

import math
import numpy as np


def analyze_wavelength_linearity(m: np.ndarray) -> dict:
    """Analyze wavelength linearity per fiber.

    Parameters
    ----------
    m : ndarray, shape (n_fibers, n_pixels)
        Wavelength calibration matrix.

    Returns
    -------
    result : dict with keys:
        slopes      : (n_fibers,) ndarray
        intercepts  : (n_fibers,) ndarray
        r2          : (n_fibers,) ndarray
        rmse        : (n_fibers,) ndarray
    """
    m = np.asarray(m, dtype=float)
    n_fibers, n_pixels = m.shape
    x = np.arange(n_pixels, dtype=float)

    slopes = np.full(n_fibers, np.nan)
    intercepts = np.full(n_fibers, np.nan)
    r2 = np.full(n_fibers, np.nan)
    rmse = np.full(n_fibers, np.nan)

    for i in range(n_fibers):
        y = m[i]
        mask = np.isfinite(y)
        if mask.sum() < 2:
            continue
        p = np.polyfit(x[mask], y[mask], 1)
        yfit = np.polyval(p, x[mask])
        ss_res = np.sum((y[mask] - yfit) ** 2)
        ss_tot = np.sum((y[mask] - np.mean(y[mask])) ** 2)
        r2[i] = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan
        rmse[i] = np.sqrt(ss_res / mask.sum())
        slopes[i] = float(p[0])
        intercepts[i] = float(p[1])

    return {
        "slopes": slopes,
        "intercepts": intercepts,
        "r2": r2,
        "rmse": rmse,
    }


def linear_fit_residuals(
    m: np.ndarray,
    slopes: np.ndarray,
    intercepts: np.ndarray,
) -> np.ndarray:
    """Return residual array Δλ = λ - (a*x + b).

    Shape: same as m (n_fibers, n_pixels).
    """
    m = np.asarray(m, dtype=float)
    n_fibers, n_pixels = m.shape
    x = np.arange(n_pixels, dtype=float)
    residuals = np.full_like(m, np.nan)
    for i in range(n_fibers):
        if np.isfinite(slopes[i]) and np.isfinite(intercepts[i]):
            fit = slopes[i] * x + intercepts[i]
            mask = np.isfinite(m[i])
            residuals[i, mask] = m[i, mask] - fit[mask]
    return residuals


def plot_linearity_summary(
    slopes: np.ndarray,
    r2: np.ndarray,
    rmse: np.ndarray,
):
    """Summary plots: slope vs fiber, RMSE distribution."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    n = len(slopes)

    ax = axes[0]
    ax.plot(np.arange(n), slopes, "o-", markersize=3)
    ax.set_xlabel("fiber")
    ax.set_ylabel("slope")
    ax.set_title("Slope per fiber")

    ax = axes[1]
    valid = rmse[np.isfinite(rmse)]
    if valid.size > 0:
        ax.hist(valid, bins=min(20, max(1, valid.size // 2)))
    ax.set_xlabel("RMSE")
    ax.set_title("RMSE distribution")
    fig.tight_layout()
    return fig, axes


def plot_fiber_line_fits(
    m: np.ndarray,
    slopes: np.ndarray,
    intercepts: np.ndarray,
    fiber_indices: list[int] | None = None,
    ncols: int = 4,
):
    """Plot data + linear fit for selected fibers in a grid.

    If fiber_indices is None, select worst RMSE fibers.
    """
    import matplotlib.pyplot as plt

    m = np.asarray(m, dtype=float)
    n_fibers, n_pixels = m.shape
    x = np.arange(n_pixels, dtype=float)

    if fiber_indices is None:
        rmse = np.full(n_fibers, np.nan)
        for i in range(n_fibers):
            if np.isfinite(slopes[i]):
                y = m[i]
                mask = np.isfinite(y)
                yfit = slopes[i] * x[mask] + intercepts[i]
                rmse[i] = np.sqrt(np.mean((y[mask] - yfit) ** 2))
        n_show = min(ncols * 2, n_fibers)
        worst = np.argsort(np.where(np.isfinite(rmse), rmse, -np.inf))[::-1][:n_show]
        fiber_indices = [int(i) for i in worst]

    n = len(fiber_indices)
    ncols = min(max(1, ncols), n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 3.2 * nrows), squeeze=False
    )
    axs = axes.ravel()

    for ax, i in zip(axs, fiber_indices):
        y = m[i].astype(float)
        mask = np.isfinite(y)
        ax.plot(x[mask], y[mask], ".", label="data", markersize=4)
        if np.isfinite(slopes[i]):
            yfit = slopes[i] * x[mask] + intercepts[i]
            ax.plot(x[mask], yfit, "-", label="fit")
            rmse_val = np.sqrt(np.mean((y[mask] - yfit) ** 2))
        else:
            rmse_val = np.nan
        ax.set_title(f"fiber {i} | RMSE={rmse_val:.3g}")
        ax.set_xlabel("pixel")
        ax.legend(fontsize=8)
        ax.minorticks_on()

    for ax in axs[n:]:
        ax.axis("off")

    fig.tight_layout()
    return fig, axes
