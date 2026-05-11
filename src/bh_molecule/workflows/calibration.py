"""Lightweight helpers for *manual* fit-constraint calibration.

These helpers summarize and visualize the *distribution* of fitted
parameters (``w_inst``, ``dx``, ``base``, ...) across a small set of
representative spectra so a human can pick production constraints by
inspection.  They intentionally **do not** auto-pick bounds: the goal is
explicit, inspectable, reproducible fitting constraints (no per-spectrum
auto-bounding, no hidden heuristics).

Workflow
--------

1. Run :func:`bh_molecule.workflows.batch_fit.run_bh_batch` (or
   :class:`bh_molecule.fit.BHFitter`) with **loose** bounds on a few
   representative shots / frames / channels.
2. Pass the resulting summary ``DataFrame`` to
   :func:`summarize_fit_distribution` and inspect the percentile table.
3. Use :func:`plot_fit_distributions` to visualize histograms vs. proposed
   bounds (overlaid as vertical lines) before adopting them.
4. Adopt bounds explicitly via ``w_inst_default`` / ``w_inst_bounds`` /
   ``fix_w_inst`` / ``dx_tol_nm`` / ``base_bound`` in the YAML or Python
   API.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


DEFAULT_CALIBRATION_PARAMS: tuple[str, ...] = ("w_inst", "dx", "base")


def summarize_fit_distribution(
    df: pd.DataFrame,
    params: Sequence[str] = DEFAULT_CALIBRATION_PARAMS,
    *,
    r2_min: float | None = None,
    extra_quality_filter: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Return per-parameter summary statistics across fits in ``df``.

    Statistics: ``n``, ``median``, ``mad`` (median absolute deviation),
    ``p05``, ``p16``, ``p50``, ``p84``, ``p95``, ``min``, ``max``.

    Parameters
    ----------
    df : DataFrame
        Output of a batch run, e.g. ``run_bh_batch(...)[0]`` or
        ``BHFitter.batch(...)``.  Must contain a column per parameter name.
    params : sequence of str
        Parameter columns to summarize.
    r2_min : float or None
        Optionally drop rows with ``R2 < r2_min`` before computing stats.
        ``None`` keeps every row (useful when the column is absent).
    extra_quality_filter : pandas.Series or None
        Optional boolean mask aligned with ``df.index``; only ``True``
        rows are used.

    Returns
    -------
    DataFrame
        Index = parameter name, columns = the statistics listed above.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    work = df.copy()
    if r2_min is not None and "R2" in work.columns:
        work = work.loc[work["R2"] >= float(r2_min)]
    if extra_quality_filter is not None:
        mask = extra_quality_filter.reindex(work.index).fillna(False)
        work = work.loc[mask]

    rows = []
    for name in params:
        if name not in work.columns:
            rows.append({"param": name, "n": 0, "median": np.nan, "mad": np.nan,
                         "p05": np.nan, "p16": np.nan, "p50": np.nan,
                         "p84": np.nan, "p95": np.nan, "min": np.nan, "max": np.nan})
            continue
        vals = work[name].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            rows.append({"param": name, "n": 0, "median": np.nan, "mad": np.nan,
                         "p05": np.nan, "p16": np.nan, "p50": np.nan,
                         "p84": np.nan, "p95": np.nan, "min": np.nan, "max": np.nan})
            continue
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        rows.append(
            {
                "param": name,
                "n": int(vals.size),
                "median": med,
                "mad": mad,
                "p05": float(np.percentile(vals, 5)),
                "p16": float(np.percentile(vals, 16)),
                "p50": med,
                "p84": float(np.percentile(vals, 84)),
                "p95": float(np.percentile(vals, 95)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        )
    return pd.DataFrame(rows).set_index("param")


def plot_fit_distributions(
    df: pd.DataFrame,
    params: Sequence[str] = DEFAULT_CALIBRATION_PARAMS,
    *,
    suggested_bounds: Optional[Mapping[str, Iterable[float]]] = None,
    bins: int = 30,
    r2_min: float | None = None,
):
    """Histogram one or more fit parameters with optional overlaid bounds.

    Parameters
    ----------
    df : DataFrame
        Batch fit summary table.
    params : sequence of str
        Columns to plot.  One subplot per parameter.
    suggested_bounds : mapping or None
        Map ``param_name -> (lo, hi)`` to overlay as vertical lines on
        the histogram — useful when iterating toward production bounds.
    bins : int
        Histogram bin count.
    r2_min : float or None
        Optional ``R2`` quality cut (passed to
        :func:`summarize_fit_distribution`'s convention).

    Returns
    -------
    fig, axes : matplotlib Figure and list of Axes
    """
    import matplotlib.pyplot as plt

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    work = df
    if r2_min is not None and "R2" in work.columns:
        work = work.loc[work["R2"] >= float(r2_min)]

    n = len(params)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.2), squeeze=False)
    axes = list(axes[0])
    suggested_bounds = suggested_bounds or {}

    for ax, name in zip(axes, params):
        if name not in work.columns:
            ax.set_title(f"{name} (missing)")
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        vals = work[name].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            ax.set_title(f"{name} (no finite values)")
            continue

        ax.hist(vals, bins=bins, color="#4a6fa5", alpha=0.85, edgecolor="white")
        med = float(np.median(vals))
        ax.axvline(med, color="k", lw=1.2, ls="-", label=f"median={med:.4g}")

        bb = suggested_bounds.get(name)
        if bb is not None:
            lo, hi = float(bb[0]), float(bb[1])
            ax.axvline(lo, color="#d8743f", lw=1.4, ls="--", label=f"lo={lo:.4g}")
            ax.axvline(hi, color="#d8743f", lw=1.4, ls="--", label=f"hi={hi:.4g}")

        ax.set_title(name)
        ax.set_xlabel(name)
        ax.set_ylabel("count")
        ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    return fig, axes


__all__ = [
    "DEFAULT_CALIBRATION_PARAMS",
    "summarize_fit_distribution",
    "plot_fit_distributions",
]
