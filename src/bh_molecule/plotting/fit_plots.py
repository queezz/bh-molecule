import math


def plot_single(res, *, nm_window, xlim=None, ylim=None, title=None, ax=None):
    """Plot a single fit result (data and optional model curve).

    Parameters
    ----------
    res : dict
        Result dictionary as returned by ``BHFitter.fit`` (must contain ``x`` and
        ``y``; may contain ``yfit``).
    nm_window : tuple
        Default x-axis window in nanometres when ``xlim`` is not provided.
    xlim : tuple, optional
        X-axis limits as ``(xmin, xmax)``. Defaults to ``nm_window``.
    ylim : tuple, optional
        Y-axis limits as ``(ymin, ymax)``.
    title : str, optional
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Axis to draw into. If ``None`` a new figure and axis are created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis containing the plot.
    """
    import matplotlib.pyplot as plt

    x, y = res["x"], res["y"]
    yfit = res.get("yfit", None)
    ax = ax or plt.subplots(figsize=(6, 5))[1]
    ax.plot(x, y, ".", ms=3, linestyle="none", label="data", zorder=2, color="k")
    if yfit is not None:
        ax.plot(x, yfit, "-", lw=1.2, label="fit", zorder=1, color="#7397de")
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("intensity [arb]")
    ax.set_xlim(*(xlim or nm_window))
    if ylim:
        ax.set_ylim(*ylim)
    if title:
        ax.set_title(title)
    ax.minorticks_on()
    ax.legend()
    ax.figure.tight_layout()
    return ax


def plot_grid(
    curves,
    *,
    nm_window,
    frames=None,
    channels=None,
    ncols=3,
    xlim=None,
    ylim=None,
    suptitle=None,
):
    """Plot a grid of fit curves.

    Parameters
    ----------
    curves : dict
        Mapping from ``(frame, channel)`` to ``(x, y, yfit)`` tuples.
    nm_window : tuple
        Default x-axis window in nanometres when ``xlim`` is not provided.
    frames : iterable, optional
        If provided, only include these frames.
    channels : iterable, optional
        If provided, only include these channels.
    ncols : int, optional
        Preferred number of columns in the plot grid.
    xlim, ylim : tuple, optional
        Axis limits forwarded to each subplot.
    suptitle : str, optional
        Optional figure-level title.

    Returns
    -------
    fig, axes : tuple
        Matplotlib figure and axes array for the created grid.
    """
    import matplotlib.pyplot as plt

    keys = sorted(
        [
            (f, ch)
            for (f, ch) in curves.keys()
            if (frames is None or f in frames) and (channels is None or ch in channels)
        ]
    )
    if not keys:
        raise ValueError("No matching curves.")
    n = len(keys)
    ncols = min(max(1, ncols), n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False
    )
    axs = axes.ravel()
    for ax, key in zip(axs, keys):
        x, y, yfit = curves[key]
        ax.scatter(x, y, s=5, zorder=2, label="data", color="k")
        ax.plot(x, yfit, lw=2, zorder=1, label="fit", color="#7397de")
        ax.set_title(f"f{key[0]} ch{key[1]}")
        ax.set_xlabel("wavelength [nm]")
        ax.set_ylabel("intensity [arb]")
        ax.set_xlim(*(xlim or nm_window))
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


def plot_overlay(
    curves,
    *,
    nm_window,
    frame,
    channels=None,
    xlim=None,
    ylim=None,
    title=None,
    cmap: str = "tab10",
    line_width: float = 1.0,
    line_alpha: float = 0.95,
    scatter_size: float = 8.0,
    scatter_alpha: float = 0.35,
    legend_cols: int | None = None,
):
    """Overlay multiple channels for a single frame.

    Parameters
    ----------
    curves : dict
        Mapping from ``(frame, channel)`` to ``(x, y, yfit)`` tuples.
    nm_window : tuple
        Default x-axis window in nanometres when ``xlim`` is not provided.
    frame : int
        Frame index to draw.
    channels : iterable, optional
        If provided, only include these channels.
    xlim, ylim : tuple, optional
        Axis limits.
    title : str, optional
        Axis title.
    cmap : str, optional
        Matplotlib colormap name used to assign channel colors.
    line_width : float, optional
        Width of the model lines.
    line_alpha : float, optional
        Transparency for model lines.
    scatter_size : float, optional
        Size for scatter points.
    scatter_alpha : float, optional
        Alpha for scatter points.
    legend_cols : int or None, optional
        Number of columns for the legend. If ``None`` a sensible default is chosen.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axis containing the overlay plot.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))

    # Determine which channels to draw and assign distinct colors
    keys = [
        (f, ch)
        for (f, ch) in curves.keys()
        if f == frame and (channels is None or ch in channels)
    ]
    ch_list = sorted({ch for (_, ch) in keys})
    n = max(len(ch_list), 1)
    cm = plt.cm.get_cmap(cmap, max(n, 10))
    color_for = {ch: cm(i % cm.N) for i, ch in enumerate(ch_list)}

    for (f, ch), (x, y, yfit) in sorted(curves.items()):
        if f != frame or (channels is not None and ch not in channels):
            continue
        color = color_for.get(ch, "#555555")
        if yfit is not None:
            ax.plot(
                x,
                yfit,
                lw=line_width,
                alpha=line_alpha,
                color=color,
                label=f"ch{ch}",
                zorder=3,
            )
        ax.scatter(
            x,
            y,
            s=scatter_size,
            alpha=scatter_alpha,
            color=color,
            edgecolors="none",
            zorder=2,
        )

    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("intensity [arb]")
    ax.set_xlim(*(xlim or nm_window))
    if ylim:
        ax.set_ylim(*ylim)
    ax.minorticks_on()
    ncols = legend_cols if legend_cols is not None else min(4, max(1, n))
    ax.legend(ncols=ncols, fontsize=8)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax


__all__ = ["plot_single", "plot_grid", "plot_overlay"]
