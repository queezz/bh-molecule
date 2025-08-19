# src/bh_molecule/plot_theme.py
from contextlib import contextmanager
import matplotlib.pyplot as plt
import matplotlib as mpl

_DEF_BG = "#44423e"
_DEF_FG = "0.92"


def use_dark(bg=_DEF_BG, fg=_DEF_FG):
    plt.style.use("dark_background")
    mpl.rcParams.update(
        {
            "figure.facecolor": bg,
            "axes.facecolor": bg,
            "savefig.facecolor": bg,
            "text.color": fg,
            "axes.labelcolor": fg,
            "axes.titlecolor": fg,
            "xtick.color": fg,
            "ytick.color": fg,
            "axes.edgecolor": fg,
            "grid.color": "0.35",
            "legend.facecolor": bg,
            "legend.edgecolor": "0.7",
        }
    )


def reset_light():
    plt.style.use("default")
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.titlecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.edgecolor": "black",
        }
    )


@contextmanager
def dark_theme(bg=_DEF_BG, fg=_DEF_FG):
    old = mpl.rcParams.copy()
    use_dark(bg=bg, fg=fg)
    try:
        yield
    finally:
        mpl.rcParams.update(old)
