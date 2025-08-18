import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math


class BHFitter:
    def __init__(
        self,
        vis,
        model,
        nm_window=(433.05, 433.90),
        p0=(1.0, 4000.0, 0.01, 0.025, 0.0, 1e-3, 1e-3),
        bounds=([0, 0, 0, 0, -10, 0, 0], [10, 10000, 1, 0.1, 10, 1, 1]),
        maxfev=40000,
        weight="none",  # "none" | "poisson" | callable(x,y)->sigma
        warm_start=False,  # reuse last params across channels/frames
    ):
        self.vis = vis
        self.model = model
        self.nm_window = tuple(map(float, nm_window))
        self.p0 = np.asarray(p0, float)
        self.bounds = (np.asarray(bounds[0], float), np.asarray(bounds[1], float))
        self.maxfev = int(maxfev)
        self.weight = weight
        self.warm_start = bool(warm_start)
        self._last_params = None
        self.param_names = ["C", "T_rot", "dx", "w_inst", "base", "I_R7", "I_R8"]
        self.param_units = ["", "K", "nm", "nm", "", "", ""]

    def _sigma(self, x, y):
        if self.weight == "none":
            return None
        if self.weight == "poisson":
            return np.sqrt(np.clip(y, 1e-12, None))
        if callable(self.weight):
            return np.asarray(self.weight(x, y), float)
        return None

    def _window_data(self, frame, channel):
        wl, spec = self.vis.spectrum(frame, channel)
        lo, hi = self.nm_window
        m = (wl >= lo) & (wl <= hi) & np.isfinite(wl) & np.isfinite(spec)
        x = wl[m].astype(float)
        y = spec[m].astype(float)
        if x.size >= 2 and np.any(np.diff(x) < 0):
            idx = np.argsort(x)
            x, y = x[idx], y[idx]
        return x, y

    def _f(self, x, C, T_rot, dx, w_inst, base, I_R7, I_R8):
        return self.model.full_fit_model(x, C, T_rot, dx, w_inst, base, I_R7, I_R8)

    # MARK: Fit
    def fit(self, frame, channel, return_fit=True, p0=None):
        x, y = self._window_data(frame, channel)
        if p0 is None:
            p0 = (
                self._last_params
                if (self.warm_start and self._last_params is not None)
                else self.p0
            )
        sigma = self._sigma(x, y)
        params, cov = curve_fit(
            self._f,
            x,
            y,
            p0=p0,
            bounds=self.bounds,
            maxfev=self.maxfev,
            sigma=sigma,
            absolute_sigma=bool(sigma is not None),
        )
        self._last_params = params.copy()
        res = {
            "frame": frame,
            "channel": channel,
            "params": params,
            "cov": cov,
            "errors": np.sqrt(np.diag(cov)),
            "summary": self._format_table(
                params, cov, self.param_names, self.param_units
            ),
            "x": x,
            "y": y,
        }
        if return_fit:
            res["yfit"] = self._f(x, *params)
        return res

    # MARK: Batch fit
    def batch(self, frames, channels, return_curves=False):
        rows, curves = [], {} if return_curves else None
        for f in frames:
            for ch in channels:
                try:
                    r = self.fit(f, ch, return_fit=return_curves)
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
                        **{n: v for n, v in zip(self.param_names, params)},
                        **{f"{n}_err": e for n, e in zip(self.param_names, errs)},
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

    # MARK: Plotting
    def plot_single(self, res, xlim=None, ylim=None, title=None, ax=None):
        x, y = res["x"], res["y"]
        yfit = res.get("yfit", None)
        ax = ax or plt.subplots(figsize=(6, 5))[1]
        ax.scatter(x, y, s=6, label="data", zorder=2, color="k")
        if yfit is not None:
            ax.plot(x, yfit, lw=2.5, label="fit", zorder=1, color="#7397de")
        ax.set_xlabel("wavelength [nm]")
        ax.set_ylabel("intensity [arb]")
        ax.set_xlim(*(xlim or self.nm_window))
        if ylim:
            ax.set_ylim(*ylim)
        if title:
            ax.set_title(title)
        ax.minorticks_on()
        ax.legend()
        ax.figure.tight_layout()
        return ax

    def plot_grid(
        self,
        curves,
        frames=None,
        channels=None,
        ncols=3,
        xlim=None,
        ylim=None,
        suptitle=None,
    ):

        keys = sorted(
            [
                (f, ch)
                for (f, ch) in curves.keys()
                if (frames is None or f in frames)
                and (channels is None or ch in channels)
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
            ax.set_xlim(*(xlim or self.nm_window))
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
        self,
        curves,
        frame,
        channels=None,
        xlim=None,
        ylim=None,
        title=None,
        *,
        cmap: str = "tab10",
        line_width: float = 1.0,
        line_alpha: float = 0.95,
        scatter_size: float = 8.0,
        scatter_alpha: float = 0.35,
        legend_cols: int | None = None,
    ):
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
        ax.set_xlim(*(xlim or self.nm_window))
        if ylim:
            ax.set_ylim(*ylim)
        ax.minorticks_on()
        ncols = legend_cols if legend_cols is not None else min(4, max(1, n))
        ax.legend(ncols=ncols, fontsize=8)
        if title:
            ax.set_title(title)
        fig.tight_layout()
        return fig, ax

    # MARK: Formatting
    @staticmethod
    def _format_table(params, cov, names, units, min_step_map=None):
        errs_raw = np.sqrt(np.diag(cov))
        min_step_map = min_step_map or {"T_rot": 100}
        rows = []

        for n, v_raw, e_raw, u in zip(names, params, errs_raw, units):
            if n.lower() == "t_rot":
                step = min_step_map["T_rot"]
                v = int(np.round(v_raw / step) * step)
                e = int(np.round(e_raw / step) * step)
                disp = f"{v} ± {e} {u}".strip()
            else:
                if not np.isfinite(e_raw) or e_raw <= 0:
                    disp = f"{v_raw:g} {u}".strip()
                else:
                    e_abs = float(abs(e_raw))
                    exp = int(np.floor(np.log10(e_abs)))
                    sig = 2 if e_abs / 10**exp < 2.5 else 1
                    e_rounded = float(f"{e_abs:.{sig}g}")

                    decimals = -int(np.floor(np.log10(e_rounded)))
                    v_rounded = round(v_raw, decimals)

                    if abs(v_rounded) < 1e-2 or abs(v_rounded) >= 1e3:
                        scale = 10.0**exp
                        v_scaled = v_rounded / scale
                        e_scaled = e_rounded / scale
                        e_scaled_str = f"{e_scaled:.{sig}g}"
                        d = (
                            len(e_scaled_str.split(".")[1])
                            if "." in e_scaled_str
                            else 0
                        )
                        v_scaled = round(v_scaled, d)
                        e_scaled = round(e_scaled, d)
                        disp = f"({v_scaled:.{d}f} ± {e_scaled:.{d}f})e{exp:+d} {u}".strip()
                    else:
                        disp = f"{v_rounded:.{decimals}f} ± {e_rounded:.{decimals}f} {u}".strip()

            rows.append({"Parameter": n, "Value": disp})

        return pd.DataFrame(rows, columns=["Parameter", "Value"])
