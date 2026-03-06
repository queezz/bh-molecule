import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


class BHFitter:
    """Blackbody-Hydrogen molecular spectrum fitter.

    This class wraps a model object providing `full_fit_model` and a
    `vis` data source exposing `spectrum(frame, channel)` to perform
    parameter estimation over a wavelength window and to visualize
    the resulting fits.

    Parameters
    ----------
    vis : object
        Data source exposing ``spectrum(frame, channel) -> (wl, spec)``.
    model : object
        Model providing ``full_fit_model(x, C, T_rot, dx, w_inst, base, I_R7, I_R8)``.
    nm_window : tuple, optional
        Wavelength window `(lo, hi)` in nanometers to select data for
        fitting. Default is ``(433.05, 433.90)``.
    p0 : tuple or array-like, optional
        Initial guess for fit parameters. Default is
        ``(1.0, 4000.0, 0.01, 0.025, 0.0, 1e-3, 1e-3)``.
    bounds : tuple, optional
        Lower and upper bounds for parameters. Should be ``(lower, upper)``
        where each is array-like. Defaults to
        ``([0, 0, 0, 0, -10, 0, 0], [10, 10000, 1, 0.1, 10, 1, 1])``.
    maxfev : int, optional
        Maximum function evaluations passed to the underlying optimizer.
    weight : {'none', 'poisson'} or callable, optional
        Weighting scheme for the fit. If ``'none'`` (default) no weighting
        is used. If ``'poisson'``, sigma is taken as ``sqrt(y)``. If a
        callable is provided it must have signature ``weight(x, y) -> sigma``.
    warm_start : bool, optional
        If True reuse the last found parameters as initial guess for the
        next fit. Defaults to False.

    Attributes
    ----------
    param_names : list of str
        Human-readable parameter names used in result tables.
    param_units : list of str
        Units corresponding to `param_names`.
    """

    def __init__(
        self,
        vis,
        model,
        nm_window=(433.05, 433.90),
        # Parameter order for p0 / bounds:
        # [C, T_rot (K), dx (nm), w_inst (nm), base, I_R7, I_R8]
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
        """Compute per-point uncertainties (sigma) for weighting the fit.

        The behaviour depends on the ``weight`` attribute of the fitter:
        - ``'none'`` -> no weighting (returns ``None``).
        - ``'poisson'`` -> Poisson-like sigma: ``sqrt(y)`` (clipped to avoid
          negative/zero values).
        - callable -> calls ``weight(x, y)`` and converts the result to an
          ndarray.

        Parameters
        ----------
        x : array_like
            Independent variable values (wavelengths).
        y : array_like
            Observed dependent variable values (intensities).

        Returns
        -------
        sigma : ndarray or None
            Per-point uncertainties suitable to pass to
            ``scipy.optimize.curve_fit`` via the ``sigma`` argument, or
            ``None`` when no weighting is used.
        """
        if self.weight == "none":
            return None
        if self.weight == "poisson":
            return np.sqrt(np.clip(y, 1e-12, None))
        if callable(self.weight):
            return np.asarray(self.weight(x, y), float)
        return None

    def _window_data(self, frame, channel):
        """Retrieve and window the spectrum from the `vis` data source.

        Pulls the wavelength and spectrum arrays for a given ``frame`` and
        ``channel`` via ``self.vis.spectrum(frame, channel)`` and returns the
        subset that falls inside ``self.nm_window``. Ensures the returned
        arrays are finite and sorted by wavelength.

        Parameters
        ----------
        frame : int
            Frame index to retrieve from the ``vis`` object.
        channel : int
            Channel index to retrieve from the ``vis`` object.

        Returns
        -------
        x : ndarray
            Wavelength values inside ``nm_window`` (sorted, float dtype).
        y : ndarray
            Corresponding spectrum values (float dtype).
        """
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
        """Wrapper that calls the underlying model's full fit function.

        Parameters
        ----------
        x : array_like
            Independent variable (wavelength) at which to evaluate the model.
        C, T_rot, dx, w_inst, base, I_R7, I_R8 : float
            Model parameters forwarded to ``model.full_fit_model``.

        Returns
        -------
        y : ndarray
            Model-evaluated dependent variable at ``x``.
        """
        return self.model.full_fit_model(x, C, T_rot, dx, w_inst, base, I_R7, I_R8)

    # MARK: Fit
    def fit(self, frame, channel, return_fit=True, p0=None):
        """Fit the model to a single frame/channel spectrum.

        Parameters
        ----------
        frame : int
            Frame index to fit.
        channel : int
            Channel index to fit.
        return_fit : bool, optional
            If True, include the fitted model curve (``yfit``) in the result
            dictionary. Default is True.
        p0 : array-like, optional
            Initial guess for the fit parameters. If ``None`` uses the
            fitter's ``p0`` or the last fit parameters when ``warm_start`` is
            enabled.

        Returns
        -------
        result : dict
            Dictionary containing fit results and metadata with keys:
            - ``frame``: frame index
            - ``channel``: channel index
            - ``params``: fitted parameter array
            - ``cov``: covariance matrix from ``curve_fit``
            - ``errors``: 1-sigma parameter uncertainties
            - ``summary``: pandas DataFrame with formatted parameter values
            - ``x``: wavelengths used for the fit
            - ``y``: observed spectrum used for the fit
            - ``yfit``: (optional) model-evaluated fit curve when
              ``return_fit`` is True
        """
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
        """Run fits for multiple frames and channels and collect results.

        Parameters
        ----------
        frames : iterable
            Iterable of frame indices to fit.
        channels : iterable
            Iterable of channel indices to fit.
        return_curves : bool, optional
            If True, also return a dictionary mapping ``(frame, channel)`` to
            the tuple ``(x, y, yfit)``. Default is False.

        Returns
        -------
        df : pandas.DataFrame
            Table of fit results with one row per attempted fit. Columns
            include parameter values, parameter errors (``<name>_err``),
            ``chi2_red``, ``R2``, ``npts`` and any error messages for failed
            fits.
        curves : dict, optional
            Only returned when ``return_curves`` is True. Dictionary keyed by
            ``(frame, channel)`` mapping to ``(x, y, yfit)``.
        """
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

    # MARK: Plotting (delegated)
    def plot_single(self, res, xlim=None, ylim=None, title=None, ax=None):
        from bh_molecule.plotting.fit_plots import plot_single as _plot_single

        return _plot_single(
            res,
            nm_window=self.nm_window,
            xlim=xlim,
            ylim=ylim,
            title=title,
            ax=ax,
        )

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
        from bh_molecule.plotting.fit_plots import plot_grid as _plot_grid

        return _plot_grid(
            curves,
            nm_window=self.nm_window,
            frames=frames,
            channels=channels,
            ncols=ncols,
            xlim=xlim,
            ylim=ylim,
            suptitle=suptitle,
        )

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
        from bh_molecule.plotting.fit_plots import plot_overlay as _plot_overlay

        return _plot_overlay(
            curves,
            nm_window=self.nm_window,
            frame=frame,
            channels=channels,
            xlim=xlim,
            ylim=ylim,
            title=title,
            cmap=cmap,
            line_width=line_width,
            line_alpha=line_alpha,
            scatter_size=scatter_size,
            scatter_alpha=scatter_alpha,
            legend_cols=legend_cols,
        )

    # MARK: Formatting
    @staticmethod
    def _format_table(params, cov, names, units, min_step_map=None):
        """Format parameter values and uncertainties for human display.

        Parameters
        ----------
        params : array-like
            Fitted parameter values.
        cov : ndarray
            Covariance matrix corresponding to ``params``.
        names : sequence of str
            Parameter names to display.
        units : sequence of str
            Units for each parameter.
        min_step_map : dict, optional
            Mapping used to round certain parameters (e.g. ``{'T_rot': 100}``).

        Returns
        -------
        df : pandas.DataFrame
            Two-column DataFrame with columns ``Parameter`` and ``Value`` where
            ``Value`` contains nicely formatted strings like ``"123 ± 4 K"``.
        """
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
