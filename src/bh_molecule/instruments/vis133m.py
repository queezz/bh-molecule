import numpy as np
import pandas as pd
from astropy.io import fits
from typing import NamedTuple


class FCPShape(NamedTuple):
    F: int  # frames (time)
    C: int  # channels (position)
    P: int  # pixels (wavelength)

    def __str__(self) -> str:
        return f"{self.F}F × {self.C}C × {self.P}P (frames × channels × pixels)"


class Vis133M:
    r"""Minimal loader/processor for VIS-1.33 m data with per-channel wavecal.

    This class wraps a FITS data cube produced by the VIS-1.33 m instrument and
    a per-channel wavelength calibration CSV. It provides convenient accessors
    and plotting helpers for frames, channels and pixels, plus simple dark
    subtraction and time-axis helpers.

    Parameters
    ----------
    fits_path : str
        Path to a FITS file containing a 3D data cube with shape ``(F, C, P)``
        (frames, channels, pixels).
    wavecal_csv : str
        Path to a CSV file containing per-channel wavelength calibration. The
        first column must contain channel indices (0-based preferred, 1-based
        accepted). Remaining columns are interpreted as wavelength values [nm]
        for each pixel.
    scale : float, optional
        Multiplicative scale factor applied to the cube data (default 1.0).

    Attributes
    ----------
    cube : ndarray
        The data cube of shape ``(F, C, P)``.
    wl_nm : ndarray
        Per-channel wavelength array with shape ``(C, P)`` in nanometres.
    header : dict
        FITS header converted to a dict.
    exptime : float | None
        Exposure time read from the FITS header when available.
    time_s : ndarray | None
        Optional time vector in seconds of length ``F``. If unset, frame
        indices are used for plotting/time axes.
    """

    def __init__(self, fits_path: str, wavecal_csv: str, scale: float = 1.0):
        hdu = fits.open(fits_path)[0]
        cube = np.asarray(hdu.data, dtype=float)  # (F, C, P)
        if cube.ndim != 3:
            raise ValueError(f"Expected 3D cube, got {cube.ndim}D")
        self.cube = cube
        self.header = dict(hdu.header)
        self.exptime = self.header.get("EXPTIME") or self.header.get("EXPOSURE")
        self.filename = fits_path
        self.scale = float(scale)
        self.time_s = None  # optional (F,) time vector in seconds

        F, C, P = cube.shape
        df = pd.read_csv(wavecal_csv, header=None)
        ch = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
        # accept 0-based (preferred) or 1-based indices
        is_zero_based = np.all(ch[:C] == np.arange(C))
        is_one_based = np.all(ch[:C] == np.arange(1, C + 1))
        if ch.size < C or not (is_zero_based or is_one_based):
            raise ValueError("Wavecal col 0 must be channel indices 0..C-1 (or 1..C).")
        wl = (
            pd.to_numeric(df.iloc[:C, 1 : P + 1].stack(), errors="coerce")
            .to_numpy()
            .reshape(C, P)
        )
        if wl.shape != (C, P):
            raise ValueError(f"Wavecal shape {wl.shape}")
        if np.isnan(wl).any():
            raise ValueError("Wavecal contains NaN")
        self.wl_nm = wl  # (C, P)

        self._dark = None  # None | scalar | (F,) | (C,) | (F,C) | ("idx", f, c)
        self._baseline_zero = False  # If True, subtract per-row minima for spectra

    @property
    def shape(self):
        """Return the shape of the internal data cube.

        Returns
        -------
        tuple
            Shape ``(F, C, P)`` where ``F`` is frames, ``C`` is channels and
            ``P`` is pixels (wavelength samples).
        """
        return self.cube.shape  # (F, C, P)

    def set_scale(self, scale: float):
        """Set the global multiplicative scale applied to the cube.

        Parameters
        ----------
        scale : float
            New scale factor.
        """
        self.scale = float(scale)

    def set_baseline_zero(self, enable: bool = True) -> None:
        """Enable/disable per-spectrum minimum subtraction.

        When enabled, each spectrum row (across pixels) will be shifted so its
        minimum is zero in methods that return spectra-like arrays.

        Parameters
        ----------
        enable : bool
            If True, subtract per-row minima from spectra results. If False,
            disable this behaviour.
        """
        self._baseline_zero = bool(enable)

    def _rowmin_stack_fp(self, stack_fp: np.ndarray) -> np.ndarray:
        """Return stack with row-wise minima (over pixels) subtracted.

        Expects shape (F, P) and subtracts min over axis=1.
        """
        return stack_fp - stack_fp.min(axis=1, keepdims=True)

    def _rowmin_row_p(self, row_p: np.ndarray) -> np.ndarray:
        """Return a 1D spectrum with its minimum subtracted."""
        return row_p - float(np.min(row_p))

    def set_dark(
        self,
        *,
        frame: int | None = None,
        channel: int | None = None,
        vector=None,
        value: float | None = None,
    ):
        """Configure dark subtraction behaviour.

        Dark can be specified in several ways:
        - ``vector``: an array-like that will be used directly as the dark
          correction (broadcasting rules apply).
        - ``value``: a scalar dark value subtracted from all pixels.
        - ``frame`` and ``channel``: record a reference index; subtraction will
          subtract the value found at that (frame, channel) location when
          applied.

        Parameters
        ----------
        frame : int | None
            Frame index used for index-based dark (with ``channel``).
        channel : int | None
            Channel index used for index-based dark (with ``frame``).
        vector : array-like | None
            Direct dark vector/array to use for subtraction.
        value : float | None
            Scalar dark value to subtract.
        """
        if vector is not None:
            self._dark = np.asarray(vector, dtype=float)
        elif value is not None:
            self._dark = float(value)
        elif frame is not None and channel is not None:
            self._dark = ("idx", int(frame), int(channel))
        else:
            self._dark = None

    def band(
        self, nm_range: tuple[float, float], subtract_dark: bool = True
    ) -> np.ndarray:
        """Sum signal within a wavelength band (per-channel) returning an
        image of shape ``(F, C)``.

        Parameters
        ----------
        nm_range : tuple of float
            ``(lo, hi)`` wavelength range in nanometres (inclusive).
        subtract_dark : bool, optional
            If True and a dark is configured via ``set_dark``, subtract the
            dark from the resulting image (default True).

        Returns
        -------
        ndarray
            Summed image of shape ``(F, C)`` in the same units as the cube
            multiplied by the current ``scale``.
        """
        lo, hi = float(nm_range[0]), float(nm_range[1])
        m = (self.wl_nm >= lo) & (self.wl_nm <= hi)  # (C, P)
        img = (self.cube * self.scale * m[None, ...]).sum(axis=2)  # (F, C)
        if subtract_dark and self._dark is not None:
            img = self._apply_dark(img)
        return img  # (F, C)

    def spectrum(
        self, frame: int, channel: int, *, zero_min: bool | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return wavelength and signal arrays for a given ``(frame, channel)``.

        Parameters
        ----------
        frame : int
            Frame index (0-based).
        channel : int
            Channel index (0-based).

        Returns
        -------
        tuple of ndarray
            ``(wavelengths, signal)`` where both arrays have shape ``(P,)`` and
            wavelengths are in nanometres. ``signal`` is scaled by the current
            ``scale`` attribute.

        Raises
        ------
        IndexError
            If ``frame`` or ``channel`` are out of range.
        """
        F, C, P = self.shape
        if not (0 <= frame < F and 0 <= channel < C):
            raise IndexError("frame/channel out of range")
        x = self.wl_nm[channel]
        row = self.cube[frame, channel]
        use_zero = self._baseline_zero if zero_min is None else bool(zero_min)
        if use_zero:
            row = self._rowmin_row_p(row)
        return x, row * self.scale  # (P,), (P,)

    def _apply_dark(self, img_fc: np.ndarray) -> np.ndarray:
        """Apply the configured dark subtraction to an ``(F, C)`` image.

        Parameters
        ----------
        img_fc : ndarray
            Image of shape ``(F, C)`` to subtract the dark from.

        Returns
        -------
        ndarray
            Dark-subtracted image.
        """
        d = self._dark
        if isinstance(d, tuple) and d and d[0] == "idx":
            _, f, c = d
            return img_fc - float(img_fc[f, c])
        return img_fc - d  # relies on NumPy broadcasting (scalar, (F,), (C,), or (F,C))

    # MARK: Nice
    # --- nice introspection ---
    @property
    def shape_fcp(self) -> FCPShape:
        """Named shape with axis meanings.

        Returns
        -------
        FCPShape
            Named tuple containing ``F``, ``C`` and ``P`` describing the
            number of frames, channels and pixels respectively.
        """
        F, C, P = self.cube.shape
        return FCPShape(F, C, P)

    @property
    def axis_legend(self) -> dict[str, str]:
        """Mapping of axis short names to human-readable descriptions.

        Returns
        -------
        dict
            Mapping with keys ``'F'``, ``'C'``, ``'P'`` describing the axes.
        """
        return {
            "F": "frame (time)",
            "C": "channel (position)",
            "P": "pixel (wavelength)",
        }

    def explain(self) -> None:
        """Print a short human-readable description of the data axes.

        This helper is convenient in interactive sessions or notebooks.
        """
        print(self.shape_fcp)
        print("Axes:", self.axis_legend)

    # --- quick views (zero copy) ---
    def frame_image(self, frame: int, *, zero_min: bool | None = None) -> np.ndarray:
        """Return the channel×pixel image for a single frame.

        Parameters
        ----------
        frame : int
            Frame index (0-based).

        Returns
        -------
        ndarray
            Array of shape ``(C, P)`` corresponding to the requested frame.
        """
        img = self.cube[frame]
        use_zero = self._baseline_zero if zero_min is None else bool(zero_min)
        if use_zero:
            img = img - img.min(axis=1, keepdims=True)
        return img

    def channel_stack(
        self, channel: int, *, zero_min: bool | None = None
    ) -> np.ndarray:
        """Return the time×pixel stack for a single channel.

        Parameters
        ----------
        channel : int
            Channel index (0-based).

        Returns
        -------
        ndarray
            Array of shape ``(F, P)`` containing the stack for the channel.
        """
        stack = self.cube[:, channel, :]
        use_zero = self._baseline_zero if zero_min is None else bool(zero_min)
        if use_zero:
            stack = self._rowmin_stack_fp(stack)
        return stack

    def pixel_map(self, pixel: int) -> np.ndarray:
        """Return the frame×channel map at a fixed detector pixel.

        Parameters
        ----------
        pixel : int
            Pixel index (0-based).

        Returns
        -------
        ndarray
            Array of shape ``(F, C)`` containing values at the given pixel.
        """
        return self.cube[:, :, pixel]

    # --- plotting convenience ---
    def plot_spectrum(
        self, frame: int, channel: int, ax=None, *, zero_min: bool | None = None
    ):
        """Plot the spectrum at a given frame and channel using Matplotlib.

        Parameters
        ----------
        frame : int
            Frame index (0-based).
        channel : int
            Channel index (0-based).
        ax : matplotlib.axes.Axes | None, optional
            Axes to plot into. If None the current axes are used.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.
        """
        import matplotlib.pyplot as plt

        x, y = self.spectrum(frame, channel, zero_min=zero_min)
        ax = ax or plt.gca()
        ax.plot(x, y)
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Signal (scaled)")
        ax.set_title(f"Spectrum: frame={frame} (time), channel={channel} (position)")
        return ax

    # MARK: Time
    def _time_axis(self, *, require_time: bool = False):
        """Return an x-axis vector and label for plotting time-like data.

        Parameters
        ----------
        require_time : bool, optional
            If True and no explicit time vector is set, raise an error. If
            False, fallback to frame indices (default False).

        Returns
        -------
        tuple
            ``(x_vector, xlabel)`` where ``x_vector`` is either ``time_s`` or a
            frame index vector and ``xlabel`` is a human-readable label.

        Raises
        ------
        RuntimeError
            If ``require_time`` is True but no time vector has been set.
        """
        F, _, _ = self.cube.shape
        if self.time_s is not None:
            return self.time_s, "time (s)"
        if require_time:
            raise RuntimeError(
                "No time vector set. Call set_time()/set_time_linspace()/set_time_period() "
                "or use require_time=False to plot against frame index."
            )
        return np.arange(F, dtype=float), "frame"

    def set_time(self, t):
        """Set an explicit time vector for the frames.

        Parameters
        ----------
        t : array-like
            Time vector (seconds) of length ``F`` where ``F`` is the number of
            frames in the cube. Must be non-decreasing and contain finite
            values.

        Raises
        ------
        ValueError
            If the vector length does not match ``F``, contains non-finite
            values, or is not non-decreasing.
        """
        t = np.asarray(t, dtype=float)
        F = self.cube.shape[0]
        if t.shape != (F,):
            raise ValueError(f"time vector length {t.size} != F {F}")
        if not np.all(np.isfinite(t)):
            raise ValueError("time vector contains non-finite values")
        if np.any(np.diff(t) < 0):
            raise ValueError("time vector must be non-decreasing")
        self.time_s = t

    def set_time_linspace(self, start_s: float, stop_s: float):
        """Set ``time_s`` to a linearly spaced vector between two times.

        Parameters
        ----------
        start_s : float
            Start time in seconds.
        stop_s : float
            Stop time in seconds.
        """
        F = self.cube.shape[0]
        self.time_s = np.linspace(float(start_s), float(stop_s), F)

    def set_time_period(self, period_s: float, start_s: float = 0.0):
        """Set ``time_s`` assuming a constant frame period.

        Parameters
        ----------
        period_s : float
            Time between successive frames in seconds.
        start_s : float, optional
            Time of the first frame (default 0.0 s).
        """
        F = self.cube.shape[0]
        self.time_s = float(start_s) + np.arange(F) * float(period_s)

    # MARK: Plotting
    def plot_channel_stack(
        self,
        channel: int,
        *,
        ax=None,
        cmap=None,
        cbar_label="intensity",
        time_line: float | None = None,
        require_time: bool = False,
    ):
        """Plot a channel stack (time × wavelength) as an image.

        Parameters
        ----------
        channel : int
            Channel index (0-based) to plot.
        ax : matplotlib.axes.Axes | None, optional
            Axes to plot into. If None the current axes are used.
        cmap : str | Colormap | None, optional
            Colormap to use for the image.
        cbar_label : str, optional
            Label for the colorbar (default "intensity").
        time_line : float | None, optional
            Optional vertical line (x coordinate in time units) to indicate a
            time-of-interest.
        require_time : bool, optional
            If True require an explicit time vector (raise if unset).

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the image.
        """
        import matplotlib.pyplot as plt

        stack = self.channel_stack(channel)  # (F,P)
        wl = self.wl_nm[channel]  # (P,)
        arr = stack.T  # -> (P,F)
        if wl.size >= 2 and wl[1] < wl[0]:
            wl, arr = wl[::-1], arr[::-1, :]

        t, xlabel = self._time_axis(require_time=require_time)
        extent = (t[0], t[-1] if t.size > 1 else 0.0, wl[0], wl[-1])

        ax = ax or plt.gca()
        im = ax.imshow(
            arr * self.scale, origin="lower", aspect="auto", extent=extent, cmap=cmap
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("wavelength (nm)")
        cb = plt.colorbar(im, ax=ax)
        cb.set_label(cbar_label)
        if time_line is not None:
            ax.axvline(float(time_line), ls="--", lw=1.5, color="w", alpha=0.8)
        return ax

    def plot_pixel_map(
        self,
        pixel: int,
        *,
        ax=None,
        cmap=None,
        cbar_label="intensity",
        channel_line: int | None = None,
        require_time: bool = False,
    ):
        """Plot a frame×channel image for a fixed pixel index.

        Parameters
        ----------
        pixel : int
            Pixel index (0-based).
        ax : matplotlib.axes.Axes | None, optional
            Axes to plot into. If None the current axes are used.
        cmap : str | Colormap | None, optional
            Colormap to use for the image.
        cbar_label : str, optional
            Label for the colorbar (default "intensity").
        channel_line : int | None, optional
            Optional horizontal line indicating a channel of interest.
        require_time : bool, optional
            If True require an explicit time vector (raise if unset).

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the image.
        """
        import matplotlib.pyplot as plt

        arr = self.pixel_map(pixel).T  # (C,F)
        C = arr.shape[0]
        t, xlabel = self._time_axis(require_time=require_time)
        extent = (t[0], t[-1] if t.size > 1 else 0.0, 1, C)

        ax = ax or plt.gca()
        im = ax.imshow(
            arr * self.scale, origin="lower", aspect="auto", extent=extent, cmap=cmap
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("channel")
        cb = plt.colorbar(im, ax=ax)
        cb.set_label(cbar_label)
        if channel_line is not None:
            ax.axhline(float(channel_line), ls="--", lw=1.5, color="w", alpha=0.8)
        return ax

    # MARK: Maps
    # ----- maps (F,C) -----
    def map_pixel_range(
        self, start: int, stop: int, *, subtract_dark: bool = True
    ) -> np.ndarray:
        """Sum over a pixel window returning an (F, C) image.

        Parameters
        ----------
        start : int
            Start pixel index (inclusive).
        stop : int
            Stop pixel index (exclusive).
        subtract_dark : bool, optional
            If True subtract a configured dark (default True).

        Returns
        -------
        ndarray
            Summed image of shape ``(F, C)``.
        """
        F, C, P = self.cube.shape
        if not (0 <= start < stop <= P):
            raise ValueError(f"pixel window [{start}:{stop}) out of 0..{P-1}")
        img = self.cube[:, :, start:stop].sum(axis=2) * self.scale  # (F,C)
        if subtract_dark and self._dark is not None:
            img = self._apply_dark(img)
        return img

    def map_band(
        self, nm_range: tuple[float, float], *, subtract_dark: bool = True
    ) -> np.ndarray:
        """Sum signal within a wavelength band (per-channel) returning (F, C).

        Parameters
        ----------
        nm_range : tuple of float
            ``(lo, hi)`` wavelength range in nanometres (inclusive).
        subtract_dark : bool, optional
            If True subtract a configured dark (default True).

        Returns
        -------
        ndarray
            Summed image of shape ``(F, C)``.
        """
        lo, hi = map(float, nm_range)
        m = (self.wl_nm >= lo) & (self.wl_nm <= hi)  # (C,P)
        img = (self.cube * m[None, ...]).sum(axis=2) * self.scale  # (F,C)
        if subtract_dark and self._dark is not None:
            img = self._apply_dark(img)
        return img

    # ----- plotting helpers (time on x, channel on y) -----
    def _plot_fc(
        self,
        img_fc: np.ndarray,
        *,
        ax=None,
        cmap=None,
        cbar_label="intensity",
        channel_line: int | None = None,
        require_time: bool = False,
    ):
        """Internal helper: plot an (F, C) image with time on x and channel on y.

        Parameters
        ----------
        img_fc : ndarray
            Image of shape ``(F, C)`` to plot.
        ax : matplotlib.axes.Axes | None, optional
            Axes to plot into. If None the current axes are used.
        cmap : str | Colormap | None, optional
            Colormap to use.
        cbar_label : str, optional
            Label for the colorbar (default "intensity").
        channel_line : int | None, optional
            Optional horizontal line indicating a channel of interest.
        require_time : bool, optional
            If True require an explicit time vector (raise if unset).

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the image.
        """
        import matplotlib.pyplot as plt

        arr = img_fc.T  # (C,F)
        C = arr.shape[0]
        t, xlabel = self._time_axis(require_time=require_time)
        extent = (t[0], t[-1] if t.size > 1 else 0.0, 1, C)

        ax = ax or plt.gca()
        im = ax.imshow(arr, origin="lower", aspect="auto", extent=extent, cmap=cmap)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("channel")
        cb = plt.colorbar(im, ax=ax)
        cb.set_label(cbar_label)
        if channel_line is not None:
            ax.axhline(float(channel_line), ls="--", lw=1.5, color="w", alpha=0.85)
        return ax

    def plot_pixel_range(
        self,
        start: int,
        stop: int,
        *,
        ax=None,
        cmap=None,
        cbar_label="intensity",
        channel_line: int | None = None,
        require_time: bool = False,
        subtract_dark: bool = True,
    ):
        """Plot the result of ``map_pixel_range(start, stop)``.

        Parameters
        ----------
        start : int
            Start pixel index (inclusive).
        stop : int
            Stop pixel index (exclusive).
        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the image.
        """
        img = self.map_pixel_range(start, stop, subtract_dark=subtract_dark)
        return self._plot_fc(
            img,
            ax=ax,
            cmap=cmap,
            cbar_label=cbar_label,
            channel_line=channel_line,
            require_time=require_time,
        )

    def plot_band_map(
        self,
        nm_range: tuple[float, float],
        *,
        ax=None,
        cmap=None,
        cbar_label="intensity",
        channel_line: int | None = None,
        require_time: bool = False,
        subtract_dark: bool = True,
    ):
        """Plot the result of ``map_band(nm_range)``.

        Parameters
        ----------
        nm_range : tuple of float
            ``(lo, hi)`` wavelength range in nanometres.


        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the image.


        See Also map_band, _plot_fc
        """
        img = self.map_band(nm_range, subtract_dark=subtract_dark)
        return self._plot_fc(
            img,
            ax=ax,
            cmap=cmap,
            cbar_label=cbar_label,
            channel_line=channel_line,
            require_time=require_time,
        )

    # MARK: Plotly
    def plot_spectrum_plotly(
        self,
        frame: int,
        channel: int,
        *,
        sort_wavelength: bool = True,
        line_shape: str = "linear",
        zero_min: bool | None = None,
    ):
        """Return an interactive Plotly figure for a spectrum.

        Parameters
        ----------
        frame : int
            Frame index (0-based).
        channel : int
            Channel index (0-based).
        sort_wavelength : bool, optional
            If True, sort the wavelength vector to be monotonic for better
            interactive behaviour (default True).
        line_shape : str, optional
            Plotly line shape (e.g. 'linear', 'spline').

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive figure containing the spectrum.

        Raises
        ------
        ImportError
            If Plotly is not available.

        Example
        -------
        > fig = s26.plot_spectrum_plotly(38, 36)
        > fig.show()
        """
        import numpy as np

        try:
            import plotly.graph_objects as go
        except Exception as e:
            raise ImportError("plotly is required for plot_spectrum_plotly()") from e

        x = np.asarray(self.wl_nm[channel], dtype=float)  # (P,)
        row = self.cube[frame, channel]
        use_zero = self._baseline_zero if zero_min is None else bool(zero_min)
        if use_zero:
            row = self._rowmin_row_p(row)
        y = np.asarray(row * self.scale, dtype=float)

        # Make wavelength monotonic for nicer interaction (optional)
        if sort_wavelength and (x.size > 1) and np.any(np.diff(x) < 0):
            idx = np.argsort(x)
            x, y = x[idx], y[idx]

        fig = go.Figure(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(shape=line_shape),
                name=f"f={frame}, ch={channel}",
                hovertemplate="λ = %{x:.3f} nm<br>I = %{y:.6g}<extra></extra>",
            )
        )
        fig.update_layout(
            title=f"Spectrum: frame={frame} (time), channel={channel} (position)",
            xaxis_title="Wavelength (nm)",
            yaxis_title="Signal (scaled)",
            legend_title_text=None,
            hovermode="x unified",
        )
        return fig
