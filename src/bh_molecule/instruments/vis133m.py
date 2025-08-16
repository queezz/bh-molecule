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
    """Minimal loader/processor for VIS-1.33 m data with per-channel wavecal."""

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

    @property
    def shape(self):
        return self.cube.shape  # (F, C, P)

    def set_scale(self, scale: float):
        self.scale = float(scale)

    def set_dark(
        self,
        *,
        frame: int | None = None,
        channel: int | None = None,
        vector=None,
        value: float | None = None,
    ):
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
        lo, hi = float(nm_range[0]), float(nm_range[1])
        m = (self.wl_nm >= lo) & (self.wl_nm <= hi)  # (C, P)
        img = (self.cube * self.scale * m[None, ...]).sum(axis=2)  # (F, C)
        if subtract_dark and self._dark is not None:
            img = self._apply_dark(img)
        return img  # (F, C)

    def spectrum(self, frame: int, channel: int) -> tuple[np.ndarray, np.ndarray]:
        F, C, P = self.shape
        if not (0 <= frame < F and 0 <= channel < C):
            raise IndexError("frame/channel out of range")
        return self.wl_nm[channel], self.cube[frame, channel] * self.scale  # (P,), (P,)

    def _apply_dark(self, img_fc: np.ndarray) -> np.ndarray:
        d = self._dark
        if isinstance(d, tuple) and d and d[0] == "idx":
            _, f, c = d
            return img_fc - float(img_fc[f, c])
        return img_fc - d  # relies on NumPy broadcasting (scalar, (F,), (C,), or (F,C))

    # --- nice introspection ---
    @property
    def shape_fcp(self) -> FCPShape:
        """Named shape with axis meanings."""
        F, C, P = self.cube.shape
        return FCPShape(F, C, P)

    @property
    def axis_legend(self) -> dict[str, str]:
        """Short legend you can print in REPL."""
        return {
            "F": "frame (time)",
            "C": "channel (position)",
            "P": "pixel (wavelength)",
        }

    def explain(self) -> None:
        """Print a one-liner about axes (handy in notebooks)."""
        print(self.shape_fcp)
        print("Axes:", self.axis_legend)

    # --- quick views (zero copy) ---
    def frame_image(self, frame: int) -> np.ndarray:
        """Return (C,P) image for a frame."""
        return self.cube[frame]

    def channel_stack(self, channel: int) -> np.ndarray:
        """Return (F,P) stack for a channel."""
        return self.cube[:, channel, :]

    def pixel_map(self, pixel: int) -> np.ndarray:
        """Return (F,C) map at a fixed detector pixel."""
        return self.cube[:, :, pixel]

    # --- plotting convenience ---
    def plot_spectrum(self, frame: int, channel: int, ax=None):
        """Plot spectrum at (frame, channel) with labels."""
        import matplotlib.pyplot as plt

        x = self.wl_nm[channel]  # (P,)
        y = self.cube[frame, channel] * self.scale
        ax = ax or plt.gca()
        ax.plot(x, y)
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Signal (scaled)")
        ax.set_title(f"Spectrum: frame={frame} (time), channel={channel} (position)")
        return ax

    # MARK: Time
    def _time_axis(self, *, require_time: bool = False):
        """Return x-axis vector and label.
        If time_s is set -> ('time (s)'), else frame index -> ('frame').
        If require_time=True and no time_s -> error.
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
        F = self.cube.shape[0]
        self.time_s = np.linspace(float(start_s), float(stop_s), F)

    def set_time_period(self, period_s: float, start_s: float = 0.0):
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
        """Sum pixels [start:stop) → (F,C). 'stop' is exclusive (Python slicing)."""
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
        """Sum wavelengths within [lo, hi] nm (per-channel mask) → (F,C)."""
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
        """Visualize map_pixel_range(start, stop)."""
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
        """Visualize map_band(nm_range)."""
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
    ):
        """Plot spectrum at (frame, channel) with Plotly (interactive).
        Example:
        >>>fig = s26.plot_spectrum_plotly(38, 36)
        >>>fig.show()
        """
        import numpy as np

        try:
            import plotly.graph_objects as go
        except Exception as e:
            raise ImportError("plotly is required for plot_spectrum_plotly()") from e

        x = np.asarray(self.wl_nm[channel], dtype=float)  # (P,)
        y = np.asarray(self.cube[frame, channel] * self.scale, dtype=float)

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
