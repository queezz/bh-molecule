"""Tests for the batch fitting bug-fix work.

These tests do not rely on private LHD FITS data; they exercise the
synthetic BH model and the small batch helpers in
``bh_molecule.workflows.batch_fit``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import curve_fit

from bh_molecule.dataio import load_v00_wavelengths
from bh_molecule.fit import BHFitter, DEFAULT_DX_TOL_NM
from bh_molecule.physics import BHModel
from bh_molecule.workflows.batch_fit import frame_plot_filename, run_bh_batch


# ---------------------------------------------------------------------------
# Wavelength-shift bounds: the fitter must allow negative `dx` so it can
# recover small CW / dispersion mismatches in either direction.
# ---------------------------------------------------------------------------


class _DummyVis:
    """Bare-bones stand-in for ``Vis133M`` exposing only ``spectrum``."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self._x = x
        self._y = y

    def spectrum(self, frame: int, channel: int):  # noqa: ARG002 - signature only
        return self._x, self._y


def test_default_bounds_allow_symmetric_shift():
    """Default `dx` bounds must include both negative and positive shifts."""
    vis = _DummyVis(np.linspace(433.05, 433.90, 200), np.zeros(200))
    fitr = BHFitter(vis=vis, model=BHModel())

    lo, hi = fitr.bounds
    dx_idx = fitr.param_names.index("dx")

    assert lo[dx_idx] < 0.0, f"dx lower bound {lo[dx_idx]} must be < 0"
    assert hi[dx_idx] > 0.0, f"dx upper bound {hi[dx_idx]} must be > 0"
    assert hi[dx_idx] - lo[dx_idx] >= 2 * 0.2, (
        "dx bounds should span at least ±0.2 nm"
    )
    assert lo[dx_idx] == pytest.approx(-DEFAULT_DX_TOL_NM)
    assert hi[dx_idx] == pytest.approx(+DEFAULT_DX_TOL_NM)


def test_default_dx_initial_guess_is_zero():
    """`p0[dx]` should be 0 so the fit can move in either direction."""
    vis = _DummyVis(np.linspace(433.05, 433.90, 200), np.zeros(200))
    fitr = BHFitter(vis=vis, model=BHModel())

    dx_idx = fitr.param_names.index("dx")
    assert fitr.p0[dx_idx] == 0.0


def test_default_dx_tol_nm_constant_in_range():
    """`DEFAULT_DX_TOL_NM` must stay in a reasonable physical range."""
    assert 0.1 <= DEFAULT_DX_TOL_NM <= 1.0


def test_set_bounds_accepts_negative_dx():
    """`set_bounds` should accept a negative dx lower bound (regression)."""
    vis = _DummyVis(np.linspace(433.05, 433.90, 200), np.zeros(200))
    fitr = BHFitter(vis=vis, model=BHModel())
    fitr.set_bounds(lower={"dx": -0.5}, upper={"dx": +0.5})

    dx_idx = fitr.param_names.index("dx")
    assert fitr.bounds[0][dx_idx] == pytest.approx(-0.5)
    assert fitr.bounds[1][dx_idx] == pytest.approx(+0.5)


# ---------------------------------------------------------------------------
# Synthetic-data recovery of a known wavelength shift.
# ---------------------------------------------------------------------------


def _synthetic_spectrum(true_dx: float):
    """Build a noisy synthetic BH-model spectrum shifted by ``true_dx`` nm."""
    rng = np.random.default_rng(0)
    v00 = load_v00_wavelengths()
    model = BHModel(v00)
    x = np.linspace(433.05, 433.90, 600)
    y_true = model.full_fit_model(
        x,
        C=0.5,
        T_rot=2500.0,
        dx=true_dx,
        w_inst=0.025,
        base=0.01,
        I_R7=1e-3,
        I_R8=1e-3,
    )
    y = y_true + rng.normal(0.0, 0.005, size=x.shape)
    return x, y, model


@pytest.mark.parametrize("true_dx", [-0.05, -0.02, +0.02, +0.05])
def test_fitter_recovers_signed_shift_on_synthetic_data(true_dx: float):
    """The fitter must recover both signs of a small wavelength offset.

    Shifts above roughly the rotational line spacing (~0.05 nm) make the
    fit ambiguous because the BH Q-branch has a regular comb of lines, so
    we test only the small offsets that the previous bound ``[0, 1]``
    used to silently clamp to zero.
    """
    x, y, model = _synthetic_spectrum(true_dx)
    vis = _DummyVis(x, y)
    fitr = BHFitter(vis=vis, model=model)

    params, _ = curve_fit(
        fitr._f, x, y, p0=fitr.p0, bounds=fitr.bounds, maxfev=fitr.maxfev
    )
    dx_idx = fitr.param_names.index("dx")
    assert params[dx_idx] == pytest.approx(true_dx, abs=0.01), (
        f"recovered dx={params[dx_idx]:+.4f}, expected {true_dx:+.2f}"
    )


# ---------------------------------------------------------------------------
# Frame PNG filenames + frames/ directory creation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "frame, channel, expected",
    [
        (0, 0, "f00_ch00.png"),
        (6, 28, "f06_ch28.png"),
        (10, 35, "f10_ch35.png"),
        (123, 7, "f123_ch07.png"),
    ],
)
def test_frame_plot_filename(frame: int, channel: int, expected: str):
    assert frame_plot_filename(frame, channel) == expected


def test_run_bh_batch_save_frames_creates_directory(monkeypatch, tmp_path):
    """``save_frames=True`` must create ``<shot>/frames/`` and emit PNGs.

    The test stubs out the FITS load + fitting heavy lifting so it can run
    without the LHD private dataset.
    """

    import matplotlib
    matplotlib.use("Agg")

    import bh_molecule.workflows.batch_fit as batch_fit_mod

    # --- Fake Vis133M -------------------------------------------------------

    class _Vis:
        def __init__(self):
            self.cube = np.zeros((4, 4, 100))

        @classmethod
        def from_files(cls, path, scale=1.0):  # noqa: ARG002
            return cls()

        def apply_cw(self, cw_nm):  # noqa: ARG002
            pass

        def set_scale(self, scale):  # noqa: ARG002
            pass

        def set_baseline_zero(self, on=True):  # noqa: ARG002
            pass

        def set_time_linspace(self, t0, t1):  # noqa: ARG002
            pass

        def spectrum(self, frame, channel):  # noqa: ARG002
            x = np.linspace(433.05, 433.90, 50)
            return x, np.zeros_like(x)

    monkeypatch.setattr(batch_fit_mod, "Vis133M", _Vis)
    monkeypatch.setattr(batch_fit_mod, "check_background_flat", lambda *a, **k: None)
    monkeypatch.setattr(
        batch_fit_mod, "scan_signal_frames", lambda *a, **k: ([1, 2], [10, 11], None)
    )
    monkeypatch.setattr(
        batch_fit_mod, "save_batch_fit_grid", lambda *a, **k: None
    )

    # --- Fake fitter that always succeeds ----------------------------------

    class _FakeFitter:
        param_names = ["C", "T_rot", "dx", "w_inst", "base", "I_R7", "I_R8"]

        def __init__(self, vis, model, **_):
            self.vis = vis
            self.model = model

        def set_bounds(self, lower=None, upper=None):  # noqa: ARG002
            pass

        def fit(self, frame, channel, return_fit=True):  # noqa: ARG002
            x = np.linspace(433.05, 433.90, 50)
            y = np.ones_like(x)
            return {
                "params": np.zeros(7),
                "errors": np.zeros(7),
                "x": x,
                "y": y,
                "yfit": y,
            }

        def plot_single(self, res, title=None):  # noqa: ARG002
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.plot(res["x"], res["y"])
            return ax

    monkeypatch.setattr(batch_fit_mod, "BHFitter", _FakeFitter)

    # ``run_bh_batch`` requires the FITS path to exist; create a placeholder.
    fake_fits = tmp_path / "999999.fits"
    fake_fits.write_bytes(b"")

    out_dir = tmp_path / "out"
    run_bh_batch(
        fake_fits,
        frames=[1, 2],
        channels=[10, 11],
        cw=431.91,
        scale=1.0,
        out_dir=out_dir,
        save_frames=True,
    )

    frames_dir = out_dir / "999999" / "frames"
    assert frames_dir.is_dir(), "frames/ directory should be created"

    pngs = sorted(p.name for p in frames_dir.glob("*.png"))
    assert pngs == [
        "f01_ch10.png",
        "f01_ch11.png",
        "f02_ch10.png",
        "f02_ch11.png",
    ], f"unexpected per-fit PNGs: {pngs}"


def test_run_bh_batch_save_frames_default_is_true(monkeypatch, tmp_path):
    """Default behavior (no ``save_frames`` arg) must save per-fit PNGs."""

    import matplotlib
    matplotlib.use("Agg")

    import bh_molecule.workflows.batch_fit as batch_fit_mod

    # Reuse the same stubs as the previous test.
    test_run_bh_batch_save_frames_creates_directory  # marker that we expect this

    class _Vis:
        cube = np.zeros((4, 4, 50))

        @classmethod
        def from_files(cls, path, scale=1.0):  # noqa: ARG002
            return cls()

        def apply_cw(self, cw_nm):  # noqa: ARG002
            pass

        def set_scale(self, scale):  # noqa: ARG002
            pass

        def set_baseline_zero(self, on=True):  # noqa: ARG002
            pass

        def set_time_linspace(self, t0, t1):  # noqa: ARG002
            pass

        def spectrum(self, frame, channel):  # noqa: ARG002
            x = np.linspace(433.05, 433.90, 30)
            return x, np.zeros_like(x)

    monkeypatch.setattr(batch_fit_mod, "Vis133M", _Vis)
    monkeypatch.setattr(batch_fit_mod, "check_background_flat", lambda *a, **k: None)
    monkeypatch.setattr(
        batch_fit_mod, "scan_signal_frames", lambda *a, **k: ([1], [10], None)
    )
    monkeypatch.setattr(
        batch_fit_mod, "save_batch_fit_grid", lambda *a, **k: None
    )

    class _FakeFitter:
        param_names = ["C", "T_rot", "dx", "w_inst", "base", "I_R7", "I_R8"]

        def __init__(self, vis, model, **_):
            self.vis = vis
            self.model = model

        def set_bounds(self, lower=None, upper=None):  # noqa: ARG002
            pass

        def fit(self, frame, channel, return_fit=True):  # noqa: ARG002
            x = np.linspace(433.05, 433.90, 30)
            y = np.ones_like(x)
            return {
                "params": np.zeros(7),
                "errors": np.zeros(7),
                "x": x,
                "y": y,
                "yfit": y,
            }

        def plot_single(self, res, title=None):  # noqa: ARG002
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.plot(res["x"], res["y"])
            return ax

    monkeypatch.setattr(batch_fit_mod, "BHFitter", _FakeFitter)

    fake_fits = tmp_path / "888888.fits"
    fake_fits.write_bytes(b"")

    run_bh_batch(
        fake_fits,
        frames=[1],
        channels=[10],
        cw=431.91,
        scale=1.0,
        out_dir=tmp_path / "out",
    )

    assert (tmp_path / "out" / "888888" / "frames" / "f01_ch10.png").is_file()


# ---------------------------------------------------------------------------
# Pretty-formatter must not blow up when an error term is large.
# ---------------------------------------------------------------------------


def test_format_table_handles_large_errors():
    """Regression: huge `e_raw` would crash with 'Format specifier missing precision'."""
    params = np.array([10.0, 10000.0, 0.5, 0.05, 9.0, 0.5, 0.5])
    cov = np.diag(np.array([1e6, 1e10, 1.0, 1.0, 100.0, 50.0, 50.0])) ** 2
    df = BHFitter._format_table(
        params,
        cov,
        ["C", "T_rot", "dx", "w_inst", "base", "I_R7", "I_R8"],
        ["", "K", "nm", "nm", "", "", ""],
    )
    assert list(df.columns) == ["Parameter", "Value"]
    assert len(df) == 7
    assert all(isinstance(v, str) for v in df["Value"])
