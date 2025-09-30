#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
fits_scroll.py — FITS cube viewer using PyQtGraph with keyboard scrolling, aspect toggle, and histogram controls.

Usage:
    python fits_scroll.py path/to/file.fits

Keys:
    Left / Right or [ / ] : prev/next frame
    A : auto-rescale (recompute percentile stretch)
    , / . : low percentile -1 / +1
    ; / ' : high percentile -1 / +1
    H : toggle histogram window
    E : toggle aspect ('equal' / 'auto')
    S : save current frame as PNG
    Q or Esc : quit
"""
import argparse
import os
import numpy as np
from astropy.io import fits
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
from pyqtgraph.exporters import ImageExporter


def percentile_limits(img, lo=1, hi=99):
    finite = np.isfinite(img)
    if not np.any(finite):
        return np.nanmin(img), np.nanmax(img)
    vmin = np.percentile(img[finite], lo)
    vmax = np.percentile(img[finite], hi)
    if vmin == vmax:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def clamp_percentiles(lo, hi):
    lo = max(0.0, min(lo, 100.0))
    hi = max(0.0, min(hi, 100.0))
    if lo >= hi:
        lo = max(0.0, hi - 0.5)
    return lo, hi


class FitsViewer(QtWidgets.QMainWindow):
    def __init__(self, data, file_path, lo, hi, aspect_equal, window_size_px):
        super().__init__()
        self.data = data.astype(float)
        self.file_path = file_path
        self.num_frames, self.height, self.width = self.data.shape
        self.frame_index = 0
        self.lo = lo
        self.hi = hi
        self.aspect_equal = aspect_equal
        self.hist_panel_visible = True

        # Configure PyQtGraph
        pg.setConfigOptions(antialias=False, imageAxisOrder="row-major")

        # Central splitter with image on the left and histogram LUT on the right
        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        layout.addWidget(self.splitter)
        self.setCentralWidget(central)

        # Left: image plot
        self.glw = pg.GraphicsLayoutWidget()
        self.plot = self.glw.addPlot()
        self.splitter.addWidget(self.glw)
        self.plot.setLabel("bottom", "X (pixels; wavelength axis)")
        self.plot.setLabel("left", "Y (pixels)")

        # ViewBox and ImageItem
        self.view_box = self.plot.getViewBox()
        self.view_box.invertY(True)  # origin at lower-left like matplotlib origin="lower"
        self.view_box.setAspectLocked(self.aspect_equal)

        self.image_item = pg.ImageItem()
        self.plot.addItem(self.image_item)

        # Right: HistogramLUTWidget for interactive levels and histogram
        self.hist_widget = pg.HistogramLUTWidget()
        try:
            self.hist_widget.gradient.loadPreset("grey")
        except Exception:
            pass
        self.hist_widget.setImageItem(self.image_item)
        self.splitter.addWidget(self.hist_widget)
        self.splitter.setStretchFactor(0, 4)
        self.splitter.setStretchFactor(1, 1)

        # Grayscale colormap
        try:
            cmap = pg.colormap.get("gray")
            lut = cmap.getLookupTable(0.0, 1.0, 256)
            self.image_item.setLookupTable(lut)
        except Exception:
            pass

        # Initial draw and window size
        self._update_image()
        if window_size_px is not None:
            self.resize(window_size_px[0], window_size_px[1])

        # Add Help menu and toolbar button for keybindings
        QActionClass = getattr(QtWidgets, "QAction", None) or getattr(QtGui, "QAction", None)
        if QActionClass is None:
            # Last resort: try attribute on QtCore (very unlikely)
            QActionClass = getattr(QtCore, "QAction", None)
        help_action = QActionClass("Key bindings", self)
        try:
            help_action.triggered.connect(self._show_keybindings)
        except Exception:
            # Some bindings may use different signals; ignore if connect fails
            pass
        menubar = self.menuBar()
        help_menu = menubar.addMenu("Help")
        help_menu.addAction(help_action)

        toolbar = self.addToolBar("help")
        toolbar.addAction(help_action)
    # ----- UI Helpers -----
    def _update_title(self):
        asp = "EQUAL" if self.aspect_equal else "AUTO"
        base = os.path.basename(self.file_path)
        self.setWindowTitle(
            f"FITS Cube Viewer — {base}  [frame {self.frame_index + 1}/{self.num_frames}]  p{self.lo:.1f}-{self.hi:.1f}  AR:{asp}"
        )

    def _current_frame(self):
        return self.data[self.frame_index]

    def _update_image(self):
        img = self._current_frame()
        vmin, vmax = percentile_limits(img, self.lo, self.hi)
        self.image_item.setImage(img, autoLevels=False)
        self.image_item.setLevels((vmin, vmax))
        self.view_box.setAspectLocked(self.aspect_equal)
        self.plot.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
        self._update_title()
        # HistogramLUTWidget updates itself when image levels change
        # Nothing else needed here

    # ----- Key handling -----
    def keyPressEvent(self, event):
        key = event.key()
        if key in (QtCore.Qt.Key.Key_Right, QtCore.Qt.Key.Key_BracketRight):
            self.frame_index = (self.frame_index + 1) % self.num_frames
            self._update_image()
        elif key in (QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_BracketLeft):
            self.frame_index = (self.frame_index - 1) % self.num_frames
            self._update_image()
        elif key == QtCore.Qt.Key.Key_A:
            # Recompute levels; already done in _update_image()
            self._update_image()
        elif key == QtCore.Qt.Key.Key_Comma:
            self.lo, self.hi = clamp_percentiles(self.lo - 1.0, self.hi)
            self._update_image()
        elif key == QtCore.Qt.Key.Key_Period:
            self.lo, self.hi = clamp_percentiles(self.lo + 1.0, self.hi)
            self._update_image()
        elif key == QtCore.Qt.Key.Key_Semicolon:
            self.lo, self.hi = clamp_percentiles(self.lo, self.hi - 1.0)
            self._update_image()
        elif key == QtCore.Qt.Key.Key_Apostrophe:
            self.lo, self.hi = clamp_percentiles(self.lo, self.hi + 1.0)
            self._update_image()
        elif key == QtCore.Qt.Key.Key_H:
            self._toggle_histogram_panel()
        elif key == QtCore.Qt.Key.Key_E:
            self.aspect_equal = not self.aspect_equal
            self._update_image()
        elif key == QtCore.Qt.Key.Key_S:
            self._save_png()
        elif key in (QtCore.Qt.Key.Key_Q, QtCore.Qt.Key.Key_Escape):
            self.close()
        else:
            super().keyPressEvent(event)

    # ----- Actions -----
    def _toggle_histogram_panel(self):
        self.hist_panel_visible = not self.hist_panel_visible
        self.hist_widget.setVisible(self.hist_panel_visible)

    def _save_png(self):
        base = os.path.splitext(self.file_path)[0]
        out = f"{base}_frame{self.frame_index:04d}.png"
        # Use exporter to capture the plot (with axes)
        exporter = ImageExporter(self.plot)
        try:
            exporter.parameters()["antialias"] = True
        except Exception:
            pass
        exporter.export(out)
        print(f"Saved {out}")

    def _show_keybindings(self):
        text = (
            "Left / Right or [ / ] : prev/next frame\n"
            "A : recompute percentiles (auto-rescale)\n"
            ", / . : low percentile -1 / +1\n"
            "; / ' : high percentile -1 / +1\n"
            "H : toggle histogram panel\n"
            "E : toggle aspect ('equal' / 'auto')\n"
            "S : save current frame as PNG\n"
            "Q or Esc : quit\n"
        )
        dlg = QtWidgets.QMessageBox(self)
        dlg.setWindowTitle("Key bindings")
        dlg.setText(text)
        dlg.exec()
    # Ensure base close behavior
    def closeEvent(self, event):
        super().closeEvent(event)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fits", help="FITS cube path")
    ap.add_argument("--axis", type=int, default=0, help="frame axis (default: 0)")
    ap.add_argument("--lo", type=float, default=1.0, help="low percentile (default: 1)")
    ap.add_argument(
        "--hi", type=float, default=99.0, help="high percentile (default: 99)"
    )
    ap.add_argument(
        "--bins", type=int, default=512, help="histogram bins (unused with HistogramLUTWidget)"
    )
    ap.add_argument("--dpi", type=int, default=300, help="DPI (used only to scale initial window size)")
    ap.add_argument(
        "--size",
        type=str,
        default="12x7",
        help="window size WxH in inches (multiplied by --dpi)",
    )
    args = ap.parse_args()

    # Parse size → pixels
    window_size_px = None
    try:
        w_in, h_in = (float(x) for x in args.size.lower().split("x"))
        window_size_px = (int(w_in * args.dpi), int(h_in * args.dpi))
    except Exception:
        pass

    data = fits.getdata(args.fits)
    if data is None:
        raise SystemExit("No data HDU found.")

    if data.ndim == 2:
        data = data[np.newaxis, ...]
    elif data.ndim == 3:
        if args.axis != 0:
            data = np.moveaxis(data, args.axis, 0)
    else:
        raise SystemExit(f"Unsupported FITS shape: {data.shape} (need 2D or 3D)")

    lo, hi = clamp_percentiles(args.lo, args.hi)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    viewer = FitsViewer(
        data=data,
        file_path=args.fits,
        lo=lo,
        hi=hi,
        aspect_equal=False,
        window_size_px=window_size_px,
    )
    viewer.show()
    run = getattr(app, "exec", None) or getattr(app, "exec_", None)
    if run is None:
        raise SystemExit("Qt application has no exec/exec_ method")
    run()


if __name__ == "__main__":
    main()
