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
        self.plot = self.glw.addPlot(row=0, col=0)
        self.splitter.addWidget(self.glw)
        self.plot.setLabel("bottom", "X (pixels; wavelength axis)")
        self.plot.setLabel("left", "Y (pixels)")

        # ViewBox and ImageItem
        self.view_box = self.plot.getViewBox()
        self.view_box.invertY(True)  # origin at lower-left like matplotlib origin="lower"
        self.view_box.setAspectLocked(self.aspect_equal)
        # Disable automatic range updates by default so overlays don't trigger zoom
        try:
            self.view_box.enableAutoRange(False)
        except Exception:
            pass
        # Track whether the user has manually changed the view (pan/zoom).
        # We avoid overriding a user-set view when updating the image.
        self.user_set_view = False
        self._programmatic_range = False
        try:
            # sigRangeChanged exists on ViewBox and is emitted when the view is panned/zoomed
            self.view_box.sigRangeChanged.connect(lambda *args: self._on_range_changed())
        except Exception:
            pass

        self.image_item = pg.ImageItem()
        self.plot.addItem(self.image_item)

        # Below: row profile plot (intensity vs X for cursor row)
        self.row_plot = self.glw.addPlot(row=1, col=0)
        self.row_plot.setLabel("bottom", "X (pixels)")
        self.row_plot.setLabel("left", "Value")
        self.row_curve = self.row_plot.plot(pen=pg.mkPen('c'))
        self.row_xline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('m'))
        self.row_plot.addItem(self.row_xline, ignoreBounds=True)
        # Track last cursor for row plot
        self.last_cursor_y = max(0, min(self.height - 1, self.height // 2))
        self.last_cursor_x = max(0, min(self.width - 1, self.width // 2))

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

        # Crosshair and value overlay
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y'))
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('y'))
        self.plot.addItem(self.vline, ignoreBounds=True)
        self.plot.addItem(self.hline, ignoreBounds=True)
        self.text_item = pg.TextItem('', anchor=(0,1), border='w', fill=pg.mkBrush(0,0,0,150))
        self.text_item.setZValue(2)
        # Add text overlay without affecting view bounds to prevent auto-zoom
        self.plot.addItem(self.text_item, ignoreBounds=True)

        # Connect mouse move on the ViewBox scene to update crosshair
        try:
            self.view_box.scene().sigMouseMoved.connect(self._on_mouse_move)
        except Exception:
            # Some bindings may expose different signal names; ignore if unavailable
            pass

        # Grayscale colormap
        try:
            cmap = pg.colormap.get("gray")
            lut = cmap.getLookupTable(0.0, 1.0, 256)
            self.image_item.setLookupTable(lut)
        except Exception:
            pass

        # Initial draw and window size
        self._update_image()
        # Initialize row profile after image is ready
        try:
            self._update_row_plot(self.last_cursor_y, self.last_cursor_x)
        except Exception:
            pass
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
        # Crosshair state (toggle via key binding 'C')
        self.crosshair_enabled = True
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
        # Set the view range to image pixel extents explicitly unless the user has
        # manually changed the view (panned/zoomed). Use a small programmatic guard
        # so the sigRangeChanged handler doesn't mark programmatic changes as user changes.
        if not getattr(self, 'user_set_view', False):
            try:
                self._programmatic_range = True
                self.view_box.setRange(xRange=(0, self.width), yRange=(0, self.height), padding=0)
                try:
                    # ensure auto-range remains disabled after programmatic set
                    self.view_box.enableAutoRange(False)
                except Exception:
                    pass
            except Exception:
                try:
                    self.plot.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
                except Exception:
                    pass
            finally:
                self._programmatic_range = False
        self._update_title()
        # HistogramLUTWidget updates itself when image levels change
        # Nothing else needed here
        # Refresh row profile for current frame at last known row
        try:
            self._update_row_plot(self.last_cursor_y, self.last_cursor_x)
        except Exception:
            pass

    def _on_mouse_move(self, pos):
        """Update crosshair and show pixel value under cursor.

        `pos` is a QPointF in scene coordinates from sigMouseMoved.
        """
        try:
            vb = self.view_box
            if vb is None:
                return
            mouse_point = vb.mapSceneToView(pos)
            # Map to pixel centers: pixels are centered at half-integer positions
            x = int(np.floor(mouse_point.x() + 0.5))
            y = int(np.floor(mouse_point.y() + 0.5))
            # clamp
            if x < 0 or y < 0 or x >= self.width or y >= self.height:
                self.text_item.setText('')
                return
            try:
                val = float(self.data[self.frame_index, y, x])
            except Exception:
                val = float('nan')
            if self.crosshair_enabled:
                # place lines through pixel center
                self.vline.setPos(x + 0.5)
                self.hline.setPos(y + 0.5)
            # small HTML-ish text for clarity
            try:
                self.text_item.setHtml(f"<div style='color: white;'>x={x} y={y}<br>val={val:.4g}</div>")
            except Exception:
                # fallback
                self.text_item.setText(f"x={x} y={y} val={val:.4g}")
            # place text near top-left of current view range
            vr = vb.viewRange()
            x_min = vr[0][0]
            y_max = vr[1][1]
            self.text_item.setPos(x_min, y_max)
            # Update row profile plot only if crosshair is enabled
            if self.crosshair_enabled:
                if y != self.last_cursor_y or x != self.last_cursor_x:
                    self.last_cursor_y = y
                    self.last_cursor_x = x
                    self._update_row_plot(y, x)
        except Exception:
            # be robust to any binding differences
            return

    def _on_range_changed(self):
        # Ignore programmatic changes triggered by setRange
        if getattr(self, '_programmatic_range', False):
            return
        self.user_set_view = True

    def _update_row_plot(self, row_index: int, x_index: int | None = None):
        """Update the row profile plot for the given row and optional x marker."""
        if row_index < 0 or row_index >= self.height:
            return
        row = self.data[self.frame_index, row_index, :].astype(float)
        xs = np.arange(self.width)
        # Set data
        self.row_curve.setData(xs, row)
        # X range fixed to image width
        try:
            self.row_plot.setXRange(0, self.width, padding=0)
        except Exception:
            pass
        # Y range via percentile limits using current lo/hi
        vmin, vmax = percentile_limits(row, self.lo, self.hi)
        # Fallback if degenerate
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = np.nanmin(row), np.nanmax(row)
            if vmin == vmax:
                vmax = vmin + 1.0
        try:
            self.row_plot.setYRange(vmin, vmax, padding=0)
        except Exception:
            pass
        # Title and marker
        try:
            self.row_plot.setTitle(f"Row {row_index}")
        except Exception:
            pass
        if x_index is None:
            x_index = self.last_cursor_x
        try:
            # place row plot x marker at pixel center
            self.row_xline.setPos(int(x_index) + 0.5)
        except Exception:
            pass

    def _set_crosshair_enabled(self, enabled: bool):
        self.crosshair_enabled = bool(enabled)
        try:
            self.vline.setVisible(self.crosshair_enabled)
            self.hline.setVisible(self.crosshair_enabled)
        except Exception:
            pass

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
        elif key == QtCore.Qt.Key.Key_C:
            # Toggle crosshair with 'C'
            self._set_crosshair_enabled(not self.crosshair_enabled)
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
        html = """
        <html>
          <head>
            <style>
              body { font-family: sans-serif; }
              table { border-collapse: collapse; width: 100%; }
              th, td { text-align: left; padding: 8px; }
              tr:nth-child(even){ background-color: #f2f2f2 }
              th { background-color: #4CAF50; color: white; }
            </style>
          </head>
          <body>
            <h2>FITS Viewer — Key bindings</h2>
            <table>
              <tr><th>Key</th><th>Action</th></tr>
              <tr><td>Left / Right or [ / ]</td><td>Previous / Next frame</td></tr>
              <tr><td>A</td><td>Recompute percentiles (auto-rescale)</td></tr>
              <tr><td>, / .</td><td>Low percentile -1 / +1</td></tr>
              <tr><td>; / '</td><td>High percentile -1 / +1</td></tr>
              <tr><td>H</td><td>Toggle histogram panel</td></tr>
              <tr><td>E</td><td>Toggle aspect (equal / auto)</td></tr>
              <tr><td>C</td><td>Cursor on/off</td></tr>
              <tr><td>S</td><td>Save current frame as PNG</td></tr>
              <tr><td>Q or Esc</td><td>Quit</td></tr>
            </table>
            <p>Tip: drag in the histogram panel to adjust levels interactively.</p>
          </body>
        </html>
        """

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Key bindings")
        layout = QtWidgets.QVBoxLayout(dlg)
        browser = QtWidgets.QTextBrowser()
        browser.setHtml(html)
        browser.setOpenExternalLinks(True)
        layout.addWidget(browser)
        # Use a plain Close button for maximum compatibility across Qt bindings
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(dlg.reject)
        btn_container = QtWidgets.QWidget()
        btn_layout = QtWidgets.QHBoxLayout(btn_container)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addWidget(btn_container)
        # Increase minimum height so the keybindings page doesn't need scrolling
        browser.setMinimumHeight(380)
        dlg.resize(500, 400)
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
