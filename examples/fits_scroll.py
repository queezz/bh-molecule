#!/usr/bin/env python3
"""
fits_scroll.py — FITS cube viewer with keyboard scrolling, aspect toggle, and histogram controls.
Improved layout (constrained), higher DPI text, and clarified axes.

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
import matplotlib.pyplot as plt
from astropy.io import fits


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fits", help="FITS cube path")
    ap.add_argument("--axis", type=int, default=0, help="frame axis (default: 0)")
    ap.add_argument("--lo", type=float, default=1.0, help="low percentile (default: 1)")
    ap.add_argument(
        "--hi", type=float, default=99.0, help="high percentile (default: 99)"
    )
    ap.add_argument(
        "--bins", type=int, default=512, help="histogram bins (default: 512)"
    )
    ap.add_argument("--dpi", type=int, default=300, help="figure DPI (default: 200)")
    ap.add_argument(
        "--size",
        type=str,
        default="12x7",
        help="figure size WxH in inches (default: 12x7)",
    )
    args = ap.parse_args()

    # Parse size
    try:
        w_in, h_in = (float(x) for x in args.size.lower().split("x"))
    except Exception:
        w_in, h_in = 12.0, 7.0

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

    n, h, w = data.shape
    idx = 0
    aspect_equal = False
    lo, hi = clamp_percentiles(args.lo, args.hi)

    # Main image fig — use constrained layout so the title doesn't get clipped; bump DPI to reduce pixelation
    fig = plt.figure(figsize=(w_in, h_in), dpi=args.dpi, constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("X (pixels; wavelength axis)")
    ax.set_ylabel("Y (pixels)")
    im = ax.imshow(
        np.zeros((h, w)),
        origin="lower",
        cmap="gray",
        interpolation="nearest",
        aspect="auto",
    )
    try:
        fig.canvas.manager.set_window_title("FITS Cube Viewer")
    except Exception:
        pass

    # Histogram fig (created lazily)
    hist_fig = None
    hist_ax = None

    def update_hist():
        nonlocal hist_fig, hist_ax
        if hist_fig is None or not plt.fignum_exists(hist_fig.number):
            hist_fig = plt.figure(
                figsize=(8, 4.5), dpi=args.dpi, constrained_layout=True
            )
            hist_ax = hist_fig.add_subplot(1, 1, 1)
            hist_ax.set_title("Histogram")
        else:
            hist_ax.cla()
        img = data[idx].astype(float)
        finite = np.isfinite(img)
        vals = img[finite].ravel()
        if vals.size == 0:
            hist_ax.text(
                0.5,
                0.5,
                "No finite data",
                ha="center",
                va="center",
                transform=hist_ax.transAxes,
            )
        else:
            hist_ax.hist(vals, bins=args.bins)
            vmin, vmax = percentile_limits(img, lo, hi)
            hist_ax.axvline(vmin, linestyle="--")
            hist_ax.axvline(vmax, linestyle="--")
            hist_ax.set_xlabel("Pixel value")
            hist_ax.set_ylabel("Count")
            hist_ax.set_title(f"Histogram  (p{lo:.1f}-{hi:.1f})")
        hist_fig.canvas.draw_idle()

    def title():
        asp = "EQUAL" if aspect_equal else "AUTO"
        ax.set_title(
            f"{os.path.basename(args.fits)}  [frame {idx+1}/{n}]  p{lo:.1f}-{hi:.1f}  AR:{asp}"
        )

    def draw():
        img = data[idx].astype(float)
        vmin, vmax = percentile_limits(img, lo, hi)
        shown = img
        im.set_data(shown)
        im.set_clim(vmin, vmax)
        ax.set_aspect("equal" if aspect_equal else "auto")
        title()
        fig.canvas.draw_idle()
        if hist_fig is not None and plt.fignum_exists(hist_fig.number):
            update_hist()

    def on_key(event):
        nonlocal idx, lo, hi, aspect_equal
        if event.key in ("right", "]"):
            idx = (idx + 1) % n
            draw()
        elif event.key in ("left", "["):
            idx = (idx - 1) % n
            draw()
        elif event.key in ("a", "A"):
            draw()
        elif event.key == ",":
            lo, hi = clamp_percentiles(lo - 1.0, hi)
            draw()
        elif event.key == ".":
            lo, hi = clamp_percentiles(lo + 1.0, hi)
            draw()
        elif event.key == ";":
            lo, hi = clamp_percentiles(lo, hi - 1.0)
            draw()
        elif event.key == "'":
            lo, hi = clamp_percentiles(lo, hi + 1.0)
            draw()
        elif event.key in ("h", "H"):
            if hist_fig is None or not plt.fignum_exists(hist_fig.number):
                update_hist()
            else:
                plt.close(hist_fig)
        elif event.key in ("e", "E"):
            aspect_equal = not aspect_equal
            draw()
        elif event.key in ("s", "S"):
            base = os.path.splitext(args.fits)[0]
            out = f"{base}_frame{idx:04d}.png"
            plt.imsave(
                out,
                im.get_array(),
                origin="lower",
                cmap="gray",
                vmin=im.get_clim()[0],
                vmax=im.get_clim()[1],
            )
            print(f"Saved {out}")
        elif event.key in ("q", "Q", "escape"):
            plt.close(fig)
            if hist_fig is not None and plt.fignum_exists(hist_fig.number):
                plt.close(hist_fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    plt.show()


if __name__ == "__main__":
    main()
