import argparse, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from .dataio import load_v00_wavelengths
from .physics import BHModel


def main():
    p = argparse.ArgumentParser(prog="bh-spectra")
    p.add_argument("--xmin", type=float, default=432.8)
    p.add_argument("--xmax", type=float, default=434.2)
    p.add_argument("--points", type=int, default=4000)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--T_rot", type=float, default=2000.0)
    p.add_argument("--dx", type=float, default=0.0)
    p.add_argument("--w_inst", type=float, default=0.02)
    p.add_argument("--base", type=float, default=0.0)
    p.add_argument("--I_R7", type=float, default=0.01)
    p.add_argument("--I_R8", type=float, default=0.01)
    p.add_argument("--out", default="spectrum.npz")
    args = p.parse_args()

    v00 = load_v00_wavelengths()
    model = BHModel(v00)
    x = np.linspace(args.xmin, args.xmax, args.points)
    y = model.full_fit_model(
        x,
        C=args.C,
        T_rot=args.T_rot,
        dx=args.dx,
        w_inst=args.w_inst,
        base=args.base,
        I_R7=args.I_R7,
        I_R8=args.I_R8,
    )
    np.savez_compressed(args.out, x=x, y=y)
    print(f"Saved: {args.out}")


def main_csv():
    p = argparse.ArgumentParser(prog="bh-spectra-csv")
    p.add_argument("--xmin", type=float, default=432.8)
    p.add_argument("--xmax", type=float, default=434.2)
    p.add_argument("--points", type=int, default=4000)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--T_rot", type=float, default=2000.0)
    p.add_argument("--dx", type=float, default=0.0)
    p.add_argument("--w_inst", type=float, default=0.02)
    p.add_argument("--base", type=float, default=0.0)
    p.add_argument("--I_R7", type=float, default=0.01)
    p.add_argument("--I_R8", type=float, default=0.01)
    p.add_argument("--out", default="spectrum.csv")
    args = p.parse_args()

    v00 = load_v00_wavelengths()
    model = BHModel(v00)
    x = np.linspace(args.xmin, args.xmax, args.points)
    y = model.full_fit_model(
        x,
        C=args.C,
        T_rot=args.T_rot,
        dx=args.dx,
        w_inst=args.w_inst,
        base=args.base,
        I_R7=args.I_R7,
        I_R8=args.I_R8,
    )
    df = pd.DataFrame({"x": x, "y": y})
    df.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")


def main_plot():
    p = argparse.ArgumentParser(prog="bh-spectra-plot")
    p.add_argument("--xmin", type=float, default=432.8)
    p.add_argument("--xmax", type=float, default=434.2)
    p.add_argument("--points", type=int, default=4000)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--T_rot", type=float, default=2000.0)
    p.add_argument("--dx", type=float, default=0.0)
    p.add_argument("--w_inst", type=float, default=0.02)
    p.add_argument("--base", type=float, default=0.0)
    p.add_argument("--I_R7", type=float, default=0.01)
    p.add_argument("--I_R8", type=float, default=0.01)
    p.add_argument("--save", help="Save plot to file (optional)")
    p.add_argument("--dpi", type=int, default=100, help="DPI for saved plot")
    p.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=[10, 6],
        help="Figure size (width height)",
    )
    args = p.parse_args()

    v00 = load_v00_wavelengths()
    model = BHModel(v00)
    x = np.linspace(args.xmin, args.xmax, args.points)
    y = model.full_fit_model(
        x,
        C=args.C,
        T_rot=args.T_rot,
        dx=args.dx,
        w_inst=args.w_inst,
        base=args.base,
        I_R7=args.I_R7,
        I_R8=args.I_R8,
    )

    # Create the plot
    plt.figure(figsize=args.figsize)
    plt.plot(x, y, linewidth=1.5)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity (a.u.)")
    plt.title(f"BH Spectrum - C={args.C}, T_rot={args.T_rot}K, dx={args.dx}")
    plt.grid(True, alpha=0.3)

    # Save if requested
    if args.save:
        plt.savefig(args.save, dpi=args.dpi, bbox_inches="tight")
        print(f"Plot saved: {args.save}")

    plt.show()
    print(
        f"Displayed spectrum with parameters: C={args.C}, T_rot={args.T_rot}K, dx={args.dx}, w_inst={args.w_inst}, base={args.base}, I_R7={args.I_R7}, I_R8={args.I_R8}"
    )
