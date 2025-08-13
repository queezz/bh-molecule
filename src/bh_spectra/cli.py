import argparse, numpy as np
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
    p.add_argument("--I_R7", type=float, default=0.5)
    p.add_argument("--I_R8", type=float, default=0.3)
    p.add_argument("--out", default="spectrum.npz")
    args = p.parse_args()

    v00 = load_v00_wavelengths()
    model = BHModel(v00)
    x = np.linspace(args.xmin, args.xmax, args.points)
    y = model.full_fit_model(x, C=args.C, T_rot=args.T_rot, dx=args.dx,
                             w_inst=args.w_inst, base=args.base,
                             I_R7=args.I_R7, I_R8=args.I_R8)
    np.savez_compressed(args.out, x=x, y=y)
    print(f"Saved: {args.out}")
