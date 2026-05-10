import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from .dataio import load_v00_wavelengths
from .physics import BHModel
from .workflows.batch_fit import run_bh_batch, run_folder_batch


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


def main_bh():
    """Entry point for 'bh' CLI with subcommands (e.g. batch)."""
    p = argparse.ArgumentParser(prog="bh")
    sub = p.add_subparsers(dest="command", required=True)

    batch_parser = sub.add_parser("batch", help="Run BH batch fitting from a YAML config")
    batch_parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to YAML config (fits_file or folder, cw, scale, out_dir, etc.)",
    )
    batch_parser.add_argument(
        "--run-fit-limit",
        type=int,
        default=None,
        metavar="N",
        help="Run only the first N fits (after frame/channel selection) for testing",
    )
    args = p.parse_args()

    if args.command == "batch":
        config_path = Path(args.config)
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_path.open() as f:
            config = yaml.safe_load(f)
        if not config:
            raise ValueError("Config file is empty")

        # Build kwargs for run_bh_batch / run_folder_batch.
        # `frames` and `channels` are passed POSITIONALLY below, so they are
        # intentionally NOT included here (passing them both ways would raise
        # ``TypeError: got multiple values for argument 'frames'``).
        kwargs = {}
        for key in (
            "cw",
            "scale",
            "out_dir",
            "dark_frame",
            "time_range",
            "background_frames",
            "band",
            "threshold_sigma",
            "bounds",
            "fitter_kwargs",
            "save_frames",
        ):
            if key in config:
                kwargs[key] = config[key]
        if args.run_fit_limit is not None:
            kwargs["run_fit_limit"] = args.run_fit_limit

        frames_cfg = config.get("frames")
        channels_cfg = config.get("channels")

        if "folder" in config:
            folder = Path(config["folder"]).expanduser()
            run_folder_batch(folder, frames_cfg, channels_cfg, **kwargs)
        elif "fits_file" in config or "fits" in config:
            fits_path = Path(config.get("fits_file", config.get("fits"))).expanduser()
            run_bh_batch(fits_path, frames_cfg, channels_cfg, **kwargs)
        else:
            raise ValueError(
                "Config must contain 'folder' or 'fits_file' (or 'fits') for input path"
            )
