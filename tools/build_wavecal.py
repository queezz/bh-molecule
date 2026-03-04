from __future__ import annotations

import argparse
from pathlib import Path

from importlib import resources

from bh_molecule.instruments.wavecal import (
    compute_calibration_from_reference,
    save_bh_wavecal_json,
)


def _resolve_csv(name_or_path: str) -> str:
    """Resolve a CSV path.

    If *name_or_path* points to an existing file on disk, it is used
    directly. Otherwise, it is looked up inside the ``bh_molecule._resources``
    package directory (useful for bundled reference spectra).
    """
    p = Path(name_or_path)
    if p.is_file():
        return str(p)

    res_dir = resources.files("bh_molecule._resources")
    candidate = res_dir.joinpath(name_or_path)
    if candidate.is_file():
        return str(candidate)

    raise FileNotFoundError(
        f"Could not find CSV '{name_or_path}' as a filesystem path or inside "
        "bh_molecule._resources"
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="build-wavecal",
        description=(
            "Compute a pixel→wavelength polynomial from a reference CSV + FITS "
            "and store the result as bh_wavecal.json in src/bh_molecule/_resources/."
        ),
    )
    p.add_argument(
        "--csv",
        required=True,
        help=(
            "Path or filename of the reference wavelength CSV. If it does not "
            "exist as a filesystem path, it is resolved relative to the "
            "bh_molecule._resources package."
        ),
    )
    p.add_argument(
        "--fits",
        required=True,
        help="Path to the reference FITS cube used for calibration.",
    )
    p.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Channel index to use when fitting the polynomial (default: 0).",
    )
    p.add_argument(
        "--degree",
        type=int,
        default=2,
        help="Polynomial degree for pixel→wavelength mapping (default: 2).",
    )
    p.add_argument(
        "--pixel-reference",
        type=int,
        default=None,
        help=(
            "Reference pixel index used to define the polynomial in terms of "
            "(pixel - pixel_reference). Defaults to the centre of the CSV "
            "pixel range."
        ),
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Optional explicit output path for bh_wavecal.json. When omitted, "
            "the file is written into src/bh_molecule/_resources/ inside the "
            "repository / installed package."
        ),
    )

    args = p.parse_args(argv)

    csv_path = _resolve_csv(args.csv)
    fits_path = str(Path(args.fits).expanduser().resolve())

    params = compute_calibration_from_reference(
        csv_path,
        fits_path,
        channel=int(args.channel),
        degree=int(args.degree),
        pixel_reference=args.pixel_reference,
    )

    if args.out is not None:
        out_path = str(Path(args.out).expanduser().resolve())
    else:
        # Default to the source tree / installed package resource location.
        res_dir = resources.files("bh_molecule._resources")
        out_path = str(res_dir.joinpath("bh_wavecal.json"))

    save_bh_wavecal_json(params, path=out_path)
    print(f"Saved wavelength calibration JSON to: {out_path}")


if __name__ == "__main__":
    main()

