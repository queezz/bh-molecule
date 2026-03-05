from __future__ import annotations

import argparse
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

from bh_molecule.instruments.wavecal import (
    compute_calibration_from_reference,
    save_bh_wavecal_json,
)

__all__ = [
    "resolve_reference_csv",
    "build_bh_wavecal",
    "compute_calibration_from_reference",
    "save_bh_wavecal_json",
    "main",
]


def resolve_reference_csv(name_or_path: str) -> str:
    """Resolve a BH reference CSV path.

    If *name_or_path* points to an existing file on disk, it is used directly.
    Otherwise, it is resolved inside the installed package resources directory
    ``bh_molecule._resources``.
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


def build_bh_wavecal(
    *,
    csv: str,
    fits: str,
    channel: int = 0,
    degree: int = 2,
    pixel_reference: int | None = None,
    out: str | None = None,
) -> Mapping[str, Any]:
    """Compute and store ``bh_wavecal.json`` (offline wavecal build step).

    This is the package-integrated replacement for the previous standalone
    builder script.

    Parameters
    ----------
    csv:
        Path or filename of the reference wavelength CSV. If it does not exist
        as a filesystem path, it is resolved inside ``bh_molecule._resources``.
    fits:
        Path to a reference FITS cube.
    channel:
        Channel index used for fitting the polynomial (default 0).
    degree:
        Polynomial degree (default 2).
    pixel_reference:
        Optional reference pixel index for polynomial centring.
    out:
        Optional explicit output path for the JSON. When omitted, the JSON is
        written to ``bh_molecule._resources/bh_wavecal.json``.

    Returns
    -------
    Mapping[str, Any]
        The calibration parameters dictionary written to JSON.
    """
    csv_path = resolve_reference_csv(csv)
    fits_path = str(Path(fits).expanduser().resolve())

    params = compute_calibration_from_reference(
        csv_path,
        fits_path,
        channel=int(channel),
        degree=int(degree),
        pixel_reference=pixel_reference,
    )

    if out is not None:
        out_path = str(Path(out).expanduser().resolve())
    else:
        res_dir = resources.files("bh_molecule._resources")
        out_path = str(res_dir.joinpath("bh_wavecal.json"))

    save_bh_wavecal_json(params, path=out_path)
    return params


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="bh-wavecal-build",
        description=(
            "Compute a pixel→wavelength polynomial from a reference CSV + FITS "
            "and store the result as bh_wavecal.json."
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
            "the file is written into bh_molecule/_resources/."
        ),
    )

    args = p.parse_args(argv)
    params = build_bh_wavecal(
        csv=args.csv,
        fits=args.fits,
        channel=int(args.channel),
        degree=int(args.degree),
        pixel_reference=args.pixel_reference,
        out=args.out,
    )
    out_path = args.out or str(resources.files("bh_molecule._resources").joinpath("bh_wavecal.json"))
    print(f"Saved wavelength calibration JSON to: {out_path}")
    print(f"reference_cw_nm={float(params['reference_cw_nm']):.6g} nm")


if __name__ == "__main__":
    main()

