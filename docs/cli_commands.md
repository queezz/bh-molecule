# BH-Molecule CLI Commands

The **BH-Molecule** package provides CLI tools for generating/plotting model spectra and for running BH batch fitting on VIS-1.33 m FITS data.

## Installation

To use the commands, you need to install the package. See the [Installation](index.md#installation) section.

**Batch fitting (FITS data):**

* `bh batch` — run BH batch fitting from a YAML config (single file or folder)

**Model spectrum generation:**

* `bh-spectra`
* `bh-spectra-csv`
* `bh-spectra-plot`

---

## Batch fitting: `bh batch`

Run BH batch fitting on VIS-1.33 m FITS data using a YAML config. Supports a single FITS file or a folder of `.fits` files. Frame and channel selection can be automatic (signal detection) or specified in the config.

```bash
bh batch --config fit_v3.yaml
```

**Options:**

| Option | Description |
|--------|-------------|
| `--config` | **(Required)** Path to YAML config file. |
| `--run-fit-limit N` | Run only the first N fits (after frame/channel selection) for quick pipeline testing. Default: no limit (full batch). |

**Example (limited run for testing):**

```bash
bh batch --config fit_v3.yaml --run-fit-limit 5
```

With `--run-fit-limit 5`, the pipeline runs the same steps (selection, fitting, saving CSV, curves, grid plots, per-fit figures) but only for the first 5 (frame, channel) pairs. Output structure and filenames are unchanged; only fewer fits are executed.

**Config file (YAML):** Must contain either `folder` (path to directory of `.fits` files) or `fits_file` / `fits` (path to a single FITS). Other keys are passed as batch options, e.g. `cw`, `scale`, `out_dir`, `dark_frame`, `time_range`, `background_frames`, `band`, `threshold_sigma`, `bounds`, `fitter_kwargs`, `frames`, `channels`. See [Instruments and calibration](instruments.md) and the batch workflow for details.

---

## Common Parameters

All three commands share a set of parameters controlling the generated spectrum:

| Parameter  | Type  | Default | Description                 |
| ---------- | ----- | ------- | --------------------------- |
| `--xmin`   | float | 432.8   | Minimum wavelength in nm    |
| `--xmax`   | float | 434.2   | Maximum wavelength in nm    |
| `--points` | int   | 4000    | Number of wavelength points |
| `--C`      | float | 1.0     | Concentration factor        |
| `--T_rot`  | float | 2000.0  | Rotational temperature in K |
| `--dx`     | float | 0.0     | Wavelength shift in nm      |
| `--w_inst` | float | 0.02    | Instrumental FWHM in nm     |
| `--base`   | float | 0.0     | Baseline offset             |
| `--I_R7`   | float | 0.5     | Intensity of R₇ branch line |
| `--I_R8`   | float | 0.3     | Intensity of R₈ branch line |

---

## Generate Spectrum Data
### `bh-spectra`

Generates model spectrum and saves it as a compressed NumPy file (`.npz`).

```bash
bh-spectra --C 5.0 --T_rot 3500 --out spectrum.npz
```

**Options:**

* `--out`: Output file name (default: `spectrum.npz`)

**Example:**

```bash
bh-spectra --xmin 433.0 --xmax 434.0 --points 2000 --C 2.0 --T_rot 2500 --out my_spectrum.npz
```

---

### `bh-spectra-csv`

Same as `bh-spectra` but outputs a CSV file instead of NumPy.

```bash
bh-spectra-csv --C 5.0 --T_rot 3500 --out spectrum.csv
```

**Options:**

* `--out`: Output CSV filename (default: `spectrum.csv`)

The resulting CSV will have columns:

* `x` — Wavelength (nm)
* `y` — Intensity (a.u.)

---

## Plot Spectrum

Generates and displays a spectrum plot using Matplotlib.

```bash
bh-spectra-plot --C 5.0 --T_rot 3500
```

**Additional options:**

* `--save`: Save plot to file (e.g. `plot.png`)
* `--dpi`: Resolution when saving (default: 100)
* `--figsize`: Width and height in inches (default: `10 6`)

---

## Example Workflow

1. Generate spectrum data
```bash
bh-spectra --C 4.0 --T_rot 3200 --out data.npz
```

2. Export same parameters to CSV
```bash
bh-spectra-csv --C 4.0 --T_rot 3200 --out data.csv
```

3. Visualize spectrum
```bash
bh-spectra-plot --C 4.0 --T_rot 3200 --save plot.png --dpi 200
```

---

## Notes

* The BH-Spectra CLI tools (`bh-spectra`, `bh-spectra-csv`, `bh-spectra-plot`) are based on the **BHModel** class and use molecular constants loaded via `load_v00_wavelengths()`.
* Adjusting `C`, `T_rot`, and `w_inst` has the most visible effect on spectrum shape.
* The **bh batch** command uses `run_bh_batch` / `run_folder_batch` from the package; `--run-fit-limit` limits the number of fits per run for workflow testing without changing output structure or filenames.
