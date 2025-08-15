# BH-Molecule CLI Commands

The **BH-Molecule** package provides three main command-line tools for generating, exporting, and plotting model spectra of the boron hydride (BH) A–X band.

## Installation

To use the commands, you need to install the package. See the [Installation](index.md#installation) section.

* `bh-spectra`
* `bh-spectra-csv`
* `bh-spectra-plot`

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

* The BH-Spectra CLI tools are based on the **BHModel** class and use molecular constants loaded via `load_v00_wavelengths()`.
* Adjusting `C`, `T_rot`, and `w_inst` has the most visible effect on spectrum shape.
