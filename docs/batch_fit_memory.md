# Batch fit memory behavior (long runs)

This page documents the memory-safety work in
`bh_molecule.workflows.batch_fit` and `bh_molecule.plotting.batch_fit_plots`.
It explains the failure mode that long folder runs of
[`run_folder_batch`](batch_fit.md) used to hit, the exact fixes applied, and
how to use the built-in process-memory diagnostics on your own campaigns.

The user-facing API is unchanged: notebooks, the CLI, output filenames,
`summary.csv`, `curves.pkl`, `grid.pdf`, and `frames/*.png` all behave
exactly as before. Only the memory profile of long runs changed.

## Symptom

On long batch runs (`run_folder_batch` over many `.fits` shots) the process
used to crash inside Agg PDF rendering with:

```text
MemoryError: bad allocation
  ...
  fig.tight_layout(rect=[0, 0, 1, 0.97])
  pdf.savefig(fig, bbox_inches="tight", dpi=200)
  ...
  RendererAgg(int(width), int(height), dpi)
```

Single shots, and even small batches, completed fine. The crash only
appeared deep into a campaign-scale run — strongly suggesting cumulative
memory pressure / fragmentation, not one impossible figure.

## Root cause

Three things in `save_batch_fit_grid()` compounded inside the per-shot
folder loop:

1. **`fig.tight_layout(rect=...)`** on a large
    `(frames × channels_per_page)` grid runs an internal full-figure
    rasterization pass just to measure ink bounds.
2. **`pdf.savefig(fig, bbox_inches="tight", dpi=200)`** runs *another*
    full-figure Agg rasterization at 200 dpi only to compute the "tight"
    crop — even though the PDF backend is vector and never uses that
    raster for the output.
3. **No incremental cleanup between shots** — `run_folder_batch` kept the
    previous shot's heavy `curves` dict (numpy `(x, y, yfit)` arrays for
    every `(frame, channel)` pair) alive in the loop's local variable
    until the next iteration reassigned it, and never collected.

Each per-page `RendererAgg(width, height, 200)` allocates a contiguous
RGBA byte buffer of roughly `width * height * 4` bytes. After enough
shots, the Windows allocator can no longer find a contiguous block of
that size and the call fails with `bad allocation` exactly where the
traceback pointed.

## Fix summary

### Plotting (`bh_molecule/plotting/batch_fit_plots.py`)

- Replaced `fig.tight_layout(rect=[0, 0, 1, 0.97])` with a fixed
    `fig.subplots_adjust(...)` layout. The page proportions are preserved
    and there is no extra "measurement" rasterization pass.
- Replaced `pdf.savefig(fig, bbox_inches="tight", dpi=200)` with a plain
    `pdf.savefig(fig)`. PDFs are vector; an explicit DPI just inflates
    the Agg buffer used by `bbox_inches="tight"` and never reaches the
    file. Data points are still rasterized (`rasterized=True`) for
    reasonable PDF size, but at the figure's default DPI.
- Added `gc.collect()` after `plt.close(fig)` so the per-page figure,
    its Agg renderer, and any temporary numpy buffers are released
    before the next page begins.

### Workflow (`bh_molecule/workflows/batch_fit.py`)

- In `run_folder_batch`, after each shot:

    ```python
    results[shot_id] = resb
    del resb, curves
    plt.close("all")
    gc.collect()
    ```

    `curves` is by far the heaviest per-shot object. Dropping it
    eagerly, closing any leftover figures, and forcing a collection
    cycle gives the next shot a clean baseline.

- Lightweight process-memory diagnostics (see below) wrap the
    `save_batch_fit_grid` call and the per-shot boundary, so any future
    regression is visible from the first run instead of only at crash
    time.

The grid PDF is visually equivalent to the old output — only the
backend path changed.

## Memory observations (A/B)

Synthetic stress test (25 sequential grid-PDF saves; 28 channels × 6
frames, 6 channels per page, ~5 pages each, no fitting):

| Variant | Per-iter wall time | RSS plateau (above baseline) | Total |
|---|---|---|---|
| **OLD** (`tight_layout` + `bbox_inches="tight"` + `dpi=200`) | ~10.6 s | ~+32 MB | ~266 s |
| **NEW** (`subplots_adjust` + plain `pdf.savefig`) | ~3.2 s | ~+29 MB | ~80 s |

Both *plateau* in RSS on this small synthetic load (modern allocators
are good), but the old path does **~3.3× more work per page** — and
that's exactly what fragments memory at real campaign scale, where each
shot has hundreds of `(frame, channel)` panels spread over a multi-page
PDF, repeated across dozens of shots.

## Memory diagnostics

When `psutil` is importable, the batch code prints a single
`[MEM] <label>: RSS = X.XX GB` line at the points that matter:

- before / after `save_batch_fit_grid(...)` inside each shot,
- after `pdf.savefig` and after `plt.close(fig) + gc.collect()` for
    each grid page,
- before `run_bh_batch` and at the end of each shot in
    `run_folder_batch` (after the cleanup step above).

If `psutil` is missing, the helper is a silent no-op — no extra
dependency is required.

### Opt-out

Diagnostics are on by default. Silence them via an environment variable
before launching the notebook / CLI:

=== "PowerShell (Windows)"

    ```powershell
    $env:BH_BATCH_MEM_LOG = "0"
    ```

=== "bash / zsh (Linux / macOS)"

    ```bash
    export BH_BATCH_MEM_LOG=0
    ```

Recognized "off" values: `0`, `false`, `no`, `off`, and the empty
string. Any other value (or the variable being unset) keeps the prints
enabled.

### Reading the output

A healthy long run shows:

- RSS climbing during the first one or two shots as matplotlib caches
    populate, then **plateauing**;
- approximately the same RSS at each `end of shot (after cleanup)` line;
- no large jumps that survive the per-shot `del curves + gc.collect()`
    boundary.

If RSS grows monotonically across many shot boundaries, suspect:

- a downstream caller holding `curves` (the lightweight `summary.csv`
    is the canonical persistent artifact — `curves` is intentionally
    dropped after the grid PDF and the `.pkl` are written);
- a custom `fitter_kwargs["preprocess"]` retaining per-shot arrays;
- third-party libraries hooked into matplotlib (custom backends,
    interactive widgets) leaking figures.

## Recommendations

- For very large campaigns where the in-memory `results` dict from
    `run_folder_batch` is not needed, discard the return value
    (`_ = run_folder_batch(...)`). The on-disk `summary.csv` files
    remain the canonical record.
- Keep `save_frames=False` in long headless runs unless the per-fit
    PNG overlays are needed for inspection. Each PNG goes through Agg
    once and is fast, but it adds up at full-campaign scale.
- If you need to regenerate just `grid.pdf` for shots that already
    have `summary.csv` + `curves.pkl`, delete the affected shot
    folder(s) and rerun — the resume logic (skip if `summary.csv`
    exists) will skip everything else and the new memory-safe code
    path will redo only the missing shots.

## See also

- [Batch fitting pipeline](batch_fit.md)
- [Batch fitting workflow (notebooks)](workflow_batch_notebooks.md)
