# Dev Environment & MkDocs

This page is the single source of truth for setting up a local development environment for `bh-molecule` and running the docs site.

---

## Part A — Dev Environment

How to prepare your venv, install the package in editable mode, and set up Jupyter.

### A1. Prerequisites
Python 3.10+, Git, and VS Code (recommended) or your preferred editor.  
> We isolate everything in `.venv/`.

### A2. Create & activate venv
**Windows (PowerShell)**
```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -U pip
```

**macOS / Linux (bash)**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

> If blocked on Windows: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
 

### A3. Editable install

From repo root:

```bash
python -m pip install -e ".[dev]"
```

Imports will work anywhere, and file edits are picked up automatically.

### A4. Jupyter setup

Optional, for running notebooks in `examples/`:

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name=bh-molecule --display-name="bh-molecule (venv)"
```

In VS Code, pick the **bh-molecule (venv)** kernel.

### A5. Git ignore

Add to `.gitignore` if missing:

```
.venv/
site/
.ipynb_checkpoints/
```

---

## Part B — MkDocs

Steps to serve or build the docs site locally.

### B1. Install MkDocs

If `.[dev]` already installed docs deps, skip this. Otherwise:

```bash
python -m pip install mkdocs mkdocs-material pymdown-extensions
```

### B2. Local preview

```bash
mkdocs serve
# or preview on LAN:
mkdocs serve -a 0.0.0.0:8000
```

### B3. Build site

```bash
mkdocs build
```

Outputs to `./site/` — do **not** commit this.

### B4. Optional pinned versions

```bash
python -m pip install \
  mkdocs==1.6.1 mkdocs-material==9.6.16 pymdown-extensions==10.16.1
```

---

## Troubleshooting

- **MkDocs not found** → Activate `venv`, or use `python -m mkdocs serve`
- **ImportError** → Check `pip install -e .` was run in this `venv`, and correct kernel selected.
- **Plugin missing** → Install in `venv` and update `mkdocs.yml`.
- **Port in use** → Change port: `mkdocs serve -a 127.0.0.1:8001`.

---
### `venv` update

If you need to recreate the venv and Windows/VS Code won’t let you delete it:
1) Switch VS Code to a different interpreter (e.g., Conda base).  
2) Reload or close VS Code.  
3) Delete/recreate `.venv`.  
4) Switch back to the project interpreter.

> **Heads-up for synced folders (Dropbox/OneDrive/Drive):**  
> If your repo lives in a synced folder and you use multiple machines, avoid putting `.venv/` inside the repo. Keep the venv **outside** the synced tree and point Cursor/VS Code to it.  
> See: [Using a venv outside a synced folder](./venv-outside-synced-folders.md)

(Add `.venv/` to `.gitignore` so a local in-repo venv won’t be committed.)



---

## Quick daily commands

**Windows**

```powershell
.\\.venv\\Scripts\\Activate.ps1
mkdocs serve
```

**macOS / Linux**

```bash
source .venv/bin/activate
mkdocs serve
```