# Using a venv **outside** a synced folder (Dropbox/OneDrive/Google Drive)

If you keep your repo inside a synced folder, excluding `.venv/` can be unreliable or annoying. Easiest fix: **create the virtual environment outside the synced tree** and point your tools to it.

This page shows how.

---

## TL;DR

* Put venvs in a local directory (e.g. `~/.venvs` or `C:\Users\<you>\.venvs`).
* Create/activate the venv there.
* Tell Cursor/VS Code to use that interpreter.
* Keep `.venv/` in `.gitignore` in case someone creates one inside the repo.

---

## Create & activate the venv

### Windows (PowerShell)
**1) Pick a home for all your venvs**

```powershell
$ENV_DIR = "$env:USERPROFILE\.venvs"
New-Item -ItemType Directory -Force $ENV_DIR | Out-Null
```

**2) Create the venv (choose ONE of these)**

Option A — with the Python launcher:

```powershell
py -3.12 -m venv "$ENV_DIR\bh-molecule"
```

Option B — with `python`:

```powershell
python -m venv "$ENV_DIR\bh-molecule"
```

**3) Activate**

```powershell
& "$ENV_DIR\bh-molecule\Scripts\Activate.ps1"
```

**4) Upgrade pip**

```powershell
python -m pip install -U pip
```

**5) Install this project (dev extras)**

```powershell
python -m pip install -e ".[dev]"
```

> If activation is blocked on Windows, run once:
>
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```


### macOS / Linux

```bash
ENV_DIR="$HOME/.venvs"
mkdir -p "$ENV_DIR"

python3 -m venv "$ENV_DIR/bh-molecule"
source "$ENV_DIR/bh-molecule/bin/activate"

python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Deactivate any time:

```bash
deactivate
```

---

## Point IDE to the external venv

1. **Ctrl+Shift+P → “Python: Select Interpreter” → “Enter interpreter path”**

   * Windows:
     `C:\Users\<you>\.venvs\bh-molecule\Scripts\python.exe`
   * macOS/Linux:
     `/Users/<you>/.venvs/bh-molecule/bin/python`

2. **Persist for this workspace** (recommended):

   Create `.vscode/settings.json`:

   **Windows**

   ```json
   {
     "python.defaultInterpreterPath": "${env:USERPROFILE}\\.venvs\\bh-molecule\\Scripts\\python.exe",
     "python.terminal.activateEnvironment": true
   }
   ```

   **macOS/Linux**

   ```json
   {
     "python.defaultInterpreterPath": "${env:HOME}/.venvs/bh-molecule/bin/python",
     "python.terminal.activateEnvironment": true
   }
   ```

(Optional) Make external venvs auto-discoverable in the picker:

* Windows (User Settings): `"python.venvPath": "C:\\Users\\<you>\\.venvs"`
* macOS/Linux (User Settings): `"python.venvPath": "~/.venvs"`

---

## Jupyter notebooks in Cursor

* Kernel picker → **Select Another Kernel** → **Python Environments** → choose the same venv,
  or run **“Python: Select Interpreter to start Jupyter server”** and pick it.

---

## Common tasks

Install/upgrade project deps:

```bash
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Add a package:

```bash
python -m pip install <package>
# Then record it in pyproject (e.g., optional-dependencies.dev) if needed.
```

Freeze (for debugging):

```bash
python -m pip freeze > freeze.txt
```

---

## Recreate the environment

If things drift, rebuild quickly:

**Windows**

```powershell
Remove-Item -Recurse -Force "$env:USERPROFILE\.venvs\bh-molecule"
python -m venv "$env:USERPROFILE\.venvs\bh-molecule"
& "$env:USERPROFILE\.venvs\bh-molecule\Scripts\Activate.ps1"
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

**macOS / Linux**

```bash
rm -rf "$HOME/.venvs/bh-molecule"
python3 -m venv "$HOME/.venvs/bh-molecule"
source "$HOME/.venvs/bh-molecule/bin/activate"
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

---

## Notes & gotchas

* **Synced folders:** keeping venvs outside avoids sync noise and conflicts (Dropbox/OneDrive/Drive).
* **.gitignore:** add `.venv/` so a local in-repo venv never gets committed:

  ```
  .venv/
  ```
* **Windows execution policy:** if activation is blocked, run once:

  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
  ```
* **Multiple machines:** each machine should have its own venv; reinstall with `-e ".[dev]"`.
