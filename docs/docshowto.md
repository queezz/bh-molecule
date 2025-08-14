# Docs Build & Preview Cheatsheet (MkDocs + venv)

This project builds the website from the `docs/` folder in CI and deploys it via a GitHub Actions workflow (see `.github/workflows/` in the repo). This page is a quick reminder for local preview/build and common pitfalls.

---

## Quick start

### First-time setup (per machine / per project)

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install mkdocs mkdocs-material pymdown-extensions
```

**macOS / Linux (bash)**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install mkdocs mkdocs-material pymdown-extensions
```

### Daily use

**Windows (PowerShell)**

```powershell
.\.venv\Scripts\Activate.ps1
mkdocs serve    # local preview at http://127.0.0.1:8000
mkdocs build    # writes static site to ./site/
```

**macOS / Linux (bash)**

```bash
source .venv/bin/activate
mkdocs serve
mkdocs build
```

**Network preview (optional)**

```powershell
mkdocs serve -a 0.0.0.0:8000
```

Open from another device on the LAN as `http://<your-pc-ip>:8000`.

---

## venv basics

* **Activate**

  * Windows (PowerShell):

    ```powershell
    .\.venv\Scripts\Activate.ps1
    ```
  * macOS / Linux (bash):

    ```bash
    source .venv/bin/activate
    ```
* **Deactivate**

```bash
deactivate
```

* **If PowerShell blocks activation once**

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

---

## CI (what the workflow does)

1. Spins up a clean Ubuntu VM.
2. Checks out this repo.
3. Installs MkDocs + plugins.
4. Runs `mkdocs build` to produce `site/`.
5. Commits `site/` to the `gh-pages` branch so GitHub Pages can serve it.

> You can view logs under the **Actions** tab on GitHub.

---

## Reproducible builds (optional pinning)

To reduce surprises from upstream updates, pin versions:

```bash
pip install mkdocs==1.6.1 mkdocs-material==9.6.16 pymdown-extensions==10.16.1
```

In CI, install the same pinned versions before `mkdocs build`.

---

## Common errors

* **`mkdocs: command not found` / `mkdocs is not recognized`** → venv not active. Activate the venv, or run `python -m mkdocs serve`.
* **Port already in use** → `mkdocs serve -a 127.0.0.1:8001`.
* **Plugin not found** → `pip install <missing-plugin>` in the venv and ensure it’s spelled correctly in `mkdocs.yml`.
* **YAML error** → check indentation and list syntax in `mkdocs.yml`.
* **Pages don’t update** → ensure repo Settings → Pages serves the `gh-pages` branch.

---

## Git hygiene

* Add to `.gitignore`:

  ```gitignore
  .venv/
  site/
  ```

  The CI runs `mkdocs build` for deployment; you don’t need to commit `site/` locally.
