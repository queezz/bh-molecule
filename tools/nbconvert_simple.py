import os
import json
import base64
from pathlib import Path

NOTEBOOKS = [
    "examples/02_quickstart.ipynb",
    "examples/03_load_spectral_data.ipynb",
    "examples/04_bh_fit.ipynb",
]
OUT_DIR = Path("docs/examples")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for nb_path in NOTEBOOKS:
    nb = json.load(open(nb_path, "r", encoding="utf-8"))
    stem = Path(nb_path).stem
    out_md = OUT_DIR / f"{stem}.md"
    lines = []
    title = f"### {stem} (from {nb_path})"
    lines.append(title)
    img_counter = 0

    for cell in nb.get("cells", []):
        ctype = cell.get("cell_type", "")
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        if ctype == "markdown":
            lines.append(src)
        elif ctype == "code":
            lines.append("```python")
            lines.append(src)
            lines.append("```")
            # outputs
            for out in cell.get("outputs", []):
                data = out.get("data", {})
                if "image/png" in data:
                    b64 = data["image/png"]
                    if isinstance(b64, list):
                        b64 = "".join(b64)
                    img_bytes = base64.b64decode(b64)
                    img_name = f"{stem}_out_{img_counter}.png"
                    img_path = OUT_DIR / img_name
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)
                    lines.append(f"![output image]({img_name})")
                    img_counter += 1
                elif "image/jpeg" in data:
                    b64 = data["image/jpeg"]
                    if isinstance(b64, list):
                        b64 = "".join(b64)
                    img_bytes = base64.b64decode(b64)
                    img_name = f"{stem}_out_{img_counter}.jpg"
                    img_path = OUT_DIR / img_name
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)
                    lines.append(f"![output image]({img_name})")
                    img_counter += 1
                else:
                    # handle text outputs
                    text = out.get("text", "")
                    if isinstance(text, list):
                        text = "".join(text)
                    if text:
                        lines.append("```")
                        lines.append(text)
                        lines.append("```")

    out_md.write_text("\n\n".join(lines), encoding="utf-8")
print("Converted notebooks to docs/examples")
