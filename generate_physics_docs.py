#!/usr/bin/env python3
"""Generate docs/physics.md from bh_molecule.physics docstrings without imports."""
import ast
import tokenize
from pathlib import Path
import textwrap
import re


def extract_docs(py_path: Path):
    # Respect the source file's declared encoding (PEP 263)
    with tokenize.open(py_path) as f:
        src = f.read()
    tree = ast.parse(src)
    docs = {}
    class_name = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "BHModel":
            class_name = node.name
            docs[class_name] = ast.get_docstring(node) or ""
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and not item.name.startswith("_"):
                    docs[item.name] = ast.get_docstring(item) or ""
    return docs


def _format_docstring(doc: str) -> str:
    """Dedent docstring text and add blank lines between parameter entries."""
    text = textwrap.dedent(doc).strip("\n")
    lines = text.splitlines()
    out: list[str] = []
    in_params = False
    for line in lines:
        heading = line.startswith("####")
        if heading:
            in_params = line.strip().lower().startswith("#### parameters")
        param_match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)", line) if in_params else None
        if param_match:
            name, rest = param_match.groups()
            line = f"`{name}` : {rest}"
            if out and out[-1].strip() and not out[-1].startswith("####"):
                out.append("")
        out.append(line)
    return "\n".join(out)


def build_markdown(docs: dict) -> str:
    lines = [
        "# Physics Module\n\n"
        "Documentation extracted from docstrings in `bh_molecule.physics`.\n\n"
        "## Molecular Constants\n\n"
        "```python\n"
        "from bh_molecule.physics import MolecularConstants\n\n"
        "constants = MolecularConstants()\n"
        "print(f\"B_e = {constants.B_e:.4f} cm⁻¹\")\n"
        "print(f\"α_e = {constants.α_e:.6f} cm⁻¹\")\n"
        "print(f\"D_e = {constants.D_e:.8f} cm⁻¹\")\n"
        "print(f\"β_e = {constants.β_e:.8f} cm⁻¹\")\n"
        "print(f\"ω_e = {constants.ω_e:.2f} cm⁻¹\")\n"
        "print(f\"ω_e_x_e = {constants.ω_e_x_e:.4f} cm⁻¹\")\n"
        "print(f\"ω_e_y_e = {constants.ω_e_y_e:.6f} cm⁻¹\")\n"
        "print(f\"T_e = {constants.T_e:.2f} cm⁻¹\")\n"
        "```\n\n"
        "## BH Model\n\n"
        "```python\n"
        "from bh_molecule.physics import BHModel\n\n"
        "model = BHModel(v00_wavelengths)\n"
        "```\n\n"
        "## Branch Enum\n\n"
        "```python\n"
        "from bh_molecule.physics import Branch\n\n"
        "print(Branch.P)  # P-branch\n"
        "print(Branch.Q)  # Q-branch\n"
        "print(Branch.R)  # R-branch\n"
        "```\n\n"
        "---\n\n"
        "Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    return "".join(lines)


def main():
    physics_file = (
        Path(__file__).resolve().parent.parent / "src" / "bh_molecule" / "physics.py"
    )
    docs = extract_docs(physics_file)
    out_path = Path(__file__).resolve().parent / "physics.md"
    out_path.write_text(build_markdown(docs), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
