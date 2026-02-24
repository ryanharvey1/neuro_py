"""Pre-build script for Zensical documentation.

Replaces the functionality of mkdocs-gen-files, mkdocs-literate-nav,
mkdocs-include-markdown, and mkdocs-jupyter with direct filesystem operations.

Run this before `zensical build`:

    python docs/prebuild.py && zensical build

Generated outputs:
    docs/reference/   - API reference markdown pages (from neuro_py source)
    docs/tutorials/   - Tutorial markdown files (converted from .ipynb)
    docs/index.md     - Homepage (copied from README.md)
    mkdocs.yml        - Nav section updated with full reference tree
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
SRC_DIR = ROOT / "neuro_py"
TUTORIALS_DIR = ROOT / "tutorials"


def generate_reference() -> list:
    """Generate API reference markdown pages.

    Replicates the logic in docs/gen_ref_pages.py but writes directly to the
    filesystem instead of the mkdocs_gen_files virtual filesystem.

    Returns
    -------
    list
        Nav structure for the reference section, suitable for mkdocs.yml.
    """
    ref_dir = DOCS_DIR / "reference"

    # Clean previous generated reference
    if ref_dir.exists():
        shutil.rmtree(ref_dir)
    ref_dir.mkdir(parents=True, exist_ok=True)

    # Collect nav entries grouped by subpackage
    # Structure: {subpackage: [(title, doc_path), ...]}
    nav_tree: dict[str, list[dict[str, str]]] = OrderedDict()

    for path in sorted(SRC_DIR.rglob("*.py")):
        rel = path.relative_to(ROOT)
        parts = rel.with_suffix("").parts  # e.g. ('neuro_py', 'process', 'peri_event')

        # Skip test directories and files
        if any(part.startswith("test") for part in parts):
            continue
        if "tests" in parts:
            continue

        module_parts = parts

        if parts[-1] == "__init__":
            module_parts = parts[:-1]
            doc_path = ref_dir / Path(*parts[:-1]) / "index.md"
        elif parts[-1].startswith("_"):
            continue
        else:
            doc_path = ref_dir / Path(*parts).with_suffix(".md")

        doc_path.parent.mkdir(parents=True, exist_ok=True)

        identifier = ".".join(module_parts)
        doc_path.write_text(
            f"---\ntitle: {identifier}\n---\n\n::: {identifier}\n"
        )

        # Build nav entry (path relative to docs/)
        rel_doc = doc_path.relative_to(DOCS_DIR).as_posix()

        if len(module_parts) == 1:
            # Top-level neuro_py package index
            pass
        elif len(module_parts) == 2:
            # Subpackage index (e.g. neuro_py.process)
            subpkg = module_parts[1]
            if subpkg not in nav_tree:
                nav_tree[subpkg] = []
            # Insert index as first entry
            nav_tree[subpkg].insert(0, {subpkg: rel_doc})
        else:
            # Module page (e.g. neuro_py.process.peri_event)
            subpkg = module_parts[1]
            mod_name = module_parts[-1]
            if subpkg not in nav_tree:
                nav_tree[subpkg] = []
            nav_tree[subpkg].append({mod_name: rel_doc})

    # Create top-level reference index
    index_md = ref_dir / "index.md"
    index_md.write_text("---\ntitle: API Reference\n---\n\n# API Reference\n")

    # Build the nav list for reference section
    ref_nav: list = [{"Overview": "reference/index.md"}]
    for subpkg, entries in nav_tree.items():
        ref_nav.append({subpkg: entries})

    print(f"  Generated {sum(1 for _ in ref_dir.rglob('*.md'))} reference pages")
    return ref_nav


def copy_and_convert_tutorials() -> None:
    """Copy tutorials and convert Jupyter notebooks to Markdown.

    Replicates docs/copy_tutorials.py (mkdocs_gen_files) + mkdocs-jupyter by
    converting .ipynb files to Markdown via ``jupyter nbconvert``.
    """
    dest = DOCS_DIR / "tutorials"

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    if not TUTORIALS_DIR.exists():
        print("  No tutorials/ directory found, skipping")
        return

    banned_dirs = {"cache", "files", "example_files", "__pycache__", "lightning_logs"}
    banned_exts = {".pbf", ".parquet", ".json", ".geojson", ".pt"}

    for src_path in sorted(TUTORIALS_DIR.glob("**/*")):
        if not src_path.is_file():
            continue
        if any(d in src_path.parts for d in banned_dirs):
            continue
        if src_path.suffix in banned_exts:
            continue

        rel = src_path.relative_to(TUTORIALS_DIR)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)

        if src_path.suffix == ".ipynb":
            # Convert notebook to Markdown via nbconvert
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "nbconvert",
                    "--to",
                    "markdown",
                    "--output-dir",
                    str(target.parent),
                    str(src_path),
                ],
                check=True,
            )
            md_name = src_path.stem + ".md"
            print(f"  Converted {rel} -> tutorials/{md_name}")
        else:
            shutil.copy2(src_path, target)

    total = sum(1 for _ in dest.rglob("*.md"))
    print(f"  {total} tutorial pages ready")


def generate_index() -> None:
    """Generate docs/index.md from README.md.

    Replaces the mkdocs-include-markdown plugin directive.
    """
    readme = ROOT / "README.md"
    index = DOCS_DIR / "index.md"

    if not readme.exists():
        print("  WARNING: README.md not found, skipping index generation")
        return

    content = readme.read_text()
    index.write_text(content)
    print("  Generated index.md from README.md")


def update_nav(ref_nav: list) -> None:
    """Update mkdocs.yml nav section with generated reference structure.

    Parameters
    ----------
    ref_nav : list
        Nav structure for the API Reference section.
    """
    config_path = ROOT / "mkdocs.yml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Build tutorial nav entries
    tutorial_nav = []
    tutorials_dir = DOCS_DIR / "tutorials"
    if tutorials_dir.exists():
        for md_file in sorted(tutorials_dir.glob("*.md")):
            # Convert filename to title: peth_tutorial -> Peth Tutorial
            title = md_file.stem.replace("_", " ").title()
            tutorial_nav.append({title: f"tutorials/{md_file.name}"})

    # Reconstruct nav
    config["nav"] = [
        {"Home": "index.md"},
        {"API Reference": ref_nav},
        {"Tutorials": tutorial_nav},
    ]

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  Updated nav: {len(ref_nav)} reference entries, {len(tutorial_nav)} tutorials")


def main() -> None:
    print("=== Zensical pre-build ===")
    print("Generating API reference pages...")
    ref_nav = generate_reference()
    print("Converting tutorials...")
    copy_and_convert_tutorials()
    print("Generating index page...")
    generate_index()
    print("Updating mkdocs.yml nav...")
    update_nav(ref_nav)
    print("=== Pre-build complete ===")


if __name__ == "__main__":
    main()
