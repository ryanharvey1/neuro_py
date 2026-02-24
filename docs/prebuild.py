"""Pre-build script for Zensical documentation.

Replaces the functionality of mkdocs-gen-files, mkdocs-literate-nav,
mkdocs-include-markdown, and mkdocs-jupyter with direct filesystem operations.

Run this before `zensical build`:

    python docs/prebuild.py && zensical build

Generated outputs:
    docs/reference/   - API reference markdown pages (from neuro_py source)
    docs/tutorials/   - Tutorial markdown files (converted from .ipynb)
    docs/index.md     - Homepage (synthesized from selected README.md sections and homepage.yml)
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


def _load_homepage_meta() -> dict:
    """Load docs/homepage.yml with module/tutorial/dep metadata."""
    meta_path = DOCS_DIR / "homepage.yml"
    return yaml.safe_load(meta_path.read_text()) if meta_path.exists() else {}


def _load_pyproject() -> dict:
    """Parse pyproject.toml and return the ``[project]`` table."""
    toml_path = ROOT / "pyproject.toml"
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        try:
            import tomllib  # type: ignore[import]
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[import,no-redef]
    return tomllib.loads(toml_path.read_text())["project"]


def _discover_subpackages() -> list[str]:
    """Return sorted list of neuro_py subpackage directory names."""
    skip = {"__pycache__", "util"}
    return sorted(
        d.name
        for d in SRC_DIR.iterdir()
        if d.is_dir() and d.name not in skip and not d.name.startswith(".")
    )


def _discover_tutorials() -> list[str]:
    """Return sorted list of tutorial stems (no extension)."""
    if not TUTORIALS_DIR.exists():
        return []
    return sorted(nb.stem for nb in TUTORIALS_DIR.glob("*.ipynb"))


def _extract_repo_info(proj: dict) -> dict:
    """Derive GitHub owner/repo and PyPI package name from pyproject.toml."""
    import re as _re

    homepage = proj.get("urls", {}).get("Homepage", "")
    m = _re.search(r"github\.com/([^/]+)/([^/]+)", homepage)
    owner = m.group(1) if m else "ryanharvey1"
    repo = m.group(2).rstrip("/") if m else "neuro_py"
    pypi_name = proj.get("name", "neuro-analysis-py")
    python_req = proj.get("requires-python", ">=3.9")
    description = proj.get("description", "")
    return {
        "owner": owner,
        "repo": repo,
        "pypi": pypi_name,
        "python": python_req,
        "description": description,
        "homepage": homepage,
    }


# Display-name overrides for subpackage card titles.
_MODULE_TITLES: dict[str, str] = {
    "io": "I/O",
    "lfp": "LFP",
}


def _parse_readme_sections() -> dict[str, str]:
    """Extract named ``##`` sections from README.md.

    Returns a dict mapping lowercase section titles (e.g. ``"installation"``,
    ``"usage"``) to the raw markdown body **between** that heading and the
    next ``##`` heading (or EOF).
    """
    import re as _re

    readme = ROOT / "README.md"
    if not readme.exists():
        return {}
    text = readme.read_text()
    # Split on ## headings, keeping the heading text
    parts = _re.split(r"^## +(.+)$", text, flags=_re.MULTILINE)
    # parts = [preamble, heading1, body1, heading2, body2, ...]
    sections: dict[str, str] = {}
    for i in range(1, len(parts) - 1, 2):
        key = parts[i].strip().lower()
        body = parts[i + 1].strip()
        sections[key] = body
    return sections


def generate_index() -> None:
    """Generate ``docs/index.md`` dynamically from project files and metadata.

    Every prebuild discovers:
    * Subpackages under ``neuro_py/``
    * Tutorial notebooks under ``tutorials/``
    * Version, authors, dependencies, URLs from ``pyproject.toml``
    * Icons, descriptions, and tutorial titles from ``docs/homepage.yml``
    """
    meta = _load_homepage_meta()
    proj = _load_pyproject()
    info = _extract_repo_info(proj)
    mod_meta = meta.get("modules", {})
    tut_meta = meta.get("tutorials", {})
    dep_icons = meta.get("dep_icons", {})

    subpackages = _discover_subpackages()
    tutorials = _discover_tutorials()
    deps = [
        d.split(">")[0].split("<")[0].split("=")[0].split("[")[0].split("@")[0].strip()
        for d in proj.get("dependencies", [])
    ]
    authors = proj.get("authors", [])

    gh = f"https://github.com/{info['owner']}/{info['repo']}"

    # --- Build markdown sections ---
    lines: list[str] = []

    # Front-matter
    lines.append("---")
    lines.append("title: Home")
    lines.append("hide:")
    lines.append("  - navigation")
    lines.append("  - toc")
    lines.append("---")
    lines.append("")

    # Hero
    lines.append("# neuro-py")
    lines.append("")
    lines.append('<div style="font-size: 0.75rem; max-width: 680px;" markdown>')
    lines.append(f"**{info['description']}**")
    lines.append(
        "Built on top of [nelpy](https://github.com/nelpy/nelpy) for core data objects,"
    )
    lines.append(
        "neuro_py provides functions for freely moving electrophysiology analysis —"
    )
    lines.append(
        "including behavior tracking, neural ensemble detection, peri-event analyses,"
    )
    lines.append("spectral methods, and robust batch analysis tools.")
    lines.append("</div>")
    lines.append("")

    # Badges (derived from pyproject URLs)
    lines.append(
        f"[![DOI](https://zenodo.org/badge/629590369.svg)](https://doi.org/10.5281/zenodo.16929395)"
    )
    lines.append(
        f"[![PyPI](https://img.shields.io/pypi/v/{info['pypi']}.svg?logo=pypi&label=PyPI&logoColor=gold)]"
        f"(https://pypi.org/project/{info['pypi']}/)"
    )
    lines.append(
        f"[![Python](https://img.shields.io/pypi/pyversions/{info['pypi']}.svg?logo=python&label=Python&logoColor=gold)]"
        f"(https://pypi.org/project/{info['pypi']}/)"
    )
    lines.append(
        f"[![Tests]({gh}/actions/workflows/ci.yml/badge.svg)]({gh}/actions/workflows/ci.yml)"
    )
    lines.append(
        f"[![Docs]({gh}/actions/workflows/deploy-docs.yml/badge.svg)]({gh}/actions/workflows/deploy-docs.yml)"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # Getting Started — loaded from README.md sections
    readme_sections = _parse_readme_sections()
    install_body = readme_sections.get("installation", "")
    usage_body = readme_sections.get("usage", "")

    lines.append("## Getting Started")
    lines.append("")
    lines.append('<div class="grid cards" markdown>')
    lines.append("")

    # Installation card
    lines.append("- :material-download:{ .lg .middle } __Installation__")
    lines.append("")
    lines.append("    ---")
    lines.append("")
    if install_body:
        for line in install_body.splitlines():
            lines.append(f"    {line}" if line.strip() else "")
    else:
        lines.append("    ```bash")
        lines.append(f"    git clone {gh}.git")
        lines.append(f"    cd {info['repo']}")
        lines.append("    pip install -e .")
        lines.append("    ```")
    lines.append("")

    # Quick Start card
    lines.append("- :material-rocket-launch:{ .lg .middle } __Quick Start__")
    lines.append("")
    lines.append("    ---")
    lines.append("")
    if usage_body:
        for line in usage_body.splitlines():
            lines.append(f"    {line}" if line.strip() else "")
    else:
        lines.append("    ```python")
        lines.append("    import neuro_py as npy")
        lines.append("    ```")
    lines.append("")

    lines.append("</div>")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Module cards — split into Core (first 6 in homepage.yml order) and More
    ordered_mods = [m for m in mod_meta if m in subpackages]
    # Append any discovered subpackages not in homepage.yml
    for sp in subpackages:
        if sp not in ordered_mods:
            ordered_mods.append(sp)

    core = ordered_mods[:6]
    more = ordered_mods[6:]

    def _mod_card(pkg: str) -> list[str]:
        m = mod_meta.get(pkg, {})
        icon = m.get("icon", "material-package-variant")
        desc = m.get("description", f"Functions in the `{pkg}` subpackage.")
        title = _MODULE_TITLES.get(pkg, pkg.capitalize())
        card: list[str] = []
        card.append(f"- :{icon}:{{ .lg .middle }} __{title}__")
        card.append("")
        card.append("    ---")
        card.append("")
        for line in desc.strip().splitlines():
            card.append(f"    {line.strip()}")
        card.append("")
        card.append(
            f"    [:octicons-arrow-right-24: View Module](reference/neuro_py/{pkg}/index.md)"
        )
        return card

    lines.append("## Core Modules")
    lines.append("")
    lines.append('<div class="grid cards" markdown>')
    lines.append("")
    for pkg in core:
        lines.extend(_mod_card(pkg))
        lines.append("")
    lines.append("</div>")
    lines.append("")
    lines.append("---")
    lines.append("")

    if more:
        lines.append("## More Modules")
        lines.append("")
        lines.append('<div class="grid cards" markdown>')
        lines.append("")
        for pkg in more:
            lines.extend(_mod_card(pkg))
            lines.append("")
        lines.append("</div>")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Tutorials
    if tutorials:
        lines.append("## Tutorials")
        lines.append("")
        lines.append("Learn by example with our hands-on notebooks:")
        lines.append("")
        lines.append('<div class="grid cards" markdown>')
        lines.append("")
        for stem in tutorials:
            t = tut_meta.get(stem, {})
            icon = t.get("icon", "material-notebook")
            title = t.get("title", stem.replace("_", " ").title())
            desc = t.get("description", f"Explore the {title} tutorial notebook.")
            lines.append(f"- :{icon}:{{ .lg .middle }} __{title}__")
            lines.append("")
            lines.append("    ---")
            lines.append("")
            lines.append(f"    {desc}")
            lines.append("")
            lines.append(
                f"    [:octicons-arrow-right-24: View Tutorial](tutorials/{stem}.md)"
            )
            lines.append("")
        lines.append("</div>")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Built With — dynamically from dependencies
    python_ver = info["python"].lstrip(">= ")
    lines.append("## Built With")
    lines.append("")
    lines.append('<div class="grid" markdown>')
    lines.append("")
    lines.append(f":material-language-python: **Python {python_ver}+**")
    lines.append("")
    for dep in deps:
        dep_name = dep.strip()
        icon = dep_icons.get(dep_name)
        if icon:
            lines.append(f":{icon}: **{dep_name}**")
            lines.append("")
    lines.append("</div>")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Community
    lines.append("## Community")
    lines.append("")
    lines.append('<div class="grid cards" markdown>')
    lines.append("")
    lines.append("- :octicons-mark-github-16:{ .lg .middle } __GitHub__")
    lines.append("")
    lines.append("    ---")
    lines.append("")
    lines.append(
        "    Star or watch the repository to stay updated with the latest changes."
    )
    lines.append("")
    lines.append(
        f"    [:octicons-arrow-right-24: View Repository]({gh}){{ .md-button .md-button--primary }}"
    )
    lines.append("")
    lines.append("- :material-heart:{ .lg .middle } __Contributing__")
    lines.append("")
    lines.append("    ---")
    lines.append("")
    lines.append(
        "    Pull requests welcome. For major changes, please open an issue first"
    )
    lines.append("    to discuss what you would like to change.")
    lines.append("")
    lines.append(
        f"    [:octicons-arrow-right-24: Open an Issue]({gh}/issues){{ .md-button .md-button--primary }}"
    )
    lines.append("")
    lines.append("- :material-test-tube:{ .lg .middle } __Testing__")
    lines.append("")
    lines.append("    ---")
    lines.append("")
    lines.append("    Run the test suite to verify everything works:")
    lines.append("")
    lines.append("    ```bash")
    lines.append("    pytest")
    lines.append("    ```")
    lines.append("")
    lines.append("</div>")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Contributors & Activity
    lines.append('<div class="grid cards" markdown>')
    lines.append("")
    lines.append("- ## Contributors")
    lines.append("")
    lines.append("    Special thanks to our contributors!")
    lines.append("")
    lines.append(
        f"    [![Contributors](https://contrib.rocks/image?repo={info['owner']}/{info['repo']})]"
        f"({gh}/graphs/contributors)"
    )
    lines.append("")
    lines.append("- ## Project Activity")
    lines.append("")
    lines.append(
        f"    [![Downloads](https://pepy.tech/badge/{info['pypi']})](https://pepy.tech/project/{info['pypi']})"
    )
    lines.append(
        f"    [![PyPI Downloads](https://img.shields.io/pypi/dm/{info['pypi']}?color=blue&label=Monthly%20Installs&logo=pypi&logoColor=gold)]"
        f"(https://pypi.org/project/{info['pypi']}/)"
    )
    lines.append(
        f"    [![Issues](https://img.shields.io/github/issues/{info['owner']}/{info['repo']}?logo=github)]"
        f"({gh}/issues)"
    )
    lines.append(
        f"    [![Last Commit](https://img.shields.io/github/last-commit/{info['owner']}/{info['repo']})]"
        f"({gh}/commits)"
    )
    lines.append("")
    lines.append("</div>")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Footer — authors from pyproject.toml
    if authors:
        author_links = []
        for a in authors:
            name = a.get("name", "")
            # Derive GitHub handle heuristic: first-last → @first-last (fallback)
            author_links.append(name)
        lines.append('<div style="text-align: center;" markdown>')
        lines.append(f"Made with :heart: by {', '.join(author_links)}")
        lines.append("</div>")
        lines.append("")

    index = DOCS_DIR / "index.md"
    index.write_text("\n".join(lines))
    n_mods = len(ordered_mods)
    n_tuts = len(tutorials)
    n_deps = sum(1 for d in deps if d.strip() in dep_icons)
    print(
        f"  Generated index.md dynamically ({n_mods} modules, {n_tuts} tutorials, {n_deps} deps)"
    )


def update_nav(ref_nav: list) -> None:
    """Update mkdocs.yml nav section with generated reference structure.

    Uses text-based replacement to preserve ``!!python/name`` tags and other
    non-standard YAML constructs that ``yaml.safe_load`` cannot parse.

    Parameters
    ----------
    ref_nav : list
        Nav structure for the API Reference section.
    """
    config_path = ROOT / "mkdocs.yml"
    text = config_path.read_text()

    # Build tutorial nav entries
    tutorial_nav = []
    tutorials_dir = DOCS_DIR / "tutorials"
    if tutorials_dir.exists():
        for md_file in sorted(tutorials_dir.glob("*.md")):
            title = md_file.stem.replace("_", " ").title()
            tutorial_nav.append({title: f"tutorials/{md_file.name}"})

    # Build the new nav YAML fragment
    nav = [
        {"Home": "index.md"},
        {"API Reference": ref_nav},
        {"Tutorials": tutorial_nav},
    ]
    nav_yaml = yaml.dump({"nav": nav}, default_flow_style=False, sort_keys=False)

    # Find and replace the nav section in the file text
    import re

    # Match "nav:" and every following line that is either blank, starts with
    # "- " (top-level entry), or starts with whitespace (nested entries).
    # Stops at the first non-blank, non-indented, non-"- " line (i.e. the
    # next top-level YAML key) or EOF.
    nav_pattern = re.compile(
        r"^nav:\s*\n(?:[ \t]*-.*\n|[ \t]+\S.*\n|\s*\n)*",
        re.MULTILINE,
    )
    match = nav_pattern.search(text)
    if match:
        text = text[: match.start()] + nav_yaml + text[match.end() :]
    else:
        # Append if no nav section exists
        text += "\n" + nav_yaml

    config_path.write_text(text)
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
