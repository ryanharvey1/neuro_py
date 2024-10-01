"""Generate the code reference pages."""
from itertools import chain
from pathlib import Path

import mkdocs_gen_files


nav = mkdocs_gen_files.Nav()
mod_symbol = ''#<code class="doc-symbol doc-symbol-nav doc-symbol-module"></code>'

root = Path(__file__).parent.parent
src = root / 'src'

for path in sorted(chain(src.rglob("*.py"), src.rglob("*.pyi"))):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    #if any([[p == "__pycache__" for p in parts]]):
    #    continue
    # if parts[0] == "__init__":
    #     continue

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    nav_parts = [f"{mod_symbol} {part}" for part in parts]
    print(nav_parts)
    nav[tuple(nav_parts)[-2:]] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        print(f"---\ntitle: {identifier}\n---\n\n::: {identifier}", file=fd)

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
