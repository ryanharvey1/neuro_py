from pathlib import Path

import pytest


def test_notebook_template_collapses_only_tagged_inputs() -> None:
    nbconvert = pytest.importorskip("nbconvert")
    nbformat = pytest.importorskip("nbformat")
    template = Path(__file__).resolve().parents[2] / "tools" / "notebook_tutorial.md.j2"
    notebook = nbformat.v4.new_notebook(
        metadata={"language_info": {"name": "python"}},
        cells=[
            nbformat.v4.new_code_cell("spectrum = px.mtspectrumc(signal, fs)"),
            nbformat.v4.new_code_cell(
                "for ax in axes:\n    ax.plot(spectrum)",
                metadata={"tags": ["collapse-plotting-code"]},
                outputs=[
                    nbformat.v4.new_output("display_data", data={"image/png": "YQ=="})
                ],
            ),
            nbformat.v4.new_markdown_cell("Figure caption."),
        ],
    )

    exporter = nbconvert.MarkdownExporter(template_file=str(template))
    markdown, resources = exporter.from_notebook_node(notebook)

    assert markdown.count('??? note "Show plotting code"') == 1
    assert markdown.startswith("```python\nspectrum = px.mtspectrumc(signal, fs)\n```")
    assert (
        "\n    ```python\n    for ax in axes:\n        ax.plot(spectrum)\n    ```"
        in markdown
    )
    (image,) = resources["outputs"]
    assert f"\n![png]({image})" in markdown
    assert "\nFigure caption." in markdown
