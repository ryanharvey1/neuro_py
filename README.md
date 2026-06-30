# neuro-py

Analysis of neuroelectrophysiology data in Python.

[![DOI](https://zenodo.org/badge/629590369.svg)](https://doi.org/10.5281/zenodo.16929395)

|         |                                                                    |
|---------|--------------------------------------------------------------------|
| CI/CD   | [![CI - Test](https://github.com/ryanharvey1/neuro_py/actions/workflows/ci.yml/badge.svg)](https://github.com/ryanharvey1/neuro_py/actions/workflows/ci.yml) [![Docs](https://github.com/ryanharvey1/neuro_py/actions/workflows/deploy-docs.yml/badge.svg)](https://github.com/ryanharvey1/neuro_py/actions/workflows/deploy-docs.yml)    |
| Package | [![PyPI - Version](https://img.shields.io/pypi/v/neuro-analysis-py.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/neuro-analysis-py/) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/neuro-analysis-py.svg?logo=python&label=Python&logoColor=gold)](https://pypi.org/project/neuro-analysis-py/) [![PyPI - Downloads](https://img.shields.io/pypi/dm/neuro-analysis-py?color=blue&label=Installs&logo=pypi&logoColor=gold)](https://pypi.org/project/neuro-analysis-py/)    |
| Repository | [![GitHub - Issues](https://img.shields.io/github/issues/ryanharvey1/neuro_py?logo=github&label=Issues&logoColor=gold)]() [![Commits](https://img.shields.io/github/last-commit/ryanharvey1/neuro_py)]() [![Contributors](https://img.shields.io/github/contributors/ryanharvey1/neuro_py)]() [![Downloads](https://pepy.tech/badge/neuro-analysis-py)](https://pepy.tech/project/neuro-analysis-py)    |
| Metadata   | [![GitHub - License](https://img.shields.io/github/license/ryanharvey1/neuro_py?logo=github&label=License&logoColor=gold)](LICENSE) [![code style - black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![docstring - numpydoc](https://img.shields.io/badge/docstring-numpydoc-blue)](https://numpydoc.readthedocs.io/en/latest/format.html) [![linting - Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)    |


Overview
========
`neuro_py` is a Python package for analysis of neuroelectrophysiology data. It is built on top of the [nelpy](https://github.com/nelpy/nelpy) package, which provides core data objects. `neuro_py` provides a set of functions for analysis of freely moving electrophysiology, including behavior tracking utilities, neural ensemble detection, peri-event analyses, robust batch analysis tools, and more. 

Tutorials are [here](https://github.com/ryanharvey1/neuro_py/tree/main/tutorials) and more will be added.
The decoding tutorial and torch/lightning-backed decoders require the optional `dl` extra.


## Installation

```bash
git clone
cd neuro_py
pip install -e .
```

For the optional deep-learning decoders, install the `dl` extra:

```bash
pip install -e .[dl]
```

To sync the `nelpy` dependency to latest version, use following instead,

```bash
pip install -e . --force-reinstall --no-cache-dir
```

## Usage

```python
import neuro_py as npy
```


## Dependencies 

For ease of use, this package uses `nelpy` core data objects. See [nelpy](https://github.com/nelpy/nelpy) 

## Testing

Use plain `pytest` for normal development and CI-like local runs:

```bash
pytest
```

To run a narrow target:

```bash
pytest tests/detectors/test_sharp_wave_ripple.py -q
```

If your local environment auto-loads unrelated third-party pytest plugins, use the wrapper to isolate the repo's own test environment:

```bash
python tools/run_pytest.py
```

This wrapper disables ambient third-party pytest plugins and, when `CI` is set, caps joblib/BLAS thread fan-out to keep the full suite stable on shared runners. Numba JIT and matplotlib backend behavior are still exercised by the tests themselves.

CI also runs a lightweight plain-`pytest` smoke check against the base install so we still catch issues outside the wrapper path.

## Type checking

The repository uses [ty](https://docs.astral.sh/ty/) for staged static type checking.

To install the development tools used by CI:

```bash
pip install -e .[dev]
```

If `ty` is installed in your active environment, run:

```bash
ty check
```

If you prefer not to install `ty` into the environment, use `uvx` against the project environment:

```bash
uvx ty check --python .venv
```

For editor integration, point your editor at the `ty` language server (`ty server`) and keep `pyproject.toml` as the source of truth for repo-specific overrides and temporary suppressions.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Authors

- [@ryanharvey1](https://www.github.com/ryanharvey1)
- [@lolaBerkowitz](https://www.github.com/lolaBerkowitz)
- [@kushaangupta](https://github.com/kushaangupta)
