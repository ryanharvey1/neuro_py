# neuro-py

Analysis of neuroelectrophysiology data in Python.

[![DOI](https://zenodo.org/badge/629590369.svg)](https://doi.org/10.5281/zenodo.16929395)

|         |                                                                    |
|---------|--------------------------------------------------------------------|
| CI/CD   | [![CI - Test](https://github.com/ryanharvey1/neuro_py/actions/workflows/ci.yml/badge.svg)](https://github.com/ryanharvey1/neuro_py/actions/workflows/ci.yml) [![Docs](https://github.com/ryanharvey1/neuro_py/actions/workflows/deploy-docs.yml/badge.svg)](https://github.com/ryanharvey1/neuro_py/actions/workflows/deploy-docs.yml)    |
| Package | [![PyPI - Version](https://img.shields.io/pypi/v/neuro-analysis-py.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/neuro-analysis-py/) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/neuro-analysis-py.svg?logo=python&label=Python&logoColor=gold)](https://pypi.org/project/neuro-analysis-py/) [![PyPI - Downloads](https://img.shields.io/pypi/dm/neuro-analysis-py?color=blue&label=Installs&logo=pypi&logoColor=gold)](https://pypi.org/project/neuro-analysis-py/)    |
| Repository | [![GitHub - Issues](https://img.shields.io/github/issues/ryanharvey1/neuro_py?logo=github&label=Issues&logoColor=gold)]() [![Commits](https://img.shields.io/github/last-commit/ryanharvey1/neuro_py)]() [![Contributors](https://img.shields.io/github/contributors/ryanharvey1/neuro_py)]() [![Downloads](https://pepy.tech/badge/neuro-analysis-py)](https://pepy.tech/project/neuro-analysis-py)    |
| Metadata   | [![GitHub - License](https://img.shields.io/github/license/ryanharvey1/neuro_py?logo=github&label=License&logoColor=gold)](LICENSE) [![code style - black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![docstring - numpydoc](https://img.shields.io/badge/docstring-numpydoc-blue)](https://numpydoc.readthedocs.io/en/latest/format.html) [![linting - Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![type checked - ty](https://img.shields.io/badge/type%20checker-ty-261230?style=flat-square)](https://docs.astral.sh/ty/)    |


Overview
========
`neuro_py` is a Python package for analysis of neuroelectrophysiology data. It is built on top of the [nelpy](https://github.com/nelpy/nelpy) package, which provides core data objects. `neuro_py` provides a set of functions for analysis of freely moving electrophysiology, including behavior tracking utilities, neural ensemble detection, peri-event analyses, robust batch analysis tools, and more. 

Documentation, tutorials, and API reference are available at [**ryanharvey1.github.io/neuro_py/**](https://ryanharvey1.github.io/neuro_py/).
The decoding tutorial and torch/lightning-backed decoders require the optional `dl` extra.


## Install

```bash
git clone https://github.com/ryanharvey1/neuro_py.git
cd neuro_py
pip install -e .
```

Install the optional `dl` extra to use the deep-learning decoders:

```bash
pip install -e .[dl]
```

## Get started

```python
import neuro_py as npy
```

See the [documentation site](https://ryanharvey1.github.io/neuro_py/) for complete examples.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Authors

- [@ryanharvey1](https://www.github.com/ryanharvey1)
- [@lolaBerkowitz](https://www.github.com/lolaBerkowitz)
- [@kushaangupta](https://github.com/kushaangupta)
