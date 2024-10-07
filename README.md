# neuro-py

Analysis of neuroelectrophysiology data in Python.

Overview
========
neuro_py is a Python package for analysis of neuroelectrophysiology data. It is built on top of the [nelpy](https://github.com/nelpy/nelpy) package, which provides core data objects. neuro_py provides a set of functions for analysis of freely moving electrophysiology, including behavior tracking utilities, neural ensemble detection, peri-event analyses, robust batch analysis tools, and more. 

Tutorials are [here](https://github.com/ryanharvey1/neuro_py/tree/master/tutorials) and more will be added. 


## Installation

```bash
pip install neuro-analysis-py
```

## Development

```bash
git clone
cd neuro_py
pip install -e .
```

To sync the `nelpy` dependency to latest version, use following instead,

```bash
pip install -e . --force-reinstall --no-cache-dir
```

## Usage

```python
import neuro_py as neuro
```


## Dependencies 

For ease of use, this package uses nelpy core data objects. See [nelpy](https://github.com/nelpy/nelpy) 

## Testing

```bash
pytest
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Authors

- [@ryanharvey1](https://www.github.com/ryanharvey1)
- [@lolaBerkowitz](https://www.github.com/lolaBerkowitz)
- [@kushaangupta](https://github.com/kushaangupta)


## License

neuro_py is distributed under the MIT license. See the [LICENSE](https://github.com/neuro_py/neuro_py/blob/master/LICENSE) file for details.



