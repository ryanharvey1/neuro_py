import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")

from neuro_py.ensemble.decoding.pipeline import _get_decoder


def test_get_decoder_invalid_name_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown decoder 'BAD'"):
        _get_decoder("BAD")
