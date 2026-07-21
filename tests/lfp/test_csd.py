import numpy as np
import pytest

from neuro_py.lfp import CSD


def test_standard_csd_returns_channel_by_time_array() -> None:
    coordinates = np.arange(4, dtype=float) * 0.05
    data = coordinates[:, None] ** 2 * np.array([[1.0, 2.0, 3.0]])

    result = CSD.get_csd("unused", data, shank=0, method="StandardCSD", channel_offset=0.05)

    np.testing.assert_allclose(
        result[1:-1], -2 * np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    )
    assert result.shape == data.shape


def test_standard_csd_uses_explicit_coordinates() -> None:
    coordinates = np.array([0.0, 0.04, 0.1, 0.18])
    data = coordinates[:, None] ** 2 * np.array([[1.0, 2.0]])

    result = CSD.get_csd(
        "unused", data, shank=0, method="StandardCSD", coords=coordinates
    )

    np.testing.assert_allclose(result[1:-1], -2 * np.array([[1.0, 2.0]] * 2))


def test_kd1csd_alias_warns_and_uses_kcsd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CSD, "get_coords", lambda basepath, shank: np.array([0.0, 0.05, 0.1]))
    data = np.arange(12, dtype=float).reshape(3, 4)

    with pytest.deprecated_call(match="KD1CSD"):
        legacy = CSD.get_csd("unused", data, shank=0, method="KD1CSD")
    current = CSD.get_csd("unused", data, shank=0, method="KCSD1D")

    np.testing.assert_allclose(legacy, current)


def test_csd_rejects_channel_time_orientation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CSD, "get_coords", lambda basepath, shank: np.array([0.0, 0.05, 0.1]))

    with pytest.raises(ValueError, match="first data dimension"):
        CSD.get_csd("unused", np.ones((4, 3)), shank=0, method="DeltaiCSD")
