"""Validate the executable normalization and event lookup in the tutorial."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as mpl_cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


def source_starting(prefix):
    path = (
        Path(__file__).resolve().parents[2] / "tutorials/weighted_regression_2d.ipynb"
    )
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    return next(
        "".join(c["source"]) for c in cells if "".join(c["source"]).startswith(prefix)
    )


@pytest.mark.parametrize(
    "name",
    [
        "simulate_degenerate_x",
        "simulate_degenerate_y",
        "simulate_no_movement",
        "simulate_both_axes",
        "simulate_anti_diagonal",
        "simulate_curved",
        "simulate_noisy_linear",
        "simulate_reverse",
        "simulate_fragmented",
    ],
)
def test_simulations_are_normalized_per_time_bin(name):
    namespace = {"np": np}
    exec(source_starting("def simulate_degenerate_x"), namespace)
    posterior = namespace[name]()
    assert np.all(np.isfinite(posterior))
    assert np.all(posterior >= 0)
    np.testing.assert_allclose(posterior.sum(axis=(0, 1)), 1, atol=1e-14)


def test_plot_looks_up_nonconsecutive_ripple_ids(monkeypatch):
    source = source_starting("if RUN_REAL_DATA:\n    from mpl_toolkits")
    ast.parse(source)
    # Input rows deliberately differ from display order, and IDs cannot be
    # valid positional array indices. Old array[rid] code would fail here.
    table = pd.DataFrame(
        {
            "ripple_index": [21, 7],
            "regression_r2": [0.8, 0.9],
            "mean_x": [20.0, 10.0],
            "mean_y": [40.0, 30.0],
            "slope_x": [4.0, 2.0],
            "slope_y": [8.0, 6.0],
            "jump": [2.0, 1.0],
            "max_jump": [5.0, 3.0],
        }
    )
    plotting = SimpleNamespace(
        set_plotting_defaults=lambda: None,
        set_size=lambda *args, **kwargs: (10, 10),
        plot_2d_replay=lambda *args, **kwargs: None,
    )
    namespace = dict(
        RUN_REAL_DATA=True,
        np=np,
        plt=plt,
        matplotlib=matplotlib,
        colors=colors,
        mpl_cm=mpl_cm,
        npy=SimpleNamespace(plotting=plotting),
        pos={0: SimpleNamespace(data=np.zeros((2, 2)))},
        beh_epochs=[0],
        TASK_EPOCH=0,
        SLIDE_BY=0.005,
        potential_replays=table,
        posterior_prob=np.ones((2, 2, 4)),
        ripple_index=np.array([7, 7, 21, 21]),
        spatial_maps=SimpleNamespace(bins=[np.arange(3), np.arange(3)]),
    )
    monkeypatch.setattr(plt, "show", lambda: None)
    try:
        exec(source, namespace)
        axes = namespace["ax"]
        for ax, rid in zip(axes, [7, 21]):
            row = table.set_index("ripple_index").loc[rid]
            centered_t = np.array([-0.0025, 0.0025])
            np.testing.assert_allclose(
                ax.lines[-1].get_xdata(), row.mean_x + row.slope_x * centered_t
            )
            np.testing.assert_allclose(
                ax.lines[-1].get_ydata(), row.mean_y + row.slope_y * centered_t
            )
            assert f"R²: {row.regression_r2:.2f}" in ax.get_title()
            assert f"Mean jump: {row.jump:.2f} cm" in ax.get_title()
            assert f"Max jump: {row.max_jump:.2f} cm" in ax.get_title()
        assert all(not ax.get_visible() for ax in axes[2:])
    finally:
        plt.close("all")


def shuffle_namespace():
    from neuro_py.ensemble.replay import weighted_regression_2d

    namespace = {"np": np, "weighted_regression_2d": weighted_regression_2d}
    exec(source_starting("N_SHUFFLES ="), namespace)
    return namespace


def test_tutorial_spatial_shuffle_geometry_and_mass():
    namespace = shuffle_namespace()
    posterior = np.arange(12.0).reshape(2, 3, 2)
    posterior[..., 0] = np.arange(6).reshape(2, 3)
    before = posterior.copy()
    result = namespace["toroidal_spatial_shuffle"](posterior, [1, 0], [2, 0])
    # Known result of an axis-0 shift of 1 and axis-1 shift of 2.
    np.testing.assert_array_equal(result[..., 0], [[4, 5, 3], [1, 2, 0]])
    np.testing.assert_array_equal(result[..., 1], posterior[..., 1])
    np.testing.assert_array_equal(result.sum(axis=(0, 1)), posterior.sum(axis=(0, 1)))
    np.testing.assert_array_equal(posterior, before)


def test_tutorial_null_is_reproducible_and_uses_correct_empirical_p():
    ns = shuffle_namespace()
    exec(source_starting("def simulate_degenerate_x"), ns)
    posterior = ns["simulate_noisy_linear"](noise=0.50)
    observed, null, p = ns["r2_shuffle_test"](posterior, n_shuffles=99, seed=42)
    again = ns["r2_shuffle_test"](posterior, n_shuffles=99, seed=42)
    np.testing.assert_array_equal(null, again[1])
    assert np.all((null >= 0) & (null <= 1))
    assert p == (np.count_nonzero(null >= observed) + 1) / 100
    assert 0.01 <= p <= 1


def test_tutorial_shuffle_ties_and_undefined_events():
    ns = shuffle_namespace()
    _, null, p = ns["r2_shuffle_test"](np.ones((2, 2, 4)), n_shuffles=9)
    np.testing.assert_array_equal(null, 0)
    assert p == 1
    with pytest.raises(ValueError, match="positive integer"):
        ns["r2_shuffle_test"](np.ones((2, 2, 4)), n_shuffles=0)
    with pytest.raises(ValueError, match="finite"):
        ns["r2_shuffle_test"](np.ones((2, 2, 1)), n_shuffles=9)
