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
