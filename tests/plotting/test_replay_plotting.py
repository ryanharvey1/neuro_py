import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from neuro_py.plotting.replay import make_replay, plot_2d_replay

matplotlib.use("Agg")


def test_plot_2d_replay_returns_fig_ax_and_stable_image_count():
    replay = make_replay(nx=20, ny=20, T=6, kind="linear", seed=0)

    fig, ax = plot_2d_replay(
        replay,
        percentile_threshold=0,
        per_frame_alpha_normalization=True,
    )

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert len(ax.images) == replay.shape[2]
    plt.close(fig)


def test_plot_2d_replay_all_zero_input_no_warnings_or_errors():
    replay = np.zeros((10, 10, 4), dtype=float)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig, ax = plot_2d_replay(replay, percentile_threshold=0)

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert len(caught) == 0
    assert len(ax.images) == 0
    plt.close(fig)


def test_make_replay_raises_for_unknown_kind():
    with pytest.raises(ValueError, match="Unknown kind"):
        make_replay(kind="unknown")


def test_plot_2d_replay_accepts_colormap_instance():
    replay = make_replay(nx=20, ny=20, T=3, kind="curved", seed=1)

    fig, ax = plot_2d_replay(replay, cmap=plt.get_cmap("cool"), percentile_threshold=0)

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert len(ax.images) == replay.shape[2]
    plt.close(fig)
