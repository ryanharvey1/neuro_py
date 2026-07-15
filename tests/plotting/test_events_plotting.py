import matplotlib.pyplot as plt
import nelpy as nel

from neuro_py.plotting.events import plot_events


def test_plot_events_returns_axes():
    fig, ax = plt.subplots()
    events = [nel.EpochArray([[0.0, 1.0]])]

    result = plot_events(events, ["event"], ax=ax)

    assert result is ax
    assert len(ax.patches) == 1
    plt.close(fig)
