import matplotlib
from matplotlib import pyplot as plt
import numpy as np


def plot_events(events, labels, cmap="tab20", gridlines=True, alpha=0.75, ax=None):
    """
    events: nested list of nelpy EpochArrays
    labels: labels related to each event

    example:

        # load sleep states
        state_dict = loading.load_SleepState_states(basepath)

        # make nelpy epoch arrays
        nrem_epochs = nel.EpochArray(state_dict['NREMstate'])
        wake_epochs = nel.EpochArray(state_dict['WAKEstate'])
        rem_epochs = nel.EpochArray(state_dict['REMstate'])

        # add to list
        events = []
        events.append(nrem_epochs)
        events.append(wake_epochs)
        events.append(rem_epochs)

        # plot
        plt.figure(figsize=(20,5))
        plot_events(events,['nrem','wake','rem'])

    Ryan H 2022
    """
    # get colormap
    cmap = matplotlib.cm.get_cmap(cmap)

    # set up y axis
    y = np.linspace(0, 1, len(events) + 1)

    # set up ax if not provided
    if ax is None:
        ax = plt.gca()

    # iter over each event
    for i, evt in enumerate(events):

        # add horizontal line underneath
        if gridlines:
            ax.axhline(y[i] + np.diff(y)[0] / 2, color="k", zorder=-100, alpha=0.1)

        # plot events
        for pair in range(evt.n_intervals):
            ax.axvspan(
                evt.starts[pair],
                evt.stops[pair],
                y[i],
                y[i + 1],
                alpha=alpha,
                color=cmap(i * 0.1),
            )

    ax.set_yticks(y[:-1] + np.diff(y)[0] / 2)
    ax.set_yticklabels(labels)
