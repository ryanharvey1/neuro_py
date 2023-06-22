import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from neuro_py.stats.stats import confidence_intervals


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


def plot_peth(peth: pd.DataFrame, ax=None, **kwargs) -> matplotlib.axes.Axes:
    """
    Plot a peth. Assumes that the index is time and the columns are trials/cells/etc.

    Parameters
    ----------
    peth : pd.DataFrame
        Peth to plot
    ax : matplotlib.axes.Axes, optional
        Axis to plot on, by default None
    **kwargs
        Keyword arguments to pass to seaborn.lineplot

    Returns
    -------
    matplotlib.axes.Axes
        Axis with plot

    Raises
    ------
    TypeError
        If peth is not a pandas dataframe

    Example
    -------
    >>> from neuro_py.plotting.events import plot_peth
    >>> from neuro_py.process import peri_event
    >>> from neuro_py.io import loading

    >>> st, cm = loading.load_spikes(basepath)
    >>> rippple_epochs = loading.load_ripples_events(basepath, return_epoch_array=True)

    >>> ripple_peth = peri_event.compute_psth(st.data, rippple_epochs.starts)
    >>> plot_peth(ripple_peth)

    """
    if ax is None:
        fig, ax = plt.subplots()

    # verify peth is a dataframe
    if not isinstance(peth, pd.DataFrame):
        raise TypeError("peth must be a pandas dataframe")

    # melt the dataframe so that the index is time and there is a column for each trial/cell/etc.
    peth_long = pd.melt(peth.reset_index(), id_vars=["index"], value_name="peth")

    # plot the peth as a lineplot with seaborn
    lineplot_ax = sns.lineplot(data=peth_long, x="index", y="peth", ax=ax, **kwargs)
    ax.set_xlabel("Time (s)")
    sns.despine(ax=ax)
    return lineplot_ax


def plot_peth_fast(peth: pd.DataFrame, ax=None,ci=.95, **kwargs) -> plt.Axes:
    """
    Plot a peth. Assumes that the index is time and the columns are trials/cells/etc.

    Less flexible, but faster version of plot_peth

    Parameters
    ----------
    peth : pd.DataFrame
        Peth to plot
    ax : plt.Axes, optional
        Axis to plot on, by default None
    **kwargs
        Keyword arguments to pass to ax.plot

    Returns
    -------
    plt.Axes
        Axis with plot

    Raises
    ------
    TypeError
        If peth is not a pandas dataframe

    Example
    -------
    >>> from neuro_py.plotting.events import plot_peth
    >>> from neuro_py.process import peri_event
    >>> from neuro_py.io import loading

    >>> st, cm = loading.load_spikes(basepath)
    >>> rippple_epochs = loading.load_ripples_events(basepath, return_epoch_array=True)

    >>> ripple_peth = peri_event.compute_psth(st.data, rippple_epochs.starts)
    >>> plot_peth_fast(ripple_peth)

    """
    if ax is None:
        fig, ax = plt.subplots()

    # verify peth is a dataframe
    if not isinstance(peth, pd.DataFrame):
        raise TypeError("peth must be a pandas dataframe")

    # plot the peth as a lineplot with matplotlib
    ax.plot(peth.index, peth.mean(axis=1), **kwargs)
    
    lower, upper = confidence_intervals(peth.values.T, conf=ci)
    ax.fill_between(peth.index, lower, upper, alpha=.5, **kwargs)

    ax.set_xlabel("Time (s)")
    sns.despine(ax=ax)

    return ax