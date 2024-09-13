import warnings
from typing import Union

import matplotlib
import matplotlib.pyplot as plt
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


def plot_peth(
    peth: pd.DataFrame,
    ax=None,
    smooth: bool = False,
    smooth_window: float = 0.30,
    smooth_std: int = 5,
    smooth_win_type: str = "gaussian",
    **kwargs
) -> matplotlib.axes.Axes:
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

    if smooth:
        # convert window to samples
        smooth_window = int(smooth_window / np.diff(peth.index)[0])
        # smooth the peth
        peth = (
            peth.rolling(
                window=smooth_window,
                win_type=smooth_win_type,
                center=True,
                min_periods=1,
            )
            .mean(std=smooth_std)
            .copy()
        )

    # melt the dataframe so that the index is time and there is a column for each trial/cell/etc.
    peth_long = pd.melt(peth.reset_index(), id_vars=["index"], value_name="peth")

    # plot the peth as a lineplot with seaborn
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        lineplot_ax = sns.lineplot(data=peth_long, x="index", y="peth", ax=ax, **kwargs)

    ax.set_xlabel("Time (s)")
    sns.despine(ax=ax)
    return lineplot_ax


def plot_peth_fast(
    peth: Union[pd.DataFrame, np.ndarray],
    ts=None,
    ax=None,
    ci: float = 0.95,
    smooth: bool = False,
    smooth_window: float = 0.30,
    smooth_std: int = 5,
    smooth_win_type: str = "gaussian",
    alpha: float = 0.2,
    **kwargs
) -> plt.Axes:
    """
    Plot a peth. Assumes that the index is time and the columns are trials/cells/etc.

    Less flexible, but faster version of plot_peth

    Parameters
    ----------
    peth : pd.DataFrame, np.ndarray
        Peth to plot
    ts : np.ndarray, optional
        Time points to plot, by default None
    ax : plt.Axes, optional
        Axis to plot on, by default None
    ci : float, optional
        Confidence interval to plot, by default 0.95
    smooth : bool, optional
        Whether to smooth the peth, by default False
    smooth_window : float, optional
        Window to smooth the peth, by default 0.30
    smooth_std : int, optional
        Standard deviation of the smoothing window, by default 5
    smooth_win_type : str, optional
        Type of smoothing window, by default "gaussian"
    alpha : float, optional
        Transparency of the confidence interval, by default 0.2

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

    # verify peth is a dataframe, if not convert it
    if not isinstance(peth, pd.DataFrame):

        # if ts is not provided, create a range of time points
        if ts is None:
            ts = np.arange(peth.shape[0])

        # transpose peth so that time is rows and trials are columns
        if len(ts) == peth.shape[1]:
            peth = peth.T

        peth = pd.DataFrame(
            index=ts,
            columns=np.arange(peth.shape[1]),
            data=peth,
        )
        # raise TypeError("peth must be a pandas dataframe")

    if smooth:
        # convert window to samples
        smooth_window = int(smooth_window / np.diff(peth.index)[0])
        # smooth the peth
        peth = (
            peth.rolling(
                window=smooth_window,
                win_type=smooth_win_type,
                center=True,
                min_periods=1,
            )
            .mean(std=smooth_std)
            .copy()
        )

    # plot the peth as a lineplot with matplotlib
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ax.plot(peth.index, np.nanmean(peth, axis=1), **kwargs)

    # drop label from kwargs, as it was already used in the plot
    kwargs.pop("label", None)

    lower, upper = confidence_intervals(peth.values.T, conf=ci)
    ax.fill_between(peth.index, lower, upper, alpha=alpha, **kwargs)

    ax.set_xlabel("Time (s)")
    sns.despine(ax=ax)

    return ax
