from typing import Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import PathPatch

from neuro_py.process.peri_event import joint_peth
from neuro_py.process.utils import average_diagonal


def set_plotting_defaults() -> None:
    """
    Set default plotting parameters for matplotlib with LaTeX-style fonts.

    This function updates matplotlib's plotting style to use serif fonts,
    sets font sizes for various elements, and ensures that SVG output uses
    non-embedded fonts for better compatibility.
    """
    tex_fonts = {
        "font.family": "serif",
        "axes.labelsize": 10,
        "font.size": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "svg.fonttype": "none",
    }

    plt.style.use("default")
    plt.rcParams.update(tex_fonts)


def set_size(
    width: Union[float, str], fraction: float = 1, subplots: Tuple[int, int] = (1, 1)
) -> Tuple[float, float]:
    """
    Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width : float or str
        Document width in points (float) or predefined document type (str).
        Supported types: 'thesis', 'beamer', 'paper'.
    fraction : float, optional
        Fraction of the width which you wish the figure to occupy, by default 1.
    subplots : tuple of int, optional
        Number of rows and columns of subplots, by default (1, 1).

    Returns
    -------
    tuple of float
        Dimensions of the figure in inches (width, height).
    """
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    elif width == "paper":
        width_pt = 595.276
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def lighten_color(color: str, amount: float = 0.5) -> str:
    """
    Lightens a hex color by blending it with white by a given percentage.

    Parameters
    ----------
    color : str
        Hex color code (e.g., '#AABBCC').
    amount : float, optional
        Fraction of the lightening, where 0 is no change and 1 is white, by default 0.5.

    Returns
    -------
    str
        Lightened hex color code (e.g., '#FFFFFF').

    Raises
    ------
    ValueError
        If the color string is not a valid hex code.

    Examples
    -------
    >>> lighten_color("#AABBCC", 0.3)
    '#c3cfdb'
    """
    try:
        c = color.lstrip("#")
        c = tuple(int(c[i : i + 2], 16) for i in (0, 2, 4))
        c = (
            int((1 - amount) * c[0] + amount * 255),
            int((1 - amount) * c[1] + amount * 255),
            int((1 - amount) * c[2] + amount * 255),
        )
        return "#%02x%02x%02x" % c
    except ValueError:
        return color


def set_equal_axis_range(ax1: plt.Axes, ax2: plt.Axes) -> None:
    """
    Synchronizes the x and y axis ranges between two matplotlib axes.

    Parameters
    ----------
    ax1 : matplotlib.axes.Axes
        The first axis to synchronize.
    ax2 : matplotlib.axes.Axes
        The second axis to synchronize.

    Examples
    -------
    >>> fig, (ax1, ax2) = plt.subplots(1, 2)
    >>> ax1.plot([1, 2, 3], [4, 5, 6])
    >>> ax2.plot([1, 2, 3], [2, 3, 4])
    >>> set_equal_axis_range(ax1, ax2)
    """
    # Get x and y axis limits for both axes
    axis_x_values = np.hstack(np.array((ax1.get_xlim(), ax2.get_xlim())))
    axis_y_values = np.hstack(np.array((ax1.get_ylim(), ax2.get_ylim())))

    ax1.set_xlim(axis_x_values.min(), axis_x_values.max())
    ax1.set_ylim(axis_y_values.min(), axis_y_values.max())
    ax2.set_xlim(axis_x_values.min(), axis_x_values.max())
    ax2.set_ylim(axis_y_values.min(), axis_y_values.max())


def restore_natural_scale(
    ax: matplotlib.axes.Axes,
    min_: float,
    max_: float,
    n_steps: int = 4,
    x_axis: bool = True,
    y_axis: bool = True,
) -> None:
    """
    Converts logarithmic scale ticks to natural scale (base 10) for the specified axes.

    This function sets the ticks on the specified axes to be evenly spaced
    in the logarithmic scale and converts them back to the natural scale
    for display.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to modify.
    min_ : float
        The minimum value for the ticks in logarithmic scale.
    max_ : float
        The maximum value for the ticks in logarithmic scale.
    n_steps : int, optional
        The number of ticks to create, by default 4.
    x_axis : bool, optional
        If True, adjust the x-axis, by default True.
    y_axis : bool, optional
        If True, adjust the y-axis, by default True.

    Returns
    -------
    None
        This function modifies the axis ticks in place.

    Examples
    -------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.set_xscale('log10')
    >>> ax.plot(np.log10([1, 10, 100]), np.log10([1, 10, 100]))
    >>> restore_natural_scale(ax, 0, 2)
    """
    ticks = np.linspace(min_, max_, n_steps)

    if x_axis:
        ax.set_xticks(ticks)
        ax.set_xticklabels(np.round(10**ticks, 3))

    if y_axis:
        ax.set_yticks(ticks)
        ax.set_yticklabels(np.round(10**ticks, 3))


def adjust_box_widths(g: sns.axisgrid.FacetGrid, fac: float) -> None:
    """
    Adjust the widths of boxes in a Seaborn-generated boxplot.

    This function iterates through the axes of the provided FacetGrid
    and modifies the widths of the boxplot boxes by a specified factor.

    Parameters
    ----------
    g : seaborn.axisgrid.FacetGrid
        The FacetGrid object containing the boxplot.
    fac : float
        The factor by which to adjust the box widths.
        A value < 1 will narrow the boxes, while > 1 will widen them.

    Returns
    -------
    None
        The function modifies the box widths in place.

    Examples
    -------
    >>> import seaborn as sns
    >>> import matplotlib.pyplot as plt
    >>> tips = sns.load_dataset("tips")
    >>> g = sns.boxplot(x="day", y="total_bill", data=tips)
    >>> adjust_box_widths(g, 0.5)  # Narrow the boxes by 50%
    """

    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5 * (xmin + xmax)
                xhalf = 0.5 * (xmax - xmin)

                # setting new width of box
                xmin_new = xmid - fac * xhalf
                xmax_new = xmid + fac * xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for line in ax.lines:
                    if np.all(line.get_xdata() == [xmin, xmax]):
                        line.set_xdata([xmin_new, xmax_new])


def plot_joint_peth(
    peth_1: np.ndarray,
    peth_2: np.ndarray,
    ts: np.ndarray,
    smooth_std: float = 2,
    labels: list = ["peth_1", "peth_2", "event"],
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot joint peri-event time histograms (PETHs) and the difference between the observed and expected responses.

    Parameters
    ----------
    peth_1 : np.ndarray
        Peri-event time histogram (PETH) for the first event. Shape: (n_events, n_time_points).
    peth_2 : np.ndarray
        Peri-event time histogram (PETH) for the second event. Shape: (n_events, n_time_points).
    ts : np.ndarray
        Time vector for the PETHs.
    smooth_std : float, optional
        Standard deviation of the Gaussian kernel used to smooth the PETHs. Default is 2.
    labels : List[str], optional
        Labels for the PETHs. Default is ["peth_1", "peth_2", "event"].

    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        Figure and axes objects for the plot.

    Examples
    -------
    >>> peth_1 = np.random.rand(10, 100)  # Example data for peth_1
    >>> peth_2 = np.random.rand(10, 100)  # Example data for peth_2
    >>> ts = np.linspace(-1, 1, 100)  # Example time vector
    >>> plot_joint_peth(peth_1, peth_2, ts)

    """

    window = [ts[0], ts[-1]]

    joint, expected, difference = joint_peth(peth_1, peth_2, smooth_std=smooth_std)

    # get average of diagonals
    corrected = average_diagonal(difference.T)
    # get center values of corrected_2
    corrected = corrected[
        difference.shape[1] // 2 : (difference.shape[1] // 2) + difference.shape[1]
    ]

    fig, ax = plt.subplots(
        2,
        4,
        figsize=(12, 4),
        gridspec_kw={"width_ratios": [0.25, 1, 1, 1], "height_ratios": [0.25, 1]},
    )
    # space between panels
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    ax[1, 1].imshow(
        joint,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        extent=[window[0], window[-1], window[0], window[-1]],
    )

    ax[0, 1].plot(
        np.linspace(window[0], window[-1], len(joint)), joint.mean(axis=0), color="k"
    )
    ax[0, 1].set_ylabel(f"{labels[1]} rate")
    ax[0, 1].axvline(0, ls="--", color="k")

    ax[1, 0].plot(
        joint.mean(axis=1), np.linspace(window[0], window[-1], len(joint)), color="k"
    )
    ax[1, 0].axhline(0, ls="--", color="k")
    ax[1, 0].set_xlabel(f"{labels[0]} rate")

    # plt.colorbar(f)
    ax[1, 2].imshow(
        expected,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        extent=[window[0], window[-1], window[0], window[-1]],
    )
    ax[1, 2].set_title("Expected")

    ax[1, 3].imshow(
        difference,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        extent=[window[0], window[-1], window[0], window[-1]],
    )

    ax[0, 3].set_title(f"corrected {labels[0]} response to {labels[1]}")
    ax[0, 3].plot(
        np.linspace(window[0], window[-1], len(corrected)),
        corrected,
        color="k",
    )
    ax[0, 3].set_xlim(window[0], window[-1])
    ax[0, 3].axvline(0, ls="--", color="k")

    for a in ax[1, 1:].ravel():
        a.plot([-1, 1], [-1, 1], "k--")
        a.axvline(0, c="w", ls="--")
        a.axhline(0, c="w", ls="--")
        a.set_xlim(window[0], window[-1])
        a.set_ylim(window[0], window[-1])
    ax[0, 0].axis("off")
    ax[0, 2].axis("off")

    ax[1, 1].set_xlabel(f"{labels[1]} time from {labels[-1]} (s)")
    ax[1, 2].set_xlabel(f"{labels[1]} time from {labels[-1]} (s)")
    ax[1, 3].set_xlabel(f"{labels[1]} time from {labels[-1]} (s)")

    ax[1, 0].set_ylabel(f"{labels[0]} time from {labels[-1]} (s)")

    # turn off x ticsk
    ax[0, 1].set_xticks([])
    ax[0, 3].set_xticks([])

    ax[1, 1].set_yticks([])
    ax[1, 2].set_yticks([])
    ax[1, 3].set_yticks([])

    ax[0, 3].set_xlabel("obs - expected")

    sns.despine()

    return fig, ax
