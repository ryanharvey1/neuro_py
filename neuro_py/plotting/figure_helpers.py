from itertools import cycle
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def clean_plot3d(ax):
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.labelpad = ax.yaxis.labelpad = ax.zaxis.labelpad = 0
    ax.xaxis._axinfo["label"]["space_factor"] = 0

    return ax


def paired_lines(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    units: Optional[str] = None,
    order: Optional[List[str]] = None,
    hue_order: Optional[List[str]] = None,
    style: Optional[str] = None,
    style_order: Optional[List[str]] = None,
    style_map: Optional[Union[Dict[str, str], List[str]]] = None,
    dodge: bool = True,
    dodge_width: float = 0.2,
    color: Optional[str] = None,
    palette: Optional[Union[str, List[str], dict]] = None,
    alpha: float = 0.5,
    lw: float = 1,
    ax: Optional[matplotlib.axes.Axes] = None,
    zorder: int = 0,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """
    Draw lines connecting paired points within each x-category.

    Designed to complement seaborn boxplot/stripplot for visualizing paired data.

    Parameters
    ----------
    data : DataFrame
        Input data.
    x : str
        Column name for x-axis categories.
    y : str
        Column name for y-axis values.
    hue : str, optional
        Column to separate points within each x-category (e.g., two maze conditions).
    units : str, optional
        Column that defines which points belong together (e.g., unique_basepath).
        Required when connecting points across hue values.
    order : list, optional
        Order of x-axis categories (matches seaborn convention).
    hue_order : list, optional
        Order of hue levels. If provided, points will be connected in this order.
    style : str, optional
        Column to map to line style (e.g., linestyle). Mimics seaborn's style mapping.
    style_order : list, optional
        Order of style levels. If provided, styles will follow this order.
    style_map : dict or list, optional
        Mapping from style level to matplotlib linestyle. If a list is provided,
        it will be cycled across style levels.
    dodge : bool, default True
        Apply dodge offset like seaborn's dodge parameter.
    dodge_width : float, default 0.2
        Width of the dodge offset between hue categories.
    color : str, optional
        Line color. If None and palette is not provided, defaults to "gray".
        Ignored if palette is provided.
    palette : str, list, or dict, optional
        Color palette for lines. Can be a seaborn palette name, list of colors,
        or dict mapping units to colors. If provided, overrides color parameter.
    alpha : float, default 0.5
        Line transparency.
    lw : float, default 1
        Line width.
    ax : matplotlib Axes, optional
        Axes to plot on. Defaults to current axes.
    zorder : int, default 0
        Z-order for the lines.
    **kwargs : additional keyword arguments
        Passed to matplotlib's plot() function (e.g., linestyle, marker, etc.).

    Returns
    -------
    ax : matplotlib Axes

    Examples
    --------
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> data = pd.DataFrame({
    ...     'condition': ['A', 'B', 'A', 'B'],
    ...     'value': [1, 2, 1.5, 2.5],
    ...     'subject': ['S1', 'S1', 'S2', 'S2']
    ... })
    >>> fig, ax = plt.subplots()
    >>> paired_lines(data, x='condition', y='value', units='subject', ax=ax)
    """
    if ax is None:
        ax = plt.gca()

    # Validate required columns
    required_cols = {x, y}
    for optional_col in (hue, units, style):
        if optional_col is not None:
            required_cols.add(optional_col)
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"paired_lines: missing required columns: {missing_cols}")

    if order is None:
        order = data[x].unique()
    else:
        unknown_x = [val for val in data[x].unique() if val not in order]
        if unknown_x:
            raise ValueError(
                f"paired_lines: x contains categories not in 'order': {unknown_x}"
            )

    x_lookup = {label: i for i, label in enumerate(order)}

    # Get hue values
    if hue:
        if hue_order is None:
            hue_vals = sorted(data[hue].unique())
        else:
            hue_vals = hue_order
            unknown_hue = [val for val in data[hue].unique() if val not in hue_order]
            if unknown_hue:
                raise ValueError(
                    f"paired_lines: hue contains categories not in 'hue_order': {unknown_hue}"
                )
    else:
        hue_vals = [None]

    # Get style values and mapping
    if style:
        if style_order is None:
            style_vals = sorted(data[style].dropna().unique())
        else:
            style_vals = style_order

        if style_map is None:
            default_styles = ["-", "--", "-.", ":"]
            style_map_resolved = {
                val: sty for val, sty in zip(style_vals, cycle(default_styles))
            }
        elif isinstance(style_map, list):
            style_map_resolved = {
                val: sty for val, sty in zip(style_vals, cycle(style_map))
            }
        else:
            style_map_resolved = style_map
    else:
        style_map_resolved = None

    # Compute dodge offset
    effective_dodge_width = dodge_width if dodge and hue else 0

    # Set up color mapping
    if palette is not None:
        if isinstance(palette, str):
            # Seaborn palette name
            colors = sns.color_palette(
                palette, n_colors=len(data[units].unique()) if units else 1
            )
            if units:
                unit_vals = sorted(data[units].unique())
                color_map = dict(zip(unit_vals, colors))
            else:
                color_map = None
        elif isinstance(palette, dict):
            color_map = palette
        elif isinstance(palette, list):
            if units:
                unit_vals = sorted(data[units].unique())
                color_map = dict(zip(unit_vals, palette))
            else:
                color_map = None
        else:
            color_map = None
    else:
        color_map = None
        if color is None:
            color = "gray"

    if hue:
        # With hue: group by x and units, connect across hue values within each x-category
        if units:
            groupby_cols = [x, units]
        else:
            groupby_cols = [x]

        for group_key, g in data.groupby(groupby_cols, sort=False):
            if units:
                x_cat, unit_val = group_key
            else:
                x_cat = group_key
                unit_val = None

            if x_cat not in x_lookup:
                continue

            # Determine line color for this unit
            if color_map and unit_val is not None:
                line_color = color_map.get(unit_val, color)
            else:
                line_color = color

            # Determine line style
            plot_kwargs = dict(kwargs)
            if style_map_resolved is not None:
                style_val = g[style].iloc[0]
                line_style = style_map_resolved.get(style_val, "-")
                plot_kwargs.pop("linestyle", None)
                plot_kwargs.pop("ls", None)
                plot_kwargs["linestyle"] = line_style

            x0 = x_lookup[x_cat]

            # Get data for each hue value in order
            hue_data = []
            for hue_val in hue_vals:
                mask = g[hue] == hue_val
                if mask.any():
                    hue_data.append((hue_val, g[mask][y].values[0]))

            # Connect consecutive pairs
            if len(hue_data) >= 2:
                # Calculate x positions with dodge
                n_hue = len(hue_vals)
                if n_hue > 1:
                    hue_offsets = np.linspace(
                        -effective_dodge_width, effective_dodge_width, n_hue
                    )
                else:
                    hue_offsets = [0]

                hue_to_offset = {
                    hue_val: offset for hue_val, offset in zip(hue_vals, hue_offsets)
                }

                # Draw lines between consecutive pairs
                for i in range(len(hue_data) - 1):
                    hue_val1, y1 = hue_data[i]
                    hue_val2, y2 = hue_data[i + 1]

                    x1 = x0 + hue_to_offset[hue_val1]
                    x2 = x0 + hue_to_offset[hue_val2]

                    ax.plot(
                        [x1, x2],
                        [y1, y2],
                        color=line_color,
                        alpha=alpha,
                        lw=lw,
                        zorder=zorder,
                        **plot_kwargs,
                    )
    else:
        # No hue: group by units only, connect across x-categories
        if units:
            for unit_val, g in data.groupby(units, sort=False):
                # Determine line color for this unit
                if color_map and unit_val is not None:
                    line_color = color_map.get(unit_val, color)
                else:
                    line_color = color

                plot_kwargs = dict(kwargs)
                if style_map_resolved is not None:
                    style_val = g[style].iloc[0]
                    line_style = style_map_resolved.get(style_val, "-")
                    plot_kwargs.pop("linestyle", None)
                    plot_kwargs.pop("ls", None)
                    plot_kwargs["linestyle"] = line_style

                # Get data for each x-category in order
                x_data = []
                for x_cat in order:
                    mask = g[x] == x_cat
                    if mask.any():
                        x_pos = x_lookup[x_cat]
                        y_val = g[mask][y].values[0]
                        x_data.append((x_pos, y_val))

                # Connect consecutive points across x-categories
                if len(x_data) >= 2:
                    for i in range(len(x_data) - 1):
                        x1, y1 = x_data[i]
                        x2, y2 = x_data[i + 1]

                        ax.plot(
                            [x1, x2],
                            [y1, y2],
                            color=line_color,
                            alpha=alpha,
                            lw=lw,
                            zorder=zorder,
                            **plot_kwargs,
                        )
        else:
            # No units and no hue: just connect points in order across x-categories
            x_data = []
            for x_cat in order:
                mask = data[x] == x_cat
                if mask.any():
                    x_pos = x_lookup[x_cat]
                    y_val = data[mask][y].values[0]
                    x_data.append((x_pos, y_val))

            # Connect consecutive points
            if len(x_data) >= 2:
                for i in range(len(x_data) - 1):
                    x1, y1 = x_data[i]
                    x2, y2 = x_data[i + 1]

                    plot_kwargs = dict(kwargs)
                    if style_map_resolved is not None:
                        style_val = data[style].iloc[0] if style else None
                        line_style = style_map_resolved.get(style_val, "-")
                        plot_kwargs.pop("linestyle", None)
                        plot_kwargs.pop("ls", None)
                        plot_kwargs["linestyle"] = line_style

                    ax.plot(
                        [x1, x2],
                        [y1, y2],
                        color=color,
                        alpha=alpha,
                        lw=lw,
                        zorder=zorder,
                        **plot_kwargs,
                    )

    return ax
