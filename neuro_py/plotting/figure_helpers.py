
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from lazy_loader import attach as _attach
from matplotlib.patches import PathPatch

__all__ = (
    "set_plotting_defaults",
    "set_size",
    "lighten_color",
    "set_equal_axis_range",
    "restore_natural_scale",
    "adjust_box_widths",
    "plot_joint_peth",
)
__getattr__, __dir__, __all__ = _attach(f"{__name__}", submodules=__all__)
del _attach


def set_plotting_defaults():
    tex_fonts = {
        #     # Use LaTeX to write all text
        "font.family": "serif",
        # Use 10pt font in plots
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "svg.fonttype": "none",
    }
    
    plt.style.use("default")
    plt.rcParams.update(tex_fonts)
    


def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.
    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
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


def lighten_color(color, amount=0.5):
    """
    Lightens a color by a certain percentage.
    This is useful for adjusting colors for a particular element of a page.
    :param color: The hex color code, e.g. #AABBCC
    :param amount: The amount to lighten the color by.
    :return: The lightened color code in hex, e.g. #FFFFFF
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


def set_equal_axis_range(ax1, ax2):
    """
    Makes x and y min and max the same between two plots
    """
    axis_x_values = np.hstack(np.array((ax1.get_xlim(), ax2.get_xlim())))
    axis_y_values = np.hstack(np.array((ax1.get_ylim(), ax2.get_ylim())))
    ax1.set_xlim(axis_x_values.min(), axis_x_values.max())
    ax1.set_ylim(axis_y_values.min(), axis_y_values.max())
    ax2.set_xlim(axis_x_values.min(), axis_x_values.max())
    ax2.set_ylim(axis_y_values.min(), axis_y_values.max())


def restore_natural_scale(ax, min_, max_, n_steps=4, x_axis=True, y_axis=True):
    """
    takes x and y ax that are in log10 and puts them into natural scale

    By default, it adjusts both x and y, but you can run this on a single
    axis or two times if you have different scales for x and y
    """
    ticks = np.linspace(min_, max_, n_steps)

    if x_axis:
        ax.set_xticks(ticks)
        ax.set_xticklabels(np.round(10**ticks, 3))

    if y_axis:
        ax.set_yticks(ticks)
        ax.set_yticklabels(np.round(10**ticks, 3))


def adjust_box_widths(g, fac):
    """
    Adjust the widths of a seaborn-generated boxplot.
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
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])


def plot_joint_peth(
    peth_1: np.ndarray,
    peth_2: np.ndarray,
    ts: np.ndarray,
    smooth_std: float = 2,
    labels: list = ["peth_1", "peth_2", "event"],
):
    """
    Plot joint peri-event time histograms (PETHs) and the difference between the observed and expected response.

    Parameters
    ----------
    peth_1 : np.ndarray
        Peri-event time histogram (PETH) for the first event. (n events x time).
    peth_2 : np.ndarray
        Peri-event time histogram (PETH) for the second event. (n events x time).
    ts : np.ndarray
        Time vector for the PETHs.
    smooth_std : float, optional
        Standard deviation of the Gaussian kernel used to smooth the PETHs.
    labels : list, optional
        Labels for the PETHs.

    Returns
    -------
    fig : plt.Figure
        Figure object.
    ax : np.ndarray
        Axes objects.


    """
    from neuro_py.process.peri_event import joint_peth
    from neuro_py.process.utils import avgerage_diagonal

    window = [ts[0], ts[-1]]

    joint, expected, difference = joint_peth(peth_1, peth_2, smooth_std=smooth_std)

    # get average of diagonals
    corrected = avgerage_diagonal(difference.T)
    # get center values of corrected_2
    corrected = corrected[difference.shape[1] // 2:(difference.shape[1] // 2) + difference.shape[1]]


    fig, ax = plt.subplots(
        2,
        4,
        figsize=(12, 4),
        gridspec_kw={"width_ratios": [0.25, 1, 1, 1], "height_ratios": [0.25, 1]},
    )
    # space between panels
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    f = ax[1, 1].imshow(
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
