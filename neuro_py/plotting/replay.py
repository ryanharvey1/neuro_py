"""
plot_2d_replay.py
-----------------
Plot a 2D replay event from a (nx, ny, T) probability matrix.
Colors encode elapsed time within the replay using a colormap.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_2d_replay(
    replay_matrix,
    ax=None,
    cmap="cool",
    extent=None,
    saturation=3,
    percentile_threshold=99,
    abs_threshold=None,
    per_frame_alpha_normalization=True,
):
    """
    Plot a single 2D replay event.

    Each time bin is drawn as a separate RGBA layer; matplotlib composites
    them naturally. Color encodes elapsed time within the replay (early→late
    following the chosen colormap). Alpha is power-scaled by each frame's
    probability relative to the global max, which preserves relative intensity
    across bins and avoids shadows from low-probability tails.

    Parameters
    ----------
    replay_matrix : np.ndarray, shape (nx, ny, T)
        Decoded probability distributions over space. Each [:, :, t] slice
        should be non-negative and ideally sum to ~1.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, a new figure is created.
    cmap : str or Colormap
        Colormap used to color time bins. Default "cool" gives cyan→magenta.
    extent : array-like [xmin, xmax, ymin, ymax], optional
        Spatial extent in data coordinates. Defaults to bin indices.
    saturation : float, > 0
        Controls how much of the probability distribution is visible via
        alpha = (p / frame.max()) ** (1 / saturation).
        saturation=1  → exponent=1, alpha scales linearly with probability.
        saturation<1  → exponent>1, low-probability regions fade faster (sparse).
        saturation>1  → exponent<1, low-probability regions boosted (dense/flat).
    percentile_threshold : float
        Per-frame values below this percentile are zeroed out.
        Combined with abs_threshold — both must pass.
    abs_threshold : float
        Absolute floor applied alongside percentile_threshold.
        Prevents near-zero values in sparse frames from leaking through.
    per_frame_alpha_normalization : bool
        If True, alpha is normalized by each frame's max. If False, alpha is normalized by the global max across all frames.
        The latter preserves relative intensity across frames, but may cause low-probability frames to be very faint.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes containing the replay plot.

    Examples
    --------
    # Figure 1: three replay types
    >>> fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    >>> for ax, kind, title in zip(
    ...    axes, ["linear", "curved", "diffuse"], ["Linear", "Curved", "Diffuse (wide)"]
    ... ):
    >>>    plot_2d_replay(make_replay(kind=kind), ax=ax, saturation=0.5)
    >>>    ax.set_title(title)
    >>> fig.suptitle("Replay types", y=1.02)
    >>> fig.tight_layout()
    >>> plt.show()

    # Figure 2: saturation comparison
    >>> mat = make_replay(kind="curved")
    >>> sat_values = [0.1, 0.5, 1.0, 2.0]

    >>> fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    >>> for ax, sat in zip(axes, sat_values):
    >>>     plot_2d_replay(mat, ax=ax, saturation=sat)
    >>>     ax.set_title(f"saturation={sat}")
    >>> fig.suptitle("Saturation comparison", y=1.02)
    >>> fig.tight_layout()
    >>> plt.show()

    # Figure 3: per-frame vs global alpha normalization
    >>> fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    >>> plot_2d_replay(mat, ax=axes[0], per_frame_alpha_normalization=True)
    >>> axes[0].set_title("Per-frame normalization")
    >>> plot_2d_replay(mat, ax=axes[1], per_frame_alpha_normalization=False)
    >>> axes[1].set_title("Global normalization")
    >>> fig.tight_layout()
    >>> plt.show()



    """
    if saturation <= 0:
        raise ValueError("saturation must be > 0")

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.get_figure()

    nx, ny, T = replay_matrix.shape
    colormap = matplotlib.colormaps[cmap]

    # convert each frame to a probability distribution (if not already), and find global max for consistent alpha scaling
    replay_matrix = replay_matrix.copy()
    for t in range(T):
        replay_matrix[:, :, t] = replay_matrix[:, :, t] / replay_matrix[:, :, t].sum()

    global_max = replay_matrix.max()

    if extent is None:
        extent = [0, nx, 0, ny]
    xmin, xmax, ymin, ymax = extent

    for t in range(T):
        frame = replay_matrix[:, :, t].T  # (ny, nx) — row=y for imshow

        # Skip frames with no spatial structure
        if frame.mean() > 0 and np.std(frame) / frame.mean() < 1e-3:
            continue

        if frame.max() == 0:
            continue

        # Combined threshold: percentile AND absolute floor
        thr = np.percentile(frame, percentile_threshold)
        if abs_threshold is not None:
            thr = max(thr, abs_threshold)

        frame = np.where(frame >= thr, frame, 0)

        if frame.max() == 0:
            continue

        rgb = np.array(colormap(t / max(T - 1, 1))[:3])
        if per_frame_alpha_normalization:
            alpha = np.power(frame / frame.max(), 1.0 / saturation)
        else:
            alpha = np.power(frame / global_max, 1.0 / saturation)

        rgba = np.zeros((*frame.shape, 4))
        rgba[..., :3] = rgb
        rgba[..., 3] = alpha

        ax.imshow(rgba, origin="lower", extent=extent, aspect="equal")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    return fig, ax


def make_replay(nx=50, ny=50, T=15, kind="linear", seed=42):
    """
    kind : "linear"   — straight trajectory across arena
           "curved"   — arc trajectory
           "diffuse"  — wide, uncertain posteriors (stress-tests saturation)

    Notes
    -----
    - This is just a helper function to generate demo replay matrices for the tutorial docstring.
    """
    rng = np.random.default_rng(seed)
    matrix = np.zeros((nx, ny, T))

    if kind == "linear":
        xs = np.linspace(8, 42, T)
        ys = np.linspace(8, 42, T)
        sigma = 3
    elif kind == "curved":
        theta = np.linspace(0, np.pi, T)
        xs = 25 + 18 * np.cos(theta)
        ys = 15 + 18 * np.sin(theta)
        sigma = 3
    elif kind == "diffuse":
        xs = np.linspace(8, 42, T) + rng.normal(0, 2, T)
        ys = np.linspace(42, 8, T) + rng.normal(0, 2, T)
        sigma = 8

    yi, xi = np.mgrid[0:ny, 0:nx]
    for t in range(T):
        matrix[:, :, t] = np.exp(
            -((xi - xs[t]) ** 2 + (yi - ys[t]) ** 2) / (2 * sigma**2)
        )

    return matrix
