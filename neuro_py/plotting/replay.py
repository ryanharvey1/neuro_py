"""
plot_2d_replay.py
-----------------
Plot a 2D replay event from a (nx, ny, T) probability matrix.
Colors encode elapsed time within the replay using a colormap.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def plot_2d_replay(
    replay_matrix,
    ax=None,
    cmap="cool",
    extent=None,
    saturation=0.5,
    percentile_threshold=99,
    abs_threshold=1e-6,
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

    Returns
    -------
    fig, ax
    """
    if saturation <= 0:
        raise ValueError("saturation must be > 0")

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.get_figure()

    nx, ny, T = replay_matrix.shape
    colormap = matplotlib.colormaps[cmap]
    global_max = replay_matrix.max()

    if extent is None:
        extent = [0, nx, 0, ny]
    xmin, xmax, ymin, ymax = extent

    for t in range(T):
        frame = replay_matrix[:, :, t].T  # (ny, nx) — row=y for imshow
        frame = frame / frame.sum()

        # Combined threshold: percentile AND absolute floor
        thr = max(np.percentile(frame, percentile_threshold), abs_threshold)
        frame = np.where(frame >= thr, frame, 0)

        if frame.max() == 0:
            continue

        # Skip frames with no spatial structure
        all_close = np.allclose(frame, frame.flat[0])
        if all_close:
            continue

        rgb = np.array(colormap(t / max(T - 1, 1))[:3])
        alpha = np.power(frame / frame.max(), 1.0 / saturation)

        rgba = np.zeros((*frame.shape, 4))
        rgba[..., :3] = rgb
        rgba[..., 3] = alpha

        ax.imshow(rgba, origin="lower", extent=extent, aspect="equal")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax
