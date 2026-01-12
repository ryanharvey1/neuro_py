__all__ = (
    "plot_events",
    "plot_peth",
    "plot_peth_fast",
    "set_plotting_defaults",
    "set_size",
    "lighten_color",
    "set_equal_axis_range",
    "restore_natural_scale",
    "adjust_box_widths",
    "plot_joint_peth",
    "clean_plot3d",
    "AngleAnnotation",
    "paired_lines",
)

from .decorators import (
    AngleAnnotation,
)
from .events import plot_events, plot_peth, plot_peth_fast
from .figure_helpers import (
    adjust_box_widths,
    clean_plot3d,
    lighten_color,
    plot_joint_peth,
    restore_natural_scale,
    set_equal_axis_range,
    set_plotting_defaults,
    set_size,
    paired_lines,
)
