from rikabplotlib.plot_utils import newplot, set_style, stamp, badge, legend, register_handle, add_whitespace, plot_event
from rikabplotlib.plot_utils import hist_with_errors, hist_with_outline, hist_stack, ratio_panel, function_with_band, line_with_band
from rikabplotlib.plot_utils import shade, ramp, blend, gradient, hue_shift, palette, colormap, diverging_colormap
from rikabplotlib.plot_utils import COLORS, CYCLE, BLUE, GREEN, ORANGE, RED, PURPLE, DARK, GRAY, Histogram, Ratio

# the wheel writes _version.py at build time, a bare source tree has only the metadata
try:
    from importlib.metadata import version
    __version__ = version("rikabplotlib")

except Exception:
    from rikabplotlib._version import __version__
