from rikabplotlib.plot_utils import newplot, set_style, stamp, badge, legend, register_handle, add_whitespace, plot_event
from rikabplotlib.plot_utils import hist_with_errors, hist_with_outline, hist_stack, ratio_panel, function_with_band, line_with_band
from rikabplotlib.plot_utils import shade, ramp, blend, gradient, hue_shift, palette, colormap, diverging_colormap
from rikabplotlib.plot_utils import COLORS, CYCLE, BLUE, GREEN, ORANGE, RED, PURPLE, DARK, GRAY, Histogram, Ratio

# the build writes _version.py next to this file, so it tracks the code actually imported
try:
    from rikabplotlib._version import __version__

except ImportError:

    # a bare git clone has no _version.py, fall back to whatever is installed
    try:
        from importlib.metadata import version
        __version__ = version("rikabplotlib")

    except Exception:
        __version__ = "0.0.0.dev0"
