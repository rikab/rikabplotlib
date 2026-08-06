import colorsys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from collections import namedtuple
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from matplotlib.ticker import MaxNLocator
from matplotlib.legend_handler import HandlerTuple
from matplotlib.offsetbox import TextArea, VPacker, AnnotationBbox


# Constants
DPI = 72
FULL_WIDTH_PX = 510
COLUMN_WIDTH_PX = 245

FULL_WIDTH_INCHES = FULL_WIDTH_PX / DPI
COLUMN_WIDTH_INCHES = COLUMN_WIDTH_PX / DPI

GOLDEN_RATIO = 1.618

# The style sheet color cycle, by name
COLORS = {"blue" : "#0C5DA5", "green" : "#00B945", "orange" : "#FF9500", "red" : "#FF2C00", "purple" : "#845B97", "dark" : "#474747", "gray" : "#9E9E9E"}

BLUE = COLORS["blue"]
GREEN = COLORS["green"]
ORANGE = COLORS["orange"]
RED = COLORS["red"]
PURPLE = COLORS["purple"]
DARK = COLORS["dark"]
GRAY = COLORS["gray"]

CYCLE = [BLUE, GREEN, ORANGE, RED, PURPLE, DARK, GRAY]

# What the histogram functions hand back
Histogram = namedtuple("Histogram", "counts errors edges centers artist")

# What ratio_panel hands back, dropped counts the bins with no defined value
Ratio = namedtuple("Ratio", "values errors dropped artist")


# Function to pick the style sheet matching a scale and a font backend
def set_style(scale = "full", use_tex = True):

    if scale == "full":
        name = "rikab_full" if use_tex else "rikab_full_notex"

    elif scale == "column":
        name = "rikab_column" if use_tex else "rikab_column_notex"

    else:
        raise ValueError("Invalid scale argument. Must be 'full' or 'column'.")

    plt.style.use("rikabplotlib.%s" % name)

    return name


def newplot(scale = "full", subplot_array = None, width = None, height = None, aspect_ratio = 1, golden_ratio = False, ratio = None, square = False, stamp = None, stamp_kwargs = None, use_tex = True, **kwargs):

    # Determine plot aspect ratio
    if golden_ratio:
        aspect_ratio = GOLDEN_RATIO

    set_style(scale, use_tex)

    # Width is the text width, aspect ratio is width over height
    base_width = FULL_WIDTH_INCHES if scale == "full" else COLUMN_WIDTH_INCHES
    fig_width = base_width
    fig_height = base_width / aspect_ratio

    if width is not None:
        fig_width = width
    if height is not None:
        fig_height = height

    # A body over a short pull panel per column, y shared across columns
    if ratio is not None:

        height_ratios = (3, 1) if ratio is True else ratio
        ncols = subplot_array[1] if subplot_array is not None else 1
        fig, columns = plt.subplots(1, ncols, figsize = (fig_width, fig_height), squeeze = False, **kwargs)

        main = []
        pull = []

        for column in columns[0]:

            # Split the column into a body over a pull, no gap
            cells = column.get_subplotspec().subgridspec(2, 1, height_ratios = height_ratios, hspace = 0)
            column.remove()

            top = fig.add_subplot(cells[0], sharey = main[0] if main else None)
            bottom = fig.add_subplot(cells[1], sharex = top, sharey = pull[0] if pull else None)
            top.tick_params(labelbottom = False)

            # the pull's top label would otherwise land on the body's bottom label
            bottom.yaxis.set_major_locator(MaxNLocator(nbins = 4, prune = "upper"))

            # Box the body and pull into one square
            if square:

                total = height_ratios[0] + height_ratios[1]
                top.set_box_aspect(height_ratios[0] / total)
                top.set_anchor("S")
                bottom.set_box_aspect(height_ratios[1] / total)
                bottom.set_anchor("N")

            main.append(top)
            pull.append(bottom)

        ax = main[0]

    # A plain grid or a single panel
    elif subplot_array is not None:
        fig, ax = plt.subplots(subplot_array[0], subplot_array[1], figsize = (fig_width, fig_height), **kwargs)

    else:
        fig, ax = plt.subplots(figsize = (fig_width, fig_height), **kwargs)

    # Plot title
    if stamp is not None:

        reference_ax = np.ravel(ax)[0]
        stamp_kwargs_default = {"style" : 'italic', "horizontalalignment" : 'right', "verticalalignment" : 'bottom', "transform" : reference_ax.transAxes}

        if stamp_kwargs is not None:
            stamp_kwargs_default.update(stamp_kwargs)

        # Text in the top right corner, right aligned:
        reference_ax.text(1, 1, stamp, **stamp_kwargs_default)

    # Body and pull per column when a ratio was asked, else the plain axes
    if ratio is not None:

        if ncols == 1:
            return fig, main[0], pull[0]

        return fig, main, pull

    return fig, ax


def add_whitespace(ax = None, upper_fraction = 1.333, lower_fraction = 1):

    # handle defualt axis
    if ax is None:
        ax = plt.gca()

    # check if log scale
    scale_str = ax.get_yaxis().get_scale()

    bottom, top = ax.get_ylim()

    if scale_str == "log":
        upper_fraction = np.power(10, upper_fraction - 1)
        lower_fraction = np.power(10, lower_fraction - 1)

    ax.set_ylim([bottom / lower_fraction, top * upper_fraction])


# function to add a stamp to figures
def stamp(left_x, top_y,
          ax=None,
          delta_y=0.06,
          textops_update=None,
          boldfirst = True,
          **kwargs):

     # handle defualt axis
    if ax is None:
        ax = plt.gca()

    # text options
    textops = {'horizontalalignment': 'left',
               'verticalalignment': 'center',
               'transform': ax.transAxes}
    if isinstance(textops_update, dict):
        textops.update(textops_update)

    # the line_N that were actually passed, in order
    indices = sorted(int(key[len("line_"):]) for key in kwargs if key.startswith("line_"))

    # add text line by line
    for row, index in enumerate(indices):

        y = top_y - row * delta_y
        t = kwargs["line_" + str(index)]

        if t is None:
            continue

        # bold the first line, through TeX only when TeX is drawing the text
        if boldfirst and row == 0:

            text = r"\textbf{%s}" % t if plt.rcParams["text.usetex"] else t
            ax.text(left_x, y, text, weight = "bold", **textops)

        else:
            ax.text(left_x, y, t, **textops)


# Function to stash a proxy artist, covers are the artists whose own keys it replaces
def register_handle(ax, handle, label, covers = None):

    if label is None:
        return

    if not hasattr(ax, "rikab_handles"):
        ax.rikab_handles = []

    ax.rikab_handles.append((handle, label, list(covers) if covers is not None else []))


# Function to draw a legend whose keys match multi-artist drawings
def legend(ax = None, **kwargs):

    if ax is None:
        ax = plt.gca()

    registered = getattr(ax, "rikab_handles", [])

    # an artist standing behind a proxy does not get a second key of its own
    covered = set()
    for handle, label, covers in registered:
        for artist in list(handle if isinstance(handle, tuple) else [handle]) + covers:
            covered.add(id(artist))

    found = [(handle, label) for handle, label in zip(*ax.get_legend_handles_labels()) if id(handle) not in covered]

    handles = [handle for handle, label in found] + [handle for handle, label, covers in registered]
    labels = [label for handle, label in found] + [label for handle, label, covers in registered]

    # a tuple of artists draws as one overlaid key, not as slices of one
    handler_map = dict(kwargs.pop("handler_map", {}))
    handler_map.setdefault(tuple, HandlerTuple(ndivide = 1))

    return ax.legend(handles, labels, handler_map = handler_map, **kwargs)


def plot_event(ax, event, R = 1.0, filename = None, color = "red", values = None, cmap = None, colorbar = False, show_circle = True, title = "", label = "Event", show = False):

    pts, ys, phis = event[:, 0], event[:, 1], event[:, 2]

    # marker area tracks the pt fraction the particle carries
    sizes = 2 * pts * 500 / np.sum(pts)

    if values is not None:
        points = ax.scatter(ys, phis, marker = 'o', s = sizes, c = values, cmap = cmap, edgecolors = "none", zorder = 10)

    else:
        points = ax.scatter(ys, phis, marker = 'o', s = sizes, color = color, edgecolors = "none", zorder = 10)

    # a legend key at a readable fixed size rather than the last particle's pt
    register_handle(ax, Line2D([], [], marker = 'o', linestyle = "none", color = "black" if values is not None else color), label)

    if show_circle:
        ax.add_patch(Circle((0, 0), R, fill = False, edgecolor = "black", linestyle = "--", zorder = 5))

    # colorbar takes its label from the argument when a string was passed
    if colorbar and values is not None:

        bar = ax.figure.colorbar(points, ax = ax)
        bar.set_label(colorbar if isinstance(colorbar, str) else "")

    # plot settings
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_xlabel('Rapidity')
    ax.set_ylabel('Azimuthal Angle')
    ax.set_xticks(np.linspace(-R, R, 5))
    ax.set_yticks(np.linspace(-R, R, 5))
    ax.set_aspect('equal')

    if title:
        ax.set_title(title)

    if filename:
        ax.figure.savefig(filename)

    if show:
        plt.show()

    return ax


    # Function to take a list of points and create a histogram of points with sqrt(N) errors, normalized to unit area
def hist_with_errors(ax, points, bins, range, weights = None, show_zero = False, show_errors = True, label = None, **kwargs):

    points = np.asarray(points, dtype = float)

    if weights is None:
        weights = np.ones_like(points)

    counts, bin_edges = np.histogram(points, bins = bins, range = range, weights = weights)
    errs2 = np.histogram(points, bins = bins, range = range, weights = weights**2)[0]

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = (bin_edges[1:] - bin_edges[:-1])

    # Check if density is a keyword argument
    density = kwargs.pop("density", False)

    # unit area divides the counts by the norm and the squared error by its square
    if density:

        norm = np.sum(counts * bin_widths)
        errs2 = errs2 / norm**2
        counts = counts / norm

    errors = np.sqrt(errs2)

    # empty bins carry nothing, negative bins always survive
    mask = np.ones_like(counts, dtype = bool) if show_zero else counts != 0

    if show_errors:

        # bars at the full style line width swallow the marker they belong to
        kwargs.setdefault("elinewidth", plt.rcParams["lines.linewidth"] / 3)

        artist = ax.errorbar(bin_centers[mask], counts[mask], errors[mask], xerr = bin_widths[mask] / 2, fmt = "o", label = label, **kwargs)

    else:
        artist = ax.scatter(bin_centers[mask], counts[mask], label = label, **kwargs)

    return Histogram(counts, errors, bin_edges, bin_centers, artist)


def hist_with_outline(ax, points, bins, range, weights = None, color = "purple", alpha_1 = 0.25, alpha_2 = 0.75, label = None,  **kwargs):

    points = np.asarray(points, dtype = float)

    if weights is None:
        weights = np.ones_like(points)

    counts, bin_edges = np.histogram(points, bins = bins, range = range, weights = weights)
    errors = np.sqrt(np.histogram(points, bins = bins, range = range, weights = weights**2)[0])

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = (bin_edges[1:] - bin_edges[:-1])

    # the drawn hists normalize themselves, so the returned counts have to follow
    if kwargs.get("density", False):

        norm = np.sum(counts * bin_widths)
        errors = errors / norm
        counts = counts / norm

    ax.hist(points, bins = bins, range = range, weights = weights, color = color, alpha = alpha_1, histtype='stepfilled', **kwargs)
    outline = ax.hist(points, bins = bins, range = range, weights = weights, color = color, alpha = alpha_2, histtype='step', label = label, **kwargs)[2]

    # a filled swatch with its outline, in place of the bare step line
    proxy = Patch(facecolor = mcolors.to_rgba(color, alpha_1), edgecolor = mcolors.to_rgba(color, alpha_2))
    register_handle(ax, proxy, label, covers = np.ravel(outline))

    return Histogram(counts, errors, bin_edges, bin_centers, proxy)


# Function to repeat the last bin so a step drawing closes on the right edge
def extend_last_bin(counts):

    return np.append(counts, counts[-1])


# Function to draw several components stacked, one color per component
def hist_stack(ax, components, bins, range, weights = None, colors = None, labels = None, alpha_1 = 0.6, alpha_2 = 1.0):

    colors = palette(len(components)) if colors is None else colors
    labels = [None] * len(components) if labels is None else labels
    weights = [None] * len(components) if weights is None else weights

    bin_edges = np.histogram_bin_edges(components[0], bins = bins, range = range)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    running = np.zeros(len(bin_centers))
    errs2 = np.zeros(len(bin_centers))

    # each component sits on the running total of the ones under it
    for points, weight, color, label in zip(components, weights, colors, labels):

        points = np.asarray(points, dtype = float)
        weight = np.ones_like(points) if weight is None else np.asarray(weight, dtype = float)

        counts = np.histogram(points, bins = bin_edges, weights = weight)[0]
        errs2 = errs2 + np.histogram(points, bins = bin_edges, weights = weight**2)[0]
        top = running + counts

        ax.fill_between(bin_edges, extend_last_bin(running), extend_last_bin(top), step = "post", color = color, alpha = alpha_1, edgecolor = "none")
        ax.step(bin_edges, extend_last_bin(top), where = "post", color = color, alpha = alpha_2)

        register_handle(ax, Patch(facecolor = mcolors.to_rgba(color, alpha_1), edgecolor = mcolors.to_rgba(color, alpha_2)), label)
        running = top

    return Histogram(running, np.sqrt(errs2), bin_edges, bin_centers, None)


# Function to fill a pull panel from a numerator and a denominator
def ratio_panel(ax, x, numerator, denominator, numerator_errors = None, denominator_errors = None, mode = "ratio", color = None, reference = True, label = None, **kwargs):

    x = np.asarray(x, dtype = float)
    numerator = np.asarray(numerator, dtype = float)
    denominator = np.asarray(denominator, dtype = float)

    # a side with no errors is a model taken as exact
    numerator_errors = np.zeros_like(numerator) if numerator_errors is None else np.asarray(numerator_errors, dtype = float)
    denominator_errors = np.zeros_like(denominator) if denominator_errors is None else np.asarray(denominator_errors, dtype = float)

    with np.errstate(divide = "ignore", invalid = "ignore"):

        # dividing only ever by the denominator keeps empty numerator bins alive
        if mode == "ratio":

            values = numerator / denominator
            errors = np.sqrt((numerator_errors / denominator)**2 + (numerator * denominator_errors / denominator**2)**2)
            baseline = 1

        elif mode == "difference":

            values = numerator - denominator
            errors = np.sqrt(numerator_errors**2 + denominator_errors**2)
            baseline = 0

        elif mode == "pull":

            values = (numerator - denominator) / np.sqrt(numerator_errors**2 + denominator_errors**2)
            errors = None
            baseline = 0

        else:
            raise ValueError("mode must be 'ratio', 'difference', or 'pull'")

    # zero denominators and undefined pulls have no place on the panel
    finite = np.isfinite(values) if errors is None else np.isfinite(values) & np.isfinite(errors)
    dropped = int(np.sum(~finite))

    if reference:
        ax.axhline(baseline, color = "black", zorder = 1)

    kwargs.setdefault("elinewidth", plt.rcParams["lines.linewidth"] / 3)
    artist = ax.errorbar(x[finite], values[finite], None if errors is None else errors[finite], fmt = "o", color = color, label = label, **kwargs)

    return Ratio(values, errors, dropped, artist)


def function_with_band(ax, f, range, params, pcov = None, color = "purple", alpha_line = 0.75, alpha_band = 0.25, label = None, rng = None, samples = 1000, **kwargs):

    x = np.linspace(range[0], range[1], 1000)
    band = None

    if pcov is not None:

        # Vary the parameters within their errors
        rng = np.random.default_rng() if rng is None else rng
        varied_params = rng.multivariate_normal(params, pcov, samples)
        y = np.array([f(x, *p) for p in varied_params])

        # Plot the band
        y_mean = np.mean(y, axis = 0)
        y_std = np.std(y, axis = 0)

        band = ax.fill_between(x, y_mean - y_std, y_mean + y_std, color = color, alpha = alpha_band, edgecolor = "none")

    y = f(x, *params)
    line = ax.plot(x, y, color = color, alpha = alpha_line, label = label, **kwargs)[0]

    # one key covering both the line and its band
    register_handle(ax, line if band is None else (band, line), label)

    return band, line


# Function to plot a central line with a spread band over per-x samples
def line_with_band(ax, x, samples, mode = "median", color = "purple", alpha_band = 0.16, label = None, **kwargs):

    samples = [np.asarray(s) for s in samples]

    if mode == "median":

        center = np.array([np.median(s) for s in samples])
        lo = np.array([np.quantile(s, 0.25) for s in samples])
        hi = np.array([np.quantile(s, 0.75) for s in samples])

    elif mode == "mean":

        center = np.array([np.mean(s) for s in samples])
        spread = np.array([np.std(s) for s in samples])
        lo = center - spread
        hi = center + spread

    else:
        raise ValueError("mode must be 'median' or 'mean'")

    band = ax.fill_between(x, lo, hi, color = color, alpha = alpha_band, edgecolor = "none")
    line = ax.plot(x, center, color = color, label = label, **kwargs)[0]

    # one key covering both the line and its band
    register_handle(ax, (band, line), label)

    return band, line


# Function to take the first n colors of the style sheet cycle
def palette(n):

    return [CYCLE[i % len(CYCLE)] for i in range(n)]


# Function to blend a color toward black or white, 0 = color, 1 = target
def shade(color, fraction, toward = "black"):

    base = np.array(mcolors.to_rgb(color))
    target = np.zeros(3) if toward == "black" else np.ones(3)

    return (1 - fraction) * base + fraction * target


# Function to make n colors ramped from a base color toward the target
def ramp(color, n, toward = "black", start = 0.0, stop = 1.0):

    return [shade(color, start + (stop - start) * i / max(n - 1, 1), toward) for i in range(n)]


# Function to mix two colors, 0 = a and 1 = b, in hue or straight in rgb
def blend(color_a, color_b, fraction, space = "hue"):

    rgb_a = np.array(mcolors.to_rgb(color_a))
    rgb_b = np.array(mcolors.to_rgb(color_b))

    if space == "rgb":
        return (1 - fraction) * rgb_a + fraction * rgb_b

    if space != "hue":
        raise ValueError("space must be 'hue' or 'rgb'")

    hue_a, saturation_a, value_a = colorsys.rgb_to_hsv(*rgb_a)
    hue_b, saturation_b, value_b = colorsys.rgb_to_hsv(*rgb_b)

    # a gray endpoint has no hue of its own, so it borrows the other one
    if saturation_a == 0:
        hue_a = hue_b
    if saturation_b == 0:
        hue_b = hue_a

    # walk the shorter arc between the two hues
    delta = (hue_b - hue_a + 0.5) % 1.0 - 0.5
    hue = (hue_a + fraction * delta) % 1.0

    return np.array(colorsys.hsv_to_rgb(hue, (1 - fraction) * saturation_a + fraction * saturation_b, (1 - fraction) * value_a + fraction * value_b))


# Function to make n colors running from one color to another
def gradient(color_a, color_b, n, space = "hue"):

    return [blend(color_a, color_b, i / max(n - 1, 1), space) for i in range(n)]


# Function to rotate a color around the hue circle, delta in turns
def hue_shift(color, delta):

    hue, saturation, value = colorsys.rgb_to_hsv(*mcolors.to_rgb(color))

    return np.array(colorsys.hsv_to_rgb((hue + delta) % 1.0, saturation, value))


# Function to make a continuous colormap, start and stop are shade fractions like in ramp
def colormap(color, toward = "white", start = 1.0, stop = 0.0, name = "rikab"):

    return mcolors.LinearSegmentedColormap.from_list(name, [shade(color, start, toward), shade(color, stop, toward)])


# Function to make a colormap running from one color through white to another
def diverging_colormap(color_low, color_high, name = "rikab_diverging"):

    return mcolors.LinearSegmentedColormap.from_list(name, [mcolors.to_rgb(color_low), (1.0, 1.0, 1.0), mcolors.to_rgb(color_high)])


# Function to stack labeled lines in one framed box, sizes from the active style
def badge(ax, lines, xy = (0.04, 0.96), loc = (0, 1), colors = None, weights = None, sizes = None, sep = 3):

    n = len(lines)
    colors = colors if colors is not None else ["black"] * n
    weights = weights if weights is not None else ["normal"] * n
    sizes = sizes if sizes is not None else [plt.rcParams["axes.labelsize"]] * n

    children = [TextArea(text, textprops = dict(color = colors[i], weight = weights[i], size = sizes[i])) for i, text in enumerate(lines)]
    stacked = VPacker(children = children, align = "center", pad = 0, sep = sep)

    box = AnnotationBbox(stacked, xy, xycoords = "axes fraction", box_alignment = loc, frameon = True, bboxprops = dict(boxstyle = "round,pad=0.4", facecolor = "white", edgecolor = "black"))
    ax.add_artist(box)

    return box
