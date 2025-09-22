import numpy as np
import matplotlib.pyplot as plt



# Constants
DPI = 72
FULL_WIDTH_PX = 510
COLUMN_WIDTH_PX = 245

FULL_WIDTH_INCHES = FULL_WIDTH_PX / DPI
COLUMN_WIDTH_INCHES = COLUMN_WIDTH_PX / DPI

GOLDEN_RATIO = 1.618

def newplot(scale = None, subplot_array = None, width = None, height = None, aspect_ratio = 1,  golden_ratio = False, stamp = None, stamp_kwargs = None, use_tex = True, **kwargs):


    # Determine plot aspect ratio
    if golden_ratio:
        aspect_ratio = GOLDEN_RATIO

    # Determine plot size if not directly set
    if scale is None:
        plot_scale = "full"
    if scale == "full":
        fig_width = FULL_WIDTH_INCHES / aspect_ratio
        fig_height = FULL_WIDTH_INCHES 
        plt.style.use('rikabplotlib.rikab_full')



        if use_tex:
            plt.style.use('rikabplotlib.rikab_full')

        else:
            plt.style.use('rikabplotlib.rikab_full_notex')


    elif scale == "column":
        fig_width = COLUMN_WIDTH_INCHES / aspect_ratio
        fig_height = COLUMN_WIDTH_INCHES 
        plt.style.use('rikabplotlib.rikab_column')
    else:
        raise ValueError("Invalid scale argument. Must be 'full' or 'column'.")


    if width is not None:
        fig_width = width
    if height is not None:
        fig_height = height

    if subplot_array is not None:
        fig, ax = plt.subplots(subplot_array[0], subplot_array[1], figsize=(fig_width, fig_height), **kwargs)
        stamp_kwargs_default = {"style" : 'italic', "horizontalalignment" : 'right', "verticalalignment" : 'bottom', "transform" : ax[0].transAxes}

    else:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), **kwargs)
        stamp_kwargs_default = {"style" : 'italic', "horizontalalignment" : 'right', "verticalalignment" : 'bottom', "transform" : ax.transAxes}


    # Plot title
    if stamp_kwargs is not None:
        stamp_kwargs_default.update(stamp_kwargs)

    if stamp is not None:
        # Text in the top right corner, right aligned:
        plt.text(1, 1, stamp, **stamp_kwargs_default)



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
    
    # add text line by line
    for i in range(len(kwargs)):
        y = top_y - i*delta_y
        t = kwargs.get('line_' + str(i))


        if t is not None:
            if boldfirst and i == 0:
                ax.text(left_x, y, r"$\textbf{%s}$" % t, weight='bold', **textops)
            else:
                ax.text(left_x, y, t, **textops)



def plot_event(ax, event, R, filename=None, color="red", title="", show=True):


    pts, ys, phis =event[:,0], event[:, 1], event[:, 2]
    ax.scatter(ys, phis, marker='o', s=2 * pts * 500/np.sum(pts), color=color, lw=0, zorder=10, label="Event")

    # Legend
    # legend = plt.legend(loc=(0.1, 1.0), frameon=False, ncol=3, handletextpad=0)
    # legend.legendHandles[0]._sizes = [150]

    # plot settings
    plt.xlim(-R, R)
    plt.ylim(-R, R)
    plt.xlabel('Rapidity')
    plt.ylabel('Azimuthal Angle')
    plt.title(title)
    plt.xticks(np.linspace(-R, R, 5))
    plt.yticks(np.linspace(-R, R, 5))

    ax.set_aspect('equal')
    if filename:
        plt.savefig(filename)
        plt.show()
        plt.close()
        return ax
    elif show:
        plt.show()
        return ax
    else:
        return ax
    


    # Function to take a list of points and create a histogram of points with sqrt(N) errors, normalized to unit area
def hist_with_errors(ax, points, bins, range, weights = None, show_zero = False, show_errors = True, label = None, **kwargs):

    if weights is None:
        weights = np.ones_like(points)

    hist, bin_edges = np.histogram(points, bins = bins, range = range, weights = weights)
    errs2 = np.histogram(points, bins = bins, range = range, weights = weights**2)[0]

    # Check if density is a keyword argument
    density = kwargs.pop("density", False)

    if density:
        bin_widths = (bin_edges[1:] - bin_edges[:-1])
        errs2 = errs2 / (np.sum(hist * bin_widths))
        hist = hist / np.sum(hist * bin_widths)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = (bin_edges[1:] - bin_edges[:-1])

    if show_errors:
        ax.errorbar(bin_centers[hist > 0], hist[hist > 0], np.sqrt(errs2[hist > 0]), xerr = bin_widths[hist > 0] / 2, fmt = "o", label = label, **kwargs)
    else:
        ax.scatter(bin_centers[hist > 0], hist[hist > 0], label = label, **kwargs)


def hist_with_outline(ax, points, bins, range, weights = None, color = "purple", alpha_1 = 0.25, alpha_2 = 0.75, label = None,  **kwargs):
    
    if weights is None:
        weights = np.ones_like(points)

    ax.hist(points, bins = bins, range = range, weights = weights, color = color, alpha = alpha_1, histtype='stepfilled', **kwargs)
    ax.hist(points, bins = bins, range = range, weights = weights, color = color, alpha = alpha_2, histtype='step', label = label, **kwargs)


    # # # Dummy plot for legend
    # if label is not None:

    #     edgecolor = mpl.colors.colorConverter.to_rgba(color, alpha=alpha_2)
    #     ax.hist(points, bins = bins, range = range, weights = weights * -1, color = color, alpha = alpha_1, lw = lw*2, label = label, edgecolor = edgecolor, **kwargs)






def function_with_band(ax, f, range, params, pcov = None, color = "purple", alpha_line = 0.75, alpha_band = 0.25, lw = 3,  **kwargs):

    x = np.linspace(range[0], range[1], 1000)

    if pcov is not None:

        # Vary the parameters within their errors
        n = 1000
        temp_params = np.random.multivariate_normal(params, pcov, n)
        y = np.array([f(x, *p) for p in temp_params])

        # Plot the band

        y_mean = np.mean(y, axis = 0)
        y_std = np.std(y, axis = 0) 

        ax.fill_between(x, y_mean - y_std, y_mean + y_std, color = color, alpha = alpha_band, **kwargs)


    y = f(x, *params)
    ax.plot(x, y, color = color, alpha = alpha_line, lw = lw, **kwargs)