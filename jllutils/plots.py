"""
Helpful generic plotting functions.
"""

import itertools
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import string

from .stats import hist2d


class IncompatibleOptions(Exception):
    """
    Error to use when incompatible options are passed
    """
    pass


class Subplots(object):
    """Helper class to create subplots given the number of plots needed

    This class is intended to simplify the case where you know how many 
    subplots you need total and want to be able to create axes as needed.

    Attributes
    ----------
    axes : numpy.ndarray
        An array of the axes created. The value for a given index will
        be None until the axis for that position is created, then it 
        will be the handle to that axis.

    Examples
    --------
    Plot each of the vectors in the list `things_to_plot` on its own
    axis, using the default of two axis across::

        sp = Subplots(len(things_to_plot))
        for vec in things_to_plot:
            ax = sp.next_subplot()
            ax.plot(vec)
    """
    def __init__(self, nplots, nx=None, ny=None, figsize=(8,6)):
        """Create a Subplots instance

        Parameters
        ----------
        nplots : int
            The total number of plots required.

        nx, ny : int
            How many subplots to use horizontally and vertically, respectivelly. 
            One of `nx` and `ny` may be `None`, in which case it is computed for
            you. Both may be specified; if the given array size is less than
            `nplots`, a `ValueError` is raised. If both are `None`, then `nx` is
            set to 2 and `ny` is computed.

        figsize : tuple
            The size of *each subplot* as a tuple, `(width, height)`, in inches.
            The final size of the figure will be `nx*width` by `ny*height`.

        Notes
        -----
        Creating a Subplots instance does not automatically instantiate all the
        axes, each axis in created as needed. This way, if you need 10 plots but
        want 3 across, the last row will not have 2 empty axes, just whitespace.
        """
        sizex, sizey = figsize
        if nx is None and ny is None:
            nx = 2
        
        if ny is None:
            ny = self._calc_second_dim(nx, nplots)
        elif nx is None:
            nx = self._calc_second_dim(ny, nplots)
        elif nx * ny < nplots:
            raise ValueError('nx * ny is less then nplots; increase one or specify one as None to calculate it automatically')

        self.nx = nx
        self.ny = nplots // nx + ((nplots % nx) > 0)
        self.fig = plt.figure(figsize=(sizex*self.nx, sizey*self.ny))
        self.iplot = 0
        self.axes = np.full([self.ny, self.nx], None)
    
    def next_subplot(self):
        """Create the next subplot and return the handle to it

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
            Handle to the newly created axes.
        """
        self.iplot += 1
        ax = self.fig.add_subplot(self.ny, self.nx, self.iplot)
        idx = np.unravel_index(self.iplot-1, self.axes.shape)
        self.axes[idx] = ax
        return ax

    @staticmethod
    def _calc_second_dim(dim1, total):
        return total // dim1 + ((total % dim1) > 0)


def create_twin_axes_with_color(twin='x', color2=None, color1=None, ax=None):
    """
    Create twinned axes where the axes are colored

    :param twin: string "x" or "y" indicating which axis is common to the two axes
    :param color2:
    :param color1:
    :return:
    """
    if ax is None:
        fig, ax1 = plt.subplots()
    else:
        fig = ax.figure
        ax1 = ax

    if twin == 'x':
        ax2 = ax1.twinx()
        spines = [ax1.spines['left'], ax2.spines['right']]
        tick_ax = 'y'
        labels = [ax1.yaxis.label, ax2.yaxis.label]
    elif twin == 'y':
        ax2 = ax1.twiny()
        spines = [ax1.spines['bottom'], ax2.spines['top']]
        tick_ax = 'x'
        labels = [ax1.xaxis.label, ax2.xaxis.label]
    else:
        raise ValueError('twin must be "x" or "y"')

    if color1:
        spines[0].set_color(color1)
        ax1.tick_params(axis=tick_ax, colors=color1)
        labels[0].set_color(color1)
    if color2:
        spines[1].set_color(color2)
        ax2.tick_params(axis=tick_ax, colors=color2)
        labels[1].set_color(color2)

    return fig, ax1, ax2


def plot_xy_error_bars(ax, x, x_error, y, y_error, **style):
    """
    Convenience method to plot x and y error simultaneously.

    This is somewhat limited compared to :func:`plot_error_bar`, as the errors must be symmetric and given as
    differences from the coordinate values (i.e. error_type='diff').

    :param ax: the axis to plot into

    :param x: the x-coordinates of the data.
    :type x: Must be a numpy array, a subclass of numpy ndarray, or something that can be made into a numpy array by
     being passed as the first argument to np.array().

    :param x_error: the error in the x-coordinate.
    :type x_error: same as ``x``

    :param y: the y-coordinates of the data.
    :type y: same as ``x``

    :param y_error: the error in the x-coordinate.
    :type y_error: same as ``x``

    :param style: keyword arguments controlling the style of the error lines.

    :return: the handles to the x and y error bars, in that order.
    """
    lx = plot_error_bar(ax, x, y, x_error, direction='x', error_type='diff', **style)
    ly = plot_error_bar(ax, x, y, y_error, direction='y', error_type='diff', **style)
    return lx, ly


def plot_error_bar(ax, x, y, error, upper_error=None, direction='y', error_type='diff', **style):
    """
    :param ax: the axis to plot into

    :param x: the x-coordinates of the data
    :type x: Must be a numpy array, a subclass of numpy ndarray, or something that can be made into a numpy array by
     being passed as the first argument to np.array().

    :param y: the y-coordinates of the data
    :type y: same as ``x``

    :param error: the error of the data or, if upper_error is also given, the lower error
    :type error: same as ``x``

    :param upper_error: the upper error of the data, if asymmetric errors are desired.
    :type error: same as ``x``, or None.

    :param direction: 'x' or 'y', indicates which axis to plot the errors on
    :type direction: str

    :param error_type: 'diff' or 'abs'; for 'diff' (default) errors are taken as the length of the error bar, i.e. how
     large the error is. In 'abs', errors are taken as the value that the end of the error bar should lie at. This
     latter mode is useful when expressing error as quantiles. Note, 'abs' requires that upper_error be given
    :type error_type: str
    :return: handles to the line objects created
    """

    error_style = {'color': 'k', 'linewidth': 0.5}
    error_style.update(style)

    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(error, np.ndarray):
        error = np.array(error)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    if upper_error is None:
        if error_type == 'abs':
            raise IncompatibleOptions('diff_type = "abs" requires that upper_error be given.')
        else:
            upper_error = error
    elif not isinstance(upper_error, np.ndarray):
        upper_error = np.ndarray(upper_error)

    # To simplify things, we'll make it so that the variable with error is i and the one without j. That way no matter
    # which direction the error bars go in, we can operate on i,j the same, and just convert back at the end.
    if direction == 'y':
        i_coords = y
        j_coords = x
    elif direction == 'x':
        i_coords = x
        j_coords = y
    else:
        raise NotImplementedError('Organizing the input coords in the right x/y dims not configured for direction = {}'.format(direction))

    plot_error_i = np.full((i_coords.size, 3), np.nan, dtype=np.float)
    plot_error_j = np.stack( (j_coords, j_coords, np.full((j_coords.size,), np.nan, dtype=np.float)), axis=1 )

    # Make the coordinates for the error bars a 3 column array: first column is for the lower end, second for the upper
    # end, and third column will always be NaNs so that the lines are technically one line object but appear separate.
    if error_type == 'abs':
        plot_error_i[:, 0] = error
        plot_error_i[:, 1] = upper_error
    elif error_type == 'diff':
        plot_error_i[:, 0] = i_coords - error
        plot_error_i[:, 1] = i_coords + error

    if direction == 'y':
        x_error = plot_error_j
        y_error = plot_error_i
    elif direction == 'x':
        x_error = plot_error_i
        y_error = plot_error_j
    else:
        raise NotImplementedError('Placing the error points in the right x/y dims not configured for direction = {}'.format(direction))

    return ax.plot(x_error.flatten(), y_error.flatten(), **error_style)


def bars(ax, x, height, width=None, relwidth=0.8, color=None, **kwargs):
    """
    Plot groups of bars together on a bar plot

    The values of the bars are determined by ``height``; if it is 1D, then this function is effectively equivalent to
    `pyplot.bar`. If it is 2D, then the number of groups of bars will be ``height.shape[0]`` and the number of bars in
    each group will be ``height.shape[1]``. Unlike `pyplot.bar`, ``height`` cannot currently be a scalar and ``x`` must
    be a vector length equal to ``height.shape[0]``.

    :param ax: the axis to plot into. If set to ``None``, the `pyplot.gca()` will be called to obtain the axis.

    :param x: the x-coordinates for the center of each group of bars. Converted to a numpy array internally if necessary
     (i.e. other iterables convertible by `numpy.array` can be given). ``x`` must provide one value per group.

    :param height: the heights of each bar. Converted to a numpy array internally. Must be 1- or 2- D.

    :param width: the width of the bar groups (*not* individual bars). If omitted, will be calculated as the smallest
     separation between x-coordinates * ``relwidth``. Can be given as a scalar value that will be applied to all groups,
     or a vector (or vector-like iterable) that specifies a unique width for each group. If specified, ``relwidth`` is
     not used.

    :param relwidth: a factor to scale the default width by. Only used if ``width`` is not specified, i.e. will only
     scale the default width.
    :type relwidth: float

    :param color: any matplotlib color specification, or a vector-like object of such specifiers with length equal to
     ``height.shape[1]``. If not given, the bars in each group will cycle through the standard 10 matplotlib colors.
     If given as a vector, the first element defines the color for the first bar in each group, second element the
     color for the second bar, and so on.

     Note that a string or a 3-element tuple of numbers will be considered a single color spec, to be applied to all
     bars.

    :param kwargs: additional keyword arguments to be passed onto `pyplot.bar`. Note that these are passed for every
     bar; there is no way currently to pass different keywords to different bars in the group.

    :return: handles to all bars plotted
    """
    if ax is None:
        ax = plt.gca()
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(height, np.ndarray):
        height = np.array(height)

    if x.ndim > 1:
        raise ValueError('x must be a scalar or vector')
    if x.size != height.shape[0]:
        raise ValueError('The length of x must be equal to the shape of height along the first dimension')

    if height.ndim > 2:
        raise ValueError('height must be at most 2D')
    elif height.ndim == 1:
        height = height.reshape((-1, 1))
    elif height.ndim != 2:
        raise NotImplementedError('height must be 1- or 2- D.')

    nbars_per_group = height.shape[1] if height.ndim > 1 else 1

    # If no width given, then compute the maximum width that would not overlap adjacent groups.
    if width is None:
        width = np.array(relwidth * np.min(np.diff(x)))
    elif not isinstance(width, np.ndarray):
        width = np.array(width)

    if width.ndim == 0:
        width = width.repeat(x.size)
    elif width.ndim != 1 or width.size != height.shape[0]:
        raise ValueError('If given as a vector, width must be the same length as the first dimension of height')

    if isinstance(color, np.ndarray) and color.ndim > 1:
        raise NotImplementedError('colors cannot currently be multidimensional')
    elif color is None:
        # Use the default matplotlib colors, via the 'CN' color specs (https://matplotlib.org/api/colors_api.html)
        color = ['C{}'.format(i % 10) for i in range(nbars_per_group)]
    elif isinstance(color, str) or _is_color_tuple(color):
        # Most color specifications are strings, so if only one is given, repeat it for every bar in the group. The only
        # other option is a color tuple.
        color = [color for i in range(nbars_per_group)]
    elif len(color) != nbars_per_group:
        raise ValueError('colors must be a single colorspec or an iterable of colorspecs with length == '
                         'height.shape[1]')

    # Now for each of bar in the groups, calculate where it should sit around the group center
    fractional_offsets = np.linspace(-0.5, 0.5, 1 + 2*nbars_per_group)[1:-1:2]

    handles = []
    for i in range(nbars_per_group):
        single_bar_widths = width / nbars_per_group
        h = ax.bar(x + width * fractional_offsets[i], height[:, i], single_bar_widths, color=color[i], **kwargs)
        handles.append(h)

    return handles


class ColorMapper(mpl.cm.ScalarMappable):
    """
    Map different values to colors in a colormap.

    This is useful when you wish to plot multiple series whose color corresponds to a data value, but do not want to
    use a scatter plot. This class can be instantiated in two ways:

    1. Call the class directly, providing a min and max value and (optionally) a colormap to use, e.g.::

        cm = ColorMapper(0, 10, cmap='viridis')

    2. Use the ``from_data`` method to automatically set the min and max values to those of some sort of data array,
       e.g.::

        cm = ColorMapper.from_data(np.arange(10))

    Either method accepts all keywords for :class:`matplotlib.cm.ScalarMappable` except ``norm`` which is set
    automatically. Calling the instance will return an RGBA tuple that can be given to the ``color`` keyword of a
    matplotlib plotting function::

        pyplot.plot(np.arange(10), color=cm(5))

    The instance of this class would then be used as the mappable for a colorbar, e.g.::

        pyplot.colorbar(cm)

    Init parameters:

    :param vmin: the bottom value for the color map.
    :type vmin: int or float

    :param vmax: the top value for the color map
    :type vmax: int or float

    :param cmap: the color map to use. May be a string that gives the name of the colormap or a Colormap instance.
    :type cmap: str or :class:`matplotlib.colors.Colormap`

    :param **kwargs: additional keyword arguments passed through to :class:`matplotlib.cm.ScalarMappable`
    """
    def __init__(self, vmin, vmax, cmap=mpl.cm.jet, **kwargs):
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        super(ColorMapper, self).__init__(norm=norm, cmap=cmap, **kwargs)
        # This is a necessary step for some reason. Not sure why
        self.set_array([])

    def __call__(self, value):
        return self.to_rgba(value)

    @classmethod
    def from_data(cls, data, **kwargs):
        """
        Create a :class:`ColorMapper` instance from a data array, with min and max values set to those of the data.

        :param data: the data array. May be any type that ``numpy.min()`` and ``numpy.max()`` will correctly return a
         scalar value for.

        :param **kwargs: additional keyword args passed through to the class __init__ method.

        :return: a new class instance
        :rtype: :class:`ColorMapper`
        """
        return cls(vmin=np.nanmin(data), vmax=np.nanmax(data), **kwargs)


def pcolordiff(x_or_data, y=None, data=None, plotting_fxn=plt.pcolormesh, fraction_of_max=1.0, cmap='RdYlBu', **plot_kwargs):
    """
    Make a pseudo-color difference plot.

    This is basically a wrapper around :func:`~matplotlib.pyplot.pcolormesh` that automatically sets the color limits to
    even values around 0 to support diverging colormaps easily. Also defaults to a diverging colormap.  May be called
    as::

        pcolordiff(data)

    to use default indices for the x and y axes or::

        pcolordiff(x, y, data).

    In this case, ``data`` need to follow the pcolor convention of being ny-by-nx.

    :param x_or_data: either the data to plot as the colors (in the one-argument form) or the x-coordinates, in any
     form compatible with :func:`~matplotlib.pyplot.pcolormesh`.
    :type x_or_data: array-like

    :param y: the y-coordinates (if using the three-argument form) in any form compatible with
     :func:`~matplotlib.pyplot.pcolormesh`.
    :type y: array-like

    :param data: the data to plot (if using the three-argument form) as a ny-by-nx array.
    :type data: array-like

    :param plotting_fxn: the underlying plotting function to use. May be any plotting function that can be called using
     the one- or three- argument forms above, and also accepts the ``cmap``, ``vmin``, and ``vmax`` keywords.
    :type plotting_fxn: callable

    :param fraction_of_max: this scales the limits of the color map. By default the limits will be set to +/-
     ``nanmax(abs(data))``. Changing this input will scale those limits, i.e. the new limits will be
     ``nanmax(abs(data)) * fraction_of_max``.
    :type fraction_of_max: float

    :param cmap: the colormap to use for the plot.

    :param plot_kwargs: additional keywords for the plotting function

    :return: All return values from the plotting function.
    """
    maxd = np.nanmax(np.abs(data)) * fraction_of_max
    if data is None:
        return plotting_fxn(x_or_data, cmap=cmap, vmin=-maxd, vmax=+maxd, **plot_kwargs)
    elif y is None:
        raise TypeError('Must call as pcolordiff(data) or pcolordiff(x, y, data). pcolordiff(data=data) is not supported')
    else:
        return plotting_fxn(x_or_data, y, data, cmap=cmap, vmin=-maxd, vmax=+maxd, **plot_kwargs)


def pcolor_categorical(xcat, ycat, zval, text_counts=False, text_style=dict(), text_fmt='{}',
                       text_skip=lambda x: np.isnan(x), ax=None, **pcolor_kws):
    """Create a pcolor plot with categorical variables

    This creates a pcolor plot with ticks centered on the colored squares. It is intended for categorical variables
    on both axes, and so every tick will be labeled. It also ensures that all elements of `zvar` are displayed.

    Parameters
    ----------
    xcat : Sequence
        The labels to use for the x ticks. Must be a sequence, but can be any type that matplotlib can display as
        tick labels.

    ycat : Sequence
        The label to use for the y ticks. Same type restrictions as `xcat`.

    zval : Array-like
        Must be a 2D numeric array. Its dimensions must be `(len(ycat), len(xcat))`. This may feel backwards, but
        follows the ordering for `pcolor` and `pcolormesh`.

    text_counts : bool
        If `True`, then the numeric value will be printed in the center of each grid cell, which can help identify
        its exact value if that is important.

    text_style : dict
        A dictionary of style keywords that can be passed to `matplotlib.pyplot.text`. The defaults are:

            * `backgroundcolor = "w"`
            * `ha = "center"`
            * `va = "center"`

        Each default is used unless explicitly overridden. (To remove the background color, pass the string `"none"`.)
        Note that because `ha` and `va` are used, `horizontalalignment` and `verticalalignment` must not be.

    text_fmt : str
        A string that can be formatted using the `format()` method; this controls how the text values are printed in
        each cell.

    text_skip : callable
        A function that, given a value from `zval`, returns `True` if the text for that cell should *not* be printed,
        and `False` otherwise. The default behavior is to print any non-NaN value.

    ax : matplotlib.axes._subplots.AxesSubplot
        The axis to plot into. If `None`, a default set of axes is created.

    pcolor_kws
        Additional keywords given to this function are passed through to `pcolormesh`.

    Returns
    -------
    matplotlib.collections.QuadMesh
        The handle returned by `pcolormesh`, suitable to pass to `colorbar`.
    """
    if ax is None:
        _, ax = plt.subplots()
    if np.ndim(zval) != 2:
        raise TypeError('zval must have exactly 2 dimensions')

    ny, nx = np.shape(zval)
    if np.size(xcat) != nx:
        raise TypeError(
            'xcat must have a number of elements (was {}) equal to the second dimension of zval (was {})'.format(
                np.size(xcat), nx))
    if np.size(ycat) != ny:
        raise TypeError(
            'ycat must have a number of elements (was {}) equal to the first dimension of zval (was {})'.format(
                np.size(ycat), ny))

    h = ax.pcolormesh(zval, **pcolor_kws)
    xticks = np.arange(0.5, nx)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xcat)

    yticks = np.arange(0.5, ny)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ycat)

    if text_counts:
        defaults = {'backgroundcolor': 'w', 'ha': 'center', 'va': 'center'}
        for k, v in defaults.items():
            text_style.setdefault(k, v)

        for i, j in itertools.product(range(nx), range(ny)):
            if text_skip(zval[j, i]):
                continue
            x = xticks[i]
            y = yticks[j]
            ax.text(x, y, text_fmt.format(zval[j, i]), text_style)

    return h


def cbextend(data, vmin=None, vmax=None, vlim=None):

    vmin = -vlim if vmin is None and vlim is not None else vmin
    vmax = vlim if vmax is None and vlim is not None else vmax
    if vmin is not None:
        extend_min = np.nanmin(data) < vmin
    else:
        extend_min = False

    if vmax is not None:
        extend_max = np.nanmax(data) > vmax
    else:
        extend_max = False

    if extend_min and extend_max:
        extend = 'both'
    elif extend_min:
        extend = 'min'
    elif extend_max:
        extend = 'max'
    else:
        extend = 'neither'

    return extend


def _is_color_tuple(t):
    """
    Test if a tuple is a valid color tuple
    :param t: the tuple to test
    :rtype: bool
    """
    if not isinstance(t, tuple):
        return False
    elif len(t) != 3:
        return False
    elif not all([isinstance(v, (float, int)) and 0 <= v <= 1 for v in t]):
        return False
    else:
        return True


def heatmap(x, y, xbins=10, ybins=10, plotfxn=plt.pcolormesh, zero_white=True, log=False, **kwargs):
    """
    Plot a 2D heatmap indicating the frequency of values occurring in x & y bins

    :param x: the x data. See :func:`hist2d` for how this may be specified.
    :type x: :class:`numpy.ndarray`

    :param y: the y data. See :func:`hist2d` for how this may be specified.
    :type y: :class:`numpy.ndarray`

    :param xbins: x bin specification. See :func:`hist2d` for how this may be specified.
    :type xbins: int or :class:`numpy.ndarray`

    :param ybins: y bin specification. See :func:`hist2d` for how this may be specified.
    :type ybins: int or :class:`numpy.ndarray`

    :param plotfxn: the function to use to plot the heatmap. Must accept x and y coordinates as vectors and the number
     of counts as a ny-by-nx array (note that the dimensions are reversed!) as the first three inputs. It will first
     try plotting with the x and y coordinates as the bin edges (i.e nx+1 and ny+1 in length); if this raises a
     ``TypeError`` then it will reduce x and y to nx and ny by calculating bin centers. This allows either a
     pcolor/pcolormesh to be used without cutting off the last row and column of the counts or a countour plot.

    :param zero_white: if ``True`` (default), then zero values are set to NaNs in the count array before plotting. This
     will make them white in most pseudo-color plots.
    :type zero_white: bool

    :param log: set to ``True`` to plot the log base 10 of the counts, rather than the linear value.
    :type log: bool

    :param **kwargs: additional keywords to be passed to the plotting function.

    :return: all return values from the plotting function.
    """
    counts, xbins, ybins = hist2d(x=x, y=y, xbins=xbins, ybins=ybins)
    counts = counts.astype(np.float)
    if zero_white:
        counts[np.isclose(counts, 0)] = np.nan
    if log:
        counts[counts <= 0] = np.nan
        counts = np.log10(counts)

    # Keeping the bins with an extra element compared to counts
    try:
        return plotfxn(xbins, ybins, counts.T, **kwargs)
    except TypeError:
        xbins = 0.5*(xbins[:-1] + xbins[1:])
        ybins = 0.5*(ybins[:-1] + ybins[1:])
        return plotfxn(xbins, ybins, counts.T, **kwargs)


def add_1to1(ax=None, limits='equal', **style):
    """Add a 1:1 line to an axis

    Parameters
    ----------

    ax : :class:`matplotlib.pyplot.Axes`
        The axis to plot on. If `None`, the current axes are used.

    leave_limits : bool
        Controls how the x- and y- limits are handled. The default,
        "keep", fixes the current limits at the time this function
        is called. "equal" will make the x- and y- limits equal, with
        the minimum limit being the minimum for either axis and likewise
        for the maximum limit. "leave" will not set the limits.

    **style
        Keyword arguments for :func:`matplotlib.pyplot.plot`. Affects
        the style of the 1:1 line.

    Notes
    -----

    If the x- and y- limits are automatic, then plotting a 1:1 line will
    typically expand them to create some space around the line. The default
    behavior of `leave_limits` is to fix the current x- and y- limits so
    that the 1:1 line touches the edge of the plot. This does mean that the
    limits will no longer automatically expand if new data is plotted, so
    it's usually best to call this function only after all other data have
    been plotted.
    """
    if ax is None:
        ax = plt.gca()

    xl = ax.get_xlim()
    yl = ax.get_ylim()
    coords = [min(xl[0], yl[0]), max(xl[1], yl[1])]
    ax.plot(coords, coords, **style)
    if limits == 'keep':
        ax.set_xlim(xl)
        ax.set_ylim(yl)
    elif limits == 'equal':
        ax.set_xlim(coords)
        ax.set_ylim(coords)
    elif limits != 'leave':
        raise ValueError('{} is not valid for `limits`. Allowed values are "keep", "equal", and "leave"'.format(limits))


def x_log_ticks(ax, base=10, fmt='adapt'):
    """
    Set ticks on an x-axis assuming the current values are of a logarithm

    The ticks will be set to integer powers of the base.

    :param cb: the colorbar instance
    :param base: the logarithm base
    :param fmt: how to format the tick labels.

        * "adapt" (default) means that one of the other two will be chosen based on the magnitude of the tick labels.
        * "simple" means that that the labels will be straight numbers, i.e. 10, 100, 0.1
        * "exp" means that the labels will be written as 10^x

    :return: None
    """
    limits = ax.get_xlim()
    _log_ticks(limits, ax.set_xticks, ax.set_xticklabels, base=base, fmt=fmt)


def y_log_ticks(ax, base=10, fmt='adapt'):
    """
    Set ticks on a y-axis assuming the current values are of a logarithm

    The ticks will be set to integer powers of the base.

    :param cb: the colorbar instance
    :param base: the logarithm base
    :param fmt: how to format the tick labels.

        * "adapt" (default) means that one of the other two will be chosen based on the magnitude of the tick labels.
        * "simple" means that that the labels will be straight numbers, i.e. 10, 100, 0.1
        * "exp" means that the labels will be written as 10^x

    :return: None
    """
    limits = ax.get_ylim()
    _log_ticks(limits, ax.set_yticks, ax.set_yticklabels, base=base, fmt=fmt)


def cb_log_ticks(cb, base=10, fmt='adapt'):
    """
    Set ticks on a colorbar assuming the current values are of a logarithm

    The ticks will be set to integer powers of the base.

    :param cb: the colorbar instance
    :param base: the logarithm base
    :param fmt: how to format the tick labels.

        * "adapt" (default) means that one of the other two will be chosen based on the magnitude of the tick labels.
        * "simple" means that that the labels will be straight numbers, i.e. 10, 100, 0.1
        * "exp" means that the labels will be written as 10^x

    :return: None
    """
    limits = cb.get_clim()
    _log_ticks(limits, cb.set_ticks, cb.set_ticklabels, base=base, fmt=fmt)


def _log_ticks(limits, tick_fxn, tick_label_fxn, base, fmt):
    """

    :param limits:
    :param tick_fxn:
    :param tick_label_fxn:
    :param base:
    :param fmt:
    :return:
    """
    def format_log(exp):
        if exp >= 0:
            return '{}'.format(base ** int(exp))
        else:
            return '{:f}'.format(base ** exp)

    ll = np.ceil(limits[0])
    ul = np.floor(limits[1])
    ticks = np.arange(ll, ul + 1)
    if fmt == 'adapt':
        if ll < -4 or ul > 5:
            fmt = 'exp'
        else:
            fmt = 'simple'

    if fmt == 'simple':
        ticklabels = ['{}'.format(format_log(x)) for x in ticks]
    elif fmt == 'exp':
        ticklabels = ['${}^{{{:d}}}$'.format(base, int(x)) for x in ticks]
    else:
        raise ValueError('format "{}" not recognized'.format(fmt))
    tick_fxn(ticks)
    tick_label_fxn(ticklabels)


def label_subplots(axs, fmt='({})', seq='lower', xpos=-0.1, ypos=0.95, style={'weight': 'bold'}):
    """
    Label subplot axes with letter labels

    Useful for paper figures where subfigures need labels such as (a), (b), (c) etc.

    :param axs: the axes to label, as a sequence of some kind. May also be a single axis.
    :param fmt: a string specifying the format of the labels. Will be formatted with the new style format method, that
     is, a pair of brackets, {}, will be replaced with the label.
    :param seq: the sequence of characters to use for labeling each subfigure. May be the string "lower" for lowercase
     ASCII letters, "upper" for upper case ASCII letters, "number" for numbers starting at 1, or any iterable.
    :param xpos: the x-postiion of the text as a fraction of the x-axis. Negative places it to the left of the edge of
     the plot.
    :param ypos: the y-position of the text as a fraction of the y-axis.
    :param style: keyword arguments to the :func:`~matplotlib.pyplot.text` function controlling the format of the text.
     Note that the "ha" (horizontalalignment) keyword is already defined.
    :return: an array of handles to the text objects in the same shape as the axes array
    """
    if isinstance(axs, np.ndarray) and axs.ndim > 1:
        orig_shape = axs.shape
        axs = axs.flatten()
    elif isinstance(axs, plt.Axes):
        orig_shape = None
        axs = np.array([axs])
    elif not all(isinstance(ax, plt.Axes) for ax in axs):
        raise TypeError('axs must be either a matplotlib.pyplot.Axes instance or a sequence of such instances')
    else:
        orig_shape = np.shape(axs)
    n_ax = np.size(axs)

    if seq == 'lower':
        seq = string.ascii_lowercase
    elif seq == 'upper':
        seq = string.ascii_uppercase
    elif seq == 'number':
        seq = range(1, len(axs)+1)
    else:
        try:
            _ = [x for x in seq]
        except TypeError:
            raise TypeError('seq must be one of the strings "lower", "upper" or "number", or else an iterable')

        if len(seq) < n_ax:
            raise ValueError('Not enough labels: {} labels for {} axes'.format(len(seq), n_ax))

    # Calculate the position for the text to be by default near the top outside left of the axes
    handles = np.full(n_ax, None)
    for i, (ax, label) in enumerate(zip(axs, seq)):
        x1, x2 = ax.get_xlim()
        dx = x2 - x1
        x = x1 + xpos * dx
        y1, y2 = ax.get_ylim()
        dy = y2 - y1
        y = y1 + ypos * dy

        handles[i] = ax.text(x, y, fmt.format(label), ha='right', **style)

    if orig_shape is None:
        handles = handles.item()
    else:
        handles = handles.reshape(orig_shape)
    return handles
