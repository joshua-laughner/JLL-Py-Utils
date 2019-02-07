"""
Helpful generic plotting functions.
"""


from matplotlib import pyplot as plt
import numpy as np


class IncompatibleOptions(Exception):
    """
    Error to use when incompatible options are passed
    """
    pass


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
        x = np.ndarray(x)
    if not isinstance(error, np.ndarray):
        error = np.ndarray
    if not isinstance(y, np.ndarray):
        y = np.ndarray(y)
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
