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