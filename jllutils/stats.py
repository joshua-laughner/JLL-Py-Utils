import numpy as np
from numpy import ma

def _rolling_input_helper(A, window, edges, force_centered):
    if A.ndim != 1:
        raise NotImplementedError('Rolling operations not yet implemented for arrays with dimension != 1')
    elif not isinstance(window, int) or window < 1:
        raise ValueError('window must be a positive integer (> 0)')

    if force_centered and window % 2 == 0:
        window += 1

    # First we pad the input array ourselves to have control over what values are used to extend the array
    extend_width = int(window/2)
    if edges == 'zeros':
        A = np.concatenate([np.zeros((extend_width,), dtype=A.dtype), A, np.zeros((extend_width,), dtype=A.dtype)])
    elif edges == 'extend':
        A = np.concatenate([np.repeat(A[0], extend_width), A, np.repeat(A[-1], extend_width)])
    else:
        raise ValueError('edges must be "zeros" or "extend"')

    return A, window


def rolling_mean2(A, window, edges='zeros', force_centered=False):
    """
    Compute a rolling mean over an array.

    :param A: The array to operate on. Must be 1D currently
    :type A: :class:`numpy.ndarray`

    :param window: the size of the averaging window to use.
    :type window: int

    :param edges: optional, how to treat points at edges:

        * 'zeros' (default) prepends and appends 0s to A in order to give enough points for the averaging windows near
          the edge of the array. So for ``A = [1, 2, 3, 4, 5]``, with a window of 3, the first window would average
          ``[0, 1, 2]`` and the last one ``[4, 5, 0]``.
        * 'extend' repeats the first and last values instead of using zeros, so in the above example, the first window
          would be ``[1, 1, 2]`` and the last one ``[4, 5, 5]``.
    :type edges: str

    :param force_centered: if ``False``, does nothing. If ``True``, forces the window to be an odd number by adding one
     if it is even. This matters because with an even number the averaging effectively acts "between" points, so for an
     array of ``[1, 2, 3]``, a window of 2 would make the windows ``[0, 1]``, ``[1, 2]``, ``[2, 3]``, and ``[3, 0]``.
     As a result, the returned array has one more element than the input. Usually it is more useful to have windows
     centered on values in A, so this paramter makes that more convenient.
    :type force_centered: bool

    :return: an array, ``B``, either the same shape as A or one longer.
    :rtype: :class:`numpy.ndarray`.
    """
    A, window = _rolling_input_helper(A, window, edges, force_centered)

    # The fact that we can do this with a convolution was described https://stackoverflow.com/a/22621523
    # The idea is, for arrays, convolution is basically a dot product of one array with another as one slides
    # over the other. So using a window function that is 1/N, where N is the window width, this is exactly an average.
    conv_fxn = np.ones((window,), dtype=A.dtype) / window

    # Using mode='valid' will restrict the running mean to t
    return np.convolve(A, conv_fxn, mode='valid')


def rolling_nanmean(A, window, edges='zeros', force_centered=False):
    if isinstance(A, ma.masked_array):
        A = A.filled(np.nan)

    A = rolling_op(np.nanmean, A, window, edges=edges, force_centered=force_centered)
    return ma.masked_where(np.isnan(A), A)


def rolling_nanstd(A, window, edges='zeros', force_centered=False):
    if isinstance(A, ma.masked_array):
        A = A.filled(np.nan)

    A = rolling_op(np.nanstd, A, window, edges=edges, force_centered=force_centered)
    return ma.masked_where(np.isnan(A), A)


def rolling_op(op, A, window, edges='zeros', force_centered=False):
    A, window = _rolling_input_helper(A, window, edges, force_centered)
    return op(rolling_window(A, window), axis=-1)


def rolling_window(a, window):
    """
    Create a view into an array that provides rolling windows in the last dimension.

    For example, given an array, ``A = [0, 1, 2, 3, 4]``, then for ``Awin = rolling_window(A, 3)``,
    ``Awin[0,:] = [0, 1, 3]``, ``Awin[1,:] = [1, 2, 3]``, etc. This works for higher dimensional arrays
    as well.

    :param a: the array to create windows on.
    :type a: :class:`numpy.ndarray`

    :param window: the width of windows to use
    :type window: int

    :return: a read-only view into the strided array of windows.

    From http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html.

    Note: the numpy docs (https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.stride_tricks.as_strided.html#numpy.lib.stride_tricks.as_strided)
    advise caution using the as_strided method used here because effectively what it does is change how numpy iterates
    through at array at the memory level.
    """

    if not isinstance(window, int) or window < 1:
        raise ValueError('window must be a positive integer (>0)')

    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides, writeable=False)