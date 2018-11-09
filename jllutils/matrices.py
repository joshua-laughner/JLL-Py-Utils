import numpy as np
from numpy import ma


def seq_mat(shape, dtype=np.int, order='C'):
    if not np.issubdtype(dtype, np.number):
        raise TypeError('dtype must be a numeric type')

    mat = np.arange(np.prod(shape), dtype=dtype)
    return np.reshape(mat, shape, order='C')


def coord_mat(shape, dtype=np.int, order='C'):

    mat = np.zeros(shape, dtype=dtype, order=order)
    for i in range(np.prod(shape)):
        idx = np.unravel_index(i, shape, order=order)
        val = np.sum([x * 10**p for p, x in enumerate(idx)])
        mat[idx] = val

    return mat


def pad_masked(m, *args, **kwargs):
    """
    Pad a masked array

    :param m: the array to pad. Cannot contain unmasked NaNs, since NaNs are used as placeholders for the mask.
    :type m: :class:`numpy.ma.masked_array`

    Remaining arguments are the same as :func:`numpy.pad`. But internally it replaces masked values with NaNs so several
    of the pad methods (maximum, minimum, mean, etc.) will always end up padding with NaNs.

    :return: the padded masked array.
    :rtype: :class:`numpy.ma.masked_array`
    """
    if ma.any(np.isnan(m)):
        raise ValueError('m contains unmasked NaNs, it will not work with pad_masked')
    elif not np.issubdtype(m.dtype, np.float):
        raise NotImplementedError('Not set up to handle non-float arrays')

    # I want to use NaN because I'm afraid of how fill values will interact with some of the methods of extending
    # the array
    a = m.filled(np.nan)
    a = np.pad(a, *args, **kwargs)
    mprime = ma.masked_where(np.isnan(a), a)
    mprime.fill_value = m.fill_value

    return mprime