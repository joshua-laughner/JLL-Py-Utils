"""
Helper methods for working with numpy arrays and creating test arrays.
"""


import numpy as np
from numpy import ma


def seq_mat(shape, dtype=np.int, order='C'):
    """
    Return an array with elements numbered as they are ordered in memory.

    :param shape: the shape to give the array
    :type shape: iterable

    :param dtype: the datatype to give the array. Must be compatible with numeric values. Default is :class:`numpy.int`.
    :type dtype: object

    :param order: the memory order to use. Default is 'C'.
    :type order: str

    :return: the sequential array
    :rtype: :class:`numpy.ndarray`
    """
    if not np.issubdtype(dtype, np.number):
        raise TypeError('dtype must be a numeric type')

    mat = np.arange(np.prod(shape), dtype=dtype)
    return np.reshape(mat, shape, order=order)


def coord_mat(shape, dtype=np.int, order='C'):
    """
    Return an array where the values correspond to the subscript indicies.

    For a 2D array, A, A[i,j] = i*10 + j. For a 3D array, A[i,j,k] = 100*i + 10*j + k, and so on.

    :param shape: the shape to give the array
    :type shape: iterable

    :param dtype: the datatype to give the array. Must be compatible with numeric values. Default is :class:`numpy.int`.
    :type dtype: object

    :param order: the memory order to use. Default is 'C'.
    :type order: str

    :return: the coordinate array
    :rtype: :class:`numpy.ndarray`
    """
    mat = np.zeros(shape, dtype=dtype, order=order)
    for i in range(np.prod(shape)):
        idx = np.unravel_index(i, shape, order=order)
        val = np.sum([x * 10**p for p, x in enumerate(reversed(idx))])
        mat[idx] = val

    return mat


def masked_full(*args, **kwargs):
    """
    Shortcut to create a masked array with all elements initialized to a given value.

    All inputs are the same as :func:`numpy.full`.

    :return: the filled array with a mask initialized to ``False``.
    :rtype: :class:`numpy.ma.masked_array`.
    """
    return ma.masked_array(np.full(*args, **kwargs))


def masked_all(shape, fill_value=np.nan, dtype=np.float, order='C'):
    """
    Create a masked array where all elements start masked.

    :param shape: The shape to give the array
    :type shape: iterable

    :param fill_value: the underlying value to give the array elements. Default is NaN.

    :param dtype: the datatype to give the array, must be compatible with the fill value. Default is a numpy float.
    :type dtype: object

    :param order: the memory order of the array. Default is 'C'
    :type order: str

    :return: the new masked array
    :rtype: :class:`numpy.ma.masked_array`
    """
    m = np.full(shape, fill_value=fill_value, dtype=dtype, order=order)
    m = ma.masked_array(m)
    m.mask = np.ones_like(m, dtype=np.bool)
    return m


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


def mask_from_criteria(*criteria):
    """
    Create a mask from one or more criteria.

    Gets around an issue where a bitwise OR cannot operate on a masked array. If any of the criteria are masked arrays,
    they are converted to numpy arrays with the masked elements set to ``True``. That is, the final mask will mask any
    elements where the corresponding element in any of the criteria was either ``True`` or already masked.

    :param criteria: one or more boolean arrays that are ``True`` where the final mask should be ``True`` as well. All
     must have the same type.
    :type criteria: :class:`numpy.ndarray` or :class:`numpy.ma.masked_array` with datatype :class:`numpy.bool_` or a
     subtype or it.

    :return: an array that is ``True`` where any of the criteria indicate an element should be masked.
    :rtype: :class:`numpy.ndarray` (dtype = :class:`numpy.bool`).
    """
    if len(criteria) == 0:
        raise ValueError('Must give at least one criterion')
    
    expected_shape = criteria[0].shape
    for idx, crit in enumerate(criteria[1:]):
        if crit.shape != expected_shape:
            raise ValueError('Criteria no. {} has a different shape than the first one'.format(idx+2))

    # If any of these are masked arrays, convert to numpy arrays first because a bit-wise OR is not
    # allowed to work with masked arrays. We'll assume that if an element is masked in the criterion,
    # it should be masked in the final mask.
    mask = np.zeros(expected_shape, dtype=np.bool)
    for crit in criteria:
        if isinstance(crit, ma.masked_array):
            crit = crit.filled(True)
        elif not isinstance(crit, np.ndarray):
            raise TypeError('Each criterion must be a numpy.ndarray or numpy.ma.masked_array')
        
        if not np.issubdtype(crit.dtype, np.bool_):
            raise TypeError('The dtype of each criterion array must be a sub-dtype of numpy.bool')

        mask |= crit

    return mask


def iter_unmasked(marray):
    """
    Iterate over unmasked elements in an array.

    :param marray: the masked array to iterate over.
    :type marray: :class:`numpy.ma.masked_array`

    :return: iterable over the unmasked elements.
    """
    xx = np.logical_not(marray.mask)
    iter_array = marray[xx]
    for element in iter_array:
        yield element
