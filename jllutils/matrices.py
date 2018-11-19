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


def masked_full(*args, **kwargs):
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
    xx = np.logical_not(marray.mask)
    iter_array = marray[xx]
    for element in iter_array:
        yield element
