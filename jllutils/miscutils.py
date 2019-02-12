"""
Utilities that don't fit any other well defined category
"""
import numpy as np

def all_or_none(val):
    """
    Return ``True`` if all or no values in the input are truthy

    :param val: the iterable to check
    :type val: iterable of truth-like values

    :return: ``True`` if all values in ``val`` are true or all are false; ``False`` if there is a mixture.
    :rtype: bool
    """
    s = sum([bool(v) for v in val])
    return s == 0 or s == len(val)


def find_block(a, axis=0, ignore_masked=True, as_slice=False, block_value=None):
    """
    Find blocks of contiguous values along a given dimension of a.

    This function will identify the index ranges for which there are contiguous values in a and the values themselves.
    For example, for::

        a = np.array([0, 1, 1, 1, 2, 3, 3, 4, 5])

    then ``find_block(a)`` would return ``[(1, 4), (5, 7)]`` and ``[1, 3]``. Note that the index ranges follow the
    Python convention that the last index is exclusive, so that ``a[1:4]`` would give ``[1, 1, 1]``.

    If ``a`` is a 2D or higher array, then the whole slice ``a[i+1, :]`` must equal ``a[i, :]`` to be considered
    contiguous. For::

        a = [[1, 2, 3],
             [1, 2, 3],
             [1, 2, 4]]

    the first two rows only would be considered a contiguous block because all their values are identical, while the
    third row is not. Finally, if ``a`` is a masked array, then by default the masked *values* are ignored but the masks
    themselves must be the same. With the same 2D ``a`` as above, if all the values in the last column were masked:

        a = [[1, 2, --],
             [1, 2, --],
             [1, 2, --]]

    then all three rows would be considered a block, but if only the bottom right element was masked:

        a = [[1, 2,  3],
             [1, 2,  3],
             [1, 2, --]]

    then the third row would *not* be part of the block, because even though its unmasked values are identical to the
    corresponding values in the previous row, the mask is not.

    :param a: the array or array-like object to find contiguous values in. Can be given as anything that `numpy.array`
     can turn into a proper array.
    :type a: array-like

    :param axis: which axis of ``a`` to operate along.
    :type axis: int

    :param ignore_masked: setting this to ``False`` modifies the behavior described above regarding masked arrays. When
     this is ``False``, then the underlying values are considered for equality. In the example where the entire last
     column of ``a`` was masked, setting ``ignore_masked = True`` would cause the third row to be excluded from the
     block again, because the values under the mask are compared.
    :type ignore_masked: bool

    :param as_slice: set to ``True`` to return the indices as slice instances instead; these can be used directly in
     ``a`` retrieve the contiguous blocks.
    :type as_slice: bool

    :param block_value: if given, this will only look for blocks matching this value. In the first example where ``a``
     was a vector, setting this to ``1`` would only return the indices (1, 4) since only the first block has values of
     1. If ``a`` if multidimensional, keep in mind that the explicit check is that ``np.all(a[i,:] == block_value)``,
     so this may be a scalar or an array, as long as each slice passes that test. If this is a masked array, then the
     mask of each slice of ``a`` must also match its mask.

    .. note::

       If ``block_value`` is a masked array and ``a`` is not, then most likely nothing will match. However, you should
       not rely on this behavior, since there may be some corner cases where this is not true, particularly if
       ``block_value`` and ``a`` are boolean arrays.

    :type block_value: array-like

    :return: A list of block start/end indices as two-element tuples (or slices, if ``as_slice = True``) and a list of
     the values of each block.
    :rtype: list(tuple(int)) and list(numpy.ndarray)

    .. note::

       Even if you pass ``a`` in as something other than a numpy array (e.g. a list of lists) the values returned will
       be numpy arrays. This is due to how ``a`` is handled internally.
    """
    if not isinstance(a, np.ndarray):
        a = np.array(a)

    block_mask = None
    if block_value is not None:
        if not isinstance(block_value, np.ndarray):
            block_value = np.array(block_value)
        elif isinstance(block_value, np.ma.masked_array): # and not isinstance(row.mask, np.bool_): # TODO: figure out why I'd needed the extra isinstance check here and for the rows
            block_mask = block_value.mask
            if ignore_masked:
                # If ignoring masked values, we cut down the value the block is supposed to equal just like we cut
                # down each row
                block_value = block_value[~block_mask]

    a = np.moveaxis(a, axis, 0)

    last_row = None
    last_mask = None
    blocks = []
    values = []
    block_start = None

    for idx, row in enumerate(a):
        if isinstance(row, np.ma.masked_array): # and not isinstance(row.mask, np.bool_):
            mask = row.mask
            if ignore_masked:
                # Ignoring masked values: masked values do not contribute one way or another to determining if two rows
                # are equal, *however*, rows must have the *same* mask, so we store the mask but remove masked values.
                # If the whole row was masked, then the last row will have to have been the same. That's why we extract
                # the mask before cutting row down.
                row = row[~mask]
            # If we're not ignoring masked values, then don't cut down row: np.array_equal will test the underlying data
            # even under masked values

        else:
            # Not a masked array? Just make the mask equal to the row so they are both equal or not together - this
            # removes the need for logic to determine whether or not to test the masks
            mask = row

        if last_row is None:
            # On the first row. Can't do anything yet.
            pass
        elif block_start is not None and not (np.array_equal(row, last_row) and np.array_equal(mask, last_mask)):
            # End of block. Add to the list of blocks. The current index is not in the block, but since Python's
            # convention is that the ending index is exclusive, we make the end of the block the first index not
            # included in it.
            blocks.append((block_start, idx))
            block_start = None
        elif block_start is None and np.array_equal(row, last_row) and np.array_equal(mask, last_mask):
            # If the current row equals the last row, then we're in a block.
            if block_value is None:
                # If no value to check against the block was given, then set the start of the block to the last index
                # (where the block actually started).
                block_start = idx-1
                values.append(row)
            elif np.all(row.shape == block_value.shape) and np.all(row == block_value):
                # If a value to check against the block was given, then only start the block if the current row
                # matches that value. If that value was a masked array, then the masks must be equal as well.
                if block_mask is None or np.all(block_mask == mask):  # if the row is the same shape, then the mask will be too
                    block_start = idx-1
                    values.append(row)

        last_row = row
        last_mask = mask

    if block_start is not None:
        # If block_start isn't None after we're done with the for loop, then there's a block that goes to the end of
        # the array, so add that.
        blocks.append((block_start, a.shape[0]))

    if as_slice:
        blocks = [slice(*b) for b in blocks]

    return blocks, values
