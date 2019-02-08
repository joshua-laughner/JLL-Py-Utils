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

    This function will identify the index ranges for which there are contiguous values in a. For example, if a was::

        a = np.array([0, 1, 1, 1, 2, 3, 3, 4, 5])

    then ``find_block(a)`` would return
    :param a:
    :param axis:
    :param ignore_masked:
    :param as_slice:
    :return:
    """
    if not isinstance(a, np.ndarray):
        a = np.array(a)

    block_mask = None
    if block_value is not None:
        if not isinstance(block_value, np.ndarray):
            block_value = np.array(block_value)
        elif isinstance(block_value, np.ma.masked_array) and not isinstance(row.mask, np.bool_):
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
        if isinstance(row, np.ma.masked_array) and not isinstance(row.mask, np.bool_):
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
