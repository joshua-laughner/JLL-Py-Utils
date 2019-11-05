"""
Utilities that don't fit any other well defined category
"""
import numpy as np


class ProgressBar(object):
    def __init__(self, end_count, style='bar+count', add_one=True, freq=1, prefix='', suffix='', auto_finish=True, width=72):
        if len(prefix) > 0 and not prefix.endswith(' '):
            prefix += ' '
        if len(suffix) > 0 and not suffix.startswith(' '):
            suffix = ' ' + suffix

        self._end_count = end_count
        self._style = style
        self._add_one = 1 if add_one else 0
        self._freq = freq
        self._prefix = prefix
        self._suffix = suffix
        self._auto_finish = auto_finish
        self._finished = False
        self._width = width

        self._last_num_char = -1

    def print_bar(self, index):
        index += self._add_one
        if (index % self._freq == 0) or (index == self._end_count) or (index == self._add_one):
            if self._style == 'count':
                self._print_counter(index)
            elif self._style == 'bar':
                self._print_bar(index)
            elif self._style.startswith('bar+'):
                num_style = self._style.split('+')[1]
                self._print_bar(index, numeric_style=num_style)
            else:
                raise NotImplementedError('Style "{}" not implemented'.format(self._style))

        if index == self._end_count and self._auto_finish:
            self.finish()

    def _print_counter(self, index):
        counter = self._format_counter(index)
        fmt = '\r{pre}{counter}{suf}'
        msg = fmt.format(pre=self._prefix, suf=self._suffix, counter=counter)
        print(msg, end='')

    def _print_bar(self, index, numeric_style='none'):
        always_update = True
        if numeric_style is None or numeric_style == 'none':
            number_fxn = lambda index: ''
            always_update = False
        elif numeric_style == 'percent':
            number_fxn = self._format_percent
        elif numeric_style == 'count':
            number_fxn = self._format_counter
        else:
            raise ValueError('numeric_style = "{}" not recognized'.format(numeric_style))

        # save characters for the number, plus the open/close bracket and a space between the bar
        # and number if the number exists, plus the prefix/suffix
        n_reserved = len(number_fxn(0))
        spacer = ''
        if n_reserved > 0:
            n_reserved += 1
            spacer = ' '
        bar_max_len = self._width - n_reserved - 2 - len(self._prefix) - len(self._suffix)
        bar_fmt = '[{{:<{}}}]'.format(bar_max_len)
        # compute how many progress characters to print. Only print if we need to update
        nchar = int(float(index)/self._end_count * bar_max_len)
        if always_update or nchar != self._last_num_char:
            bar = bar_fmt.format('#'*nchar)
            number = number_fxn(index)
            line = '\r{pre}{bar}{spacer}{num}{suf}'.format(pre=self._prefix, bar=bar, spacer=spacer, num=number, suf=self._suffix)
            print(line, end='')
            
    def _format_counter(self, index):
        nchr = len(str(self._end_count))
        fmt = r'{{idx:{}}}/{{end}}'.format(nchr)
        return fmt.format(idx=index, end=self._end_count)

    def _format_percent(self, index):
        percent = float(index) / self._end_count * 100
        return '{:6.2f}%'.format(percent)

    def finish(self):
        if not self._finished:
            # print a newline so that the next message goes onto a new line
            print('')
            # record that we finished so that the user can manually call this after an auto-finish
            # without finishing twice
            self._finished = True
        


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


def find(a, axis=None, pos='all'):
    """
    Find indices where an array has a truth-like value.

    This function provides a different way of getting indices where ``a`` is truthy than :func:`numpy.argwhere` or
    :func:`numpy.nonzero`. Its functionality varies slightly whether an axis is specified.

     * If ``axis`` is ``None``, then ``find`` behaves like :func:`numpy.flatnonzero`, finding the linear indices of
       truthy values in ``a``.
     * If ``axis`` is given, then ``pos`` must be "first" or "last" and ``find`` will return the first or last index of
       a truthy value along that dimension as an (N-1) dimensional array (assuming ``a`` is N-D).

    :param a: the array to find truthy values in
    :type a: array-like

    :param axis: which axis to operate along. The behavior of ``find`` changes slightly if ``axis`` is not specified,
     see the main description for details.
    :type axis: int or None

    :param pos: the position of the indices to return. May be "all", "first", or "last". If ``axis`` is specified, then
     ``pos`` must be "first" or "last".
    :type pos: str

    :return: indices of truthy values in ``a``. If ``axis`` is not specified, these will be linear indices for the
     flattened version of ``a``. If ``axis`` is specified, these will be linear indexes along the requested axis.
    :rtype: int or array-like
    """
    if axis is None:
        xx = np.flatnonzero(a)
        if pos == 'first':
            return xx[0]
        elif pos == 'last':
            return xx[-1]
        elif pos == 'all':
            return xx
        else:
            raise ValueError('pos must be "all", "first", or "last"')

    else:
        shape_arr = np.ones_like(a.shape)
        shape_arr[axis] = -1
        indices = np.arange(a.shape[axis], dtype=np.float).reshape(shape_arr)
        indices = np.broadcast_to(indices, a.shape).copy()
        indices[~a] = np.nan

        if pos == 'last':
            return np.nanargmax(indices, axis=axis).astype(np.int)
        elif pos == 'first':
            return np.nanargmin(indices, axis=axis).astype(np.int)
        elif pos == 'all':
            raise NotImplementedError('No behavior implemented for pos = "all" along a specific axis')
        else:
            raise ValueError('pos must be "all", "first", or "last"')


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
