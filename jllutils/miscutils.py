"""
Utilities that don't fit any other well defined category
"""
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
import fnmatch
import logging
from typing import Optional, Union

import numpy as np
import os
from pathlib import Path
import re
import string
import warnings


class ProgressBar(object):
    """Display a text-based progress bar

    For loops over large numbers of items, this allows you to display the progress without taking up n lines. It will
    continually reprint the progress bar on the current line. Note that it has no mechanism to detect if you are
    printing other messages in the for loop, so if you do that, the messages will get jumbled up with the progress bar.
    """
    def __init__(self, end_count, style='bar+count', add_one=True, freq=1, prefix='', suffix='', auto_finish=True,
                 force_finish=True, width=72, enabled=True):
        """Create a ProgressBar instance

        Parameters
        ----------
        end_count : int
            the number of times the loop will run, i.e. this is the iteration when the task is considered complete.

        style : str
            how to display the progress bar. Options are:
                * "count" will display a simple counter, e.g. " 1/25" that updates
                * "percent" will display the percentage complete, e.g. "  4.00%"
                * "bar" will display a progress bar, e.g. "[#####     ]"
                * "bar+?" where "?" is one of the other style will display the graphical bar plus the numeric style
                  at the end, e.g. "bar+count" would give "[####    ] 10/20"

        add_one : bool
            if `True`, then when given an index to update the progress bar, 1 is added to it when calculating the
            progress. This way, passing the 0-based index of the item will give the correct progress if the total number
            of iterations was given as the `end_count`. If your index will already have 1 added or if the `end_count`
            given was the last index, then set `add_one` to `False`.

        freq : int
            how often to update the progress bar. The bar will only be updated every `freq` calls to print the bar. This
            is useful to limit printing to the screen if you have a very fast loop.

        prefix : str
            a string to write before the rest of the progress bar. There will always be at least one space between the
            prefix and the rest of the bar.

        suffix : str
            a string to write after the rest of the progress bar. There will always be at least one space between the
            rest of the bar and the suffix.

        auto_finish : bool
            when `True`, `finish()` is called automatically when the index reaches the end count.

        force_finish : bool
            when `True`, the bar will be set to the end count when `finish()` is called.

        width : int
            number of characters wide to make the progress bar, including the prefix and suffix.

        enabled : bool
            whether this progress bar responds to calls to `print_bar` or `finish`. Can be used to easily disable
            the bar during less verbose execution.
        """
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
        self._force_finish = force_finish
        self._finished = not enabled
        self._width = width

        self._curr_index = 0
        self._last_num_char = -1
        self._freq_count = 0

    @classmethod
    def iteron(cls, iterable, **class_kws):
        """Print a progress bar while iterating over an object

        Since a progress bar is used almost exclusively inside a for loop, this method combines
        iterating over an object and printing a progress bar for that iteration.

        Parameters
        ----------

        iterable
            Any iterable object. The progress bar will have its end count set to the iterable's
            length. See notes.

        **class_kws
            Keywords for the init method of this class. Note that `add_one` will be ignored because
            it always needs to be `True` given how the iteration worked internally.


        Returns
        -------

        iterator
            Iterates over the elements of `iterable`. For each element, the progress bar is automatically
            updated.


        Examples
        --------

        Given an iterable, `x`, the progress bar would be printed automatically::

            xvec = np.arange(10)
            for x in ProgressBar.iteron(x):
                ...

        Notes
        -----

        `len()` is used to figure out the length of the iterator to set the end count of the progress bar.
        If `len()` cannot be used on your iterable (e.g. if it is a generator object or an array object that
        `len` gives the wrong length for) then you should *either* (a) pass the correct end count as the
        `end_count` keyword or (b) instantiate an instance of the class yourself and call the `print_bar`
        method within the for loop to print the bar.
        """
        # Modifiy some of the keywords if needed
        if 'end_count' not in class_kws:
            class_kws['end_count'] = len(iterable)
        if 'add_one' in class_kws:
            warnings.warn('The add_one keyword is ignored by iteron')
        class_kws['add_one'] = True

        pbar = cls(**class_kws)
        for i, x in enumerate(iterable):
            pbar.print_bar(i)
            yield x

    def print_bar(self, index=None):
        """Update the progress bar on screen

        Parameters
        ----------
        index : int
            The index representing how far through the progress the program is. If not given, then an internal index
            that advances by one each time this method is called is used instead.
        """
        self._curr_index += 1
        if index is None:
            # We're implicitly adding one already so the setting of _add_one doesn't matter
            index = self._curr_index
        else:
            index += self._add_one

        if (self._freq_count == 0) or (index == self._end_count) or (index == self._add_one):
            if self._style == 'count':
                self._print_counter(index, self._format_counter)
            elif self._style == 'percent':
                self._print_counter(index, self._format_percent)
            elif self._style == 'bar':
                self._print_bar(index)
            elif self._style.startswith('bar+'):
                num_style = self._style.split('+')[1]
                self._print_bar(index, numeric_style=num_style)
            else:
                raise NotImplementedError('Style "{}" not implemented'.format(self._style))

        self._freq_count = (self._freq_count - 1) % self._freq

        if index == self._end_count and self._auto_finish:
            self.finish()

    def _print_counter(self, index, format_fxn):
        counter = format_fxn(index)
        fmt = '\r{pre}{counter}{suf}'
        msg = fmt.format(pre=self._prefix, suf=self._suffix, counter=counter)
        print(msg, end='')

    def _print_bar(self, index, numeric_style='none'):
        always_update = True
        if numeric_style is None or numeric_style == 'none':
            def number_fxn(index): return ''
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
        """Finish the progress bar

        This will print a newline (so that any following text is printed on a separate line) and, if `force_finish` was
        set, update the bar to the end position. This only has an effect the first time it is called on a given
        instance.
        """
        if not self._finished:
            # record that we finished so that the user can manually call this after an auto-finish
            # without finishing twice. Do this first to avoid a recursion error if we force finishing
            self._finished = True

            # Move the counter to the end
            if self._force_finish:
                self.print_bar(self._end_count - self._add_one)

            # print a newline so that the next message goes onto a new line
            print('')


class ProgressMessage(object):
    """Print a series of messages with each one overwriting the previous one.

    This is a companion to :class:`ProgressBar`. This class is build for cases where a large but indeterminate
    number of progress messages need to be shown. Unlike :class:`ProgressBar`, this does not need or accept the
    total number of iterations expected.
    """
    def __init__(self, prefix='', suffix='', add_one=True, format='{prefix}{idx}{suffix}', width=72, truncate=True,
                 auto_advance=True, enabled=True):
        """Create a ProgressMessage instance.

        Parameters
        ----------
        format : str
            A format string (using `{}` style formatting) that will be used for each message. All formatting must be
            done by key value (that is, each pair of curly braces must have a key name: "{prefix}", not "{}"). 
            Three keywords are always available: "prefix", "suffix", and "idx", which will be replaced with the prefix
            string, suffix string, and current index, respectively. Other keywords can be used, so long as they are
            given to each call of :meth:`print_message`.

        prefix : str
            A prefix string to use. It will replace "{prefix}" in `format`. If it does not end with a space, one is
            added.

        suffix : str
            A suffix string to use. It will replace "{suffix}" in `format`. If it does not begin with a space, one is
            added.

        add_one : bool
            Whether one should be added to the index value before inserting in the message, effectively switching to
            1-based indexing.

        width : int
            Number of characters wide each message may be.

        truncate : bool
            When `True`, messages longer than `width` are truncated to `width`. If this is `False`, messages can 
            exceed the width, which also means characters from previous messages may be left behind.

        auto_advance : bool
            When `True`, the internal index is advanced by 1 after each call to :meth:`print_message`.

        enabled : bool
            whether this progress bar responds to calls to `print_bar` or `finish`. Can be used to easily disable
            the bar during less verbose execution.
        """
        self._prefix = prefix
        if not self._prefix.endswith(' '):
            self._prefix += ' '
        self._suffix = suffix
        if not self._suffix.startswith(' '):
            self._suffix = ' ' + self._suffix

        self._add_one = int(add_one)
        self._width = width
        self._format = format
        self._final_format = '\r{{:<{}}}'.format(width)
        self._curr_index = 0
        self._truncate = truncate
        self._auto_adv = auto_advance

        self._finished = not enabled

    def format_message(self, index, **kwargs):
        tmp = self._format.format(prefix=self._prefix, idx=index + self._add_one, suffix=self._suffix, **kwargs)
        if self._truncate and len(tmp) > self._width:
            tmp = tmp[:self._width]
        return self._final_format.format(tmp)

    def print_message(self, index=None, **kwargs):
        """Print a message

        Parameters
        ----------
        index : Optional[int]
            Index to use for the "{idx}" value in the message. Will have 1 added to it if `add_one` was `True` during
            instantiation. If not given, the current internal count of how many messages have been printed is used.

        kwargs
            Additional keywords, passed to the message formatter and so need to provide for all formatting elements
            in the message except for "{prefix}", "{suffix}", and "{idx}".
        """
        if self._finished:
            return
        if index is None:
            index = self._curr_index
        else:
            self._curr_index = index
        msg = self.format_message(index, **kwargs)
        print(msg, end='')
        if self._auto_adv:
            self._curr_index += 1

    def finish(self):
        """Mark this progress as complete.

        Has two effects: future calls to :meth:`print_message` do nothing and prints a newline to "close out" the
        message. Note that unlike :class:`ProgressBar`, which can infer when it is complete, :class:`ProgressMessage`
        cannot (as it is intended for use when the total number of elements is not known) so you must call `finish`
        to avoid the next `print` happening on the same line as the last progress message.
        """
        if not self._finished:
            self._finished = True
            print('')


class CleanupFiles(object):
    """Context manager that cleans up certain files when the context block exits

    Usage::

        with CleanupFiles() as cleaner:
            ...
            cleaner.add_file(tmp_file)
            ...
    """
    def __init__(self, verbose=True):
        """Create a CleanupFiles instance

        Parameters
        ----------
        verbose : bool
            Set to `True` to print a message for each file removed
        """
        self._files = []
        self._verbose = verbose

    def add_file(self, filename):
        """Add a file to be deleted at the end of a context block

        Parameters
        ----------
        filename : str
            Path to the file to delete
        """
        self._files.append(filename)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for f in self._files:
            if self._verbose:
                print('Cleaning up {}'.format(f))
            os.remove(f)


class _File(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        self._path = Path(*args, **kwargs)
        if not self._path.exists():
            raise OSError(f'{self._path} does not exist')

    def __repr__(self):
        klass = self.__class__.__name__
        filepath = str(self._path)
        return f'{klass}("{filepath}")'

    def __str__(self):
        return str(self._path)

    def __getattr__(self, item):
        return getattr(self._path, item)

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __hash__(self):
        pass


class RealFile(_File):
    """Representation of a specific, concrete file stored on disk.

    This class is almost identical to :class:`pathlib.Path` except that it checks equality by file, not path. This makes
    it easy to tell whether two paths point to the same file or not. This class follows symbolic links, so comparing
    `f1 == f2` will return `True` if one is a symbolic link to the other, or both are symbolic links to the same "real"
    file. If you want to distinguish symbolic links from the files they point to, use :class:`LinkFile` instead.

    The constructor for this class is the same as for :class:`pathlib.Path`, except that if you try to construct an
    instance for a file that does not exist, an error is raised immediately.

    Notes
    -----
    * You can check for equality between this class and a :class:`pathlib.Path` or a string, and it will have the same
      behavior as constructing a :class:`RealFile` instance from those types.
    * This class has not been tested on Windows as of 2021-04-15

    See Also
    --------
    * :class:`LinkFile` - considers symbolic links and "real" files different when checking for equality
    """
    def __eq__(self, other):
        if isinstance(other, Path):
            their_stat = other.stat()
        elif isinstance(other, str):
            their_stat = Path(other).stat()
        elif isinstance(other, self.__class__):
            their_stat = other._path.stat()
        else:
            raise NotImplementedError(f'Equality between {self.__class__.__name__} and {other.__class__.__name__} not implemented')

        my_stat = self._path.stat()
        return (my_stat.st_ino == their_stat.st_ino) and (my_stat.st_dev == their_stat.st_dev)

    def __hash__(self):
        my_stat = self._path.stat()
        return my_stat.st_ino + 2**64 * my_stat.st_dev


class LinkFile(_File):
    """Representation of a specific, concrete file stored on disk.

    This class is almost identical to :class:`pathlib.Path` except that it checks equality by file, not path. This makes
    it easy to tell whether two paths point to the same file or not. This class does not follow symbolic links, so
    comparing `f1 == f2` will return `False` if one is a symbolic link to the other, or both are symbolic links to the
    same "real" file. If you want to treat symbolic links as the same as the files they point to, use :class:`RealFile`
    instead.

    The constructor for this class is the same as for :class:`pathlib.Path`, except that if you try to construct an
    instance for a file that does not exist, an error is raised immediately.

    Notes
    -----
    * You can check for equality between this class and a :class:`pathlib.Path` or a string, and it will have the same
      behavior as constructing a :class:`LinkFile` instance from those types.
    * This class has not been tested on Windows as of 2021-04-15

    See Also
    --------
    * :class:`RealFile` - considers symbolic links the same as the file they point to when checking for equality
    """
    def __eq__(self, other):
        if isinstance(other, Path):
            their_stat = other.lstat()
        elif isinstance(other, str):
            their_stat = Path(other).lstat()
        elif isinstance(other, self.__class__):
            their_stat = other._path.lstat()
        else:
            raise NotImplementedError(f'Equality between {self.__class__.__name__} and {other.__class__.__name__} not implemented')

        my_stat = self._path.lstat()
        return (my_stat.st_ino == their_stat.st_ino) and (my_stat.st_dev == their_stat.st_dev)

    def __hash__(self):
        my_stat = self._path.lstat()
        return my_stat.st_ino + 2**64 * my_stat.st_dev


def find_files(path: str, name: Optional[str] = None, iname: Optional[str] = None, xtype: Optional[str] = None):
    """Find files and directories in a tree, similar to the unix ``find`` program.

    Note that a small subset of the filters from ``find`` are available.

    Parameters
    ----------
    path
        Path under which to search.

    name
        Glob-style pattern to match against file base names (case sensitive).

    iname
        Glob-style pattern to match against file base names (case insensitive).

    xtype
        Which type a path is: "d" for directory, "f" for regular file (not link),
        "l" for symbolic link. Note that ``type`` is not an option because that is
        a Python keyword.

    Returns
    -------
    files
        A list of files matching all of the criteria. If no criteria were given, all
        files and directories under ``path`` are returned.
    """
    matched_files = []
    for root, dirs, files in os.walk(path):
        to_check = [(d, True) for d in dirs]
        to_check.extend((f, False) for f in files)
        for this_path, is_dir in to_check:
            if name and not fnmatch.fnmatchcase(this_path, name):
                continue
            if iname and not fnmatch.fnmatch(this_path, iname):
                continue
            if xtype:
                is_link = os.path.islink(os.path.join(root, this_path))
                if xtype == 'd' and not is_dir:
                    continue
                elif xtype == 'f' and is_dir and is_link:
                    continue
                elif xtype == 'l' and not is_dir and not is_link:
                    continue

            matched_files.append(os.path.join(root, this_path))
    return matched_files


def all_or_none(val):
    """Return ``True`` if all or no values in the input are truthy

    Parameters
    ----------
    val : sequence
        the iterable to check

    Returns
    -------
    bool
        `True` if all values in `val` are true or all are false; `False` if there is a mixture.
    """
    s = sum([bool(v) for v in val])
    return s == 0 or s == len(val)


def find_where(a, axis=None, pos='all'):
    """Find indices where an array has a truth-like value.

    This function provides a different way of getting indices where ``a`` is truthy than :func:`numpy.argwhere` or
    :func:`numpy.nonzero`. Its functionality varies slightly whether an axis is specified.

     * If ``axis`` is ``None``, then ``find`` behaves like :func:`numpy.flatnonzero`, finding the linear indices of
       truthy values in ``a``.
     * If ``axis`` is given, then ``pos`` must be "first" or "last" and ``find`` will return the first or last index of
       a truthy value along that dimension as an (N-1) dimensional array (assuming ``a`` is N-D).

    Parameters
    ----------
    a : numpy.ndarray
        the array to find truthy values in

    axis : int or None
        which axis to operate along. The behavior of ``find`` changes slightly if ``axis`` is not specified,
        see the main description for details.

    pos : str
        the position of the indices to return. May be "all", "first", or "last". If ``axis`` is specified, then
        ``pos`` must be "first" or "last".

    Returns
    -------
    int or array-like
        indices of truthy values in ``a``. If ``axis`` is not specified, these will be linear indices for the
        flattened version of ``a``. If ``axis`` is specified, these will be linear indexes along the requested axis.
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
        indices = np.arange(a.shape[axis], dtype=float).reshape(shape_arr)
        indices = np.broadcast_to(indices, a.shape).copy()
        indices[~a] = np.nan

        if pos == 'last':
            return np.nanargmax(indices, axis=axis).astype(int)
        elif pos == 'first':
            return np.nanargmin(indices, axis=axis).astype(int)
        elif pos == 'all':
            raise NotImplementedError('No behavior implemented for pos = "all" along a specific axis')
        else:
            raise ValueError('pos must be "all", "first", or "last"')


def find_block(a, axis=0, ignore_masked=True, as_slice=False, block_value=None):
    """Find blocks of contiguous values along a given dimension of a.

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

    Parameters
    ----------
    a : array-like
        the array or array-like object to find contiguous values in. Can be given as anything that `numpy.array`
        can turn into a proper array.

    axis : int
        which axis of ``a`` to operate along.

    ignore_masked : bool
        setting this to ``False`` modifies the behavior described above regarding masked arrays. When
        this is ``False``, then the underlying values are considered for equality. In the example where the entire last
        column of ``a`` was masked, setting ``ignore_masked = True`` would cause the third row to be excluded from the
        block again, because the values under the mask are compared.

    as_slice : bool
        set to ``True`` to return the indices as slice instances instead; these can be used directly in
        ``a`` retrieve the contiguous blocks.

    block_value : array-like
        if given, this will only look for blocks matching this value. In the first example where ``a``
        was a vector, setting this to ``1`` would only return the indices (1, 4) since only the first block has values of
        1. If ``a`` if multidimensional, keep in mind that the explicit check is that ``np.all(a[i,:] == block_value)``,
        so this may be a scalar or an array, as long as each slice passes that test. If this is a masked array, then the
        mask of each slice of ``a`` must also match its mask.

    Returns
    -------
    list[tuple[int]]
        A list of block start/end indices as two-element tuples (or slices, if ``as_slice = True``)

    list[numpy.ndarray]
        a list of the values of each block.

    Notes
    -----
    Even if you pass ``a`` in as something other than a numpy array (e.g. a list of lists) the second return value will
    be numpy arrays. This is due to how ``a`` is handled internally.

    If ``block_value`` is a masked array and ``a`` is not, then most likely nothing will match. However, you should
    not rely on this behavior, since there may be some corner cases where this is not true, particularly if
   ``block_value`` and ``a`` are boolean arrays.
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


def file_iter(files, make_abs=False):
    """Iterate over file paths, yielding both the path and basename together

    Parameters
    ----------
    files : Sequence[str]
        a sequence of paths to files
    make_abs : bool
        whether to make the path to the file an absolute path or not

    Yields
    ------
    str
        path to the file, possibly forced to be an absolute path
    str
        the basename of the file
    """
    for f in files:
        if make_abs:
            f = os.path.abspath(f)
        fbase = os.path.basename(f)
        yield f, fbase


def split_outside_group(s, grp_start, grp_stop=None, splitchar=string.whitespace, merge=None):
    """Split a string on a character or set of characters that occur outside specified delimiters

    This function deals with the problem of, e.g. splitting a string into parts where individual parts may contain the
    split character, but should not split because it is inside a "group". For example, given::

        "John Doe"  50  6.0

    we may want to split on spaces to parse this line of the table, but "John Doe" should not be split because it is
    grouped by quotation marks. This function handles that.

    Parameters
    ----------
    s : str
        The string to split

    grp_start : str
        The character that opens a group. It will be considered escaped (and not counted) if preceded by a backslash.

    grp_stop : str
        The character that closes a group. If not specified, is taken to be the same as `open`. Also escaped by
        backslashes.

    splitchar : str
        A string specifying a character or characters to split on. The default is to split on whitespace.

    merge : bool
        How to treat consecutive instances of `splitchar`. If this is `True`, then consecutive splitting characters
        are treated as one. If `False`, then each one is treated separately, meaning that 0-length strings will end
        up in the output list. By default, this will be `True` if `splitchar` is any combination of whitespace 
        characters and `False` otherwise.

    Returns
    -------
    list
        The list of substrings after being split
    """
    out = []
    grp_cnt = 0
    start = 0
    grp_stop = grp_start if grp_stop is None else grp_stop
    same_char = grp_start == grp_stop
    if merge is None:
        merge = all([c in string.whitespace for c in splitchar])

    if len(grp_start) != 1:
        raise ValueError('open must be a single character')
    elif len(grp_stop) != 1:
        raise ValueError('close must be a single character')

    def count_change(c):
        if same_char:
            # Cannot have nested groups with the same character opening and closing a group. So if we're in a group,
            # we leave it, and vice versa.
            return 1 if grp_cnt == 0 else -1
        elif c == grp_start:
            return 1
        elif c == grp_stop:
            return -1

    def is_escaped(idx):
        if idx == 0:
            return False
        elif s[idx-1] == '\\':
            return True
        else:
            return False

    for i, c in enumerate(s):
        if i < start:
            # i will be < start if start was advanced to merge consecutive split characters
            continue
        elif c in splitchar and grp_cnt == 0:
            out.append(s[start:i])
            start = i+1
            while merge and s[start] in splitchar:
                # advance start to the next non-split character
                start += 1
        elif c in (grp_start, grp_stop) and not is_escaped(i):
            grp_cnt += count_change(c)

        if grp_cnt < 0:
            raise ValueError('Given string has an unmatched {} at position {}'.format(grp_stop, i))

    # Group count should be 0. If not, then there were more open characters than closing characters
    if grp_cnt > 0:
        raise ValueError('Given string had an unmatched {}'.format(grp_start))

    # Handle the last piece. Check that we haven't just added something, i.e. that the split character isn't the last
    # one in the string. If it is, then we should add an empty element
    if start == len(s) + 1:
        out.append('')
    else:
        out.append(s[start:])

    return out


def parse_fortran_format(fmt_str, rtype='fmtstr', bare_fmt=False):
    # Fortran uses "i" for integers, Python uses "d"
    # Fortran uses "a" for strings, Python uses "s"
    # "pe" comes up as a way to change the power of 10 that an
    #   exponential is, Python has no equivalent.
    fmttype_subs = {'pe': 'e', 'i': 'd', 'a': 's'}

    fmt_str = fmt_str.strip()
    if not fmt_str.startswith('(') or not fmt_str.endswith(')'):
        raise ValueError('The first and last non-whitespace characters of the format string must be open and close '
                         'parentheses, respectively.')
    fmt_parts = split_outside_group(fmt_str[1:-1], '(', ')', ',')
    colspecs = []
    types = []
    format_str = ''

    idx = 0
    for part in fmt_parts:
        if '(' in part:
            match = re.match(r'(\d*)(\(.+\))', part)
            repeats = 1 if len(match.group(1)) == 0 else int(match.group(1))
            subfmt = match.group(2)
            subfmt, subspecs, subtypes = parse_fortran_format(subfmt, rtype='all')

            for i in range(repeats):
                for start, stop in subspecs:
                    colspecs.append((idx+start, idx+stop))
                types += subtypes
                format_str += subfmt
                idx += stop
        elif re.match(r'\d+x', part):
            # In Fortran, specifiers like "1x" mean that n spaces do not contain information and are just spacers
            nspaces = int(re.match(r'\d+', part, re.IGNORECASE).group())
            format_str += nspaces * ' '
            idx += nspaces
        else:
            match = re.match(r'(\d*)([a-z]+)(\d+)(\.\d+)?', part, re.IGNORECASE)
            # This is not always right. For specs like "1pe12.4", the "1" is not repeats,
            # but is associated with the "p".
            repeats, fmttype, width, prec = match.groups()
            # Substitute in the python-compatible format type
            fmttype = fmttype_subs.get(fmttype, fmttype)
            prec = '' if prec is None else prec

            pyfmt = '{w}{p}{t}'.format(w=width, p=prec, t=fmttype)
            if not bare_fmt:
                pyfmt = '{:' + pyfmt + '}'
            repeats = 1 if len(repeats) == 0 else int(repeats)
            width = int(width)

            for i in range(repeats):
                colspecs.append((idx, idx+width))
                types.append(pyfmt)
                format_str += pyfmt
                idx += width

    if rtype == 'all':
        return format_str, colspecs, types
    elif rtype == 'fmtstr':
        return format_str
    elif rtype == 'colspecs':
        return colspecs
    elif rtype == 'types':
        return types
    else:
        raise NotImplementedError('rtype == {}'.format(rtype))


@contextmanager
def temporary_working_dir(working_dir):
    """Temporarily change the working directory

    Useful in cases where you need to call functions that expect to be in
    a directory with a particular structure. Use as a context manager
    to ensure that the working directory is changed back to the
    original one when the block exits, for example::

        # cwd is '.'
        with temporary_working_dir('./run-dir'):
            # cwd is './run-dir'
            dir_dependent_function(...)
        # cwd is back to '.', even if dir_dependent_function errors

    Paramters
    --------
    working_dir
        Path to the directory to change to; relative paths are interpreted
        relative to the current working directory.
    """
    curr_dir = Path(os.getcwd()).expanduser().resolve()
    os.chdir(working_dir)
    yield
    os.chdir(curr_dir)


def logging_level(level_name: str) -> int:
    """Get the numeric logging level that corresponds to a level name

    Parameters
    ----------
    level_name
        The name to search for. Will be upper-cased for the search, so both "warning"
        and "Warning" will be searched for as "WARNING"

    Returns
    -------
    level
        The integer value of the log level.

    Raises
    ------
    ValueError
        If the given name does not correspond to any level number between 0 and ``logging.CRITICAL``.
    """
    level_name = level_name.upper()
    for level in range(0, logging.CRITICAL+1):
        this_name = logging.getLevelName(level)
        if level_name == this_name:
            return level

    raise ValueError(f'No level corresponding to the string {level_name}')


@contextmanager
def logging_context(level: Union[int, str] = logging.CRITICAL, logger: Optional[logging.Logger] = None):
    """Temporarily suspend log messages below a certain level

    Used as a context manager, any logging within the context block will be limited to messages at or
    above ``level`` in severity. If ``logger`` is not given, then this will apply globally. If ``logger``
    is given, then only that logger will be affected.

    Notes
    -----
    * The behavior differs slightly from :func:`logging.disable`; in that function, messages with severity
      ``level`` will be suppressed. Here, that level is kept.
    * If you pass a string for ``level``, it will be upper-cased before passing to the various log functions
      on the assumption that all log levels are upper case.
    * Likewise, if passing a string while ``logger = None``, it must be a name corresponding to a value between
      0 and ``logging.CRITICAL`` knowns by :func:`logging.getLevelName`.
    * This uses :func:`logging.disable` to affect the global logging (i.e. when ``logger = None``) and 
      ``logger.setLevel(level)`` when ``logger`` is not ``None``.  If these methods are used separately
      inside this context manager, then they can override the value set by this manager, *and* their
      value will be overridden when the block exits.

    Examples
    --------

    Set all logging to only WARNING messages and higher::

        with logging_pause('WARNING'):
            logging.warning('Look out!')  # will print
            logging.info('Pothole ahead.'')  # will not print

    Set only a custom logger to hide messages except CRITICAL ones::

        cust_logger = logging.getLogger('my-custom-logger')
        with logging_pause(logger=cust_logger):
            cust_logger.warning('Look out!')  # will not print
            logging.warning('Look out ahead!')  # will print

    Calling ``logging.disable`` inside a ``logging_pause`` block::

        with logging_pause(level='WARNING'):
            logging.info('Hi')  # will not print
            logging.warning('Eek!')  # will print

            logging.disable(logging.WARNING)
            logging.warning('Uh oh!')  # will *not* print because of how logging.disable works

        logging.warning('Did we make it?')  # will print because when the with block exits, logging.disable is called with logging.NOTSET
    """
    if logger is None:
        if isinstance(level, str):
            level = logging_level(level)
        logging.disable(level-1)
        yield
        logging.disable(logging.NOTSET)
    else:
        if isinstance(level, str):
            level = level.upper()
        curr_level = logger.level
        logger.setLevel(level)
        yield
        logger.setLevel(curr_level)



def compare_unmasked_arrays(a, b, return_stats: bool = False):
    """Compare two non-masked numpy arrays

    This function will check that ``a`` and ``b`` are the same with :func:`np.allclose`
    if both ``a`` and ``b`` are floating point types, and :func:`np.array_equal` otherwise.
    By default, it returns a boolean that will be ``True`` if they pass the given test.
    Setting the ``return_stats`` keyword to ``True`` will instead return a dictionary
    that includes this boolean as the key "equal", along with various statistics about the
    differences between a and b.
    """
    a_is_masked = hasattr(a, 'mask')
    b_is_masked = hasattr(b, 'mask')
    if a_is_masked or b_is_masked:
        raise TypeError(f'This function compares regular (not masked) arrays only (a is masked: {a_is_masked}, b is masked: {b_is_masked})')

    if np.issubdtype(a.dtype, np.floating) and np.issubdtype(b.dtype, np.floating):
        ok = np.allclose(a, b, equal_nan=True)
    else:
        ok = np.array_equal(a, b)

    if not return_stats:
        return ok

    shapes_match = a.shape == b.shape
    stats = {'equal': ok, 'shapes_match': shapes_match}
    if shapes_match:
        delta = b - a 
        stats['mindiff'] = np.nanmin(delta)
        stats['maxdiff'] = np.nanmax(delta)
        stats['meandiff'] = np.nanmean(delta)
        stats['mediandiff'] = np.nanmedian(delta)
    else:
        stats['mindiff'] = np.nan
        stats['maxdiff'] = np.nan
        stats['meandiff'] = np.nan
        stats['mediandiff'] = np.nan

    return stats


def compare_masked_arrays(a, b, return_stats: bool = False):
    """Compare two masked numpy arrays

    This function will check that ``a`` and ``b`` are the same with :func:`np.ma.allclose`
    if both ``a`` and ``b`` are floating point types. Otherwise, it will check the values with
    :func:`np.array_equal` and verify that the masks are equivalent (meaning a mask of
    ``np.False_`` and one that is an array of all ``False`` values are considered the same).
    By default, it returns a boolean that will be ``True`` if they pass the given test.
    Setting the ``return_stats`` keyword to ``True`` will instead return a dictionary
    that includes this boolean as the key "equal", along with various statistics about the
    differences between a and b.
    """
    a_not_masked = not hasattr(a, 'mask')
    b_not_masked = not hasattr(b, 'mask')
    if a_not_masked or b_not_masked:
        raise TypeError(f'This function compares masked arrays only (a is masked: {not a_not_masked}, b is masked: {not b_not_masked})')

    if np.issubdtype(a.dtype, np.floating) and np.issubdtype(b.dtype, np.floating):
        ok = np.ma.allclose(a, b, masked_equal=True)
        mask_ok = None
    else:
        val_ok = np.array_equal(a, b)
        mask_ok = _masks_equal(a, b)
        ok = val_ok and mask_ok


    if not return_stats:
        return ok

    if mask_ok is None:
        # Means we skipped calculating this before, do so now for the return dict.
        mask_ok = _masks_equal(a, b)

    shapes_match = a.shape == b.shape
    stats = {'equal': ok, 'masks_equal': mask_ok, 'shapes_match': shapes_match}
    if shapes_match:
        delta = b - a 
        stats['mindiff'] = np.ma.min(delta)
        stats['maxdiff'] = np.ma.max(delta)
        stats['meandiff'] = np.ma.mean(delta)
        stats['mediandiff'] = np.ma.median(delta)
    else:
        stats['mindiff'] = np.nan
        stats['maxdiff'] = np.nan
        stats['meandiff'] = np.nan
        stats['mediandiff'] = np.nan

    return stats


def _masks_equal(a, b):
    def mask_is_all_false(m):
        if np.ndim(m) == 0:
            return not m
        else:
            return not np.any(m)

    # First, just test if two masks are the same. If so, we can just return.
    if np.array_equal(a.mask, b.mask):
        return True

    # Failing that first doesn't doesn't mean the masks are different; we might
    # have a np.False_ mask for one and a proper array that is all False for the
    # other, so check if both evaluate to an all-False mask
    if not np.any(a.mask) and not np.any(b.mask):
        return True

    # If the arrays aren't elementwise equal and don't evaluate to all false,
    # then they really aren't equal
    return False
