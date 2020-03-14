"""
Helpful methods for working with files.

At the top level of this module are methods for working with standard Python I/O, that is, text files, binary files,
and pickles. This module also contains two submodules, :mod:`ncio` and :mod:`h5io`, that have functions for working
with netCDF and HDF5 files, respectively. This can either be referenced directly from :mod:`fileops`::

    from jllutils import fileops
    fileops.ncio

or imported separately::

    from jllutils.fileops import ncio
"""


import contextlib
from io import IOBase as _IOBase
import pickle
import sys
from typing import Iterable, Union

# Import to make accessible through this module
from .subutils import ncdf as ncio, hdf5 as h5io


class SmartHandle(object):
    """An improved handle to standard files

    This solves the problem of opening either a file on disk or the special stdin/stdout file objects. If the filename
    is "-" or `None`, then stdin or stdout is opened (depending on the mode - stdin for read modes, stdout for any other
    mode). Otherwise the file specified is opened as normal.

    Parameters
    ----------
    filename: str or None
        The path to the file to open. If "-" or `None`, then stdin/stdout is "opened"
    mode: str
        The permission mode to open the file under, may be any mode recognized by the :func:`open` builtin.

    Examples
    --------

    You can use this pretty much anywhere in place of :func:`open`. The following would write "This is a log message"
    to the :file:`logfile.txt` file:

    >>> f = SmartHandle('logfile.txt', 'w')
    >>> f.write('This is a log message\\n')
    >>> f.close()

    But we can also use this to write that to stdout:

    >>> f = SmartHandle('-', 'w')  # stdout is chosen because the mode is write
    >>> f.write('This is a log message\\n')
    >>> f.close()  # closing will have no effect and can be included or omitted safely

    This works in `with` blocks as well
    >>> with SmartHandle('-', 'w') as f:
    >>>      f.write('This is a log message\\n')

    This is helpful if, say, you want to print messages to the screen normally but have the ability to redirect them
    to a file. A single `with` block can write to either a file or stdout depending on the value of the filename, which
    helps prevent having to duplicate your code.
    """
    def __init__(self, filename, mode):
        if filename is None or filename == '-':
            if mode.startswith('r'):
                self._handle = sys.stdin
            else:
                self._handle = sys.stdout
            self._do_close = False
        else:
            self._handle = open(filename, mode)
            self._do_close = True

    def __getattr__(self, item):
        # this allows pass through of all read/write methods to the _handle
        return getattr(self._handle, item)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the open file.

        This has no effect if it was stdin or stdout opened.
        """
        if self._do_close:
            self._handle.close()


def smart_handle(filename, mode, return_close=False):
    """
    Return a handle to a file or stdin/stdout.

    :param filename: the path to the file to open. If ``None`` or ``'-'``, then stdin (mode ``'r'`` or ``'rb'``) or
     stdout will be the handle returned.
    :type filename: str or None

    :param mode: permission mode to open the file under (e.g. ``'r'``, ``'w'``, ``'a'``, ...)
    :type mode: str

    :return: handle to the file, stdin, or stdout.
    """
    if filename is None or filename == '-':
        if mode.startswith('r'):
            handle = sys.stdin
        else:
            handle = sys.stdout
        do_close = False
    else:
        handle = open(filename, mode)
        do_close = True

    if return_close:
        return handle, do_close
    else:
        return handle


@contextlib.contextmanager
def smart_open(filename, mode):
    """An improved open method for standard files

    This solves the problem of opening either a file on disk or the special stdin/stdout file objects. If the filename
    is "-" or `None`, then stdin or stdout is opened (depending on the mode - stdin for read modes, stdout for any other
    mode). Otherwise the file specified is opened as normal.

    Parameters
    ----------
    filename: str or None
        The path to the file to open. If "-" or `None`, then stdin/stdout is "opened"
    mode: str
        The permission mode to open the file under, may be any mode recognized by the :func:`open` builtin.

    Returns
    -------
    SmartHandle

    Notes
    -----

    This is just a convenience function wrapped around :class:`SmartHandle` to provide a similar interface as the
    :func:`open` builtin.

    Examples
    --------

    You can use this pretty much anywhere in place of :func:`open`. The following would write "This is a log message"
    to the :file:`logfile.txt` file:

    >>> f = smart_open('logfile.txt', 'w')
    >>> f.write('This is a log message\\n')
    >>> f.close()

    But we can also use this to write that to stdout:

    >>> f = smart_open('-', 'w')  # stdout is chosen because the mode is write
    >>> f.write('This is a log message\\n')
    >>> f.close()  # closing will have no effect and can be included or omitted safely

    This works in `with` blocks as well

    >>> with smart_open('-', 'w') as f:
    >>>      f.write('This is a log message\\n')

    This is helpful if, say, you want to print messages to the screen normally but have the ability to redirect them
    to a file. A single `with` block can write to either a file or stdout depending on the value of the filename, which
    helps prevent having to duplicate your code.
    """
    return SmartHandle(filename, mode)


class MultiFileHandle(_IOBase):
    """Class for reading or writing to or from multiple files at once

    Parameters
    ----------
    *files
        Each file to read from or write to, as separate arguments. Uses :class:`SmartHandle` internally, so stdin/stdout
        may be specified with "-" or `None`.
    mode
        Permission mode for the files. May be "r", "rb", "w", "wb", "a", "ab". Must be specified as a keyword argument.
    read_rtype : str
        How to return values read from files. By default, they are returned as a dict, with the filenames being the
        keys. Other options are "list" and "tuple", which return the values as lists/tuples respectively in the order
        the file names were specified.
    """
    def __init__(self, *files, mode='r', read_rtype='dict'):
        self._file_names = files
        self._handles = tuple([SmartHandle(f, mode=mode) for f in files])
        self._read_rtype = read_rtype.lower()

        allowed_rtypes = ('dict', 'list', 'tuple')
        if self._read_rtype not in allowed_rtypes:
            raise ValueError('read_rtype must be one of: {}'.format(', '.join(allowed_rtypes)))

    def close(self):
        """Close all attached handles.
        """
        for f in self._handles:
            f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _format_read(self, values):
        """Format the return values from read, or other returning functions.
        """
        if self._read_rtype == 'dict':
            return values
        elif self._read_rtype == 'list':
            return [values[name] for name in self._file_names]
        elif self._read_rtype == 'tuple':
            return tuple([values[name] for name in self._file_names])
        else:
            raise NotImplementedError('No method for return a read type of "{}" implemented'.format(self._read_rtype))

    def _return_internal(self, fxn_name, args, kwargs):
        """Call any returning method on all the attached file handles.

        Parameters
        ----------
        fxn_name
            the name of the method to call
        args
            positional arguments for the method
        kwargs
            keyword arguments for the method

        Returns
        -------
            dict, list or tuple : formatted return value for the call.
        """
        values = dict()
        for name, handle in zip(self._file_names, self._handles):
            fxn = getattr(handle, fxn_name)
            values[name] = fxn(*args, **kwargs)
        return self._format_read(values)

    def read(self, *args, **kwargs):
        """Read at most n characters from all open streams.

        Accepts same arguments as the `read` methods on all open streams.

        Returns
        -------
        dict, list, or tuple :
            the characters read from all the files. If `read_rtype` was "dict" when this class was instantiated, then
            the return value will be a dictionary with the stream names as keys and the read strings as values.
            Otherwise, it will be a list or tuple with the read strings in the order the files were specified when this
            instance was created.
        """
        return self._return_internal('read', args, kwargs)

    def readline(self, *args, **kwargs):
        """Read one line from all open streams.

        Accepts same arguments as the `readline` methods on all open streams.

        Returns
        -------
        dict, list, or tuple :
            the characters read from all the files. If `read_rtype` was "dict" when this class was instantiated, then
            the return value will be a dictionary with the stream names as keys and the read strings as values.
            Otherwise, it will be a list or tuple with the read strings in the order the files were specified when this
            instance was created.
        """
        return self._return_internal('readline', args, kwargs)

    def readlines(self, *args, **kwargs):
        """Read lines as a list from all open streams.

        Accepts same arguments as the `readlines` methods on all open streams.

        Returns
        -------
        dict, list, or tuple :
            the characters read from all the files. If `read_rtype` was "dict" when this class was instantiated, then
            the return value will be a dictionary with the stream names as keys and the read strings as values.
            Otherwise, it will be a list or tuple with the read strings in the order the files were specified when this
            instance was created.
        """
        return self._return_internal('readlines', args, kwargs)

    def write(self, *args, **kwargs):
        """
        Write data to all open streams.

        Accepts same arguments as `write` method on all open streams.

        Parameters
        ----------
        text
            Text or binary data to write to all open streams.
        """
        for f in self._handles:
            f.write(*args, **kwargs)

    def writelines(self, lines: Iterable[Union[bytes, bytearray]]) -> None:
        """Write lists of data to all open streams.

        Accepts same arguments as `writelines` method on all open streams.

        Parameters
        ----------
        lines
            Iterable of text or binary data to write to all open streams.
        """
        for f in self._handles:
            f.writelines(lines)

    def seek(self, *args, **kwargs):
        """Change position of all streams.

        Accepts same arguments as `seek` method on all open streams.

        See Also
        --------
        io.BufferedReader.seek : builtin for reading in binary mode
        io.BufferedWriter.seek : builtin for writing in binary mode
        io.TextIOWrapper.seek : builtin for opening text files
        """
        return self._return_internal('seek', args, kwargs)


def dump_pickle(filename, obj):
    """Save an object to a pickle file

    Handles opening a file to save a pickle to, saving the data to the pickle, and closing the file in a single
    convenient function call.

    Parameters
    ----------
    filename : str
        the file name to give the pickle output file. Will be overwritten if it exists
    obj
        the Python object to pickle

    Examples
    --------

    >>> data = {'a': [1,2,3], 'b': "flora"}
    >>> dump_pickle("demo.pkl", data)
    """
    with open(filename, 'wb') as pkfile:
        pickle.dump(obj, pkfile, protocol=pickle.HIGHEST_PROTOCOL)


def grab_pickle(filename):
    """Load a pickle file

    Handles opening a file to save a pickle to, saving the data to the pickle, and closing the file in a single
    convenient function call.

    Parameters
    ----------
    filename : str
        the pickle file to load

    Returns
    -------
    object
        the Python object stored in the pickle file

    Examples
    --------

    Assuming the example from :func:`dump_pickle` was already run:

    >>> grab_pickle("demo.pkl")
    {'a': [1,2,3], 'b': "flora"}
    """
    with open(filename, 'rb') as pkfile:
        return pickle.load(pkfile)