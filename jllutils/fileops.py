"""
Helpful methods for working with files.
"""


import contextlib
from io import IOBase
import pickle
import sys
from typing import Iterable, Union

# Import to make accessible through this module
from .subutils import ncdf as ncio, hdf5 as h5io


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
    """
    Context manager that can return a handle to a file or stdin/stdout.

    This can be used with the ``with`` keyword to arbitrarily read from/write to a file or stdin/stdout and safely
    close the file when done with it.

    :param filename: the path to the file to open. If ``None`` or ``'-'``, then stdin (mode ``'r'`` or ``'rb'``) or
     stdout will be the handle returned.
    :type filename: str or None

    :param mode: permission mode to open the file under (e.g. ``'r'``, ``'w'``, ``'a'``, ...)
    :type mode: str

    :return: handle to the file, stdin, or stdout.
    """
    handle, do_close = smart_handle(filename, mode, return_close=True)

    try:
        yield handle
    finally:
        if do_close:
            handle.close()


class MultiFileHandle(IOBase):
    """
    Class for reading or writing to or from multiple files at once
    """
    def __init__(self, *files, mode='r', read_rtype='dict'):
        """
        Open multiple files for reading or writing
        :param files: give each file to read from or write to.
        :type files: str

        :param mode: access mode, "r", "rb", "w", "wb", "a", "ab". Must be specified as a keyword argument
        :type mode: str

        :param read_rtype: how to return values read from files. By default, they are returned as a dict, with the
         filenames being the keys. Other options are "list" and "tuple", which return the values as lists/tuples
         respectively in the order the file names were specified.
        :type read_rtype: str
        """
        self._file_names = files
        self._handles = tuple([smart_handle(f, mode=mode) for f in files])
        self._read_rtype = read_rtype.lower()

        allowed_rtypes = ('dict', 'list', 'tuple')
        if self._read_rtype not in allowed_rtypes:
            raise ValueError('read_rtype must be one of: {}'.format(', '.join(allowed_rtypes)))

    def close(self):
        """
        Close all attached handles.
        :return:
        """
        for f in self._handles:
            f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _format_read(self, values):
        """
        Format the return values from read, or other returning functions.
        :param values:
        :return:
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
        """
        Call any returning method on all the attached file handles.

        :param fxn_name: the name of the method to call
        :param args: positional arguments for the method
        :param kwargs: keyword arguments for the method
        :return: formatted return value for the call.
        """
        values = dict()
        for name, handle in zip(self._file_names, self._handles):
            fxn = getattr(handle, fxn_name)
            values[name] = fxn(*args, **kwargs)
        return self._format_read(values)

    def read(self, *args, **kwargs):
        return self._return_internal('read', args, kwargs)

    def readline(self, *args, **kwargs):
        return self._return_internal('readline', args, kwargs)

    def readlines(self, *args, **kwargs):
        return self._return_internal('readlines', args, kwargs)

    def write(self, *args, **kwargs):
        for f in self._handles:
            f.write(*args, **kwargs)

    def writelines(self, lines: Iterable[Union[bytes, bytearray]]) -> None:
        for f in self._handles:
            f.writelines(lines)

    def seek(self, *args, **kwargs):
        return self._return_internal('seek', args, kwargs)


def dump_pickle(filename, obj):
    """
    Save an object to a pickle file

    :param filename: the file name to give the pickle output file. Will be overwritten if it exists
    :type filename: str

    :param obj: the Python object to pickle

    :return: None
    """
    with open(filename, 'wb') as pkfile:
        pickle.dump(obj, pkfile, protocol=pickle.HIGHEST_PROTOCOL)


def grab_pickle(filename):
    """
    Load a pickle file

    :param filename: the pickle file to load

    :return: the Python object stored in the pickle file
    """
    with open(filename, 'rb') as pkfile:
        return pickle.load(pkfile)