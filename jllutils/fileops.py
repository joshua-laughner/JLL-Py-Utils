"""
Helpful methods for working with files.
"""


import contextlib
import pickle
import sys

# Import to make accessible through this module
from .subutils import ncdf as ncio, hdf5 as h5io


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
    if filename is None or filename == '-':
        if mode.startswith('r'):
            handle = sys.stdin
        else:
            handle = sys.stdout
        do_close = False
    else:
        handle = open(filename, mode)
        do_close = True

    try:
        yield handle
    finally:
        if do_close:
            handle.close()


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