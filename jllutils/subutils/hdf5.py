"""
This module focuses on making HDF5 files easier to access, including finding variables within HDF5 files.
"""

import contextlib
import h5py
import re


class MultipleChildrenFoundError(Exception):
    """
    Error to use when multiple groups/datasets match in a recursive find and only one was desired
    """
    pass


@contextlib.contextmanager
def smart_h5(name_or_handle, mode='r'):
    """Allow context managing an HDF5 file object or filename

    Use as a context manager, i.e. `with smart_h5(...) as hobj:` when you may have either a path to a HDF5 file or
    an already open HDF5 file handle. In the first case, it will behave identically to `with h5py.File(...) as hobj:`, 
    automatically closing the handle when the ``with`` block is exited. If given an existing handle, it will not close 
    the handle automatically.

    Parameters
    ----------
    name_or_handle : str or :class:`h5py.File`
        the path to the HDF5 file, or an already open file handle. 

    mode : str
        the mode to open the file in. Only has an effect if passing a path; if a handle is given, then this input is 
        ignored and the mode of the existing file takes precedence.
        
    Returns
    -------
    `contextlib._GeneratorContextManager`
        context manager for a HDF5 file dataset.
        
    Warnings
    --------
    Because this returns a context manager it *cannot* be used outside a `with` block; that is, `f = smart_h5(...)` will
    not work as expected.
    
    Examples
    --------
    >>> with smart_h5('demo.h5') as ds:
    >>>     print(ds.keys())
    ['temperature', 'pressure']

    This can also accept an already-open HDF5 file handle.

    >>> f = h5py.File('demo.h5', 'r')
    >>> with smart_h5(f) as ds:
    >>>     print(ds.keys())
    ['temperature', 'pressure']

    This is useful in a function that might be called by directly or by another function. Consider a case where a
    HDF5 file has a quality flag variables that indicates which data is safe to use where it has a value of 0. You
    might want a function that automatically filters data read from such a file, and another than computes a derived
    quantity from variables in that file:
    
    >>> def read_and_filter_h5_var(h5file, varname):
    >>>     with smart_h5(h5file) as ds:
    >>>         data = ds.variables[varname][:]
    >>>         quality_flag = ds.variables['qual_flag'][:]
    >>>         return data[quality_flag == 0]
    >>>
    >>> def compute_potential_temperature(h5file):
    >>>     with smart_h5(h5file) as ds:
    >>>         p = read_and_filter_h5_var(ds, 'pressure')
    >>>         t = read_and_filter_h5_var(ds, 'temperature')
    >>>     return t * (1000/p) ** 0.286

    While we could just do:

    >>> def compute_potential_temperature(h5file):
    >>>     read_and_filter_h5_var(h5file, 'pressure')
    >>>     read_and_filter_h5_var(h5file, 'temperature')
    >>>     return t * (1000/p) ** 0.286

    the former provides a slightly cleaner implementation because we're not opening and closing the HDF5 file multiple
    times. This is more important if writing to the file.
    """

    if isinstance(name_or_handle, h5py.File):
        handle = name_or_handle
        do_close = False
    else:
        handle = h5py.File(name_or_handle, mode)
        do_close = True

    try:
        yield handle
    finally:
        if do_close:
            handle.close()


def h5dump(file_or_handle):
    """Print a visual tree of groups and datasets in an HDF5 file.

    Parameters
    ----------
    file_or_handle : str or :class:`h5py.File`
        the path or handle to the HDF5 file to dump

    Returns
    -------
    None
        prints the tree to the screen
    """
    bullets = ('*', '-', '>')

    def h5dump_internal(fobj, depth):
        prefix_str = '  ' * depth + bullets[depth % len(bullets)]
        for k, v in fobj.items():
            print(prefix_str, k, '({})'.format(type(v).__name__))
            if isinstance(v, h5py.Group):
                h5dump_internal(v, depth+1)

    with smart_h5(file_or_handle) as f:
        h5dump_internal(f, 0)


def h5find(file_or_handle, target_name, target_type='dataset', unique=True, return_type='path', read_h5=False,
           use_regex=False):
    """Find a group or dataset in an HDF5 file.

    Parameters
    ----------

    file_or_handle : str or :class:`h5py.File`
        either the path to the HDF5 file or an instance of :class:`h5py.File` already open. However, see notes under
        `return_type` about limitations of passing a file name.

    target_name : str
        the name of the group/dataset to search for. If `use_regex` is `False`, this is matched with simple
        equality. If `use_regex` is `True`, this is matched with `re.match`.

    unique : bool
        if `True` then only one group or dataset may be matched. If more are matched, an error is raised; if none, then
        `None` is returned.  If exactly one is matched, then the matched value is returned. See `return_type` for more
        information.

    target_type : str or type
        controls what types to match. By default, only match datasets, not groups. This can be changed to match groups
        by passing the string "group" or both datasets and groups with "both". Alternately, pass a type or collection of
        types to serve as the second argument to :func:`isinstance`, only children matching those types will be
        returned.

    return_type : str
        controls how the matched children are returned, in conjunction with ``read_h5`` and ``unique``. Options are:

      * "path" - return the path(s) to the matching child(ren) (e.g. "/HDFEOS/SWATHS/N2O/Data Fields/N2O"). If
        ``unique`` is ``False``, then a list of paths is returned. If `unique` is `True`, then the matching path is
        returned as a string.
      * "object" - return the object(s) that matched, as a list if `unique` is ``False``, or the sole object
        otherwise. If `read_h5` is `True` and the child matched is a dataset, the contents are read in and returned.
        If `read_h5` is `True` and the child is not a Dataset, a string representation of it is returned. Otherwise,
        the h5py object (i.e. :class:`h5py.Dataset` or :class:`h5py.Group`) is returned.
      * "dict" - returns a dictionary where the keys are the paths and the values the objects. This is not modified when
        `unique` is `True`, however, and an error is still raised if >1 child matches in that case.

     Because of how :class:`h5py.File` instances work, dataset or group objects pointing to a closed file cannot be
     accessed. Therefore, if a filename is passed as `file_or_handle`, then either `return_type` must be "path" or
     `read_h5` must be `True`. If not, then useless closed handles would be returned.

    read_h5 : bool
        causes datasets to be read in, and string representations of other types to be stored, rather than the original
        h5py objects. See `return_type` for details.

    use_regex : bool
        affects how `target_name` is matched against child names. See `target_name` for details.

    Returns
    -------
    object
        the matching child object(s) from the HDF5 file. See `return_type` for details.

    Raises
    ------
    MultipleChildrenFoundError
        if `unique` is `True` and multiple children are matched.
    """
    if target_type in ('dataset', 'dset'):
        target_type = h5py.Dataset
    elif target_type == 'group':
        target_type = h5py.Group
    elif target_type is None or target_type == 'both':
        target_type = (h5py.Dataset, h5py.Group)

    if not isinstance(target_type, (list, tuple)):
        target_type = (target_type,)

    allowed_rtypes = ('object', 'path', 'dict')
    if return_type not in allowed_rtypes:
        raise ValueError('return_type must be one of: {}'.format(', '.join(allowed_rtypes)))

    if return_type != 'path' and not read_h5 and not isinstance(file_or_handle, h5py.File):
        raise ValueError('To use a return type other than "path", you must either pass an h5py file handle as the '
                         'first arguement (instead of a filename) or set read_h5 = True. Otherwise, the '
                         'groups/datasets returned will already be closed and not be usable.')

    if use_regex:
        match_fxn = lambda k: re.match(target_name, k)
    else:
        match_fxn = lambda k: k == target_name

    matches = dict()

    def find_internal(grp, path):
        for k, v in grp.items():
            this_path = path + k
            if match_fxn(k) and isinstance(v, target_type):
                if read_h5 and isinstance(v, h5py.Dataset):
                    matches[this_path] = v[:]
                elif read_h5:
                    matches[this_path] = repr(v)
                else:
                    matches[this_path] = v
            if isinstance(v, h5py.Group):
                find_internal(v, this_path + '/')

    with smart_h5(file_or_handle) as fobj:
        filename = fobj.filename
        find_internal(fobj, '/')

    if return_type == 'object':
        matches = list(matches.values())
    elif return_type == 'path':
        matches = list(matches.keys())

    if unique:
        if len(matches) > 1:
            raise MultipleChildrenFoundError('Multiple children of type(s) {} found matching {} in {}'
                                             .format(', '.join(t.__name__ for t in target_type), target_name, filename))
        elif len(matches) == 0:
            matches = None
        else:
            matches = matches[0]

    return matches


def h5getpath(h5handle, path):
    """Get a group or dataset from an arbitrary path

    Parameters
    ----------
    h5handle : :class:`h5py.File` or :class:`h5py.Group`
        handle to an HDF5 File or Group.
    path : list(str)
        elements of the path to the dataset or group in the HDF5 file

    Returns
    -------
    :class:`h5py.Group` or :class:`h5py.Dataset`
        The group or dataset pointed to by the path

    Examples
    --------
    Assume that "demo.h5" contains the "sst" dataset in the "oceans" group which is under the "data" group (i.e. the
    path is data/oceans/sst):

    >>> import h5py
    >>> with h5py.File('demo.h5', 'r') as f:
    >>>     sst_dset = h5getpath(f, ['data', 'oceans', 'sst'])
    """
    f = h5handle
    for p in path:
        f = f[p]
    return f
