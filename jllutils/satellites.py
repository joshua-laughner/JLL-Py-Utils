import h5py
import os

from .subutils.ncdf import NcWrapper
from .subutils.sat_filters import SatelliteFilter
from .subutils import sat_filters as filters


class SatelliteData(object):
    """
    Wrapper class for loading satellite data

    This class homogenizing working with satellite data in several ways:

    1. it uses NcWrapper for netCDF files so that groups and variables can be accessed using the same syntax as with
       h5py
    2. it accepts a filter instance that automatically filters bad data based on quality filtering.

    Note that data wrapped by this class can be accessed two ways:

    1. Directly indexing the object (e.g. ``satdat['xco2']``) will access the dataset or group specified in the top
       level. This will **NOT** invoke the filtering.
    2. Using the :meth:`get` method, which **WILL** invoke filtering.
    """
    def __init__(self, data_file, filter=None, file_type=None):
        """
        Open a satellite .h5 or .nc file

        :param data_file: the path to the file to open
        :type data_file: path-like

        :param filter: the filter object to use to determine which values are bad. If ``None`` then no filter will be
         applied.
        :type file_type: :class:`SatelliteFilter`

        :param file_type: which type a file is. May be "h5" or "hdf5" for and HDF5 file or "nc", "nc4", "ncdf", or
         "netcdf" (case insensitive) for a netCDF file. If ``None``, will infer from the file extension. Alternately,
         you may pass a callable that, given the file name as the sole argument, opens the dataset and follows the
         h5py convention for indexing groups/datasets.
        :type filter: str or callable
        """
        self._filter = filter
        wrapper = self._detect_filetype(data_file, file_type)
        self._dataset = wrapper(data_file)

    def _detect_filetype(self, filename, file_type):
        """
        Determine what function to use to open the given file.
        :param filename: path to the file
        :param file_type: user-specified type. May be None (for auto-detection), string, or callable that opens the
         file.
        :return: the callable to use to open the file
        """
        file_type = file_type.lower() if isinstance(file_type, str) else file_type
        if file_type is None:
            ext = os.path.splitext(filename)[1][1:]  # get extension without the dot
            return self._detect_filetype(filename, ext)
        elif file_type in ('h5', 'hdf5'):
            return lambda f: h5py.File(f, 'r')
        elif file_type in ('nc', 'nc4', 'ncdf', 'netcdf'):
            return NcWrapper
        elif isinstance(file_type, str):
            raise ValueError('File type "{}" not recognized'.format(file_type))
        else:
            return file_type

    def __getitem__(self, item):
        """
        Retrieve a group or dataset from the file. Bypass filtering.
        :param item: group or dataset name.
        :return: the group or dataset.
        """
        return self._dataset[item]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self._dataset.close()

    def keys(self):
        """
        Return a tuple of all keys in the top level of the open file
        """
        return self._dataset.keys()

    def get(self, *path):
        """
        Get a dataset from the file and apply filtering.

        :param path: internal path to the dataset. If dataset not in the top level, either specify the full path using
         forward slashes to separate parts, or give each part as a separate argument. That is, the following are
         equivalent::

            satdat.get('Meteorology/psurf')
            satdat.get('Meteorology', 'psurf')

        :type path: str

        :return: the dataset with values rejected by the filter (given when constructing the instance) either replaced
         with a fill value, or masked, depending on the value given to the ``masked`` keyword of the filter.
        :rtype: :class:`numpy.ndarray` or :class:`numpy.ma.masked_array`
        """
        if len(path) == 1:
            path = path[0].split('/')
        elif any('/' in p for p in path):
            raise ValueError('Either use slashes to separate path parts or pass each part as its own argument; do not mix')

        data = self._dataset
        for p in path:
            data = data[p]

        if self._filter is not None:
            return self._filter.apply(data[tuple()], self._dataset)
        else:
            return data
