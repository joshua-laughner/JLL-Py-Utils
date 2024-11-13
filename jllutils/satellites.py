import h5py
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
import os

from .subutils.ncdf import NcWrapper


def plot_pixel_polygons(loncorn: np.ndarray, latcorn: np.ndarray, values: np.ndarray, ax=None, **collection_kws) -> PatchCollection:
    """Plot satellite pixels as polygons.

    Parameters
    ----------
    loncorn, latcorn 
        The longitude and latitude corners of the pixels. Must be 2D arrays, the first dimension must be the number
        of pixels and the second dimension the number of corners. Any number of corners >= 3 are allowed, but the number
        of corners must be the same in both arrays. If your axes use a map projection other than PlateCarree or equirectangular,
        then the safest approach is to transform these coordinates into the map projection yourself.

    values
        The values to color the pixels by. Must be a 1D array with the same number of elements as the first dimensions
        of loncorn and latcorn.

    ax
        Which axes to plot into. If not given, the axes returned by ``plt.gca()`` are used.

    collection_kws
        Additional keywords are passed through to :func:`matplotlib.collections.PatchCollection`. Most commonly used ones are:

        - ``alpha``: set transparency (0 to 1)
        - ``clim``: set colorbar limits as ``(vmin, vmax)``
        - ``cmap``: set the color map
        - ``edgecolor``, ``edgecolors``, or ``ec``: set the pixel edge color
        - ``label``: set the legend label
        - ``linestyle``, ``linestyles`` or ``ls``: set the edge line style
        - ``linewidth``, ``linewidths``, or ``lw``: set the edge line width
        - ``norm``: set the colorbar normalization
        - ``transform``: set which coordiate system to interpret the lat/lon values in. For maps in non-equirectangular projections,
          setting this to ``cartopy.crs.PlateCarree()`` might work to transform the pixels into the new projection (not tested).
        - ``zorder``: set where the pixels are rendered in the stack relative to other elements.

    Returns
    -------
    patch_col
        The patch collection; this can be used as the first argument to ``plt.colorbar``.
    """
    shape_lon = np.shape(loncorn)
    shape_lat = np.shape(latcorn)
    shape_values = np.shape(values)
    if shape_lon != shape_lat:
        raise ValueError(f'loncorn and latcorn must be the same shape, got {shape_lon} and {shape_lat}')
    if shape_lon[0] != shape_values[0]:
        raise ValueError(f'loncorn, latcorn, and values must have the same length in the first dimension, but got {shape_lon[0]}, {shape_lat[0]}, and {shape_values[0]}')
    if np.ndim(values) != 1:
        raise ValueError(f'values must be 1D, got {np.ndim(values)}D')
    
    pixels = []
    for xc, yc in zip(loncorn, latcorn):
        coords = np.vstack([xc, yc]).T
        pixels.append(Polygon(coords, closed=True))
        
    ax = ax or plt.gca()
    collection_kws.setdefault('in_layout', True)  # try to make sure this gets included in layout calculations
    p = PatchCollection(pixels, **collection_kws)
    p.set_array(values)
    ax.add_collection(p)
    
    # For whatever reason, adding a collection does not update the axis
    # limits, so we do that manually.
    if ax.get_autoscalex_on():
        xmin = np.nanmin(loncorn)
        xmax = np.nanmax(loncorn)
        ax.set_xbound(xmin, xmax)
    if ax.get_autoscaley_on():
        ymin = np.nanmin(latcorn)
        ymax = np.nanmax(latcorn)
        ax.set_ybound(ymin, ymax)
    return p


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

    def get(self, *path, do_filter=True):
        """
        Get a dataset from the file and apply filtering.

        :param path: internal path to the dataset. If dataset not in the top level, either specify the full path using
         forward slashes to separate parts, or give each part as a separate argument. That is, the following are
         equivalent::

            satdat.get('Meteorology/psurf')
            satdat.get('Meteorology', 'psurf')

        :type path: str

        :param do_filter: controls whether to apply the filter. Must be given as keyword.
        :type do_filter: bool

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

        data = data[tuple()]
        if self._filter is not None and do_filter:
            return self._filter.apply(data, self._dataset)
        else:
            return data
