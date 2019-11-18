from __future__ import print_function, absolute_import, division, unicode_literals

import cftime
from collections import OrderedDict
import contextlib
import datetime as dt
from hashlib import sha1
import netCDF4 as ncdf
import numpy as np
import pandas as pd
import re


class VarnameConflictError(Exception):
    """
    Error indicating a conflict with variable names
    """
    pass


class FindingDimensionError(Exception):
    """
    Error indicating trouble finding a dimension
    """
    pass


class DimensionMatchingError(Exception):
    """
    Error indicating that a variable does not match the expected dimensions
    """
    pass


@contextlib.contextmanager
def smart_nc(name_or_handle, mode='r'):
    """
    Allow context managing a netCDF4 dataset or filename

    Use as a context manager, i.e. ``with smart_nc(...) as nh:`` when you may have either a path to a netCDF file or
    an already open netCDF dataset handle. In the first case, it will behave identically to
    ``with netCDF4.Dataset(...) as nh:``, automatically closing the handle when the ``with`` block is exited. If given
    an existing handle, it will not close the handle automatically.

    :param name_or_handle: the path to the netCDF file, or an already open Dataset handle.
    :type name_or_handle: str or :class:`netCDF4.Dataset`

    :param mode: the mode to open the dataset in. Only has an effect if passing a path; if a handle is given, then this
     input is ignored and the mode of the existing dataset takes precedence.
    :type mode: str

    :return: context manager for a netCDF dataset.
    """

    if isinstance(name_or_handle, ncdf.Dataset):
        handle = name_or_handle
        do_close = False
    else:
        handle = ncdf.Dataset(name_or_handle, mode)
        do_close = True

    try:
        yield handle
    finally:
        if do_close:
            handle.close()


def dataframe_to_ncdf(df, ncfile, index_name=None, index_attrs=None, attrs=None):
    """
    Save a dataframe as a netCDF file

    :param df: the dataframe to save
    :type df: :class:`pandas.DataFrame`

    :param ncfile: the path of the netCDF4 file to create, or a handle to an existing netCDF4 group.
    :type ncfile: str or :class:`netCDF4.Group`

    :param index_name: the name to give the variable created from the index. This name will always be used if given, but
     only needs to be given if the index is not named.
    :type index_name: str

    :param index_attrs: a dict-like object specifying the attributes to give the index dimension variable. Note that if
     the index is a DatetimeIndex, the "units", "calendar", and "base_date" attributes will be automatically provided.
    :type index_attrs: dict-like

    :param attrs: a dict-like object specifying the attributes for each column of the dataframe. The keys of the top
     dict must correspond to column names, and the values will themselves be dicts specifying the attribute names and
     values. Not all columns need exist; any columns absent from this dict will simply have no attributes.
    :type attrs: dict-like

    :return: None, writes a netCDF4 file.
    """
    if index_name is None:
        if df.index.name is None:
            raise TypeError('If the index of your dataframe is not named, then you must provide a value for index_name')
        else:
            index_name = df.index.name

    if isinstance(df.index, pd.MultiIndex):
        raise NotImplementedError('Dataframes with a multi index are not supported')

    if index_attrs is None:
        index_attrs = dict()
    if attrs is None:
        attrs = dict()

    # Create a netcdf file, write the dimension. If it is a datetime index, write it as a time variable, otherwise,
    # write it as a standard dimension.
    with smart_nc(ncfile, 'w') as nch:
        if isinstance(df.index, pd.DatetimeIndex):
            dim = make_nctimedim_helper(nch, index_name, df.index, **index_attrs)
        else:
            dim = make_ncdim_helper(nch, index_name, df.index.to_numpy(), **index_attrs)

        # Write each column of the data frame as a netCDF variable with the index as its dimension
        for colname, data in df.items():
            col_attrs = attrs[colname] if colname in attrs else dict()
            make_ncvar_helper(nch, colname, data.to_numpy(), [dim], **col_attrs)
        

def ncdf_to_dataframe(ncfile, target_dim=None, unmatched_vars='silent', top_vars_only=False, no_leading_slash=True, fullpath=False):
    """
    Read in a netCDF file's 1D variables into a Pandas dataframe

    This function will scan a netCDF file for all 1D variables along a specific dimension and read them in as a Pandas
    dataframe. It will return a second dataframe with the attributes of the variables.  It will iterate through groups
    and store their variables as well, unless ``top_vars_only`` is ``True``.

    :param ncfile: the netCDF file to read. Alternatively a handle to a netCDF group.
    :type ncfile: str or :class:`netCDF4.Dataset`.

    :param target_dim: the name of the dimension to target - only variables that have this as their sole dimension will
     be included. If this is not specified, then the following logic is applied:

        1. Is there only one dimension? If yes, use that, if not:
        2. Is there one (and only one) unlimited dimension? If so, that dimension is used. If there is >1, an error is
           raised. If there are none, then raise an error.

     Note that this only searches dimensions in the top level group. That is, if you specify the dimension 'time',
     'time' must be in the top group's dimensions, and not any child groups.
    :type target_dim: str

    :param unmatched_vars: controls what happens if a variable with the wrong dimensions is found:

        * "silent" (default) - skips that variable silently.
        * "warn" - print a warning to stdout
        * "error" - raise an exception

    :type unmatched_vars: str

    :param top_vars_only: set to ``True`` to only store variables in the top group; do not recurse into lower groups.
    :type top_vars_only: bool

    :param no_leading_slash: by default, variables will be named with their full path in the netCDF file, but with the
     initial / removed. Set this to ``True`` to enable that initial slash.
    :type no_leading_slash: bool

    :return: two dataframes, one with the data (variables are columns, dimension as index), one with attributes
     (attribute name will be the index)
    :rtype: :class:`pandas.DataFrame`, :class:`pandas.DataFrame`,
    """

    target_dim, dim_size, dim_values = _find_1d_dim(ncfile, target_dim)
    var_dict = OrderedDict()
    attr_dfs_list = []
    with smart_nc(ncfile) as nh:
        _ncdf_to_df_internal(nh, dim_name=target_dim, dim_size=dim_size, var_dict=var_dict, att_dfs=attr_dfs_list,
                             top_vars_only=top_vars_only, unmatched_vars=unmatched_vars.lower(),
                             no_leading_slash=no_leading_slash, fullpath=fullpath)

    var_df = pd.DataFrame(var_dict, index=dim_values)
    attr_df = pd.concat(attr_dfs_list, axis=1, sort=True)
    return var_df, attr_df


def _find_1d_dim(ncfile, target_dim):
    with smart_nc(ncfile) as nh:
        if target_dim is not None:
            if target_dim in nh.dimensions:
                dim_name = target_dim
            else:
                raise FindingDimensionError('Specified dimension "{}" is not in the netCDF file {}'
                                            .format(target_dim, nh.filepath()))
        else:
            dim_name = None
            if len(nh.dimensions) == 1:
                dim_name = list(nh.dimensions.keys())[0]
            else:
                for name, dim in nh.dimensions.items():
                    if dim.isunlimited():
                        if dim_name is None:
                            dim_name = name
                        else:
                            raise FindingDimensionError('Multiple unlimited dimensions ({}, {}) found'.format(dim_name, name))

                if dim_name is None:
                    raise FindingDimensionError('Could not automatically determine which dimension to use as the index. '
                                                'Pass a dimension name manually.')

        dim_size = nh.dimensions[dim_name].size
        if dim_name in nh.variables:
            tmp_dim_values = nh.variables[dim_name][:].filled(np.nan)
            try:
                # Is this a time axis?
                dim_values = pd.DatetimeIndex(ncdf.num2date(tmp_dim_values, nh.variables[dim_name].units))
            except (ValueError, AttributeError):
                # Not a datetime axis or missing units, can't convert to datetime
                dim_values = tmp_dim_values
        else:
            dim_values = pd.RangeIndex(dim_size)

    return dim_name, dim_size, dim_values


def _ncdf_to_df_internal(nch, dim_name, dim_size, var_dict, att_dfs, top_vars_only, unmatched_vars, no_leading_slash, fullpath):
    path = nch.path
    dim_tuple = (dim_name,)
    for varname, variable in nch.variables.items():
        if variable.dimensions != dim_tuple and variable.size != dim_size:
            if unmatched_vars == 'silent':
                pass
            elif unmatched_vars == 'warn':
                print('Warning: variable "{}" is not 1D with dimension "{}", skipping'.format(varname, dim_name))
            elif unmatched_vars == 'error':
                raise DimensionMatchingError('Variable "{}" is not 1D with dimension "{}"'.format(varname, dim_name))
            else:
                raise ValueError('Value "{}" for unmatched_vars is not valid. Must be "silent", "warn", or "error"')
            continue

        if fullpath:
            full_varname = re.sub(r'//+', '/', path + '/' + varname)
        else:
            full_varname = varname

        if no_leading_slash:
            full_varname = re.sub(r'^/', '', full_varname)

        if full_varname in var_dict:
            # This really should not happen
            raise VarnameConflictError('{} is already present in the variable dictionary'.format(full_varname))

        if np.issubdtype(variable.dtype, np.floating):
            var_array = variable[:].filled(np.nan)
        else:
            var_array = variable[:].data
        var_dict[full_varname] = var_array
        # Read the attributes and put them into their own dataframe
        this_att_dict = {name: [variable.getncattr(name)] for name in variable.ncattrs()}
        this_att_df = pd.DataFrame(this_att_dict, index=[full_varname]).T
        att_dfs.append(this_att_df)

    if top_vars_only:
        return

    for group in nch.groups.values():
        _ncdf_to_df_internal(group, dim_name=dim_name, dim_size=dim_size, var_dict=var_dict, att_dfs=att_dfs,
                             top_vars_only=top_vars_only, unmatched_vars=unmatched_vars,
                             no_leading_slash=no_leading_slash)


def make_ncdim_helper(nc_handle, dim_name, dim_var, **attrs):
    """
    Create a netCDF dimension and its associated variable simultaneously

    Typically in a netCDF file, each dimension should have a variable with the same name that defines the coordinates
    for that dimension. This function streamlines the process of creating a dimension with its associated variable.

    :param nc_handle: the handle to a netCDF file open for writing, returned by :class:`netCDF4.Dataset`
    :type nc_handle: :class:`netCDF4.Dataset`

    :param dim_name: the name to give both the dimension and its associated variable
    :type dim_name: str

    :param dim_var: the variable to use when defining the dimension's coordinates. The dimensions length will be set
     by the size of this vector. This must be a 1D numpy array or comparable type.
    :type dim_var: :class:`numpy.ndarray`

    :param attrs: keyword-value pairs defining attribute to attach to the dimension variable.

    :return: the dimension object
    :rtype: :class:`netCDF4.Dimension`
    """
    if np.ndim(dim_var) != 1:
        raise ValueError('Dimension variables are expected to be 1D')
    dim = nc_handle.createDimension(dim_name, np.size(dim_var))
    var = nc_handle.createVariable(dim_name, dim_var.dtype, dimensions=(dim_name,))
    var[:] = dim_var
    var.setncatts(attrs)
    return dim


def make_nctime(timedata, base_date=dt.datetime(1970,1,1), time_units='seconds', calendar='gregorian',
                base_date_nc_time=True):
    allowed_time_units = ('seconds', 'minutes', 'hours', 'days')
    if time_units not in allowed_time_units:
        raise ValueError('time_units must be one of: {}'.format(', '.join(allowed_time_units)))

    units_str = '{} since {}'.format(time_units, base_date.strftime('%Y-%m-%d %H:%M:%S'))
    # date2num requires that the dates be given as basic datetimes. We'll handle converting Pandas timestamps, either
    # as a series or datetime index, but other types will need handled by the user.
    try:
        date_arr = ncdf.date2num(timedata, units_str, calendar=calendar)
    except TypeError:
        if isinstance(timedata, np.ndarray):
            dim_var = timedata.astype('datetime64[s]').tolist()
        else:
            dim_var = [d.to_pydatetime() for d in timedata]
        date_arr = ncdf.date2num(dim_var, units_str, calendar=calendar)
    if base_date_nc_time:
        base_date = cftime.date2num(base_date, units_str, calendar=calendar)
    time_info_dict = {'units': units_str, 'calendar': calendar, 'base_date': base_date}
    return date_arr, time_info_dict


def make_nctimedim_helper(nc_handle, dim_name, dim_var, base_date=dt.datetime(1970, 1, 1), time_units='seconds',
                          calendar='gregorian', **attrs):
    """
    Create a CF-style time dimension.

    The Climate and Forecast (CF) Metadata Conventions define standardized conventions for how to represent geophysical
    data. Time is one of the trickiest since there are multiple ways of identifying dates that can be ambiguous. The
    standard representation is to give time in units of seconds/minutes/hours/days since a base time in a particular
    calendar. This function handles creating the a time dimension and associated variable from any array-like object of
    datetime-like object.

    For more information, see:

        http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html#time-coordinate

    :param nc_handle: the handle to a netCDF file open for writing, returned by :class:`netCDF4.Dataset`
    :type nc_handle: :class:`netCDF4.Dataset`

    :param dim_name: the name to give both the dimension and its associated variable
    :type dim_name: str

    :param dim_var: the variable to use when defining the dimension's coordinates. The dimensions length will be set
     by the size of this vector. This must be a 1D numpy array or comparable type.
    :type dim_var: :class:`numpy.ndarray`

    :param base_date: the date and time to make the time coordinate relative to. The default is midnight, 1 Jan 1970.
    :type base_date: datetime-like object

    :param time_units: a string indicating what unit to use as the count of time between the base date and index date.
     Options are 'seconds', 'minutes', 'hours', or 'days'.  This is more restrictive than the CF convention, but avoids
     the potential pitfalls of using months or years.
    :type time_units: str

    :param calendar: one of the calendar types defined in the CF conventions document (section 4.4.1)
    :type calendar: str

    :param attrs: keyword-value pairs defining attribute to attach to the dimension variable.

    :return: the dimension object, and a dictionary describing the units, calendar, and base date of the time dimension
    :rtype: :class:`netCDF4.Dimension`, dict
    """
    date_arr, time_info_dict = make_nctime(dim_var, base_date=base_date, time_units=time_units, calendar=calendar, base_date_nc_time=True)
    attrs = attrs.copy()
    attrs.update(time_info_dict)
    dim = make_ncdim_helper(nc_handle, dim_name, date_arr, **attrs)
    return dim


def make_nctimevar_helper(nc_handle, var_name, var_data, dims, base_date=dt.datetime(1970, 1, 1), time_units='seconds',
                          calendar='gregorian', **attrs):
    """
    Create a netCDF variable for times and store the data converted to CF-compliant units simultaneously.

    This function uses :func:`make_nctime` internally to convert the ``var_data`` to a CF-compliant array
    then uses :func:`make_ncvar_helper` to create the variable.

    :param nc_handle: the handle to a netCDF file open for writing, returned by :class:`netCDF4.Dataset`
    :type nc_handle: :class:`netCDF4.Dataset`

    :param var_name: the name to give the variable
    :type var_name: str

    :param var_data: the array to store in the netCDF variable.
    :type var_data: :class:`numpy.ndarray` or list(datetimes)

    :param dims: the dimensions to associate with this variable. Must be a collection of either dimension names or
     dimension instances. Both types may be mixed. This works well with :func:`make_ncdim_helper`, since it returns the
     dimension instances.
    :type dims: list(:class:`netCDF4.Dimensions` or str)

    :param base_date: the date and time to make the time coordinate relative to. The default is midnight, 1 Jan 1970.
    :type base_date: datetime-like object

    :param time_units: a string indicating what unit to use as the count of time between the base date and index date.
     Options are 'seconds', 'minutes', 'hours', or 'days'.  This is more restrictive than the CF convention, but avoids
     the potential pitfalls of using months or years.
    :type time_units: str

    :param calendar: one of the calendar types defined in the CF conventions document (section 4.4.1)
    :type calendar: str
    
    :param attrs: keyword-value pairs defining attribute to attach to the dimension variable.

    :return: the variable object
    :rtype: :class:`netCDF4.Variable`
    """
    date_arr, time_info_dict = make_nctime(var_data, base_date=base_date, time_units=time_units, calendar=calendate, base_date_nc_time=True)
    attrs = attrs.copy()
    attrs.update(time_info_dict)
    return make_ncvar_helper(nc_handle, var_name, date_arr, dims, make_cf_time_auto=False, **attrs)


def make_ncvar_helper(nc_handle, var_name, var_data, dims, make_cf_time_auto=True, **attrs):
    """
    Create a netCDF variable and store the data for it simultaneously.

    This function combines call to :func:`netCDF4.createVariable` and assigning the variable's data and attributes.

    :param nc_handle: the handle to a netCDF file open for writing, returned by :class:`netCDF4.Dataset`
    :type nc_handle: :class:`netCDF4.Dataset`

    :param var_name: the name to give the variable
    :type var_name: str

    :param var_data: the array to store in the netCDF variable.
    :type var_data: :class:`numpy.ndarray`

    :param dims: the dimensions to associate with this variable. Must be a collection of either dimension names or
     dimension instances. Both types may be mixed. This works well with :func:`make_ncdim_helper`, since it returns the
     dimension instances.
    :type dims: list(:class:`netCDF4.Dimensions` or str)

    :param make_cf_time_auto: when this is ``True``, then any array with a datetime64 data type will be converted to
     a CF-compliant time variable automatically. If ``False``, no conversion will be attempted, which may results in
     a TypeError.
    :type make_cf_time_auto: bool

    :param attrs: keyword-value pairs defining attribute to attach to the dimension variable.

    :return: the variable object
    :rtype: :class:`netCDF4.Variable`
    """
    dim_names = tuple([d if isinstance(d, str) else d.name for d in dims])
    if np.issubdtype(var_data.dtype, np.datetime64) and make_cf_time_auto:
        var_data, time_info_dict = make_nctime(var_data)
        attrs.update(time_info_dict)
    var = nc_handle.createVariable(var_name, var_data.dtype, dimensions=dim_names)
    var[:] = var_data
    var.setncatts(attrs)
    return var


def make_dependent_file_hash(dependent_file):
    """
    Create an SHA1 hash of a file.

    :param dependent_file: the path to the file to hash.
    :type dependent_file: str

    :return: the SHA1 hash
    :rtype: str
    """
    hashobj = sha1()
    with open(dependent_file, 'rb') as fobj:
        block = fobj.read(4096)
        while block:
            hashobj.update(block)
            block = fobj.read(4096)

    return hashobj.hexdigest()


def add_dependent_file_hash(nc_handle, hash_att_name, dependent_file):
    """
    Add an SHA1 hash of another file as an attribute to a netCDF file.

    This is intended to create an attribute that list the SHA1 hash of a file that the netCDF file being created
    depends on.

    :param nc_handle: a handle to the netCDF4 dataset to add the attribute to.
    :type nc_handle: :class:`netCDF4.Dataset`

    :param hash_att_name: the name to give the attribute that will store the hash. It is recommended to include the
     substring "sha1" so that it is clear what hash function was used.
    :type hash_att_name: str

    :param dependent_file: the file to generate the hash of.
    :type dependent_file: str

    :return: None
    """
    hash_hex = make_dependent_file_hash(dependent_file)
    nc_handle.setncattr(hash_att_name, hash_hex)


class NcWrapper(object):
    def __init__(self, dataset, no_masked=True):
        if isinstance(dataset, str):
            self._nc_handle = ncdf.Dataset(dataset)
        else:
            self._nc_handle = dataset
        self._no_masked = no_masked

    def __getitem__(self, item):
        ds = self._nc_handle
        if isinstance(ds, ncdf.Variable):
            data = ds[item]
            if self._no_masked:
                try:
                    return data.filled(np.nan)
                except AttributeError:
                    return data
        elif item in ds.groups:
            return self.__class__(ds.groups[item])
        elif item in ds.variables:
            return self.__class__(ds.variables[item])
        else:
            raise KeyError(item)

    def __getattr__(self, item):
        return getattr(self._nc_handle, item)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self):
        return self._nc_handle.__repr__()

    def close(self):
        if self._nc_handle is not None:
            self._nc_handle.close()
            self._nc_handle = None

    def keys(self):
        ds = self._nc_handle
        group_keys = list(ds.groups.keys())
        var_keys = list(ds.variables.keys())
        return tuple(group_keys) + tuple(var_keys)