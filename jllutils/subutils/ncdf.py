"""
This module contains various functions to work more efficiently with netCDF files. It is particularly focused on making
it easier to create netCDF files.
"""
from __future__ import print_function, absolute_import, division, unicode_literals
from typing import Optional

import cftime
from collections import OrderedDict
import contextlib
import datetime as dt
from hashlib import sha1
import netCDF4 as ncdf
from netrc import netrc
import numpy as np
import pandas as pd
import re
import xarray as xr

from pydap.client import open_url
from pydap.cas.urs import setup_session


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
    """Allow context managing a netCDF4 dataset or filename

    Use as a context manager, i.e. `with smart_nc(...) as nh:` when you may have either a path to a netCDF file or
    an already open netCDF dataset handle. In the first case, it will behave identically to
    `with netCDF4.Dataset(...) as nh:`, automatically closing the handle when the ``with`` block is exited. If given
    an existing handle, it will not close the handle automatically.

    Parameters
    ----------
    name_or_handle : str or :class:`netCDF4.Dataset`
        the path to the netCDF file, or an already open Dataset handle.
    mode : str
        the mode to open the dataset in. Only has an effect if passing a path; if a handle is given, then this
        input is ignored and the mode of the existing dataset takes precedence.

    Returns
    -------
    :class:`contextlib._GeneratorContextManager`
        context manager for a netCDF dataset.

    Examples
    --------
    >>> with smart_nc('demo.nc') as ds:
    >>>     print(ds.variables.keys())
    ['temperature', 'pressure']

    This can also accept an already-open netCDF dataset handle.

    >>> ncds = netCDF4.Dataset('demo.nc')
    >>> with smart_nc(ncds) as ds:
    >>>     print(ds.variables.keys())
    ['temperature', 'pressure']

    This is useful in a function that might be called by directly or by another function. Consider a case where a
    netCDF file has a quality flag variables that indicates which data is safe to use where it has a value of 0. You
    might want a function that automatically filters data read from such a file, and another than computes a derived
    quantity from variables in that file:

    >>> def read_and_filter_nc_var(ncfile, varname):
    >>>     with smart_nc(ncfile) as ds:
    >>>         data = ds.variables[varname][:]
    >>>         quality_flag = ds.variables['qual_flag'][:]
    >>>         return data[quality_flag == 0]
    >>>
    >>> def compute_potential_temperature(ncfile):
    >>>     with smart_nc(ncfile) as ds:
    >>>         p = read_and_filter_nc_var(ds, 'pressure')
    >>>         t = read_and_filter_nc_var(ds, 'temperature')
    >>>     return t * (1000/p) ** 0.286

    While we could just do:

    >>> def compute_potential_temperature(ncfile):
    >>>     read_and_filter_nc_var(ncfile, 'pressure')
    >>>     read_and_filter_nc_var(ncfile, 'temperature')
    >>>     return t * (1000/p) ** 0.286

    the former provides a slightly cleaner implementation because we're not opening and closing the netCDF file multiple
    times. This is more important if writing to the file.
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
    """Save a dataframe as a netCDF file

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        the dataframe to save

    ncfile : str or :class:`netCDF4.Group`
        the path of the netCDF4 file to create, or a handle to an existing netCDF4 group.

    index_name : str
         the name to give the variable created from the index. This name will always be used if given, but only needs
         to be given if the index is not named.

    index_attrs : dict
        a dict-like object specifying the attributes to give the index dimension variable. Note that if the index is a
        DatetimeIndex, the "units", "calendar", and "base_date" attributes will be automatically provided.

    attrs : dict
     a dict-like object specifying the attributes for each column of the dataframe. The keys of the top dict must
     correspond to column names, and the values will themselves be dicts specifying the attribute names and values. Not
     all columns need exist; any columns absent from this dict will simply have no attributes.

    Examples
    --------

    Usually a dataframe's index is not named, so we must provide the name to use in the netCDF file with the
    `index_name` keyword.

    >>> df = pd.DataFrame({'a': range(10), 'b': range(0, 100, 10)}, index=pd.date_range('2019-01-01', '2019-02-01', periods=10))
    >>> dataframe_to_ncdf(df, 'demo.nc', index_name='time')

    However, if the index is named, the `index_name` is not necessary:

    >>> df.index.name = 'date'
    >>> dataframe_to_ncdf(df, 'demo.nc')

    Providing attributes:

    >>> var_attrs = {'a': {'long_name': 'The A variable'}, 'b': {'long_name': 'The B variable'}}
    >>> dataframe_to_ncdf(df, 'demo.nc', attrs=var_attrs)
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
        

def ncdf_to_dataframe(ncfile, target_dim=None, unmatched_vars='silent', top_vars_only=False, no_leading_slash=True, fullpath=False,
                      convert_time=True, time_vars=tuple(), ret_attrs=False):
    """
    Read in a netCDF file's 1D variables into a Pandas dataframe

    This function will scan a netCDF file for all 1D variables along a specific dimension and read them in as a Pandas
    dataframe. It will return a second dataframe with the attributes of the variables.  It will iterate through groups
    and store their variables as well, unless `top_vars_only` is `True`.

    Parameters
    ----------
    ncfile : str or :class:`netCDF4.Dataset`
        the netCDF file to read. Alternatively a handle to a netCDF group.

    target_dim : str
        the name of the dimension to target - only variables that have this as their sole dimension will
        be included. If this is not specified, then the following logic is applied:

            1. Is there only one dimension? If yes, use that, if not:
            2. Is there one (and only one) unlimited dimension? If so, that dimension is used. If there is >1, an error
               is raised. If there are none, then raise an error.

        Note that this only searches dimensions in the top level group. That is, if you specify the dimension 'time',
        'time' must be in the top group's dimensions, and not any child groups.

    unmatched_vars : str
        controls what happens if a variable with the wrong dimensions is found:

            * "silent" (default) - skips that variable silently.
            * "warn" - print a warning to stdout
            * "error" - raise an exception

    recurse : bool
        set to `True` to only store variables in the top group; do not recurse into lower groups.

    no_leading_slash : bool
        If columns are named with their full path from the netCDF file, the initial / will be removed. Set this to
        `True` to keep that initial slash.

    fullpath : bool or None
        Control whether columns are named by their full path or not. A full path would include the group names that
        the variables fall under. The default is to use full names only if `recurse` is `True`. This can be set to
        `True` or `False` to override that behavior.

    convert_time : bool
        try to convert time variables automatically. Time variables are recognized if they have the
        "calendar" attribute.

    time_vars : Sequence[str]
        A sequence of strings giving variable names to always convert to DatetimeIndex variables.

    ret_attrs : bool
        if `True`, return a second dataframe with the attributes. Otherwise, just return the data dataframe.
    
    Returns
    -------
    :class:`pandas.DataFrame`
        Data frame containing the 1D variables with the appropriate dimension from the specified netCDF file.
    :class:`pandas.DataFrame`
        Data frame containing the attributes for the variables; attribute names will be the index. Only returned
        if `ret_attrs` is `True`.
    """

    target_dim, dim_size, dim_values = _find_1d_dim(ncfile, target_dim)
    var_dict = OrderedDict()
    attr_dfs_list = []
    if fullpath:
        # use fullpaths if recursing, otherwise do not
        fullpath = not top_vars_only
    with smart_nc(ncfile) as nh:
        _ncdf_to_df_internal(nh, dim_name=target_dim, dim_size=dim_size, var_dict=var_dict, att_dfs=attr_dfs_list,
                             top_vars_only=top_vars_only, unmatched_vars=unmatched_vars.lower(),
                             no_leading_slash=no_leading_slash, fullpath=fullpath, auto_time=convert_time, time_vars=time_vars)

    var_df = pd.DataFrame(var_dict, index=dim_values)
    attr_df = pd.concat(attr_dfs_list, axis=1, sort=True)
    if ret_attrs:
        return var_df, attr_df
    else:
        return var_df


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
                dim_values = get_nctime(nh.variables[dim_name])
            except (ValueError, AttributeError):
                # Not a datetime axis or missing units, can't convert to datetime
                dim_values = tmp_dim_values
        else:
            dim_values = pd.RangeIndex(dim_size)

    return dim_name, dim_size, dim_values


def _ncdf_to_df_internal(nch, dim_name, dim_size, var_dict, att_dfs, top_vars_only, unmatched_vars, no_leading_slash, fullpath, auto_time, time_vars):
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

        # Read the attributes and put them into their own dataframe
        this_att_dict = {name: [variable.getncattr(name)] for name in variable.ncattrs()}
        this_att_df = pd.DataFrame(this_att_dict, index=[full_varname]).T

        if varname in time_vars or ('calendar' in this_att_dict and auto_time):
            var_array = get_nctime(variable)
        elif np.issubdtype(variable.dtype, np.floating):
            var_array = variable[:].filled(np.nan)
        else:
            var_array = variable[:].data
        var_dict[full_varname] = var_array
        att_dfs.append(this_att_df)

    if top_vars_only:
        return

    for group in nch.groups.values():
        _ncdf_to_df_internal(group, dim_name=dim_name, dim_size=dim_size, var_dict=var_dict, att_dfs=att_dfs,
                             top_vars_only=top_vars_only, unmatched_vars=unmatched_vars, fullpath=fullpath,
                             no_leading_slash=no_leading_slash, auto_time=auto_time, time_vars=time_vars)


def get_nctime(ncvar, fill_action='nothing'):
    """Read a netCDF time variable and convert it to a Pandas DatetimeIndex

    Parameters
    ----------
    ncvar : netCDF4.Variable
        The time *variable* in the netCDF dataset. Note this is not the array, so for a dataset `ds`
        this would be e.g. `ds['time']`, and not `ds['time'][:]`.

    fill_action : str, int, or float
        How to handle fill values in the time variable. The default value is `"nothing"`, which will
        do nothing and attempt to interpret them as all other values in the array. This may result
        in an overflow error if the fill values are large. Passing `"replace"` here will instead
        replace them with the base date for the variable (often 1 Jan 1970). Alternatively, pass
        an integer or float to use instead as the fill value (it will still be reinterpreted as a
        date).

    Returns
    -------
    pandas.DatetimeIndex
        The netCDF times as a DatetimeIndex.
    """
    if fill_action == 'nothing':
        vardat = ncvar[:].data
    elif fill_action == 'replace':
        vardat = ncvar[:].filled(0.0)
    elif isinstance(fill_action, (int, float)):
        vardat = ncvar[:].filled(fill_action)
    else:
        raise NotImplementedError('Unknown fill_action "{}"'.format(fill_action))
    cf_time = ncdf.num2date(vardat, ncvar.units)
    return pd.DatetimeIndex(cf_time.astype('datetime64[s]'))


def make_ncdim_helper(nc_handle, dim_name, dim_var, unlimited=False, **attrs):
    """Create a netCDF dimension and its associated variable simultaneously

    Typically in a netCDF file, each dimension should have a variable with the same name that defines the coordinates
    for that dimension. This function streamlines the process of creating a dimension with its associated variable.

    Parameters
    ----------
    nc_handle : :class:`netCDF4.Dataset`
        the handle to a netCDF file open for writing, returned by :class:`netCDF4.Dataset`

    dim_name : str
        the name to give both the dimension and its associated variable

    dim_var : :class:`numpy.ndarray`
        the variable data to use when defining the dimension's coordinates. The dimensions length will be set
        by the size of this vector. This must be a 1D numpy array or comparable type.

    unlimited : bool
        Whether this dimension should be an unlimited (i.e. record) dimension in the netCDF file.

    attrs
        keyword-value pairs defining attribute to attach to the dimension variable.

    Returns
    -------
    :class:`netCDF4.Dimension`
        the dimension object.

    See Also
    --------
    make_nctimedim_helper: for making time dimensions where the data needs converted from a datetime to an CF-compliant time variable

    Examples
    --------

    Create latitude and longitude dimensions with "units" and "long_name" attributes:

    >>> lon = np.arange(-180.0, 181.0, 5.0)
    >>> lat = np.arange(-90.0, 91.0, 2.5)
    >>> with netCDF4.Dataset('demo.nc', 'w') as nch:
    >>>     make_ncdim_helper(nch, 'longitude', lon, units='degrees_east',  long_name='longitude')
    >>>     make_ncdim_helper(nch, 'latitude',  lat, units='degrees_north', long_name='latitude')
    """
    if np.ndim(dim_var) != 1:
        raise ValueError('Dimension variables are expected to be 1D')
    dim = nc_handle.createDimension(dim_name, np.size(dim_var) if not unlimited else None)
    var = nc_handle.createVariable(dim_name, dim_var.dtype, dimensions=(dim_name,))
    var[:] = dim_var
    var.setncatts(attrs)
    return dim


def make_nctimedim_attrs(base_date=dt.datetime(1970, 1, 1), time_units='seconds', calendar='gregorian',
                         base_date_nc_time=True):
    allowed_time_units = ('seconds', 'minutes', 'hours', 'days')
    if time_units not in allowed_time_units:
        raise ValueError('time_units must be one of: {}'.format(', '.join(allowed_time_units)))

    units_str = '{} since {}'.format(time_units, base_date.strftime('%Y-%m-%d %H:%M:%S'))
    if base_date_nc_time:
        base_date = cftime.date2num(base_date, units_str, calendar=calendar)
    return {'units': units_str, 'calendar': calendar, 'base_date': base_date}


def make_nctime(timedata, base_date=dt.datetime(1970, 1, 1), time_units='seconds', calendar='gregorian',
                base_date_nc_time=True):
    """Make a CF-compliant time array

    CF conventions expect that time data is given as time since a given date
    (http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html#time-coordinate), for example,
    1 Jan 1971 could be given as 365 days since 1 Jan 1970, or 8760 hours since 1 Jan 1970, etc. This function will
    convert most typical Python datetime representations into an array of CF-style values and return the necessary
    netCDF attributes to include as metadata.

    Parameters
    ----------
    timedata
        A sequence of datetime values. Currently, supported types are:

            * a list of :class:`datetime.datetime` objects
            * a numpy array convertible to the `datetime64[s]` type, or
            * a sequence of Pandas Timestamps (including a :class:`~pandas.DatetimeIndex`)

        Note that only 1D sequences have been tested. 2D and higher sequences may not work.

    base_date : class:`datetime.datetime`
        The date that will be used as the reference, e.g. in "days since 1970-01-01" it would be 1 Jan 1970.

    time_units : str
        What unit to use as the counter, e.g. in "days since 1970-01-01" it would be "days". Allowed values are
        "seconds", "minutes", "hours", and "days". "months" and "years" are NOT allowed because they defined in CF
        conventions in ways that are easy to use improperly.

    calendar : str
        What calendar to use when computing the number of days/hours/minutes/seconds between two times. For modern
        times, the default "gregorian" is recommended unless you have a specific reason not to use it.

    base_date_nc_time : bool
        The default behavior is to return the base date in the attribute dictionary in CF-type units. Set this keyword
        to `False` to disable that behavior.

    Returns
    -------
    :class:`numpy.ndarray`
        The input `timedata` converted to CF-compliant values.
    dict
        A dictionary with the "units", "calendar" and "base_date" attributes that should be assigned to the netCDF
        variable that will store the time data. These attributes are a necessary part of the CF convention for time
        variables.

    See Also
    --------

    make_nctimedim_helper: for creating time dimensions in netCDF files

    Examples
    --------

    Convert a list of Python datetimes to a CF-compliant array. Note that the default unit of time is "seconds since
    1970-01-01":

    >>> from datetime import datetime as dtime
    >>> timevec, timeattrs = make_nctime([dtime(1970,1,1), dtime(1970,5,1), dtime(1970,9,1), dtime(1971,1,1)])
    >>> timevec
    array([       0., 10368000., 20995200., 31536000.])
    >>> timeattrs
     {'units': 'seconds since 1970-01-01 00:00:00',
      'calendar': 'gregorian',
      'base_date': 0.0}

    Convert a Pandas DatetimeIndex, using hours as the unit of time:

    >>> import pandas as pd
    >>> dt_index = pd.date_range('1970-01-01', '1971-01-01', freq='4MS')
    >>> timevec, timeattrs = make_nctime(dt_index, time_units='days')
    >>> timevec
    array([  0., 120., 243., 365.])
    >>> timeattrs
    {'units': 'days since 1970-01-01 00:00:00',
     'calendar': 'gregorian',
     'base_date': 0.0}
    """
    time_info_dict = make_nctimedim_attrs(base_date=base_date, time_units=time_units, calendar=calendar,
                                          base_date_nc_time=base_date_nc_time)
    units_str = time_info_dict['units']
    calendar = time_info_dict['calendar']
    # date2num requires that the dates be given as basic datetimes. We'll handle converting Pandas timestamps, either
    # as a series or datetime index, but other types will need handled by the user.
    try:
        date_arr = ncdf.date2num(timedata, units_str, calendar=calendar)
    except (TypeError,AttributeError):
        # AttributeError required for some versions of netCDF4 and numpy where date2num tries to access "year" on a
        # numpy datetime64, which doesn't have that attribute.
        if isinstance(timedata, np.ndarray):
            dim_var = timedata.astype('datetime64[s]').tolist()
        else:
            dim_var = [d.to_pydatetime() for d in timedata]
        date_arr = ncdf.date2num(dim_var, units_str, calendar=calendar)

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
    date_arr, time_info_dict = make_nctime(var_data, base_date=base_date, time_units=time_units, calendar=calendar, base_date_nc_time=True)
    attrs = attrs.copy()
    attrs.update(time_info_dict)
    return make_ncvar_helper(nc_handle, var_name, date_arr, dims, make_cf_time_auto=False, **attrs)


def make_ncvar_helper(nc_handle, var_name, var_data, dims, make_cf_time_auto=True, **attrs):
    """Create a netCDF variable and store the data for it simultaneously.

    This function combines call to :func:`netCDF4.createVariable` and assigning the variable's data and attributes.
    
    Parameters
    ----------

    nc_handle : :class:`netCDF4.Dataset`
        the handle to a netCDF file open for writing, returned by :class:`netCDF4.Dataset`

    var_name : str
        the name to give the variable

    var_data : :class:`numpy.ndarray`
        the array to store in the netCDF variable.

    dims : list(:class:`netCDF4.Dimensions` or str)
        the dimensions to associate with this variable. Must be a collection of either dimension names or dimension
        instances. Both types may be mixed. This works well with :func:`make_ncdim_helper`, since that returns the
        dimension instances.

    make_cf_time_auto : bool
        when this is `True`, then any array with a datetime64 data type will be converted to a CF-compliant time
        variable automatically. If ``False``, no conversion will be attempted, which may result in a TypeError.

    attrs
        keyword-value pairs defining attribute to attach to the dimension variable.

    Returns
    -------
    :class:`netCDF4.Variable`
        the variable object

    Examples
    --------

    Create a variable for sea surface temperature that has dimensions of latitude and longitude, assuming those
    dimensions already exist:

    >>> import netCDF4, numpy as np
    >>> sst = np.random.rand(91, 180)  # assuming 2 degree resolution in both dimensions
    >>> with netCDF4.Dataset('demo.nc', 'a') as nch:  # use append as the mode if the dimensions exist in the file
    >>>     make_ncvar_helper(nch, 'sst', sst, dims=['latitude', 'longitude'], units='K',
    >>>                       long_name='sea surface temperature')

    If you make the dimensions at the same time, it is convenient to use the dimension objects to specify the dimensions
    in the call to make_ncvar_helper:

    >>> import netCDF4, numpy as np
    >>> lat = np.arange(-90., 91., 2)
    >>> lon = np.arange(-180., 180., 2)
    >>> sst = np.random.rand(lat.size, lon.size)
    >>> with netCDF4.Dataset('demo.nc', 'w') as nch:
    >>>     latdim = ncio.make_ncdim_helper(nch, 'lat', lat)
    >>>     londim = ncio.make_ncdim_helper(nch, 'lon', lon)
    >>>     make_ncvar_helper(nch, 'sst', sst, dims=[latdim, londim], units='K', long_name='sea surface temperature')
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
    """Create an SHA1 hash of a file.

    Parameters
    ----------
    dependent_file : str
        the path to the file to hash.

    Returns
    -------
    str
        the SHA1 hash
        
    Examples
    --------
    >>> make_dependent_file_hash('demo.nc')                                                                                                                                                
    'e65e1a1344aa1dc67702703a28966ea0b5294798'
    """
    hashobj = sha1()
    with open(dependent_file, 'rb') as fobj:
        block = fobj.read(4096)
        while block:
            hashobj.update(block)
            block = fobj.read(4096)

    return hashobj.hexdigest()


def add_dependent_file_hash(nc_handle, hash_att_name, dependent_file):
    """Add an SHA1 hash of another file as an attribute to a netCDF file.

    This is intended to create an attribute that list the SHA1 hash of a file that the netCDF file being created
    depends on.

    nc_handle : :class:`netCDF4.Dataset`
        a handle to the netCDF4 dataset to add the attribute to.

    hash_att_name : str
        the name to give the attribute that will store the hash. It is recommended to include the substring "sha1" so
        that it is clear what hash function was used.

    dependent_file : str
        the path to the file to generate the hash of.

    Examples
    --------

    Add a hash for a file containing satellite data that went into the netCDF file "demo.nc":

    >>> import netCDF4
    >>> with netCDF4.Dataset('demo.nc', 'a') as nch:
    >>>     add_dependent_file_hash(nch, 'sat_data_sha1', 'satellite_data.h5')
    """
    hash_hex = make_dependent_file_hash(dependent_file)
    nc_handle.setncattr(hash_att_name, hash_hex)


class NcWrapper(object):
    """Provides an h5py-like interface to netCDF4 files

    In Earth sciences, we often work with both HDF5 and netCDF files. In Python, the former is usually read with the
    h5py package and the latter with netCDF4 or xarray packages. However, h5py and netCDF4 provide different interfaces
    to their respective file types, which means writing code that is agnostic about which file type it is reading is
    quite challenging. This class serves as a drop in replacement for :class:`netCDF4.Dataset` that provides an
    interface similar to h5py for netCDF files.

    Parameters
    ----------
    dataset : str
        The path to the netCDF file to open

    no_masked : bool
        If `True` (default) then when returning data from a variable, masked values are replaced with NaNs and the
        data is returned as a regular :class:`numpy.ndarray` rather than a masked array. If `False`, masked arrays
        are returned.

    Warnings
    --------

    Currently the way `no_masked` is implemented means it will only work retrieving variables with a float datatype. If
    you expect to read other datatypes, then you will currently have to set `no_masked` to `False` and handle the masked
    arrays yourself.

    Examples
    --------

    Assume that "demo.nc" contains the "lat" and "lon" variables in the top level, and the "sst" variable in the "ocean"
    group. First list the available variables and groups:

    >>> f = NcWrapper('demo.nc')
    >>> f.keys()
    ('ocean', 'lat', 'lon')

    Access the "lat" variable object (not the data):

    >>> f['lat']
    <class 'netCDF4._netCDF4.Variable'>
    float64 lat(lat)
    unlimited dimensions:
    current shape = (91,)
    filling on, default _FillValue of 9.969209968386869e+36 used

    Access the "ocean" group, then list the groups and variables it contains:

    >>> f['ocean']
    <class 'netCDF4._netCDF4.Group'>
    group /ocean:
        dimensions(sizes):
        variables(dimensions): float64 sst(lat,lon)
        groups:
    >>> f['ocean'].keys()
    ('sst',)

    Read in the sst variable:

    >>> f['ocean']['sst'][:]
    array([[288.79069806, 277.81187056, 291.48221822, ..., 297.99602979,
        283.23083654, 288.35196569],
       [274.94932278, 291.88013601, 289.1781155 , ..., 297.54587896,
        285.10436042, 289.01523051],
       [297.99372506, 285.2359472 , 294.12834898, ..., 291.61429846,
        297.50201179, 292.88045642],
       ...,
       [285.39244074, 293.42634441, 275.57705086, ..., 294.31560397,
        297.301303  , 289.94240014],
       [287.1197515 , 291.43977814, 280.08993929, ..., 276.61085751,
        283.6424262 , 278.17390702],
       [280.50048818, 290.92144857, 284.18603123, ..., 285.939394  ,
        277.45671306, 276.28845933]])

    Access the "units" attribute of the "sst" variable. Note that this is not yet been homogenized with the h5py-style
    interface:

    >>> f['ocean']['sst'].units
    'K'

    Any call to an attribute for a netCDF dataset wrapped by this class that the class does not implement is passed
    through to the underlying dataset:

    >>> f['ocean']['sst'].ncattrs()
    ['units', 'long_name']

    Close the dataset as normal:

    >>> f.close()

    This wrapper also works in `with` blocks:

    >>> with ncio.NcWrapper('/home/josh/scratch/demo.nc') as f:
    >>>     print(f['ocean']['sst'].ncattrs())
    ['units', 'long_name']
    """
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
        """Close the open netCDF file
        """
        if self._nc_handle is not None:
            self._nc_handle.close()
            self._nc_handle = None

    def keys(self):
        """List the available keys for this dataset - includes both groups and variables.

        Returns
        -------
        tuple
            the group and variable keys. Group keys are always listed first.
        """
        ds = self._nc_handle
        group_keys = list(ds.groups.keys())
        var_keys = list(ds.variables.keys())
        return tuple(group_keys) + tuple(var_keys)


def copy_nc_group(src, dst, recursive=True, exclude_dimensions=None, exclude_variables=None, exclude_groups=None, exclude_attributes=None):
    """Copy any group, including the root group, and (optionally) all of its descendants from one netCDF file into another

    Parameters
    ----------
    src : netCDF4.Group
        Group to copy

    dst : netCDF4.Group
        Group to copy into

    recursive : bool
        If ``True``, all child groups are recursively copied. If ``False``, then groups contained within
        ``src`` will be created in ``dst`` but left empty.

    exclude_dimensions : Sequence[str]
        A list of dimension names not to copy into ``dst``. If ``recursive`` is ``True``, this affects all
        subordinate groups as well.

    exclude_variables : Sequence[str]
        A list of variable names not to copy into ``dst``. If ``recursive`` is ``True``, this affects all
        subordinate groups as well.

    exclude_groups : Sequence[str]
        A list of group names not to copy into ``dst``. Groups named in this will not be copied, nor will any
        groups under the excluded group. If ``recursive`` is ``True``, this check is done within each group
        copied.

    exclude_attributes : Sequence[str]
        A list of attribute names not to copy into ``dst``. Only affects group level attributes; variable
        attributes are not affected. If ``recursive`` is ``True``, this affects all subordinate groups as well.

    Limitations
    -----------
    * Due to their complexity, character arrays can only be two dimensional and the second dimension *must* be 
      the length of the individual strings (that is, each row in the character array must be one string).
    * Variable length strings are supported, but must be one dimensions (i.e. a vector of strings).
    * Other variable length types are not specifically excluded, but have not been tested and are not
      guaranteed to work.
    """
    exclude_dimensions = set() if exclude_dimensions is None else set(exclude_dimensions)
    exclude_variables = set() if exclude_variables is None else set(exclude_variables)
    exclude_groups = set() if exclude_groups is None else set(exclude_groups)
    exclude_attributes = set() if exclude_attributes is None else set(exclude_attributes)

    grp_attrs = {k: v for k, v in src.__dict__.items() if k not in exclude_attributes}
    dst.setncatts(grp_attrs)

    for dimname, dimension in src.dimensions.items():
        if dimname in exclude_dimensions:
            continue
        dimlen = len(dimension) if not dimension.isunlimited() else None
        dst.createDimension(dimname, dimlen)

    for varname in src.variables.keys():
        if varname in exclude_variables:
            continue

        copy_nc_var(src, dst, varname)
        
        
    for grpname, group in src.groups.items():
        if grpname in exclude_groups:
            continue

        newgrp = dst.createGroup(grpname)
        if recursive:
            copy_nc_group(group, newgrp, recursive=recursive, exclude_dimensions=exclude_dimensions, exclude_variables=exclude_variables,
                          exclude_groups=exclude_groups, exclude_attributes=exclude_attributes)


def copy_nc_var(src, dst, varname, vardata=None, convert_str='no'):
    """Copy a single netCDF variable from one group into another.

    Parameters
    ----------
    src : netCDF4.Group
        Group to copy the variable from

    dst : netCDF4.Group
        Group to copy the variable into

    varname : str
        Name of the variable to copy

    vardata : Optional[numpy.ndarray]
        If given, this is written as the data in ``dst`` rather than the original 
        data in ``src``. Useful for subsetting the data.

    convert_str : str
        If "no" (default), then variable length strings and character arrays are both left as is.
        If "stoc" (string to character), variable length strings are converted to character arrays.
        ("ctos" is planned but not yet implemented.)

    Limitations
    -----------
    * Due to their complexity, character arrays can only be two dimensional and the second dimension *must* be 
      the length of the individual strings (that is, each row in the character array must be one string).
    * Variable length strings are supported, but must be one dimensions (i.e. a vector of strings).
    * Other variable length types are not specifically excluded, but have not been tested and are not
      guaranteed to work.
    """
    def get_string(el):
        """NetCDF arrays are awfully inconsistent about whether iterating
        produces a string or an array with one string. This will try to handle
        the case of extracting that string
        """
        try:
            return el.item()
        except AttributeError:
            return el

    variable = src[varname]
    if vardata is None:
        vardata = variable[tuple()]

    if variable.dtype is str and convert_str == 'stoc':
        strlen = max(len(get_string(s)) for s in vardata)
        # For WHATEVER reason, `astype('S')` truncated some strings, so we have to calculate
        # the length manually.
        vardata = ncdf.stringtochar(np.array(vardata).astype(f'S{strlen}'))
        strdim = f'c{strlen}'
        if strdim not in dst.dimensions:
            dst.createDimension(strdim, strlen)
        newvar = dst.createVariable(varname, vardata.dtype, (list(variable.dimensions)[0], strdim))
    else:
        newvar = dst.createVariable(varname, variable.datatype, variable.dimensions)
        newvar.setncatts(variable.__dict__)

    if variable.dtype is str and convert_str != 'stoc':
        # Handle vline strings. I was able to do newvar[:] = variable[:], but passing 
        # an array in as vardata didn't work well for some reason. So we loop here as 
        # well as with regular character arrays.
        #
        # But if we converted to a character array, then it should be able to just copy
        # the whole thing at once in the else block.
        if variable.ndim != 1:
            raise NotImplementedError('Cannot copy vlen strings with >1 dimension')
        for i in range(vardata.shape[0]):
            newvar[i] = get_string(vardata[i])
        
    elif re.match(r'[\|<][SU]\d+', str(variable.dtype)):
        # Handle 2D char arrays. It is important that this comes after the attributes are set
        # so that the characters are interpreted correctly, and we have to do one at a time
        # or the slices don't match in size
        for i in range(vardata.shape[0]):
            newvar[i] = get_string(vardata[i])

    else:
        # Numeric types should just copy directly
        newvar[tuple()] = vardata

        
def ncdump(file_or_handle, var_att=None, list_atts=False, list_att_values=False, grep=None, grep_re=None, _indent_level=0):
    """Print a tree visualization of a netCDF file

    Prints out names of variables in a netCDF file. If groups a present, they are printed with variables
    contained in them shown indented by one level. Attributes (with or without values) can also be printed.
    In the tree, lines starting with "*" are groups, "-" are variables, and "+" are attributes.
    
    Parameters
    ----------
    file_or_handle : str, netCDF4.Dataset, or netCDF4.Group
        The path to or open dataset/group handle of a netCDF file to print.

    var_att : Optional[str]
        The name of an attribute to print the value of after each variable name. Useful for attributes
        like "long_name" or "description" that provide more information about what a particular variable
        is. If this argument is not given, or the attribute isn't present on a variable, nothing is printed
        after the variable name.

    list_atts : bool
        If ``True``, then prints out the names of attributes under variables (but not their values).
        This keyword is ignored if ``list_att_values`` is ``True``.

    list_att_values : bool
        If ``True``, then prints out the names and values of attributes under variables.

    grep : Optional[str]
        If given, will only print variables whose names include this string (case insensitive).

    grep_re: Optional[str | re.Pattern]
        If given, will only print variables whose names return a match to the given regular expression
        (called with ``search``). This can either be a string which is a valid pattern or a compiled
        :class:`re.Pattern` (i.e. the return value from :func:`re.compile`). The latter is the only way
        to include flags.
    """
    if grep:
        grep = grep.lower()
    if isinstance(grep_re, str):
        grep_re = re.compile(grep_re)
        
    indent = '  ' * _indent_level
    with smart_nc(file_or_handle) as ds:
        for grpname, group in ds.groups.items():
            print('{indent}* {name}:'.format(indent=indent, name=grpname))
            ncdump(group, var_att=var_att, list_atts=list_atts, list_att_values=list_att_values, grep=grep, grep_re=grep_re, _indent_level=_indent_level+1)
        for varname, var in ds.variables.items():
            if grep and grep not in varname.lower():
                continue
            if grep_re and not grep_re.search(varname):
                continue

            dims = ['{dname} [{dlen}]'.format(dname=d, dlen=_find_dim_in_group_or_parents(ds, d).size) for d in var.dimensions]
            dims = ', '.join(dims)
            if var_att is not None and var_att in var.ncattrs():
                print('{indent}- {name} ({dims}): {att}'.format(indent=indent, name=varname, dims=dims, att=var.getncattr(var_att)))
            else:
                print('{indent}- {name} ({dims})'.format(indent=indent, name=varname, dims=dims))

            if list_atts or list_att_values:
                att_indent = '  ' * (_indent_level+1)
                for att in var.ncattrs():
                    if list_att_values:
                        print('{indent}+ {att} = {val}'.format(indent=att_indent, att=att, val=var.getncattr(att)))
                    else:
                        print('{indent}+ {att}'.format(indent=att_indent, att=att))


def _find_dim_in_group_or_parents(grp, dimname):
    if dimname in grp.dimensions:
        return grp.dimensions[dimname]

    while grp.parent is not None:
        grp = grp.parent
        if dimname in grp.dimensions:
            return grp.dimensions[dimname]

    return KeyError('No dimension named "{name}" found in this group or any parent'.format(dimname))


def read_opendap_url(url: str, variables: dict, date: Optional[dt.datetime] = None, host='urs.earthdata.nasa.gov', keep_as_xarray: bool = False):
    """Read data from an OpenDAP URL

    Parameters
    ----------
    url
        The OpenDAP URL to download from. Date elements in the URL can be replaced with ``{date:FMT}`` where ``FMT`` is a datetime 
        formatting string. If you pass a URL with this in it, you must also pass the ``date`` argument.

    variables
        A list or dictionary listing variables to read from the OpenDAP repo. If a list, then it must be variable names in the root
        group of the OpenDAP repo, and the output dictionary will use those names as keys. If a dictionary, then the keys will be the
        keys in the output and the values are the variables in the OpenDAP repo.

        Currently I've not worked out whether OpenDAP repos can have netCDF-like groups, or how to access them if they do.

    date
        The date to download; may be omitted if that is already included in the URL (i.e. the URL has no ``{date}`` format elements.)

    host
        The host name in your ~/.netrc file to use credentials from to login.

    Returns
    -------
    data
        A dictionary with the variables as numpy arrays in the values. 
    """

    if date is not None:
        print(f'Downloading data for {date:%Y-%m-%d}')
        url = url.format(date=date)
    else:
        print(f'Downloading data from {url}')

    if not isinstance(variables, dict):
        variables = {k: k for k in variables}
        
    user, _, password = netrc().hosts[host]
    session = setup_session(user, password, check_url=url)
    pydap_ds = open_url(url, session=session)
    
    xr_store = xr.backends.PydapDataStore(pydap_ds)
    with xr.open_dataset(xr_store) as ds:
        data = dict()
        for key, variable in variables.items():
            if keep_as_xarray:
                data[key] = ds[variable]
            else:
                data[key] = ds[variable].data

    return data
