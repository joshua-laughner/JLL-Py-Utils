"""
This modules contains functions that are helpful additions to Pandas dataframes.

It also monkey-patches several functions to the DataFrame and Series classes to make them accessible in the expected
object-oriented fashion. This includes:

    * `DataFrame.to_netcdf`: :func:`jllutils.subutils.ncdf.dataframe_to_ncdf`
    * `DataFrame.interpolate_to` and `Series.interpolate_to`: :func:`interpolate_series_to`

When using the monkey-patched methods, the first argument of the function is automatically passed. That is,
interpolating the dataframe `df` to the values `vals` can be done with::

    interpolate_series_to(df, vals)

or::

    df.interpolate_to(vals)

The two are completely equivalent.
"""

import pandas as pd
from .subutils import ncdf


def interpolate_series_to(series, values, method='index', axis=0, limit_direction='both', limit_area=None,
                          downcast=None, **kwargs):
    """Interpolate a Pandas series or dataframe to specified index values

    This function is similar to the `"index"` method of `interpolate` on Pandas series/dataframes in that it does
    interpolation using the index values as the :math:`x`-coordinate, however this allows you to specify the values
    to interpolate to, rather than just interpolating to fill in NaN values in the series/dataframe.

    Parameters
    ----------
    series : :class:`pandas.Series` or :class:`pandas.DataFrame`
        The series or dataframe to intepolate
    values
        The values to interpolate to. Must be compatible with the index. The returned series or dataframe will have
        these as the new index.
    method
        How to do the interpolate. Any method accepted by :meth:`pandas.Series.interpolate` is valid.
    axis
        Which axis to interpolate along. May be 0 ("index") or 1 ("columns")
    limit_direction
        Controls which direction NaNs are filled. May be "both", "forward", or "backward". If "forward" or "backward",
        then target values before/after the first/last value in the index, respectively, will not be filled.
    limit_area
        Controls whether only interpolation ("inside"), only extrapolation ("outside") or both (`None`) are done.
    downcast
        Downcast types if possible.
    kwargs
        Additional keyword arguments to pass to :meth:`pandas.Series.interpolate` and the interpolating function.

    Returns
    -------
    :class:`pandas.Series` or :class:`pandas.DataFrame`
        The input series of dataframe interpolated to the requested values

    Warnings
    --------

    Duplicate values in the original series/dataframe may cause issues. If you are getting errors about operations
    failing on a duplicate axis, or if the output contains more values than expected, check your original for
    duplicate indices.

    Notes
    -----

    This is implemented by first reindexing the input series to an index containing a union of its original values and
    all unique elements of `values`, then interpolating, and finally extracting the elements indexed by `values`. This
    is *not* done in place so large dataframes being interpolated to many points may use substantial memory.

    Examples
    --------

    >>> ser = pd.Series([1, 2, 3], index=[0, 10, 20])
    >>> interpolate_series_to(ser, range(1,20,3))
    1     1.1
    4     1.4
    7     1.7
    10    2.0
    13    2.3
    16    2.6
    19    2.9
    dtype: float64

    By default, constant value extrapolation is allowed:

    >>> interpolate_series_to(ser, range(-10, 31, 10))
    -10    1.0
     0     1.0
     10    2.0
     20    3.0
     30    3.0
    dtype: float64

    Restrict to interpolation only:

    >>> interpolate_series_to(ser, range(-10, 31, 10), limit_area='inside')
    -10    NaN
     0     1.0
     10    2.0
     20    3.0
     30    NaN
    dtype: float64

    Note that the monkey-patched interpolate_to method is an alias to this:

    >>> ser.interpolate_to(range(5))
    0    1.0
    1    1.1
    2    1.2
    3    1.3
    4    1.4
    dtype: float64
    """
    # input values must be unique for the index, otherwise when we do the final call to loc we can get
    # duplicate rows. Try to use a `unique()` method on the values object if present as it often preserves
    # types better (e.g. for Pandas DatetimeIndex objects), but if `values` does not have a `unique()`
    # method, then fall back on `pd.unique()`.
    try:
        uvalues = values.unique()
    except AttributeError:
        uvalues = pd.unique(values)

    new_index = series.index.union(uvalues)
    new_df = series.reindex(new_index).interpolate(method=method, axis=axis, limit=limit, limit_direction=limit_direction,
                                                   limit_area=limit_area, downcast=downcast, inplace=False, **kwargs)
    return new_df.loc[values]


# Monkey-patch these into pandas
pd.DataFrame.to_netcdf = ncdf.dataframe_to_ncdf
pd.DataFrame.interpolate_to = interpolate_series_to
pd.Series.interpolate_to = interpolate_series_to
