import pandas as pd
from .subutils import ncdf


def interpolate_series_to(series, values, method='linear', axis=0, limit=None, limit_direction='forward',
                          limit_area=None, downcast=None, **kwargs):
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
