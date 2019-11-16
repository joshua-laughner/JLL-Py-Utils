import pandas as pd
from .subutils import ncdf


def interpolate_series_to(series, values, method='linear', axis=0, limit=None, limit_direction='forward',
                          limit_area=None, downcast=None, **kwargs):
    new_index = series.index.union(values)
    new_df = series.reindex(new_index).interpolate(method=method, axis=axis, limit=limit, limit_direction=limit_direction,
                                                   limit_area=limit_area, downcast=downcast, inplace=False, **kwargs)
    return new_df.loc[values]


# Monkey-patch these into pandas
pd.DataFrame.to_netcdf = ncdf.dataframe_to_ncdf
pd.DataFrame.interpolate_to = interpolate_series_to
pd.Series.interpolate_to = interpolate_series_to
