try:
    import ephem
except ImportError:
    print('pyephem not installed, solar noon functions will not work')
import matplotlib.pyplot as plt
import netCDF4 as ncdf
import numpy as np
import pandas as pd

def shapefile_to_csv(shpfile, csvfile, attrs=tuple(), sort_attr=None):
    """Convert a GIS shapefile (``.shp``) to a .csv file.

    The output .csv file will have at least the columns "lon", "lat", and "polygon_index". "lon" and "lat" have the coordinates
    of the records, while "polygon_index" is a 0-based index that increments by 1 for each record or (in the case of MultiPolygon
    geometries) each sub-polygon within the record. Thus, when reading the .csv, plotting the lats/lons associated with each index
    separately will correctly avoid connecting separate polygons with lines.

    If ``attrs`` is specified, then each attribute will be an additional column in the .csv.

    Parameters
    ----------
    shpfile : pathlike
        Path to the ``.shp`` file to convert, note that it may require some additional files (e.g. ``.shx``) are present in the same directory.

    csvfile : pathlike
        Path to write the .csv file as. Will be overwritten if already exists.

    attrs : Sequence[str]
        A sequence of attributes on the individual records in the shapefile that should be copied into the .csv file. For example,
        records of US states often contain a "STUSPS" attribute which gives the abbreviation for that state.

    sort_attr : Optional[str]
        The name of an attribute to sort the records by before writing them.

    Notes
    -----
    This function is intended to write boundaries of shapes, so currently the following geometry types are handled:

    * :class:`shapely.geometry.linestring.LineString` - the coordinates are written as lon and lat.
    * :class:`shapely.geometry.polygon.Polygon` - the boundary coordinates are written as lon and lat.
    * :class:`shapely.geometry.multipolygon.MultiPolygon` - the boundary coordinates of each polygon under ``boundary``
      are written as lon and lat; each child polygon is given a unique polygon_index.

    Any other :mod:`shapely` geometry types will raise a :class:`NotImplementedError`.
    """
    try:
        import cartopy.io.shapereader as shpreader
    except ImportError:
        raise ImportError('cartopy is a soft dependency for JLLUtils; please install it to use this function')

    reader = shpreader.Reader(shpfile)
    records = reader.records()
    if sort_attr is not None:
        records = sorted(records, key=lambda r: r.attributes[sort_attr])

    with open(csvfile, 'w') as f:
        f.write('lon,lat,polygon_index')
        if attrs:
            f.write(',{}\n'.format(','.join(attrs)))
        else:
            f.write('\n')

        curr_poly_index = 0
        for rec in records:
            geo = rec.geometry
            lon, lat, poly_index, curr_poly_index = _get_lat_lon_poly(geo, curr_poly_index)

            # Prep the attributes ones, since they'll be the same for every row in this geometry
            if attrs:
                attr_string = ','.join(rec.attributes[a] for a in attrs)
                attr_string = ',{}\n'.format(attr_string)
            else:
                attr_string = '\n'

            for x, y, i in zip(lon, lat, poly_index):
                f.write('{},{},{}{}'.format(x, y, i, attr_string))


def shapefile_to_ncdf(shpfile, ncfile, attrs=tuple(), sort_attr=None):
    """Convert a GIS shapefile (``.shp``) to a netCDF 4 file.

    The output .nc4 file will have at least the variables "lon", "lat", which are the coordinates of the records. These will be
    variable length data types, so each record or (in the case of MultiPolygon geometries) each sub-polygon within the record will
    be its own row (and each row can be a different length). Thus, when reading the .nc4, plotting each row of lat/lon separately
    will correctly avoid connecting separate polygons with lines.

    If ``attrs`` is specified, then each attribute will be an additional variable in the .nc4.

    Parameters
    ----------
    shpfile : pathlike
        Path to the ``.shp`` file to convert, note that it may require some additional files (e.g. ``.shx``) are present in the same directory.

    ncfile : pathlike
        Path to write the .nc4 file as. Will be overwritten if already exists.

    attrs : Sequence[str]
        A sequence of attributes on the individual records in the shapefile that should be copied into the .nc4 file. For example,
        records of US states often contain a "STUSPS" attribute which gives the abbreviation for that state.

    sort_attr : Optional[str]
        The name of an attribute to sort the records by before writing them.

    Notes
    -----
    This function is intended to write boundaries of shapes, so currently the following geometry types are handled:

    * :class:`shapely.geometry.linestring.LineString` - the coordinates are written as lon and lat.
    * :class:`shapely.geometry.polygon.Polygon` - the boundary coordinates are written as lon and lat.
    * :class:`shapely.geometry.multipolygon.MultiPolygon` - the boundary coordinates of each polygon under ``boundary``
      are written as lon and lat; each child polygon is given a unique polygon_index.

    Any other :mod:`shapely` geometry types will raise a :class:`NotImplementedError`.
    """
    try:
        import cartopy.io.shapereader as shpreader
    except ImportError:
        raise ImportError('cartopy is a soft dependency for JLLUtils; please install it to use this function')

    reader = shpreader.Reader(shpfile)
    records = reader.records()
    if sort_attr is not None:
        records = sorted(records, key=lambda r: r.attributes[sort_attr])

    lons = []
    lats = []
    attr_dict = {a: [] for a in attrs}

    for rec in records:
        # Don't need the polygon index here, so pass in a dummy value for it
        lon, lat, _, _ = _get_lat_lon_poly(rec.geometry, 0)
        lons.append(np.array(lon))
        lats.append(np.array(lat))
        for a in attrs:
            att_val = rec.attributes[a]
            attr_dict[a].append(att_val)

    lons = np.array(lons, dtype=object)
    lats = np.array(lats, dtype=object)
    for k, v in attr_dict.items():
        attr_dict[k] = np.array(v)

    with ncdf.Dataset(ncfile, 'w') as ds:
        ds.createDimension('polygon', np.size(lons))
        vlen_t = ds.createVLType(np.float64, "polygon_vlen")
        lon_var = ds.createVariable('lon', vlen_t, ['polygon'])
        lon_var[:] = lons
        lat_var = ds.createVariable('lat', vlen_t, ['polygon'])
        lat_var[:] = lats

        for k, v in attr_dict.items():
            attr_var = ds.createVariable(k, v.dtype, ['polygon'])
            attr_var[:] = v


def _get_lat_lon_poly(geo, curr_poly_index):
    try:
        import shapely
    except ImportError:
        raise ImportError('shapely is a soft dependency for JLLUtils; please install it to use this function')

    if isinstance(geo, shapely.geometry.linestring.LineString):
        lon, lat = geo.coords.xy
        poly_index = np.full(len(lon), curr_poly_index, dtype=np.int64)
        curr_poly_index += 1
    elif isinstance(geo, shapely.geometry.polygon.Polygon):
        lon, lat = geo.boundary.coords.xy
        poly_index = np.full(len(lon), curr_poly_index, dtype=np.int64)
        curr_poly_index += 1
    elif isinstance(geo, shapely.geometry.multipolygon.MultiPolygon):
        lon = []
        lat = []
        poly_index = []
        for poly in geo.boundary:
            this_lon, this_lat = poly.coords.xy
            this_pidx = np.full(len(this_lon), curr_poly_index)
            curr_poly_index += 1
            lon.append(this_lon)
            lat.append(this_lat)
            poly_index.append(this_pidx)

        lon = np.concatenate(lon)
        lat = np.concatenate(lat)
        poly_index = np.concatenate(poly_index)
    else:
        raise NotImplementedError('CSV conversion not implemented for geometry type {}'.format(type(geo)))

    return lon, lat, poly_index, curr_poly_index


def plot_shapes_from_csv(shpcsv, ax=None, filter_fxn=None, keep_extent=True, **style):
    """Plot geographic shapes from a .csv file created by :func:`shapefile_to_csv`

    shpcsv : pathlike
        Path to the .csv file.

    ax
        Axes to plot into; if not given, the current axes are used.

    filter_fxn : callable
        A function that, given the .csv dataframe as the sole argument, returns a
        logical or numerical index vector for the rows of the dataframe to plot.

    keep_extent : bool
        If ``True``, keeps the bounds that the map had when this function was called. Note that
        this does so by calling `set_extent`, so any future plots to these axis will not update
        the limits either.

    **style
        Keywords to pass to :func:`matplotlib.pyplot.plot`. Note that "color" is set to black by default.
    """
    if ax is None:
        ax = plt.gca()

    style.setdefault('color', 'k')

    shpdf = pd.read_csv(shpcsv)
    if filter_fxn is not None:
        xx = filter_fxn(shpdf)
        shpdf = shpdf.loc[xx]

    extent = ax.get_xlim() + ax.get_ylim()

    for _, feature in shpdf.groupby('polygon_index'):
        ax.plot(feature['lon'].to_numpy(), feature['lat'].to_numpy(), **style)

    if keep_extent:
        ax.set_extent(extent)


def plot_shapes_from_nc(shpnc, ax=None, keep_extent=True, **style):
    """Plot geographic shapes from a netCDF 4 file created by :func:`shapefile_to_ncdf`

    shpnc : pathlike
        Path to the netCDF 4 file.

    ax
        Axes to plot into; if not given, the current axes are used.

    keep_extent : bool
        If ``True``, keeps the bounds that the map had when this function was called. Note that
        this does so by calling `set_extent`, so any future plots to these axis will not update
        the limits either.

    **style
        Keywords to pass to :func:`matplotlib.pyplot.plot`. Note that "color" is set to black by default.
    """
    if ax is None:
        ax = plt.gca()

    style.setdefault('color', 'k')

    with ncdf.Dataset(shpnc) as ds:
        lon = ds['lon'][:]
        lat = ds['lat'][:]

    extent = ax.get_xlim() + ax.get_ylim()

    for x, y in zip(lon, lat):
        ax.plot(x, y, **style)

    if keep_extent:
        ax.set_extent(extent)


def great_circle_distance(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray, earth_radius: float = 6378.137) -> np.ndarray:
    """Compute the great circle distances between two sets of lat/lons, in kilometers by default

    Parameters
    ----------
    lon1, lat1
        Arrays of the starting coordinates.

    lon2, lat2
        Arrays of the ending coordinates.

    earth_radius
        Value to use for the radius of Earth in the distance calculation. The units of this value set the units of the output.

    Returns
    -------
    distances
        An array of distances between the points, in the same units as ``earth_radius``.

    Notes
    -----
    1. This uses Numpy array operations internally, so either/both of the lat/lon pairs may be scalar floats if you need just a single
       distance or multiple distances to a single point.
    2. This uses the Haversine formula, which is known to have rounding errors for antipodal points. 
    """
    if np.size(lon1) != np.size(lat1):
        raise TypeError('lon1 and lat1 are different sizes')
    if np.size(lon2) != np.size(lat2):
        raise TypeError('lon2 and lat2 are different sizes')

    lon1 = np.deg2rad(lon1)
    lat1 = np.deg2rad(lat1)
    lon2 = np.deg2rad(lon2)
    lat2 = np.deg2rad(lat2)
    dlon = np.abs(lon2 - lon1)
    dlat = np.abs(lat2 - lat1)

    inner = np.sin(dlat/2)**2 + (1 - np.sin(dlat/2)**2 - np.sin((lat1 + lat2)/2)**2) * np.sin(dlon/2)**2
    central_angle = 2 * np.arcsin(np.sqrt(inner))
    return central_angle * earth_radius


def format_lon(lon: float, fmt: str = '{lon:.2f} {ew}') -> str:
    """Format a longitude as a string with "E" or "W" correctly indicated

    Parameters
    ----------
    lon
        The longitude value to format. Must be a scalar.

    fmt
        Format string to use. The key ``lon`` will be replaced by the longitude value
        (modified as described in the "Returns" section) and the key ``ew`` will be replaced
        with the E/W indicator.

    Returns
    -------
    lonstr
        The longitude string, using ``fmt`` as the template. The longitude value will be
        constrained to the range 0 to 180 degrees; values less than 0 or greater than 180
        are converted to degrees west.
    """
    if lon < 0:
        ew = 'W'
        lon = -lon
    elif lon > 180:
        ew = 'W'
        lon -= 360
    else:
        ew = 'E'

    return fmt.format(lon=lon, ew=ew)


def format_lat(lat: float, fmt: str = '{lat:.2f} {ns}') -> str:
    """Format a latitude as a string with "N" or "S" correctly indicated.

    Parameters
    ----------
    lat
        The latitude value to format. Must be a scalar.

    fmt
        Format string to use. The key ``lat`` will be replaced by the latitude value
        (modified as described in the "Returns" section) and the key ``ns`` will be replaced
        with the N/S indicator.

    Returns
    -------
    latstr
        The latitude string, using ``fmt`` as the template. The latitude value will always
        be between 0 and 90, with values less than 90 originally considered as S.
    """
    if lat < 0:
        ns = 'S'
        lat = -lat
    else:
        ns = 'N'

    return fmt.format(lat=lat, ns=ns)


def get_solar_noon_for_observer(after, observer):
    """Return solar noon in UTC for the given observer's location

    Parameters
    ----------
    after: datetime-like
        A datetime-like object giving the UTC time after which to find solar noon for.

    observer: ephem.Observer
        An observer for the location to compute solar noon for.

    Returns
    -------
    solar_noon
        Solar noon as a :class:`datetime.datetime` in UTC.
    """
    sun = ephem.Sun()
    return observer.next_transit(sun, start=ephem.Date(after)).datetime()


def get_solar_noon_for_lat_lon(after, lat, lon):
    """Convenience method to compute solar noon for a given latitude and longitude.

    For repeated calculations of the same location, it is more efficient to use
    :func:`make_ephem_observer` to construct the observer once and call
    :func:`get_solar_noon_for_observer` instead.

    Parameters
    ----------
    after: datetime-like
        A datetime-like object giving the UTC time after which to find solar noon for.

    lat: float
        Latitude for the observer (south is negative).

    lon: float
        Longitude for the observer (west is negative).

    Returns
    -------
    solar_noon
        Solar noon as a :class:`datetime.datetime` in UTC.
    """
    observer = make_ephem_observer(lat, lon)
    return get_solar_noon_for_observer(after, observer)


def make_ephem_observer(lat, lon):
    """Create a pyephem Observer for a given latitude and longitude.

    Parameters
    ----------
    lat: float
        Latitude for the observer (south is negative).

    lon: float
        Longitude for the observer (west is negative).

    Returns
    -------
    observer: ephem.Observer
        Observer at the given latitude and longitude.
    """
    observer = ephem.Observer()
    observer.lat, observer.long = '{:.4f}'.format(lat), '{:.4f}'.format(lon)
    return observer
