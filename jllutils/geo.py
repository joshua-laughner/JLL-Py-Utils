from __future__ import print_function, absolute_import, division, unicode_literals
import numpy as np


class GridError(Exception):
    pass


class GridInputError(GridError):
    pass


class GridQueryError(GridError):
    pass


def equirect_grid(center_lon, center_lat, lon_res, lat_res, lon_lim, lat_lim, edges=False):
    lon_edge_vec = np.arange(lon_lim[0], lon_lim[1] + 1.0, lon_res)
    lat_edge_vec = np.arange(lat_lim[0], lat_lim[1] + 1.0, lat_res)

    if not np.all(np.in1d(lon_lim, lon_edge_vec)) or not np.all(np.in1d(lat_lim, lat_edge_vec)):
        raise GridInputError('Input resolution does not divide the domain evenly')

    lon_vec = (lon_edge_vec[:-1] + lon_edge_vec[1:])/2.0
    lat_vec = (lat_edge_vec[:-1] + lat_edge_vec[1:])/2.0

    if edges:
        lon_grid, lat_grid = np.meshgrid(lon_edge_vec, lat_edge_vec, indexing='ij')
        # Want lon_grid unstaggered in the second dimension, lat_grid in the first. Since they're just repeated
        # along those dimensions we can just cut off the lat row/column
        lon_grid = lon_grid[:, :-1]
        lat_grid = lat_grid[:-1, :]
        return lon_grid, lat_grid
    else:
        return np.meshgrid(lon_vec, lat_vec, indexing='ij')


class MapGrid(object):
    _allowed_proj = ('equirect',)
    _predef_domains = {'global': {'center': (0.0, 0.0), 'lon_lim': (-180.0, 180.0), 'lat_lim': (-90.0, 90.0)}}


    # Build a dictionary of methods to call to produce the grid for each projection. Each method must take center_lon,
    # center_lat, lon_res, lat_res, lon_lim, lat_lim and the keyword edges and return a lat/lon grid of center points
    # (if edges is false) or edge points (if true). Edge points are to be given as a 2D array staggered in the
    # appropriate dimension.
    _grid_methods = {'equirect': equirect_grid}

    @property
    def grid_lon_centers(self):
        lon_centers, _ = self.make_grid(edges=False)
        return lon_centers

    @property
    def grid_lat_centers(self):
        _, lat_centers = self.make_grid(edges=False)
        return lat_centers

    @property
    def grid_lon_edges(self):
        lon_edges, _ = self.make_grid(edges=True)
        return lon_edges

    @property
    def grid_lat_edges(self):
        _, lat_edges = self.make_grid(edges=True)
        return lat_edges

    def __init__(self, projection, domain, lonres, latres):
        if projection not in self._grid_methods.keys():
            raise GridInputError('{} is not an allowed projection. Allowed values are: {}'
                                 .format(projection, ', '.join(self._grid_methods.keys())))

        self._domain = self._setup_domain(domain, lonres, latres)
        self._projection = projection
        self._make_grid = self._grid_methods[projection]

    def _setup_domain(self, domain_in, lon_res, lat_res):
        if isinstance(domain_in, str):
            domain_in = self._predef_domains[domain_in]
        elif not isinstance(domain_in, str):
            raise GridInputError('domain must be a string or dictionary')

        try:
            _center_lon, _center_lat = domain_in['center']
            _lon_lim = sorted(domain_in['lon_lim'])
            _lat_lim = sorted(domain_in['lat_lim'])
        except KeyError:
            raise GridInputError('If given as a dictionary, domain must have the keys "center", "lon_lim", and "lat_lim"')

        if np.size(_lat_lim) != 2:
            raise GridInputError('Lat limits must be a 2-element collection')
        if np.size(_lon_lim) != 2:
            raise GridInputError('Lon limits must be a 2-element collection')

        return {'center_lon': _center_lon, 'center_lat': _center_lat, 'lon_res': lon_res,
                'lat_res': lat_res, 'lon_lim': _lon_lim, 'lat_lim': _lat_lim}

    def make_grid(self, edges=False):
        return self._grid_methods[self._projection](edges=edges, **self._domain)

    def find_grid_indices(self, lon, lat):
        if np.shape(lon) != np.shape(lat):
            raise GridInputError('lon and lat must be the same shape')

        if np.shape(lon) == np.shape(0):
            # if inputs are scalar, make them arrays
            lon = np.array([lon])
            lat = np.array([lat])
            return_scalar = True
        else:
            return_scalar = False

        x_inds = np.zeros_like(lon, dtype=int)
        y_inds = np.zeros_like(lat, dtype=int)

        lon_edges, lat_edges = self.make_grid(edges=True)

        for idx, (i, j) in enumerate(zip(lon, lat)):
            xx = (i >= lon_edges[:-1, :]) & (i < lon_edges[1:, :])
            yy = (j >= lat_edges[:, :-1]) & (j < lat_edges[:, 1:])
            zz = xx & yy
            if np.sum(zz) == 1:
                x_inds[idx], y_inds[idx] = np.argwhere(zz)[0]
            elif np.sum(zz) == 0 and np.isclose(i, lon_edges[-1, 0]) and np.isclose(j, lat_edges[0, -1]):
                # Check if the point is on the last edge, which we ignored b/c of the upper inequality. If so, then the
                # indices are just the last ones in each dimension
                x_inds[idx], y_inds[idx] = [x - 1 for x in zz.shape]
            elif np.sum(zz) == 0:
                raise GridQueryError('Cannot find {}, {} on the grid'.format(i, j))
            else:
                raise GridQueryError('{}, {} placed in multiple grid cells'.format(i, j))

        if return_scalar:
            return x_inds.item(), y_inds.item()
        else:
            return x_inds, y_inds
