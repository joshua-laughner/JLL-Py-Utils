"""Functions for specialized interpolation
"""

import numpy as np

from . import miscutils
from . import geography as jgeo

from typing import Optional, Sequence, Dict

DEFAULT_DIST_CUTOFF = 300.0  # km
DEFAULT_TIME_CUTOFF = 45*60  # seconds = ~half an orbit
DEFAULT_DIST_WT_POWER = 1.0

def idw_interpolation_with_cutoff_lowmem(src_vals: Dict[str, np.ndarray], tgt_vals: Dict[str, np.ndarray], vars_to_interp: Optional[Sequence[str]] = None, 
                                         dist_cutoff_km: float = DEFAULT_DIST_CUTOFF, time_cutoff: float = DEFAULT_TIME_CUTOFF, time_rel_weight: float = 1.0,
                                         min_count_weight: Optional[float] = None, dist_weight_power: float = DEFAULT_DIST_WT_POWER) -> Dict[str, np.ndarray]:
    """
    Perform an inverse distance weighted interpolation from ``src_vals`` to the coordinates defined in ``tgt_vals``.

    Parameters
    ----------
    src_vals
        A dictionary containing 1D numpy arrays. It must have at least the keys "lon", "lat", and "timestamp", which point
        to the longitude, latitude, and time (in any numerical representation, such as a Unix timestamp or Julian day),
        respectively. Additionally, it must contain all of the ``vars_to_interp`` as keys.

    tgt_vals
        A dictionary containing 1D numpy arrays for the keys "lon", "lat", and "timestamp", which have the same meaning
        as in ``src_vals``. These may be given following any convention (e.g. longitude given as west is negative *or*
        west is >180), but the convention used must match between ``src_vals`` and ``tgt_vals``.

    vars_to_interp
        The keys in ``src_vals`` to interpolate to the coordinates in ``tgt_vals``. If not given, all keys in ``src_vals``
        except "lon", "lat", and "timestamp" are interpolated.

    dist_cutoff_km
        The maximum distance in kilometers between a point in ``tgt_vals`` and one in ``src_vals`` at which the source value
        can contribute to the interpolation.

    time_cutoff
        The maximum absolute time difference, in the same units as "timestamp" in ``src_vals`` and ``tgt_vals``, at which a
        source value can contribute to the value of an interpolated value. The default value is intended for interpolating
        values from one sun-synchronous satellite to another; those orbits are typically about 90 to 100 minutes in duration,
        so the cutoff is set to half of that to prevent points from different orbits being interpolated together.

    time_rel_weight
        How much to weight the time difference compared to the distance when computing the interpolation weights. The default,
        ``1.0``, weights both time and distance equally. A value of ``0`` would essentially exclude the time difference from
        the weights, but retain the time cutoff.

    min_count_weight
        The minimum weight a source value must have to be counted as contributing to the interpolation for the purpose of the
        "counts" output. If not given, then for each interpolation, points whose weight is at least 0.1% of the maximum will
        count.

    dist_weight_power
        The exponent to apply to distance in the weights. The default, 1, will make the distance weights be ``1/distance``.
        Increasing to e.g. 2 would make the weights ``1/distance**2``.

    Returns
    -------
    src_on_tgt
        A dictionary with all the ``vars_to_interp`` as keys, which will be the values interpolated to the coordinates in
        ``tgt_vals``. It will also contain the keys "weighted_dist" (the mean weighted distance from all the contributing
        source points to a given target point), "weighted_dt" (the same, except for time), and "counts" (the number of source
        points with weights above the threshold specified by ``min_count_weight``).
    """
    if vars_to_interp is None:
        vars_to_interp = [k for k in src_vals if k not in {'lon', 'lat', 'timestamp'}]
    n = tgt_vals['lon'].size
    src_on_tgt = {k: np.full(n, np.nan) for k in vars_to_interp}
    src_on_tgt['weighted_dist'] = np.full(n, np.nan)
    src_on_tgt['weighted_dt'] = np.full(n, np.nan)
    src_on_tgt['counts'] = np.zeros(n, dtype=int)
    pbar = miscutils.ProgressBar(n, prefix='Intepolating with IDW lowmem')
    for i in range(n):
        pbar.print_bar()
        weights, weight_debug = _calc_weights(
            src_vals, tgt_vals, i, dist_cutoff_km=dist_cutoff_km, 
            time_cutoff=time_cutoff, time_rel_weight=time_rel_weight,
            dist_power=dist_weight_power,
        )
        if not np.all(np.isnan(weights)):
            for k in vars_to_interp:
                src_on_tgt[k][i] = np.nansum(weights * src_vals[k])
            src_on_tgt['weighted_dist'][i] = np.nansum(weights * weight_debug['distances'])
            src_on_tgt['weighted_dt'][i] = np.nansum(weights * weight_debug['delta_times'])
            wt_threshold = min_count_weight if min_count_weight is not None else 0.001 * np.nanmax(weights)
            src_on_tgt['counts'][i] = np.sum(weights > wt_threshold)
    return src_on_tgt


def _dist_matrix(src_vals, tgt_vals):
    nsrc = src_vals['lon'].size
    ntgt = tgt_vals['lon'].size
    distances = np.full([ntgt, nsrc], np.nan)
    for itgt in range(ntgt):
        distances[itgt] = jgeo.great_circle_distance(src_vals['lon'], src_vals['lat'], tgt_vals['lon'][itgt], tgt_vals['lat'][itgt])
    return distances


def _delta_time_matrix(src_vals, tgt_vals, min_dt_sec=60):
    nsrc = src_vals['timestamp'].size
    ntgt = tgt_vals['timestamp'].size
    delta_times = np.full([ntgt, nsrc], np.nan)
    for itgt in range(ntgt):
        dt = np.abs(src_vals['timestamp'] - tgt_vals['timestamp'][itgt])
        delta_times[itgt] = np.clip(dt, a_min=60, a_max=None)
    return delta_times


def _calc_weights(src_vals, tgt_vals, i, dist_cutoff_km=DEFAULT_DIST_CUTOFF, time_cutoff=DEFAULT_TIME_CUTOFF, time_rel_weight=1.0, 
                  dist_power=DEFAULT_DIST_WT_POWER):
    # these must be 1D arrays still, hence the slice for the index
    this_tgt = {'lon': tgt_vals['lon'][i:i+1], 'lat': tgt_vals['lat'][i:i+1], 'timestamp': tgt_vals['timestamp'][i:i+1]}
    distances = _dist_matrix(src_vals, this_tgt).squeeze()
    delta_times = _delta_time_matrix(src_vals, this_tgt).squeeze()
    in_dist_cutoff = distances <= dist_cutoff_km
    in_time_cutoff = delta_times <= time_cutoff
    cutoff_crit = in_dist_cutoff & in_time_cutoff
    dist_weights = np.where(cutoff_crit, 1/(distances**dist_power), 0)
    dist_weights /= np.sum(dist_weights)
    
    time_weights = np.where(cutoff_crit, 1/delta_times, 0)
    time_weights /= np.sum(time_weights)

    weights = (dist_weights + time_rel_weight * time_weights) / (1.0 + time_rel_weight)
    debug_info = {
        'distances': distances,
        'times': src_vals['timestamp'],
        'delta_times': delta_times,
        'dist_weights': dist_weights,
        'time_weights': time_weights,
        'in_dist_cutoff': in_dist_cutoff,
        'in_time_cutoff': in_time_cutoff,
    }
    return weights, debug_info
