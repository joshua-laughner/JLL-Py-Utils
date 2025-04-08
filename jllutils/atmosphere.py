import numpy as np

from typing import Optional


class PresBoundsError(Exception):
    """An exception used to indicate that pressure bounds are too
    close together to calculate reliable pressure weights
    """
    pass


def calculate_subcol_pressure_weight(pressure: np.ndarray, pmin: Optional[float] = None, pmax: Optional[float] = None) -> np.ndarray:
    """Calculates simplified pressure weights to use when computing a trace gas subcolumn
    
    Parameters
    ----------
    pressure
        A numpy 1D array of pressures for one sounding in a TROPESS file. The pressures
        must be ordered surface-to-space (i.e. must monotonically decrease) and must not
        contain any fill values (usually -999)
        
    pmin
        The top (minimum) pressure for the desired subcolumn. If omitted, the subcolumn
        will extend to the top of the atmosphere.
        
    pmax
        The bottom (maximum) pressure for the desired subcolumn. If omitted, the subcolumn
        will extend to the surface.
        
    Returns
    -------
    pres_weights
        A numpy 1D array that will be the same shape as ``pressure`` and will contain the pressure
        weights needed to estimate the subcolumn bounded by ``pmin`` and ``pmax``.
        
    Raises
    ------
    ValueError
        - if ``pressure`` contains fill values, or
        - if ``pressure`` is not monotonically decreasing
        
    PresBoundsError
        if no level in ``pressure`` is between ``pmin`` and ``pmax``. This includes when ``pmax`` < ``pmin``.
    """
    # Check for fill values
    if not np.all(pressure > -0.1):
        raise ValueError('Pressure vector must not contain fill values')
    
    # Check input assumptions about pressure order
    if not np.all(np.diff(pressure) < 0):
        raise ValueError('Pressure is not monotonically decreasing')
        
    if pmin is not None and pmax is not None and pmax < pmin:
        raise PresBoundsError(f'pmax ({pmax:.2f}) must be greater than pmin ({pmin:.2f})')
    
    # Find the levels that fall inside the pressure bounds
    i_first = None
    i_last = None
    if pmax is not None and pmax < pressure[0]:
        i_first = np.flatnonzero(pressure < pmax + 0.01)[0]
    if pmin is not None and pmin > pressure[-1]:
        i_last = np.flatnonzero(pressure > pmin - 0.01)[-1]
        
    # Calculate the initial pressure weights as if pmin and pmax are not given
    layer_pw = pressure[:-1] - pressure[1:]
    layer_pw /= np.sum(layer_pw)
    level_pw = np.zeros_like(pressure)
    level_pw[:-1] += 0.5*layer_pw
    level_pw[1:] += 0.5*layer_pw
    
    if i_first is None and i_last is None:
        # No pressure bounds, return weights for all levels
        return level_pw
    
    
    num_lev = _num_levels_in_bounds(i_first, i_last, len(pressure))
    
    if num_lev >= 2:
        # The pressure bounds are large enough for at least 2 levels to be inside them.
        # That means no level needs modified to account for both the pmin and pmax truncating
        # the column integral.
        return _adjust_edge_pres_weights(
            level_pw=level_pw, layer_pw=layer_pw, pressure=pressure, pmin=pmin, pmax=pmax, i_first=i_first, i_last=i_last
        )
    elif num_lev == 1:
        i = _get_common_level(i_first, i_last, len(pressure))
        return _adjust_edge_pres_weights_one_level(
            level_pw=level_pw, layer_pw=layer_pw, pressure=pressure, pmin=pmin, pmax=pmax, i=i,
        )
    else:
        raise PresBoundsError(f'No levels fall in side the pressure bounds {pmax:.1f} to {pmin:.1f}')
        
        
def _adjust_edge_pres_weights(level_pw: np.ndarray, layer_pw: np.ndarray, pressure: np.ndarray, 
                              pmin: Optional[float], pmax: Optional[float], i_first: Optional[int], i_last: Optional[int]):
    """This function updates and returns ``level_pw`` with smaller weights in the levels near the pressure bounds
    to account for the pressure bound cutting off part of a layer.
    """
    if i_first is not None:
        p_frac = (pmax - pressure[i_first])/(pressure[i_first-1] - pressure[i_first])
        level_pw[i_first] = 0.5 * layer_pw[i_first] + (1 - 0.5*p_frac) * layer_pw[i_first-1] * p_frac
        level_pw[i_first-1] = 0.5 * p_frac * layer_pw[i_first-1] * p_frac
        level_pw[:i_first-1] = 0.0
        
    if i_last is not None:
        p_frac = (pressure[i_last] - pmin)/(pressure[i_last] - pressure[i_last+1])
        level_pw[i_last] = 0.5 * layer_pw[i_last-1] + (1 - 0.5 * p_frac) * layer_pw[i_last] * p_frac
        level_pw[i_last+1] = 0.5 * p_frac * layer_pw[i_last] * p_frac
        level_pw[i_last+2:] = 0.0
        
    return level_pw
    
    
def _adjust_edge_pres_weights_one_level(level_pw: np.ndarray, layer_pw: np.ndarray, pressure: np.ndarray, 
                                        pmin: Optional[float], pmax: Optional[float], i: int):
    """This function updates and returns ``level_pw`` for the case where the pressure limits only include
    a single level.
    """
    if pmin is None:
        # Set pmin to the top of atmosphere if not given
        pmin = 0
    if pmax is None:
        # Set pmax to the surface pressure if not given
        pmax = np.max(pressure)
        
    p_frac_below = (pmax - pressure[i])/(pressure[i-1] - pressure[i])
    p_frac_above = (pressure[i] - pmin)/(pressure[i] - pressure[i+1])
    level_pw[:i-1] = 0.0
    level_pw[i+2:] = 0.0
    level_pw[i-1] = 0.5 * p_frac_below * layer_pw[i-1] * p_frac_below
    level_pw[i] = (1 - 0.5 * p_frac_below) * layer_pw[i-1] * p_frac_below + (1 - 0.5 * p_frac_above) * layer_pw[i] * p_frac_above
    level_pw[i+1] = 0.5 * p_frac_above * layer_pw[i] * p_frac_above
    return level_pw
    
# ----------------- #    
# Utility functions #
# ----------------- #
    
def _num_levels_in_bounds(i_first: int, i_last: int, npres: int):
    """Helper function to count the number of levels between two
    optional indices
    """
    if i_first is None:
        i_first = 0
    if i_last is None:
        i_last = npres - 1
    
    return i_last - i_first + 1


def _get_common_level(i_first: int, i_last: int, npres: int):
    """Helper function to identify a single level between two
    optional indices
    """
    if i_first == i_last:
        return i_first
    if i_first is None and i_last == 0:
        return 0
    if i_last is None and i_first == npres - 1:
        return npres-1
    
    raise PresBoundsError(f'Cannot get common level with i_first = {i_first} and i_last = {i_last}')
