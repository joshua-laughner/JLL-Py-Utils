from .subutils.leapseconds import utc_to_gps, utc_to_tai, gps_to_utc, tai_to_utc, tai_to_gps, gps_to_tai, tai93_to_utc


_tz2offset = {'EST': -5, 'EDT': -4, 'CST': -6, 'CDT': -5, 'MST': -7, 'MDT': -6, 'PST': -8, 'PDT': -7}
_std_tz = ('EST', 'CST', 'MST', 'PST')
_daysave_tz = ('EDT', 'CDT', 'MDT', 'PDT')


def start_of_month(date_in):
    """Return the start of a month as the same type as the input

    Parameters
    ----------

    date_in
        a datetime-like object specifying a date to find the beginning of the month for. Must be a type
        that can be constructed with a call type(date_in)(year, month, day).


    Returns
    -------

    datetime-like
        date of the first of the month containing `date_in`. Will be same type as `date_in`.
    """
    return type(date_in)(date_in.year, date_in.month, 1)


def tzname_to_offset(tz):
    """Return the UTC offset for a given timezone

    Parameters
    ----------

    tz : str
        the timezone abbreviation

    Returns
    -------

    int
        the UTC offset
    """
    try:
        return _tz2offset[tz.upper()]
    except KeyError:
        raise ValueError('Time zone abbreviation "{}" not recognized. Valid timezones are: {}'.format(tz.upper(), ', '.join(_tz2offset.keys())))


def offset_to_tzname(offset, daylight_savings=False):
    """Return the timezone abbreviation for a given offset

    Parameters
    ----------
    
    offset : int
        the UTC offset

    daylight_savings : bool 
        specifies if the offset given is for daylight savings (`True`) or standard time (`False`)

    
    Returns
    -------
    
    str
        the timezone abbrevation. If the timezone cannot be found, then "UTC+/-n" is returned.
    """
    if daylight_savings:
        offset2tz = {_tz2offset[k]: k for k in _daysave_tz}
    else:
        offset2tz = {_tz2offset[k]: k for k in _std_tz}

    try:
        return offset2tz[offset]
    except KeyError:
        return 'UTC{:+03d}'.format(offset)


def get_utc_offset(offset):
    """Ensure a UTC offset is numeric.

    Parameters
    ----------

    offset : str, int, or float
        The UTC offset to coerce to a numeric offset. May be numeric
        (e.g. +1, -8) or a timezone string supported by `tzname_to_offset`.


    Returns
    -------

    int or float
        The UTC offset. If the input was a string, the corresponding numeric
        offset is returned. If the input was already numeric, it is returned unchanged.        
    """
    if isinstance(offset, str):
        return tzname_to_offset(offset)
    else:
        return offset


def hour_of_day(dates, utc_offset=0):
    """Calculate hour after midnight for a date or dates

    Parameters
    ----------

    dates : datetime-like
        The date or dates to compute hours past midnight for. See Notes on acceptable
        types.

    utc_offset : int, float, or str
        A UTC offset to apply to the hours. Assumes the input dates are UTC. May be a 
        numeric offset (e.g. +1, -8) or a timezone string (e.g. "PST").


    Returns
    -------
    
    float or array-like
        The hours past midnight of the input date or dates. The exact return type
        depends on the input type.

    
    Notes
    -----

    This function assumes that the input date(s) have attributes `hour`, `minute`,
    and `second`. This means that Pandas :class:`~pandas.DatetimeIndex` objects will
    work, but lists of datetimes will not.
    """
    hours = (dates.hour + get_utc_offset(utc_offset)) % 24
    minutes = dates.minute / 60
    seconds = dates.second / 3600
    return hours + minutes + seconds

