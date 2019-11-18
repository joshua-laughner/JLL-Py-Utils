from .subutils.leapseconds import utc_to_gps, utc_to_tai, gps_to_utc, tai_to_utc, tai_to_gps, gps_to_tai, tai93_to_utc


_tz2offset = {'EST': -5, 'EDT': -4, 'CST': -6, 'CDT': -5, 'MST': -7, 'MDT': -6, 'PST': -8, 'PDT': -7}
_std_tz = ('EST', 'CST', 'MST', 'PST')
_daysave_tz = ('EDT', 'CDT', 'MDT', 'PDT')


def start_of_month(date_in):
    """
    Return the start of a month as the same type as the input

    :param date_in: a datetime-like object specifying a date to find the beginning of the month for. Must be a type
     that can be constructed with a call type(date_in)(year, month, day).

    :return: date of the first of the month containing ``date_in``
    """
    return type(date_in)(date_in.year, date_in.month, 1)


def tzname_to_offset(tz):
    """
    Return the UTC offset for a given timezone

    :param tz: the timezone abbreviation
    :type tz: str

    :return: the UTC offset
    :rtype: int
    """
    try:
        return _tz2offset[tz.upper()]
    except KeyError:
        raise ValueError('Time zone abbreviation "{}" not recognized. Valid timezones are: {}'.format(tz.upper(), ', '.join(_tz2offset.keys())))


def offset_to_tzname(offset, daylight_savings=False):
    """
    Return the timezone abbreviation for a given offset

    :param offset: the UTC offset
    :type offset: int

    :param daylight_savings: specifies if the offset given is for daylight
     savings (``True``) or standard time (``False``)
    :type dayight_savings: bool

    :return: the timezone abbrevation. If the timezone cannot be found, then "UTC+/-n" is returned.
    :rtype: str
    """
    if daylight_savings:
        offset2tz = {_tz2offset[k]: k for k in _daysave_tz}
    else:
        offset2tz = {_tz2offset[k]: k for k in _std_tz}

    try:
        return offset2tz[offset]
    except KeyError:
        return 'UTC{:+03d}'.format(offset)

