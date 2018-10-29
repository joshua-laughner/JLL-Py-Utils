import datetime as base_datetime


class datetime(base_datetime.datetime):
    """
    datetime(year, month, day[, hour[, minute[, second[, microsecond[, tzinfo]]]]])

    Replacement datetime object, for compatibility with :class:`timedelta` from this module.
    """
    def __add__(self, other):
        if isinstance(other, timedelta):
            return timedelta.__add__(other, self)
        else:
            return super(datetime, self).__add__(other)

    def __sub__(self, other):
        if isinstance(other, timedelta):
            return timedelta.__add__(-other, self)
        else:
            return super(datetime, self).__sub__(other)


class timedelta(object):
    """
    timedelta(years=0, months=0, weeks=0, days=0, hours=0, minutes=0, seconds=0, milliseconds=0, microseconds=0)

    Represents a difference between two :class:`datetime` values. Only compatible with the :class:`datetime` class from
    this module, not the built-in :mod:`datetime` module. Extends the behavior of the built-in module by adding support
    for year and month differences. Note that this means the behavior is more complex than the built-in timedelta.

    When adding to a datetime, years and months are added first, then days, minutes, etc. This matters only if the
    result of adding the years and months results in a day that does not exist, e.g.::

        datetime(2010, 1, 31) + timedelta(months=1)

    Since there is no Feb 31st, the result will actually be Feb 28th, i.e. in the event that adding the requested number
    of months results in a date where the day of the original date does not exist in the new month, the last day of the
    new month is used instead. The order of addition (i.e. years and months first, then days, etc.) matters in a few
    cases such as::

        datetime(2010, 1, 28) + timedelta(months=1, days=1)

    In this case, if the days we added first, it would be trying to add a month to Jan 29th, which following the rules
    above would yield Feb 28th. But, since the months are added first, the actual result is Mar 1st.

    Just as in the built-in timedelta, the number of days may be greater than the number of days in any given month, so
    if you want the date exactly 90 days later, rather than 3 months later, ``date + timedelta(days=90)`` is the proper
    way to do that.
    """
    def __init__(self, years=0, months=0, weeks=0, days=0, hours=0, minutes=0, seconds=0, milliseconds=0, microseconds=0):
        # Like the regular timedelta we'll transform everything into months, days, seconds, and microsceonds
        self._months = years * 12 + months
        self._days = weeks * 7 + days
        self._seconds = hours * 3600 + minutes * 60 + seconds
        self._microseconds = milliseconds * 1000 + microseconds

    def total_seconds(self, start_date=None):
        """
        Compute the total number of seconds (fractional, if necessary) represented by the timedelta.
        :param start_date: optional, if given, computes the number of seconds assuming that the timedelta starts from
         the given date. Only matters if the timedelta contains years or months as, e.g. the number of seconds between
         Jan 1 and Feb 1 is different than between Feb 1 and Mar 1, or between Jan 1, 2010 and Jan 1, 2011 is different
         than between Jan 1, 2012 and Jan 1, 2013 (leap year).
        :return: the number of seconds
        :rtype: float
        """
        if start_date is None:
            # If a start date is not given, then we'll have to make some assumptions about how many days there are per
            # month
            return float(self._months * 30 * 24 * 3600 + self._days * 24 * 3600 + self._seconds + self._microseconds/100000.0)
        else:
            # If there is a start date given, then we can be more careful.
            return float((start_date + self - start_date).total_seconds())

    def _base_timedelta(self):
        """
        Return just the days and smaller components as a standard datetime.timedelta.
        :return:
        """
        return base_datetime.timedelta(days=self._days, seconds=self._seconds, microseconds=self._microseconds)

    def __add__(self, other):
        if not isinstance(other, datetime):
            raise TypeError('Add only defined for timedelta and datemath.datetime objects')

        new_date = add_months(other, self._months)
        return new_date + self._base_timedelta()

    def __sub__(self, other):
        return timedelta.__add__(-self, other)

    def __neg__(self):
        return timedelta(months=-self._months, days=-self._days, seconds=-self._seconds, microseconds=-self._microseconds)


def add_months(date_in, months=1):
    """
    Add a number of months to a given date.

    This function behaves the same as the :class:`timedelta` for cases where adding the months will create a new date
    for a day that doesn't exist, e.g.::

        add_months(datetime(2010, 1, 31), 1)

    will return Feb 28th, 2010 since Feb 31st does not exist. However::

        add_months(datetime(2010, 1, 31), 2)

    will return Mar 31st, 2010. That is, passing through a shorter month does not change the final day, it is only
    adjusted once the final month is reached.

    This function is compatible with the built-in :class:`datetime.date` and :class:`datetime.datetime` classes, unlike
    the :class:`timedelta` class from this module.

    :param date_in: the date to add months to.
    :type date_in: :class:`datetime.datetime`, :class:`datetime.date`, or :class:`datetime`

    :param months: the number of months to add, may be negative
    :type months: int

    :return: a new datetime instance
    :rtype: same as input
    """
    if months == 0:
        return date_in
    elif months > 0:
        step = 1
        jump_month = 1
    else:
        step = -1
        jump_month = 12

    original_day = date_in.day

    while months != 0:
        new_month = (date_in.month - 1 + step) % 12 + 1
        if new_month == jump_month:
            new_year = date_in.year + step
        else:
            new_year = date_in.year

        date_in = date_in.replace(year=new_year, month=new_month, day=1)
        months -= step

    eom = eom_day(date_in)
    final_day = original_day if original_day <= eom else eom

    return date_in.replace(day=final_day)


def eom_day(date_in):
    """
    Get the last day of the given month.

    :param date_in: the date to find the end-of-month day for
    :type date_in: :class:`datetime.date`, :class:`datetime.datetime`, or :class:`datetime`.

    :param hms: optional, if ``False`` (default) then hours and smaller are unchanged. If ``True``, then the datetime
     is set to the very first microsecond of the given month. If ``date_in`` is a :class:`datetime.date`, then this
     parameter has no effect.
    :type hms: bool

    :return: the last day of the month
    :rtype: int
    """
    tmp_date = (date_in.replace(day=1) + base_datetime.timedelta(days=32)).replace(day=1) - base_datetime.timedelta(days=1)
    return tmp_date.day


def som_date(date_in, hms=False):
    if not hms or isinstance(date_in, base_datetime.date):
        return date_in.replace(day=1)
    else:
        sub = date_in - datetime(year=date_in.year, month=date_in.month, day=1)
        return date_in - sub


def eom_date(date_in, hms=False):
    """
    Return a datetime object for the last day of the month

    :param date_in: the date to move to the end of its month
    :type date_in:  :class:`datetime.date`, :class:`datetime.datetime`, or :class:`datetime`.

    :param hms: optional, if ``False`` (default) then hours and smaller are unchanged. If ``True``, then the datetime
     is set to the very last microsecond of the given month. If ``date_in`` is a :class:`datetime.date`, then this
     parameter has no effect.
    :type hms: bool

    :return: the input date with the day changed to the last one in the month
    :rtype: same as ``date_in``
    """
    if not hms or isinstance(date_in, base_datetime.date):
        return date_in.replace(day=eom_day(date_in))
    else:
        return som_date(add_months(date_in), True) - base_datetime.timedelta(microseconds=1)
