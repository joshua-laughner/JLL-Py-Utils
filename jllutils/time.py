from .subutils.leapseconds import utc_to_gps, utc_to_tai, gps_to_utc, tai_to_utc, tai_to_gps, gps_to_tai, tai93_to_utc


def start_of_month(date_in):
    """
    Return the start of a month as the same type as the input

    :param date_in: a datetime-like object specifying a date to find the beginning of the month for. Must be a type
     that can be constructed with a call type(date_in)(year, month, day).

    :return: date of the first of the month containing ``date_in``
    """
    return type(date_in)(date_in.year, date_in.month, 1)
