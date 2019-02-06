"""
Utilities that don't fit any other well defined category
"""


def all_or_none(val):
    """
    Return ``True`` if all or no values in the input are truthy

    :param val: the iterable to check
    :type val: iterable of truth-like values

    :return: ``True`` if all values in ``val`` are true or all are false; ``False`` if there is a mixture.
    :rtype: bool
    """
    s = sum([bool(v) for v in val])
    return s == 0 or s == len(val)
