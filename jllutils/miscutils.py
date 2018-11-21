"""
Utilities that don't fit any other well defined category
"""

def all_or_none(val):
    s = sum([bool(v) for v in val])
    return s == 0 or s == len(val)
