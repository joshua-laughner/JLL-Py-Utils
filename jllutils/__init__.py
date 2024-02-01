from . import plots as jplots
try:
    from . import stats as jstats
except ImportError as e:
    from warnings import warn
    warn(f'Cannot import stats module: {e}')
    __all__ = ['databases', 'datemath', 'display', 'fileops', 'matrices', 'miscutils', 'jplots']
else:
    __all__ = ['databases', 'datemath', 'display', 'fileops', 'matrices', 'miscutils', 'jplots', 'jstats']