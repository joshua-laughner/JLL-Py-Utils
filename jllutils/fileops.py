import contextlib
import sys


@contextlib.contextmanager
def smart_open(filename, mode):
    if filename is None or filename == '-':
        if mode.startswith('r'):
            handle = sys.stdin
        else:
            handle = sys.stdout
        do_close = False
    else:
        handle = open(filename, mode)
        do_close = True

    try:
        yield handle
    finally:
        if do_close:
            handle.close()
