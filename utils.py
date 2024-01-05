import os
from contextlib import contextmanager

@contextmanager
def temp_stdout_removal():
    old = os.dup(1)
    try:
        os.close(1)
        yield
    finally:
        os.dup(old)
        os.close(old)
