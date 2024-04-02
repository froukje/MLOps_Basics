import time
from functools import wraps

def timing(f):
    """Decorator for timing functions
    Usage:
    @timing
    def function(a):
        pass
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print(f"function: {f.__name__} took: {end - start}:.5f sec")
        return result
    return wrapper