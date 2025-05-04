import time


def decorator_time_measurement_log(description: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            retval = func(*args, **kwargs)
            end = time.time()
            print("[", "%1.4f" % round(end - start, 4), "s", "]", " ", description)
            return retval

        return wrapper

    return decorator
