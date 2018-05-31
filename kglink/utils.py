# coding:utf8
"""
copyright Asan AGIBETOV <asan.agibetov@gmail.com>

Benchmark utils functions 

"""
# Standard-library imports
import time
import cProfile


def profileit(log_message):
    def inner(func):
        def wrapper(*args, **kwargs):
            # datafn = func.__name__ + ".profile" # Name the data file sensibly
            prof = cProfile.Profile()
            retval = prof.runcall(func, *args, **kwargs)

            print("*" * 80)
            print(log_message)
            print("*" * 80)

            # if write to file then prof.dump_stats(datafn)
            prof.print_stats()

            return retval

        return wrapper
    return inner


def timeit(f):
    def another_f(*args, **kwargs):
        start = time.time()
        res = f(*args, **kwargs)
        delta = time.time() - start

        return res, delta

    return another_f
