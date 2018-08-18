import random

import cupy
import numpy

import util


sizes = [2 ** i for i in range(20)] * 100
random.seed(0)
random.shuffle(sizes)


def dummy_call():
    pass

def mem_empty():
    cupy.cuda.alloc(0)

def mem_1K():
    cupy.cuda.alloc(1024)

def mem_1M():
    cupy.cuda.alloc(1024 * 1024)


cnt = 1000
util.measure(dummy_call, "dummy_call", cnt)
util.measure(mem_empty, "mem_0B", cnt)
util.measure(mem_1K, "mem_1K", cnt)
util.measure(mem_1M, "mem_1M", cnt)
sizes = [(0, "0B"), (1024, "1K"), (1024 * 1024, "1M")]

for xp in [cupy, numpy]:
    if xp is cupy:
        str = "cupy"
    else:
        str = "numpy"
    if xp is cupy:
        for size, s in sizes:
            memptr = cupy.cuda.alloc(size)
            def alloc():
                xp.ndarray((size,), dtype='b', memptr=memptr)
            util.measure(alloc, "%s_memptr_%s" % (str, s), cnt)
    for size, s in sizes:
        def alloc():
             xp.ndarray((size,), dtype='b')
        util.measure(alloc, "%s_alloc_%s" % (str, s), cnt)
    for size, s in sizes:
        def alloc():
            xp.empty((size,), dtype='b')
        util.measure(alloc, "%s_empty_%s" % (str, s), cnt)
    for size, s in sizes:
        def zeros():
            xp.zeros((size,), dtype='b')
        util.measure(zeros, "%s_zeros_%s" % (str, s), cnt)
    for size, s in sizes:
        def ones():
            xp.ones((size,), dtype='b')
        util.measure(ones, "%s_ones_%s" % (str, s), cnt)

