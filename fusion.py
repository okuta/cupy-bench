import cupy
import numpy

import util


def saxpy(a, x, y):
    return a * x + y


@cupy.fuse()
def saxpy_fuse(a, x, y):
    return a * x + y


for xp in [cupy, numpy]:
    for size in [2 ** i for i in range(20)]:
        if xp is numpy and size > 2 ** 13:
            continue
        a = numpy.float32(2.0)
        x = xp.ones((1024, size), 'f')
        y = xp.ones((1024, size), 'f')

        def f():
            saxpy(a, x, y)
        str = "cp" if xp is cupy else "np"
        util.measure(f, "saxpy_%s  , %8d" % (str, size))

xp = cupy
for size in [2 ** i for i in range(20)]:
    a = numpy.float32(2.0)
    x = xp.ones((1024, size), 'f')
    y = xp.ones((1024, size), 'f')

    def f():
        saxpy_fuse(a, x, y)
    util.measure(f, "saxpy_fuse, %8d" % (size))

