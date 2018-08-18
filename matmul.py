import numpy
import cupy

import util


for xp in [cupy, numpy]:
    for size in [2 ** i for i in range(12)]:
        a = xp.zeros((size, size), dtype='f')
        b = xp.zeros((size, size), dtype='f')
        def f():
            xp.matmul(a, b)
        str = "cupy" if xp is cupy else "numpy"
        util.measure(f, "matmul_%s, %5d" % (str, size), 5)

