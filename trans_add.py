import cupy
import numpy

import util


for xp in [cupy, numpy]:
    for trans in [False, True]:
        for size in [2 ** i for i in range(16)]:
            if trans:
                a = xp.zeros((32, size), dtype='f').T
            else:
                a = xp.zeros((size, 32), dtype='f')
            b = xp.zeros((size, 32), dtype='f')
            def f():
                a + b
            head = "t" if trans else "n"
            str = "cp" if xp is cupy else "np"
            util.measure(f, "%s_add_%s, %5d" % (head, str, size))

