import cupy
import numpy

import util


for xp in [cupy, numpy]:
    for trans in [False, True]:
        for size in [2 ** i for i in range(16)]:
            if trans:
                a = xp.ones((32, size), 'f').T
            else:
                a = xp.ones((size, 32), 'f')
            b = xp.ones((size, 32), 'f')
            def f():
                a + b
            head = "t" if trans else "n"
            str = "cp" if xp is cupy else "np"
            util.measure(f, "%s_add_%s, %5d" % (head, str, size))

for xp in [cupy, numpy]:
    for trans in [False, True]:
        for size in [2 ** i for i in range(16)]:
            if trans:
                a = xp.ones((32, size), 'f').T
            else:
                a = xp.ones((size, 32), 'f')
            b = xp.ones((size, 32), 'f')
            def f():
                a ** b
            head = "t" if trans else "n"
            str = "cp" if xp is cupy else "np"
            util.measure(f, "%s_pow_%s, %5d" % (head, str, size))

