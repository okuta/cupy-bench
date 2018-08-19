import numpy
import numpy
import cupy

import util


for xp in [cupy, numpy]:
    for size in [2 ** i for i in range(10, 16, 5)]:
        st = xp.random.RandomState()
        st.seed(0)
        str = "cupy" if xp is cupy else "numpy"
        def f():
            st.beta(2, 2, size=(size,))
        util.measure(f, "beta_%s    , %5d" % (str, size), 5)
        def f():
            st.binomial(10, 0.5, size=(size,))
        util.measure(f, "binomial_%s, %5d" % (str, size), 5)
        def f():
            st.lognormal(size=(size,))
        util.measure(f, "lognormal%s, %5d" % (str, size), 5)
        def f():
            st.normal(size=(size,))
        util.measure(f, "normal%s   , %5d" % (str, size), 5)
