#!/bin/bash -x

python -c "import cupy; print(cupy.__version__)"

python -c "import inspect, cupy, numpy;f = lambda x:len([mem for mem in inspect.getmembers(x) if inspect.isfunction(mem[1])]); print(f(numpy), f(cupy))"

python alloc.py

python alloc_empty.py

python matmul.py

python trans_add.py

python adam.py

