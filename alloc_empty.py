import gc
import random

import cupy

import util


sizes = [2 ** i for i in range(10, 25)] * 100
random.seed(0)
random.shuffle(sizes)

def f1():
    [cupy.cuda.alloc(s) for s in sizes]

def f2():
    buf = []
    for i, s in enumerate(sizes):
        buf.append(cupy.cuda.alloc(s))
        if i % 10 == 0:
            buf[i // 10] = None

def f3():
    [cupy.empty((s,), dtype='b') for s in sizes]

def f4():
    buf = []
    for i, s in enumerate(sizes):
        buf.append(cupy.empty((s,), dtype='b'))
        if i % 10 == 0:
            buf[i // 10] = None

util.measure(f1, "alloc         ")

cupy.get_default_memory_pool().free_all_blocks()
gc.collect()
util.measure(f2, "alloc_and_free")

cupy.get_default_memory_pool().free_all_blocks()
gc.collect()
util.measure(f3, "empty         ")

cupy.get_default_memory_pool().free_all_blocks()
gc.collect()
util.measure(f4, "empty_and_free")

