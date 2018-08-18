import time

import numpy

import cupy


def measure(func, name="hoge", cnt=10, times=20):
    stream = cupy.cuda.Stream.null
    gpu_times = []
    cpu_times = []
    for _ in range(times):
        start_gpu = stream.record()
        start_cpu = time.time()
        for i in range(cnt):
            func()
        end_cpu = time.time()
        end_gpu = stream.record()
        end_gpu.synchronize()
        elapsed_gpu = cupy.cuda.get_elapsed_time(start_gpu, end_gpu) / cnt
        elapsed_cpu = (end_cpu - start_cpu) / cnt * 1000
        gpu_times.append(elapsed_gpu)
        cpu_times.append(elapsed_cpu)

    gpu_times = sorted(gpu_times)[2:-2]
    cpu_times = sorted(cpu_times)[2:-2]
    print("%s, %f, %f" % (
        name, numpy.average(gpu_times), numpy.average(cpu_times)))

