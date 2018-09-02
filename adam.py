import cupy
import numpy

import util


adam = cupy.ElementwiseKernel(
    'T grad, T lr, T one_minus_beta1, T one_minus_beta2, '
    'T eps, T eta, T weight_decay_rate',
    'T param, T m, T v',
    '''m += one_minus_beta1 * (grad - m);
    v += one_minus_beta2 * (grad * grad - v);
    param -= eta * (lr * m / (sqrt(v) + eps) +
        weight_decay_rate * param);''',
    'adam')

@cupy.fuse()
def adam_fuse(
        grad, lr, one_minus_beta1, one_minus_beta2, eps, eta,
        weight_decay_rate, param, m, v):
    xp = cupy.get_array_module(grad)
    m += one_minus_beta1 * (grad - m)
    v += one_minus_beta2 * (grad * grad - v)
    param -= eta * (lr * m / (xp.sqrt(v) + eps) + weight_decay_rate * param)


class hp:
    lr = 0.1
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    eta = 1.0
    weight_decay_rate = 0.9


def call_adam(grad, data, state_m, state_v):
    adam(grad, hp.lr, 1 - hp.beta1, 1 - hp.beta2, hp.eps, hp.eta,
         hp.weight_decay_rate, data, state_m, state_v)


def call_adam_fuse(grad, data, state_m, state_v):
    adam_fuse(grad, numpy.float32(hp.lr), numpy.float32(1 - hp.beta1),
              numpy.float32(1 - hp.beta2), numpy.float32(hp.eps),
              numpy.float32(hp.eta), numpy.float32(hp.weight_decay_rate),
              data, state_m, state_v)

sizes = [1, 10, 100, 1000, 2000, 5000]
for size in sizes:
    zero = cupy.zeros((size, size))

    def f():
        call_adam(zero, zero, zero, zero)
    util.measure(f, "adam     , %4d" % (size), 100)

for size in sizes:
    zero = cupy.zeros((size, size))

    def f():
        call_adam_fuse(zero, zero, zero, zero)
    util.measure(f, "adam_fuse, %4d" % (size), 100)

for size in sizes[:4]:
    zero = numpy.zeros((size, size))

    def f():
        call_adam_fuse(zero, zero, zero, zero)
    util.measure(f, "adam_np  , %4d" % (size), 10)
