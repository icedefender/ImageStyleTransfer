import chainer
import chainer.functions as F
import argparse
import os
import pylab

from chainer import cuda, optimizers, serializers

xp = cuda.cupy
cuda.get_device(0).use()


def set_optimizer(model, alpha=0.00001, beta=0.5):
    optimizer = optimizers.Adam(alpha=alpha, beta1=beta)
    optimizer.setup(model)

    return optimizer

