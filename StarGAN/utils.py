import chainer

from chainer import optimizers


def set_optimizer(model, alpha, beta):
    optimizer = optimizers.Adam(alpha=alpha, beta1=beta)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001))

    return optimizer
