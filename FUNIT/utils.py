from chainer import optimizers


def set_optimizer(model, alpha=0.0001):
    optimizer = optimizers.RMSprop(lr=alpha)
    optimizer.setup(model)

    return optimizer