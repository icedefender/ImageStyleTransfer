import numpy as np
import chainer
import argparse
import chainer.functions as F
import chainer.links as L

from pathlib import Path
from dataset import DatasetLoader
from model import Generator, Discriminator
from utils import set_optimizer
from chainer import serializers, cuda

xp = cuda.cupy
cuda.get_device(0).use()


def adversarial_loss_dis(discriminator, fake, real, y_label, x_label):
    fake = discriminator(fake, y_label)
    real = discriminator(real, x_label)

    return F.mean(F.softplus(-real)) + F.mean(F.softplus(fake))


def adversarial_loss_gen(discriminator, y_fake, x_fake, x, y_label):
    fake = discriminator(y_fake, y_label)

    loss = F.mean(F.softplus(-fake))
    loss += 10 * F.mean_absolute_error(x_fake, x)

    return loss


def train(epochs, batchsize, iterations, nc_size, data_path, modeldir):
    # Dataset definition
    dataset = DatasetLoader(data_path, nc_size)

    # Model Definition & Optimizer Definition
    generator = Generator(nc_size)
    generator.to_gpu()
    gen_opt = set_optimizer(generator, 0.0001, 0.5)
    
    discriminator = Discriminator(nc_size)
    discriminator.to_gpu()
    dis_opt = set_optimizer(discriminator, 0.0001, 0.5)

    for epoch in range(epochs):
        sum_gen_loss = 0
        sum_dis_loss = 0
        for batch in range(0, iterations, batchsize):
            x, x_label, y, y_label = dataset.train(batchsize)

            y_fake = generator(x, y_label)
            y_fake.unchain_backward()

            loss = adversarial_loss_dis(discriminator, y_fake, x, y_label, x_label)

            discriminator.cleargrads()
            loss.backward()
            dis_opt.update()
            loss.unchain_backward()

            sum_dis_loss += loss.data

            y_fake = generator(x, y_label)
            x_fake = generator(y_fake, x_label)
            x_id = generator(x, x_label)

            loss = adversarial_loss_gen(discriminator, y_fake, x_fake, x, y_label)

            if epoch < 20:
                loss += 10 * F.mean_absolute_error(x_id, x)

            generator.cleargrads()
            loss.backward()
            gen_opt.update()
            loss.unchain_backward()

            sum_gen_loss += loss.data

            if batch == 0:
                serializers.save_npz(f"{modeldir}/generator_{epoch}.model", generator)
                serializers.save_npz("discriminator.model", discriminator)

        print(f"epoch: {epoch} disloss: {sum_dis_loss/iterations} genloss: {sum_gen_loss/iterations}")


parser = argparse.ArgumentParser(description="StarGAN")
parser.add_argument("--e", type=int, default=1000, help="the number of epochs")
parser.add_argument("--b", type=int, default=64, help="batch size")
parser.add_argument("--i", type=int, default=2000, help="the number of iterations")
parser.add_argument("--n", type=int, default=6, help="the number of categories")

args = parser.parse_args()

data_path = Path("./face_stargan/")
modeldir = Path("./modeldir")
modeldir.mkdir(exist_ok=True)

train(args.e, args.b, args.i, args.n, data_path, modeldir)
