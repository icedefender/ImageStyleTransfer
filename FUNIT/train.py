import chainer
import chainer.functions as F
import numpy as np
import argparse

from pathlib import Path
from chainer import serializers, cuda
from utils import set_optimizer
from evaluation import Evaluation
from dataset import DatasetLoader
from model import Discriminator, Generator
from sn_net import SNGenerator, SNDiscriminator

xp = cuda.cupy
cuda.get_device(0).use()


class FUNITLossFunction:
    def __init__(self):
        pass

    @staticmethod
    def dis_hinge_loss(fake, real):
        return F.mean(F.relu(1. - real)) + F.mean(F.relu(1. + fake))

    @staticmethod
    def gen_hinge_loss(fake):
        return -F.mean(fake)

    def dis_loss(self, discriminator, y, s, si):
        _, fake = discriminator(y, si)
        _, real = discriminator(s, si)

        loss = self.dis_hinge_loss(fake, real)

        return loss

    def gen_loss(self, discriminator, y_convert, y_recon, s, c, si, ci):
        fake_convert_feat, fake_convert = discriminator(y_convert, si)
        fake_recon_feat, fake_recon = discriminator(y_recon, ci)
        real_feat_c, _ = discriminator(c, ci)
        real_feat_s, _ = discriminator(s, si)

        adv_loss = (self.gen_hinge_loss(fake_convert) + self.gen_hinge_loss(fake_recon)) * 0.5
        recon_loss = F.mean_absolute_error(c, y_recon)
        fm_loss = F.mean_absolute_error(fake_convert_feat, real_feat_s)
        fm_loss += F.mean_absolute_error(fake_recon_feat, real_feat_c)

        return (adv_loss, recon_loss, fm_loss)

    def gradient_penalty(self, discriminator, s, c, si):
        # Add Pertubation
        rnd = xp.random.uniform(0, 1, s.shape).astype(xp.float32)
        s_pertubed = rnd * s + (1 - rnd) * c

        _, y_pertubed = discriminator(s_pertubed, si)
        y_pertubed = F.mean(y_pertubed, axis=(1, 2)).reshape(1, 1)
        grad = chainer.grad([y_pertubed], [s_pertubed], enable_double_backprop=True)
        grad = F.sqrt(F.batch_l2_norm_squared(grad[0]))

        loss = 10 * F.mean_squared_error(grad, xp.ones_like(grad.data))

        return loss


def train(epochs, iterations, dataset_path, test_path, outdir, batchsize, testsize,
                  recon_weight, fm_weight, gp_weight, spectral_norm=False):
    # Dataset Definition
    dataloader = DatasetLoader(dataset_path, test_path)
    c_valid, s_valid = dataloader.test(testsize)

    # Model & Optimizer Definition
    if spectral_norm:
        generator = SNGenerator()
    else:
        generator = Generator()
    generator.to_gpu()
    gen_opt = set_optimizer(generator)

    discriminator = Discriminator()
    discriminator.to_gpu()
    dis_opt = set_optimizer(discriminator)

    # Loss Function Definition
    lossfunc = FUNITLossFunction()

    # Evaluator Definition
    evaluator = Evaluation()

    for epoch in range(epochs):
        sum_loss = 0
        for batch in range(0, iterations, batchsize):
            c, ci, s, si = dataloader.train(batchsize)

            y = generator(c, s)
            y.unchain_backward()

            loss = lossfunc.dis_loss(discriminator, y, s, si)
            loss += lossfunc.gradient_penalty(discriminator, s, y, si)

            discriminator.cleargrads()
            loss.backward()
            dis_opt.update()
            loss.unchain_backward()

            y_conert = generator(c, s)
            y_recon = generator(c, c)

            adv_loss, recon_loss, fm_loss = lossfunc.gen_loss(discriminator, y_conert, y_recon, s, c, si, ci)
            loss = adv_loss + recon_weight * recon_loss + fm_weight * fm_loss

            generator.cleargrads()
            loss.backward()
            gen_opt.update()
            loss.unchain_backward()

            sum_loss += loss.data

            if batch == 0:
                serializers.save_npz('generator.model', generator)
                serializers.save_npz('discriminator.model', discriminator)

                with chainer.using_config('train', False):
                    y = generator(c_valid, s_valid)
                y.unchain_backward()

                y = y.data.get()
                c = c_valid.data.get()
                s = s_valid.data.get()

                evaluator(y, c, s, outdir, epoch, testsize)

        print(f"epoch: {epoch}")
        print(f"loss: {sum_loss / iterations}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FUNIT")
    parser.add_argument('--e', type=int, default=1000, help="the number of epochs")
    parser.add_argument('--i', type=int, default=2000, help="the number of iterations")
    parser.add_argument('--b', type=int, default=1, help="batch size")
    parser.add_argument('--t', type=int, default=3, help="test size")
    parser.add_argument('--rw', type=float, default=0.1, help="the weight of reconstruction loss")
    parser.add_argument('--fw', type=float, default=1.0, help="the weight of feature matching loss")
    parser.add_argument('--gw', type=float, default=10.0, help="the weight of gradient penalty")
    parser.add_argument('--spectral', action='store_true', help="enable spectral normalization")
    args = parser.parse_args()

    outdir = Path('output')
    outdir.mkdir(exist_ok=True)

    dataset_path = Path('./Dataset/face_illustration/face_stargan')
    test_path = Path('./Dataset/face_illustration/face_test')

    train(args.e, args.i, dataset_path, test_path, outdir, args.b, args.t,
               args.rw, args.fw, args.gw, args.spectral)