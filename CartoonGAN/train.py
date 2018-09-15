import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Chain, optimizers, initializers, serializers
import os
import numpy as np
import argparse
from model import Discriminator, Generator, VGG

xp = cuda.cupy
cuda.get_device(0).use()

def set_optimzier(mode, alpha=0.0002, beta=0.5):
    optimizer = optimizers.Adam(alpha=alpha, beta1 = beta)
    optimizer.setup(model)

    return optimizer

outdir = "./output/"
if not os.path.exists(outdir):
    os.mkdir(outdir)

parser = argparse.ArgumentParser(description="cartoongan")
parser.add_argument("--epochs",default=1000,type=int,help="the number of epochs")
parser.add_argument("--batchsize",default=16,type=int,help="batchsize")
parser.add_argument("--interval",default=5,type=int,help="the interval of snapshot")
parser.add_argument("--cw",default=10.0,type=float,help="the weights of content loss")

args=parser.parse_args()
epochs = args.epochs
batchsize = args.batchsize
interval = args.interval
content_weight = args.cw

generator = Generator()
generator.to_gpu()
gen_opt = set_optimizer(generator)

discriminator = Discriminator()
discriminator.to_gpu()
dis_opt = set_optimizer(discriminator)

vgg = VGG()
vgg.to_gpu()
vgg_opt.update()
vgg.base.disable_update()

for epoch in range(epochs):
    sum_gen_loss = 0
    sum_dis_loss = 0
    for batch in range(0,Ntrain,batchsize):
        image_box = []
        style_box = []
        smooth_box = []
        for i in range(batchsize):
        
        x = xp.array(image_box).astype(xp.float32)
        t = xp.array(style_box).astype(xp.float32)
        s = xp.array(smooth_box).astype(xp.float32)

        x = chainer.as_variable(x)
        t = chainer.as_variable(t)
        s = chainer.as_variable(s)

        y = generator(x)
        y_dis = discriminator(y)
        s_dis = discriminator(s)
        dis_loss = F.mean(F.softplus(y_dis)) + F.mean(F.softplus(s_dis))

        y.unchain_backward()

        t_dis = discriminator(t)
        dis_loss += F.mean(F.softplus(-t_dis))

        discriminator.cleargrads()
        dis_loss.backward()
        dis_opt.update()
        dis_loss.unchain_backward()

        y = generator(x)
        y_dis = discriminator(y)
        s_dis = discriminator(s)
        gen_loss = F.mean(F.softplus(-y_dis)) + F.mean(F.softplus(-s_dis))
        content_loss = F.mean_absolute_error(vgg(t) - vgg(y))
        gen_loss += content_weight * content_loss

        generator.cleargrads()
        gen_loss.backward()
        gen_opt.update()
        gen_loss.unchain_backward()

        if epoch % interval == 0 and batch == 0:
            serializers.save_npz("generator.model", generator)
            with chainer.using_config("train", False):
                y_test = generator(x_test)
            y_test = y_test.data.get()
