import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain, cuda, initializers, optimizers, Variable, serializers
import os
import numpy as np
import pylab
import argparse
from model import VGG, Decoder_g1, Decoder_g2, UNet, Discriminator, Grayscale
from prepare  import prepare_dataset, prepare_dataset_vgg

def set_optimizer(model, alpha = 0.0002, beta = 0.5):
    optimizer = optimizers.Adam(alpha = alpha, beta1 = beta)
    optimizer.setup(model)

    return optimizer

def BCE(x,t):
    return F.average(x - x * t + F.softplus(-x))

xp = cuda.cupy
cuda.get_device(0).use()

out_dir = "./output/"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

parser = argparse.ArgumentParser(description="anime style transfer")
parser.add_argument("--epoch", default=1000, help = "the number of epochs")
parser.add_argument("--batchsize", default = 2, type = int,help = "batch size")
parser.add_argument("--testsize", default  = 4, help = "test size")
parser.add_argument("--lam1", default = 10.0, type = float, help = "weight of the content loss")
parser.add_argument("--lamg1", default = 3.0, type = float, help = "weight of the g1 content loss")
parser.add_argument("--lamg2", default = 9.0, type = float, help = "weight of the g2 loss")
parser.add_argument("--interval", default = 10, help = "interval of the snapshot")
args = parser.parse_args()

epochs = args.epoch
batchsize = args.batchsize
testsize = args.testsize
lambda1 = args.lam1
lambda_g1 = args.lamg1
lambda_g2 = args.lamg2
interval = args.interval

#x_train = np.load("../Pix2pix/line.npy").astype(np.float32)
#t_train = np.load("../Pix2pix/trim.npy").astype(np.float32)
#Ntrain, channels, width, height = x_train.shape
line_path = "/usr/MachineLearning/Dataset/line/train_lin/"
color_path = "/usr/MachineLearning/Dataset/trim/train_trim/"
Ntrain = 5000
channels = 3
width = 128
height = 128

vgg = VGG()
vgg.to_gpu()
vgg_opt = set_optimizer(vgg)
vgg.base.disable_update()

decoder_g1 = Decoder_g1(in_ch = 256, out_ch = 1)
decoder_g1.to_gpu()
g1_opt = set_optimizer(decoder_g1)

decoder_g2 = Decoder_g2(in_ch = 512, out_ch = 3)
decoder_g2.to_gpu()
g2_opt = set_optimizer(decoder_g2)

unet = UNet()
unet.to_gpu()
unet_opt = set_optimizer(unet)

discriminator = Discriminator()
discriminator.to_gpu()
dis_opt = set_optimizer(discriminator)

gray = Grayscale()
gray.to_gpu()

for epoch in range(epochs):
    sum_unet_loss = 0
    sum_dis_loss = 0
    for batch in range(0, Ntrain, batchsize):
        x = xp.zeros((batchsize, channels, width, height), dtype = xp.float32)
        t = xp.zeros((batchsize, channels, width, height), dtype = xp.float32)
        line_box = []
        color_box = []
        color_vgg_box = []

        for index in range(batchsize):
            rnd = np.random.randint(Ntrain)
            line_name =  "trim_free_" + str(rnd) + ".png"
            while line_name == "trim_free_0.png":
                rnd = np.random.randint(Ntrain)
                line_name =  "trim_free_" + str(rnd) + ".png"
            line = prepare_dataset(line_path + line_name)
            line_box.append(line)

            color_name =  "trim_free_" + str(rnd) + ".png"
            color = prepare_dataset(color_path + color_name)
            color_box.append(color)

            color_vgg_name =  "trim_free_" + str(rnd) + ".png"
            color_vgg = prepare_dataset_vgg(color_path + color_vgg_name)
            color_vgg_box.append(color_vgg)

        x = xp.array(line_box).astype(xp.float32)
        t = xp.array(color_box).astype(xp.float32)
        t_vgg = xp.array(color_vgg_box).astype(xp.float32)

        x = chainer.as_variable(x)
        t = chainer.as_variable(t)
        t_vgg = chainer.as_variable(t_vgg)

        z_tag, z = vgg(t_vgg)

        y, _, _ = unet(x,z)

        y_cls = discriminator(y)
        t_cls = discriminator(t)
        t_cls = t_cls + xp.ones_like(t_cls) - z_tag

        fake_loss = F.sum(F.softplus(y_cls)) / batchsize
        real_loss = F.sum(F.softplus(-t_cls)) / batchsize
        loss_dis = fake_loss + real_loss

        discriminator.cleargrads()
        vgg.cleargrads()
        loss_dis.backward()
        dis_opt.update()
        vgg_opt.update()
        loss_dis.unchain_backward()

        _, z = vgg(t_vgg)

        y, g1, g2 = unet(x,z)
        g1 = decoder_g1(g1)
        g2 = decoder_g2(g2)
        y_cls = discriminator(y)

        dis_loss = F.sum(F.softplus(-y_cls)) / batchsize
        content_loss = F.mean_absolute_error(y,t)
        g1_loss = F.mean_absolute_error(g1, gray(t))
        g2_loss = F.mean_absolute_error(g2, t)

        l1_loss = lambda1 * content_loss + lambda_g1 * g1_loss + lambda_g2 * g2_loss
        unet_loss = l1_loss + dis_loss

        unet.cleargrads()
        vgg.cleargrads()
        decoder_g1.cleargrads()
        decoder_g2.cleargrads()

        unet_loss.backward()
        unet_opt.update()
        g1_opt.update()
        g2_opt.update()
        vgg_opt.update()
        unet_loss.unchain_backward()

        sum_dis_loss += loss_dis.data.get()
        sum_unet_loss += unet_loss.data.get()

    serializers.save_npz("unet.model", unet)
    print("epoch : {}".format(epoch))
    print("UNet loss : {}".format(sum_unet_loss / Ntrain))
    print("Discriminator loss : {}".format(sum_dis_loss / Ntrain))