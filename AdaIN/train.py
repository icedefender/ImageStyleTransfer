import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Chain, serializers, optimizers
import numpy as np
from model import Decoder, VGG, adain、calc_mean_std
import pylab
import os
import argparse

def set_optimizer(model, alpha=0.0002, beta=0.5):
    optimizer = optimizers.Adam(alpha = alpha, beta1 = beta)
    optimzier.setup(optimizer)

    return optimizer

def style_loss(inp, target):
    inp_std, inp_mean = calc_mean_std(inp)
    target_std, target_mean = calc_mean_std(target)

    return F.mean_squared_error(inp_std, target_std) + F.mean_squared_error(inp_mean, target_mean)

parser = argparse.ArgumentParser(description="AdaIN")
parser.add_argument("--epoch", default = 1000, type = int, help = "the number of epochs")
parser.add_argument("--batchsize", default = 8, type = int, help = "batch size")
parser.add_argument("--testsize", default = 4, type = int, help = "testsize")
parser.add_argument("--interval", default = 10, type = int, help = "the interval of snapshot")
parser.add_argument("--sw", default = 10.0, type = float, help = "the weight of the style loss")
parser.add_argument("--cw", default = 1.0, type = float, help = "the weight of the content loss")

args = parser.parse_args()
epochs = args.epoch
batchsize = args.batchsize
testsize = args.testsize
interval = args.interval
style_weight = args.sw
content_weight = args.cw

content_path = ""
style_path = ""
content_list = os.listdir(content_path)
style_list = os.listdir(style_path)

decoder = Decoder()
decoder.to_gpu()
dec_opt = set_optimizer(decoder)

vgg = VGG()
vgg.to_gpu()
vgg_opt = set_optimizer(vgg)
vgg.base.disable_update()

for epoch in range(epochs):
    sum_loss = 0
    for batch in range(0, Ntrain, batchsize):
        content_box = []
        style_box = []
        for index in range(batchsize):
            rnd = np.random.randint(Ntrain)
            filename = content_path + content_list[rnd]
            content = prepare_dataset(filename)
            filename = style_path + style_list[rnd]
            style = prepare_dataset(filename)
            content_box.append(content)
            style_box.append(style)

        content = xp.array(content_box).astype(xp.float32)
        style = xp.array(style_box).astype(xp.float32)

        content = chainer.as_variable(content)
        style = chainer.as_variable(style)

        style_feat1, style_feat2, style_feat3, style_feat4 = vgg(style)
        content_feat = vgg(content, last_only = True)

        t = adain(style_feat4, content_feat)
        t = alpha * t + (1-alpha) * content_feat

        g_t = decoder(t)
        g_t_feats1, g_t_feats2, g_t_feats3, g_t_feats4 = vgg(g_t)

        loss_content = F.mean_squared_error(g_t_feats4, t)
        loss_style = style_loss(style_feat1, g_t_feats1)
        loss_style += style_loss(style_feat2, g_t_feats2)
        loss_style += style_loss(style_feat3, g_t_feats3)
        loss_style += style_loss(style_feat4, g_t_feats4)

        loss = content_weight * loss_content + style_weight * loss_style

        decoder.cleargrads()
        vgg.cleargrads()

        loss.backward()
        loss.unchain_backward()

        dec_opt.update()
        vgg_opt.update()

        sum_loss += loss.data.get()

        if epoch % interval == 0 and batch == 0:
            serializers.save_npz("decoder.model", decoder)

    print("epoch : {}".format(epoch))
    print("Loss : {}".format(sum_loss/Ntrain))