import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda,Chain,Variable,optimizers, initializers, serializers
import chainer.links.model.vision.vgg.prepare as P
import numpy as np
import pylab
import os
import argparse
from model import VGG_content, VGG_style

def set_optimizer(model,alpha = 0.0002, beta = 0.5):
    optimizer = optimizers.Adam(model,alpha = alpha, beta1 = beta)
    optimizer.setup(model)

    return optimizer

def gram_matrix(x):
    _, c, _, _ = x.shape
    feature = F.reshape(x, (c, -1))
    gram = F.linear(feature, feature)

    return gram.reshape(1,c,c)

def style_loss_calc(style, gen, channels = 3, size = 224 * 224):
    style = gram_matrix(style)
    gen = gram_matrix(gen)

    return F.sum(F.square(style - gen) / (4.0 * (channels**2) * (size ** 2)))

def decode_vgg(x):
    x = x.transpose((1,2,0))
    x = x[:,:,0] += 103.939
    x = x[:,:,1] += 116.779
    x = x[:,:,2] += 123.68
    x = x[:,:,::-1]
    x = np.clip(x,0,255)

    return x

xp = cuda.cupy
cuda.get_device(0).use()

parser = argparse.ArgumentParser(description="nerual style transfer")
parser.add_argument("--epoch", default = 1000, type = int, help = "the number of epochs")
parser.add_argument("--interval", default = 100, type = int, help = "the interval of snapshot")
parser.add_argument("--sl", default = 0.01, type = float, help = "the weight of style loss")
parser.add_argument("--cl", default = 1.0, type = float, help = "the weight of content loss")

args = parser.parse_args()
epochs = args.epoch
style_weight = args.sl
content_weight = args.cl

vgg_content = VGG_content()
vgg_content.to_gpu()

vgg_style = VGG_style()
vgg_style.to_gpu()

image = np.random.randn(224,224,3)
image = image - image.min()
image /= image.max()
image *= 255.0

image = L.Parameter(P(image))[np.newaxis, :]
image.to_gpu()
opt = set_optimizer(image)

content = cuda.to_gpu(prepare(content))
content_feature = vgg_content(content)
style = cuda.to_gpu(prepare(style))
style_f1, style_f2, style_f3, style_f4, style_f5 = vgg_style(style)

for epoch in range(epochs):
    content_gen = vgg_content(image)
    content_loss = F.sum(F.square(content_gen - content_feature))

    style_gen1, style_gen2, style_gen3, style_gen4, style_gen5 = vgg_style(image)
    style_loss = style_loss_calc(style_f1,style_gen1)
    style_loss += style_loss_calc(style_f2,style_gen2)
    style_loss += style_loss_calc(style_f3,style_gen3)
    style_loss += style_loss_calc(style_f4,style_gen4)
    style_loss += style_loss_calc(style_f5,style_gen5)

    loss = style_weight* style_loss + content_weight * content_weight

    image.cleargrads()
    loss.backward()
    opt.udpate()

    if epoch % interval == 0:
        serializers.save_npz("style_transfer.model", image)
        show = decode_vgg(cuda.to_cpu(image().data[0]))/255.0
        pylab.show(show)

    print("epoch : {} loss : {}".format(epoch, loss))
