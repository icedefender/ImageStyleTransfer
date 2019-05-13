import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Chain, optimizers, initializers, serializers
import os
import numpy as np
import argparse
import pylab
from model import Discriminator, Generator, VGG
from prepare import prepare_trim_content, prepare_trim_style

xp = cuda.cupy
cuda.get_device(0).use()

def set_optimizer(model, alpha=0.0002, beta=0.5, beta2 = 0.999):
    optimizer = optimizers.Adam(alpha=alpha, beta1 = beta, beta2 = 0.999)
    optimizer.setup(model)

    return optimizer

def calc_loss(fake, real):
    sum_loss = 0
    for f, r in zip(fake, real):
        sum_loss += F.mean_absolute_error(f, r)

    return sum_loss

outdir = "./output_pretrain256/"
if not os.path.exists(outdir):
    os.mkdir(outdir)

model_dir = "./model/"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

parser = argparse.ArgumentParser(description="cartoongan")
parser.add_argument("--epochs",default=1000,type=int,help="the number of epochs")
parser.add_argument("--batchsize",default=8,type=int,help="batchsize")
parser.add_argument("--interval",default=1,type=int,help="the interval of snapshot")
parser.add_argument("--cw",default=10.0,type=float,help="the weights of content loss")
parser.add_argument("--Ntrain", default=37000, type = int, help = "Ntrain")
parser.add_argument("--Nstyle",default = 1990, type = int, help = "Nstyle")
parser.add_argument("--testsize", default = 4, type = int, help = "testsize")

args=parser.parse_args()
epochs = args.epochs
batchsize = args.batchsize
interval = args.interval
content_weight = args.cw
Ntrain = args.Ntrain
Nstyle = args.Nstyle
testsize = args.testsize

image_path = "./Dataset/coco/test2015/"
image_list = os.listdir(image_path)
style_path = "./Dataset/background/bg/"
smooth_path = "./Dataset/edge_smooth/background/"

test_box = []
for _ in range(testsize):
    rnd = np.random.randint(Ntrain, Ntrain + 1000)
    test_name = image_path + image_list[rnd]
    test,_,_ = prepare_trim_content(test_name)
    test_box.append(test)

x_test = xp.array(test_box).astype(xp.float32)
x_test = chainer.as_variable(x_test)

generator = Generator()
generator.to_gpu()
gen_opt = set_optimizer(generator)

vgg = VGG()
vgg.to_gpu()
vgg_opt = set_optimizer(vgg)
vgg.base.disable_update()

for epoch in range(epochs):
    sum_gen_loss = 0
    sum_dis_loss = 0
    for batch in range(0, 2000, batchsize):
        image_box = []
        style_box = []
        for i in range(batchsize):
            rnd1 = np.random.randint(Ntrain)
            image_name = image_path + image_list[rnd1]
            image,_,_ = prepare_trim_content(image_name)
            image_box.append(image)

        x = xp.array(image_box).astype(xp.float32)
        x = chainer.as_variable(x)

        y = generator(x)
        vgg_x = vgg(x)
        vgg_y = vgg(y)
        content_loss = calc_loss(vgg_x , vgg_y)
        gen_loss = content_weight * content_loss

        vgg.cleargrads()
        generator.cleargrads()
        gen_loss.backward()
        gen_opt.update()
        vgg_opt.update()
        gen_loss.unchain_backward()

        sum_gen_loss += gen_loss.data.get()

        if epoch % interval == 0 and batch == 0:
            serializers.save_npz("./model/generator_pretrain_{}.model".format(epoch), generator)
            with chainer.using_config("train", False):
                y_test = generator(x_test)
            y_test = y_test.data.get()
            x_t = x_test.data.get()
            for i in range(testsize):
                tmp = (np.clip(x_t[i]*127.5 + 127.5, 0, 255)).transpose(1, 2, 0).astype(np.uint8)
                pylab.subplot(testsize, 2, 2*i+1)
                pylab.imshow(tmp)
                pylab.axis("off")
                pylab.savefig('%s/visualize_%d.png'%(outdir, epoch))
                tmp = (np.clip(y_test[i]*127.5 + 127.5, 0, 255)).transpose(1, 2, 0).astype(np.uint8)
                pylab.subplot(testsize, 2, 2*i+2)
                pylab.imshow(tmp)
                pylab.axis("off")
                pylab.savefig('%s/visualize_%d.png'%(outdir, epoch))

    print("epoch : {}".format(epoch))
    print("Generator : {} Discriminator : {}".format(sum_gen_loss / Ntrain, sum_dis_loss / Ntrain))