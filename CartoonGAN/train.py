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

outdir = "./output/"
if not os.path.exists(outdir):
    os.mkdir(outdir)

parser = argparse.ArgumentParser(description="cartoongan")
parser.add_argument("--epochs",default=1000,type=int,help="the number of epochs")
parser.add_argument("--batchsize",default=3,type=int,help="batchsize")
parser.add_argument("--interval",default=1,type=int,help="the interval of snapshot")
parser.add_argument("--cw",default=10.0,type=float,help="the weights of content loss")
parser.add_argument("--Ntrain", default=20000, type = int, help = "Ntrain")
parser.add_argument("--Nstyle",default = 4500, type = int, help = "Nstyle")
parser.add_argument("--testsize", default = 2, type = int, help = "testsize")

args=parser.parse_args()
epochs = args.epochs
batchsize = args.batchsize
interval = args.interval
content_weight = args.cw
Ntrain = args.Ntrain
Nstyle = args.Nstyle
testsize = args.testsize

image_path = "/usr/MachineLearning/Dataset/trim/coco_trim/"
image_list = os.listdir(image_path)
style_path = "/usr/MachineLearning/Dataset/macross/"
smooth_path = "/usr/MachineLearning/Dataset/edge_smooth/"

test_box = []
for _ in range(testsize):
    rnd = np.random.randint(Ntrain, Ntrain + 1000)
    test_name = image_path + image_list[rnd]
    test,_,_ = prepare_trim_content(test_name)
    test_box.append(test)

x_test = xp.array(test_box).astype(xp.float32)
x_test = chainer.as_variable(x_test)

generator = Generator()
serializers.load_npz("./model/generator_pretrain_2.model", generator)
generator.to_gpu()
gen_opt = set_optimizer(generator)

discriminator = Discriminator()
discriminator.to_gpu()
dis_opt = set_optimizer(discriminator)

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
        smooth_box = []
        for i in range(batchsize):
            rnd1 = np.random.randint(Ntrain)
            image_name = image_path + image_list[rnd1]
            image,_,_ = prepare_trim_content(image_name)
            image_box.append(image)

            rnd2 = np.random.randint(Nstyle)
            style_name = style_path + "macross_" + str(rnd2) + ".png"
            style,width,height = prepare_trim_content(style_name)
            style_box.append(style)

            smooth_name = smooth_path + "edge_smooth" + str(rnd2) + ".png"
            smooth_edge = prepare_trim_style(smooth_name,width,height)
            smooth_box.append(smooth_edge)
        
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
        gen_loss = F.mean(F.softplus(-y_dis))
        content_loss = F.mean_absolute_error(vgg(x) , vgg(y))
        gen_loss += content_weight * content_loss

        generator.cleargrads()
        gen_loss.backward()
        gen_opt.update()
        gen_loss.unchain_backward()

        sum_dis_loss += dis_loss.data.get()
        sum_gen_loss += gen_loss.data.get()

        if epoch % interval == 0 and batch == 0:
            serializers.save_npz("generator.model", generator)
            with chainer.using_config("train", False):
                y_test = generator(x_test)
            y_test = y_test.data.get()
            x_t = x_test.data.get()
            for i in range(testsize):
                tmp = (np.clip(x_t[i]*127.5 + 127.5,0,255)).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(testsize, 2, 2*i+1)
                pylab.imshow(tmp)
                pylab.axis("off")
                pylab.savefig('%s/visualize_%d.png'%(outdir, epoch))
                tmp = (np.clip(y_test[i]*127.5 + 127.5,0,255)).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(testsize, 2, 2*i+2)
                pylab.imshow(tmp)
                pylab.axis("off")
                pylab.savefig('%s/visualize_%d.png'%(outdir, epoch))

    print("epoch : {}".format(epoch))
    print("Generator : {} Discriminator : {}".format(sum_gen_loss / Ntrain, sum_dis_loss / Ntrain))