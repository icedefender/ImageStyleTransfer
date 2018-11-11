import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, optimizers, serializers
import numpy as np
import argparse
import pylab
import cv2 as cv
import os
from model import Generator,Discriminator
from prepare import prepare_dataset

xp=cuda.cupy
cuda.get_device(0).use()

def set_optimizer(model,alpha=0.0002,beta=0.5):
    optimizer=optimizers.Adam(alpha=alpha,beta1=beta)
    optimizer.setup(model)

    return optimizer

parser=argparse.ArgumentParser(description="HairStyleTransfer")
parser.add_argument("--epochs",default=1000,type=int,help="the number of epochs")
parser.add_argument("--iterations",default=2000,type=int,help="the number of iteration")
parser.add_argument("--batchsize",default=16,type=int,help="batch size")
parser.add_argument("--testsize",default=4,type=int,help="test size")
parser.add_argument("--cw",default=10.0,type=float,help="the weight of cycle loss")
parser.add_argument("--iw",default=10.0,type=float,help="the weigt of identity loss")
parser.add_argument("--Ntrain",default=900,type=int,help="the number of training images")
parser.add_argument("--interval",default=1,type=int,help="the interval of snapshot")
parser.add_argument("--size",default=128,type=int,help="image width")
parser.add_argument("--cluster",default=2,type=int,help="the number of clusters")

args=parser.parse_args()
epochs=args.epochs
iterations=args.iterations
batchsize=args.batchsize
testsize=args.testsize
cycle_weight=args.cw
identity_weight=args.iw
Ntrain=args.Ntrain
interval=args.interval
size=args.size
cluster=args.cluster

x_path="/medium/"
y_path="/twintail/"
x_list=os.listdir(x_path)
y_list=os.listdir(y_path)

test_box=[]
binary_box=[]
for _ in range(testsize):
    rnd=np.random.randint(Ntrain,Ntrain+50)
    filename=x_path+x_list[rnd]
    image,binary=prepare_dataset(filename,size,cluster)
    test_box.append(image)
    binary_box.append(binary)

test_twin=chainer.as_variable(xp.array(test_box).astype(xp.float32))
binary=chainer.as_variable(xp.array(binary_box).astype(xp.float32))
test_twin=F.concat([test_twin,binary])

outdir="./output"
if not os.path.exists(outdir):
    os.mkdir(outdir)

generator_xy=Generator()
generator_xy.to_gpu()
gen_xy_opt=set_optimizer(generator_xy)

generator_yx=Generator()
generator_yx.to_gpu()
gen_yx_opt=set_optimizer(generator_yx)

discriminator_xy=Discriminator()
discriminator_xy.to_gpu()
dis_xy_opt=set_optimizer(discriminator_xy)

discriminator_yx=Discriminator()
discriminator_yx.to_gpu()
dis_yx_opt=set_optimizer(discriminator_yx)

for epoch in range(epochs):
    sum_dis_loss=0
    sum_gen_loss=0
    for batch in range(0,iterations,batchsize):
        x_box=[]
        x_binary_box=[]
        y_box=[]
        y_binary_box=[]
        for index in range(batchsize):
            rnd=np.random.randint(1,Ntrain)
            filename=x_path+x_list[rnd]
            image,binary=prepare_dataset(filename,size,cluster)
            x_box.append(image)
            x_binary_box.append(binary)
            rnd=np.random.randint(1,Ntrain)
            filename=y_path+y_list[rnd]
            image,binary=prepare_dataset(filename,size,cluster)
            y_box.append(image)
            y_binary_box.append(binary)

        x=chainer.as_variable(xp.array(x_box).astype(xp.float32))
        x_bianry=chainer.as_variable(xp.array(x_binary_box).astype(xp.float32))
        x=F.concat([x,x_bianry])
        y=chainer.as_variable(xp.array(y_box).astype(xp.float32))
        y_bianry=chainer.as_variable(xp.array(y_binary_box).astype(xp.float32))
        y=F.concat([y,y_bianry])

        x_y=generator_xy(x)
        y_x=generator_yx(y)

        y_real=discriminator_xy(y)
        y_fake=discriminator_xy(x_y)
        x_real=discriminator_yx(x)
        x_fake=discriminator_yx(y_x)

        dis_loss_y=F.mean(F.softplus(-y_real)) + F.mean(F.softplus(y_fake))
        dis_loss_x=F.mean(F.softplus(-x_real)) + F.mean(F.softplus(x_fake))

        y_fake.unchain_backward()
        x_fake.unchain_backward()

        discriminator_xy.cleargrads()
        dis_loss_y.backward()
        dis_xy_opt.update()
        dis_loss_y.unchain_backward()

        discriminator_yx.cleargrads()
        dis_loss_x.backward()
        dis_yx_opt.update()
        dis_loss_x.unchain_backward()

        x_y=generator_xy(x)
        x_y_x=generator_yx(x_y)

        y_x=generator_yx(y)
        y_x_y=generator_xy(y_x)

        y_fake=discriminator_xy(x_y)
        x_fake=discriminator_yx(y_x)

        gen_loss=F.mean(F.softplus(-y_fake))
        gen_loss+=F.mean(F.softplus(-x_fake))

        cycle_loss_x=F.mean_absolute_error(x,x_y_x)
        cycle_loss_y=F.mean_absolute_error(y,y_x_y)

        identity_loss_x=F.mean_absolute_error(x_y,x)
        identity_loss_y=F.mean_absolute_error(y_x,y)

        gen_loss+=cycle_weight*(cycle_loss_x+cycle_loss_y) + identity_weight*(identity_loss_x+identity_loss_y)

        generator_xy.cleargrads()
        generator_yx.cleargrads()
        gen_loss.backward()
        gen_xy_opt.update()
        gen_yx_opt.update()
        gen_loss.unchain_backward()

        sum_dis_loss=dis_loss_x.data.get() + dis_loss_y.data.get()
        sum_gen_loss=gen_loss.data.get()

        if epoch%interval==0 and batch==0:
            serializers.save_npz("generator_xy.model",generator_xy)
            serializers.save_npz("generator_yx.model",generator_yx)
            with chainer.using_config("train",False):
                y=generator_xy(test_twin)
            y=y.data.get()
            test=test_twin.data.get()
            for i in range(testsize):
                tmp=np.clip(test[i][0:3]*127.5+127.5,0,255).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(testsize,4,4*i+1)
                pylab.imshow(tmp)
                pylab.axis("off")
                pylab.savefig("%s/visualize_%d.png"%(outdir,epoch))
                tmp=np.clip(test[i][3:6]*127.5+127.5,0,255).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(testsize,4,4*i+2)
                pylab.imshow(tmp)
                pylab.axis("off")
                pylab.savefig("%s/visualize_%d.png"%(outdir,epoch))
                tmp=np.clip(y[i][0:3]*127.5+127.5,0,255).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(testsize,4,4*i+3)
                pylab.imshow(tmp)
                pylab.axis("off")
                pylab.savefig("%s/visualize_%d.png"%(outdir,epoch))
                tmp=np.clip(y[i][3:6]*127.5+127.5,0,255).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(testsize,4,4*i+4)
                pylab.imshow(tmp)
                pylab.axis("off")
                pylab.savefig("%s/visualize_%d.png"%(outdir,epoch))

    print("epoch:{}".format(epoch))
    print("Discriminator loss:{}".format(sum_dis_loss/iterations))
    print("Generator loss:{}".format(sum_gen_loss/iterations))