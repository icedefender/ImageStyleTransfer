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
parser.add_argument("--Ntrain",default=850,type=int,help="the number of training images")
parser.add_argument("--interval",default=5,type=int,help="the interval of snapshot")
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
x_tag_path="/label/medium/"
y_tag_path="/label/twin/"
x_list=os.listdir(x_tag_path)
y_list=os.listdir(y_tag_path)
Nx = len(x_list) - 50
Ny = len(y_list)

test_box=[]
binary_box=[]
for _ in range(testsize):
    rnd=np.random.randint(Nx,Nx+50)
    filename=x_tag_path+x_list[rnd]
    binary=prepare_dataset(filename,size,cluster)
    filename=x_path+x_list[rnd]
    image=prepare_dataset(filename,size,cluster)
    test_box.append(image)
    binary_box.append(binary)

test_img=chainer.as_variable(xp.array(test_box).astype(xp.float32))
test_mask=chainer.as_variable(xp.array(binary_box).astype(xp.float32))

outdir="./output"
if not os.path.exists(outdir):
    os.mkdir(outdir)

generator_xy = Generator()
generator_xy.to_gpu()
gen_xy_opt = set_optimizer(generator_xy)

generator_yx = Generator()
generator_yx.to_gpu()
gen_yx_opt = set_optimizer(generator_yx)

discriminator_xy = Discriminator()
discriminator_xy.to_gpu()
dis_xy_opt = set_optimizer(discriminator_xy)

discriminator_yx = Discriminator()
discriminator_yx.to_gpu()
dis_yx_opt = set_optimizer(discriminator_yx)

for epoch in range(epochs):
    sum_dis_loss=0
    sum_gen_loss=0
    for batch in range(0,iterations,batchsize):
        x_box=[]
        x_binary_box=[]
        y_box=[]
        y_binary_box=[]
        for index in range(batchsize):
            rnd=np.random.randint(1,Nx)
            aug=np.random.randint(2)
            filename=x_tag_path+x_list[rnd]
            binary=prepare_dataset(filename,size,aug)
            filename=x_path+x_list[rnd]
            image=prepare_dataset(filename,size,aug)
            x_box.append(image)
            x_binary_box.append(binary)
            rnd=np.random.randint(1,Ny)
            aug=np.random.randint(2)
            filename=y_tag_path+y_list[rnd]
            binary=prepare_dataset(filename,size,aug)
            filename=y_path+y_list[rnd]
            image=prepare_dataset(filename,size,aug)
            y_box.append(image)
            y_binary_box.append(binary)

        x_img=chainer.as_variable(xp.array(x_box).astype(xp.float32))
        x_mask=chainer.as_variable(xp.array(x_binary_box).astype(xp.float32))
        y_img=chainer.as_variable(xp.array(y_box).astype(xp.float32))
        y_mask=chainer.as_variable(xp.array(y_binary_box).astype(xp.float32))

        x_y_img, x_y_mask=generator_xy(x_img, x_mask)
        y_x_img, y_x_mask=generator_yx(y_img, y_mask)

        y_real=discriminator_xy(y_img, y_mask)
        y_fake=discriminator_xy(x_y_img, x_y_mask)
        x_real=discriminator_yx(x_img, x_mask)
        x_fake=discriminator_yx(y_x_img, y_x_mask)

        dis_loss_y=F.mean(F.softplus(-y_real)) + F.mean(F.softplus(y_fake))
        dis_loss_x=F.mean(F.softplus(-x_real)) + F.mean(F.softplus(x_fake))

        x_y_img.unchain_backward()
        x_y_mask.unchain_backward()
        y_x_img.unchain_backward()
        y_x_mask.unchain_backward()

        discriminator_xy.cleargrads()
        dis_loss_y.backward()
        dis_xy_opt.update()
        dis_loss_y.unchain_backward()

        discriminator_yx.cleargrads()
        dis_loss_x.backward()
        dis_yx_opt.update()
        dis_loss_x.unchain_backward()

        x_y_img, x_y_mask=generator_xy(x_img, x_mask)
        x_y_x_img, x_y_x_mask=generator_yx(x_y_img, x_y_mask)

        y_x_img, y_x_mask=generator_yx(y_img, y_mask)
        y_x_y_img, y_x_y_mask=generator_xy(y_x_img, y_x_mask)

        y_fake=discriminator_xy(x_y_img, x_y_mask)
        x_fake=discriminator_yx(y_x_img, y_x_mask)

        gen_loss=F.mean(F.softplus(-y_fake))
        gen_loss+=F.mean(F.softplus(-x_fake))

        cycle_loss_x=F.mean_absolute_error(x_img,x_y_x_img) + F.mean_absolute_error(x_mask, x_y_x_mask)
        cycle_loss_y=F.mean_absolute_error(y_img,y_x_y_img) + F.mean_absolute_error(y_mask, y_x_y_mask)

        x_idt_img, x_idt_mask = generator_yx(x_img, x_mask)
        y_idt_img, y_idt_mask = generator_xy(y_img, y_mask)

        identity_loss_x=F.mean_absolute_error(x_idt_img,x_img) + F.mean_absolute_error(x_idt_mask, x_mask)
        identity_loss_y=F.mean_absolute_error(y_idt_img,y_img) + F.mean_absolute_error(y_idt_mask, y_mask)

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
            pylab.rcParams['figure.figsize'] = (16.0,16.0)
            pylab.clf()

            serializers.save_npz("generator_xy.model",generator_xy)
            serializers.save_npz("generator_yx.model",generator_yx)

            with chainer.using_config("train",False):
                for i in range(testsize):
                    y_img, y_mask=generator_xy(test_img[i].reshape(1,3,128,128), test_mask[i].reshape(1,3,128,128))
                    y_img_t=y_img.data.get()
                    y_mask_t=y_mask.data.get()
                    t_img=test_img[i].data.get()
                    t_mask=test_mask[i].data.get()

                    y_img.unchain_backward()
                    y_mask.unchain_backward()

                    tmp=(np.clip(t_img*127.5+127.5,0,255)).transpose(1,2,0).astype(np.uint8)
                    pylab.subplot(testsize,4,4*i+1)
                    pylab.imshow(tmp)
                    pylab.axis("off")
                    pylab.savefig("%s/visualize_%d.png"%(outdir,epoch))
                    tmp=(np.clip(t_mask*127.5+127.5,0,255)).transpose(1,2,0).astype(np.uint8)
                    pylab.subplot(testsize,4,4*i+2)
                    pylab.imshow(tmp)
                    pylab.axis("off")
                    pylab.savefig("%s/visualize_%d.png"%(outdir,epoch))
                    tmp=np.clip(y_img_t[0]*127.5+127.5,0,255).transpose(1,2,0).astype(np.uint8)
                    pylab.subplot(testsize,4,4*i+3)
                    pylab.imshow(tmp)
                    pylab.axis("off")
                    pylab.savefig("%s/visualize_%d.png"%(outdir,epoch))
                    tmp=np.clip(y_mask_t[0]*127.5+127.5,0,255).transpose(1,2,0).astype(np.uint8)
                    pylab.subplot(testsize,4,4*i+4)
                    pylab.imshow(tmp)
                    pylab.axis("off")
                    pylab.savefig("%s/visualize_%d.png"%(outdir,epoch))

    print("epoch:{}".format(epoch))
    print("Discriminator loss:{}".format(sum_dis_loss/iterations))
    print("Generator loss:{}".format(sum_gen_loss/iterations))