import chainer
import chainer.functions as F
import argparse
import os
import pylab
import cv2
import copy
import numpy as np

from chainer import cuda, optimizers, serializers
from model import Discriminator, Generator

xp = cuda.cupy
cuda.get_device(0).use()


def prepare(filename):
    img = cv2.imread(filename)
    img = cv2.resize(img, (128, 128))
    img = img[:, :, ::-1]
    img = img.transpose(2, 0, 1)
    img = (img - 127.5) / 127.5

    return img


def set_optimizer(model, alpha=0.0001):
    optimizer = optimizers.RMSprop(lr=alpha)
    optimizer.setup(model)

    return optimizer


def hinge_loss(fake, real=None):
    if real is not None:
        return F.mean(F.relu(1 - real)) + F.mean(F.relu(1 + fake))

    else:
        return -F.mean(fake)


def making_onehot(color):
    cls_list_copy = copy.copy(cls_list)
    cls_list_copy.remove(color)
    
    return cls_list_copy


def making_index(color):
    if color == 'black':
        return 0
    elif color == 'white':
        return 1
    elif color == 'blue':
        return 2
    elif color == 'pink':
        return 3
    else:
        return 4

parser = argparse.ArgumentParser(description='FUNIT')
parser.add_argument('--e', default=1000, type=int, help="the number of epochs")
parser.add_argument('--b', default=8, type=int, help="batch size")
parser.add_argument('--t', default=3, type=int, help="test size")
parser.add_argument('--rw', default=0.1, type=float, help="the weight of reconstruction loss")
parser.add_argument('--fw', default=1.0, type=float, help="the weight of feature matching loss")
parser.add_argument('--gw', default=10.0, type=float, help="the weight of gradient penalty")

args = parser.parse_args()
epochs = args.e
batchsize = args.b
testsize = args.t
reconstruction_weight = args.rw
fmatching_weight = args.fw
gp_weight = args.gw

outdir = './output'
if not os.path.exists(outdir):
    os.mkdir(outdir)

generator = Generator()
generator.to_gpu()
gen_opt = set_optimizer(generator)

discriminator = Discriminator()
discriminator.to_gpu()
dis_opt = set_optimizer(discriminator)

img_path = './Dataset/face_illustration/face_stargan/'
img_list = os.listdir(img_path)
img_len = len(img_list)

test_path = './Dataset/face_illustration/face_test/'
test_list = os.listdir(test_path)

cls_list = ['black', 'white', 'blue', 'pink', 'gold']

c_test_box = []
s_test_box = []
for index in range(testsize):
    content_color = np.random.choice(cls_list)
    dir_path = img_path + content_color + '/'
    dir_list = os.listdir(dir_path)
    rnd = np.random.randint(len(dir_list))
    img_name = dir_path + dir_list[rnd]
    img = prepare(img_name)
    c_test_box.append(img)

    img_name = test_path + test_list[index]
    img = prepare(img_name)
    s_test_box.append(img)

c_test = chainer.as_variable(xp.array(c_test_box).astype(xp.float32))
s_test = chainer.as_variable(xp.array(s_test_box).astype(xp.float32))

for epoch in range(epochs):
    sum_dis_loss = 0
    sum_gen_loss = 0

    for batch in range(0, 5000, batchsize):
        c_box = []
        c_index = []
        s_box = []
        s_index = []
        for _ in range(batchsize):
            content_color = np.random.choice(cls_list)
            dir_path = img_path + content_color + '/'
            dir_list = os.listdir(dir_path)
            rnd = np.random.randint(len(dir_list))
            img_name = dir_path + dir_list[rnd]
            index = making_index(content_color)
            img = prepare(img_name)
            c_box.append(img)
            c_index.append(index)

            style_list = making_onehot(str(content_color))
            style_color = np.random.choice(style_list)
            dir_path = img_path + style_color + '/'
            dir_list = os.listdir(dir_path)
            rnd = np.random.randint(len(dir_list))
            img_name = dir_path + dir_list[rnd]
            index = making_index(style_color)
            img = prepare(img_name)
            s_box.append(img)
            s_index.append(index)

        c = chainer.as_variable(xp.array(c_box).astype(xp.float32))
        s = chainer.as_variable(xp.array(s_box).astype(xp.float32))

        c_i = chainer.as_variable(xp.array(c_index).astype(xp.int32))
        s_i = chainer.as_variable(xp.array(s_index).astype(xp.int32))

        y = generator(c, s)
        y.unchain_backward()

        _, fake, fake_cls = discriminator(y)
        _, real, real_cls = discriminator(c)
        dis_loss = hinge_loss(fake, real)
        dis_loss += F.softmax_cross_entropy(real_cls, c_i)
        dis_loss += F.softmax_cross_entropy(fake_cls, s_i)

        #eps = xp.random.uniform(0,1,size = batchsize).astype(xp.float32)[:,None,None,None]
        #x_mid = eps * real + (1.0 - eps) * fake
        #_, y_mid, _ = discriminator(x_mid)

        #grad,  = chainer.grad([y_mid], [x_mid], enable_double_backprop=True)
        #grad = F.sqrt(F.sum(grad*grad, axis=(1, 2, 3)))
        #loss_gp = gp_weight * F.mean_squared_error(grad, xp.ones_like(grad.data))

        #dis_loss += loss_gp

        discriminator.cleargrads()
        dis_loss.backward()
        dis_opt.update()
        dis_loss.unchain_backward()

        y = generator(c, s)
        y_recon = generator(c, c)
        fake_feat, fake, fake_cls = discriminator(y)
        real_feat, _, _ = discriminator(s)
        
        gen_loss = hinge_loss(fake)
        gen_loss += F.softmax_cross_entropy(fake_cls, s_i)
        recon_loss = F.mean_absolute_error(c, y_recon)
        fm_loss = F.mean_absolute_error(fake_feat, real_feat)

        gen_loss += reconstruction_weight * recon_loss + fmatching_weight * fm_loss

        generator.cleargrads()
        gen_loss.backward()
        gen_opt.update()
        gen_loss.unchain_backward()

        sum_dis_loss += dis_loss.data
        sum_gen_loss += gen_loss.data

        if batch == 0:
            pylab.rcParams['figure.figsize'] = (16.0,16.0)
            pylab.clf()

            serializers.save_npz('./generator.model', generator)
            serializers.save_npz('./discriminator.model', discriminator)

            with chainer.using_config('train', False):
                y = generator(c_test, s_test)
            y.unchain_backward()
            y = y.data.get()
            c = c_test.data.get()
            s = s_test.data.get()

            for i in range(testsize):
                tmp=(np.clip(c[i]*127.5 + 127.5, 0, 255)).transpose(1, 2, 0).astype(np.uint8)
                pylab.subplot(testsize,3,3*i+1)
                pylab.imshow(tmp)
                pylab.axis("off")
                pylab.savefig("%s/visualize_%d.png"%(outdir, epoch))
                tmp=np.clip(s[i]*127.5+127.5, 0, 255).transpose(1, 2, 0).astype(np.uint8)
                pylab.subplot(testsize, 3, 3*i+2)
                pylab.imshow(tmp)
                pylab.axis("off")
                pylab.savefig("%s/visualize_%d.png"%(outdir, epoch))
                tmp=np.clip(y[i]*127.5 + 127.5, 0, 255).transpose(1, 2, 0).astype(np.uint8)
                pylab.subplot(testsize, 3, 3*i+3)
                pylab.imshow(tmp)
                pylab.axis("off")
                pylab.savefig("%s/visualize_%d.png"%(outdir, epoch))

    print('epoch : {}'.format(epoch))
    print('Discriminator loss: {}'.format(sum_dis_loss/5000))
    print('Generator loss: {}'.format(sum_gen_loss/5000))