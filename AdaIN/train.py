import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Chain, serializers, optimizers
import numpy as np
from model import Decoder, VGG, adain,calc_mean_std
from prepare import prepare_dataset, prepare_trim, prepare_trim_full, prepare_trim_test
import pylab
import os
import argparse

xp = cuda.cupy
cuda.get_device(0).use()

def set_optimizer(model, alpha=0.0001, beta=0.9):
    optimizer = optimizers.Adam(alpha = alpha, beta1 = beta)
    optimizer.setup(model)

    return optimizer

def style_loss(inp, target):
    inp_std, inp_mean = calc_mean_std(inp)
    target_std, target_mean = calc_mean_std(target)

    return F.mean_squared_error(inp_std, target_std) + F.mean_squared_error(inp_mean, target_mean)

def prepare_test(image_path, image_list, N):
    test_list = []
    for i in range(testsize):
        rnd = np.random.randint(N+1, N+200)
        filename = image_path + image_list[rnd]
        image = prepare_trim_test(filename)
        test_list.append(image)
    
    return xp.array(test_list).astype(xp.float32)

image_out_dir = "./output"
if not os.path.exists(image_out_dir):
    os.mkdir(image_out_dir)

parser = argparse.ArgumentParser(description="AdaIN")
parser.add_argument("--epoch", default = 1000, type = int, help = "the number of epochs")
parser.add_argument("--batchsize", default = 3, type = int, help = "batch size")
parser.add_argument("--testsize", default = 2, type = int, help = "testsize")
parser.add_argument("--interval", default = 1, type = int, help = "the interval of snapshot")
parser.add_argument("--Ncontent", default = 24000, type = int, help="the numer of content images")
parser.add_argument("--Nstyle", default = 21000, type = int, help = "the number of style images")
parser.add_argument("--sw", default = 10.0, type = float, help = "the weight of the style loss")
parser.add_argument("--cw", default = 1.0, type = float, help = "the weight of the content loss")
parser.add_argument("--alpha", default = 1.0, type = float, help = "the value of alpha")

args = parser.parse_args()
epochs = args.epoch
batchsize = args.batchsize
testsize = args.testsize
interval = args.interval
style_weight = args.sw
content_weight = args.cw
Ncontent = args.Ncontent
Nstyle = args.Nstyle
alpha = args.alpha

content_path = "./coco/"
style_path = "./wiki/"
content_list = os.listdir(content_path)
style_list = os.listdir(style_path)
content_test = prepare_test(content_path, content_list, Ncontent)
content_test = chainer.as_variable(content_test)
style_test = prepare_test(style_path, style_list, Nstyle)
style_test = chainer.as_variable(style_test)

decoder = Decoder()
decoder.to_gpu()
dec_opt = set_optimizer(decoder)

vgg = VGG()
vgg.to_gpu()
vgg_opt = set_optimizer(vgg)
vgg.base.disable_update()

for epoch in range(epochs):
    sum_loss = 0
    for batch in range(0, 3000, batchsize):
        content_box = []
        style_box = []
        for index in range(batchsize):
            rnd1 = np.random.randint(Ncontent)
            rnd2 = np.random.randint(Nstyle)
            filename = content_path + content_list[rnd1]
            content = prepare_trim(filename)
            filename = style_path + style_list[rnd2]
            style = prepare_trim(filename)
            content_box.append(content)
            style_box.append(style)

        content = xp.array(content_box).astype(xp.float32)
        style = xp.array(style_box).astype(xp.float32)

        content = chainer.as_variable(content)
        style = chainer.as_variable(style)

        style_feat1, style_feat2, style_feat3, style_feat4 = vgg(style)
        content_feat = vgg(content, last_only = True)

        t = adain(content_feat, style_feat4)
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
            with chainer.using_config("train", False):
                style_feat1, style_feat2, style_feat3, style_feat4 = vgg(style_test)
                content_feat = vgg(content_test, last_only = True)

                t = adain(content_feat, style_feat4)
                t = alpha * t + (1-alpha) * content_feat
                g_t = decoder(t)

            g_t = g_t.data.get()
            for i_ in range(testsize):
                con = content_test.data.get()
                con = (np.clip((con[i_,:,:,:])*127.5 + 127.5, 0, 255)).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(testsize,3,3*i_+1)
                pylab.imshow(con)
                pylab.axis('off')
                pylab.savefig('%s/visualize_%d.png'%(image_out_dir, epoch))
                sty = style_test.data.get()
                sty = (np.clip((sty[i_,:,:,:])*127.5 + 127.5, 0, 255)).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(testsize,3,3*i_+2)
                pylab.imshow(sty)
                pylab.axis('off')
                pylab.savefig('%s/visualize_%d.png'%(image_out_dir, epoch))
                tmp = (np.clip((g_t[i_,:,:,:])*127.5 + 127.5, 0, 255)).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(testsize,3,3*i_+3)
                pylab.imshow(tmp)
                pylab.axis('off')
                pylab.savefig('%s/visualize_%d.png'%(image_out_dir, epoch))

    print("epoch : {}".format(epoch))
    print("Loss : {}".format(sum_loss/Ncontent))