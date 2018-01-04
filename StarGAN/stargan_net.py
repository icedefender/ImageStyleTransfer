import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import math
from chainer import Variable, cuda, Chain
from instance_normalization import InstanceNormalization

xp = cuda.cupy

class Generator_ResBlock6(Chain):
    def __init__(self, nc_size):
        self.nc_size = nc_size
        w1 = chainer.initializers.Uniform(scale = math.sqrt(1/(3+nc_size)/7/7))
        w2 = chainer.initializers.Uniform(scale = math.sqrt(1/32/4/4))
        w3 = chainer.initializers.Uniform(scale = math.sqrt(1/64/4/4))
        w_res = chainer.initializers.Uniform(scale = math.sqrt(1/128/3/3))
        w10 = chainer.initializers.Uniform(scale = math.sqrt(1/128/3/3))
        w11 = chainer.initializers.Uniform(scale = math.sqrt(1/64/3/3))
        w12 = chainer.initializers.Uniform(scale = math.sqrt(1/32/7/7))
        super(Generator_ResBlock6,self).__init__(
            conv1 = L.Convolution2D(3 + nc_size,32,7,1,3,initialW = w1, nobias = True),
            conv2 = L.Convolution2D(32,64,4,2,1,initialW = w2, nobias = True),
            conv3 = L.Convolution2D(64,128,4,2,1,initialW = w3, nobias = True),
            conv4_1 = L.Convolution2D(128,128,3,1,1,initialW = w_res, nobias = True),
            conv4_2 = L.Convolution2D(128,128,3,1,1,initialW = w_res, nobias = True),
            conv5_1 = L.Convolution2D(128,128,3,1,1,initialW = w_res, nobias = True),
            conv5_2 = L.Convolution2D(128,128,3,1,1,initialW = w_res, nobias = True),
            conv6_1 = L.Convolution2D(128,128,3,1,1,initialW = w_res, nobias = True),
            conv6_2 = L.Convolution2D(128,128,3,1,1,initialW = w_res, nobias = True),
            conv7_1 = L.Convolution2D(128,128,3,1,1,initialW = w_res, nobias = True),
            conv7_2 = L.Convolution2D(128,128,3,1,1,initialW = w_res, nobias = True),
            conv8_1 = L.Convolution2D(128,128,3,1,1,initialW = w_res, nobias = True),
            conv8_2 = L.Convolution2D(128,128,3,1,1,initialW = w_res, nobias = True),
            conv9_1 = L.Convolution2D(128,128,3,1,1,initialW = w_res, nobias = True),
            conv9_2 = L.Convolution2D(128,128,3,1,1,initialW = w_res, nobias = True),
            conv10 = L.Convolution2D(128,64,3,1,1,initialW = w10, nobias = True),
            conv11 = L.Convolution2D(64,32,3,1,1,initialW = w11, nobias = True),
            conv12 = L.Convolution2D(32,3,7,1,3,initialW = w12, nobias = True),

            bnc1 = L.BatchNormalization(32, use_gamma = True, use_beta = True),
            bnc2 = L.BatchNormalization(64, use_gamma = True, use_beta = True),
            bnc3 = L.BatchNormalization(128, use_gamma = True, use_beta = True),
            bnc4_1 = L.BatchNormalization(128, use_gamma = True, use_beta = True),
            bnc4_2 = L.BatchNormalization(128, use_gamma = True, use_beta = True),
            bnc5_1 = L.BatchNormalization(128, use_gamma = True, use_beta = True),
            bnc5_2 = L.BatchNormalization(128, use_gamma = True, use_beta = True),
            bnc6_1 = L.BatchNormalization(128, use_gamma = True, use_beta = True),
            bnc6_2 = L.BatchNormalization(128, use_gamma = True, use_beta = True),
            bnc7_1 = L.BatchNormalization(128, use_gamma = True, use_beta = True),
            bnc7_2 = L.BatchNormalization(128, use_gamma = True, use_beta = True),
            bnc8_1 = L.BatchNormalization(128, use_gamma = True, use_beta = True),
            bnc8_2 = L.BatchNormalization(128, use_gamma = True, use_beta = True),
            bnc9_1 = L.BatchNormalization(128, use_gamma = True, use_beta = True),
            bnc9_2 = L.BatchNormalization(128, use_gamma = True, use_beta = True),
            bnc10 = L.BatchNormalization(64, use_gamma = True, use_beta = True),
            bnc11 = L.BatchNormalization(32, use_gamma = True, use_beta = True),
            )
    def __call__(self,x,domain):
        B, ch, H, W = x.shape
        B, A = domain.shape
        domain_map = xp.broadcast_to(domain, (H,W,B,A))
        domain_map = Variable(xp.transpose(domain_map, (2,3,0,1)))
        h = F.concat([x, domain_map], axis = 1)

        h = F.relu(self.bnc1(self.conv1(h)))
        h = F.relu(self.bnc2(self.conv2(h)))
        h = F.relu(self.bnc3(self.conv3(h)))
        h4_1 = F.relu(self.bnc4_1(self.conv4_1(h)))
        h = F.relu(self.bnc4_2(self.conv4_2(h4_1))) + h
        h5_1 = F.relu(self.bnc5_1(self.conv5_1(h)))
        h = F.relu(self.bnc5_2(self.conv5_2(h5_1))) + h
        h6_1 = F.relu(self.bnc6_1(self.conv6_1(h)))
        h = F.relu(self.bnc6_2(self.conv6_2(h6_1))) + h
        h7_1 = F.relu(self.bnc7_1(self.conv7_1(h)))
        h = F.relu(self.bnc7_2(self.conv7_2(h7_1))) + h
        h8_1 = F.relu(self.bnc8_1(self.conv8_1(h)))
        h = F.relu(self.bnc8_2(self.conv8_2(h8_1))) + h
        h9_1 = F.relu(self.bnc9_1(self.conv9_1(h)))
        h = F.relu(self.bnc9_2(self.conv9_2(h9_1))) + h
        h = F.unpooling_2d(h,4,2,1,cover_all = False)
        h = F.relu(self.bnc10(self.conv10(h)))
        h = F.unpooling_2d(h,4,2,1,cover_all = False)
        h = F.relu(self.bnc11(self.conv11(h)))
        h = F.tanh(self.conv12(h))

        return h

class Discriminator(Chain):
    def __init__(self):
        w1 = chainer.initializers.Uniform(scale = math.sqrt(1/3/4/4))
        w2 = chainer.initializers.Uniform(scale = math.sqrt(1/32/4/4))
        w3 = chainer.initializers.Uniform(scale = math.sqrt(1/64/4/4))
        w4 = chainer.initializers.Uniform(scale = math.sqrt(1/128/4/4))
        w5 = chainer.initializers.Uniform(scale = math.sqrt(1/256/4/4))
        w6 = chainer.initializers.Uniform(scale = math.sqrt(1/512/4/4))
        w_out = chainer.initializers.Uniform(scale = math.sqrt(1/1024/3/3))
        w_cls = chainer.initializers.Uniform(scale = math.sqrt(1/1024/2/2))
        super(Discriminator,self).__init__(
            conv1 = L.Convolution2D(3,32,4,2,1,initialW = w1,nobias = True),
            conv2 = L.Convolution2D(32,64,4,2,1,initialW = w2, nobias=True),
            conv3 = L.Convolution2D(64,128,4,2,1,initialW = w3, nobias=True),
            conv4 = L.Convolution2D(128,256,4,2,1,initialW = w4, nobias=True),
            conv5 = L.Convolution2D(256,512,4,2,1,initialW = w5, nobias  = True),
            conv6 = L.Convolution2D(512,1024,4,2,1,initialW = w6, nobias = True),
            conv_out = L.Convolution2D(1024,1,3,1,1,nobias = True, initialW = w_out),
            conv_cls = L.Convolution2D(1024,4, 2, 1,0, nobias = True, initialW = w_cls),

            )

    def __call__(self,x):
        h = F.leaky_relu((self.conv1(x)), slope = 0.01)
        h = F.leaky_relu((self.conv2(h)), slope = 0.01)
        h = F.leaky_relu((self.conv3(h)), slope = 0.01)
        h = F.leaky_relu((self.conv4(h)), slope = 0.01)
        h = F.leaky_relu((self.conv5(h)), slope = 0.01)
        h = F.leaky_relu((self.conv6(h)), slope = 0.01)
        h_out = self.conv_out(h)
        h_cls = self.conv_cls(h)
        h_cls = F.reshape(h_cls, (x.shape[0],4))

        return h_out, h_cls