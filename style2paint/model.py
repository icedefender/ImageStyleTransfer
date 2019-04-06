import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain, cuda
import numpy as np
import math
xp = cuda.cupy

class VGG(Chain):
    def __init__(self):
        super(VGG, self).__init__()
        w = chainer.initializers.GlorotNormal()
        with self.init_scope():
            self.base = L.VGG16Layers()
            self.l0 = L.Linear(4096, 512, initialW = w)

    def __call__(self,x):
        h_true = self.base(x, layers=["fc7"])["fc7"]
        h_true.unchain_backward()
        h_add = self.l0(h_true)

        return h_true, h_add

class Grayscale(Chain):
    def __init__(self):
        super(Grayscale, self).__init__()

    def __call__(self,x):
        batch, _, width, height = x.shape
        result = xp.zeros((batch, 1, width, height)).astype(xp.float32)
        x = x.data.get()
        x = cuda.to_gpu(x)
        for i in range(batch):
            result[i,:,:,:] = 0.299 * x[i, 0] + 0.587 * x[i, 1] + 0.114 * x[i,2]

        return chainer.as_variable(result)

class CBR(Chain):
    def __init__(self, in_ch, out_ch, sample="down", activation=F.relu):
        super().__init__()
        self.activation = activation
        self.sample = sample

        w = chainer.initializers.GlorotNormal()
        with self.init_scope():
                self.cdown = L.Convolution2D(in_ch, out_ch, 4,2,1,initialW = w)
                self.cup = L.Convolution2D(in_ch, out_ch, 3,1,1,initialW=w)
                self.cpara = L.Convolution2D(in_ch, out_ch, 3,1,1,initialW=w)

                self.bn = L.BatchNormalization(out_ch)

    def __call__(self,x):
        if self.sample == 'down':
            h = self.activation(self.bn(self.cdown(x)))

        elif self.sample == 'up':
            h = F.unpooling_2d(x,2,2,0,cover_all=False)
            h = self.activation(self.bn(self.cup(h)))

        else:
            h = self.activation(self.bn(self.cpara(x)))

        return h

class Decoder_g1(Chain):
    def __init__(self, in_ch, out_ch, base = 16):
        super().__init__()
        w = chainer.initializers.GlorotNormal()

        with self.init_scope():
            self.dc0 = CBR(in_ch, base*8, sample="up", activation=F.leaky_relu)
            self.dc1 = CBR(base*8, base*8, sample="up", activation=F.leaky_relu)
            self.dc2 = CBR(base*8, base*4, sample="up", activation=F.leaky_relu)
            self.dc3 = CBR(base*4, base*2, sample="up", activation=F.leaky_relu)
            self.dc4 = L.Convolution2D(base*2, out_ch, 3,1,1,initialW =w)

    def __call__(self,x):
        h = self.dc0(x)
        h = self.dc1(h)
        h = self.dc2(h)
        h = self.dc3(h)
        h = self.dc4(h)

        return h

class Decoder_g2(Chain):
    def __init__(self, in_ch, out_ch, base = 32):
        super().__init__()
        w = chainer.initializers.GlorotNormal()

        with self.init_scope():
            self.dc0 = CBR(in_ch, base*8, sample="up", activation=F.leaky_relu)
            self.dc1 = CBR(base*8, base*8, sample="up", activation=F.leaky_relu)
            self.dc2 = CBR(base*8, base*4, sample="up", activation=F.leaky_relu)
            self.dc3 = CBR(base*4, base*2, sample="up", activation=F.leaky_relu)
            self.dc4 = L.Convolution2D(base*2, out_ch, 3,1,1,initialW =w)

    def __call__(self,x):
        h = self.dc0(x)
        h = self.dc1(h)
        h = self.dc2(h)
        h = self.dc3(h)
        h = self.dc4(h)

        return h

class UNet(Chain):
    def __init__(self, base = 16):
        super (UNet, self).__init__()
        w = chainer.initializers.GlorotNormal()

        with self.init_scope():
            self.c0 = L.Convolution2D(3, base, 3,1,1,initialW=w)
            self.c1 = CBR(base, base, sample="same",activation=F.leaky_relu)
            self.c2 = CBR(base, base*2, sample="down", activation=F.leaky_relu)
            self.c3 = CBR(base*2, base*4, sample="down", activation=F.leaky_relu)
            self.c4 = CBR(base*4, base*8, sample="down", activation=F.leaky_relu)
            self.c5 = CBR(base*8, base*16, sample="down",activation=F.leaky_relu)
            self.c6 = CBR(base*16, base*32, sample="down", activation=F.leaky_relu)
            self.mid0 = CBR(base*64, base*32, sample="up", activation=F.leaky_relu)
            self.dc0 = CBR(base*(32 + 16), base*8, sample="up", activation=F.relu)
            self.dc1 = CBR(base*(8 + 8), base*4, sample="up", activation=F.relu)
            self.dc2 = CBR(base*(4 + 4), base*2, sample = "up", activation=F.relu)
            self.dc3 = CBR(base*(2 + 2), base, sample = "up", activation=F.relu)
            self.dc4 = CBR(base*(1 + 1), base, sample="same", activation=F.relu)
            self.dc5 = L.Convolution2D(base, 3, 3,1,1,initialW=w)

    def __call__(self,x,z):
        batch, _ = z.shape
        z = z.reshape(batch, 512,1,1)
        enc0 = self.c0(x)
        enc1 = self.c1(enc0)
        enc2 = self.c2(enc1)
        enc3 = self.c3(enc2)
        enc4 = self.c4(enc3)
        enc5 = self.c5(enc4)
        enc6 = self.c6(enc5)
        z = F.broadcast_to(z,(batch, 512, 4, 4))
        dec = F.concat([enc6, z], axis = 1)
        dec0 = self.mid0(dec)
        dec = self.dc0(F.concat([enc5, dec0]))
        dec = self.dc1(F.concat([enc4, dec]))
        dec = self.dc2(F.concat([enc3, dec]))
        dec = self.dc3(F.concat([enc2, dec]))
        dec = self.dc4(F.concat([enc1, dec]))
        dec = self.dc5(dec)

        return dec, enc5, dec0

class Discriminator(Chain):
    def __init__(self, base = 32):
        super(Discriminator, self).__init__()
        w = chainer.initializers.GlorotNormal()

        with self.init_scope():
            self.c0 = L.Convolution2D(3, base, 4,2,1,initialW=w)
            self.c1 = L.Convolution2D(base, base*2, 4,2,1,initialW=w)
            self.c2 = L.Convolution2D(base*2, base*4, 4,2,1,initialW=w)
            self.c3 = L.Convolution2D(base*4, base*8, 4,2,1,initialW=w)
            self.c4_cls = L.Linear(None, 1, initialW = chainer.initializers.HeNormal(math.sqrt(0.02*math.sqrt(8*8*256)/2)))

    def __call__(self,x):
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.c1(h))
        h = F.leaky_relu(self.c2(h))
        h = F.leaky_relu(self.c3(h))
        h = self.c4_cls(h)

        return h