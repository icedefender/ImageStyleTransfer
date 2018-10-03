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
        w = chainer.initializers.Normal(0.02)
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
    def __init__(self, in_ch, out_ch, bn=True, sample="down", activation=F.relu):
        super().__init__()
        self.bn = bn
        self.activation = activation

        w = chainer.initializers.Normal(0.02)
        with self.init_scope():
            if sample == "down":
                self.c = L.Convolution2D(in_ch, out_ch, 4,2,1,initialW = w)
            elif sample == "up":
                self.c = L.Deconvolution2D(in_ch, out_ch, 4,2,1,initialW=w)
            else:
                self.c = L.Convolution2D(in_ch, out_ch, 1,1,0,initialW=w)
            if bn:
                self.batchn = L.BatchNormalization(out_ch)

    def __call__(self,x):
        h = self.c(x)
        if self.bn:
            h = self.batchn(h)
        if self.activation is not None:
            h = self.activation(h)

        return h

class Decoder_g1(Chain):
    def __init__(self, in_ch, out_ch, base = 16):
        super().__init__()
        w = chainer.initializers.Normal(0.02)

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
        w = chainer.initializers.Normal(0.02)

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
        w = chainer.initializers.Normal(0.02)

        with self.init_scope():
            self.c0 = L.Convolution2D(3, base, 3,1,1,initialW=w)
            self.c1 = CBR(base, base, sample="same",activation=F.leaky_relu)
            self.c2 = CBR(base, base*2, sample="down", activation=F.leaky_relu)
            self.c3 = CBR(base*2, base*2, sample="same", activation=F.leaky_relu)
            self.c4 = CBR(base*2, base*4, sample="down", activation=F.leaky_relu)
            self.c5 = CBR(base*4, base*4, sample="same", activation=F.leaky_relu)
            self.c6 = CBR(base*4, base*8, sample="down", activation=F.leaky_relu)
            self.c7 = CBR(base*8, base*8, sample="same", activation=F.leaky_relu)
            self.c8 = CBR(base*8, base*16, sample="down",activation=F.leaky_relu)
            self.c9 = CBR(base*16, base*16, sample="same", activation=F.leaky_relu)
            self.c10 = CBR(base*16, base*32, sample="down", activation=F.leaky_relu)

            self.mid0 = CBR(base*64, base*32, sample="up", activation=F.leaky_relu)

            self.dc0 = CBR(base*32, base*32, sample="same", activation=F.relu)
            self.dc1 = CBR(base*(32 + 16), base*8, sample="up", activation=F.relu)
            self.dc2 = CBR(base*8, base*8, sample="same", activation=F.relu)
            self.dc3 = CBR(base*(8 + 8), base*4, sample="up", activation=F.relu)
            self.dc4 = CBR(base*4, base*4, sample="same", activation=F.relu)
            self.dc5 = CBR(base*(4 + 4), base*2, sample = "up", activation=F.relu)
            self.dc6 = CBR(base*2, base*2, sample="same", activation=F.relu)
            self.dc7 = CBR(base*(2 + 2), base * 4, sample = "up", activation=F.relu)
            self.dc8 = CBR(base*4, base*4, sample="same", activation=F.relu)
            self.dc9 = L.Convolution2D(base*4, 3, 3,1,1,initialW=w)

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
        enc7 = self.c7(enc6)
        enc8 = self.c8(enc7)
        enc9 = self.c9(enc8)
        enc10 = self.c10(enc9)
        z = F.broadcast_to(z,(batch, 512, 4, 4))

        del enc0, enc1, enc2, enc4, enc6, enc8

        dec = F.concat([enc10, z], axis = 1)
        dec = self.mid0(dec)

        del enc10, z

        dec0 = self.dc0(dec)
        dec1 = self.dc1(F.concat([enc9, dec0]))
        dec2 = self.dc2(dec1)
        del dec1
        dec3 = self.dc3(F.concat([enc7, dec2]))
        del enc7, dec2
        dec4 = self.dc4(dec3)
        del dec3
        dec5 = self.dc5(F.concat([enc5, dec4]))
        del enc5, dec4
        dec6 = self.dc6(dec5)
        del dec5
        dec7 = self.dc7(F.concat([enc3, dec6]))
        del enc3,  dec6
        dec8 = self.dc8(dec7)
        del dec7
        dec9 = self.dc9(dec8)
        del dec8

        return dec9, enc9, dec0

class Discriminator(Chain):
    def __init__(self, base = 32):
        super(Discriminator, self).__init__()
        w = chainer.initializers.Normal(0.02)

        with self.init_scope():
            self.c0 = CBR(3, base, sample="down",activation=F.relu)
            self.c2 = CBR(base, base*2, sample="down", activation=F.relu)
            self.c4 = CBR(base*2, base*4, sample="down", activation=F.relu)
            self.c6 = CBR(base*4, base*8, sample="down", activation=F.relu)
            self.c8_cls = L.Linear(256 * 8 * 8, 4096, initialW = chainer.initializers.HeNormal(math.sqrt(0.02*math.sqrt(8*8*256)/2)))

    def __call__(self,x):
        h = self.c0(x)
        h = self.c2(h)
        h = self.c4(h)
        h = self.c6(h)
        h = self.c8_cls(h)

        return h