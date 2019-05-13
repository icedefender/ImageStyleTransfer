import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Chain, initializers
from instance_normalization import InstanceNormalization

class VGG(Chain):
    def __init__(self):
        super(VGG,self).__init__()
        with self.init_scope():
            self.base = L.VGG19Layers()

    def __call__(self, x):
        h2 = self.base(x, layers=["conv4_4"])["conv4_4"]

        return [h2]

class CCBR_down(Chain):
    def __init__(self, in_ch, out_ch):
        super(CCBR_down, self).__init__()
        w = initializers.Normal(0.02)
        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch,4,2,1,initialW = w)
            self.c1 = L.Convolution2D(out_ch, out_ch,3,1,1,initialW = w)

            self.bn0 = InstanceNormalization(out_ch)

    def __call__(self,x):
        h =self.c0(x)
        h =F.relu(self.bn0(self.c1(h)))

        return h

class CCBR_up(Chain):
    def __init__(self, in_ch, out_ch):
        super(CCBR_up, self).__init__()
        w = initializers.Normal(0.02)
        with self.init_scope():
            self.dc0 = L.Convolution2D(in_ch, out_ch, 3,1,1,initialW = w)
            self.dc1 = L.Convolution2D(out_ch, out_ch, 3,1,1,initialW = w)

            self.bn1 = InstanceNormalization(out_ch)

    def __call__(self,x):
        h = x
        h = self.dc0(F.unpooling_2d(h, 2,2,0,cover_all=False))
        h = F.relu(self.bn1(self.dc1(h)))

        return h

class Resblock(Chain):
    def __init__(self, in_ch, out_ch):
        super(Resblock, self).__init__()
        w = initializers.Normal(0.02)
        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, 3,1,1,initialW = w)
            self.c1 = L.Convolution2D(out_ch, out_ch, 3,1,1,initialW = w)

            self.bn0 = InstanceNormalization(out_ch)
            self.bn1 = InstanceNormalization(out_ch)

    def __call__(self,x):
        h = F.relu(self.bn0(self.c0(x)))
        h = self.bn1(self.c1(h))

        return h + x

class Generator(Chain):
    def __init__(self,base =64):
        super(Generator, self).__init__()
        w = initializers.Normal(0.02)
        with self.init_scope():
            self.c0 = L.Convolution2D(3, base, 7,1,3,initialW = w)
            self.down0 = CCBR_down(base, base*2)
            self.down1 = CCBR_down(base*2, base*4)
            self.res0 = Resblock(base*4, base*4)
            self.res1 = Resblock(base*4, base*4)
            self.res2 = Resblock(base*4, base*4)
            self.res3 = Resblock(base*4, base*4)
            self.res4 = Resblock(base*4, base*4)
            self.res5 = Resblock(base*4, base*4)
            self.res6 = Resblock(base*4, base*4)
            self.res7 = Resblock(base*4, base*4)
            self.up0 = CCBR_up(base*4, base*2)
            self.up1 = CCBR_up(base*2, base)
            self.c1 = L.Convolution2D(base, 3,7,1,3,initialW = w)

            self.bn0 = InstanceNormalization(base)

    def __call__(self,x):
        h = F.relu(self.bn0(self.c0(x)))
        h = self.down0(h)
        h = self.down1(h)
        h = self.res0(h)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = self.res6(h)
        #h = self.res7(h)
        h = self.up0(h)
        h = self.up1(h)
        h = self.c1(h)

        return F.tanh(h)

class Discriminator(Chain):
    def __init__(self, base=32):
        super(Discriminator, self).__init__()
        w = initializers.Normal(0.02)
        with self.init_scope():
            self.c0 = L.Convolution2D(3, base, 3,1,1,initialW = w)
            self.c1 = L.Convolution2D(base, base*2, 4,2,1,initialW = w)
            self.c2 = L.Convolution2D(base*2, base*4,3,1,1,initialW = w)
            self.c3 = L.Convolution2D(base*4, base*4,4,2,1,initialW = w)
            self.c4 = L.Convolution2D(base*4, base*8,3,1,1,initialW = w)
            self.c5 = L.Convolution2D(base*8, base*8,3,1,1,initialW = w)
            self.c6 = L.Convolution2D(base*8, 1,3,1,1,initialW = w)

            self.bn0 = InstanceNormalization(base*4)
            self.bn1 = InstanceNormalization(base*8)
            self.bn2 = InstanceNormalization(base*8)

    def __call__(self, x):
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.c1(h))
        h = F.leaky_relu(self.bn0(self.c2(h)))
        h = F.leaky_relu(self.c3(h))
        h = F.leaky_relu(self.bn1(self.c4(h)))
        h = F.leaky_relu(self.bn2(self.c5(h)))
        h = self.c6(h)

        return h
