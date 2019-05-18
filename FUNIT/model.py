import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Chain, initializers

xp = cuda.cupy
cuda.get_device(0).use()


class CBR(Chain):
    def __init__(self, in_ch, out_ch, down=False, up=False):
        w = initializers.Normal(0.02)
        self.up = up
        self.down = down
        super(CBR, self).__init__()
        with self.init_scope():
            self.cup = L.Convolution2D(in_ch, out_ch, 3, 1, 1, initialW=w)
            self.cdown = L.Convolution2D(in_ch, out_ch, 4, 2, 1, initialW=w)

            self.bn0 = L.BatchNormalization(out_ch)

    def __call__(self, x):
        if self.up:
            h = F.unpooling_2d(x, 2, 2, 0, cover_all=False)
            h = F.relu(self.bn0(self.cup(h)))

        elif self.down:
            h = F.relu(self.bn0(self.cdown(x)))

        else:
            h = F.relu(self.bn0(self.cup(x)))

        return h


class ResBlock(Chain):
    def __init__(self, in_ch, out_ch, adain=True):
        w = initializers.Normal(0.02)
        self.adain = adain
        super(ResBlock, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(out_ch, out_ch, 3, 1, 1, initialW=w)

            self.bn0 = L.BatchNormalization(out_ch)
            self.bn1 = L.BatchNormalization(out_ch)

    def __call__(self, x):
        if self.adain:
            h = F.relu(adain(self.c0(x)))
            h = F.relu(adain(self.c1(h)))

        else:
            h = F.relu(self.bn0(self.c0(x)))
            h = F.relu(self.bn1(self.c1(h)))

        return h + x


class ContentEncoder(Chain):
    def __init__(self, base=64):
        w = initializers.Normal(0.02)
        super(ContentEncoder, self).__init__()

        with self.init_scope():
            self.cbr0 = CBR(3, base)
            self.cbr1 = CBR(base, base*2, down=True)
            self.cbr2 = CBR(base*2, base*4, down=True)
            self.cbr3 = CBR(base*4, base*8, down=True)
            self.res0 = ResBlock(base*8, base*8)
            self.res1 = ResBlock(base*8, base*8)

    def __call__(self, x):
        h = self.cbr0(x)
        h = self.cbr1(h)
        h = self.cbr2(h)
        h = self.cbr3(h)
        h = self.res0(h)
        h = self.res1(h)

        return h


class ClassEncoder(Chain):
    def __init__(self, base=64):
        w = initializers.Normal(0.02)
        super(ClassEncoder, self).__init__()

        with self.init_scope():
            self.cbr0 = CBR(3, base)
            self.cbr1 = CBR(base, base*2, down=True)
            self.cbr2 = CBR(base*2, base*4, down=True)
            self.cbr3 = CBR(base*4, base*8, down=True)
            self.cbr4 = CBR(base*8, base*16, down=True)

    def __call__(self, x):
        h = self.cbr0(x)
        h = self.cbr1(h)
        h = self.cbr2(h)
        h = self.cbr3(h)
        h = self.cbr4(h)
        batch, cha, height, width = h.shape
        h = F.reshape(F.average_pooling_2d(h, (height, width)),(batch, cha))
        h = F.mean(h, axis=0)
        h = F.tile(h, (batch, 1))

        return h


class Decoder(Chain):
    def __init__(self, base=64):
        w = initializers.Normal(0.02)
        super(Decoder, self).__init__()
        with self.init_scope():
            self.l0 = L.Linear(base*16, base*4)
            self.l1 = L.Linear(base*4, base*4)
            self.l2 = L.Linear(base*4, base*4)

            self.res0 = ResBlock(base*8, base*8, adain=True)
            self.res1 = ResBlock(base*8, base*8, adain=True)
            self.cbr0 = CBR(base*8, base*4, up=True)
            self.cbr1 = CBR(base*4, base*2, up=True)
            self.cbr2 = CBR(base*2, base, up=True)
            self.c = L.Convolution2D(base, 3, 3, 1, 1, initialW=w)

    def __call__(self, x):
        h = self.res0(x)
        h = self.res1(h)
        h = self.cbr0(h)
        h = self.cbr1(h)
        h = self.cbr2(h)
        h = self.c(h)

        return h
