import chainer.functions as F
import chainer.links as L
import math

from chainer import Variable, cuda, Chain, initializers
from sn import SNConvolution2D

xp = cuda.cupy
cuda.get_device(0).use()


class CBR(Chain):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, pad=1,
                 act=F.relu, up=False, down=False, sn=False):
        super(CBR, self).__init__()
        scale = math.sqrt(1 / (in_ch) / kernel / kernel)
        w = initializers.Uniform(scale=scale)
        self.up = up
        self.down = down
        self.act = act
        self.sn = sn
        with self.init_scope():
            if sn:
                self.c0 = SNConvolution2D(in_ch, out_ch, kernel, stride, pad,
                                          initialW=w, nobias=True)
            else:
                self.c0 = L.Convolution2D(in_ch, out_ch, kernel, stride, pad,
                                          initialW=w, nobias=True)
            self.bn0 = L.BatchNormalization(out_ch, use_gamma=True, use_beta=True)

    def __call__(self, x):
        if self.sn:
            if self.down:
                h = self.act(self.c0(x))

            elif self.up:
                h = F.unpooling_2d(x, 2, 2, 0, cover_all=False)
                h = self.act(self.c0(h))

            else:
                h = self.act(self.c0(x))

        else:
            if self.down:
                h = self.act(self.bn0(self.c0(x)))

            elif self.up:
                h = F.unpooling_2d(x, 2, 2, 0, cover_all=False)
                h = self.act(self.bn0(self.c0(h)))

            else:
                h = self.act(self.bn0(self.c0(x)))

        return h


class CBRDis(Chain):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, pad=1,
                 act=F.relu, up=False, down=False):
        super(CBRDis, self).__init__()
        scale = math.sqrt(1 / (in_ch) / kernel / kernel)
        w = initializers.Uniform(scale=scale)
        self.up = up
        self.down = down
        self.act = act
        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, kernel, stride, pad,
                                      initialW=w, nobias=True)

    def __call__(self, x):
        if self.down:
            h = self.act(self.c0(x))

        elif self.up:
            h = F.unpooling_2d(x, 2, 2, 0, cover_all=False)
            h = self.act(self.c0(h))

        else:
            h = self.act(self.c0(x))

        return h


class ResBlock(Chain):
    def __init__(self, in_ch, out_ch, sn=False):
        super(ResBlock, self).__init__()

        with self.init_scope():
            self.cbr0 = CBR(in_ch, out_ch, sn=sn)
            self.cbr1 = CBR(out_ch, out_ch, sn=sn)

    def __call__(self, x):
        h = self.cbr0(x)
        h = self.cbr1(h) + x

        return h


class Generator(Chain):
    def __init__(self, nc_size, base=64, sn=False):
        super(Generator, self).__init__()
        scale = math.sqrt(1 / base / 7 / 7)
        w = initializers.Uniform(scale=scale)
        with self.init_scope():
            self.c0 = CBR(3 + nc_size, base, 7, 1, 3, sn=sn)
            self.c1 = CBR(base, base*2, 4, 2, 1, down=True, sn=sn)
            self.c2 = CBR(base*2, base*4, 4, 2, 1, down=True, sn=sn)
            self.r0 = ResBlock(base*4, base*4, sn=sn)
            self.r1 = ResBlock(base*4, base*4, sn=sn)
            self.r2 = ResBlock(base*4, base*4, sn=sn)
            self.r3 = ResBlock(base*4, base*4, sn=sn)
            self.r4 = ResBlock(base*4, base*4, sn=sn)
            self.r5 = ResBlock(base*4, base*4, sn=sn)
            self.c3 = CBR(base*4, base*2, 3, 1, 1, up=True, sn=sn)
            self.c4 = CBR(base*2, base, 3, 1, 1, up=True, sn=sn)
            if sn:
                self.c5 = SNConvolution2D(base, 3, 7, 1, 3, initialW=w, nobias=True)
            else:
                self.c5 = L.Convolution2D(base, 3, 7, 1, 3, initialW=w, nobias=True)

    def __call__(self, x, domain):
        batch, ch, height, width = x.shape
        _, label = domain.shape
        domain_map = F.broadcast_to(domain, (height, width, batch, label))
        domain_map = F.transpose(domain_map, (2, 3, 0, 1))
        h = F.concat([x, domain_map], axis=1)

        h = self.c0(h)
        h = self.c1(h)
        h = self.c2(h)
        h = self.r0(h)
        h = self.r1(h)
        h = self.r2(h)
        h = self.r3(h)
        h = self.r4(h)
        h = self.r5(h)
        h = self.c3(h)
        h = self.c4(h)
        h = F.tanh(self.c5(h))

        return h


class Discriminator(Chain):
    def __init__(self, nc_size, base=64):
        self.nc_size = nc_size
        wscale = math.sqrt(1 / (base*32) / 3 / 3)
        wout = initializers.Uniform(scale=wscale)
        super(Discriminator, self).__init__()

        with self.init_scope():
            self.c0 = CBRDis(3, base, 4, 2, 1, down=True, act=F.leaky_relu)
            self.c1 = CBRDis(base, base*2, 4, 2, 1, down=True, act=F.leaky_relu)
            self.c2 = CBRDis(base*2, base*4, 4, 2, 1, down=True, act=F.leaky_relu)
            self.c3 = CBRDis(base*4, base*8, 4, 2, 1, down=True, act=F.leaky_relu)
            self.c4 = CBRDis(base*8, base*16, 4, 2, 1, down=True, act=F.leaky_relu)
            self.c5 = CBRDis(base*16, base*32, 4, 2, 1, down=True, act=F.leaky_relu)
            self.cout = L.Convolution2D(base*32, 1, 1, nobias=True, initialW=wout)
            self.cinp = L.Convolution2D(base*32, 6, 1, 1, 0, nobias=True, initialW=wout)
            self.cma0 = L.Convolution2D(base*64+nc_size, base*32, 1, 1, 0, nobias=True, initialW=wout)
            self.cma1 = L.Convolution2D(base*32, 1, 1, 1, 0, nobias=True, initialW=wout)

    def _extract(self, x):
        h = self.c0(x)
        h = self.c1(h)
        h = self.c2(h)
        h = self.c3(h)
        h = self.c4(h)
        h = self.c5(h)

        return h

    def __call__(self, x, y, label, method):
        if method == "adv":
            hx = self._extract(x)
            h = self.cout(hx)

        elif method == "inp":
            hx = self._extract(x)
            h = self.cinp(hx)
            h = F.mean(h, axis=(2, 3))

        elif method == "mat":
            hx = self._extract(x)
            hy = self._extract(y)

            batch, ch, height, width = hx.shape
            _, domain = label.shape
            domain_map = F.broadcast_to(label, (height, width, batch, domain))
            domain_map = F.transpose(domain_map, (2, 3, 0, 1))
            hmat = F.leaky_relu(self.cma0(F.concat([hx, hy, domain_map])))
            h = self.cma1(hmat)

        return h
