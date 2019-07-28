import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers, cuda, serializers, Variable, initializers, Chain
from chainer.functions.connection import convolution_2d
from chainer.links.connection.convolution_2d import Convolution2D
from chainer.functions.connection import deconvolution_2d
from chainer.links.connection.deconvolution_2d import Deconvolution2D
from chainer.functions.connection import linear
from chainer.links.connection.linear import Linear
import numpy as np

xp = cuda.cupy


def _l2normalize(v, eps = 1e-12):
    return v / (((v**2).sum())**0.5 + eps)


def max_singular_value(W, u = None, Ip = 1):
    if u is None:
        u = xp.random.normal(size = (1, W.shape[0])).astype(xp.float32)
    _u = u

    for _ in range(Ip):
        _v = _l2normalize(xp.dot(_u, W.data),eps = 1e-12)
        _u = _l2normalize(xp.dot(_v, W.data.transpose()), eps = 1e-12)
    sigma = F.math.sum.sum(F.connection.linear.linear(_u, F.array.transpose.transpose(W))* _v)
    return sigma, _u, _v


class SNConvolution2D(Convolution2D):
    def __init__(self, in_channels, out_channels, ksize, stride = 1, pad = 0, nobias = True, initialW = None, initial_bias = None, use_gamma = False, Ip = 1):
         self.Ip = Ip
         self.u = None
         self.use_gamma = use_gamma
         super(SNConvolution2D,self).__init__(in_channels, out_channels, ksize, stride, pad, nobias, initialW, initial_bias)

    @property
    def W_bar(self):
        W_mat = self.W.reshape(self.W.shape[0], -1)
        sigma, _u, _ = max_singular_value(W_mat, self.u, self.Ip)
        sigma = F.array.broadcast.broadcast_to(sigma.reshape((1,1,1,1)), self.W.shape)
        self.u = _u
        return self.W / sigma

    def _initialize_params(self, in_size):
        super(SNConvolution2D,self)._initialize_params(in_size)
        if self.use_gamma:
            W_mat = self.W.data.reshape(self.W.shape[0], -1)
            _, s, _ = np.linalg.svd(W_mat)
            with self.init_scope():
                self.gamma = chainer.Parameter(s[0], (1,1,1,1))

    def __call__(self,x):
        if self.W.data is None:
            self._initialize_params(x.shape[1])
        return convolution_2d.convolution_2d(x, self.W_bar, self.b, self.stride, self.pad)


class SNDeconvolution2D(Deconvolution2D):
    def __init__(self, in_channels, out_channels, ksize, stride = 1, pad = 0, nobias = True, initialW = None, initial_bias = None, use_gamma = False, Ip = 1):
         self.Ip = Ip
         self.u = None
         self.use_gamma = use_gamma
         super(SNDeconvolution2D,self).__init__(in_channels, out_channels, ksize, stride, pad, nobias, initialW, initial_bias)

    @property
    def W_bar(self):
        W_mat = self.W.reshape(self.W.shape[0], -1)
        sigma, _u, _ = max_singular_value(W_mat, self.u, self.Ip)
        sigma = F.array.broadcast.broadcast_to(sigma.reshape((1,1,1,1)), self.W.shape)
        self.u = _u
        return self.W / sigma

    def _initialize_params(self, in_size):
        super(SNDeconvolution2D,self)._initialize_params(in_size)
        if self.use_gamma:
            W_mat = self.W.data.reshape(self.W.shape[0], -1)
            _, s, _ = np.linalg.svd(W_mat)
            with self.init_scope():
                self.gamma = chainer.Parameter(s[0], (1,1,1,1))

    def __call__(self,x):
        if self.W.data is None:
            self._initialize_params(x.shape[1])
        return deconvolution_2d.deconvolution_2d(x, self.W_bar, self.b, self.stride, self.pad)


class SNLinear(Linear):
    def __init__(self, in_size, out_size, use_gamma = False, nobias = False, initialW = None, initial_bias = None, Ip = 1):
        self.Ip = Ip
        self.u = None
        self.use_gamma = use_gamma
        super(SNLinear, self).__init__(in_size, out_size, nobias, initialW, initial_bias)

    @property
    def W_bar(self):
        sigma, _u, _ = max_singular_value(self.W, self.u, self.Ip)
        sigma = F.array.broadcast.broadcast_to(sigma.reshape((1, 1)), self.W.shape)
        self.u = _u
        return self.W / sigma

    def _initialize_params(self, in_size):
        super(SNLinear, self)._initialize_params(in_size)
        if self.use_gamma:
            _, s, _ = np.linalg.svd(self.W.data)
            with self.init_scope():
                self.gamma = chainer.Parameter(s[0], (1, 1))

    def __call__(self,x):
        if self.W.data is None:
            self._initialize_params(x.size // x.shape[0])
        return linear.linear(x, self.W_bar, self.b)


def calc_style_mean_std(feature, eps=1e-5):
    mean = F.mean(feature, axis=1).reshape(feature.shape[0], 1)
    sigma = F.average((feature - F.tile(mean, (1, 256)))**2, axis=1) + eps
    std = F.sqrt(sigma).reshape(feature.shape[0], 1, 1, 1)
    mean = F.reshape(mean, (feature.shape[0], 1, 1, 1))

    return mean, std


def calc_content_mean_std(feature, eps = 1e-5):
    batch, channels, height, width = feature.shape
    feature_mean = F.mean(feature.reshape(batch, channels, -1), axis = 2)
    feature_sigma = F.average((feature - F.tile(feature_mean.reshape(batch, channels, 1, 1), (1,1, height, width)))**2, axis=(2, 3))
    feature_std = F.sqrt(feature_sigma).reshape(batch, channels, 1, 1)
    feature_mean = feature_mean.reshape(batch, channels, 1,1)

    return feature_std, feature_mean


def adain(content_feature, style_feature):
    batch, channel, height, width = content_feature.shape
    style_std, style_mean = style_feature[:, 512:1024], style_feature[:, :512]
    style_std = style_std.reshape(batch, channel, 1, 1)
    style_mean = style_mean.reshape(batch, channel, 1, 1)
    style_mean = F.broadcast_to(style_mean, shape = (batch, channel, height, width))
    style_std = F.broadcast_to(style_std, shape = (batch, channel, height, width))
    
    content_std, content_mean = calc_content_mean_std(content_feature)
    content_mean = F.broadcast_to(content_mean, shape = (batch, channel, height, width))
    content_std = F.broadcast_to(content_std, shape = (batch, channel, height, width))
    normalized_feat = (content_feature - content_mean) / content_std

    return normalized_feat * style_std + style_mean


class CBR(Chain):
    def __init__(self, in_ch, out_ch, down=False, up=False):
        w = initializers.Normal(0.02)
        self.up = up
        self.down = down
        super(CBR, self).__init__()
        with self.init_scope():
            self.cup = SNConvolution2D(in_ch, out_ch, 3, 1, 1, initialW=w)
            self.cdown = SNConvolution2D(in_ch, out_ch, 4, 2, 1, initialW=w)

    def __call__(self, x):
        if self.up:
            h = F.unpooling_2d(x, 2, 2, 0, cover_all=False)
            h = F.relu(self.cup(h))

        elif self.down:
            h = F.relu(self.cdown(x))

        else:
            h = F.relu(self.cup(x))

        return h


class ResBlock(Chain):
    def __init__(self, in_ch, out_ch, adain=False):
        w = initializers.Normal(0.02)
        self.adain = adain
        super(ResBlock, self).__init__()
        with self.init_scope():
            self.c0 = SNConvolution2D(in_ch, out_ch, 3, 1, 1, initialW=w)
            self.c1 = SNConvolution2D(out_ch, out_ch, 3, 1, 1, initialW=w)

    def __call__(self, x, style=None):
        if self.adain:
            h = F.relu(adain(self.c0(x), style))
            h = F.relu(adain(self.c1(h), style))

        else:
            h = F.relu(self.c0(x))
            h = F.relu(self.c1(h))

        return h + x


class Dis_ResBlock(Chain):
    def __init__(self, in_ch, out_ch):
        w = initializers.Normal(0.02)
        super(Dis_ResBlock, self).__init__()
        with self.init_scope():
            self.c0 = SNConvolution2D(in_ch, out_ch, 3, 1, 1, initialW=w)
            self.c1 = SNConvolution2D(out_ch, out_ch, 3, 1, 1, initialW=w)
            self.c_sc = SNConvolution2D(in_ch, out_ch, 1, 1, 0, initialW=w)

    def __call__(self, x):
        h = F.relu(self.c0(x))
        h = F.relu(self.c1(h))

        h_sc = F.relu(self.c_sc(x))

        return h + h_sc


class ContentEncoder(Chain):
    def __init__(self, base=64):
        w = initializers.Normal(0.02)
        super(ContentEncoder, self).__init__()

        with self.init_scope():
            self.c0 = SNConvolution2D(3, base, 7, 1, 3, initialW=w)
            self.cbr1 = CBR(base, base*2, down=True)
            self.cbr2 = CBR(base*2, base*4, down=True)
            self.cbr3 = CBR(base*4, base*8, down=True)
            self.res0 = ResBlock(base*8, base*8)
            self.res1 = ResBlock(base*8, base*8)

    def __call__(self, x):
        h = F.relu(self.c0(x))
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
            self.c0 = SNConvolution2D(3, base, 7, 1, 3, initialW=w)
            self.cbr1 = CBR(base, base*2, down=True)
            self.cbr2 = CBR(base*2, base*4, down=True)
            self.cbr3 = CBR(base*4, base*8, down=True)
            self.cbr4 = CBR(base*8, base*16, down=True)
            self.cout = SNConvolution2D(base*16, base*16, 1, 1, 0, initialW=w)

    def __call__(self, x):
        h = F.relu(self.c0(x))
        h = self.cbr1(h)
        h = self.cbr2(h)
        h = self.cbr3(h)
        h = self.cbr4(h)
        batch, cha, height, width = h.shape
        h = F.average_pooling_2d(h, (height, width))
        h = self.cout(h).reshape(batch, cha)
        #h = F.mean(h, axis=0)
        #h = F.tile(h, (batch, 1))

        return h


class Decoder(Chain):
    def __init__(self, base=64):
        w = initializers.Normal(0.02)
        super(Decoder, self).__init__()
        with self.init_scope():
            self.l0 = SNLinear(base*16, base*4)
            self.l1 = SNLinear(base*4, base*4)
            self.l2 = SNLinear(base*4, base*16)

            self.res0 = ResBlock(base*8, base*8, adain=True)
            self.res1 = ResBlock(base*8, base*8, adain=True)
            self.cbr0 = CBR(base*8, base*4, up=True)
            self.cbr1 = CBR(base*4, base*2, up=True)
            self.cbr2 = CBR(base*2, base, up=True)
            self.c = SNConvolution2D(base, 3, 7, 1, 3, initialW=w)

    def __call__(self, content, style):
        h = F.relu(self.l0(style))
        h = F.relu(self.l1(h))
        hstyle = F.relu(self.l2(h))

        h = self.res0(content, hstyle)
        h = self.res1(h, hstyle)
        h = self.cbr0(h)
        h = self.cbr1(h)
        h = self.cbr2(h)
        h = self.c(h)

        return F.tanh(h)


class SNGenerator(Chain):
    def __init__(self):
        super(SNGenerator, self).__init__()
        with self.init_scope():
            self.content = ContentEncoder()
            self.cls = ClassEncoder()
            self.decoder = Decoder()

    def __call__(self, content, style):
        zx = self.content(content)
        zy = self.cls(style)
        h = self.decoder(zx, zy)

        return h


class SNDiscriminator(Chain):
    def __init__(self, cls_len=5, base=64):
        super(SNDiscriminator, self).__init__()
        w = initializers.Normal(0.02)
        self.cls_len = cls_len
        with self.init_scope():
            self.c0 = SNConvolution2D(3, base, 7, 1, 3, initialW=w)
            self.res0 = Dis_ResBlock(base, base*2)
            self.res1 = Dis_ResBlock(base*2, base*2)
            self.res2 = Dis_ResBlock(base*2, base*4)
            self.res4 = Dis_ResBlock(base*4, base*8)
            self.res6 = Dis_ResBlock(base*8, base*8)
            self.res8 = Dis_ResBlock(base*8, base*16)
            self.c1 = SNConvolution2D(base*16, self.cls_len, 3, 1, 1, initialW=w)

    def __call__(self, x, label):
        h = F.relu(self.c0(x))
        h = self.res0(h)
        h = F.average_pooling_2d(h, 3, 2, 1)
        h = self.res1(h)
        h = F.average_pooling_2d(h, 3, 2, 1)
        h = self.res2(h)
        h = F.average_pooling_2d(h, 3, 2, 1)
        h = self.res4(h)
        h = F.average_pooling_2d(h, 3, 2, 1)
        h = self.res6(h)
        h = F.average_pooling_2d(h, 3, 2, 1)
        h_feat = self.res8(h)

        h_cls = self.c1(h_feat)
        #h_cls = F.reshape(F.average_pooling_2d(h_cls, (8, 8)), (h_cls.shape[0], 5))
        h_cls = h_cls[:, label, :, :]

        return h_feat, h_cls