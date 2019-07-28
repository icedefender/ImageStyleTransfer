import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Chain, initializers
from instance_normalization import InstanceNormalization

xp = cuda.cupy
cuda.get_device(0).use()


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
    def __init__(self, in_ch, out_ch, norm=True, down=False, up=False):
        w = initializers.Normal(0.02)
        self.up = up
        self.down = down
        self.norm = norm
        super(CBR, self).__init__()
        with self.init_scope():
            self.cup = L.Convolution2D(in_ch, out_ch, 3, 1, 1, initialW=w)
            self.cdown = L.Convolution2D(in_ch, out_ch, 4, 2, 1, initialW=w)

            self.bn0 = InstanceNormalization(out_ch)

    def __call__(self, x):
        if self.up:
            h = F.unpooling_2d(x, 2, 2, 0, cover_all=False)
            if self.norm:
                h = F.relu(self.bn0(self.cup(h)))
            else:
                h = F.relu(self.cup(h))

        elif self.down:
            if self.norm:
                h = F.relu(self.bn0(self.cdown(x)))
            else:
                h = F.relu(self.cdown(x))

        else:
            if self.norm:
                h = F.relu(self.bn0(self.cup(x)))
            else:
                h = F.relu(self.cup(x))

        return h


class ResBlock(Chain):
    def __init__(self, in_ch, out_ch, norm=True, adain=False):
        w = initializers.Normal(0.02)
        self.adain = adain
        self.norm = norm
        super(ResBlock, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(out_ch, out_ch, 3, 1, 1, initialW=w)

            self.bn0 = InstanceNormalization(out_ch)
            self.bn1 = InstanceNormalization(out_ch)

    def __call__(self, x, style=None):
        if self.adain:
            h = F.relu(adain(self.c0(x), style))
            h = F.relu(adain(self.c1(h), style))

        else:
            if self.norm:
                h = F.relu(self.bn0(self.c0(x)))
                h = F.relu(self.bn1(self.c1(h)))
            else:
                h = F.relu(self.c0(x))
                h = F.relu(self.c1(h))

        return h + x


class Dis_ResBlock(Chain):
    def __init__(self, in_ch, out_ch):
        w = initializers.Normal(0.02)
        super(Dis_ResBlock, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(out_ch, out_ch, 3, 1, 1, initialW=w)
            self.c_sc = L.Convolution2D(in_ch, out_ch, 1, 1, 0, initialW=w)

            self.in0 = L.GroupNormalization(out_ch)
            self.in1 = L.GroupNormalization(out_ch)
            self.in_sc = L.GroupNormalization(out_ch)

    def __call__(self, x):
        h = F.relu(self.in0(self.c0(x)))
        h = F.relu(self.in1(self.c1(h)))

        h_sc = F.relu(self.in_sc(self.c_sc(x)))

        return h + h_sc


class ContentEncoder(Chain):
    def __init__(self, base=64):
        w = initializers.Normal(0.02)
        super(ContentEncoder, self).__init__()

        with self.init_scope():
            self.c0 = L.Convolution2D(3, base, 7, 1, 3, initialW=w)
            self.bn0 = InstanceNormalization(base)
            self.cbr1 = CBR(base, base*2, down=True)
            self.cbr2 = CBR(base*2, base*4, down=True)
            self.cbr3 = CBR(base*4, base*8, down=True)
            self.res0 = ResBlock(base*8, base*8)
            self.res1 = ResBlock(base*8, base*8)

    def __call__(self, x):
        h = F.relu(self.bn0(self.c0(x)))
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
            self.c0 = L.Convolution2D(3, base, 7, 1, 3, initialW=w)
            self.cbr1 = CBR(base, base*2, norm=False,down=True)
            self.cbr2 = CBR(base*2, base*4, norm=False, down=True)
            self.cbr3 = CBR(base*4, base*8, norm=False, down=True)
            self.cbr4 = CBR(base*8, base*16, norm=False, down=True)
            self.cout = L.Convolution2D(base*16, base*16, 1, 1, 0, initialW=w)

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
            self.l0 = L.Linear(base*16, base*4)
            self.l1 = L.Linear(base*4, base*4)
            self.l2 = L.Linear(base*4, base*16)

            self.res0 = ResBlock(base*8, base*8, adain=True)
            self.res1 = ResBlock(base*8, base*8, adain=True)
            self.cbr0 = CBR(base*8, base*4, up=True)
            self.cbr1 = CBR(base*4, base*2, up=True)
            self.cbr2 = CBR(base*2, base, up=True)
            self.c = L.Convolution2D(base, 3, 7, 1, 3, initialW=w)

    def __call__(self, content, style):
        h = F.relu(self.l0(style))
        h = F.relu(self.l1(h))
        hstyle = self.l2(h)

        h = self.res0(content, hstyle)
        h = self.res1(h, hstyle)
        h = self.cbr0(h)
        h = self.cbr1(h)
        h = self.cbr2(h)
        h = self.c(h)

        return F.tanh(h)


class Generator(Chain):
    def __init__(self):
        super(Generator, self).__init__()
        with self.init_scope():
            self.content = ContentEncoder()
            self.cls = ClassEncoder()
            self.decoder = Decoder()

    def __call__(self, content, style):
        zx = self.content(content)
        zy = self.cls(style)
        h = self.decoder(zx, zy)

        return h


class Discriminator(Chain):
    def __init__(self, cls_len=6, base=64):
        super(Discriminator, self).__init__()
        w = initializers.Normal(0.02)
        self.cls_len = cls_len
        with self.init_scope():
            self.c0 = L.Convolution2D(3, base, 7, 1, 3, initialW=w)
            self.res0 = Dis_ResBlock(base, base*2)
            self.res1 = Dis_ResBlock(base*2, base*2)
            self.res2 = Dis_ResBlock(base*2, base*4)
            self.res4 = Dis_ResBlock(base*4, base*8)
            self.res6 = Dis_ResBlock(base*8, base*8)
            self.res8 = Dis_ResBlock(base*8, base*16)
            self.c1 = L.Convolution2D(base*16, self.cls_len, 3, 1, 1, initialW=w)

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