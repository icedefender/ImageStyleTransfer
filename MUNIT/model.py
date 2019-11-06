import torch
import torch.nn as nn

from torch.nn import init
from torch.autograd import Variable
from torchvision import models


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)

    return feat_mean, feat_std


def adain(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = style_feat[:, :256], style_feat[:, 256:]
    style_mean = style_mean.unsqueeze(dim=2).unsqueeze(dim=3)
    style_std = style_std.unsqueeze(dim=2).unsqueeze(dim=3)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)

    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False, layer=None):
        super(Vgg19, self).__init__()
        self.layer = layer

        vgg_pretrained_features = models.vgg19(pretrained=True).features

        self.slice = nn.Sequential()
        for x in range(37):
            self.slice.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h = self.slice(x)

        return h


class Vgg19Norm(nn.Module):
    def __init__(self, base=512):
        super(Vgg19Norm, self).__init__()
        self.vgg = Vgg19()
        self.norm = nn.InstanceNorm2d(base, affine=False)

    def forward(self, x):
        h = self.vgg(x)
        h = self.norm(h)

        return h


class CIR(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, pad=1, norm=False):
        super(CIR, self).__init__()

        if norm:
            self.cia = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel, stride, pad),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU()
            )

        else:
            self.cia = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel, stride, pad),
                nn.ReLU()
            )

    def forward(self, x):
        return self.cia(x)


class CIL(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, pad=1, norm=False):
        super(CIL, self).__init__()

        if norm:
            self.cia = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel, stride, pad),
                nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU()
            )

        else:
            self.cia = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel, stride, pad),
                nn.LeakyReLU()
            )

    def forward(self, x):
        return self.cia(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, norm=False):
        super(Up, self).__init__()

        if norm:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU()
            )

        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                nn.ReLU()
            )

    def forward(self, x):
        return self.up(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm=False):
        super(ResBlock, self).__init__()

        if norm:
            self.res = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, 1, 1),
                nn.InstanceNorm2d(out_ch)
            )

        else:
            self.res = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            )

    def forward(self, x):
        return self.res(x) + x


class AdaINResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(AdaINResBlock, self).__init__()

        self.c0 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.relu0 = nn.ReLU()
        self.c1 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)

    def forward(self, x, z):
        h = self.c0(x)
        h = self.relu0(adain(h, z))
        h = self.c1(h)
        h = adain(h, z)

        return h + x


class ContentEncoder(nn.Module):
    def __init__(self, base=64):
        super(ContentEncoder, self).__init__()

        self.ce = nn.Sequential(
            CIR(3, base, 7, 1, 3, norm=True),
            CIR(base, base*2, 4, 2, 1, norm=True),
            CIR(base*2, base*4, 4, 2, 1, norm=True),
            ResBlock(base*4, base*4, norm=True),
            ResBlock(base*4, base*4, norm=True),
            ResBlock(base*4, base*4, norm=True),
            ResBlock(base*4, base*4, norm=True)
        )

        init_weights(self.ce)

    def forward(self, x):
        return self.ce(x)


class StyleEncoder(nn.Module):
    def __init__(self, base=64):
        super(StyleEncoder, self).__init__()

        self.se = nn.Sequential(
            CIR(3, base, 7, 1, 3),
            CIR(base, base*2, 4, 2, 1),
            CIR(base*2, base*4, 4, 2, 1),
            CIR(base*4, base*4, 4, 2, 1),
            CIR(base*4, base*4, 4, 2, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base*4, 8, 1, 1, 0)
        )

        init_weights(self.se)

    def forward(self, x):
        h = self.se(x)
        h = h.squeeze(3).squeeze(2)

        return h


class MLP(nn.Module):
    def __init__(self, base=256):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(8, base*2),
            nn.ReLU(),
            nn.Linear(base*2, base*2),
            nn.ReLU(),
            nn.Linear(base*2, base*2)
        )

        init_weights(self.mlp)

    def forward(self, x):
        return self.mlp(x)


class Decoder(nn.Module):
    def __init__(self, base=64):
        super(Decoder, self).__init__()

        self.ar0 = AdaINResBlock(base*4, base*4)
        self.ar1 = AdaINResBlock(base*4, base*4)
        self.ar2 = AdaINResBlock(base*4, base*4)
        self.ar3 = AdaINResBlock(base*4, base*4)
        self.out = nn.Sequential(
            Up(base*4, base*2, norm=True),
            Up(base*2, base, norm=True),
            nn.Conv2d(base, 3, 7, 1, 3),
            nn.Tanh()
        )

        init_weights(self.ar0)
        init_weights(self.ar1)
        init_weights(self.ar2)
        init_weights(self.ar3)
        init_weights(self.out)

    def forward(self, x, z):
        h = self.ar0(x, z)
        h = self.ar1(h, z)
        h = self.ar2(h, z)
        h = self.ar3(h, z)
        h = self.out(h)

        return h


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.ce = ContentEncoder()
        self.se = StyleEncoder()
        self.mlp = MLP()
        self.decoder = Decoder()

    def encode(self, x):
        h_ce = self.ce(x)
        h_se = self.se(x)

        return h_ce, h_se

    def decode(self, x, z):
        z = self.mlp(z)

        return self.decoder(x, z)


class MUNIT(nn.Module):
    def __init__(self):
        super(MUNIT, self).__init__()

        self.ga = Generator()
        self.gb = Generator()

    def _latent_generate(self, batchsize):
        sa_l = Variable(torch.randn(batchsize, 8).cuda())
        sb_l = Variable(torch.randn(batchsize, 8).cuda())

        return sa_l, sb_l

    def forward(self, a, b):
        batchsize = a.size(0)
        sa_l, sb_l = self._latent_generate(batchsize)

        # Reconstruction
        c_a, s_a = self.ga.encode(a)
        c_b, s_b = self.gb.encode(b)

        a_recon = self.ga.decode(c_a, s_a)
        b_recon = self.gb.decode(c_b, s_b)

        # Cross Domain
        ba = self.ga.decode(c_b, sa_l)
        ab = self.gb.decode(c_a, sb_l)

        # Cycle-consistent
        c_b_recon, s_a_recon = self.ga.encode(ba)
        c_a_recon, s_b_recon = self.gb.encode(ab)

        aba = self.ga.decode(c_a_recon, s_a)
        bab = self.gb.decode(c_b_recon, s_b)

        return (c_a, sa_l, c_b, sb_l, a_recon, b_recon, ba, ab, c_b_recon, s_a_recon, c_a_recon, s_b_recon, aba, bab)


class Discriminator(nn.Module):
    def __init__(self, base=64):
        super(Discriminator, self).__init__()
        self.cnns = nn.ModuleList()
        for _ in range(3):
            self.cnns.append(self._make_nets(base))
        self.down = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def _make_nets(self, base):
        model = nn.Sequential(
            CIL(3, base, 4, 2, 1),
            CIL(base, base*2, 4, 2, 1),
            CIL(base*2, base*4, 4, 2, 1),
            CIL(base*4, base*8, 4, 2, 1),
            nn.Conv2d(base*8, 1, 1, 1, 0)
        )

        init_weights(model)

        return model

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            h = model(x)
            outputs.append(h)
            x = self.down(x)

        return outputs
