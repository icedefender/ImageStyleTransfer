import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn import init


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


class AdaILN(nn.Module):
    def __init__(self, in_ch):
        super(AdaILN, self).__init__()

        self.ro = Parameter(torch.Tensor(1, in_ch, 1, 1))
        self.ro.data.fill_(0.9)

    def forward(self, x, gamma, beta):
        i_mean = torch.mean(torch.mean(x, dim=2, keepdim=True), dim=3, keepdim=True)
        i_var = torch.var(torch.var(x, dim=2, keepdim=True), dim=3, keepdim=True)
        i_std = torch.sqrt(i_var + 1e-5)
        i_h = (x - i_mean) / i_std

        l_mean = torch.mean(torch.mean(torch.mean(x, dim=1, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True)
        l_var = torch.var(torch.var(torch.var(x, dim=1, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True)
        l_std = torch.sqrt(l_var + 1e-5)
        l_h = (x - l_mean) / l_std

        h = self.ro.expand(x.size(0), -1, -1, -1) * i_h + (1 - self.ro.expand(x.size(0), -1, -1, -1)) * l_h
        h = h * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)

        return h


class InstanceLayerNormalization(nn.Module):
    def __init__(self, in_ch):
        super(InstanceLayerNormalization, self).__init__()

        self.ro = Parameter(torch.Tensor(1, in_ch, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, in_ch, 1, 1))
        self.beta = Parameter(torch.Tensor(1, in_ch, 1, 1))
        self.ro.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, x):
        i_mean = torch.mean(torch.mean(x, dim=2, keepdim=True), dim=3, keepdim=True)
        i_var = torch.var(torch.var(x, dim=2, keepdim=True), dim=3, keepdim=True)
        i_std = torch.sqrt(i_var + 1e-5)
        i_h = (x - i_mean) / i_std

        l_mean = torch.mean(torch.mean(torch.mean(x, dim=1, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True)
        l_var = torch.var(torch.var(torch.var(x, dim=1, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True)
        l_std = torch.sqrt(l_var + 1e-5)
        l_h = (x - l_mean) / l_std

        h = self.ro.expand(x.size(0), -1, -1, -1) * i_h + (1 - self.ro.expand(x.size(0), -1, -1, -1)) * l_h
        h = h * self.gamma.expand(x.size(0), -1, -1, -1) + self.beta.expand(x.size(0), -1, -1, -1)

        return h


class CIR(nn.Module):
    def __init__(self, in_ch, out_ch, down=False, up=False):
        self.down = down
        self.up = up
        super(CIR, self).__init__()

        if self.down:
            self.cir = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, 4, 2, 0, bias=False),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        elif self.up:
            self.cir = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, 3, 1, 0, bias=False),
                InstanceLayerNormalization(out_ch),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        h = self.cir(x)

        return h


class CIRDis(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CIRDis, self).__init__()

        self.cir = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(
                nn.Conv2d(in_ch, out_ch, 4, 2, 0, bias=True)
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        h = self.cir(x)

        return h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResBlock, self).__init__()

        self.res = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_ch, out_ch, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(out_ch)
        )

    def forward(self, x):
        h = self.res(x)
        
        return h + x


class ResBlockAdaILN(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResBlockAdaILN, self).__init__()

        self.pad0 = nn.ReflectionPad2d(1)
        self.c0 = nn.Conv2d(in_ch, out_ch, 3, 1, 0, bias=False)
        self.norm0 = AdaILN(out_ch)
        self.relu0 = nn.ReLU(inplace=True)

        self.pad1 = nn.ReflectionPad2d(1)
        self.c1 = nn.Conv2d(out_ch, out_ch, 3, 1, 0, bias=False)
        self.norm1 = AdaILN(out_ch)

    def __call__(self, x, gamma, beta):
        h = self.pad0(x)
        h = self.relu0(self.norm0(self.c0(h), gamma, beta))
        h = self.pad1(h)
        h = self.norm1(self.c1(h), gamma, beta)

        return h + x


class Attention(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Attention, self).__init__()

        self.gap_l = nn.Linear(in_ch, 1, bias=False)
        self.gmp_l = nn.Linear(in_ch, 1, bias=False)
        self.c0 = nn.Conv2d(in_ch * 2, out_ch, 1, 1, 0, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        gap = F.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_l(gap.view(gap.size(0), -1))
        gap_weight = list(self.gap_l.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = F.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_l(gmp.view(gmp.size(0), -1))
        gmp_weight = list(self.gmp_l.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        logit = torch.cat([gap_logit, gmp_logit], dim=1)
        h = torch.cat([gap, gmp], dim=1)
        h = self.relu(self.c0(h))

        return h, logit


class AttentionDis(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(AttentionDis, self).__init__()

        self.gap_l = nn.utils.spectral_norm(nn.Linear(in_ch, 1, bias=False))
        self.gmp_l = nn.utils.spectral_norm(nn.Linear(in_ch, 1, bias=False))
        self.c0 = nn.Conv2d(in_ch * 2, out_ch, 1, 1, 0, bias=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        gap = F.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_l(gap.view(gap.size(0), -1))
        gap_weight = list(self.gap_l.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = F.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_l(gmp.view(gmp.size(0), -1))
        gmp_weight = list(self.gmp_l.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        logit = torch.cat([gap_logit, gmp_logit], dim=1)
        h = torch.cat([gap, gmp], dim=1)
        h = self.relu(self.c0(h))

        return h, logit


class Generator(nn.Module):
    def __init__(self, base=64, layers=6):
        super(Generator, self).__init__()

        self.in_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, base, 7, 1, bias=False),
            nn.InstanceNorm2d(base),
            nn.ReLU(inplace=True)
        )
        init_weights(self.in_block)

        self.down = nn.Sequential(
            CIR(base, base*2, down=True),
            CIR(base*2, base*4, down=True),
            ResBlock(base*4, base*4),
            ResBlock(base*4, base*4),
            ResBlock(base*4, base*4),
            ResBlock(base*4, base*4),
        #    ResBlock(base*4, base*4),
        #    ResBlock(base*4, base*4)
        )
        init_weights(self.down)

        self.attn = Attention(base*4, base*4)
        init_weights(self.attn)

        self.linear = nn.Sequential(
            nn.Linear(32 * 32 * base * 4, base*4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(base*4, base*4),
            nn.ReLU(inplace=True)
        )
        init_weights(self.linear)

        self.gamma = nn.Linear(base*4, base*4, bias=False)
        self.beta = nn.Linear(base*4, base*4, bias=False)

        init_weights(self.gamma)
        init_weights(self.beta)

        self.res0 = ResBlockAdaILN(base*4, base*4)
        self.res1 = ResBlockAdaILN(base*4, base*4)
        self.res2 = ResBlockAdaILN(base*4, base*4)
        self.res3 = ResBlockAdaILN(base*4, base*4)
        #self.res4 = ResBlockAdaILN(base*4, base*4)
        #self.res5 = ResBlockAdaILN(base*4, base*4)

        init_weights(self.res0)
        init_weights(self.res1)
        init_weights(self.res2)
        init_weights(self.res3)
        #init_weights(self.res4)
        #init_weights(self.res5)

        self.up = nn.Sequential(
            CIR(base*4, base*2, up=True),
            CIR(base*2, base, up=True)
        )

        init_weights(self.up)

        self.out_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(base, 3, 7, 1, 0, bias=False),
            nn.Tanh()
        )
        init_weights(self.out_block)

    def forward(self, x):
        h = self.in_block(x)
        h = self.down(h)
        h, logit = self.attn(h)
        hmap = torch.sum(h, dim=1, keepdim=True)
        h_ = self.linear(h.view(h.size(0), -1))
        gamma, beta = self.gamma(h_), self.beta(h_)
        h = self.res0(h, gamma, beta)
        h = self.res1(h, gamma, beta)
        h = self.res2(h, gamma, beta)
        h = self.res3(h, gamma, beta)
        #h = self.res4(h, gamma, beta)
        #h = self.res5(h, gamma, beta)
        h = self.up(h)
        h = self.out_block(h)

        return h, logit, hmap


class Discriminator(nn.Module):
    def __init__(self, base=64):
        super(Discriminator, self).__init__()

        self.down = nn.Sequential(
            CIRDis(3, base),
            CIRDis(base, base*2),
            CIRDis(base*2, base*4),
            CIRDis(base*4, base*8),
            CIRDis(base*8, base*8),
        )
        init_weights(self.down)

        self.attn = AttentionDis(base*8, base*8)
        init_weights(self.attn)

        self.out = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(
                nn.Conv2d(base * 8, 1, 4, 1, 0, bias=False)
            )
        )
        init_weights(self.out)

    def forward(self, x):
        h = self.down(x)
        h, logit = self.attn(h)
        hmap = torch.sum(h, dim=1, keepdim=True)
        h = self.out(h)

        return h, logit, hmap


class RhoClipper(object):

    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):

        if hasattr(module, 'ro'):
            w = module.ro.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.ro.data = w
