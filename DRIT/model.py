import torch
import torch.nn as nn
from torch.autograd import Variable


class Conv(nn.Module):
    """
    Convolution Module
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=True, w_init='linear'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              bias=bias)

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        x = self.conv(x)
        return x


class AddGaussian(nn.Module):
    def __init__(self):
        super(AddGaussian, self).__init__()

    def forward(self, x):
        if self.training == False:
            return x

        noise = Variable(torch.randn(x.size()).cuda())
        return x + noise


class CBR(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, down=False, up=False):
        super(CBR, self).__init__()

        if down:
            self.cbr = nn.Sequential(
                Conv(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU()
            )

        elif up:
            self.cbr = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                Conv(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU()
            )

        else:
            self.cbr = nn.Sequential(
                Conv(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU()
            )

    def forward(self, x):
        return self.cbr(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResBlock, self).__init__()

        self.res = nn.Sequential(
            Conv(in_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(),
            Conv(out_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm2d(out_ch),
        )

    def forward(self, x):
        h = self.res(x)

        return h + x


class ConcatResBlock(nn.Module):
    def __init__(self, base=256, attr_dim=256):
        super(ConcatResBlock, self).__init__()

        self.c0 = nn.Sequential(
            Conv(base, base, 3, 1, 1),
            nn.InstanceNorm2d(base)
        )
        self.c1 = nn.Sequential(
            Conv(base, base, 3, 1, 1),
            nn.InstanceNorm2d(base)
        )
        self.attr0 = nn.Sequential(
            Conv(base + attr_dim, base + attr_dim, 1, 1, 0),
            nn.ReLU(inplace=True),
            Conv(base + attr_dim, base, 1, 1, 0),
            nn.ReLU(inplace=True)
        )
        self.attr1 = nn.Sequential(
            Conv(base + attr_dim, base + attr_dim, 1, 1, 0),
            nn.ReLU(inplace=True),
            Conv(base + attr_dim, base, 1, 1, 0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, z):
        z = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
        h = self.c0(x)
        h = self.attr0(torch.cat([h, z], dim=1))
        h = self.c1(h)
        h = self.attr1(torch.cat([h, z], dim=1))

        return h + x


class ContentEncoder(nn.Module):
    def __init__(self, base=64):
        super(ContentEncoder, self).__init__()

        self.enc = nn.Sequential(
            CBR(3, base, kernel=7, stride=1, padding=3),
            CBR(base, base*2, kernel=4, stride=2, padding=1, down=True),
            CBR(base*2, base*4, kernel=4, stride=2, padding=1, down=True),
            ResBlock(base*4, base*4),
            ResBlock(base*4, base*4),
            ResBlock(base*4, base*4),
            ResBlock(base*4, base*4)
        )

    def forward(self, x):
        return self.enc(x)


class AttributeEncoder(nn.Module):
    def __init__(self, base=64):
        super(AttributeEncoder, self).__init__()

        self.enc = nn.Sequential(
            CBR(3, base, kernel=7, stride=1, padding=3),
            CBR(base, base*2, kernel=4, stride=2, padding=1),
            CBR(base*2, base*4, kernel=4, stride=2, padding=1),
            CBR(base*4, base*4, kernel=4, stride=2, padding=1),
            CBR(base*4, base*4, kernel=4, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(1),
            Conv(base*4, 8, 1, 1, 0)
        )

    def forward(self, x):
        h = self.enc(x)
        h = h.view(h.size(0), -1)

        return h


class Decoder(nn.Module):
    def __init__(self, base=64):
        super(Decoder, self).__init__()

        self.dec0 = ConcatResBlock()
        self.dec1 = ConcatResBlock()
        self.dec2 = ConcatResBlock()
        self.dec3 = ConcatResBlock()

        self.dec  = nn.Sequential(
            CBR(base*4, base*2, up=True),
            CBR(base*2, base, up=True),
            Conv(base, 3, 7, 1, 3),
            nn.Tanh()
        )

        self.mlp = nn.Sequential(
            nn.Linear(8, base*4),
            nn.ReLU(inplace=True),
            nn.Linear(base*4, base*4),
            nn.ReLU(inplace=True),
            nn.Linear(base*4, base*16)
        )

    def forward(self, x, z):
        z =  self.mlp(z)
        z0, z1, z2, z3 = torch.split(z, 256, dim=1)
        z0, z1, z2, z3 = z0.contiguous(), z1.contiguous(), z2.contiguous(), z3.contiguous()
        h = self.dec0(x, z0)
        h = self.dec1(h, z1)
        h = self.dec2(h, z2)
        h = self.dec3(h, z3)
        h = self.dec(h)

        return h


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.enc_x = ContentEncoder()
        self.enc_attr_x = AttributeEncoder()
        self.enc_y = ContentEncoder()
        self.enc_attr_y = AttributeEncoder()
        self.dec_x = Decoder()
        self.dec_y = Decoder()

    def _reconstruct(self, content, attr, switch='x'):
        if switch == 'x':
            return self.dec_x(content, attr)

        else:
            return self.dec_y(content, attr)

    def _get_ramdom(self, enc, nz=8):
        batchsize = enc.size(0)
        z = torch.randn(batchsize, nz).cuda()

        return z

    def _mock_inference(self, content, switch='x'):
        if switch == 'x':
            latent = self._get_ramdom(content)
            y = self.dec_y(content, latent)
            y_attr = self.enc_attr_y(y)

            return latent, y, y_attr

        else:
            latent = self._get_ramdom(content)
            x = self.dec_x(content, latent)
            x_attr = self.enc_attr_x(x)

            return latent, x, x_attr

    def forward(self, a, b):
        ha = self.enc_x(a)
        ha_attr = self.enc_attr_x(a)

        hb = self.enc_y(b)
        hb_attr = self.enc_attr_y(b)

        ya = self.dec_x(hb, ha_attr)
        yb = self.dec_y(ha, hb_attr)

        recon_a = self._reconstruct(ha, ha_attr, switch='x')
        recon_b = self._reconstruct(hb, hb_attr, switch='y')

        infer_a = self._mock_inference(ha, switch='y')
        infer_b = self._mock_inference(hb, switch='x')

        return ha, hb, ha_attr, hb_attr, ya, yb, recon_a, recon_b, infer_a, infer_b


class ContentDiscriminator(nn.Module):
    def __init__(self, base=256):
        super(ContentDiscriminator, self).__init__()

        self.dis = nn.Sequential(
            CBR(base, base, kernel=4, stride=2, padding=1,down=True),
            CBR(base, base, kernel=4, stride=2, padding=1,down=True),
            CBR(base, base, kernel=4, stride=2, padding=1,down=True),
            CBR(base, base, kernel=4, stride=1, padding=0)
        )

        self.linear = nn.Linear(base, 1)

    def forward(self, x):
        h = self.dis(x)
        h = h.view(h.size(0), -1)
        h = self.linear(h)

        return h


class DomainDiscriminator(nn.Module):
    def __init__(self, base=64):
        super(DomainDiscriminator, self).__init__()

        self.dis = nn.Sequential(
            CBR(3, base, kernel=4, stride=2, padding=1,down=True),
            CBR(base, base*2, kernel=4, stride=2, padding=1, down=True),
            CBR(base*2, base*4, kernel=4, stride=2, padding=1, down=True),
            CBR(base*4, base*8, kernel=4, stride=2, padding=1, down=True),
            CBR(base*8, base*8, kernel=4, stride=2, padding=1, down=True),
        )

        self.linear = nn.Linear(base*8*4*4, 1)

    def forward(self, x):
        h = self.dis(x)
        h = h.view(h.size(0), -1)
        h = self.linear(h)

        return h