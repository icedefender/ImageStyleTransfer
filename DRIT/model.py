import torch
import torch.nn as nn


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


class CBR(nn.Module):
    def __init__(self, in_ch, out_ch, down=False, up=False):
        super(CBR, self).__init__()

        if down:
            self.cbr = nn.Sequential(
                Conv(in_ch, out_ch, 4, 2, 1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU()
            )

        elif up:
            self.cbr = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                Conv(in_ch, out_ch, 3, 1, 1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU()
            )

        else:
            self.cbr = nn.Sequential(
                Conv(in_ch, out_ch, 3, 1, 1),
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


class Encoder(nn.Module):
    def __init__(self, base=64):
        super(Encoder, self).__init__()

        self.enc = nn.Sequential(
            CBR(3, base, down=True),
            CBR(base, base*2, down=True),
            CBR(base*2, base*4, down=True),
            CBR(base*4, base*4, down=True),
            ResBlock(base*4, base*4),
            ResBlock(base*4, base*4),
            ResBlock(base*4, base*4),
            ResBlock(base*4, base*4),
            ResBlock(base*4, base*4),
            ResBlock(base*4, base*4)
        )

    def forward(self, x):
        return self.enc(x)


class Decoder(nn.Mldule):
    def __init__(self, base=64):
        super(Decoder, self).__init__()

        self.dec = nn.Sequential(
            CBR(base*8, base*4, up=True),
            CBR(base*4, base*2, up=True),
            CBR(base*2, base, up=True),
            CBR(base, base, up=True),
            Conv(base, 3, 7, 1, 3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.dec(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.enc_x = Encoder()
        self.enc_attr_x = Encoder()
        self.enc_y = Encoder()
        self.enc_attr_y = Encoder()
        self.dec_x = Decoder()
        self.dec_y = Decoder()

    def forward(self, a, b):
        ha = self.enc_x(a)
        ha_attr = self.enc_attr_x(a)

        hb = self.enc_y(b)
        hb_attr = self.enc_attr_y(b)

        ya = self.dec_x(torch.cat([ha_attr, hb]))
        yb = self.dec_y(torch.cat([hb_attr, ha]))

        return ha, hb, ya, yb


class ContentDiscriminator(nn.Module):
    def __init__(self, base=256):
        super(ContentDiscriminator, self).__init__()

        self.dis = nn.Sequential(
            CBR(base, base, down=True),
            CBR(base, base, down=True),
            CBR(base, base, down=True),
            CBR(base, base, down=True),
            Conv(256, 1, 1, 1, 0)
        )

    def forward(self, x):
        return self.dis(x)


class DomainDiscriminator(nn.Module):
    def __init__(self, base=64):
        super(DomainDiscriminator, self).__init__()

        self.dis = nn.Sequential(
            CBR(3, base, down=True),
            CBR(base, base*2, down=True),
            CBR(base*2, base*4, down=True),
            CBR(base*4, base*8, down=True),
            Conv(base*8, 1, 1, 1, 0)
        )

    def forward(self, x):
        return self.dis(x)