import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
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


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def normalize(features, eps=1e-5):
    size = features.size()
    batch, channels = size[:2]
    feat_variance = features.view(batch, channels, -1).var(dim=2) + eps
    feat_standard = feat_variance.sqrt().view(batch, channels, 1, 1)
    feat_mean = features.view(batch, channels, -1).mean(dim=2).view(batch, channels, 1, 1)

    normalized = (features - feat_mean.expand(size)) / feat_standard

    return normalized


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False, layer=None):
        super(Vgg19, self).__init__()
        self.layer = layer

        vgg_pretrained_features = models.vgg19(pretrained=True).features

        if layer == 'four':
            self.slice = nn.Sequential()
            for x in range(21):
                self.slice.add_module(str(x), vgg_pretrained_features[x])

        elif layer == 'five':
            self.slice = nn.Sequential()
            for x in range(30):
                self.slice.add_module(str(x), vgg_pretrained_features[x])

        else:
            self.slice1 = torch.nn.Sequential()
            self.slice2 = torch.nn.Sequential()
            self.slice3 = torch.nn.Sequential()
            self.slice4 = torch.nn.Sequential()
            self.slice5 = torch.nn.Sequential()
            for x in range(2):
                self.slice1.add_module(str(x), vgg_pretrained_features[x])
            for x in range(2, 7):
                self.slice2.add_module(str(x), vgg_pretrained_features[x])
            for x in range(7, 12):
                self.slice3.add_module(str(x), vgg_pretrained_features[x])
            for x in range(12, 21):
                self.slice4.add_module(str(x), vgg_pretrained_features[x])
            for x in range(21, 30):
                self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        if self.layer == 'four':
            h = self.slice(x)

        elif self.layer == 'five':
            h = self.slice(x)

        else:
            h_relu1 = self.slice1(x)
            h_relu2 = self.slice2(h_relu1)
            h_relu3 = self.slice3(h_relu2)
            h_relu4 = self.slice4(h_relu3)
            h_relu5 = self.slice5(h_relu4)
            h = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

        return h


class SANet(nn.Module):
    def __init__(self, base=512):
        super(SANet, self).__init__()

        self.c1 = nn.Conv2d(base, base, 1, 1, 0)
        self.c2 = nn.Conv2d(base, base, 1, 1, 0)
        self.c3 = nn.Conv2d(base, base, 1, 1, 0)
        self.softmax = nn.Softmax(dim=1)

        #init_weights(self.c1, init_type='normal')
        #init_weights(self.c2, init_type='normal')
        #init_weights(self.c3, init_type='normal')

    def forward(self, content, style):
        batch, channels, width, height = content.size()
        norm_content = normalize(content)
        norm_style = normalize(style)

        c = self.c1(norm_content).view(batch, -1, width * height).permute(0, 2, 1)
        s1 = self.c2(norm_style).view(batch, -1, width * height)
        s2 = self.c3(style).view(batch, -1, width * height)

        attn = self.softmax(torch.bmm(c, s1))

        return torch.bmm(s2, attn.permute(0, 2, 1)).view(batch, channels, height, width)


class CBR(nn.Module):
    def __init__(self, in_ch, out_ch, up=False):
        super(CBR, self).__init__()
        self.up = up

        if up:
            self.cbr = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(in_ch, out_ch, 3),
                #nn.InstanceNorm2d(out_ch),
                nn.ReLU()
            )

        else:
            self.cbr = nn.Sequential(
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.Conv2d(in_ch, out_ch, 3),
                #nn.InstanceNorm2d(out_ch),
                nn.ReLU()
            )

        #init_weights(self.cbr, init_type='normal')

    def forward(self, x):
        return self.cbr(x)


class Decoder(nn.Module):
    def __init__(self, base=64):
        super(Decoder, self).__init__()

        self.dec = nn.Sequential(
            CBR(base*8, base*4),
            CBR(base*4, base*4, up=True),
            CBR(base*4, base*4),
            CBR(base*4, base*4),
            CBR(base*4, base*2, up=True),
            CBR(base*2, base*2),
            CBR(base*2, base, up=True),
            CBR(base, base),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(base, 3, 3),
            nn.Tanh()
        )

        #init_weights(self.dec, init_type='normal')

    def forward(self, x):
        return self.dec(x)


class Model(nn.Module):
    def __init__(self, base=512):
        super(Model, self).__init__()

        self.vgg4 = Vgg19(layer='four')
        self.vgg5 = Vgg19(layer='five')
        self.sa4 = SANet()
        self.sa5 = SANet()
        self.c0 = nn.Conv2d(base, base, 1, 1, 0)
        self.c1 = nn.Conv2d(base, base, 1, 1, 0)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.c2 = nn.Conv2d(base, base, 3)
        self.in0 = nn.InstanceNorm2d(base)
        self.dec = Decoder()
        self.vgg = Vgg19()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.relu = nn.ReLU()

    def forward(self, content, style):
        c4 = self.vgg4(content)
        s4 = self.vgg4(style)
        c5 = self.vgg5(content)
        s5 = self.vgg5(style)

        h4 = self.relu(self.c0(self.relu(self.sa4(c4, s4)) + c4))
        h5 = self.relu(self.c1(self.relu(self.sa5(c5, s5)) + c5))

        h = self.c2(self.pad((h4 + self.up(h5))))
        h = self.relu(h)
        h = self.dec(h)

        h_dec = self.vgg(h)
        h_style = self.vgg(style)

        return c4, c5, h_dec, h_style, h