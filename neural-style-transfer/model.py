import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda,Chain
import numpy as np

xp = cuda.cupy

class VGG_content(Chain):
    def __init__(self):
        super(VGG_content, self).__init__()

        with init_scope():
            self.base = L.VGG16Layers()

    def __call__(self, x):
        h = self.base(x, layers=['conv5_2'])['conv5_2']

        return h

class VGG_style(Chain):
    def __init__(self):
        super(VGG_style, self).__init__()

        with init_scope():
            self.base = L.VGG16Layers()

    def __call__(self, x):
        h1 = self.base(x, layers=['conv1_1'])['conv1_1']
        h2 = self.base(x, layers=['conv2_1'])['conv2_1'] 
        h3 = self.base(x, layers=['conv3_1'])['conv3_1']
        h4 = self.base(x, layers=['conv4_1'])['conv4_1']
        h5 = self.base(x, layers=['conv5_1'])['conv5_1']

        return h1,h2,h3,h4,h5