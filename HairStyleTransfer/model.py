import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda,Chain,initializers
import numpy as np
from instance_normalization import InstanceNormalization

xp=cuda.cupy
cuda.get_device(0).use()

class CBR(Chain):
    def __init__(self,in_ch,out_ch,up=False,down=False,predict=False,activation=F.relu):
        super(CBR,self).__init__()
        w=initializers.Normal(0.02)
        self.up=up
        self.down=down
        self.activation=activation
        self.predict=predict
        with self.init_scope():
            self.cpara=L.Convolution2D(in_ch,out_ch,3,1,1,initialW=w)
            self.cdown=L.Convolution2D(in_ch,out_ch,4,2,1,initialW=w)

            self.bn0=L.BatchNormalization(out_ch)
            self.in0=InstanceNormalization(out_ch)

    def __call__(self,x):
        if x.shape[0] == 1:
            if self.up:
                h=F.unpooling_2d(x,2,2,0,cover_all=False)
                h=self.activation(self.in0(self.cpara(h)))

            elif self.down:
                h=self.activation(self.in0(self.cdown(x)))

            else:
                h=self.activation(self.in0(self.cpara(x)))

        else:
            if self.up:
                h=F.unpooling_2d(x,2,2,0,cover_all=False)
                h=self.activation(self.bn0(self.cpara(h)))

            elif self.down:
                h=self.activation(self.bn0(self.cdown(x)))

            else:
                h=self.activation(self.bn0(self.cpara(x)))

        return h

class ResBlock(Chain):
    def __init__(self,in_ch,out_ch):
        super(ResBlock,self).__init__()
        with self.init_scope():
            self.cbr0=CBR(in_ch,out_ch)
            self.cbr1=CBR(out_ch,out_ch)

    def __call__(self,x):
        h=self.cbr0(x)
        h=self.cbr1(h)

        return h+x

class Generator(Chain):
    def __init__(self,base=32):
        super(Generator,self).__init__()
        w=initializers.Normal(0.02)
        with self.init_scope():
            self.c0=L.Convolution2D(3+3,base,7,1,3,initialW=w)
            self.cbr0=CBR(base,base*2,down=True)
            self.cbr1=CBR(base*2,base*4,down=True)
            self.res0=ResBlock(base*4,base*4)
            self.res1=ResBlock(base*4,base*4)
            self.res2=ResBlock(base*4,base*4)
            self.res3=ResBlock(base*4,base*4)
            self.res4=ResBlock(base*4,base*4)
            self.res5=ResBlock(base*4,base*4)
            self.cbr2=CBR(base*4,base*2,up=True)
            self.cbr3=CBR(base*2,base,up=True)
            self.c1=L.Convolution2D(base,3+3,7,1,3,initialW=w)

            self.bn0=L.BatchNormalization(base)
            self.in0=InstanceNormalization(base)

    def __call__(self,x):
        if x.shape[0]==1:
            h=F.relu(self.in0(self.c0(x)))
        else:
            h=F.relu(self.bn0(self.c0(x)))
        h=self.cbr0(h)
        h=self.cbr1(h)
        h=self.res0(h)
        h=self.res1(h)
        h=self.res2(h)
        h=self.res3(h)
        h=self.res4(h)
        h=self.res5(h)
        h=self.cbr2(h)
        h=self.cbr3(h)
        h=self.c1(h)

        return F.tanh(h)

class Discriminator(Chain):
    def __init__(self,base=32):
        w = initializers.Normal(0.02)
        super(Discriminator,self).__init__()
        with self.init_scope():
            self.cbr0=CBR(6,base,down=True,activation=F.leaky_relu)
            self.cbr1=CBR(base,base*2,down=True,activation=F.leaky_relu)
            #self.cbr1_1=CBR(base*2,base*2,activation=F.leaky_relu)
            self.cbr2=CBR(base*2,base*4,down=True,activation=F.leaky_relu)
            #self.cbr2_1=CBR(base*4,base*4,activation=F.leaky_relu)
            self.cbr3=CBR(base*4,base*8,down=True,activation=F.leaky_relu)
            #self.cbr3_1=CBR(base*8,base*8,activation=F.leaky_relu)
            self.cout=L.Convolution2D(base*8,1,3,1,1,initialW=w)

    def __call__(self,x):
        h=self.cbr0(x)
        h=self.cbr1(h)
        #h=self.cbr1_1(h)
        h=self.cbr2(h)
        #h=self.cbr2_1(h)
        h=self.cbr3(h)
        #h=self.cbr3_1(h)
        h=self.cout(h)

        return h

class UNet(Chain):
    def __init__(self,base=64):
        super(UNet,self).__init__()
        w=initializers.Normal(0.02)
        with self.init_scope():
            self.c0=L.Convolution2D(6,base,3,1,1,initialW=w)
            self.cbr0=CBR(base,base*2,down=True,predict=True)
            self.cbr1=CBR(base*2,base*4,down=True,predict=True)
            self.cbr2=CBR(base*4,base*8,down=True,predict=True)
            self.cbr3=CBR(base*8,base*8,down=True,predict=True)
            self.cbr4=CBR(base*16,base*8,up=True,predict=True)
            self.cbr5=CBR(base*16,base*4,up=True,predict=True)
            self.cbr6=CBR(base*8,base*2,up=True,predict=True)
            self.cbr7=CBR(base*4,base*1,up=True,predict=True)
            self.c1=L.Convolution2D(base*2,3,3,1,1,initialW=w)

            self.bn0=InstanceNormalization(base)

    def __call__(self,x):
        h1=F.relu(self.bn0(self.c0(x)))
        h2=self.cbr0(h1)
        h3=self.cbr1(h2)
        h4=self.cbr2(h3)
        h5=self.cbr3(h4)
        h=self.cbr4(F.concat([h5,h5]))
        h=self.cbr5(F.concat([h4,h]))
        h=self.cbr6(F.concat([h3,h]))
        h=self.cbr7(F.concat([h2,h]))
        h=self.c1(F.concat([h1,h]))

        return F.tanh(h)