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
            self.c0_img=L.Convolution2D(3,base,7,1,3,initialW=w)
            self.cbr0_img=CBR(base,base*2,down=True)
            self.cbr1_img=CBR(base*2,base*4,down=True)
            self.res0_img=ResBlock(base*4,base*4)
            self.res1_img=ResBlock(base*4,base*4)
            self.res2_img=ResBlock(base*4,base*4)
            self.res3_img=ResBlock(base*4,base*4)
            self.res4_img=ResBlock(base*4,base*4)
            self.res5_img=ResBlock(base*4,base*4)
            self.res6_img=ResBlock(base*4,base*4)
            self.res7_img=ResBlock(base*4,base*4)
            self.res8_img=ResBlock(base*4,base*4)
            self.cbr2_img=CBR(base*8,base*2,up=True)
            self.cbr3_img=CBR(base*2,base,up=True)
            self.c1_img=L.Convolution2D(base,3,7,1,3,initialW=w)

            self.bn0_img=L.BatchNormalization(base)
            self.in0_img=InstanceNormalization(base)

            self.c0_mask=L.Convolution2D(3,base,7,1,3,initialW=w)
            self.cbr0_mask=CBR(base,base*2,down=True)
            self.cbr1_mask=CBR(base*2,base*4,down=True)
            self.res0_mask=ResBlock(base*4,base*4)
            self.res1_mask=ResBlock(base*4,base*4)
            self.res2_mask=ResBlock(base*4,base*4)
            self.res3_mask=ResBlock(base*4,base*4)
            self.res4_mask=ResBlock(base*4,base*4)
            self.res5_mask=ResBlock(base*4,base*4)
            self.res6_mask=ResBlock(base*4,base*4)
            self.res7_mask=ResBlock(base*4,base*4)
            self.res8_mask=ResBlock(base*4,base*4)
            self.cbr2_mask=CBR(base*12,base*2,up=True)
            self.cbr3_mask=CBR(base*2,base,up=True)
            self.c1_mask=L.Convolution2D(base,3,7,1,3,initialW=w)

            self.bn0_mask=L.BatchNormalization(base)
            self.in0_mask=InstanceNormalization(base)

    def encode_img(self,x):
        if x.shape[0]==1:
            h=F.relu(self.in0_img(self.c0_img(x)))
        else:
            h=F.relu(self.bn0_img(self.c0_img(x)))
        h=self.cbr0_img(h)
        h=self.cbr1_img(h)
        h=self.res0_img(h)
        h=self.res1_img(h)
        h=self.res2_img(h)
        h=self.res3_img(h)
        h=self.res4_img(h)
        h=self.res5_img(h)
        h=self.res6_img(h)
        h=self.res7_img(h)
        h=self.res8_img(h)

        return h

    def decode_img(self,x):
        h=self.cbr2_img(x)
        h=self.cbr3_img(h)
        h=self.c1_img(h)

        return F.tanh(h)

    def encode_mask(self,x):
        if x.shape[0]==1:
            h=F.relu(self.in0_mask(self.c0_mask(x)))
        else:
            h=F.relu(self.bn0_mask(self.c0_mask(x)))
        h=self.cbr0_mask(h)
        h=self.cbr1_mask(h)
        h=self.res0_mask(h)
        h=self.res1_mask(h)
        h=self.res2_mask(h)
        h=self.res3_mask(h)
        h=self.res4_mask(h)
        h=self.res5_mask(h)
        h=self.res6_mask(h)
        h=self.res7_mask(h)
        h=self.res8_mask(h)

        return h

    def decode_mask(self,x):
        h=self.cbr2_mask(x)
        h=self.cbr3_mask(h)
        h=self.c1_mask(h)

        return F.tanh(h)

    def __call__(self,img,mask):
        enc_img = self.encode_img(img)
        enc_mask = self.encode_mask(mask)

        latent_img = F.concat([enc_img, enc_mask])
        latent_mask = F.concat([enc_img, enc_mask, enc_mask])

        dec_img = self.decode_img(latent_img)
        dec_mask = self.decode_mask(latent_mask)

        return dec_img, dec_mask

class Discriminator(Chain):
    def __init__(self,base=32):
        w = initializers.Normal(0.02)
        super(Discriminator,self).__init__()
        with self.init_scope():
            self.cbr0_img=CBR(3,base,down=True,activation=F.leaky_relu)
            self.cbr1_img=CBR(base,base*2,down=True,activation=F.leaky_relu)
            self.cbr2_img=CBR(base*2,base*4,down=True,activation=F.leaky_relu)
            
            self.cbr0_mask=CBR(3,base,down=True,activation=F.leaky_relu)
            self.cbr1_mask=CBR(base,base*2,down=True,activation=F.leaky_relu)
            self.cbr2_mask=CBR(base*2,base*4,down=True,activation=F.leaky_relu)

            self.cbr3=CBR(base*8,base*16,down=True,activation=F.leaky_relu)
            self.cout=L.Convolution2D(base*16,1,3,1,1,initialW=w)

    def encode_img(self,x):
        h=self.cbr0_img(x)
        h=self.cbr1_img(h)
        h=self.cbr2_img(h)

        return h

    def encode_mask(self,x):
        h=self.cbr0_mask(x)
        h=self.cbr1_mask(h)
        h=self.cbr2_mask(h)

        return h

    def __call__(self,img,mask):
        enc_img = self.encode_img(img)
        enc_mask = self.encode_mask(mask)
        h = self.cbr3(F.concat([enc_img, enc_mask]))
        h = self.cout(h)

        return h