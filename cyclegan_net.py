import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, cuda, Chain

class Generator_ResBlock6(Chain):
	def __init__(self):
		w = chainer.initializers.Normal(0.02)
		super(Generator_ResBlock6,self).__init__(
			conv1 = L.Convolution2D(3,32,7,1,3,initialW =w),
			conv2 = L.Convolution2D(32,64,4,2,1,initialW = w),
			conv3 = L.Convolution2D(64,128,4,2,1,initialW = w),
			conv4_1 = L.Convolution2D(128,128,3,1,1),
			conv4_2 = L.Convolution2D(128,128,3,1,1),
			conv5_1 = L.Convolution2D(128,128,3,1,1),
			conv5_2 = L.Convolution2D(128,128,3,1,1),
			conv6_1 = L.Convolution2D(128,128,3,1,1),
			conv6_2 = L.Convolution2D(128,128,3,1,1),
			conv7_1 = L.Convolution2D(128,128,3,1,1),
			conv7_2 = L.Convolution2D(128,128,3,1,1),
			conv8_1 = L.Convolution2D(128,128,3,1,1),
			conv8_2 = L.Convolution2D(128,128,3,1,1),
			conv9_1 = L.Convolution2D(128,128,3,1,1),
			conv9_2 = L.Convolution2D(128,128,3,1,1),
			conv10 = L.Convolution2D(128,64,3,1,1,initialW = w),
			conv11 = L.Convolution2D(64,32,3,1,1,initialW = w),
			conv12 = L.Convolution2D(32,3,7,1,3,initialW = w),

			bnc1 = L.BatchNormalization(32),
			bnc2 = L.BatchNormalization(64),
			bnc3 = L.BatchNormalization(128),
			bnc4_1 = L.BatchNormalization(128),
			bnc4_2 = L.BatchNormalization(128),
			bnc5_1 = L.BatchNormalization(128),
			bnc5_2 = L.BatchNormalization(128),
			bnc6_1 = L.BatchNormalization(128),
			bnc6_2 = L.BatchNormalization(128),
			bnc7_1 = L.BatchNormalization(128),
			bnc7_2 = L.BatchNormalization(128),
			bnc8_1 = L.BatchNormalization(128),
			bnc8_2 = L.BatchNormalization(128),
			bnc9_1 = L.BatchNormalization(128),
			bnc9_2 = L.BatchNormalization(128),
			bnc10 = L.BatchNormalization(64),
			bnc11 = L.BatchNormalization(32),
			)
	def forward(self,x):
		h = F.relu(self.bnc1(self.conv1(x)))
		h = F.relu(self.bnc2(self.conv2(h)))
		h = F.relu(self.bnc3(self.conv3(h)))
		h4_1 = F.relu(self.bnc4_1(self.conv4_1(h)))
		h = F.relu(self.bnc4_2(self.conv4_2(h4_1))) + h
		h5_1 = F.relu(self.bnc5_1(self.conv5_1(h)))
		h = F.relu(self.bnc5_2(self.conv5_2(h5_1))) + h
		h6_1 = F.relu(self.bnc6_1(self.conv6_1(h)))
		h = F.relu(self.bnc6_2(self.conv6_2(h6_1))) + h
		h7_1 = F.relu(self.bnc7_1(self.conv7_1(h)))
		h = F.relu(self.bnc7_2(self.conv7_2(h7_1))) + h
		h8_1 = F.relu(self.bnc8_1(self.conv8_1(h)))
		h = F.relu(self.bnc8_2(self.conv8_2(h8_1))) + h
		h9_1 = F.relu(self.bnc9_1(self.conv9_1(h)))
		h = F.relu(self.bnc9_2(self.conv9_2(h9_1))) + h
		h = F.unpooling_2d(h,2,2,0,cover_all = False)
		h = F.relu(self.bnc10(self.conv10(h)))
		h = F.unpooling_2d(h,2,2,0,cover_all = False)
		h = F.relu(self.bnc11(self.conv11(h)))
		h = F.tanh(self.conv12(h))

		return h

class Discriminator(Chain):
	def __init__(self):
		w = chainer.initializers.Normal(0.02)
		super(Discriminator,self).__init__(
			conv1 = L.Convolution2D(3,64,4,2,1,initialW = w),
			conv2 = L.Convolution2D(64,128,4,2,1,initialW = w),
			conv3 = L.Convolution2D(128,256,4,2,1,initialW = w),
			conv4 = L.Convolution2D(256,512,4,2,1,initialW = w),
			conv5 = L.Convolution2D(512,1,initialW = w),

			bnc1 = L.BatchNormalization(64),
			bnc2 = L.BatchNormalization(128),
			bnc3 = L.BatchNormalization(256),
			bnc4 = L.BatchNormalization(512),
			)

	def forward(self,x):
		h = F.leaky_relu(self.bnc1(self.conv1(x)))
		h = F.leaky_relu(self.bnc2(self.conv2(h)))
		h = F.leaky_relu(self.bnc3(self.conv3(h)))
		h = F.leaky_relu(self.bnc4(self.conv4(h)))
		h = F.leaky_relu(self.bnc1(self.conv1(x)))
		h = self.conv5(h)

		return h