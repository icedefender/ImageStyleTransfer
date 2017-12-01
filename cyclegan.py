import numpy as np
import os
import pylab
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, cuda, initializers, serializers
from cyclegan_net import Generator_ResBlock6, Discriminator

xp = cuda.cupy

image_out_dir = './output'
if not os.path.exists(image_out_dir):
    os.mkdir(image_out_dir)

def cal_l2_sum(h,t):
	return F.sum((h-t)**2) / np.prod(h.data.shape)

def loss_func_adv_dis_fake(h):
	return cal_l2_sum(h,0.1)

def loss_func_adv_dis_real(h):
	return cal_l2_sum(h,0.9)

def loss_func_adv_gen(h):
	return cal_l2_sum(h,0.9)

def set_optimizer(model, alpha, beta):
	optimizer = chainer.optimizers.Adam(alpha = alpha, beta1 = beta)
	optimizer.setup(model)
	optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001))
	return optimizer

x_train = np.load('face_black.npy').astype(np.float32)[0:3000]
y_train = np.load('face_white.npy').astype(np.float32)[0:3000]
x_test = np.load('face_black.npy').astype(np.float32)[3010:3021]
y_test = np.load('face_white.npy').astype(np.float32)[3010:3021]

epochs = 500
batchsize = 15
interval = 10
Ntrain = x_train.shape[0]
Ntest = x_test.shape[0]
lambda1 = 10
lambda2 = 3

gen_g_model = Generator_ResBlock6()
cuda.get_device(0).use()
gen_g_model.to_gpu()

dis_x_model = Discriminator()
cuda.get_device(0).use()
dis_x_model.to_gpu()

gen_f_model = Generator_ResBlock6()
cuda.get_device(0).use()
gen_f_model.to_gpu()

dis_y_model = Discriminator()
cuda.get_device(0).use()
dis_y_model.to_gpu()

gen_g_opt = set_optimizer(gen_g_model, 0.002, 0.5)
dis_x_opt = set_optimizer(dis_x_model, 0.002, 0.5)
gen_f_opt = set_optimizer(gen_f_model, 0.002, 0.5)
dis_y_opt = set_optimizer(dis_y_model, 0.002, 0.5)

for epoch in range(epochs):
	sum_dis_x_loss = 0
	sum_dis_y_loss = 0
	sum_gen_loss = 0
	for batch in range(0, Ntrain, batchsize):
		x = np.zeros((batchsize, 3, 128, 128)).astype(np.float32)
		y = np.zeros((batchsize, 3, 128, 128)).astype(np.float32)

		for i in range(batchsize):
			rnd = np.random.randint(Ntrain)
			x[i,:,:,:] = x_train[rnd]
			y[i,:,:,:] = y_train[rnd]

		x = Variable(cuda.to_gpu(x))
		y = Variable(cuda.to_gpu(y))

		x_y = gen_g_model.forward(x)
		x_y_x = gen_f_model.forward(x_y)

		y_x = gen_f_model.forward(y)
		y_x_y = gen_g_model.forward(y_x)

		loss_dis_y_fake = loss_func_adv_dis_fake(dis_y_model.forward(x_y))
		loss_dis_y_real = loss_func_adv_dis_real(dis_y_model.forward(y))
		loss_dis_y = loss_dis_y_fake + loss_dis_y_real

		loss_dis_x_fake = loss_func_adv_dis_fake(dis_x_model.forward(y_x))
		loss_dis_x_real = loss_func_adv_dis_real(dis_x_model.forward(x))
		loss_dis_x = loss_dis_x_fake + loss_dis_x_real

		dis_y_model.cleargrads()
		loss_dis_y.backward()
		loss_dis_y.unchain_backward()
		dis_y_opt.update()

		dis_x_model.cleargrads()
		loss_dis_x.backward()
		loss_dis_x.unchain_backward()
		dis_x_opt.update()

		loss_gen_g_adv = loss_func_adv_gen(dis_y_model.forward(x_y))
		loss_gen_f_adv = loss_func_adv_gen(dis_x_model.forward(y_x))
		loss_cycle_x = lambda1*F.mean_absolute_error(x_y_x,x)
		loss_cycle_y = lambda1*F.mean_absolute_error(y_x_y,y)
		loss_gen = lambda2*loss_gen_g_adv + lambda2*loss_gen_f_adv + loss_cycle_y + loss_cycle_x

		gen_f_model.cleargrads()
		gen_g_model.cleargrads()
		loss_gen.backward()
		loss_gen.unchain_backward()
		gen_f_opt.update()
		gen_g_opt.update()

		sum_dis_y_loss += loss_dis_y.data.get()
		sum_dis_x_loss += loss_dis_x.data.get()
		sum_gen_loss += loss_gen.data.get()

		if epoch % interval == 0 and batch == 0:
			for i in range(Ntest):
				black = (x_test[i]*127.5 + 127.5).transpose(1,2,0)
				black = black.astype(np.uint8)

				pylab.imshow(black)
				pylab.axis('off')
				pylab.savefig('./output/raw_black_%d.png' %i)

				x = Variable(cuda.to_gpu(x_test[i]))
				x = x.reshape(1,3,128,128)
				with chainer.using_config('train', False):
					x_y = gen_g_model.forward(x)
				x_y = x_y.data.get()
				tmp = (np.clip(x_y[0,:,:,:]*127.5 + 127.5, 0, 255)).transpose(1,2,0).astype(np.uint8)
				pylab.imshow(tmp)
				pylab.axis('off')
				pylab.savefig( './output/convert_white_%d.png' %i)

				white = (y_test[i]*127.5 + 127.5).transpose(1,2,0)
				white = white.astype(np.uint8)
				pylab.imshow(white)
				pylab.axis('off')
				pylab.savefig('./output/raw_white_%d.png' %i)

				y = Variable(cuda.to_gpu(y_test[i]))
				y = y.reshape(1,3,128,128)
				with chainer.using_config('train', False):
					y_x = gen_f_model.forward(y)
				y_x = y_x.data.get()
				tmp = (np.clip(y_x[0,:,:,:]*127.5 + 127.5, 0, 255)).transpose(1,2,0).astype(np.uint8)
				pylab.imshow(tmp)
				pylab.axis('off')
				pylab.savefig( './output/convert_black_%d.png' %i)

	if epoch % 50 == 0:
		serializers.save_npz('black2white.model', gen_g_model)
		serializers.save_npz('white2black.model', gen_f_model)
			
	print('{} Dis_y:{} Dis_x:{} Gen:{}'.format(epoch, sum_dis_y_loss/Ntrain, sum_dis_x_loss/Ntrain, sum_gen_loss/Ntrain))

