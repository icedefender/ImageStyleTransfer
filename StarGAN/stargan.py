import numpy as np
import os
import pylab
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, cuda, initializers, serializers, optimizers
from stargan_net import Generator_ResBlock6, Discriminator

xp = cuda.cupy

image_out_dir = './output_color5'
if not os.path.exists(image_out_dir):
    os.mkdir(image_out_dir)

def set_optimizer(model, alpha, beta):
	optimizer = optimizers.Adam(alpha = alpha, beta1 = beta)
	optimizer.setup(model)
	optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001))
	return optimizer

def one_hot_x(domain_num,nc_size = 4):
    vec = xp.zeros(nc_size).astype(xp.int32)
    vec[domain_num] = 1
    return vec

def one_hot_y(domain_num, nc_size = 4):
    vec = xp.zeros(nc_size).astype(xp.int32)
    domain = np.random.randint(nc_size)
    while(domain == domain_num):
        domain = np.random.randint(nc_size)
    vec[domain] = 1
    return vec

train = np.load('target.npy').astype(np.float32)
label = xp.load('label.npy').astype(xp.int32)

epochs = 500
batchsize = 20
interval = 10
lambda_adv = 1.0
lambda_cls = 1.0
lambda_gp = 10.0
lambda_rec = 10.0
Ntrain = train.shape[0]
nc_size = 4
interval = 10

Ncolor = int(Ntrain / nc_size)

gen_model = Generator_ResBlock6(nc_size)
cuda.get_device(0).use()
gen_model.to_gpu()

dis_model = Discriminator()
cuda.get_device(0).use()
dis_model.to_gpu()

gen_opt = set_optimizer(gen_model, 0.0001, 0.5)
dis_opt = set_optimizer(dis_model, 0.0001, 0.5)

for epoch in range(epochs):
    sum_gen_loss = 0
    sum_dis_loss = 0
    for batch in range(0, Ntrain, batchsize):
        x = np.zeros((batchsize, 3,128,128)).astype(np.float32)
        x_domain = xp.zeros((batchsize, nc_size)).astype(np.int32)
        y_domain = xp.zeros((batchsize, nc_size)).astype(np.int32)
        start_point = 0
        for j in range(batchsize):
            if j % (batchsize/nc_size) == 0:
                start_point += 1
            rnd = np.random.randint(Ncolor * (start_point -1) , Ncolor * (start_point))
            x[j, :,:,:] = train[rnd]
            x_domain[j] = one_hot_x(label[rnd])
            y_domain[j] = one_hot_y(label[rnd])

        x_domain_in = x_domain.astype(xp.float32)
        y_domain_in = y_domain.astype(xp.float32)
        x_real = Variable(cuda.to_gpu(x))

        y_fake = gen_model(x_real, y_domain_in)
        x_fake = gen_model(y_fake, x_domain_in)

        out_real, out_real_domain = dis_model(x_real)
        out_fake, out_fake_domain = dis_model(y_fake)

        loss_gen_adv = F.sum(-out_fake) / batchsize
        loss_gen_cls = F.sigmoid_cross_entropy(out_fake_domain, Variable(y_domain))
        loss_gen_rec = F.mean_absolute_error(x_real, x_fake)

        loss_gen = lambda_adv*loss_gen_adv + lambda_cls * loss_gen_cls + lambda_rec * loss_gen_rec

        gen_model.cleargrads()
        loss_gen.backward()
        gen_opt.update()

        loss_dis_adv = F.sum(-out_real) / batchsize
        loss_dis_adv += F.sum(out_fake) / batchsize
        loss_dis_cls = F.sigmoid_cross_entropy(out_real_domain, Variable(x_domain))

        loss_dis = lambda_adv * loss_dis_adv + lambda_cls * loss_dis_cls

        dis_model.cleargrads()
        loss_dis.backward()
        dis_opt.update()

        eps = xp.random.uniform(0,1,size = batchsize).astype(xp.float32)[:, None, None, None]
        x_mid = eps * x_real + (1.0 - eps) * y_fake
        x_mid_v = Variable(x_mid.data)
        y_mid, _ = dis_model(x_mid_v)
        y_mid = F.sum(y_mid)
        dydx, = chainer.grad([y_mid],[x_mid_v],enable_double_backprop = True)
        dydx = F.sqrt(F.sum(dydx*dydx, axis = (1,2,3)))
        loss_gp = F.mean_squared_error(dydx, xp.ones_like(dydx.data))

        loss_dis = lambda_gp * loss_gp

        dis_model.cleargrads()
        loss_dis.backward()
        dis_opt.update()

        sum_dis_loss += loss_dis.data.get()
        sum_gen_loss += loss_gen.data.get()

        if epoch % interval == 0 and batch == 0:
            serializers.save_npz('starGAN_generator.model',gen_model)
            serializers.save_npz('starGAN_discriminator.model',dis_model)
            vector = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
            vector = xp.array(vector).astype(xp.int32)

            for color in range(nc_size):
                testimg = train[300+Ncolor*color]
                testimg_out = (testimg*127.5 + 127.5).transpose(1,2,0)
                testimg_out = testimg_out.astype(np.uint8)

                for index in range(nc_size + 1):
                    if index == 0:
                        pylab.subplot(nc_size,nc_size + 1, color*(nc_size + 1) + index+1)
                        pylab.imshow(testimg_out)
                        pylab.axis('off')
                        pylab.savefig('%s/convert_%d.png'%(image_out_dir, epoch))
                    
                    else:
                        pylab.subplot(nc_size,nc_size + 1,color*(nc_size+1) + index+1)
                        x = Variable(cuda.to_gpu(testimg))
                        x = x.reshape(1,3,128,128)
                        cls = vector[index-1].astype(xp.float32).reshape(1,nc_size)
                        with chainer.using_config('train', False):
                            y1 = gen_model(x,cls)
                        y1 = y1.data.get()
                        tmp = (np.clip(y1[0,:,:,:]*127.5 + 127.5, 0, 255)).transpose(1,2,0).astype(np.uint8)
                        pylab.imshow(tmp)
                        pylab.axis('off')
                        pylab.savefig('%s/convert_%d.png'%(image_out_dir, epoch))

    print('epoch:{} Dis:{} Gen:{}'.format(epoch, sum_dis_loss/Ntrain, sum_gen_loss/Ntrain))