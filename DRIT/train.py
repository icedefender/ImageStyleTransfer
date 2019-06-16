import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
import pylab

from model import Generator, ContentDiscriminator, DomainDiscriminator
from dataset import CollateFn, HairDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable


def adversarial_content_D(discriminator, content_x, content_y):
    fake = discriminator.forward(content_x.detach())
    real = discriminator.forward(content_y.detach())

    fake = F.sigmoid(fake)
    real = F.sigmoid(real)

    ones = Variable(torch.cuda.FloatTensor(fake.shape[0], 1).fill_(1.0), requires_grad=False)
    zeros = Variable(torch.cuda.FloatTensor(real.shape[0], 1).fill_(1.0), requires_grad=False)

    loss = nn.BCELoss()(real, ones) + nn.BCELoss()(fake, zeros)

    return loss


def adversarial_content_G(discriminator, content_x, content_y):
    fake_x = discriminator.forward(content_x)
    fake_y = discriminator.forward(content_y)

    fake_x = F.sigmoid(fake_x)
    fake_y = F.sigmoid(fake_y)

    half_x = 0.5 * torch.ones((fake_x.size(0))).cuda()
    half_y = 0.5 * torch.ones((fake_y.size(0))).cuda()

    loss = nn.BCELoss()(fake_x, half_x) + nn.BCELoss()(fake_y, half_y)

    return loss


def adversarial_domain_D(discriminator, pred_fake, pred_real):
    fake = discriminator.forward(pred_fake.detach())
    real = discriminator.forward(pred_real)

    fake = F.sigmoid(fake)
    real = F.sigmoid(real)

    ones = Variable(torch.cuda.FloatTensor(fake.shape[0], 1).fill_(1.0), requires_grad=False)
    zeros = Variable(torch.cuda.FloatTensor(real.shape[0], 1).fill_(1.0), requires_grad=False)

    loss = nn.BCELoss()(real, ones) + nn.BCELoss()(fake, zeros)

    return loss


def adversarial_domain_G(discriminator, pred_fake):
    fake = discriminator.forward(pred_fake)
    fake = F.sigmoid(fake)

    ones = Variable(torch.cuda.FloatTensor(fake.shape[0], 1).fill_(1.0), requires_grad=False)
    loss = nn.BCELoss()(fake, ones)

    return loss


def cross_cycle_consistency_loss(x, y, fake_x, fake_y):
    return nn.L1Loss()(x, fake_x) + nn.L1Loss()(y, fake_y)


def l2_regularize(x):
    x_2 = torch.pow(x, 2)
    loss = torch.mean(x_2)

    return loss


def train(epochs, batchsize, s_interval, c_weight, kl_weight, x_path, y_path):
    generator = Generator()
    generator.cuda()
    generator.train()

    content_discriminator = ContentDiscriminator()
    content_discriminator.cuda()
    content_discriminator.train()

    domain_x_discriminator = DomainDiscriminator()
    domain_x_discriminator.cuda()
    domain_x_discriminator.train()

    domain_y_discriminator = DomainDiscriminator()
    domain_y_discriminator.cuda()
    domain_y_discriminator.train()

    g_optim = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    cdis_optim = torch.optim.Adam(content_discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    ddis_x_optim = torch.optim.Adam(domain_x_discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    ddis_y_optim = torch.optim.Adam(domain_y_discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

    dataset = HairDataset(medium_path=x_path, twin_path=y_path)
    collator = CollateFn()

    iterations = 0

    for epoch in range(epochs):
        dataloader = DataLoader(dataset,
                                                           batch_size=batchsize,
                                                           shuffle=True,
                                                           collate_fn=collator.train,
                                                           drop_last=True,
                                                           num_workers=0)
        progress_bar = tqdm(dataloader)

        for index, data in enumerate(progress_bar):
            iterations += 1
            x, y = data

            # discriminator update
            enc_x, enc_y, _, _, fake_x, fake_y, _, _, infers_x, infers_y = generator.forward(x, y)
            _, infer_x, _ = infers_x
            _, infer_y, _ = infers_y
            dis_loss = adversarial_content_D(content_discriminator, enc_x, enc_y)
            dis_loss += adversarial_domain_D(domain_x_discriminator, fake_x, x)
            dis_loss += adversarial_domain_D(domain_y_discriminator ,fake_y, y)
            dis_loss += adversarial_domain_D(domain_x_discriminator, infer_x, x)
            dis_loss += adversarial_domain_D(domain_y_discriminator, infer_y, y)

            cdis_optim.zero_grad()
            ddis_x_optim.zero_grad()
            ddis_y_optim.zero_grad()
            dis_loss.backward()
            cdis_optim.step()
            ddis_x_optim.step()
            ddis_y_optim.step()

            # generator update
            enc_x, enc_y, attr_x, attr_y, fake_x, fake_y, recon_x, recon_y, infers_x, infers_y = generator.forward(x ,y)
            latent_x, infer_x, infer_attr_x = infers_x
            latent_y, infer_y, infer_attr_y = infers_y
            _, _, _, _, fake_xyx, fake_yxy, _, _, _, _ = generator.forward(fake_x, fake_y)
            gen_loss = adversarial_content_G(content_discriminator, enc_x, enc_y)
            gen_loss += adversarial_domain_G(domain_x_discriminator, fake_x)
            gen_loss += adversarial_domain_G(domain_y_discriminator, fake_y)
            gen_loss += adversarial_domain_G(domain_x_discriminator, infer_x)
            gen_loss += adversarial_domain_G(domain_y_discriminator, infer_y)
            gen_loss += c_weight * cross_cycle_consistency_loss(x, y, fake_xyx, fake_yxy)
            gen_loss += c_weight * cross_cycle_consistency_loss(x, y, recon_x, recon_y)
            gen_loss += c_weight * cross_cycle_consistency_loss(latent_x, latent_y, infer_attr_x, infer_attr_y)
            #gen_loss += kl_weight * (l2_regularize(attr_x) + l2_regularize(attr_y))

            g_optim.zero_grad()
            gen_loss.backward()
            g_optim.step()

            if iterations % s_interval == 1:
                torch.save(generator.state_dict(), './model/model_{}.pt'.format(iterations))

                pylab.rcParams['figure.figsize'] = (16.0,16.0)
                pylab.clf()

                with torch.no_grad():
                    _, _, _, _, _, fake_y, _, _, _, _ = generator.forward(x, y)
                    fake_y = fake_y[:2].detach().cpu().numpy()
                    real_x = x[:2].detach().cpu().numpy()

                    for i in range(1):
                        tmp = (np.clip(real_x[i] * 127.5 + 127.5, 0, 255)).transpose(1, 2, 0).astype(np.uint8)
                        pylab.subplot(2, 2, 2 * i + 1)
                        pylab.imshow(tmp)
                        pylab.axis("off")
                        pylab.savefig("outdir/visualize_{}.png".format(iterations))
                        tmp = (np.clip(fake_y[i] * 127.5 + 127.5, 0, 255)).transpose(1, 2, 0).astype(np.uint8)
                        pylab.subplot(2, 2, 2 * i + 2)
                        pylab.imshow(tmp)
                        pylab.axis("off")
                        pylab.savefig("outdir/visualize_{}.png".format(iterations))

            print('iteration: {} dis loss: {} gen loss: {}'.format(iterations, dis_loss, gen_loss))

if __name__ == "__main__":
    model_dir = './model/'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    visualize_dir = './outdir/'
    if not os.path.exists(visualize_dir):
        os.mkdir(visualize_dir)

    parser = argparse.ArgumentParser(description='DRIT')
    parser.add_argument('--e', type=int, default=1000, help="the number of epochs")
    parser.add_argument('--b', type=int, default=16, help="batch size")
    parser.add_argument('--i', type=int, default=2000, help="the interval os snapshot")
    parser.add_argument('--cw', type=float, default=10.0, help='the weight of content loss')
    parser.add_argument('--kw', type=float, default=0.01, help="the weight of kl-divergence loss")

    args = parser.parse_args()
    epochs = args.e
    batchsize = args.b
    interval = args.i
    c_weight = args.cw
    kl_weight = args.kw

    x_path = './face_blackhair/'
    y_path = './face_whitehair/'

    train(epochs, batchsize, interval, c_weight, kl_weight, x_path, y_path)