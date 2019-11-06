import torch
import torch.nn as nn
import numpy as np
import argparse

from torch.utils.data import DataLoader
from model import MUNIT, Discriminator, Vgg19Norm
from dataset import HairDataset, CollateFn
from pathlib import Path
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import pylab

l1loss = nn.L1Loss()
mseloss = nn.MSELoss()
softplus = nn.Softplus()


def reconstruction_loss(y, t):
    return l1loss(y, t)


def perceptual_loss(vgg, y, t):
    y_feat = vgg(y)
    t_feat = vgg(t)

    return mseloss(y_feat, t_feat)


def adversarial_dis_loss(discriminator, y, t):
    sum_loss = 0
    fake_list = discriminator(y.detach())
    real_list = discriminator(t)
    for fake, real in zip(fake_list, real_list):
        loss = torch.mean(softplus(-real)) + torch.mean(softplus(fake))
        sum_loss += loss

    return sum_loss


def adversarial_gen_loss(discriminator, y):
    sum_loss = 0
    fake_list = discriminator(y)
    for fake in fake_list:
        loss = torch.mean(softplus(-fake))
        sum_loss += loss

    return sum_loss


def train(epochs, batchsize, interval, c_path, s_path):
    # Dataset definition
    dataset = HairDataset(c_path, s_path)
    collator = CollateFn()

    # Model & Optimizer Definition
    munit = MUNIT()
    munit.cuda()
    munit.train()
    m_opt = torch.optim.Adam(munit.parameters(),
                             lr=0.0001,
                             betas=(0.5, 0.999),
                             weight_decay=0.0001)

    discriminator_a = Discriminator()
    discriminator_a.cuda()
    discriminator_a.train()
    da_opt = torch.optim.Adam(discriminator_a.parameters(),
                              lr=0.0001,
                              betas=(0.5, 0.999),
                              weight_decay=0.0001)

    discriminator_b = Discriminator()
    discriminator_b.cuda()
    discriminator_b.train()
    db_opt = torch.optim.Adam(discriminator_b.parameters(),
                              lr=0.0001,
                              betas=(0.5, 0.999),
                              weight_decay=0.0001)

    vgg = Vgg19Norm()
    vgg.cuda()
    vgg.train()

    iterations = 0

    for epoch in range(epochs):
        dataloader = DataLoader(dataset,
                                batch_size=batchsize,
                                shuffle=True,
                                drop_last=True,
                                collate_fn=collator)
        dataloader = tqdm(dataloader)

        for i, data in enumerate(dataloader):
            iterations += 1
            a, b = data
            _, _, _, _, _, _, ba, ab, _, _, _, _, _, _ = munit(a, b)

            loss = adversarial_dis_loss(discriminator_a, ba, a)
            loss += adversarial_dis_loss(discriminator_b, ab, b)

            da_opt.zero_grad()
            db_opt.zero_grad()
            loss.backward()
            da_opt.step()
            db_opt.step()

            c_a, s_a, c_b, s_b, a_recon, \
                b_recon, ba, ab, c_b_recon, s_a_recon, c_a_recon, s_b_recon, aba, bab = munit(a, b)

            loss = adversarial_gen_loss(discriminator_a, ba)
            loss += adversarial_gen_loss(discriminator_b, ab)
            loss += 10 * reconstruction_loss(a_recon, a)
            loss += 10 * reconstruction_loss(b_recon, b)
            loss += reconstruction_loss(c_a, c_a_recon)
            loss += reconstruction_loss(c_b, c_b_recon)
            loss += reconstruction_loss(s_a, s_a_recon)
            loss += reconstruction_loss(s_b, s_b_recon)
            loss += 10 * reconstruction_loss(aba, a)
            loss += 10 * reconstruction_loss(bab, b)
            loss += perceptual_loss(vgg, ba, b)
            loss += perceptual_loss(vgg, ab, a)

            m_opt.zero_grad()
            loss.backward()
            m_opt.step()

            if iterations % interval == 1:
                torch.save(munit.load_state_dict, f"./modeldir/model_{iterations}.pt")

                pylab.rcParams['figure.figsize'] = (16.0, 16.0)
                pylab.clf()

                munit.eval()

                with torch.no_grad():
                    _, _, _, _, _, _, _, ab, _, _, _, _, _, _ = munit(a, b)
                    fake = ab.detach().cpu().numpy()
                    real = a.detach().cpu().numpy()

                    for i in range(batchsize):
                        tmp = (np.clip(real[i] * 127.5 + 127.5, 0, 255)).transpose(1, 2, 0).astype(np.uint8)
                        pylab.subplot(4, 4, 2 * i + 1)
                        pylab.imshow(tmp)
                        pylab.axis("off")
                        pylab.savefig("outdir/visualize_{}.png".format(iterations))
                        tmp = (np.clip(fake[i] * 127.5 + 127.5, 0, 255)).transpose(1, 2, 0).astype(np.uint8)
                        pylab.subplot(4, 4, 2 * i + 2)
                        pylab.imshow(tmp)
                        pylab.axis("off")
                        pylab.savefig("outdir/visualize_{}.png".format(iterations))

                munit.train()

            print(f"iter: {iterations} loss: {loss.data}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MUNIT")
    parser.add_argument("--e", type=int, default=2000, help="the number of epochs")
    parser.add_argument("--b", type=int, default=4, help="batch size")
    parser.add_argument("--i", type=int, default=2000, help="the interval of snapshot")
    args = parser.parse_args()

    c_path = Path("./face_stargan/0/")
    s_path = Path("./face_stargan/5/")

    modeldir = Path("./modeldir")
    modeldir.mkdir(exist_ok=True)

    outdir = Path("./outdir")
    outdir.mkdir(exist_ok=True)

    train(args.e, args.b, args.i, c_path, s_path)