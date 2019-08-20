import torch
import torch.nn as nn
import argparse

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import UGATITDataset, ImageCollate
from model import Generator, Discriminator, RhoClipper

mseloss = nn.MSELoss()
l1loss = nn.L1Loss()
bceloss = nn.BCEWithLogitsLoss()


def discriminator_loss(fake, real, fake_logit, real_logit):
    adv_loss = mseloss(real, torch.ones_like(real)) + mseloss(fake, torch.zeros_like(fake))
    adv_loss += mseloss(real_logit, torch.ones_like(real_logit)) + mseloss(fake_logit, torch.zeros_like(fake_logit))

    return adv_loss


def generator_loss(fake, real, fake_logit, fake_g_logit,
                   fake_xyx, fake_id, fake_id_logit):
    adv_loss = mseloss(fake, torch.ones_like(fake))
    adv_loss += mseloss(fake_logit, torch.ones_like(fake_logit))

    cycle_loss = l1loss(fake_xyx, real)
    identity_loss = l1loss(fake_id, real)

    cam_loss = bceloss(fake_g_logit, torch.ones_like(fake_g_logit))
    cam_loss += bceloss(fake_id_logit, torch.zeros_like(fake_id_logit))

    return adv_loss + 10 * cycle_loss + 10 * identity_loss + 1000 * cam_loss


def train(epochs, s_path, t_path, batchsize, interval):
    dataset = UGATITDataset(s_path, t_path)
    print(dataset)
    collator = ImageCollate()

    generator_st = Generator()
    generator_st.cuda()
    generator_st.train()
    optim_gen_st = torch.optim.Adam(generator_st.parameters(), lr=0.0001, betas=(0.5, 0.999))

    generator_ts = Generator()
    generator_ts.cuda()
    generator_ts.train()
    optim_gen_ts = torch.optim.Adam(generator_ts.parameters(), lr=0.0001, betas=(0.5, 0.999))

    discriminator_gt = Discriminator()
    discriminator_gt.cuda()
    discriminator_gt.train()
    optim_dis_gt = torch.optim.Adam(discriminator_gt.parameters(), lr=0.0001, betas=(0.5, 0.999))

    discriminator_gs = Discriminator()
    discriminator_gs.cuda()
    discriminator_gs.train()
    optim_dis_gs = torch.optim.Adam(discriminator_gs.parameters(), lr=0.0001, betas=(0.5, 0.999))

    #discriminator_rt = Discriminator()
    #discriminator_rt.cuda()
    #discriminator_rt.train()
    #optim_dis_rt = torch.optim.Adam(discriminator_rt.parameters(), lr=0.0001, betas=(0.5, 0.999))

    #discriminator_rs = Discriminator()
    #discriminator_rs.cuda()
    #discriminator_rs.train()
    #optim_dis_rs = torch.optim.Adam(discriminator_rs.parameters(), lr=0.0001, betas=(0.5, 0.999))

    clipper = RhoClipper(0, 1)

    iteration = 0

    for epoch in range(epochs):
        dataloader = DataLoader(dataset,
                                batch_size=batchsize,
                                shuffle=True,
                                collate_fn=collator,
                                drop_last=True)
        progress_bar = tqdm(dataloader)

        for i, data in enumerate(progress_bar):
            iteration += 1
            s, t = data

            fake_t, _, _ = generator_st(s)
            fake_s, _, _ = generator_ts(t)

            real_gs, real_gs_logit, _ = discriminator_gs(s)
            real_gt, real_gt_logit, _ = discriminator_gt(t)
            fake_gs, fake_gs_logit, _ = discriminator_gs(fake_s)
            fake_gt, fake_gt_logit, _ = discriminator_gt(fake_t)

            loss = discriminator_loss(fake_gt, real_gt, fake_gt_logit, real_gt_logit)
            loss += discriminator_loss(fake_gs, real_gs, fake_gs_logit, real_gs_logit)

            optim_dis_gs.zero_grad()
            optim_dis_gt.zero_grad()
            loss.backward()
            optim_dis_gs.step()
            optim_dis_gt.step()

            fake_t, fake_gen_t_logit, _ = generator_st(s)
            fake_s, fake_gen_s_logit, _ = generator_ts(t)

            fake_sts, _, _ = generator_ts(fake_t)
            fake_tst, _, _ = generator_st(fake_s)

            fake_t_id, fake_t_id_logit, _ = generator_st(t)
            fake_s_id, fake_s_id_logit, _ = generator_ts(s)

            fake_gs, fake_gs_logit, _ = discriminator_gs(fake_s)
            fake_gt, fake_gt_logit, _ = discriminator_gt(fake_t)

            loss = generator_loss(fake_gs, s, fake_gs_logit, fake_gen_s_logit, fake_sts, fake_s_id, fake_s_id_logit)
            loss += generator_loss(fake_gt, t, fake_gt_logit, fake_gen_t_logit, fake_tst, fake_t_id, fake_t_id_logit)

            optim_gen_st.zero_grad()
            optim_gen_ts.zero_grad()
            loss.backward()
            optim_gen_st.step()
            optim_gen_ts.step()

            generator_st.apply(clipper)
            generator_ts.apply(clipper)

            if iteration % interval == 0:
                torch.save(generator_st.state_dict(), f"./model/model_st_{iteration}.pt")
                torch.save(generator_ts.state_dict(), f"./model/model_ts_{iteration}.pt")

            print(f"iteration: {iteration} Loss: {loss.data}")


if __name__ == "__main__":
    model_dir = Path('./model')
    model_dir.mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(description="UGATIT")
    parser.add_argument('--e', default=1000, type=int, help="the number of epochs")
    parser.add_argument('--b', default=16, type=int, help="batch size")
    parser.add_argument('--i', default=1000, type=int, help="the interval of snapshot")

    args = parser.parse_args()

    c_path = Path('./img_align_celeba/')
    s_path = Path('./waifu_getchu/')

    train(args.e, c_path, s_path, args.b, args.i)