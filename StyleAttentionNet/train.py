import torch
import torch.nn as nn
import numpy as np
import argparse
import os

from model import Model, normalize
from dataset import CSDataset, ImageCollate
from torch.utils.data import DataLoader
from tqdm import tqdm


def calc_content_loss(y4, y5, c4, c5):
    loss = nn.MSELoss()(normalize(y4), normalize(c4))
    loss += nn.MSELoss()(normalize(y5), normalize(c5))

    return loss


def calc_identity1_loss(cc, ss, c, s):
    loss = nn.MSELoss()(cc, c) + nn.MSELoss()(ss, s)

    return loss


def calc_identity2_loss(cc_list, ss_list, c_list, s_list):
    sum_loss = 0
    for cc, c in zip(cc_list, c_list):
        sum_loss += nn.MSELoss()(cc, c)

    for ss, s in zip(ss_list, s_list):
        sum_loss += nn.MSELoss()(ss, s)

    return sum_loss


def calc_mean_std(features, eps=1e-5):
    batch, channels, height, width = features.size()
    feature_mean = features.view(batch, channels, -1).mean(dim=2)
    feature_var = features.view(batch, channels, -1).var(dim=2) + eps
    feature_std = feature_var.sqrt()

    return feature_mean, feature_std


def calc_style_loss(y_list, s_list):
    sum_loss = 0
    for y, s in zip(y_list, s_list):
        y_mean, y_std = calc_mean_std(y)
        s_mean, s_std = calc_mean_std(s)

        sum_loss += nn.MSELoss()(y_mean, s_mean)
        sum_loss += nn.MSELoss()(y_std, s_std)

    return sum_loss


def train(content_path, style_path, epochs, batchsize, interval,
          c_weight, s_weight, i1_weight, i2_weight):
    dataset = CSDataset(c_path=content_path, s_path=style_path)
    collator = ImageCollate()

    model = Model()
    model.cuda()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    iterations = 0

    for epoch in range(epochs):
        dataloader = DataLoader(dataset,
                                batch_size=batchsize,
                                shuffle=True,
                                collate_fn=collator,
                                drop_last=True)
        progress_bar = tqdm(dataloader)

        for i, data in enumerate(progress_bar):
            iterations += 1
            c, s = data

            c4, c5, y_list, s_list, y = model(c, s)
            _, _, yc_list, cc_list, yc = model(c, c)
            _, _, ys_list, ss_list, ys = model(s, s)

            loss = c_weight * calc_content_loss(y_list[3], y_list[4], c4, c5)
            loss += s_weight * calc_style_loss(y_list, s_list)
            loss += i1_weight * calc_identity1_loss(yc, ys, c, s)
            loss += i2_weight * calc_identity2_loss(yc_list, ys_list, cc_list, ss_list)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iterations % interval == 0:
                torch.save(model.state_dict(), './model/model_{}.pt'.format(iterations))
                
            print('iteration: {} Loss: {}'.format(iterations, loss.data[0]))


if __name__ == "__main__":
    model_dir = './model/'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    parser = argparse.ArgumentParser(description='StyleAttentionNetwork')
    parser.add_argument('--e', default=1000, type=int, help="the number of epochs")
    parser.add_argument('--b', default=16, type=int, help="batch size")
    parser.add_argument('--cw', default=1.0, type=float, help="the weight of content loss")
    parser.add_argument('--sw', default=3.0, type=float, help="the weight of style loss")
    parser.add_argument('--iw1', default=1.0, type=float, help="the weight of identity loss1")
    parser.add_argument('--iw2', default=50.0, type=float, help="the weight of identity loss2")
    parser.add_argument('--i', default=1000, type=int, help="the interval of snapshot")

    args = parser.parse_args()
    epochs = args.e
    batchsize = args.b
    c_weight = args.cw
    s_weight = args.sw
    i1_weight = args.iw1
    i2_weight = args.iw2
    interval = args.i

    content_path = './coco2015/test2015/'
    style_path = './Dataset/wikiart/'

    train(content_path, style_path, epochs, batchsize, interval,
          c_weight, s_weight, i1_weight, i2_weight)