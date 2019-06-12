import torch
import torch.nn as nn
import numpy as np
import argparse
import os

from model import Model
from dataset import CSTestDataset, ImageCollate
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import pylab


def test(content_path, style_path, model_path, batchsize):
    dataset = CSTestDataset(c_path=content_path, s_path=style_path)
    collator = ImageCollate(test=True)

    model = Model()
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()

    dataloader = DataLoader(dataset,
                            batch_size=batchsize,
                            shuffle=True,
                            collate_fn=collator,
                            drop_last=True)
    progress_bar = tqdm(dataloader)

    for index, data in enumerate(progress_bar):
        pylab.rcParams['figure.figsize'] = (16.0,16.0)
        pylab.clf()
        c, s = data

        with torch.no_grad():
            _, _, _, _, y = model(c, s)

        y = y.detach().cpu().numpy()
        c = c.detach().cpu().numpy()
        s = s.detach().cpu().numpy()

        for i in range(batchsize):
            tmp = (np.clip(c[i]*127.5+127.5, 0, 255)).transpose(1, 2, 0).astype(np.uint8)
            pylab.subplot(batchsize, 3, 3*i+1)
            pylab.imshow(tmp)
            pylab.axis("off")
            pylab.savefig("outdir/visualize_{}.png".format(index))
            tmp = (np.clip(s[i]*127.5+127.5, 0, 255)).transpose(1, 2, 0).astype(np.uint8)
            pylab.subplot(batchsize, 3, 3*i+2)
            pylab.imshow(tmp)
            pylab.axis("off")
            pylab.savefig("outdir/visualize_{}.png".format(index))
            tmp = np.clip(y[i]*127.5+127.5, 0, 255).transpose(1, 2, 0).astype(np.uint8)
            pylab.subplot(batchsize, 3, 3*i+3)
            pylab.imshow(tmp)
            pylab.axis("off")
            pylab.savefig("outdir/visualize_{}.png".format(index))


if __name__ == "__main__":
    output_dir = './outdir/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    parser = argparse.ArgumentParser(description='StyleAttentionNetwork')
    parser.add_argument('--e', default=1000, type=int, help="the number of epochs")
    parser.add_argument('--b', default=3, type=int, help="batch size")
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

    content_path = './coco2017/test2017/'
    style_path = './wikiart/'
    model_path = './model/model_53000.pt'

    test(content_path, style_path, model_path, batchsize)