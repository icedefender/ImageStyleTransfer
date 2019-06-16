import os
import torch
import numpy as np
import cv2 as cv
from torch.utils.data import Dataset, DataLoader


class HairDataset(Dataset):
    def __init__(self, medium_path, twin_path):
        super(HairDataset, self).__init__()

        self.mpath = medium_path
        self.mlist = os.listdir(self.mpath)
        self.mlen = len(self.mlist)
        self.tpath = twin_path
        self.tlist = os.listdir(self.tpath)
        self.tlen = len(self.tlist)

    def __len__(self):
        return self.mlen - 50

    def __getitem__(self, idx):
        m_path = self.mpath + self.mlist[idx]

        rnd = np.random.randint(self.tlen)
        t_path = self.tpath + self.tlist[rnd]

        return (m_path, t_path)


class CollateFn():
    def __init__(self):
        pass

    def _prepare(self, path,size=128):
        img = cv.imread(path)
        img = cv.resize(img, (size, size))
        img = img[:, :, ::-1]
        img = (img.transpose(2, 0, 1) - 127.5) / 127.5

        return img

    def train(self, batch):
        x_box = []
        y_box = []

        for b in batch:
            x_path, y_path = b
            x = self._prepare(x_path)
            y = self._prepare(y_path)
            x_box.append(x)
            y_box.append(y)

        x = torch.FloatTensor(np.array(x_box).astype(np.float32))
        y = torch.FloatTensor(np.array(y_box).astype(np.float32))

        x = x.cuda()
        y = y.cuda()

        return (x, y)