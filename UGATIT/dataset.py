import numpy as np
import cv2 as cv
import torch

from torch.utils.data import Dataset
from pathlib import Path


class UGATITDataset(Dataset):
    def __init__(self, c_path: Path, s_path: Path):
        self.cpath = c_path
        self.spath = s_path

        self.clist = list(c_path.glob('*.jpg'))
        self.slist = list(s_path.glob('*.png'))

    def __str__(self):
        return f"source: {len(self.clist)} target: {len(self.slist)}"

    def __len__(self):
        return len(self.clist)

    def __getitem__(self, index):
        c_name = self.clist[index]

        rnd = np.random.randint(len(self.slist))
        s_name = self.slist[rnd]

        return c_name, s_name


class UGATITDatasetTest(Dataset):
    def __init__(self, c_path: Path):
        self.cpath = c_path

        self.clist = list(c_path.glob('*.jpg'))

    def __str__(self):
        return f"source: {len(self.clist)}"

    def __len__(self):
        return len(self.clist)

    def __getitem__(self, index):
        c_name = self.clist[index]

        return c_name


class ImageCollate:
    def __init__(self, size=128):
        self.size = size

    def _source_prepare(self, filename):
        img = cv.imread(str(filename))
        height, width = img.shape[0], img.shape[1]
        mid = int(height/2)
        center_dist = int(width/2)

        cropped = img[mid-center_dist: mid+center_dist, :]
        cropped = cv.resize(cropped, (self.size, self.size))

        #rnd = np.random.randint(10000)
        #cv.imwrite(f"./test/aaa_{rnd}.png", cropped)

        cropped = cropped[:, :, ::-1]
        cropped = cropped.transpose(2, 0, 1)
        cropped = (cropped - 127.5) / 127.5

        return cropped

    def _target_prepare(self, filename):
        img = cv.imread(str(filename))
        img = cv.resize(img, (self.size, self.size))
        img = img[:, :, ::-1]
        img = (img.transpose(2, 0, 1) - 127.5) / 127.5

        return img

    def __call__(self, batch):
        s_box = []
        t_box = []

        for b in batch:
            s_name, t_name = b
            s = self._source_prepare(s_name)
            t = self._target_prepare(t_name)

            s_box.append(s)
            t_box.append(t)

        s = np.array(s_box).astype(np.float32)
        t = np.array(t_box).astype(np.float32)

        s = torch.FloatTensor(s)
        t = torch.FloatTensor(t)

        s = s.cuda()
        t = t.cuda()

        return s, t


class ImageCollateTest:
    def __init__(self, size=128):
        self.size = size

    def _source_prepare(self, filename):
        img = cv.imread(str(filename))
        height, width = img.shape[0], img.shape[1]
        mid = int(height/2)
        center_dist = int(width/2)

        cropped = img[mid-center_dist: mid+center_dist, :]
        cropped = cv.resize(cropped, (self.size, self.size))

        cropped = cropped[:, :, ::-1]
        cropped = cropped.transpose(2, 0, 1)
        cropped = (cropped - 127.5) / 127.5

        return cropped

    def __call__(self, batch):
        s_box = []

        for b in batch:
            s_name = b
            s = self._source_prepare(s_name)
            s_box.append(s)

        s = np.array(s_box).astype(np.float32)
        s = torch.FloatTensor(s)
        s = s.cuda()

        return s
