import torch
import numpy as np
import os
import cv2 as cv

from torch.utils.data import Dataset


class CSDataset(Dataset):
    def __init__(self, c_path, s_path):
        self.cpath = c_path
        self.spath = s_path
        self.content_list = os.listdir(c_path)
        self.ncontent = len(self.content_list)
        self.style_list = os.listdir(s_path)
        self.nstyle = len(self.style_list)

    def __len__(self):
        return self.ncontent - 100

    def __getitem__(self, index):
        c_name = self.cpath + self.content_list[index]

        rnd = np.random.randint(self.nstyle)
        s_name = self.spath + self.style_list[rnd]

        return (c_name, s_name)


class CSTestDataset(Dataset):
    def __init__(self, c_path, s_path):
        self.cpath = c_path
        self.spath = s_path
        self.content_list = os.listdir(c_path)
        self.ncontent = len(self.content_list)
        self.style_list = os.listdir(s_path)
        self.nstyle = len(self.style_list)

    def __len__(self):
        return 99

    def __getitem__(self, index):
        c_name = self.cpath + self.content_list[index]

        rnd = np.random.randint(self.nstyle)
        s_name = self.spath + self.style_list[rnd]

        return (c_name, s_name)


class ImageCollate():
    def __init__(self, test=False):
        self.test = test

    def _preapre(self, filename):
        image_path = filename
        image = cv.imread(image_path)
        if not image is None:
            height, width = image.shape[0], image.shape[1]

            if height > width:
                scale = 378 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv.resize(image, (new_width, new_height))
            
            if height <= width:
                scale = 378 / height
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv.resize(image, (new_width, new_height))

            rnd1 = np.random.randint(100)
            rnd2 = np.random.randint(100)
            image = image[rnd1: rnd1 + 256, rnd2: rnd2 + 256]

            height, width = image.shape[0], image.shape[1]

            image = image[:, :, ::-1]
            image = image.transpose(2, 0, 1)
            image = (image - 127.5)/127.5

            return image

    def _test_preapre(self, filename):
        image_path = filename
        image = cv.imread(image_path)
        if not image is None:
            height, width = image.shape[0], image.shape[1]

            if height > width:
                scale = 513 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv.resize(image, (new_width, new_height))
            
            if height <= width:
                scale = 513 / height
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv.resize(image, (new_width, new_height))

            #rnd1 = np.random.randint(100)
            #rnd2 = np.random.randint(100)
            image = image[:512, :512]

            height, width = image.shape[0], image.shape[1]

            image = image[:, :, ::-1]
            image = image.transpose(2, 0, 1)
            image = (image - 127.5)/127.5

            return image

    def __call__(self, batch):
        c_box = []
        s_box = []
        for b in batch:
            c_name, s_name = b
            if self.test:
                s = self._test_preapre(s_name)
                c = self._test_preapre(c_name)
            
            else:
                s = self._preapre(s_name)
                c = self._preapre(c_name)

            c_box.append(c)
            s_box.append(s)

        c = np.array(c_box).astype(np.float32)
        s = np.array(s_box).astype(np.float32)

        c = torch.FloatTensor(c)
        s = torch.FloatTensor(s)

        c = c.cuda()
        s = s.cuda()

        return (c, s)
