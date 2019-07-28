import cv2 as cv
import numpy as np
import copy
import chainer

from chainer import cuda
from pathlib import Path

xp = cuda.cupy
cuda.get_device(0).use()


class DatasetLoader:
    def __init__(self, dataset_path, test_path):
        self.path = dataset_path
        self.test_path = test_path
        self.testlist = list(self.test_path.glob('*.png'))
        self.cls_list = ['black', 'white', 'blue', 'pink', 'gold', 'red']

    @staticmethod
    def _coordinate(path, size=128):
        img = cv.imread(str(path))
        img = cv.resize(img, (size, size), interpolation=cv.INTER_CUBIC)
        if np.random.randint(2):
            img = img[:, ::-1, :]
        img = img[:, :, ::-1]
        img = (img.transpose(2, 0, 1) - 127.5) / 127.5

        return img

    @staticmethod
    def _making_index(color):
        if color == 'black':
            return 0
        elif color == 'white':
            return 1
        elif color == 'blue':
            return 2
        elif color == 'pink':
            return 3
        elif color == 'red':
            return 4
        else:
            return 5

    @staticmethod
    def _making_onehot(color, cls_list):
        cls_list_copy = copy.copy(cls_list)
        cls_list_copy.remove(color)

        return cls_list_copy

    @staticmethod
    def _variable(array_list, np_type='float'):
        if np_type == 'float':
            return chainer.as_variable(xp.array(array_list).astype(xp.float32))

        elif np_type == 'int':
            return chainer.as_variable(xp.array(array_list).astype(xp.int32))

        else:
            raise AttributeError

    def _prepare(self, cls_list):
        color = np.random.choice(cls_list)
        self.pathlist = list(self.path.glob(f"{color}/*.png"))
        path = np.random.choice(self.pathlist)

        index = self._making_index(color)
        img = self._coordinate(path)

        return img, index, color

    def train(self, batchsize):
        c_img_box = []
        c_index_box = []
        s_img_box = []
        s_index_box = []

        for _ in range(batchsize):
            cls_list = copy.copy(self.cls_list)
            c_img, c_index, c_color = self._prepare(cls_list)

            stl_list = self._making_onehot(c_color, cls_list)
            s_img, s_index, _ = self._prepare(stl_list)

            c_img_box.append(c_img)
            c_index_box.append(c_index)
            s_img_box.append(s_img)
            s_index_box.append(s_index)

        c_img = self._variable(c_img_box, np_type='float')
        s_img = self._variable(s_img_box, np_type='float')
        #c_index = self._variable(c_index_box, np_type='int')
        #s_index = self._variable(s_index_box, np_type='int')

        return (c_img, c_index, s_img, s_index)

    def test(self, testsize):
        c_img_box = []
        s_img_box = []

        for index in range(testsize):
            cls_list = copy.copy(self.cls_list)
            c_img, _, _ = self._prepare(cls_list)

            test_path = self.testlist[index]
            s_img = self._coordinate(test_path)

            c_img_box.append(c_img)
            s_img_box.append(s_img)

        c_img = self._variable(c_img_box)
        s_img = self._variable(s_img_box)

        return (c_img, s_img)