import chainer
import numpy as np
import copy
import cv2 as cv

from pathlib import Path
from chainer import cuda

xp = cuda.cupy
cuda.get_device(0).use()


class DatasetLoader:
    def __init__(self, data_path, nc_size):
        self.data_path = data_path
        self.nc_size = nc_size
        self.label_list = [0, 1, 2, 3, 4, 5]

    @staticmethod
    def _label_remove(label_list, source):
        label_list.remove(source)

        return label_list

    @staticmethod
    def _variable(array_list, array_type='float'):
        if array_type == 'float':
            return chainer.as_variable(xp.array(array_list).astype(xp.float32))
        
        else:
            return chainer.as_variable(xp.array(array_list).astype(xp.int32))

    def _onehot_convert(self, label):
        onehot = np.zeros(self.nc_size)
        onehot[label] = 1

        return onehot

    def _prepare_sp(self, path):
        img = cv.imread(str(path))
        img = cv.resize(img, (128, 128))
        img = img[:, :, ::-1]
        img = (img.transpose(2, 0, 1) - 127.5) / 127.5

        return img

    def train(self, batchsize):
        x_sp_box = []
        x_label_box = []
        y_sp_box = []
        y_label_box = []
        z_sp_box = []
        z_label_box = []
        for _ in range(batchsize):
            label_list = copy.copy(self.label_list)
            rnd = np.random.choice(label_list)
            onehot = self._onehot_convert(rnd)
            pathlist = list((self.data_path/Path(str(rnd))).glob('*.png'))
            sp_path = np.random.choice(pathlist)
            sp = self._prepare_sp(sp_path)

            x_sp_box.append(sp)
            x_label_box.append(onehot)

            label_list = self._label_remove(label_list, rnd)
            rnd = np.random.choice(label_list)
            onehot = self._onehot_convert(rnd)
            pathlist = list((self.data_path/Path(str(rnd))).glob('*.png'))
            sp_path = np.random.choice(pathlist)
            sp = self._prepare_sp(sp_path)

            y_sp_box.append(sp)
            y_label_box.append(onehot)

            label_list = self._label_remove(label_list, rnd)
            rnd = np.random.choice(label_list)
            onehot = self._onehot_convert(rnd)
            pathlist = list((self.data_path/Path(str(rnd))).glob('*.png'))
            sp_path = np.random.choice(pathlist)
            sp = self._prepare_sp(sp_path)

            z_sp_box.append(sp)
            z_label_box.append(onehot)

        x_sp = self._variable(x_sp_box)
        x_label = self._variable(x_label_box, array_type='float')
        y_sp = self._variable(y_sp_box)
        y_label = self._variable(y_label_box, array_type='float')
        z_sp = self._variable(z_sp_box)
        z_label = self._variable(z_label_box, array_type='float')

        return (x_sp, x_label, y_sp, y_label, z_sp, z_label)

    def test(self, testsize):
        testlist = list(self.data_path.glob("*.png"))
        test_box = []

        for path in testlist:
            img = self._prepare_sp(str(path))
            test_box.append(img)

        x_test = self._variable(test_box)

        return x_test
