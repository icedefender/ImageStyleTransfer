import cv2
import os
import numpy as np

kernel_size = 5
kernel = np.ones((kernel_size, kernel_size), np.uint8)
gauss = cv2.getGaussianKernel(kernel_size, 0)
gauss = gauss * gauss.transpose(1, 0)

rgb_img = cv2.imread("image.jpg")
gray_img = cv2.imread("image.jpg")
rgb_img = cv2.resize(rgb_img, (256, 256))
pad_img = np.pad(rgb_img, ((2,2), (2,2), (0,0)), mode='reflect')
gray_img = cv2.resize(gray_img, (256, 256))
edges = cv2.Canny(gray_img, 100, 200)
dilation = cv2.dilate(edges, kernel)

gauss_img = np.copy(rgb_img)
idx = np.where(dilation != 0)
for i in range(np.sum(dilation != 0)):
    gauss_img[idx[0][i], idx[1][i], 0] = np.sum(np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0], gauss))
    gauss_img[idx[0][i], idx[1][i], 1] = np.sum(np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1], gauss))
    gauss_img[idx[0][i], idx[1][i], 2] = np.sum(np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2], gauss))

result = np.concatenate((rgb_img, gauss_img), 1)

cv2.imwrite("test.jpg", result)
