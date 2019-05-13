import cv2
import os
import numpy as np

def prepare_trim_content(filename):
    image_path = filename
    image = cv2.imread(image_path)
    if not image is None:
        height, width = image.shape[0], image.shape[1]

        if height > width:
            scale = 386 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        if height <= width:
            scale = 386 / height
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))

        rnd1 = np.random.randint(new_height - 384)
        rnd2 = np.random.randint(new_width - 384)

        leftup = rnd1
        leftdown = rnd1 + 224
        rightup = rnd2 + 224
        rightdown = rnd2
        
        image = image[leftup : leftdown , rightdown : rightup]

        height, width = image.shape[0], image.shape[1]

        image = image[:,:,::-1]
        image = image.transpose(2,0,1)
        image = (image-127.5)/127.5

        return image, rnd1, rnd2

def prepare_trim_style(filename, rnd1, rnd2):
    image_path = filename
    image = cv2.imread(image_path)
    if not image is None:
        height, width = image.shape[0], image.shape[1]

        if height > width:
            scale = 386 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        if height <= width:
            scale = 386 / height
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))

        leftup = rnd1
        leftdown = rnd1 + 224
        rightup = rnd2 + 224
        rightdown = rnd2
        
        image = image[leftup : leftdown , rightdown : rightup]

        image = image[:,:,::-1]
        image = image.transpose(2,0,1)
        image = (image - 127.5)/127.5

        return image

def edge_smoothed(filename):
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss * gauss.transpose(1, 0)

    rgb_img = cv2.imread(filename)
    gray_img = cv2.imread(filename,0)
    #rgb_img = cv2.resize(rgb_img, (256, 256))
    pad_img = np.pad(rgb_img, ((2,2), (2,2), (0,0)), mode='reflect')
    #gray_img = cv2.resize(gray_img, (256, 256))
    edges = cv2.Canny(gray_img, 100, 200)
    dilation = cv2.dilate(edges, kernel)

    gauss_img = np.copy(rgb_img)
    idx = np.where(dilation != 0)
    for i in range(np.sum(dilation != 0)):
        gauss_img[idx[0][i], idx[1][i], 0] = np.sum(np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0], gauss))
        gauss_img[idx[0][i], idx[1][i], 1] = np.sum(np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1], gauss))
        gauss_img[idx[0][i], idx[1][i], 2] = np.sum(np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2], gauss))

    image = gauss_img
    height, width = image.shape[0], image.shape[1]

    if height > width:
        scale = 257 / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
    
    if height <= width:
        scale = 257 / height
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))

    rnd1 = np.random.randint(new_height - 128)
    rnd2 = np.random.randint(new_width - 128)

    leftup = rnd1
    leftdown = rnd1 + 128
    rightup = rnd2 + 128
    rightdown = rnd2
    
    image = image[leftup : leftdown , rightdown : rightup]

    image = image[:,:,::-1]
    image = image.transpose(2,0,1)
    image = (image-127.5)/127.5

    return image