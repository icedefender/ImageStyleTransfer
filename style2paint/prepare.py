import cv2
import os
import numpy as np
import copy

def sharpImage(img, sigma, k_sigma, p):
    sigma_large = sigma * k_sigma
    G_small = cv2.GaussianBlur(img,(0, 0), sigma)
    G_large = cv2.GaussianBlur(img,(0, 0), sigma_large)
    S = (1+p) * G_small - p * G_large

    return S

def softThreshold(SI, epsilon, phi):
    T = np.zeros(SI.shape)
    SI_bright = SI >= epsilon
    SI_dark = SI < epsilon
    T[SI_bright] = 1.0
    T[SI_dark] = 1.0 + np.tanh( phi * (SI[SI_dark] - epsilon))

    return T

def xdog(img, sigma, k_sigma, p, epsilon, phi):
    S = sharpImage(img, sigma, k_sigma, p)
    SI = np.multiply(img, S)
    T = softThreshold(SI, epsilon, phi)

    return T

def making_mask(line_mask, color):
    choice = np.random.choice(['width', 'height', 'diag'])

    if choice == 'width':
        rnd_height = np.random.randint(4, 8)
        rnd_width = np.random.randint(4, 64)

        rnd1 = np.random.randint(224 - rnd_height)
        rnd2 = np.random.randint(224 - rnd_width)
        line_mask[rnd1:rnd1+rnd_height, rnd2:rnd2+rnd_width] = color[rnd1:rnd1+rnd_height, rnd2:rnd2+rnd_width]

    elif choice == 'height':
        rnd_height = np.random.randint(4, 64)
        rnd_width = np.random.randint(4, 8)

        rnd1 = np.random.randint(224 - rnd_height)
        rnd2 = np.random.randint(224 - rnd_width)
        line_mask[rnd1:rnd1+rnd_height, rnd2:rnd2+rnd_width] = color[rnd1:rnd1+rnd_height, rnd2:rnd2+rnd_width]

    elif choice == 'diag':
        rnd_height = np.random.randint(4, 8)
        rnd_width = np.random.randint(4, 64)

        rnd1 = np.random.randint(224 - rnd_height - rnd_width - 1)
        rnd2 = np.random.randint(224 - rnd_width)

        for index in range(rnd_width):
            line_mask[rnd1 + index : rnd1 + rnd_height + index, rnd2 + index] = color[rnd1 + index : rnd1 + rnd_height + index, rnd2 + index]

    return line_mask

def prepare_dataset(line_path, color_path, size=224):
    line = cv2.imread(line_path)
    color = cv2.imread(color_path)
    choice = np.random.choice(['horizon', 'normal'])
    if choice == 'horizon':
        color = color[:, ::-1, :]

    color = cv2.resize(color, (128, 128), interpolation=cv2.INTER_CUBIC)
    color_vgg = cv2.resize(color, (224, 224), interpolation=cv2.INTER_CUBIC)
    line = cv2.resize(line, (128,128), interpolation=cv2.INTER_CUBIC)

    color = color[:,:,::-1]
    color = color.transpose(2,0,1)
    color = (color-127.5)/127.5

    line = line[:,:,::-1]
    line = line.transpose(2,0,1)
    line = (line - 127.5)/127.5

    color_vgg = color_vgg[:,:,::-1]
    color_vgg = color_vgg.transpose(2,0,1)
    color_vgg = (color_vgg - 127.5) / 127.5
    
    return color, line, color_vgg