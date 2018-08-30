import os
import numpy as np  
import cv2 as cv
from PIL import Image

def prepare_dataset(filename, size = 512):
    if filename.endswith(".png"):
        image_orig = cv.imread(filename)
        image_orig = cv.resize(image_orig,(size,size))
        image = image_orig[:,:,::-1]
        image = image.transpose(2,0,1)
        image = (image-127.5)/127.5
        
        return image

def prepare_trim(filename, size = 256):
    image_path = filename
    image = cv.imread(image_path)
    if not image is None:
        height, width = image.shape[0], image.shape[1]

        rnd1 = np.random.randint(255)
        rnd2 = np.random.randint(255)

        leftup = rnd1
        leftdown = rnd1 + 256
        rightup = rnd2 + 256
        rightdown = rnd2
        
        image = image[leftup : leftdown , rightdown : rightup]

        image = image[:,:,::-1]
        image = image.transpose(2,0,1)
        image = (image-127.5)/127.5

        return image

def prepare_trim_full(filename):
    image_path = filename
    image = cv.imread(image_path)
    if not image is None:
        height, width = image.shape[0], image.shape[1]

        if height > width:
            scale = 512 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv.resize(image, (new_width, new_height))
        
        if height <= width:
            scale = 512 / height
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv.resize(image, (new_width, new_height))

        leftup = (int(new_height/2)) - 256
        leftdown = (int(new_height/2))+ 256
        rightup = (int(new_width/2)) + 256
        rightdown = (int(new_width/2)) - 256
        
        image = image[leftup : leftdown , rightdown : rightup]

        height, width = image.shape[0], image.shape[1]

        #leftup = (int(height/2)) - 128
        #leftdown = (int(height/2)) + 128
        #rightup = (int(width/2)) + 128
        #rightdown = (int(width/2)) - 128

        #image = image[leftup : leftdown , rightdown : rightup]

        image = image[:,:,::-1]
        image = image.transpose(2,0,1)
        image = (image-127.5)/127.5

        return image

def prepare_trim_test(filename):
    image_path = filename
    image = cv.imread(image_path)
    if not image is None:
        height, width = image.shape[0], image.shape[1]

        if height > width:
            scale = 512 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv.resize(image, (new_width, new_height))
        
        if height <= width:
            scale = 512 / height
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv.resize(image, (new_width, new_height))

        rnd1 = np.random.randint(100, 512)
        rnd2 = np.random.randint(100,512)

        leftup = (int(new_height/2)-80) - 128
        leftdown = (int(new_height/2)-80) + 128
        rightup = (int(new_width/2)) + 128
        rightdown = (int(new_width/2)) - 128
        
        image = image[leftup : leftdown , rightdown : rightup]

        height, width = image.shape[0], image.shape[1]

        #leftup = (int(height/2)) - 128
        #leftdown = (int(height/2)) + 128
        #rightup = (int(width/2)) + 128
        #rightdown = (int(width/2)) - 128

        #image = image[leftup : leftdown , rightdown : rightup]

        image = image[:,:,::-1]
        image = image.transpose(2,0,1)
        image = (image-127.5)/127.5

        return image