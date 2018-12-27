import os
import numpy as np
import cv2 as cv

def kmeans(image,cluster=2):
    img = image.reshape((-1,3))
    img = np.float32(img)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv.kmeans(img,cluster,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    binary = res.reshape((image.shape))

    return binary

def prepare_dataset(filename,size,aug):
    image=cv.imread(filename)
    if image is not None:
        image=cv.resize(image,(size,size),interpolation=cv.INTER_CUBIC)
        image=image[:,:,::-1]
        if aug == 0:
            image = image[:,::-1,:]
        image=image.transpose(2,0,1)
        image=(image-127.5) / 127.5

        return image

def prepare_mask(filename,size,aug):
    image=cv.imread(filename)
    if image is not None:
        image=cv.resize(image,(size,size),interpolation=cv.INTER_CUBIC)
        image=image[:,:,::-1]
        mask=image[:,:,2].reshape(128,128,1)
        if aug == 0:
            mask = mask[:,::-1,:]
        mask=mask.transpose(2,0,1)
        mask=mask/255.0

        return mask