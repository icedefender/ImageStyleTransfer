import numpy as np
import chainer 
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Chain, initializers

xp = cuda.cupy

def content_transform(feature):
    feature = feature.data
    co_matrix = xp.cov(xp.dot(feature, feature.T))
    e,d,et = xp.linalg.svd(co_matrix, full_matrices = True)
    d = xp.diag(d)
    d = xp.power(d, -0.5)
    trans = xp.dot(xp.dot(xp.dot(et,d), et.T), feature)

    return trans

def style_transform(feature, style_feature):
    style_feature = style_feature.data
    style_mean = xp.mean(style_feature)
    co_matrix = xp.cov(xp.dot(style_feature, style_feature.T))
    e,d,et = xp.linalg.svd(co_matrix, full_matrices = True)
    d = xp.diag(d)
    d = xp.power(d, 0.5)
    trans = xp.dot(xp.dot(xp.dot(et,d), et.T), feature)
    trans = trans + style_mean

    return truns

def WCT(content_feature, style_feature, alpha = 1.0):
    fc = content_transform(content_feature)
    fcs = style_transform(fc, style_transform)

    return alpha * fcs + (1 - alpha) * content_feature