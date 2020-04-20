import os
from openslide import OpenSlide, deepzoom
import numpy
from matplotlib import pyplot as plt
import sys
import cv2
import glob
from skimage.io import sift, imread, imsave, imshow
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from skimage.util.shape import view_as_windows
from skimage.color import rgb2hed

def detect_dab(image, thr = 225, freq = 3):
    """
    Input: image in rgb
    """
    dab = False
    image = rgb2hed(image)
    hist = get_hist(image)
    if numpy.sum(hist[thr:]) > freq:
        dab = True
    return dab

def get_hist(image, mode = 'hed', channel = 2, min_val = -0.55, max_val = -0.2):

    if mode == 'rgb':
        image = rgb2hed(image)
    im_channel = image[:, :, channel]
    x = (min_val, max_val)
    y = (0, 255)
    a = (y[0]-y[1])/(x[0]-x[1])
    b = y[1] - a*x[1]
    for i in range(im_channel.shape[0]):
        for j in range(im_channel.shape[1]):
            im_channel[i, j] = a*(im_channel[i, j]) + b
            im_channel[i, j] = int(im_channel[i,j])

    im_channel = im_channel.astype('int64')
    im_channel = numpy.reshape(im_channel, im_channel.shape[0]*im_channel.shape[1])

    hist = numpy.bincount(im_channel, minlength=255)

    return hist
