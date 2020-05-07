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
import pickle
import argparse


def dab(image, thr=225, freq=3):
    """
    Input: image in rgb
    """
    dab = False
    image = rgb2hed(image)
    hist = get_hist(image)
    if numpy.sum(hist[thr:]) > freq:
        dab = True
    return dab


def get_hist(image, mode='hed', channel=2, min_val=-0.55, max_val=-0.2):

    if mode == 'rgb':
        image = rgb2hed(image)
    im_channel = image[:, :, channel]
    x = (min_val, max_val)
    y = (0, 255)
    a = (y[0]-y[1])/(x[0]-x[1])
    b = y[1] - a*x[1]
    for i in range(im_channel.shape[0]):
        for j in range(im_channel.shape[1]):
            im_channel[i, j] = abs(a*(im_channel[i, j]) + b)
            im_channel[i, j] = abs(int(im_channel[i, j]))

    im_channel = im_channel.astype('int64')
    im_channel = numpy.reshape(im_channel, im_channel.shape[0]*im_channel.shape[1])

    hist = numpy.bincount(im_channel, minlength=255)

    return hist


def divide_dab(path):
    """
    Divides set of images according to the presence of DAB staining

    Arguments:
        - path: str, path to image folder

    Returns:
        - image_positive, image_negative: list
    """

    image_path = os.path.join(path, "*.jpg")
    image_path = glob.glob(image_path)
    image_positive = []
    image_negative = []
    n = len(image_path)

    for im in tqdm(image_path):
        image = imread(im)
        if dab(image):
            image_positive.append(im)
        else:
            image_negative.append(im)

    print('Division in path ' + path + ' completed.')
    print('Number of positive elements: {} out of {}'.format(len(image_positive), n))
    print()

    return image_positive, image_negative, n


def pickle_save(file, path, name):
    file_path = os.path.join(path, name)
    with open(file_path, "wb") as f:
        pickle.dump(file, f)


def detect_dab(list_slides, outpath):

    classifier = []
    list_positive = []
    for i in range(len(list_slides)):
        print('Getting positive patches from slide {} out of {}'.format(i+1, len(list_slides)))
        list_positive_x, list_negative_x, n = divide_dab(list_slides[i][1])
        c = numpy.zeros((n, 2))
        c = c.astype(int)
        for im in list_positive_x:
            name = os.path.basename(im)
            number = name.split('-')
            number = int(number[1])
            c[number][0] = number
            c[number][1] = 1
            list_positive.append(im)
        for im in list_negative_x:
            name = os.path.basename(im)
            number = name.split('-')
            number = int(number[1])
            c[number][0] = number
            c[number][1] = 0
        classifier.append((list_slides[i][0], list_slides[i][1], c))

    name = outpath
    name = os.path.basename(name)
    name = os.path.splitext(name)[0]
    name = name.split('_')
    level = name[1]
    tile_size = name[3]
    pickle_save(classifier, outpath, 'class_{}_{}.p'.format(level, tile_size))
    pickle_save(list_positive, outpath, 'list_positive_{}_{}.p'.format(level, tile_size))

    return classifier, list_positive


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script that discriminates patches positives to DAB.')
    parser.add_argument('-l', '--list_slides', type=str, help='file with slide list')
    parser.add_argument('-o', '--outpath', type=str, help='path to outfolder')

    args = parser.parse_args()

    outpath = args.outpath
    list_slides = args.list_slides
    with open(list_slides, "rb") as f:
        list_slides = pickle.load(f)

    classifier, list_positive = detect_dab(list_slides, outpath)
