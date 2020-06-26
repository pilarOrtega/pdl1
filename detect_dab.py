import os
from openslide import OpenSlide, deepzoom
import numpy
from matplotlib import pyplot as plt
import sys
import glob
from skimage.io import sift, imread, imsave, imshow
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from skimage.util.shape import view_as_windows
from skimage.exposure import histogram
from skimage.color import rgb2hed
import pickle
import argparse
from joblib import Parallel, delayed
import time
from numba import jit
from auxiliary_functions.pickle_functions import *


def dab(image, thr=85, freq=10):
    """
    Detects if an image has DAB tinction or not

    Arguments:
        - image: PIL image, in rgb

    Returns:
        - dab: bool, True if image has DAB, False if not
    """
    dab = False
    image = rgb2hed(image)
    hist = histogram(image[:, :, 2], source_range='dtype')[0]
    if numpy.sum(hist[thr:]) > freq:
        dab = True
    return dab


def divide_dab(path, threshold):
    """
    Divides set of images according to the presence of DAB staining

    Arguments:
        - path: str, path to image folder
        - threshold: int, threshold level for DAB detection

    Returns:
        - image_positive, image_negative: list
        - n: int, total number of images
    """

    # Collects all images .jpg from path
    image_path = os.path.join(path, "*.jpg")
    image_path = glob.glob(image_path)
    image_positive = []
    image_negative = []
    n = len(image_path)

    # Detects DAB presence in each image. DAB positive images are stored in image_positive list
    for im in tqdm(image_path):
        image = imread(im)
        if dab(image, thr=threshold):
            image_positive.append(im)
        else:
            image_negative.append(im)

    print('Division in path ' + path + ' completed.')
    print('Number of positive elements: {} out of {}'.format(len(image_positive), n))
    print()

    return image_positive, image_negative, n


def detect_dab_delayed(slide, threshold):
    """
    Gets the DAB positive patches for a slide and creates the classifier asarray

    Arguments:
        - Slide: str, path to slide folder with the individual patches
        - Threshold: int, threshold level for DAB detection

    Returns:
        - classifier: list with three elements (1) slide name (2) path to slide
            folder (3) classifier array. The classifier array is an array with
            shape (n, 4) - with n being the total number of patches. Classifier
            array saves for each patch the patch number (from 0 to n-1), the x
            coordenate, the y coordenate and a 1 if DAB positive or 0 if DAB
            negative
        - list_positive: list with the paths to all patches that are positive
            to DAB
    """
    # Divide all patches in positive or negative to DAB
    list_positive, list_negative, n = divide_dab(slide, threshold=threshold)
    # Creates a numpy array with n rows and 4 columns (number, x, y, positive)
    c = numpy.zeros((n, 4))
    c = c.astype(int)
    for im in list_positive:
        # Extracts the patch number from the patch path
        name = os.path.basename(im)
        name = os.path.splitext(name)[0]
        number = name.split('#')[1]
        number = number.split('-')
        slide_number = int(number[0])
        x = int(number[2])
        y = int(number[3])
        # Loads data in classifier matrix
        c[slide_number][0] = slide_number
        c[slide_number][1] = x
        c[slide_number][2] = y
        # Positive column = 1 if patch is in list_positive
        c[slide_number][3] = 1
    for im in list_negative:
        name = os.path.basename(im)
        name = os.path.splitext(name)[0]
        number = name.split('#')[1]
        number = number.split('-')
        slide_number = int(number[0])
        x = int(number[2])
        y = int(number[3])
        c[slide_number][0] = slide_number
        c[slide_number][1] = x
        c[slide_number][2] = y
        # Positive column = 0 if patch is in list_negative
        c[slide_number][3] = 0
    classifier = (os.path.basename(slide), slide, c)

    return classifier, list_positive, list_negative


def detect_dab(list_slides, outpath, jobs, threshold, level=16, tile_size=224):
    """
    For a list of slides, this function gets the patches that have DAB tinction
    on them (DAB positive patches). It saves two .p files in the outpath
    directory with classifier and list positive.

    Arguments:
        - list_slides: list with slides to evaluate. Each list element has two
            values (1) Slidename (2) path to folder where the patches from the
            slide are stored.
        - outpath: str, path to the folder where the results will be saved. It
            has the format /level_{}_ts_{}_{other information}
        - jobs: int, number of processors to be used during parallelization
        - threshold: int, threshold level for DAB detection

    Returns:
        - classifier: list with lentgh len(list_slides). For each slide, it
            contains three elements: (1) slide name (2) path to slide folder (3)
            classifier array. The classifier array is an array with shape (n, 4)
            - with n being the total number of patches. Classifier array saves
            for each patch the patch number (from 0 to n-1), the x coordenate,
            the y coordenate and a 1 if DAB positive or 0 if DAB negative
        - list positive: list with the path to all patches that resulted DAB
            positive for all slides in list_slides
    """
    # Parallelization
    start = time.time()
    result = Parallel(n_jobs=jobs)(delayed(detect_dab_delayed)(s[1], threshold) for s in (list_slides))
    end = time.time()
    print('Total time DAB detection: {:.4f} s'.format(end-start))

    classifier = [result[i][0] for i in range(len(list_slides))]
    list_positive = []
    for i in range(len(list_slides)):
        for p in result[i][1]:
            list_positive.append(p)

    pickle_save(classifier, outpath, 'class_{}_{}.p'.format(level, tile_size))
    pickle_save(list_positive, outpath, 'list_positive_{}_{}.p'.format(level, tile_size))

    return classifier, list_positive


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script that discriminates patches positives to DAB.')
    parser.add_argument('-l', '--list_slides', type=str, help='file with slide list')
    parser.add_argument('-o', '--outpath', type=str, help='path to outfolder')
    parser.add_argument('-t', '--threshold', type=int, default=85)
    parser.add_argument('-j', '--jobs', type=int)

    args = parser.parse_args()

    outpath = args.outpath
    list_slides = args.list_slides
    jobs = args.jobs
    threshold = args.threshold
    with open(list_slides, "rb") as f:
        list_slides = pickle.load(f)

    classifier, list_positive = detect_dab(list_slides, outpath, jobs=jobs, threshold=threshold)
