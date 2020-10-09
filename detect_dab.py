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


def divide_dab(classifier, threshold):
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
    # Collects all images .jpg from path
    image_path = os.path.join(classifier[0], "*.jpg")
    image_path = glob.glob(image_path)
    n = len(image_path)

    # Detects DAB presence in each image. DAB positive images are stored in image_positive list
    for im in tqdm(image_path):
        image = imread(im)
        name = os.path.basename(im)
        name = os.path.splitext(name)[0]
        number = name.split('#')[1]
        number = number.split('-')
        slide_number = int(number[0])
        if not dab(image, thr=threshold):
            classifier[1][slide_number][3] = 0

    return classifier


def detect_dab(classifier, outpath, jobs, threshold, level=16, tile_size=224):
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
    classifier = Parallel(n_jobs=jobs)(
        delayed(divide_dab)(c, threshold) for c in (classifier))
    end = time.time()
    print('Total time DAB detection: {:.4f} s'.format(end - start))

    pickle_save(classifier, outpath, 'class_{}_{}.p'.format(level, tile_size))

    return classifier


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Script that discriminates patches positives to DAB.')
    parser.add_argument('-c', '--classifier', type=str, help='Classifier file')
    parser.add_argument('-o', '--outpath', type=str, help='path to outfolder')
    parser.add_argument('-t', '--threshold', type=int, default=85)
    parser.add_argument('-j', '--jobs', type=int)

    args = parser.parse_args()

    outpath = args.outpath
    classifier = args.classifier
    jobs = args.jobs
    threshold = args.threshold
    with open(classifier, "rb") as f:
        classifier = pickle.load(f)

    classifier = detect_dab(classifier, outpath,
                            jobs=jobs, threshold=threshold)
