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

    Returns:
        - image_positive, image_negative: list
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


def pickle_save(file, path, name):
    file_path = os.path.join(path, name)
    with open(file_path, "wb") as f:
        pickle.dump(file, f)


def detect_dab_delayed(slide, threshold):
    list_positive, list_negative, n = divide_dab(slide[1], threshold=threshold)
    c = numpy.zeros((n, 4))
    c = c.astype(int)
    for im in list_positive:
        name = os.path.basename(im)
        name = os.path.splitext(name)[0]
        number = name.split('-')
        slide_number = int(number[1])
        x = int(number[3])
        y = int(number[4])
        c[slide_number][0] = slide_number
        c[slide_number][1] = x
        c[slide_number][2] = y
        c[slide_number][3] = 1
    for im in list_negative:
        name = os.path.basename(im)
        name = os.path.splitext(name)[0]
        number = name.split('-')
        slide_number = int(number[1])
        x = int(number[3])
        y = int(number[4])
        c[slide_number][0] = slide_number
        c[slide_number][1] = x
        c[slide_number][2] = y
        c[slide_number][3] = 0
    classifier = (slide[0], slide[1], c)

    return classifier, list_positive


def detect_dab(list_slides, outpath, jobs, threshold):

    # Parallelization
    start = time.time()
    result = Parallel(n_jobs=jobs)(delayed(detect_dab_delayed)(s, threshold) for s in (list_slides))
    end = time.time()

    classifier = [result[i][0] for i in range(len(list_slides))]
    list_positive = [result[i][1] for i in range(len(list_slides))]
    # classifier = []
    # list_positive = []
    # for i in range(len(list_slides)):
    #     classifier.append(result[i][0])
    #     for p in result[i][1]:
    #         list_positive.append(p)

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
