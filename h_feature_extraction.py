import os
import numpy
import argparse
from matplotlib import pyplot as plt
import glob
import csv
from skimage.io import sift, imread, imsave
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score
import pickle
from tqdm import tqdm
from auxiliary_functions.feature_list_division import *
from auxiliary_functions.pickle_functions import *
from joblib import Parallel, delayed
from skimage.util.shape import view_as_windows
from openslide import OpenSlide, deepzoom
from scipy.special import softmax
from auxiliary_functions.get_patch_reshaped import *


def get_probability_images(slidename, classifier, slide_folder, features=30):
    result = []
    slidepath = os.path.join(slide_folder, slidename)
    slide = OpenSlide(slidepath)
    slide_dz = deepzoom.DeepZoomGenerator(slide, tile_size=(224 - 2), overlap=1)
    tiles = slide_dz.level_tiles[16]
    preview = numpy.zeros((tiles[0], tiles[1], features + 1))
    for x in classifier:
        im_x = int(x[1])
        im_y = int(x[2])
        if x[3] == 0:
            preview[im_x][im_y][0] = 1
        else:
            for f in range(features):
                preview[im_x][im_y][f+1] = x[4+f]
    image_name = os.path.join(outpath, '{}#{}features.npy'.format(slidename, features))
    numpy.save(image_name, preview)
    return image_name


def features_ex(im, patch_shape):
    features = []
    image = numpy.load(im)
    image = image.astype(float)
    patches = view_as_windows(image, patch_shape)
    patches = numpy.ascontiguousarray(patches)
    print(im)
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patch_name = os.path.basename(im)
            patch_name = patch_name.split('#')[0]
            patch_name = os.path.join(os.path.dirname(outpath), patch_name)
            patch_name = glob.glob(os.path.join(patch_name, '*-{}-{}.jpg'.format(i+1, j+1)))
            if not patch_name == []:
                patch = patches[i][j].reshape(patch_shape[0] * patch_shape[1] * patch_shape[2])
                features.append((patch_name[0], patch))
                print(0)
    return features


def h_feature_extraction(distances, outpath, slide_folder):

    # function to perform the softmax to the probability maps
    for d in tqdm(distances):
        npatches = len(d[2])
        for i in range(0, npatches):
            if not d[2][i, 3] == 0:
                d[2][i, 4:] = softmax(-d[2][i, 4:])

    n_features = distances[0][2].shape[1] - 4
    image_list = Parallel(n_jobs=-3, backend='loky')(delayed(get_probability_images)(d[0], d[2], slide_folder, features=n_features) for d in tqdm(distances))

    patch_shape = (3, 3, n_features + 1)
    features = Parallel(n_jobs=-1)(delayed(features_ex)(i, patch_shape) for i in tqdm(image_list))

    total_features = []
    for f in features:
        total_features.extend(f)

    pickle_save(total_features, outpath, 'hierarchical_features.p')

    return features
