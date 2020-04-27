__author__ = 'Pilar Ortega Arevalo'
__copyright__ = 'Copyright (C) 2019 IUCT-O'
__license__ = 'GNU General Public License'
__version__ = '1.2.1'
__status__ = 'prod'

import os
from openslide import OpenSlide, deepzoom
import numpy
import argparse
from matplotlib import pyplot as plt
import Pysiderois_Arnaud.pysliderois.tissue as tissue
import sys
import cv2
from PythonSIFT import pysift
import glob
from skimage.io import sift, imread, imsave
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from skimage.util.shape import view_as_windows
from skimage.color import rgb2grey
from skimage.feature import daisy
import detect_dab
import pickle
import csv
import class_to_cluster

# Importing Keras libraries
from keras.utils import np_utils
from keras.applications import VGG16
from keras.applications import imagenet_utils

###############################################################################
# THINGS TO IMPROVE #
# Save path, not images V
# Display preview clusters
# Improve feature extraction Dense and Daisy
# Try different clustering methods (Kmeans, DBSCAN)
# Try different feature extraction method
# Better screen display
# Results
###############################################################################

###############################################################################
#
# FUNCTIONS
#
###############################################################################


def get_patches(slidepath, outpath, level=10, tissue_ratio=0.25, size=256):
    """
    Function that divides a slide into patches with different resolution

    Arguments:
        - slidepath: str, path to the image to patchify
        - outpath: str, path to the folder in which the patches will be saved
        - level: int, level in which image is patchified. The bigger the level,
          the higher the number of patches and the resolution of the images.
        - tissue_ratio: float, minimum surface of tissue tile to be considered
        - size: int, side number of pixels (n pixels size*size)

    Returns:
        - n: int, number of patches
        - outpath: str, folder in which the patches are stored

    """

    # Opens the slide with OpenSlide
    slide = OpenSlide(slidepath)
    slide_dz = deepzoom.DeepZoomGenerator(slide, tile_size=(size - 2), overlap=1)
    PATH = os.path.join(outpath, "level_{}".format(level))

    # Makes directory to store the patches
    try:
        os.mkdir(PATH)
        print("Directory", PATH, "created")
    except FileExistsError:
        print("Directory", PATH, "already exists")

    # Asures that the chosen level is valid
    if level < slide_dz.level_count:
        tiles = slide_dz.level_tiles[level]
        print('Level {} : {} tiles (empty tiles included)'.format(level, slide_dz.level_tiles[level][0]*slide_dz.level_tiles[level][1]))
        print()
    else:
        print('Invalid level')
        return

    # Saves tiles if detects tissue presence higher than tissue_ratio
    n = 0
    print("Saving tiles image " + slidepath + "...")
    for i in tqdm(range(tiles[0])):
        for j in range(tiles[1]):
            tile = slide_dz.get_tile(level, (i, j))
            tile_path = os.path.join(PATH, '{}_slide1_level{}_{}_{}.jpg'.format(n, level, i, j))
            image = numpy.array(tile)[..., :3]
            mask = tissue.get_tissue_from_rgb(image)
            if mask.sum() > tissue_ratio * tile.size[0] * tile.size[1]:
                tile.save(tile_path)
                n = n + 1
    print('Total of {} tiles in level {}'.format(n, level))
    print()

    return n, PATH


def get_features_SIFT(path, total):
    n = 0
    des_list = []
    image_path = os.path.join(path, '*.jpg')
    print('Obtaining SIFT features from images in ' + path + '...')
    for im in tqdm(glob.glob(image_path)):
        image = cv2.imread(im, 0)
        n = n + 1
        keypoints, descriptors = pysift.computeKeypointsAndDescriptors(image)
        des_list.append((im, descriptors))
    # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]
    for im, descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))  # Stacking the descriptors
    return descriptors


def get_features(image_list, nclusters=256, method='Dense'):

    """
    Gets the histogram of features of the given set of images. It obtains the
    features by means of a KMeans clustering algorithm.

    Arguments:
        - image_list: list, image set
        - nclusters: int, number of visual words in which the features are clustered
        - method: str, Dense or Daisy

    Returns:
        - features: list, contains tuples with image path + histogram of features
    """
    kmeans = MiniBatchKMeans(n_clusters=nclusters)
    # This for loop passes the window "patch_shape" to extract individual 8x8x3 patches all along the tiles.
    # The extracted patches are used to fit the kmeans classifier
    features = []
    image_list_path = os.path.dirname(image_list[0])
    print('Extracting dense features from images in ' + image_list_path)

    if method == 'Dense':
        patch_shape = (8, 8, 3)

        for im in tqdm(image_list):
            image = imread(im)
            image = numpy.asarray(image)
            image = image.astype(float)
            patches = view_as_windows(image, patch_shape)
            plines = patches.shape[0]
            pcols = patches.shape[1]
            patches_reshaped = patches.reshape(plines, pcols, patch_shape[0] * patch_shape[1] * patch_shape[2])
            patches_reshaped = patches_reshaped.reshape(plines * pcols, patch_shape[0] * patch_shape[1] * patch_shape[2])
            kmeans.partial_fit(patches_reshaped)

        # This loop gets again the features of each tile and gets a list of the histograms of each individual tile
        for im in tqdm(image_list):
            image = imread(im)
            image = numpy.asarray(image)
            image = image.astype(float)
            patches = view_as_windows(image, patch_shape)
            plines = patches.shape[0]
            pcols = patches.shape[1]
            patches_reshaped = patches.reshape(plines, pcols, patch_shape[0] * patch_shape[1] * patch_shape[2])
            patches_reshaped = patches_reshaped.reshape(plines * pcols, patch_shape[0] * patch_shape[1] * patch_shape[2])
            result = kmeans.predict(patches_reshaped)
            histogram = numpy.histogram(result, bins=nclusters - 1)
            features.append((im, histogram[0]))

        return features

    elif method == 'Daisy':
        patch_shape = (8, 8)
        p = 0
        q = 0
        r = 0
        # extraction
        for im in tqdm(image_list):
            image = imread(im)
            image = numpy.asarray(rgb2grey(image))
            daisyzy = daisy(image, step=1, radius=8, rings=3)
            # daisy has shape P, Q, R
            p = daisyzy.shape[0]
            q = daisyzy.shape[1]
            r = daisyzy.shape[2]
            daisyzy_reshaped = daisyzy.reshape(p * q, r)
            kmeans.partial_fit(daisyzy_reshaped)

        for im in tqdm(image_list):
            image = imread(im)
            image = numpy.asarray(rgb2grey(image))
            daisyzy = daisy(image, step=1, radius=8, rings=3)
            # daisy has shape P, Q, R
            p = daisyzy.shape[0]
            q = daisyzy.shape[1]
            r = daisyzy.shape[2]
            daisyzy_reshaped = daisyzy.reshape(p * q, r)
            result = kmeans.predict(daisyzy_reshaped)
            histogram = numpy.histogram(result, bins=nclusters - 1)
            features.append((im, histogram[0]))

        return features

    else:
        print('Method not valid')
        return


def get_features_CNN(image_list, model='VGG16'):
    """
    Extracts image features using CNN

    Arguments:
        - image_list: list, image set
        - model: str, VGG16 or Xception

    Returns:
        - features: list, contains tuples with image path + histogram of features
    """
    if model == 'VGG16':
        print('Loading network...')
        model = VGG16(weights='imagenet', include_top=False)
        model.summary()

        batch = []
        for im in tqdm(image_list):
            image = imread(im)
            image = numpy.asarray(image)
            image = imagenet_utils.preprocess_input(image)
            batch.append(image)

        batch = numpy.array(batch)
        features = model.predict(batch, batch_size=32)
        features_flatten = features.reshape((features.shape[0], features.shape[1] * features.shape[2] * features.shape[3]))

        features = []
        for i in range(len(features_flatten)):
            features.append(os.path.basename(image_list[i]), features_flatten[i])

        return features


def divide_dab(path, classifier):
    """
    Divides set of images according to the presence of DAB staining

    Arguments:
        - path: str, path to image folder
        - classifier: int arr

    Returns:
        - classifier: int arr
        - image_positive, image_negative: list
    """

    image_path = os.path.join(path, "*.jpg")
    image_path = glob.glob(image_path)
    image_positive = []
    image_negative = []

    for im in tqdm(image_path):
        image = imread(im)
        name = os.path.basename(im)
        number = name.split('_')
        number = int(number[0])
        classifier[number][0] = number
        if detect_dab.detect_dab(image):
            classifier[number][1] = 1
            image_positive.append(im)
        else:
            classifier[number][1] = 0
            image_negative.append(im)

    print('Division in path ' + path + ' completed.')

    return classifier, image_positive, image_negative


def image_cluster(features, classifier, n, method='Kmeans'):
    """
    """
    features, image_list = feature_list_division(features)
    features_1 = []
    features_0 = []

    if len(image_list) < 2:
        return classifier, features_1, features_0

    if method == 'Kmeans':
        cls = MiniBatchKMeans(n_clusters=2)
        labels = cls.fit_predict(features)

    else:
        print('Method not valid')

    for im in tqdm(image_list):
        # Gets the index of the image
        index = image_list.index(im)
        image_name = os.path.basename(im)
        number = image_name.split('_')
        number = int(number[0])
        image = imread(im)
        if labels[index] == 1:
            classifier[number][n] = 1
            features_1.append((im, features[index]))
        if labels[index] == 0:
            classifier[number][n] = 0
            features_0.append((im, features[index]))

    return classifier, features_1, features_0


def feature_list_division(list_features):
    """
    Gets a list with elements ('image_name', 'array of features') and returns a
    numpy array with the features and a separate list with the image_names
    """

    features = []
    image_list = []
    for i in range(len(list_features)):
        image_list.append(list_features[i][0])
        features.append(list_features[i][1])
    features = numpy.array(features)

    return features, image_list


def pickle_save(file, path, name):
    file_path = os.path.join(path, name)
    with open(file_path, "wb") as f:
        pickle.dump(file, f)


def pickle_load(file_name):
    with open(file_name, "rb") as f:
        file = pickle.load(f)
    return file

###############################################################################
#
# MAIN
#
###############################################################################


if __name__ == "__main__":

    # Manage parameters
    parser = argparse.ArgumentParser(description='Script that divides a WSI in individual patches and classifies the resulting tiles in similarity groups. PRUEBA CSV')
    parser.add_argument('-S', '--Slide', type=str, help='path to slide')
    parser.add_argument('--outpath', type=str, help='path to outfolder')
    parser.add_argument('-n', '--n_division', type=int, default=4, help='number of divisions [Default: %(default)s]')
    parser.add_argument('-l', '--level', type=int, default=13,  help='division level of slide [Default: %(default)s]')
    parser.add_argument('--tissue_ratio', type=float, default=0.25, help='tissue ratio per patch [Default: %(default)s]')
    parser.add_argument('--tile_size', type=int, default=256, help='tile heigth and width in pixels [Default: %(default)s]')
    parser.add_argument('--feature_method', type=str, default='Dense', help='features extracted from individual patches [Default: %(default)s]')
    parser.add_argument('--flag', type=int, default=0, help='step of the process, from 1 to 5')
    group_f1 = parser.add_argument_group('Flag 1')
    group_f1.add_argument('--path_1', type=str, help='path to the folder with the patches')
    group_f2 = parser.add_argument_group('Flag 2')
    group_f2.add_argument('--list_positive', type=str, help='path to list_positive.p')
    group_f2.add_argument('--classifier', type=str, help='path to classifier.p')
    group_f3 = parser.add_argument_group('Flag 3')
    group_f3.add_argument('--feat_file', type=str, help='path to feat_file.txt')

    args = parser.parse_args()

    flag = args.flag
    # Gets patches from initial slide
    if flag < 1:
        n, outpath = get_patches(args.Slide, args.outpath, args.level, args.tissue_ratio, args.tile_size)
        flag = 1

    if flag < 2:
        if args.flag == 1:
            outpath = args.path_1
            outpath_images = os.path.join(outpath, '*.jpg')
            n = len(glob.glob(outpath_images))

        n_columns = args.n_division + 2
        print('Number of columns: {}. Number of rows: {}'.format(n_columns, n))
        classifier = numpy.zeros((n, n_columns))
        classifier = classifier.astype(int)
        classifier, list_positive, list_negative = divide_dab(outpath, classifier)
        flag = 2

        pickle_save(classifier, outpath, 'classifier.p')
        pickle_save(list_positive, outpath, 'list_positive.p')

    if flag < 3:
        if args.flag == 2:
            classifier = pickle_load(args.classifier)
            list_postive = pickle_load(args.list_positive)

        # Extract features from positive images

        if args.feature_method == 'Dense':
            features = get_features(list_positive, nclusters=256, method=args.feature_method)

        if args.feature_method == 'Daisy':
            features = get_features(list_positive, nclusters=256, method=args.feature_method)

        if args.feature_method == 'CNN':
            features = get_features_CNN(list_positive)

        pickle_save(features, outpath, 'features.txt')

        flag == 3

    if flag < 4:
        if args.flag == 3:
            features = pickle_load(args.feat_file)
            classifier = pickle_load(args.classifier)

        param = []
        param.append(features)
        n_division = args.n_division
        k = 0

        for i in range(n_division):
            n_level = i + 2
            for j in range(2**i):
                index = j + 2**i - 1
                curr_features = param[index]
                classifier, f1, f0 = image_cluster(curr_features, classifier, n_level)
                param.append(f1)
                param.append(f0)
                number_divisions = 2**i
                print('Division completed - division {} out of {} in level {}'.format(j, number_divisions, i))
                print()

        pickle_save(classifier, outpath, 'classifier.p')

        # Save to csvfile

        csv_cluster = 'cluster_division.csv'
        csv_features = 'features.csv'
        csv_file_path_cluster = os.path.join(outpath, csv_cluster)
        csv_columns = ["Slide_number"]
        csv_columns.append('Positive')
        for i in range(args.n_division):
            csv_columns.append('Level_{}'.format(i))

        with open(csv_file_path_cluster, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, csv_columns)
            writer.writeheader()
            for i in range(classifier.shape[0]):
                row = {'Slide_number': classifier[i][0], 'Positive': classifier[i][1]}
                for j in range(args.n_division):
                    row["Level_{}".format(j)] = classifier[i][j+2]
                writer.writerow(row)

        csv_file_path_features = os.path.join(outpath, csv_features)
        final_feat, final_imag_list = feature_list_division(features)
        csv_columns = ["Slidename"]
        csv_columns.append('Number')
        csv_columns.append('X')
        csv_columns.append('Y')
        shape_feat = final_feat.shape
        for i in range(shape_feat[1]):
            csv_columns.append('feature_{}'.format(i))

        with open(csv_file_path_features, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, csv_columns)
            writer.writeheader()
            for im in final_imag_list:
                index = final_imag_list.index(im)
                im = os.path.basename(im)
                data = im.split('.')[0]
                data = data.split('_')
                row = {'Slidename': data[1], 'Number': data[0], 'X': data[3], 'Y': data[4]}
                for i in range(shape_feat[1]):
                    row['feature_{}'.format(i)] = final_feat[index][i]
                writer.writerow(row)

        # Save images to clusters
        clusters = os.path.join(outpath, 'clusters')
        outpath_images = os.path.join(outpath, '*.jpg')
        try:
            os.mkdir(clusters)
            print("Directory", clusters, "created")
        except FileExistsError:
            print("Directory", clusters, "already exists")

        for i in range(2**n_division):
            dir = os.path.join(clusters, '{}'.format(i))
            try:
                os.mkdir(dir)
                print('Directory', dir, 'created')
            except FileExistsError:
                print('Directory', dir, 'already exists')
            list_ref_images = class_to_cluster.class_to_cluster(classifier, n_division, i)
            list_images = glob.glob(outpath_images)
            for im_ref in list_ref_images:
                image_name = os.path.basename(list_images[im_ref])
                image = imread(list_images[im_ref])
                image_path = os.path.join(dir, image_name)
                imsave(image_path, image)
