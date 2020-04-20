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

###############################################################################
#
# FUNCTIONS
#
###############################################################################

def get_patches(slidepath, outpath, level = 10, tissue_ratio = 0.25, size = 256):
    """
    Function that divides a slide into patches with different resolution

    Arguments:
        - slidepath: str, path to the image to patchify
        - outpath: str, path to the folder in which the patches will be saved
        - level: int, level in which image is patchified. The bigger the level,
          the higher the number of patches and the resolution of the images.
        - tissue_ratio: float, minimum surface of tissue tile to be considered

    Returns:
        - n: int, number of patches
        - outpath: str, folder in which the patches are stored

    """

    #Opens the slide with OpenSlide
    slide = OpenSlide(slidepath)
    slide_dz = deepzoom.DeepZoomGenerator(slide, tile_size = (size - 2), overlap = 1)
    PATH = outpath + "/level_{}".format(level)

    #Makes directory to store the patches
    try:
        os.mkdir(PATH)
        print("Directory", PATH, "created")
    except FileExistsError:
        print("Directory", PATH, "already exists")

    #Asures that the chosen level is valid
    if level < slide_dz.level_count:
        tiles = slide_dz.level_tiles[level]
        print('Level {} : {} tiles (empty tiles included)'.format(level, slide_dz.level_tiles[level][0]*slide_dz.level_tiles[level][1]))
    else:
        print('Invalid level')
        return

    #Saves tiles if detects tissue presence higher than tissue_ratio
    n=0
    print("Saving tiles image "+ slidepath + "...")
    for i in tqdm(range(tiles[0])):
      for j in range(tiles[1]):
        tile = slide_dz.get_tile(level,(i,j))
        tile_path = PATH + '/{}_slide1_level{}_{}_{}.jpg'.format(n,level,i,j)
        image = numpy.array(tile)[..., :3]
        mask = tissue.get_tissue_from_rgb(image)
        if mask.sum() > tissue_ratio * tile.size[0] * tile.size[1]:
            tile.save(tile_path)
            n = n + 1
    print('Total of {} tiles in level {}'.format(n,level))
    return n, PATH

def get_features_SIFT(image_path, total):
    n=0
    des_list = []
    image_path = image_path + '/*.jpg'
    print('Obtaining SIFT features from images in ' + image_path + '...')
    for im in tqdm(glob.glob(image_path)):
        image = cv2.imread(im, 0)
        n = n + 1
        keypoints, descriptors = pysift.computeKeypointsAndDescriptors(image)
        #print('Features obtained from image '+ str(n)+ ' of ' + str(total))
        des_list.append((im, descriptors))
        #if n%20 == 0:
          #sys.stdout.write(str(n)+" ")
          #sys.stdout.flush()
    # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]
    for im, descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))  # Stacking the descriptors
    return descriptors

def get_features(path, nclusters = 256, method = 'Dense'):

    """
    Gets the histogram of features of the given set of images. It obtains the
    features by means of a KMeans clustering algorithm.

    Arguments:
        - path: str, address to the folder with the images
        - nclusters: int, number of visual words in which the features are clustered
        - method: str, Dense or Daisy
    """
    kmeans = MiniBatchKMeans(n_clusters = nclusters)
    #This for loop passes the window "patch_shape" to extract individual 8x8x3 patches all along the tiles.
    #The extracted patches are used to fit the kmeans classifier
    image_path = path + "/*.jpg"
    features = []
    image_list = []

    if method == 'Dense':
        print('Extracting dense features from images in '+ path)
        patch_shape = (8, 8, 3)
        for im in tqdm(glob.glob(image_path)):
            image = imread(im)
            image = numpy.asarray(image)
            image = image.astype(float)
            patches = view_as_windows(image, patch_shape)
            plines = patches.shape[0]
            pcols = patches.shape[1]
            patches_reshaped = patches.reshape(plines, pcols, patch_shape[0] * patch_shape[1] * patch_shape[2])
            patches_reshaped = patches_reshaped.reshape(plines * pcols, patch_shape[0] * patch_shape[1] * patch_shape[2])
            kmeans.partial_fit(patches_reshaped)

        #This loop gets again the features of each tile and gets a list of the histograms of each individual tile
        for im in tqdm(glob.glob(image_path)):
            image = imread(im)
            image = numpy.asarray(image)
            image = image.astype(float)
            patches = view_as_windows(image, patch_shape)
            plines = patches.shape[0]
            pcols = patches.shape[1]
            patches_reshaped = patches.reshape(plines, pcols, patch_shape[0] * patch_shape[1] * patch_shape[2])
            patches_reshaped = patches_reshaped.reshape(plines * pcols, patch_shape[0] * patch_shape[1] * patch_shape[2])
            result = kmeans.predict(patches_reshaped)
            histogram = numpy.histogram(result, bins = nclusters - 1)
            #features.append((im[len(path):], histogram[0]))
            features.append(histogram[0])
            image_list.append(im[len(path):])

        #return features
        return features, image_list

    elif method == 'Daisy':
        print('Extracting Daisy features from images in '+ path)
        patch_shape = (8, 8)
        p = 0
        q = 0
        r = 0
        # extraction
        for im in tqdm(glob.glob(image_path)):
            image = imread(im)
            image = numpy.asarray(rgb2grey(image))
            daisyzy = daisy(image, step=1, radius=8, rings=3)
            # daisy has shape P, Q, R
            p = daisyzy.shape[0]
            q = daisyzy.shape[1]
            r = daisyzy.shape[2]
            daisyzy_reshaped = daisyzy.reshape(p * q, r)
            kmeans.partial_fit(daisyzy_reshaped)

        for im in tqdm(glob.glob(image_path)):
            image = imread(im)
            image = numpy.asarray(rgb2grey(image))
            daisyzy = daisy(image, step=1, radius=8, rings=3)
            # daisy has shape P, Q, R
            p = daisyzy.shape[0]
            q = daisyzy.shape[1]
            r = daisyzy.shape[2]
            daisyzy_reshaped = daisyzy.reshape(p * q, r)
            result = kmeans.predict(daisyzy_reshaped)
            histogram = numpy.histogram(result, bins = nclusters - 1)
            features.append(histogram[0])
            image_list.append(im[len(path):])

        return features, image_list

    else:
        print('Method not valid')
        return

def divide_dab(path, positive, negative):
    for im in tqdm(glob.glob(path + '/*.jpg')):
        image = imread(im)
        if detect_dab.detect_dab(image):
            im_name = positive + im[len(path):]
            imsave(im_name, image)
        else:
            im_name = negative + im[len(path):]
            imsave(im_name, image)
    print('Division in path '+ path + ' completed.')

def image_cluster(features, path, image_list, method = 'Kmeans'):
    """
    features: array, size nxm
    path: str, address to save the images
    images_paths: list, len n
    """

    if method == 'Kmeans':
        cls = MiniBatchKMeans(n_clusters=2)
        labels = cls.fit_predict(features)

    else:
        print('Method not valid')

    #Cluster_1
    path_1 = path + "/1"
    try:
        os.mkdir(path_1)
        print("Directory", path_1, "created")
    except FileExistsError:
        print("Directory", path_1, "already exists")

    #Cluster_0
    path_0 = path + "/0"
    try:
        os.mkdir(path_0)
        print("Directory", path_0, "created")
    except FileExistsError:
        print("Directory", path_0, "already exists")

    features_1 = []
    list_1 = []
    features_0 = []
    list_0 = []

    images_paths = glob.glob(path + '/*.jpg')
    for im in tqdm(images_paths):
        #Gets the index of the image
        index = image_list.index(im[len(path):])
        image = imread(im)
        if labels[index] == 1:
            tile_path = path_1 + im[len(path):]
            imsave(tile_path, image)
            features_1.append(features[index])
            list_1.append(im[len(path):])
        if labels[index] == 0:
            tile_path = path_0 + im[len(path):]
            imsave(tile_path, image)
            features_0.append(features[index])
            list_0.append(im[len(path):])

    return features_1, features_0, list_1, list_0, path_1, path_0

###############################################################################
#
# MAIN
#
###############################################################################

if __name__ == "__main__":

    #Manage parameters
    parser = argparse.ArgumentParser(description='Script that divides a WSI in individual patches and classifies the resulting tiles in similarity groups.')
    parser.add_argument('-S', '--Slide', type = str, required = True, help = 'path to slide')
    parser.add_argument('--outpath', type = str, required = True, help = 'path to outfolder')
    parser.add_argument('-n', '--n_division', type = int, default = 4, help = 'number of divisions [Default: %(default)s]')
    parser.add_argument('-l', '--level', type = int, default = 13,  help = 'division level of slide [Default: %(default)s]')
    parser.add_argument('--tissue_ratio', type = float, default = 0.25, help = 'tissue ratio per patch [Default: %(default)s]')
    parser.add_argument('--tile_size', type = int, default = 256, help = 'tile heigth and width in pixels [Default: %(default)s]')
    parser.add_argument('-f', '--feature_method', type = str, default = 'Dense', help = 'features extracted from individual patches [Default: %(default)s]')
    parser.add_argument('--flag', type = int, default = 0, help = 'step of the process, from 1 to 5')
    group_f1 = parser.add_argument_group('Flag 1')
    group_f1.add_argument('-p', '--path_1', type = str, help = 'path to the folder with the patches')
    #group_f1.add_argument('-pos', '--path_pos', type = str, metavar = '', help = 'path to positive folder')
    group_f2 = parser.add_argument_group('Flag 2')
    group_f2.add_argument('--positive', type = str, help = 'path to positives')
    group_f3 = parser.add_argument_group('Flag 3')
    group_f3.add_argument('--feat_file', type = str, help = 'path to feat_file.txt')
    group_f3.add_argument('--il_file', type = str, help = 'path to il_file.txt')

    args = parser.parse_args()

    flag = args.flag
    #Gets patches from initial slide
    if flag < 1:
        n, outpath = get_patches(args.Slide, args.outpath, args.level, args.tissue_ratio, args.tile_size)
        flag = 1

    if flag < 2:
        if args.flag == 1:
            outpath = args.path_1
        #Classifies patches between presence or not
        positive = outpath + "/positive"
        negative = outpath + "/negative"
        try:
            os.mkdir(positive)
            print("Directory", positive, "created")
        except FileExistsError:
            print("Directory", positive, "already exists")

        try:
            os.mkdir(negative)
            print("Directory", negative, "created")
        except FileExistsError:
            print("Directory", negative, "already exists")

        divide_dab(outpath, positive, negative)
        flag = 2

    if flag < 3:
        #Extract features from positive images
        ### Quizá sea mejor dejar features y imagelist como una lista de dos columnas
        if args.flag == 2:
            positive = args.positive
        features, image_list = get_features(positive, nclusters = 256, method = args.feature_method)
        feat_file = positive + '/features.txt'
        il_file = positive + '/image_list.txt'
        with open(feat_file, "wb") as f:
            pickle.dump(features, f)

        with open(il_file, "wb") as f:
            pickle.dump(image_list, f)

        flag == 3

    if flag < 4:
        if args.flag == 3:
            with open(args.feat_file, "rb") as f:
                features = pickle.load(f)

            with open(args.il_file, "rb") as f:
                image_list = pickle.load(f)

            print('Features and image_list loaded')
            positive = args.positive

        features = numpy.array(features)

        param = []
        param.append((features, positive, image_list))
        n = args.n_division
        k = 0

        for i in range(n):
            for j in tqdm(range(2**i)):
                index = j + 2**i - 1
                features = param[index][0]
                path = param[index][1]
                list = param[index][2]
                f1, f0, l1, l0, p1, p0 = image_cluster(features, path, list)
                param.append((f1, p1, l1))
                param.append((f0, p0, l0))
                print('Folder '+ path + ' succesfully divided into ' + p1 + ' and ' + p0)

        # f1 = numpy.array(f1)
        # f0 = numpy.array(f0)
        # f11, f10, l11, l10, p11, p10 = image_cluster(f1, p1, l1)
        # f01, f00, l01, l00, p01, p00 = image_cluster(f0, p0, l0)
        #
        # f11 = numpy.array(f11)
        # f10 = numpy.array(f10)
        # f01 = numpy.array(f01)
        # f00 = numpy.array(f00)
        # f111, f110, l111, l110, p111, p110 = image_cluster(f11, p11, l11)
        # f101, f100, l101, l100, p101, p100 = image_cluster(f10, p10, l10)
        # f011, f010, l011, l010, p011, p010 = image_cluster(f01, p01, l01)
        # f001, f000, l001, l000, p001, p000 = image_cluster(f00, p00, l00)
