import os
from openslide import OpenSlide, deepzoom
import numpy
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

"""
Dense Features

This script gets the raw dense features of the patches of a given level, obtains the
histogram of features and classifies the patches in two clusters.

"""
#Path to the folder with image tiles of level X
level = 15
path = "/content/drive/My Drive/Slide1/level_{}/".format(level)
print('Obtaining dense features from images in ' + path + '...')

def get_hist_of_feat(path, nclusters, method = 'Dense'):

    kmeans = MiniBatchKMeans(n_clusters = nclusters)
    #This for loop passes the window "patch_shape" to extract individual 8x8x3 patches all along the tiles.
    #The extracted patches are used to fit the kmeans classifier
    patch_shape = (8, 8, 3)
    image_path = path + "*.jpg"

    if method == 'Dense':
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

    if method == 'Daisy':
        

    #This loop gets again the features of each tile and gets a list of the histograms of each individual tile
    features = []
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
        features.append(histogram[0])

    return features

def image_cluster(features, path, images_paths):
    """
    features: array, size nxm
    path: str, address to save the images
    images_paths: list, len n
    """
    cls = MiniBatchKMeans(n_clusters=2)
    labels = cls.fit_predict(features)

    #Cluster_1
    path_1 = path + "1/"
    try:
      os.mkdir(path_1)
      print("Directory", path_1, "created")
    except FileExistsError:
      print("Directory", path_1, "already exists")

    #Cluster_0
    path_0 = path + "0/"
    try:
      os.mkdir(path_0)
      print("Directory", path_0, "created")
    except FileExistsError:
      print("Directory", path_0, "already exists")

    n = 0
    cluster_1 = []
    cluster_0 = []
    for im in tqdm(images_paths):
        image = imread(im)
        if labels[n] == 1:
          tile_path = path_1 + '{}.jpg'.format(n)
          imsave(tile_path, image)
          cluster_1.append((tile_path, features[n]))
        if labels[n] == 0:
          tile_path = path_0 + '{}.jpg'.format(n)
          imsave(tile_path, image)
          cluster_0.append((tile_path, features[n]))
        n = n + 1

     return cluster_1, cluster_0

features = get_hist_of_feat(path, 256)
features = numpy.array(features)
images_paths = glob.glob(path + '*.jpg')
c1, c0 = image_cluster(features, path, images_paths)

#Divides the tiles in 2 clusters and provides a label to each tile
n = len(c0)
paths_c0 = []
features_c0 = numpy.zeros((n,255), int)
for i in range(n):
  paths_c0.append(c0[i][0])
  features_c0[i] = c0[i][1]

c01, c00 = image_cluster(features_c0, path, paths_c0)
