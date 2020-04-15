import os
from openslide import OpenSlide, deepzoom
import numpy
from matplotlib import pyplot as plt
import Pysiderois_Arnaud.pysliderois.tissue as tissue
import sys
import cv2
from PythonSIFT import pysift
import glob
from skimage.io import sift, imread
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from skimage.util.shape import view_as_windows

def get_patches(slidepath, outpath, level = 10):
    """
    Function that divides the slide in slidepath into level_tiles

    Arguments:
        - slidepath: str, path to the image to patchify
        - outpath: str, path to the resulting patches
        - level: int, level in which image is patchified

    Returns:
        - n: int, number of patches
        - outpath: str,

    """
    slide = OpenSlide(slidepath)
    slide_dz = deepzoom.DeepZoomGenerator(slide)
    PATH = outpath + "/level_{}".format(level)

    try:
        os.mkdir(PATH)
        print("Directory", PATH, "created")
    except FileExistsError:
        print("Directory", PATH, "already exists")

    tiles = slide_dz.level_tiles[level]
    n=0
    print("Saving tiles image "+ slidepath + "...")
    for i in range(tiles[0]):
      for j in range(tiles[1]):
        tile = slide_dz.get_tile(level,(i,j))
        tile_path = PATH + '/{}_slide1_level{}_{}_{}.jpg'.format(n,level,i,j)
        image = numpy.array(tile)[..., :3]
        mask = tissue.get_tissue_from_rgb(image)
        if mask.sum() > 0.25 * tile.size[0] * tile.size[1]:
            tile.save(tile_path)
            n = n + 1
            if n%20 == 0:
              sys.stdout.write(str(n)+" ")
              sys.stdout.flush()
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

def get_features_dense(image_path, total):
    image_path = image_path + '/*.jpg'
    print('Obtaining dense features from images in ' + image_path + '...')
    nclusters = 256
    kmeans = MiniBatchKMeans(n_clusters = nclusters)
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

    print('Obtaining histogram of features...')
    for im in tqdm(glob.glob(image_path)):
         image = imread(im)
         image = numpy.asarray(image)
         image = image.astype(float)
         patches = view_as_windows(image, patch_shape)
         plines = patches.shape[0]
         pcols = patches.shape[1]
         patches_reshaped = patches.reshape(plines, pcols, patch_shape[0] * patch_shape[1] * patch_shape[2])
         patches_reshaped = patches_reshaped.reshape(plines * pcols, patch_shape[0] * patch_shape[1] * patch_shape[2])
         frequence = kmeans.predict(patches_reshaped)
         histogram =

slidefile1 = "/Users/pilarortega/Desktop/STAGE_ONCOPOLE/slides/NVA_RC.PDL1.V1_18T040165.2B.4963.PDL1.mrxs"
outpath = "/Users/pilarortega/Desktop/STAGE_ONCOPOLE/patches"
n, outpath = get_patches(slidefile1, outpath, 15)
print("Tiles are saved in " + outpath)
#lista = get_features(outpath, n)
print("Features obtained. Proceding to KMeans clustering...")
k = 100
kmeans = MiniBatchKMeans(n_clusters=k)
