import pickle
import os
import numpy
import tensorflow as tf
from matplotlib import pyplot as plt
import argparse
from PIL import Image
from kmeanstf import KMeansTF
from tqdm import tqdm
import time

from sklearn.cluster import MiniBatchKMeans
from skimage.color import rgb2hed
from skimage.util.shape import view_as_windows
from sklearn.metrics.pairwise import paired_distances
from auxiliary_functions.pickle_functions import *
from auxiliary_functions.get_patch_reshaped import *

# Importing Keras libraries
from keras.applications.xception import preprocess_input


def get_features(image_list, n_words=256, learn_ratio=50):

    # This for loop passes the window "patch_shape" to extract individual 8x8x3
    # patches all along the tiles.
    # The extracted patches are used to fit the kmeans classifier
    features = []
    image_list_path = os.path.dirname(image_list[0])
    image_list_path = os.path.dirname(image_list_path)
    print('Extracting feature from images in '.format(image_list_path))

    inertia = []
    pdab_list = []
    ph_list = []
    patch_shape = (8, 8)
    print('****')
    print('Step 0: Sliding window')
    start = time.time()
    # Fits k-means in 1/50 of the images
    for i in tqdm(range(0, len(image_list), learn_ratio)):
        with Image.open(image_list[i]) as image:
            image = numpy.asarray(rgb2hed(image))
            if (image.shape[0] == image.shape[1]):
                im_dab = image[:, :, 2]
                im_h = image[:, :, 0]
                im_dab = preprocess_input(im_dab)
                im_h = preprocess_input(im_h)
                im_dab = im_dab.astype(float)
                im_h = im_h.astype(float)
                p_dab = view_as_windows(im_dab, patch_shape)
                p_h = view_as_windows(im_h, patch_shape)
                p_dab = get_patch_reshaped(p_dab, patch_shape)
                p_h = get_patch_reshaped(p_h, patch_shape)
                pdab_list.extend(p_dab)
                ph_list.extend(p_h)
    end = time.time()
    print('Total time Sliding Window: {:.4f} s'.format(end - start))
    print()

    print('****')
    print('Step 1: KMeans fitting (MiniBatchKMeans)')
    start = time.time()
    kmeans_dab = MiniBatchKMeans(n_clusters=n_words)
    kmeans_h = MiniBatchKMeans(n_clusters=n_words)
    kmeans_dab.fit(p_dab)
    kmeans_h.fit(p_h)
    end = time.time()
    print('Inertia DAB: {}'.format(kmeans_dab.inertia_))
    print('Inertia H: {}'.format(kmeans_h.inertia_))
    print()
    print('Total time MiniBatchKMeans fitting: {:.4f} s'.format(end - start))
    print()

    print('****')
    print('Step 1: KMeans fitting (KMeansTF)')
    start = time.time()
    kmeanstf_dab = KMeansTF(n_clusters=n_words)
    kmeanstf_h = KMeansTF(n_clusters=n_words)
    kmeanstf_dab.fit(p_dab)
    kmeanstf_h.fit(p_h)
    end = time.time()
    print('Inertia DAB: {}'.format(kmeanstf_dab.inertia_))
    print('Inertia H: {}'.format(kmeanstf_h.inertia_))
    print()
    print('Total time KMeansTF fitting: {:.4f} s'.format(end - start))
    print()

    # Generate the filter set for the convolution layer
    kernel_in = []
    k_shape = (8, 8, 1)
    for i in range(k_shape[0]):
        for j in range(k_shape[1]):
            k = numpy.zeros(k_shape)
            k[i, j, :] = 1
            kernel_in.append(k)

    kernel_in = numpy.asarray(kernel_in)
    kernel_in = kernel_in.reshape(
        (k_shape[0], k_shape[1], k_shape[2], k_shape[0] * k_shape[1]))
    kernel_in = tf.constant(kernel_in, dtype=tf.float32)

    inertia = []
    pdab_list = []
    ph_list = []
    patch_shape = (8, 8)
    print('****')
    print('Step 0: Tensorflow Conv2D')
    start = time.time()
    # Fits k-means in 1/50 of the images
    for i in tqdm(range(0, len(image_list), learn_ratio)):
        with Image.open(image_list[i]) as image:
            image = numpy.asarray(rgb2hed(image))
            if (image.shape[0] == image.shape[1]):
                im_dab = image[:, :, 2]
                im_h = image[:, :, 0]
                im_dab = preprocess_input(im_dab)
                im_h = preprocess_input(im_h)
                im_dab = im_dab.astype(float)
                im_h = im_h.astype(float)
                im_dab = im_dab.reshape(1, 224, 224, 1)
                im_h = im_h.reshape(1, 224, 224, 1)
                im_dab = tf.constant(im_dab, dtype=tf.float32)
                im_h = tf.constant(im_h, dtype=tf.float32)
                p_dab = tf.nn.conv2d(
                    input=im_dab, filters=kernel_in, strides=1, padding='SAME')
                p_h = tf.nn.conv2d(input=im_h, filters=kernel_in,
                                   strides=1, padding='SAME')
                shape = p_dab.numpy().shape
                pdab_list.extend(p_dab.numpy().reshape(
                    shape[0] * shape[1] * shape[2], shape[3]))
                ph_list.extend(p_h.numpy().reshape(
                    shape[0] * shape[1] * shape[2], shape[3]))
    end = time.time()
    print('Total time Tensorflow Conv2D: {:.4f} s'.format(end - start))
    print()

    inertia = []
    pdab_list = []
    ph_list = []
    patch_shape = (8, 8)
    print('****')
    print('Step 0: Tensorflow ExtractPatches')
    start = time.time()
    # Fits k-means in 1/50 of the images
    for i in tqdm(range(0, len(image_list), learn_ratio)):
        with Image.open(image_list[i]) as image:
            image = numpy.asarray(rgb2hed(image))
            if (image.shape[0] == image.shape[1]):
                im_dab = image[:, :, 2]
                im_h = image[:, :, 0]
                im_dab = preprocess_input(im_dab)
                im_h = preprocess_input(im_h)
                im_dab = im_dab.astype(float)
                im_h = im_h.astype(float)
                im_dab = im_dab.reshape(1, 224, 224, 1)
                im_h = im_h.reshape(1, 224, 224, 1)
                im_dab = tf.constant(im_dab, dtype=tf.float32)
                im_h = tf.constant(im_h, dtype=tf.float32)
                p_dab = tf.image.extract_patches(im_dab, sizes=[1, k_shape[0], k_shape[1], 1], strides=[
                                                 1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
                p_h = tf.image.extract_patches(im_h, sizes=[1, k_shape[0], k_shape[1], 1], strides=[
                                               1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
                shape = p_dab.numpy().shape
                pdab_list.extend(p_dab.numpy().reshape(
                    shape[0] * shape[1] * shape[2], shape[3]))
                ph_list.extend(p_h.numpy().reshape(
                    shape[0] * shape[1] * shape[2], shape[3]))
    end = time.time()
    print('Total time Tensorflow ExtractPatches: {:.4f} s'.format(end - start))
    print()

    return features, kmeans_dab, kmeans_h, kmeanstf_dab, kmeanstf_h


if __name__ == "__main__":

    print('##################################################################')
    print('#')
    print('# TEST FEATURE EXTRACTION')
    print('#')
    print('##################################################################')

    list_positive = '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/2_Tests_2209/list_positive_16_224.p'
    list_positive = pickle_load(list_positive)
    outpath = '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/'
    features, kmeans_dab, kmeans_h, kmeanstf_dab, kmeanstf_h = get_features(
        list_positive, n_words=100, learn_ratio=1000)

    print('##################################################################')
    print('#')
    print('# DISPLAYING FEATURES')
    print('#')
    print('##################################################################')

    print('Getting Figure KMeans DAB')
    figsize = (13, 11)
    fig, axes = plt.subplots(10, 10, figsize=figsize, sharex=True, sharey=True)
    ax = axes.ravel()
    for i, patch in enumerate(kmeans_dab.cluster_centers_):
        ax[i].imshow(patch.reshape((8, 8)), cmap=plt.cm.gray,
                     interpolation='nearest')
        plt.xticks(())
        plt.yticks(())

    plt.suptitle('Features', fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    plt.show()
    name = os.path.join(outpath, 'Dict_features_KMeans_DAB.jpg')
    fig.savefig(name, bbox_inches='tight', dpi=fig.dpi)

    print('Getting Figure KMeans H')
    fig, axes = plt.subplots(10, 10, figsize=figsize, sharex=True, sharey=True)
    ax = axes.ravel()
    for i, patch in enumerate(kmeans_h.cluster_centers_):
        ax[i].imshow(patch.reshape((8, 8)), cmap=plt.cm.gray,
                     interpolation='nearest')
        plt.xticks(())
        plt.yticks(())

    fig.suptitle('Features', fontsize=16)
    fig.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    plt.show()
    name = os.path.join(outpath, 'Dict_features_KMeans_H.jpg')
    fig.savefig(name, bbox_inches='tight', dpi=fig.dpi)

    print('Getting Figure KMeansTF DAB')
    fig, axes = plt.subplots(10, 10, figsize=figsize, sharex=True, sharey=True)
    ax = axes.ravel()
    for i, patch in enumerate(kmeanstf_dab.cluster_centers_.numpy()):
        ax[i].imshow(patch.reshape((8, 8)), cmap=plt.cm.gray,
                     interpolation='nearest')
        plt.xticks(())
        plt.yticks(())

    plt.suptitle('Features', fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    plt.show()
    name = os.path.join(outpath, 'Dict_features_KMeansTF_DAB.jpg')
    fig.savefig(name, bbox_inches='tight', dpi=fig.dpi)

    print('Getting Figure KMeansTF H')
    fig, axes = plt.subplots(10, 10, figsize=figsize, sharex=True, sharey=True)
    ax = axes.ravel()
    for i, patch in enumerate(kmeanstf_h.cluster_centers_.numpy()):
        ax[i].imshow(patch.reshape((8, 8)), cmap=plt.cm.gray,
                     interpolation='nearest')
        plt.xticks(())
        plt.yticks(())

    plt.suptitle('Features', fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    plt.show()
    name = os.path.join(outpath, 'Dict_features_KMeansTF_H.jpg')
    fig.savefig(name, bbox_inches='tight', dpi=fig.dpi)
