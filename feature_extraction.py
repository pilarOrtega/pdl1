import os
import numpy
import argparse
from tqdm import tqdm
from skimage.io import sift, imread, imsave
from sklearn.cluster import MiniBatchKMeans
from skimage.color import rgb2grey, rgb2hed
from skimage.feature import daisy
from skimage.util.shape import view_as_windows
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.manifold import TSNE
import pickle
import itertools
import csv
from joblib import Parallel, delayed
import time
from numba import jit, njit
from slideminer.finetuning.fitting import *
from auxiliary_functions.pickle_functions import *
from auxiliary_functions.feature_list_division import *
from auxiliary_functions.imagetoDAB import *


# Importing Keras libraries
from keras.utils import np_utils
from keras.applications import VGG16, Xception
from keras.applications import imagenet_utils
from keras.applications.xception import preprocess_input


def get_patch_reshaped(patches, patch_shape):
    plines = patches.shape[0]
    pcols = patches.shape[1]
    if len(patch_shape) == 3:
        patches_reshaped = patches.reshape(plines, pcols, patch_shape[0] * patch_shape[1] * patch_shape[2])
        patches_reshaped = patches_reshaped.reshape(plines * pcols, patch_shape[0] * patch_shape[1] * patch_shape[2])
    if len(patch_shape) == 2:
        patches_reshaped = patches.reshape(plines, pcols, patch_shape[0] * patch_shape[1])
        patches_reshaped = patches_reshaped.reshape(plines * pcols, patch_shape[0] * patch_shape[1])
    return patches_reshaped


def hof_dense(im, kmeans, nclusters, method='DenseDAB'):
    """
    Function that gets the histogram of features (HoF) of a given image im for
    dense features.

    Arguments:
        - im: str, path to image
        - kmeans: sklearn MiniBatchKMeans fitted with a subset of all patches
        - nclusters: int, number of bins of the final HoF
        - dab: bool, set to true for only getting the HoF in the DAB channel

    Returns:
        - features: list with two elements: (1) im - path to image (2) histogram
            of features of the image
    """
    features = []
    image = imread(im)
    if method == 'DenseDAB':
        patch_shape = (8, 8)
        image = rgb2hed(image)
        image = image[:, :, 2]
    elif method == 'DenseH':
        patch_shape = (8, 8)
        image = rgb2hed(image)
        image = image[:, :, 0]
    else:
        patch_shape = (8, 8, 3)

    image = preprocess_input(image)
    image = numpy.asarray(image)
    image = image.astype(float)
    patches = view_as_windows(image, patch_shape)
    patches = numpy.ascontiguousarray(patches)
    patches_reshaped = get_patch_reshaped(patches, patch_shape)
    result = kmeans.predict(patches_reshaped)
    histogram = numpy.histogram(result, bins=nclusters - 1)
    features.extend((im, histogram[0]))
    return features


def hof_daisy(im, kmeans, nclusters, method='Daisy'):
    """
    Function that gets the histogram of features (HoF) of a given image im for
    daisy features.

    Arguments:
        - im: str, path to image
        - kmeans: sklearn MiniBatchKMeans fitted with a subset of all patches
        - nclusters: int, number of bins of the final HoF
        - dab: bool, set to true for only getting the HoF in the DAB channel

    Returns:
        - features: list with two elements: (1) im - path to image (2) histogram
            of features of the image
    """
    features = []
    image = imread(im)
    if method == 'DaisyDAB':
        image = numpy.asarray(rgb2hed(image))
        image = image[:, :, 2]
    if method == 'DaisyH':
        image = numpy.asarray(rgb2hed(image))
        image = image[:, :, 0]
    if method == 'Daisy':
        image = numpy.asarray(rgb2grey(image))
    image = preprocess_input(image)
    daisyzy = daisy(image, step=1, radius=8, rings=3)
    # daisy has shape P, Q, R
    p = daisyzy.shape[0]
    q = daisyzy.shape[1]
    r = daisyzy.shape[2]
    daisyzy_reshaped = daisyzy.reshape(p * q, r)
    result = kmeans.predict(daisyzy_reshaped)
    histogram = numpy.histogram(result, bins=nclusters - 1)
    features.extend((im, histogram[0]))
    return features


def get_features(image_list, nclusters=256, method='Dense'):

    """
    Gets the histogram of features of the given set of images. It obtains the
    features by means of a KMeans clustering algorithm.

    Arguments:
        - image_list: list which contains the list of images to extract HoF.
            Each element of the list is a path to an image.
        - nclusters: int, number of visual words in which the features are
            clustered. Default 256
        - method: str, Dense, DenseDAB, Daisy or DaisyDAB

    Returns:
        - features: list, contains tuples with image path and histogram of
            features for each image.
    """
    kmeans = MiniBatchKMeans(n_clusters=nclusters)
    # This for loop passes the window "patch_shape" to extract individual 8x8x3 patches all along the tiles.
    # The extracted patches are used to fit the kmeans classifier
    features = []
    image_list_path = os.path.dirname(image_list[0])
    image_list_path = os.path.dirname(image_list_path)
    print('Extracting features ({} method) from images in '.format(method) + image_list_path)

    if method in ['Dense', 'DenseDAB', 'DenseH']:

        start1 = time.time()
        print('Step 1: KMeans fitting')
        # Fits k-means in 1/50 of the images
        for i in tqdm(range(0, len(image_list), 50)):
            image = imread(image_list[i])
            if method == 'Dense':
                patch_shape = (8, 8, 3)
                image = numpy.asarray(image)
            if method == 'DenseDAB':
                patch_shape = (8, 8)
                image = numpy.asarray(rgb2hed(image))
                image = image[:, :, 2]
            if method == 'DenseH':
                patch_shape = (8, 8)
                image = numpy.asarray(rgb2hed(image))
                image = image[:, :, 0]

            image = preprocess_input(image)
            image = image.astype(float)
            patches = view_as_windows(image, patch_shape)
            patches_reshaped = get_patch_reshaped(patches, patch_shape)
            kmeans.partial_fit(patches_reshaped)
        end1 = time.time()
        print('Total time KMeans fitting: {:.4f} s'.format(end1-start1))

        start2 = time.time()
        # This loop gets again the features of each tile and gets a list of the histograms of each individual tile
        print('Step 2: Histogram of features extraction')
        features = Parallel(n_jobs=-2)(delayed(hof_dense)(im, kmeans, nclusters, method=method) for im in tqdm(image_list))
        end2 = time.time()
        print('Total time KMeans fitting: {:.4f} s'.format(end2-start2))

        print()
        print('Feature extraction completed')
        print()
        return features, kmeans

    elif method in ['Daisy', 'DaisyDAB', 'DaisyH']:
        patch_shape = (8, 8)
        p = 0
        q = 0
        r = 0
        # extraction
        start1 = time.time()
        print('Step 1: KMeans fitting')
        for i in tqdm(range(0, len(image_list), 50)):
            image = imread(image_list[i])
            if method == 'Daisy':
                image = numpy.asarray(rgb2grey(image))
            if method == 'DaisyDAB':
                image = numpy.asarray(rgb2hed(image))
                image = image[:, :, 2]
            if method == 'DaisyH':
                image = numpy.asarray(rgb2hed(image))
                image = image[:, :, 0]

            image = preprocess_input(image)
            daisyzy = daisy(image, step=1, radius=8, rings=3)
            # daisy has shape P, Q, R
            p = daisyzy.shape[0]
            q = daisyzy.shape[1]
            r = daisyzy.shape[2]
            daisyzy_reshaped = daisyzy.reshape(p * q, r)
            kmeans.partial_fit(daisyzy_reshaped)
        end1 = time.time()
        print('Total time KMeans fitting: {:.4f} s'.format(end1-start1))

        start2 = time.time()
        print('Step 2: Histogram of features extraction')
        features = Parallel(n_jobs=-3, backend='loky')(delayed(hof_daisy)(im, kmeans, nclusters, method=method) for im in tqdm(image_list))
        end2 = time.time()
        print('Total time KMeans fitting: {:.4f} s'.format(end2-start2))

        print()
        print('Feature extraction completed')
        print()
        return features, kmeans

    else:
        print('Method not valid')
        return


def get_features_CNN(image_list, outpath, model='VGG16', da=False):
    """
    Extracts image features using CNN

    Arguments:
        - image_list: list, image set
        - model: str, VGG16, VGG16DAB, Xception or XceptionDAB

    Returns:
        - features: list, contains tuples with image path + histogram of features
    """
    features = []
    if model in ['VGG16', 'VGG16DAB', 'VGG16H']:
        print('Loading network...')
        model = VGG16(weights='imagenet', include_top=False, pooling='avg')
        model.summary()

        for im in tqdm(image_list):
            image = imread(im)
            if model == 'VGG16DAB':
                image = imagetoDAB(image)
            if model == 'VGG16H':
                image = imagetoDAB(image, h=True)
            image = numpy.asarray(image)
            image = numpy.expand_dims(image, axis=0)
            image = imagenet_utils.preprocess_input(image)
            curr_feat = model.predict(image)
            curr_feat = curr_feat.flatten()
            features.append((im, curr_feat))

    if model in ['Xception', 'XceptionDAB', 'XceptionH']:
        print('Loading network...')

        if da:
            outdir = os.path.join(outpath, model)
            if not os.path.exists(outdir):
                if model == 'Xception':
                    domain_adaption(image_list, outdir, 224, pdl1=True)
                if model == 'XceptionDAB':
                    domain_adaption(image_list, outdir, 224, pdl1=True, dab=True)
                if model == 'XceptionH':
                    domain_adaption(image_list, outdir, 224, pdl1=True, h=True)
            weights_dir = os.path.join(outdir, 'weights')
            model = load_model(outdir, 5)
            model.summary()

        model = Xception(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

        for im in tqdm(image_list):
            image = imread(im)
            if model == 'XceptionDAB':
                image = imagetoDAB(image)
            if model == 'XceptionH':
                image = imagetoDAB(image, h=True)
            image = numpy.asarray(image)
            if image.shape == (224, 224, 3):
                image = numpy.expand_dims(image, axis=0)
                image = preprocess_input(image)
                curr_feat = model.predict(image)
                curr_feat = curr_feat.flatten()
                features.append((im, curr_feat))

    return features


def feature_reduction(list_features):
    features, image_list = feature_list_division(list_features)
    # We take the features that explain 90% of the variance
    pca = PCA(n_components=0.9)
    pca = pca.fit(features)
    #features_tsne = TSNE(n_components=2, random_state=123).fit_transform(features_pca)
    features_pca = pca.transform(features)
    initial_features = features.shape[1]
    pca_features = features_pca.shape[1]
    #final_features = features_tsne.shape[1]
    print('Number of features reduced from {} to {}'.format(initial_features, pca_features))
    print()
    # StandardScaler normalizes the data
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_pca)
    # List comprehension
    result = [(image_list[i], features_scaled[i]) for i in range(len(image_list))]
    # result = []
    # for i in range(len(image_list)):
    #     result.append((image_list[i], features_scaled[i]))

    return result, pca, scaler


def feature_extraction(list_positive, outpath, feature_method, da=False):

    print('[INFO] Extracting features from {} positive images'.format(len(list_positive)))

    start = time.time()
    # Extract features from positive images
    if feature_method in ['Dense', 'DenseDAB', 'DenseH', 'Daisy', 'DaisyDAB', 'DaisyH']:
        features, kmeans = get_features(list_positive, nclusters=256, method=feature_method)
        pickle_save(kmeans, outpath, 'kmeans_features_{}_level{}.p'.format(feature_method, 16))
    if feature_method in ['VGG16', 'VGG16DAB', 'VGG16H', 'Xception', 'XceptionDAB', 'XceptionH']:
        features = get_features_CNN(list_positive, outpath, model=feature_method, da=da)
    end = time.time()
    print('Feature extraction completed in time {:.4f} s'.format(end-start))

    print('Saving features...')
    start = time.time()
    name = outpath
    name = os.path.basename(name)
    name = os.path.splitext(name)[0]
    name = name.split('_')
    level = name[1]

    pickle_save(features, outpath, 'features_{}_level{}.p'.format(feature_method, level))

    start = time.time()
    features, pca, scaler = feature_reduction(features)
    end = time.time()
    print('Feature reduction completed in time {:.4f} s'.format(end-start))

    pickle_save(pca, outpath, 'pca_{}_level{}.p'.format(feature_method, level))
    pickle_save(scaler, outpath, 'scaler_{}_level{}.p'.format(feature_method, level))
    pickle_save(features, outpath, 'features_{}_level{}.p'.format(feature_method, level))

    csv_features = 'features_{}_level{}.csv'.format(feature_method, level)
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
        for im in tqdm(final_imag_list):
            index = final_imag_list.index(im)
            im_name = os.path.basename(im)
            data = os.path.splitext(im_name)[0]
            slidename = data.split('#')[0]
            data = data.split('#')[1]
            data = data.split('-')
            row = {'Slidename': slidename, 'Number': data[0], 'X': data[2], 'Y': data[3]}
            for i in range(shape_feat[1]):
                row['feature_{}'.format(i)] = final_feat[index][i]
            writer.writerow(row)
    end = time.time()
    print('Csv file correctly saved in {:.4f} s'.format(end-start))
    return features


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script that obtains the histogram of features of a set of images using different methods')
    parser.add_argument('-l', '--list_positive', type=str, help='file with slide list')
    parser.add_argument('-o', '--outpath', type=str, help='path to outfolder')
    parser.add_argument('-f', '--feature_method', type=str, help='feature method')
    parser.add_argument('-d', '--device', default="0", type=str, help='GPU device to use [Default: %(default)s]')
    parser.add_argument('--da', action='store_true')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    with open(args.list_positive, "rb") as f:
        list_positive = pickle.load(f)

    outpath = args.outpath
    feature_method = args.feature_method
    da = args.da

    features = feature_extraction(list_positive, outpath, feature_method)
