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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
import itertools
import csv

# Importing Keras libraries
from keras.utils import np_utils
from keras.applications import VGG16, Xception
from keras.applications import imagenet_utils
from keras.applications.xception import preprocess_input


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
    image_list_path = os.path.dirname(image_list_path)
    print('Extracting features ({} method) from images in '.format(method) + image_list_path)

    if method == 'Dense':
        patch_shape = (8, 8, 3)

        print('Step 1: KMeans fitting')
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
        print('Step 2: Histogram of features extraction')
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

        print('Feature extraction completed')
        print()
        return features

    elif method == 'Daisy' or method == 'Daisy_DAB':
        patch_shape = (8, 8)
        p = 0
        q = 0
        r = 0
        # extraction
        print('Step 1: KMeans fitting')
        for im in tqdm(image_list):
            image = imread(im)
            if method == 'Daisy':
                image = numpy.asarray(rgb2grey(image))
            if method == 'Daisy_DAB':
                image = numpy.asarray(rgb2hed(image))
                image = image[:, :, 2]
            daisyzy = daisy(image, step=1, radius=8, rings=3)
            # daisy has shape P, Q, R
            p = daisyzy.shape[0]
            q = daisyzy.shape[1]
            r = daisyzy.shape[2]
            daisyzy_reshaped = daisyzy.reshape(p * q, r)
            kmeans.partial_fit(daisyzy_reshaped)

        print('Step 2: Histogram of features extraction')
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

        print('Feature extraction completed')
        print()
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
    features = []
    if model == 'VGG16':
        print('Loading network...')
        model = VGG16(weights='imagenet', include_top=False, pooling='avg')
        model.summary()

        for im in tqdm(image_list):
            image = imread(im)
            image = numpy.asarray(image)
            image = numpy.expand_dims(image, axis=0)
            image = imagenet_utils.preprocess_input(image)
            curr_feat = model.predict(image)
            curr_feat = curr_feat.flatten()
            features.append((im, curr_feat))

    if model == 'Xception':
        print('Loading network...')
        model = Xception(weights='imagenet', include_top=False, pooling='avg')
        model.summary()

        for im in tqdm(image_list):
            image = imread(im)
            image = numpy.asarray(image)
            image = numpy.expand_dims(image, axis=0)
            image = preprocess_input(image)
            curr_feat = model.predict(image)
            curr_feat = curr_feat.flatten()
            features.append((im, curr_feat))

    return features


def feature_reduction(list_features):
    features, image_list = feature_list_division(list_features)
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features)
    features_tsne = TSNE(random_state=123).fit_transform(features_pca)

    initial_features = features.shape[1]
    final_features = features_tsne.shape[1]
    print('Number of features reduced from {} to {}'.format(initial_features, final_features))
    print()
    result = []
    for i in range(len(image_list)):
        result.append((image_list[i], features_tsne[i]))

    return result


def pickle_save(file, path, name):
    file_path = os.path.join(path, name)
    with open(file_path, "wb") as f:
        pickle.dump(file, f)


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


parser = argparse.ArgumentParser(description='Script that discriminates patches positives to DAB.')
parser.add_argument('-l', '--list_positive', type=str, help='file with slide list')
parser.add_argument('-o', '--outpath', type=str, help='path to outfolder')
parser.add_argument('-f', '--feature_method', type=str, choices=['Dense', 'Daisy', 'Daisy_DAB', 'VGG16', 'Xception'], help='feature method')

args = parser.parse_args()

with open(args.list_positive, "rb") as f:
    list_positive = pickle.load(f)

outpath = args.outpath
feature_method = args.feature_method

print('[INFO] Extracting features from {} positive images'.format(len(list_positive)))

# Extract features from positive images
if feature_method in ['Dense', 'Daisy', 'Daisy_DAB']:
    features = get_features(list_positive, nclusters=256, method=feature_method)

if feature_method in ['VGG16', 'Xception']:
    features = get_features_CNN(list_positive, model=feature_method)

features, image_list = feature_list_division(features)
features = feature_reduction(features)
# StandardScaler normalizes the data
features = StandardScaler().fit_transform(features)
features_reduced = []
for i in range(len(image_list)):
    features_reduced.append((image_list[i], features[i]))

name = args.list_positive
name = os.path.basename(name)
name = os.path.splitext(name)[0]
name = name.split('_')
level = name[1]

pickle_save(features_reduced, outpath, 'features_{}_level{}.p'.format(feature_method, level))

csv_features = 'features_{}_level{}.csv'.format(feature_method, level)
csv_file_path_features = os.path.join(outpath, csv_features)
final_feat, final_imag_list = feature_list_division(features_reduced)
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
        im_name = os.path.basename(im)
        data = os.path.splitext(im_name)[0]
        data = data.split('-')
        row = {'Slidename': data[0], 'Number': data[1], 'X': data[3], 'Y': data[4]}
        for i in range(shape_feat[1]):
            row['feature_{}'.format(i)] = final_feat[index][i]
        writer.writerow(row)
