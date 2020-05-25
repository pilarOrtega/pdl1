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


# Importing Keras libraries
from keras.utils import np_utils
from keras.applications import VGG16, Xception
from keras.applications import imagenet_utils
from keras.applications.xception import preprocess_input


def imagetoDAB(image):
    image_hed = rgb2hed(image)
    d = image_hed[:, :, 2]
    img_dab = np.zeros_like(image)
    img_dab[:, :, 0] = d
    img_dab[:, :, 1] = d
    img_dab[:, :, 2] = d

    return img_dab


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
    list_features_batch = []
    if model in ['VGG16', 'VGG16DAB']:
        print('Loading network...')
        model = VGG16(weights='imagenet', include_top=False, pooling='avg')
        model.summary()

        x = 0
        n = 0
        for im in tqdm(image_list):
            image = imread(im)
            if model == 'VGG16DAB':
                image = imagetoDAB(image)
            image = numpy.asarray(image)
            image = numpy.expand_dims(image, axis=0)
            image = imagenet_utils.preprocess_input(image)
            curr_feat = model.predict(image)
            curr_feat = curr_feat.flatten()
            features.append((im, curr_feat))
            n += 1
            # When list has 100000 images + features, it is saved with pickle and a new list starts
            if n == 100000:
                features_batch = os.path.join(outpath, 'features_{}_batch{}.p'.format(model, x))
                with open(features_batch, "wb") as f:
                    pickle.dump(features, f)
                list_features_batch.append(features_batch)
                features = []
                n = 0
                x += 1

        features_batch = os.path.join(outpath, 'features_{}_batch{}.p'.format(model, x))
        with open(features_batch, "wb") as f:
            pickle.dump(features, f)
        list_features_batch.append(features_batch)

    if model in ['Xception', 'XceptionDAB']:
        print('Loading network...')
        model = Xception(weights='imagenet', include_top=False, pooling='avg')
        model.summary()

        x = 0
        n = 0
        for im in tqdm(image_list):
            image = imread(im)
            if model == 'XceptionDAB':
                image = imagetoDAB(image)
            image = numpy.asarray(image)
            image = numpy.expand_dims(image, axis=0)
            image = preprocess_input(image)
            curr_feat = model.predict(image)
            curr_feat = curr_feat.flatten()
            features.append((im, curr_feat))
            n += 1
            # When list has 100000 images + features, it is saved with pickle and a new list starts
            if n == 100000:
                features_batch = os.path.join(outpath, 'features_{}_batch{}.p'.format(model, x))
                with open(features_batch, "wb") as f:
                    pickle.dump(features, f)
                list_features_batch.append(features_batch)
                features = []
                n = 0
                x += 1

        features_batch = os.path.join(outpath, 'features_{}_batch{}.p'.format(model, x))
        with open(features_batch, "wb") as f:
            pickle.dump(features, f)
        list_features_batch.append(features_batch)

    return list_features_batch


def feature_reduction(list_features):
    features, image_list = feature_list_division(list_features)
    pca = IncrementalPCA(n_components=50)
    features_pca = pca.fit_transform(features)
    features_tsne = TSNE(n_components=2, random_state=123).fit_transform(features_pca)

    initial_features = features.shape[1]
    pca_features = features_pca.shape[1]
    final_features = features_tsne.shape[1]
    print('Number of features reduced from {} to {} ({} features after PCA)'.format(initial_features, final_features, pca_features))
    print()
    # StandardScaler normalizes the data
    features_scaled = StandardScaler().fit_transform(features_tsne)
    result = []
    for i in range(len(image_list)):
        result.append((image_list[i], features_scaled[i]))

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


def feature_extraction(list_positive, outpath, feature_method):

    print('[INFO] Extracting features from {} positive images'.format(len(list_positive)))

    start = time.time()
    # Extract features from positive images
    list_features_batch = get_features_CNN(list_positive, model=feature_method)
    end = time.time()
    print('Feature extraction completed in time {:.4f} s'.format(end-start))

    for lf in list_features_batch:
        with open(lf, "rb") as f:
            features = pickle.load(f)
        features = feature_reduction(features)
        with open(lf, "wb") ad f:
            pickle.dump(features, f)

    features = []
    for lf in list_features_batch:
        with open(lf, "rb") as f:
            batch = pickle.load(f)
            features.extend(batch)

    name = outpath
    name = os.path.basename(name)
    name = os.path.splitext(name)[0]
    name = name.split('_')
    level = name[1]

    csv_features = 'features_{}_level{}.csv'.format(feature_method, level)
    csv_file_path_features = os.path.join(outpath, csv_features)
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
        final_feat, final_imag_list = feature_list_division(features)
        for im in final_imag_list:
            index = final_imag_list.index(im)
            im_name = os.path.basename(im)
            data = os.path.splitext(im_name)[0]
            data = data.split('-')
            row = {'Slidename': data[0], 'Number': data[1], 'X': data[3], 'Y': data[4]}
            for i in range(shape_feat[1]):
                row['feature_{}'.format(i)] = final_feat[index][i]
            writer.writerow(row)

    return features


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script that discriminates patches positives to DAB.')
    parser.add_argument('-l', '--list_positive', type=str, help='file with slide list')
    parser.add_argument('-o', '--outpath', type=str, help='path to outfolder')
    parser.add_argument('-f', '--feature_method', type=str, choices=['Dense', 'DenseDAB', 'Daisy', 'DaisyDAB', 'VGG16', 'VGG16', 'Xception', 'XceptionDAB'], help='feature method')
    parser.add_argument('-d', '--device', default="0", type=str, help='GPU device to use [Default: %(default)s]')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    with open(args.list_positive, "rb") as f:
        list_positive = pickle.load(f)

    outpath = args.outpath
    feature_method = args.feature_method

    features = feature_extraction(list_positive, outpath, feature_method)
