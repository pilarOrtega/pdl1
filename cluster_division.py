import os
import numpy
import argparse
from matplotlib import pyplot as plt
import glob
import csv
from skimage.io import sift, imread, imsave
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import davies_bouldin_score
import pickle
from tqdm import tqdm
from auxiliary_functions.feature_list_division import *
from auxiliary_functions.pickle_functions import *


def cluster_division(features, classifiers_0, outpath, feature_method, ncluster=16, save=False, level=16, init=[]):
    """
    Arguments:
        - features:
        - classifiers_0: list with lentgh equal to the number of slides studied.
            Each element of the list contains three elements: (1) str, Slidename
            (2) str, Path to the folder in which patches are stored (3) arr,
            classification array with shape (n_patches, 4). These 4 columns
            contains 1) Patch number 2) X coordenate 3) Y coordenate 4) 1 if
            the patch is DAB positive or 0 otherwise
    """
    # Creates empty list in which the list of features is subsequentelly stored
    # Initially, we store the first feature array, including all DAB positive
    # patches
    param = []
    param.append(features)

    classifiers = []
    for s in classifiers_0:
        n_samples = s[2].shape[0]
        n_features = s[2].shape[1] + 3
        c = numpy.zeros((n_samples, n_features))
        c[:, : - 3] = s[2]
        classifiers.append((s[0], s[1], c))

    print('[INFO] Dividing patches into clusters')
    print('Total of {} images to be divided in {} clusters'.format(len(features), ncluster))
    print()
    features, image_list = feature_list_division(features)
    slide_list = []
    for x in classifiers:
        slide_list.append(x[0])
    if init == []:
        cls = MiniBatchKMeans(n_clusters=ncluster)
        cls = cls.fit(features)
        pickle_save(cls, outpath, 'model-{}-{}-BottomUp.p'.format(feature_method, level))
    else:
        cls = MiniBatchKMeans()
        cls.cluster_centers_ = init
    labels = cls.predict(features)
    distances = cls.transform(features)
    score = davies_bouldin_score(features, labels)
    print('Davies-Bouldin Score: {}'.format(score))
    for im in tqdm(image_list):
        index = image_list.index(im)
        image_name = os.path.basename(im)
        image_name = image_name.split('#')[1]
        number = image_name.split('-')
        number = int(number[0])
        slide_path = os.path.dirname(im)
        index_slide = slide_list.index(os.path.basename(slide_path))
        indices = distances[index].argsort()
        for i in range(3):
            classifiers[index_slide][2][number][4+i] = indices[i]

    print('Improving clusters...')
    n = 0
    for im in tqdm(image_list):
        index = image_list.index(im)
        indices = distances[index].argsort()
        if init != []:
            if (distances[index][indices[1]]-distances[index][indices[0]]) <= 0.5:
                if indices[1] < len(init) && indices[0] >= len(init):
                    print('Initial: {}, {}'.format(classifiers[index_slide][2][number][4], classifiers[index_slide][2][number][5]))
                    classifiers[index_slide][2][number][4], classifiers[index_slide][2][number][5] = classifiers[index_slide][2][number][5], classifiers[index_slide][2][number][4]
                    print('Final: {}, {}'.format(classifiers[index_slide][2][number][4], classifiers[index_slide][2][number][5]))
                    n += 1

    pickle_save(classifiers, outpath, 'class-{}-{}-BottomUp.p'.format(feature_method, level))

    if save:
        print('[INFO] Saving csv files...')
        print()
        for x in classifiers:
            csv_cluster = '{}-{}-level{}.csv'.format(x[0], feature_method, level)
            c = x[2]
            csv_file_path_cluster = os.path.join(outpath, csv_cluster)
            csv_columns = ["Patch_number"]
            csv_columns.append('X')
            csv_columns.append('Y')
            csv_columns.append('Positive')
            csv_columns.append('Cluster')

            with open(csv_file_path_cluster, 'w') as csv_file:
                writer = csv.DictWriter(csv_file, csv_columns)
                writer.writeheader()
                for i in range(c.shape[0]):
                    row = {'Patch_number': c[i][0], 'X': c[i][1], 'Y': c[i][2], 'Positive': c[i][3], 'Cluster': c[i][4]}
                    writer.writerow(row)

    return classifiers


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script that divides a set of patches into cluster hierarchically using KMeans.')
    parser.add_argument('-f', '--list_features', type=str, help='file with feature list')
    parser.add_argument('-c', '--classifiers', type=str, help='path to classification file')
    parser.add_argument('-o', '--outpath', type=str, help='path to outfolder')
    parser.add_argument('--nclusters', type=int, default=23)
    parser.add_argument('-i', '--init', default=0, help='File to initiation features [Default: %(default)s]')

    args = parser.parse_args()

    features = pickle_load(args.list_features)
    classifiers = pickle_load(args.classifiers)
    outpath = args.outpath
    init = args.init

    feature_method = args.list_features
    feature_method = os.path.basename(feature_method)
    feature_method = os.path.splitext(feature_method)[0]
    feature_method = feature_method.split('_')[1]

    if init == 0:
        classifiers = cluster_division(features, classifiers, outpath, feature_method, ncluster=args.nclusters)
    else:
        init = pickle_load(init)
        classifiers = cluster_division(features, classifiers, outpath, feature_method, init=init)
