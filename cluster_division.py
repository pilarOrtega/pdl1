import os
import numpy
import argparse
from matplotlib import pyplot as plt
import glob
import csv
from skimage.io import sift, imread, imsave
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import davies_bouldin_score
import pickle
from tqdm import tqdm


def image_cluster(features, classifiers, n, method='Kmeans'):

    features, image_list = feature_list_division(features)
    slide_list = []
    for x in classifiers:
        slide_list.append(x[0])
    features_1 = []
    features_0 = []

    if len(image_list) < 2:
        return classifiers, features_1, features_0

    if method == 'Kmeans':
        cls = MiniBatchKMeans(n_clusters=2)
        labels = cls.fit_predict(features)
        score = davies_bouldin_score(features, labels)
        print('Davies-Bouldin Score: {}'.format(score))
    else:
        print('Method not valid')

    for im in tqdm(image_list):
        # Gets the index of the image
        index = image_list.index(im)
        image_name = os.path.basename(im)
        number = image_name.split('-')
        number = int(number[1])
        image = imread(im)
        slide_path = os.path.dirname(im)
        index_slide = slide_list.index(os.path.basename(slide_path))
        if labels[index] == 1:
            classifiers[index_slide][2][number][n] = 1
            features_1.append((im, features[index]))
        if labels[index] == 0:
            classifiers[index_slide][2][number][n] = 0
            features_0.append((im, features[index]))

    return classifiers, features_1, features_0


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


def cluster_division(features, classifiers_0, n_division, outpath, feature_method):
    param = []
    param.append(features)
    k = 0

    classifiers = []
    for s in classifiers_0:
        n_samples = s[2].shape[0]
        n_features = s[2].shape[1] + n_division
        c = numpy.zeros((n_samples, n_features))
        c[:, : - n_division] = s[2]
        classifiers.append((s[0], s[1], c))

    print('[INFO] Dividing patches into clusters')
    print('Total of {} images to be divided in {} clusters'.format(len(features), 2**n_division))
    print()
    for i in range(n_division):
        n_level = i + 4
        for j in range(2**i):
            index = j + 2**i - 1
            curr_features = param[index]
            classifiers, f1, f0 = image_cluster(curr_features, classifiers, n_level)
            param.append(f1)
            param.append(f0)
            number_divisions = 2**i
            print('Division completed - division {} out of {} in level {}'.format(j+1, number_divisions, i))
            print('    {} images in cluster {}'.format(len(f1), 1))
            print('    {} images in cluster {}'.format(len(f0), 0))
            print()

    name = outpath
    name = os.path.basename(name)
    name = os.path.splitext(name)[0]
    name = name.split('_')
    level = name[1]
    pickle_save(classifiers, outpath, 'class{}-{}.p'.format(feature_method, level))

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
        for i in range(n_division):
            csv_columns.append('Level_{}'.format(i))

        with open(csv_file_path_cluster, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, csv_columns)
            writer.writeheader()
            for i in range(c.shape[0]):
                row = {'Patch_number': c[i][0], 'X': c[i][1], 'Y': c[i][2], 'Positive': c[i][3]}
                for j in range(n_division):
                    row["Level_{}".format(j)] = c[i][j+4]
                writer.writerow(row)

    return classifiers


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script that discriminates patches positives to DAB.')
    parser.add_argument('-f', '--list_features', type=str, help='file with feature list')
    parser.add_argument('-c', '--classifiers', type=str, help='path to classification file')
    parser.add_argument('-n', '--n_division', type=int, default=4, help='number of divisions [Default: %(default)s]')
    parser.add_argument('-o', '--outpath', type=str, help='path to outfolder')

    args = parser.parse_args()

    with open(args.list_features, "rb") as f:
        features = pickle.load(f)
    with open(args.classifiers, "rb") as f:
        classifiers = pickle.load(f)
    outpath = args.outpath
    n_division = args.n_division

    feature_method = args.list_features
    feature_method = os.path.basename(feature_method)
    feature_method = os.path.splitext(feature_method)[0]
    feature_method = feature_method.split('_')[1]

    classifiers = cluster_division(features, classifiers, n_division, outpath, feature_method)