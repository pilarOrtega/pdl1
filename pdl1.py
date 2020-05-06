__author__ = 'Pilar Ortega Arevalo'
__copyright__ = 'Copyright (C) 2019 IUCT-O'
__license__ = 'GNU General Public License'
__status__ = 'prod'

import os
import numpy
import argparse
from matplotlib import pyplot as plt
import glob
from skimage.io import sift, imread, imsave
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
from skimage.util.shape import view_as_windows
from skimage.color import rgb2grey, rgb2hed
from skimage.feature import daisy
import pickle
import csv
import class_to_cluster as ctc
import itertools


###############################################################################
#
# FUNCTIONS
#
###############################################################################

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
        print('Document ' + file_name + ' correctly loaded')
        print()
    return file

###############################################################################
#
# MAIN
#
###############################################################################


if __name__ == "__main__":

    # Manage parameters
    parser = argparse.ArgumentParser(description='Script that divides a WSI in individual patches and classifies the resulting tiles in similarity groups.')
    parser.add_argument('-S', '--Slide', type=str, help='path to slide')
    parser.add_argument('--outpath', type=str, required='True', help='path to outfolder')
    parser.add_argument('-n', '--n_division', type=int, default=4, help='number of divisions [Default: %(default)s]')
    parser.add_argument('-l', '--level', type=int, default=13,  help='division level of slide [Default: %(default)s]')
    parser.add_argument('--tissue_ratio', type=float, default=0.25, help='tissue ratio per patch [Default: %(default)s]')
    parser.add_argument('--tile_size', type=int, default=256, help='tile heigth and width in pixels [Default: %(default)s]')
    parser.add_argument('--feature_method', type=str, default='Dense', help='features extracted from individual patches [Default: %(default)s]')
    parser.add_argument('--flag', type=int, default=0, help='step of the process, from 1 to 5')
    parser.add_argument('--save_cluster', action='store_true', help='Set to True when cluster division desired')
    group_f1 = parser.add_argument_group('Flag 1')
    group_f1.add_argument('--classifier', type=str, help='path to classifier.p')
    group_f2 = parser.add_argument_group('Flag 2')
    group_f2.add_argument('--list_positive', type=str, help='path to list_positive.p')
    group_f3 = parser.add_argument_group('Flag 3')
    group_f3.add_argument('--feat_file', type=str, help='path to feat_file.txt')

    args = parser.parse_args()

    flag = args.flag
    outpath = args.outpath
    feature_method = args.feature_method
    tile_size = args.tile_size

    if feature_method == 'VGG16':
        tile_size = 224
    if feature_method == 'Xception':
        tile_size = 299

    print()
    print('[INFO] Starting execution from step {}'.format(flag))
    print()

    try:
        os.mkdir(outpath)
        print("Directory", outpath, "created")
        print()
    except FileExistsError:
        print("Directory", outpath, "already exists")
        print()

    # Gets patches from initial slide
    if flag < 1:
        slides = args.Slide
        slides = os.path.join(slides, '*.mrxs')
        level = args.level

        n_columns = args.n_division + 2
        classifiers = []
        n = 0
        for s in glob.glob(slides):
            print('[INFO] Extracting patches from slide {}'.format(s))
            n_s, outpath_slide = get_patches(s, outpath, level, args.tissue_ratio, tile_size)
            classifier = numpy.zeros((n_s, n_columns))
            classifier = classifier.astype(int)
            classifiers.append((os.path.basename(s), outpath_slide, classifier))
            n = n + n_s

        pickle_save(classifiers, outpath, 'classifiers.p')

        flag = 1

    if flag < 2:
        if args.flag == 1:
            classifiers = pickle_load(args.classifier)

        list_positive = []
        print('[INFO] Extracting positive patches...')
        for i in range(len(classifiers)):
            print('Getting positive patches from slide {} out of {}'.format(i, len(classifiers)-1))
            classifier, list_positive_x, list_negative_x = divide_dab(classifiers[i][1], classifiers[i][2])
            classifiers[i] = (classifiers[i][0], classifiers[i][1], classifier)
            list_positive += list_positive_x
        flag = 2

        pickle_save(classifiers, outpath, 'classifiers.p')
        pickle_save(list_positive, outpath, 'list_positive.p')

    if flag < 3:
        if args.flag == 2:
            classifiers = pickle_load(args.classifier)
            list_positive = pickle_load(args.list_positive)
            outpath = args.outpath

        print('[INFO] Extracting features from {} positive images'.format(len(list_positive)))

        # Extract features from positive images
        if feature_method in ['Dense', 'Daisy', 'Daisy_DAB']:
            features = get_features(list_positive, nclusters=256, method=feature_method)

        if feature_method in ['VGG16', 'Xception']:
            features = get_features_CNN(list_positive, model=feature_method)

        features = feature_reduction(features)

        pickle_save(features, outpath, 'features.p')

        flag = 3

    if flag < 4:
        if args.flag == 3:
            features = pickle_load(args.feat_file)
            classifiers = pickle_load(args.classifier)
            outpath = args.outpath

        param = []
        param.append(features)
        n_division = args.n_division
        k = 0

        print('[INFO] Dividing patches into clusters')
        print('Total of {} images to be divided in {} clusters'.format(len(features), 2**n_division))
        print()
        for i in range(n_division):
            n_level = i + 2
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

        pickle_save(classifiers, outpath, 'classifiers.p')

        # Save to csvfile
        print('[INFO] Saving csv files...')
        print()
        for x in classifiers:
            csv_cluster = '{}.csv'.format(x[0])
            classifier = x[2]
            csv_file_path_cluster = os.path.join(outpath, csv_cluster)
            csv_columns = ["Patch_number"]
            csv_columns.append('Positive')
            for i in range(args.n_division):
                csv_columns.append('Level_{}'.format(i))

            with open(csv_file_path_cluster, 'w') as csv_file:
                writer = csv.DictWriter(csv_file, csv_columns)
                writer.writeheader()
                for i in range(classifier.shape[0]):
                    row = {'Patch_number': classifier[i][0], 'Positive': classifier[i][1]}
                    for j in range(args.n_division):
                        row["Level_{}".format(j)] = classifier[i][j+2]
                    writer.writerow(row)

        csv_features = 'features.csv'
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
                im_name = os.path.basename(im)
                data = os.path.splitext(im_name)[0]
                data = data.split('-')
                row = {'Slidename': data[0], 'Number': data[1], 'X': data[3], 'Y': data[4]}
                for i in range(shape_feat[1]):
                    row['feature_{}'.format(i)] = final_feat[index][i]
                writer.writerow(row)

        # Save images to cluster
        print('[INFO] Saving images into clusters...')
        if args.save_cluster:
            for x in classifiers:
                cluster_list = ctc.get_clusterlist(x[1], x[2], n_division)
                ctc.save_cluster_folder(x[1], cluster_list, n_division)
                csv_file_cluster_list = os.path.join(x[1], 'cluster_list.csv')
                csv_columns = ["Slidename"]
                csv_columns.append('Number')
                csv_columns.append('X')
                csv_columns.append('Y')
                csv_columns.append('Cluster')
                with open(csv_file_cluster_list, 'w') as csv_file:
                    writer = csv.DictWriter(csv_file, csv_columns)
                    writer.writeheader()
                    for im in cluster_list:
                        index = final_imag_list.index(im[0])
                        im_name = os.path.basename(im[0])
                        data = os.path.splitext(im_name)[0]
                        data = data.split('-')
                        row = {'Slidename': data[0], 'Number': data[1], 'X': data[3], 'Y': data[4], 'Cluster': im[1]}
                        writer.writerow(row)
