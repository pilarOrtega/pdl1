import os
from tqdm import tqdm
import glob
import csv
import numpy
import argparse
from skimage.io import sift, imread, imsave
from show_random_imgs import *
import pickle
from joblib import Parallel, delayed
import time
from auxiliary_functions.get_clusterlist import *


def save_cluster_folder(outpath, cluster_list, feature_method):
    clusters = os.path.join(outpath, 'clusters_{}'.format(feature_method))
    try:
        os.mkdir(clusters)
        print("Directory", clusters, "created")
    except FileExistsError:
        print("Directory", clusters, "already exists")

    dir_list = []
    nclusters = max([x[1] for x in cluster_list]) + 1
    for i in range(int(nclusters)):
        dir = os.path.join(clusters, '{}_cluster_{}'.format(feature_method, i))
        try:
            os.mkdir(dir)
            print('Directory', dir, 'created')
        except FileExistsError:
            print('Directory', dir, 'already exists')
        dir_list.append(dir)

    print()
    print('Saving images into clusters...')

    for x in tqdm(cluster_list):
        index = int(x[1])
        image_name = os.path.basename(x[0])
        image = imread(x[0])
        image_path = os.path.join(dir_list[index], image_name)
        imsave(image_path, image)


def class_to_cluster(classifier, level, cluster):
    binary_format = '0{}b'.format(level)
    cluster_bin = format(cluster, binary_format)
    list_images_cluster = []
    for i in range(classifier.shape[0]):
        flag = True
        if classifier[i][1] == 0:
            continue
        for j in range(level):
            if (classifier[i][j+2] != int(cluster_bin[j])):
                flag = False
                break
        if flag:
            list_images_cluster.append(classifier[i][0])

    return list_images_cluster


def read_csv(file_path):
    with open(file_path, newline='') as file:
        reader = csv.reader(file, delimiter=',')
        list = []
        for row in reader:
            list.append(row)
        return list


def list_to_array(list):
    list_array = numpy.zeros((len(list)-1, len(list[0])))
    for i in range(list_array.shape[0]):
        for j in range(list_array.shape[1]):
            list_array[i][j] = list[i+1][j]
    list_array = list_array.astype(int)
    return list_array


def save_cluster(cluster_list, outpath, feature_method, x=4, y=8, figsize=(13, 7), save_folder=False):

    list_cluster = []
    nclusters = max([x[1] for x in cluster_list]) + 1
    for i in range(int(nclusters)):
        list_images = []
        for im in cluster_list:
            if im[1] == i:
                list_images.append(im[0])
        cluster_name = os.path.join(outpath, '{}_cluster_{}_ndivision_{}.png'.format(feature_method, i, n_division))
        list_cluster.append((cluster_name, list_images))

    start = time.time()
    Parallel(n_jobs=1)(delayed(show_random_imgs)(l[1], x, y, figsize, save_fig=True, name=l[0]) for l in tqdm(list_cluster))
    end = time.time()
    print('Total time get images: {:.4f} s'.format(end-start))

    if save_folder:
        print('Saving images from slide ' + c[1])
        print()
        save_cluster_folder(outpath, cluster_list, feature_method)
        csv_file_cluster_list = os.path.join(c[1], 'cluster_list_{}.csv'.format(feature_method))
        csv_columns = ["Slidename"]
        csv_columns.append('Number')
        csv_columns.append('X')
        csv_columns.append('Y')
        csv_columns.append('Cluster')
        with open(csv_file_cluster_list, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, csv_columns)
            writer.writeheader()
            for im in cluster_list:
                im_name = os.path.basename(im[0])
                data = os.path.splitext(im_name)[0]
                slidename = data.split('#')[0]
                data = data.split('#')[1]
                data = data.split('-')
                row = {'Slidename': slidename, 'Number': data[0], 'X': data[2], 'Y': data[3]}
                writer.writerow(row)

    return cluster_list

###############################################################################
#
# MAIN
#
###############################################################################


if __name__ == "__main__":

    # Manage parameters
    parser = argparse.ArgumentParser(description='Reads data from csv file and saves patches in corresponding clusters')
    parser.add_argument('-c', '--cluster_list', type=str, help='path to cluster list file')
    parser.add_argument('-o', '--outpath', type=str, help='path to outfolder')
    parser.add_argument('-s', '--save_folder', action='store_true', help='saves patches in cluster folders')
    args = parser.parse_args()

    classifiers = args.classifiers
    outpath = args.outpath
    save_folder = args.save_folder
    feature_method = os.path.basename(classifiers)
    feature_method = os.path.splitext(classifiers)[0]
    feature_method = feature_method.split('-')[1]
    with open(classifiers, "rb") as f:
        classifiers = pickle.load(f)

    if save_folder:
        cluster_list = save_cluster(classifiers, outpath, feature_method, save_folder=save_folder)
    else:
        cluster_list = save_cluster(classifiers, outpath, feature_method)
