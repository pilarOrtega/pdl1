import os
from tqdm import tqdm
import glob
import csv
import numpy
import argparse
from skimage.io import sift, imread, imsave
import show_random_imgs as sri


def save_cluster_folder(outpath, cluster_list, n_division):
    clusters = os.path.join(outpath, 'clusters')
    try:
        os.mkdir(clusters)
        print("Directory", clusters, "created")
    except FileExistsError:
        print("Directory", clusters, "already exists")

    dir_list = []
    for i in range(2**n_division):
        dir = os.path.join(clusters, '{}'.format(i))
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


def get_clusterlist(outpath, classifier, n_division):
    cluster_list = []
    image_list = glob.glob(os.path.join(outpath, '*.jpg'))
    for im in tqdm(image_list):
        image_name = os.path.basename(im)
        number = image_name.split('-')
        number = int(number[1])

        if classifier[number][1] == 0:
            continue

        cluster = 0
        for j in range(n_division):
            exp = n_division - j - 1
            cluster = cluster + classifier[number][j+2] * (2**exp)
        cluster_list.append((im, cluster))

    return cluster_list


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

###############################################################################
#
# MAIN
#
###############################################################################


if __name__ == "__main__":

    # Manage parameters
    parser = argparse.ArgumentParser(description='Reads data from csv file and saves patches in corresponding clusters')
    parser.add_argument('-c', '--csv_files', type=str, nargs='+', help='path(s) to csv file(s)')
    parser.add_argument('-s', '--save_cluster', action='store_true', default=False, help='saves a copy of the patches in cluster folders')
    parser.add_argument('-v', '--show_images', action='store_true', default=False, help='show random set of images of each cluster')
    args = parser.parse_args()

    classifiers = []
    csv_files = args.csv_files
    for file in csv_files:
        print('Reading file '+file)
        list_file = read_csv(file)
        path = os.path.dirname(file)
        slide = os.path.basename(file)
        slide = os.path.splitext(slide)[0]
        path = os.path.join(path, slide)
        classifiers.append((slide, path, list_to_array(list_file)))

    for x in classifiers:
        n_division = (x[2].shape[1]) - 2
        cluster_list = get_clusterlist(x[1], x[2], n_division)
        if args.save_cluster:
            print('Saving images from slide ' + x[1])
            print()
            save_cluster_folder(x[1], cluster_list, n_division)
        if args.show_images:
            print('Showing sample images from slide ' + x[1])
            print()
            for i in range(2**n_division):
                image_list = []
                print('Cluster {}'.format(i))
                for image in cluster_list:
                    if image[1] == i:
                        image_list.append(image[0])
                sri.show_random_imgs(image_list, 2, 8, (7,6))
