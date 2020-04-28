import os
import tqdm
import glob
from skimage.io import sift, imread, imsave

def save_cluster_folder(outpath, classifier, n_division):
    clusters = os.path.join(outpath, 'clusters')
    outpath_images = os.path.join(outpath, '*.jpg')
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

    cluster_list = []
    print()
    print('Saving images into clusters...')
    for im in tqdm(glob.glob(outpath_images)):
        image_name = os.path.basename(im)
        number = image_name.split('-')
        number = int(number[0])

        if classifier[number][1] == 0:
            continue

        cluster = 0
        for j in range(n_division):
            exp = n_division - j - 1
            cluster = cluster + classifier[number][j+2] * (2**exp)
        cluster_list.append((im, cluster))

    for x in cluster_list:
        index = x[1]
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
