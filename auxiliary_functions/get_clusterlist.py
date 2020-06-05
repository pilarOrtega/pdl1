import pickle
import glob
import os
from tqdm import tqdm

def pickle_save(file, path, name):
    file_path = os.path.join(path, name)
    with open(file_path, "wb") as f:
        pickle.dump(file, f)

def get_clusterlist(outpath, classifier, n_division):
    cluster_list = []
    image_list = glob.glob(os.path.join(outpath, '*.jpg'))
    #print('Get cluster list from {}'.format(outpath))
    for im in image_list:
        image_name = os.path.basename(im)
        number = image_name.split('#')[1]
        number = number.split('-')
        number = int(number[0])

        if classifier[number][3] == 0:
            continue

        cluster = 0
        if n_division == 1:
            cluster = classifier[number][4]
        else:
            for j in range(n_division):
                exp = n_division - j - 1
                cluster = cluster + classifier[number][j+4] * (2**exp)
        cluster_list.append((image_name, cluster))

    return cluster_list

def extract_complete_clusterlist(classifier, outpath, feature_method):

    # Creamos una lista con todas las imagenes y su cluster
    print('Getting complete clusterlist for {}'.format(feature_method))
    clusterlist = []
    for c in classifier:
        ndivision = (c[2].shape[1]) - 4
        clusterlist.extend(get_clusterlist(c[1], c[2], ndivision))

    nclusters = max([x[1] for x in clusterlist]) + 1
    print('nclusters: {}'.format(nclusters))
    # Creamos un set de cada cluster
    for i in range(int(nclusters)):
        cluster = {x[0] for x in clusterlist if x[1] == i}
        name = 'cluster_{}_{}.p'.format(feature_method, i)
        pickle_save(cluster, outpath, name)

    return clusterlist, nclusters
