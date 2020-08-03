import pickle
import glob
import os
from tqdm import tqdm

def pickle_save(file, path, name):
    file_path = os.path.join(path, name)
    with open(file_path, "wb") as f:
        pickle.dump(file, f)

def get_clusterlist(outpath, classifier):
    cluster_list = []
    #print('Get cluster list from {}'.format(outpath))
    for c in classifier:
        slidename = os.path.basename(outpath)
        im = os.path.join(outpath, '{}#{}-level{}-{}-{}.jpg'.format(slidename, int(c[0]), 16, int(c[1]), int(c[2])))
        if c[3] == 0:
            continue
        cluster = c[4]
        cluster_list.append((im, cluster))

    return cluster_list

def extract_complete_clusterlist(classifier, feature_method):

    # Creamos una lista con todas las imagenes y su cluster
    print('Getting complete clusterlist for {}'.format(feature_method))
    clusterlist = []
    for c in classifier:
        clusterlist.extend(get_clusterlist(c[1], c[2]))

    nclusters = max([x[1] for x in clusterlist]) + 1

    return clusterlist, nclusters

def create_cluster_set(clusterlist, outpath, feature_method):
    nclusters = max([x[1] for x in clusterlist]) + 1
    print('nclusters: {}'.format(nclusters))
    # Creamos un set de cada cluster
    for i in range(int(nclusters)):
        cluster = {x[0] for x in clusterlist if x[1] == i}
        name = 'cluster_{}_{}.p'.format(feature_method, i)
        pickle_save(cluster, outpath, name)

    return clusterlist, nclusters
