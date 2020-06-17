import Cluster_Ensembles as CE
import glob
import os
import numpy
import argparse
from auxiliary_functions.get_clusterlist import *
from auxiliary_functions.pickle_functions import *
from auxiliary_functions.cl_to_class import *


def cooperative_cluster(data, feature_method):
    cluster_runs = []
    for d in data:
        classifier = pickle_load(d)
        clusterlist, n = extract_complete_clusterlist(classifier, feature_method)
        clusterlist.sort(key=lambda c: c[0])
        images = [c[0] for c in clusterlist]
        labels = [c[1] for c in clusterlist]
        cluster_runs.append(labels)

    cluster_runs = numpy.asarray(cluster_runs)
    for i in range(cluster_runs.shape[0]):
        for j in range(cluster_runs.shape[1]):
            cluster_runs[i, j] = int(cluster_runs[i, j])
    # Cluster run is an array shape (M,N), being M number of clustering methods and N number of samples
    cluster_runs = cluster_runs[:, :230000]
    consensus_labels = CE.cluster_ensembles(cluster_runs, verbose = True, N_clusters_max = 16)
    clusterlist = [(images[i], consensus_labels[i]) for i in range(len(consensus_labels))]
    return clusterlist


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cooperative clustering')
    parser.add_argument('-o', '--outpath', type=str, help='path to outfolder')
    parser.add_argument('-d', '--data', default='None', nargs='+', help='Add files to compare')
    parser.add_argument('-f', '--feature_method', type=str, help='feature method')
    parser.add_argument('-c', '--classifier', default='None')
    args = parser.parse_args()

    outpath = args.outpath
    feature_method = args.feature_method
    data = args.data
    classifier = args.classifier

    if data == 'None':
        data = os.path.join(outpath, '*-{}-*.p'.format(feature_method))
        data = glob.glob(data)

    if classifier == 'None':
        classifier = os.path.join(outpath, 'class_16_224.p')

    classifier = pickle_load(classifier)

    clusterlist = cooperative_cluster(data, feature_method)
    class_cooperative = cl_to_class(clusterlist, classifier)

    name = 'Cooperative_{}.p'.format(feature_method)
    pickle_save(clusterlist, outpath, name)

    name = 'Class_cooperative_{}.p'.format(feature_method)
    pickle_save(class_cooperative, outpath, name)
