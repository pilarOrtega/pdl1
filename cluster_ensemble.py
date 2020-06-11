import Cluster_Ensembles as CE
import glob
import os
import numpy
import argparse
from auxiliary_functions.get_clusterlist import *
from auxiliary_functions.pickle_functions import *


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
    # Cluster run is an array shape (M,N), being M number of clustering methods and N number of samples
    consensus_labels = CE.cluster_ensembles(cluster_runs, verbose = True, N_clusters_max = 16)
    clusterlist = [(images[i], consensus_labels[i]) for i in range(len(consensus_labels))]
    return clusterlist


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cooperative clustering')
    parser.add_argument('-o', '--outpath', type=str, help='path to outfolder')
    parser.add_argument('-d', '--data', default='None', nargs='+', help='Add files to compare')
    parser.add_argument('-f', '--feature_method', type=str, help='feature method')
    args = parser.parse_args()

    outpath = args.outpath
    feature_method = args.feature_method
    data = args.data

    if len(data) == 1:
        data = os.path.join(outpath, '*-{}-*.p'.format(feature_method))
        data = glob.glob(data)

    clusterlist = cooperative_cluster(data, feature_method)

    name = 'Cooperative_{}.p'.format(feature_method)
    pickle_save(clusterlist, outpath, name)
