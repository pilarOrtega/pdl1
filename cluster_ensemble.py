import Cluster_Ensembles as CE
import glob
import os
import numpy
import argparse
from auxiliary_functions.get_clusterlist import *
from auxiliary_functions.pickle_functions import *


def cooperative_cluster(outpath, feature_method):
    outpath = '/data/DeepLearning/ABREU_Arnaud/Pilar_stage/tests/level_16_ts_224_V2_final/'
    data = os.path.join(outpath, '*-{}-*.p'.format(feature_method))
    data = glob.glob(data)
    M = len(data)
    outpath_temp = os.path.join(outpath, 'temp')
    try:
        os.mkdir(outpath_temp)
        print("Directory", outpath, "created")
    except FileExistsError:
        print("Directory", outpath, "already exists")

    cluster_runs = []
    for d in data:
        classifier = pickle_load(d)
        clusterlist, n = extract_complete_clusterlist(classifier, outpath_temp, feature_method)
        clusterlist.sort(key=lambda c: c[0])
        images = [c[0] for c in clusterlist]
        labels = [c[1] for c in clusterlist]
        cluster_runs.append(labels)

    cluster_runs = numpy.asarray(cluster_runs)
    # Cluster run is an array shape (M,N), being M number of clustering methods and N number of samples
    consensus_labels = CE.cluster_ensembles(cluster_runs, verbose = True, N_clusters_max = 50)
    clusterlist = [(images[i], consensus_labels[i]) for i in range(len(consensus_labels))]

    name = 'Cooperative_{}.p'.format(feature_method)

    pickle_save(clusterlist, outpath, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cooperative clustering')
    parser.add_argument('-o', '--outpath', type=str, help='path to outfolder')
    parser.add_argument('-f', '--feature_method', type=str, help='feature method')
    args = parser.parse_args()

    outpath = args.outpath
    feature_method = args.feature_method

    cooperative_cluster(outpath, feature_method)
