import pickle
from auxiliary_functions.get_clusterlist import *
import os
import numpy
from matplotlib import pyplot as plt
import argparse


parser = argparse.ArgumentParser(description='Compares two classifiers between them')
parser.add_argument('-c1', '--classifier_1', type=str)
parser.add_argument('-c2', '--classifier_2', type=str)
parser.add_argument('-n', '--ndivision', type=int)
parser.add_argument('-o', '--outpath', type=str, help='path to outfolder')
args = parser.parse_args()
# Function that gets the intersection between two clusters from different
# feature methods

classifiers_1 = args.classifier_1
classifiers_2 = args.classifier_2
ndivision = args.ndivision
outpath = args.outpath

feature_method_1 = os.path.basename(classifiers_1)
feature_method_1 = feature_method_1.split('-')[1]
feature_method_1 = feature_method_1 + '(1)'

feature_method_2 = os.path.basename(classifiers_2)
feature_method_2 = feature_method_2.split('-')[1]
feature_method_2 = feature_method_2 + '(2)'

with open(classifiers_1, "rb") as f:
    classifiers_1 = pickle.load(f)

with open(classifiers_2, "rb") as f:
    classifiers_2 = pickle.load(f)


extract_complete_clusterlist(classifiers_1, ndivision, outpath, feature_method_1)
extract_complete_clusterlist(classifiers_2, ndivision, outpath, feature_method_2)


# Comparar clusters
grid = numpy.zeros((2**ndivision, 2**ndivision))
for i in range(2**ndivision):
    path_cluster_1 = 'cluster_{}_{}_{}.p'.format(feature_method_1, ndivision, i)
    path_cluster_1 = os.path.join(outpath, path_cluster_1)
    with open(path_cluster_1, "rb") as f:
        cluster_1 = pickle.load(f)
    for j in range(2**ndivision):
        path_cluster_2 = 'cluster_{}_{}_{}.p'.format(feature_method_2, ndivision, j)
        path_cluster_2 = os.path.join(outpath, path_cluster_2)
        with open(path_cluster_2, "rb") as f:
            cluster_2 = pickle.load(f)
        grid[i, j] = (len(cluster_1 & cluster_2)/len(cluster_1 | cluster_2))*100
        os.remove(path_cluster_2)
    os.remove(path_cluster_1)


# Display grid
fig = plt.figure()
plt.imshow(grid, cmap='hot')
plt.colorbar()
plt.xlabel(feature_method_2)
plt.ylabel(feature_method_1)
fig.tight_layout()
plt.show()
name = os.path.join(outpath, 'compare_{}_{}.png'.format(feature_method_1, feature_method_2))
fig.savefig(name, bbox_inches='tight', dpi=fig.dpi)
plt.close(fig)
