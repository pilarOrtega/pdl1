import pickle
from auxiliary_functions.get_clusterlist import *
import os
import numpy
from matplotlib import pyplot as plt
import argparse
import shutil


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

outpath_temp = os.path.join(outpath, 'temp')
os.mkdir(outpath_temp)
extract_complete_clusterlist(classifiers_1, ndivision, outpath_temp, feature_method_1)
extract_complete_clusterlist(classifiers_2, ndivision, outpath_temp, feature_method_2)


# Comparar clusters
grid = numpy.zeros((2**ndivision, 2**ndivision))
grid_1 = numpy.zeros((2**ndivision, 2**ndivision))
grid_2 = numpy.zeros((2**ndivision, 2**ndivision))
for i in range(2**ndivision):
    path_cluster_1 = 'cluster_{}_{}_{}.p'.format(feature_method_1, ndivision, i)
    path_cluster_1 = os.path.join(outpath_temp, path_cluster_1)
    with open(path_cluster_1, "rb") as f:
        cluster_1 = pickle.load(f)
    for j in range(2**ndivision):
        path_cluster_2 = 'cluster_{}_{}_{}.p'.format(feature_method_2, ndivision, j)
        path_cluster_2 = os.path.join(outpath_temp, path_cluster_2)
        with open(path_cluster_2, "rb") as f:
            cluster_2 = pickle.load(f)
        grid[i, j] = (len(cluster_1 & cluster_2)/len(cluster_1 | cluster_2))*100
        grid_1[i, j] = (len(cluster_1 & cluster_2)/len(cluster_1))*100
        grid_2[i, j] = (len(cluster_1 & cluster_2)/len(cluster_2))*100

shutil.rmtree(outpath_temp)

# Display grid
fig, axes = plt.subplots(1, 3, figsize=(13, 17))
ax = axes.ravel()

ax[0].imshow(grid, cmap='hot')
ax[0].set_title('To union of clusters')
ax[1].imshow(grid_1, cmap='hot')
ax[1].set_title('To cluster 1')
ax[2].imshow(grid_2, cmap='hot')
ax[2].set_title('To cluster 2')
plt.xlabel(feature_method_2)
plt.ylabel(feature_method_1)
plt.colorbar()
fig.tight_layout()
plt.show()
name = os.path.join(outpath, 'compare_{}_{}.png'.format(feature_method_1, feature_method_2))
fig.savefig(name, bbox_inches='tight', dpi=fig.dpi)
plt.close(fig)
