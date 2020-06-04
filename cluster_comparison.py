import pickle
from auxiliary_functions.get_clusterlist import *
import os
import numpy
from matplotlib import pyplot as plt
import argparse
import shutil
from sklearn.metrics.cluster import adjusted_rand_score
from prettytable import PrettyTable


parser = argparse.ArgumentParser(description='Compares two classifiers between them')
parser.add_argument('-c1', '--classifier_1', type=str)
parser.add_argument('-c2', '--classifier_2', type=str)
parser.add_argument('-o', '--outpath', type=str, help='path to outfolder')
args = parser.parse_args()
# Function that gets the intersection between two clusters from different
# feature methods

classifiers_1 = args.classifier_1
classifiers_2 = args.classifier_2
outpath = args.outpath

feature_method_1 = os.path.basename(classifiers_1)
feature_method_1 = feature_method_1.split('-')
feature_method_1 = feature_method_1[1] + feature_method_1[2] + '(1)'

feature_method_2 = os.path.basename(classifiers_2)
feature_method_2 = feature_method_2.split('-')
feature_method_2 = feature_method_2[1] + feature_method_2[2] + '(2)'

with open(classifiers_1, "rb") as f:
    classifiers_1 = pickle.load(f)

with open(classifiers_2, "rb") as f:
    classifiers_2 = pickle.load(f)

outpath_temp = os.path.join(outpath, 'temp')
os.mkdir(outpath_temp)
clusterlist1, n1 = extract_complete_clusterlist(classifiers_1, outpath_temp, feature_method_1)
clusterlist2, n2 = extract_complete_clusterlist(classifiers_2, outpath_temp, feature_method_2)
n1 = int(n1)
n2 = int(n2)

# Get Adjusted Rand Score
clusterlist1.sort(key=lambda c: c[0])
clusterlist2.sort(key=lambda c: c[0])
labels1 = numpy.asarray([c[1] for c in clusterlist1])
labels2 = numpy.asarray([c[1] for c in clusterlist2])
score = adjusted_rand_score(labels1, labels2)
print('Adjusted rand score: {}'.format(score))

# Comparar clusters
grid = numpy.zeros((n1, n2))
grid_1 = numpy.zeros((n1, n2))
grid_2 = numpy.zeros((n1, n2))
for i in range(n1):
    path_cluster_1 = 'cluster_{}_{}.p'.format(feature_method_1, i)
    path_cluster_1 = os.path.join(outpath_temp, path_cluster_1)
    with open(path_cluster_1, "rb") as f:
        cluster_1 = pickle.load(f)
    for j in range(n2):
        path_cluster_2 = 'cluster_{}_{}.p'.format(feature_method_2, j)
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

im0 = ax[0].imshow(grid, cmap='hot')
ax[0].set_title('To union of clusters')
plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
im1 = ax[1].imshow(grid_1, cmap='hot')
ax[1].set_title('To cluster 1')
plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
im2 = ax[2].imshow(grid_2, cmap='hot')
ax[2].set_title('To cluster 2')
plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
for i in range(3):
    ax[i].set_xlabel(feature_method_2)
    ax[i].set_ylabel(feature_method_1)
fig.suptitle('Score {}'.format(score), va = 'baseline')
fig.tight_layout()
plt.show()
name = os.path.join(outpath, 'compare_{}_{}.png'.format(feature_method_1, feature_method_2))
fig.savefig(name, bbox_inches='tight', dpi=fig.dpi)
plt.close(fig)

pairs_grid = [(i, grid[i].argmax(),  max(grid[i])) for i in range(grid.shape[0])]
pairs_grid1 = [(i, grid_1[i].argmax(),  max(grid_1[i])) for i in range(grid_1.shape[0])]
pairs_grid2 = [(i, grid_2[:,i].argmax(),  max(grid_2[:,i])) for i in range(grid_2.shape[1])]

print('*** View max values ***')
print('To union of clusters:')
t = PrettyTable([feature_method_1, feature_method_2, Percentage])
for i in range(len(pairs_grid)):
    t.add_row([pairs_grid[i][0], pairs_grid[i][1], pairs_grid[i][2]])
print(t)
print()
print('To cluster {}:'.format(feature_method_1))
t = PrettyTable([feature_method_1, feature_method_2, Percentage])
for i in range(len(pairs_grid)):
    t.add_row([pairs_grid1[i][0], pairs_grid1[i][1], pairs_grid1[i][2]])
print(t)
print()
print('To cluster {}:'.format(feature_method_2))
t = PrettyTable([feature_method_2, feature_method_3, Percentage])
for i in range(len(pairs_grid)):
    t.add_row([pairs_grid2[i][0], pairs_grid2[i][1], pairs_grid2[i][2]])
print(t)
