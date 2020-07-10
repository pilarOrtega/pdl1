from patch_division import *
from detect_dab import *
from feature_extraction import *
from cluster_division import *
from show_preview import *
from save_cluster import *
from cluster_comparison import *
from auxiliary_functions.get_clusterlist import *
from auxiliary_functions.pickle_functions import *
import argparse
import statistics
from matplotlib import pyplot as plt


import time
from tqdm import tqdm
from joblib import Parallel, delayed


def compare(features, classifiers, outpath, feature_method, i):
    scores = []
    classifiers_1 = cluster_division(features, classifiers, 1, outpath, feature_method, method='BottomUp', ncluster=i)
    classifiers_2 = cluster_division(features, classifiers, 1, outpath, feature_method, method='BottomUp', ncluster=i)
    classifiers_3 = cluster_division(features, classifiers, 1, outpath, feature_method, method='BottomUp', ncluster=i)
    classifiers_4 = cluster_division(features, classifiers, 1, outpath, feature_method, method='BottomUp', ncluster=i)
    classifiers_5 = cluster_division(features, classifiers, 1, outpath, feature_method, method='BottomUp', ncluster=i)
    print()
    clusterlist1, n = extract_complete_clusterlist(classifiers_1, feature_method)
    clusterlist2, n = extract_complete_clusterlist(classifiers_2, feature_method)
    clusterlist3, n = extract_complete_clusterlist(classifiers_3, feature_method)
    clusterlist4, n = extract_complete_clusterlist(classifiers_4, feature_method)
    clusterlist5, n = extract_complete_clusterlist(classifiers_5, feature_method)


    # Get Adjusted Rand Score
    clusterlist1.sort(key=lambda c: c[0])
    clusterlist2.sort(key=lambda c: c[0])
    clusterlist3.sort(key=lambda c: c[0])
    clusterlist4.sort(key=lambda c: c[0])
    clusterlist5.sort(key=lambda c: c[0])

    labels1 = numpy.asarray([c[1] for c in clusterlist1])
    labels2 = numpy.asarray([c[1] for c in clusterlist2])
    labels3 = numpy.asarray([c[1] for c in clusterlist3])
    labels4 = numpy.asarray([c[1] for c in clusterlist4])
    labels5 = numpy.asarray([c[1] for c in clusterlist5])

    score1 = adjusted_rand_score(labels1, labels2)
    score2 = adjusted_rand_score(labels1, labels3)
    score3 = adjusted_rand_score(labels1, labels4)
    score4 = adjusted_rand_score(labels2, labels3)
    score5 = adjusted_rand_score(labels2, labels4)
    score6 = adjusted_rand_score(labels3, labels4)
    score7 = adjusted_rand_score(labels1, labels5)
    score8 = adjusted_rand_score(labels2, labels5)
    score9 = adjusted_rand_score(labels3, labels5)
    score10 = adjusted_rand_score(labels4, labels5)

    scores = [score1, score2, score3, score4, score5, score6, score7, score8, score9, score10]
    score = sum(scores)/len(scores)
    result = []
    std = statistics.stdev(scores)
    result.extend((i, score, std))
    return result

def best_kmeans(features, classifiers, outpath, feature_method, min=1, max=50, step=5):
    outpath_temp = os.path.join(outpath, 'temp')
    os.mkdir(outpath_temp)
    classifiers = pickle_load(classifiers)
    features = pickle_load(features)
    scores = Parallel(n_jobs=-10)(delayed(compare)(features, classifiers, outpath_temp, feature_method, i) for i in tqdm(range(min, max, step)))
    shutil.rmtree(outpath_temp)
    scores = numpy.array(scores)
    pickle_save(scores, outpath, 'Scores_{}.p'.format(feature_method))
    fig, axes = plt.subplots(1, 1)
    x_values = [s[0] for s in scores]
    y_values = [s[1] for s in scores]
    axes.plot(x_values, y_values)
    plt.show()
    name = os.path.join(outpath, 'Scores_{}.png'.format(feature_method))
    fig.savefig(name, bbox_inches='tight', dpi=fig.dpi)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script to obtain the best k for each feature extractor')
    parser.add_argument('-f', '--features', type=str, help='path to features file')
    parser.add_argument('-c', '--classifiers', type=str, help='path to classifier file')
    parser.add_argument('-o', '--outpath', type=str, help='name of the out file (.png)')
    parser.add_argument('--feature_method', type=str)
    parser.add_argument('--min', type=int, default=1)
    parser.add_argument('--max', type=int, default=50)
    parser.add_argument('-s', '--step', type=int, default=5)

    args = parser.parse_args()

    best_kmeans(args.features, args.classifiers, args.outpath, args.feature_method, min=args.min, max=args.max, step=args.step)
