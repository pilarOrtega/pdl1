from auxiliary_functions.get_patch_features import *
from auxiliary_functions.pickle_functions import *
from auxiliary_functions.read_csv import *
from cluster_division import *
from show_preview import *
import numpy
import argparse
import os
import matplotlib.patches as mpatches
from matplotlib import gridspec
from matplotlib import pyplot as plt


def obtain_init_array(list_patches, features, model):
    init_arr = [get_patch_features(p, features).flatten() for p in list_patches]
    for x in model.cluster_centers_:
        init_arr.append(x)
    return numpy.asarray(init_arr)


def weighted_clustering(data, features, outpath, feature_method, classifiers, slide_folder, model):
    labels = read_csv(data)
    labels_dict = {}
    labels_dict[0] = 'Background'
    labels_dict[1] = 'Negative'
    label_color_dict = {}
    data = []
    for label in labels:
        labels_dict[(int(label[0])+2)] = label[2]
        label_color_dict[(int(label[0])+2)] = (float(label[3]), float(label[4]), float(label[5]))
        data.append(label[1])
    init_color_dict = './dict/color_dict.csv'
    init_color_dict = read_csv(init_color_dict)
    init_color_dict = {(int(c[0])): (float(c[1]), float(c[2]), float(c[3])) for c in init_color_dict}
    label_color_dict[0] = init_color_dict[0]
    label_color_dict[1] = init_color_dict[1]

    init_arr = obtain_init_array(data, features, model)
    features_mod = []
    for f in features:
        for c in classifiers:
            if c[1] in f[0]:
                features_mod.append(f)
    classifiers = cluster_division(features_mod, classifiers, outpath, feature_method, init=init_arr)

    preview = []
    for c in classifiers:
        preview.append(get_preview(c[0], c[2], 16, 224, slide_folder))

    unique_labels = [(label_color_dict[key], labels_dict[key]) for key in labels_dict]
    unique_labels = set(unique_labels)
    patchList = []
    for patch in unique_labels:
        data_key = mpatches.Patch(color=patch[0], label=patch[1])
        patchList.append(data_key)

    n_label_clusters = len(data)
    n_unlabel_clusters = model.cluster_centers_.shape[0]
    n_complet = n_label_clusters + n_unlabel_clusters + 2
    color_dict = {}
    for i in range(n_complet):
        if i < (n_label_clusters+2):
            color_dict[i] = label_color_dict[i]
        else:
            false_i = len(init_color_dict)-(i-n_label_clusters)
            color_dict[i] = init_color_dict[false_i]
            data_key = mpatches.Patch(color=init_color_dict[false_i], label='Cluster {}'.format(i-2))
            patchList.append(data_key)


    fig = plt.figure()
    spec = gridspec.GridSpec(ncols=1, nrows=1)
    ax0 = fig.add_subplot(spec[0])
    ax0.legend(handles=patchList, loc='center left', prop={'size': 15})
    ax0.axis("off")
    name = os.path.join(outpath, 'legend_'+feature_method+'.png')
    fig.savefig(name, bbox_inches='tight', dpi=fig.dpi)

    for im in preview:
        image = numpy.array([[color_dict[x] for x in row] for row in im[1]])
        image2 = numpy.transpose(image, (1, 0, 2))
        plt.imsave(os.path.join(outpath,'cmap_{}_{}.png'.format(im[0], feature_method)), image2)

    return classifiers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cooperative clustering')
    parser.add_argument('-o', '--outpath', type=str, help='path to outfolder')
    parser.add_argument('-d', '--data', help='CSV file with label information')
    parser.add_argument('-f', '--features', type=str, help='Feature file')
    parser.add_argument('-c', '--classifiers', type=str, help='Path to classifier file')
    parser.add_argument('-s', '--slides', type=str, help='Path to slide folder')
    parser.add_argument('-m', '--model', type=str, default='None', help='Path to the Kmeans trained model')
    args = parser.parse_args()

    outpath = args.outpath
    data = args.data
    features = pickle_load(args.features)
    feature_method = os.path.basename(args.features)
    feature_method = feature_method.split('_')[1]
    classifiers = args.classifiers
    slides = args.slides
    if args.model == 'None':
        model = os.path.join(outpath, 'model-{}-16-BottomUp.p'.format(feature_method))
    model = pickle_load(model)

    classifiers = weighted_clustering(data, features, outpath, feature_method, classifiers, slides, model)

    pickle_save(classifiers, outpath, 'class-wKmeans-{}-{}-BottomUp.p'.format(feature_method, level))
