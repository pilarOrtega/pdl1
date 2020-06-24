from auxiliary_functions.get_patch_features import *
from auxiliary_functions.pickle_functions import *
from auxiliary_functions.read_csv import *
from cluster_division import *
from show_preview import *
import numpy
import argparse
import os

def obtain_init_array(list_patches, features):
    init_arr = [get_patch_features(p, features).flatten() for p in list_patches]
    return numpy.asarray(init_arr)

def weighted_clustering(data, features, outpath, feature_method, classifiers, slide_folder):
    labels = read_csv(data)
    labels_dict = { int(label[0]) : label[2]  for label in labels }
    color_dict = { int(label[0]) : (float(label[3]), float(label[4]), float(label[5])) for label in labels}
    data = [label[1] for label in labels]

    init_arr = obtain_init_array(data, features)
    features_mod = []
    for f in features:
        for c in classifiers:
            if c[1] in f[0]:
                features_mod.append(f)
    classifiers = cluster_division(features_mod, classifiers, 4, outpath, feature_method, method='BottomUp', init=init_arr)

    preview = []
    for c in classifiers:
        preview.append(get_preview(c[0], c[2], 16, 224, slide_folder, 4, method='BottomUp'))

    for im in preview:
        image = numpy.array([[color_dict[x] for x in row] for row in im[1]])
        patchList = []
        for key in color_dict:
            data_key = mpatches.Patch(color=color_dict[key], label=labels_dict[key])
            patchList.append(data_key)

        fig = plt.figure(figsize=(13,4))
        spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[2, 1])
        ax0 = fig.add_subplot(spec[0])
        ax1 = fig.add_subplot(spec[1])

        ax0.imshow(image)
        ax1.legend(handles=patchList, loc='center left', prop={'size': 15})
        ax1.axis("off")
        title = im[0] + ' ' + feature_method
        fig.suptitle(title, va='baseline', fontsize = 15)
        fig.tight_layout()
        plt.show()
        name = os.path.join(outpath, im[0]+feature_method+'.png')
        fig.savefig(name, bbox_inches='tight', dpi=fig.dpi)

    return classifiers

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cooperative clustering')
    parser.add_argument('-o', '--outpath', type=str, help='path to outfolder')
    parser.add_argument('-d', '--data', help='CSV file with label information')
    parser.add_argument('-f', '--features', type=str, help='Feature file')
    parser.add_argument('-c', '--classifiers', type=str, help='Path to classifier file')
    parser.add_arguement('-s', '--slides', type=str, help='Path to slide folder')
    args = parser.parse_args()

    outpath = args.outpath
    data = args.data
    features = pickle_load(args.features)
    feature_method = os.path.basename(args.features)
    feature_method = feature_method.split('_')[1]
    classifiers = args.classifiers
    slides = args.slides

    classifiers = weighted_clustering(data, features, outpath, feature_method, classifiers, slides)

    pickle_save(classifiers, outpath, 'class-wKmeans-{}-{}-{}.p'.format(feature_method, level, method))
