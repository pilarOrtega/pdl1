from auxiliary_functions.get_patch_features import *
from auxiliary_functions.pickle_functions import *
import numpy
import argparse
import os

def obtain_init_array(list_patches, features):
    init_arr = [get_patch_features(p, features) for p in list_patches]
    return numpy.asarray(init_arr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cooperative clustering')
    parser.add_argument('-o', '--outpath', type=str, help='path to outfolder')
    parser.add_argument('-d', '--data', default='None', nargs='+', help='Add files to compare')
    parser.add_argument('-f', '--features', type=str, help='Feature file')
    args = parser.parse_args()

    outpath = args.outpath
    data = args.data
    features = pickle_load(args.features)

    init_arr = obtain_init_array(data, features)

    feature_method = os.path.basename(args.features)
    feature_method = feature_method.split('_')[1]
    pickle_save(init_arr, outpath, 'init_array_{}.p'.format(feature_method))
