from patch_division import *
from detect_dab import *
from feature_extraction import *
from cluster_division import *


# Manage parameters
parser = argparse.ArgumentParser(description='Script that divides a WSI in individual patches and classifies the resulting tiles in similarity groups.')
parser.add_argument('-s', '--slides', type=str, help='path to slides folder')
parser.add_argument('-o', '--outpath', type=str, required='True', help='path to outfolder')
parser.add_argument('-l', '--level', type=int, default=13,  help='division level of slide [Default: %(default)s]')
parser.add_argument('-tr', '--tissue_ratio', type=float, default=0.25, help='tissue ratio per patch [Default: %(default)s]')
parser.add_argument('-ts', '--tile_size', type=int, default=256, help='tile heigth and width in pixels [Default: %(default)s]')
parser.add_argument('-f', '--feature_method', type=str, default='Dense', help='features extracted from individual patches [Default: %(default)s]')
parser.add_argument('-n', '--n_division', type=int, default=4, help='number of divisions [Default: %(default)s]')

args = parser.parse_args()

slides = args.slides
outpath = args.outpath
n_division = args.n_division
level = args.level
tissue_ratio = args.tissue_ratio
tile_size = args.tile_size
feature_method = args.feature_method
n_division = args.n_division

slide_list = patch_division(slides, outpath, level, tile_size, tissue_ratio)
classifiers, list_positive = detect_dab(slide_list, outpath)
features = feature_extraction(list_positive, outpath, feature_method)
classifiers = cluster_division(features, classifiers, n_division, outpath, feature_method)
