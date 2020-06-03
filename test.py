from patch_division import *
from detect_dab import *
from feature_extraction import *
from cluster_division import *
from show_preview import *
from save_cluster import *
import pickle
import time


def pickle_load(file_name):
    with open(file_name, "rb") as f:
        file = pickle.load(f)
        print('Document ' + file_name + ' correctly loaded')
        print()
    return file


# Manage parameters
parser = argparse.ArgumentParser(description='Script that divides a WSI in individual patches and classifies the resulting tiles in similarity groups.')
parser.add_argument('-s', '--slides', type=str, help='path to slides folder')
parser.add_argument('-o', '--outpath', type=str, required='True', help='path to outfolder')
parser.add_argument('-l', '--level', type=int, default=16,  help='division level of slide [Default: %(default)s]')
parser.add_argument('-tr', '--tissue_ratio', type=float, default=0.5, help='tissue ratio per patch [Default: %(default)s]')
parser.add_argument('-ts', '--tile_size', type=int, default=224, help='tile heigth and width in pixels [Default: %(default)s]')
parser.add_argument('-f', '--feature_method', type=str, default='Dense', help='features extracted from individual patches [Default: %(default)s]')
parser.add_argument('-n', '--n_division', type=int, default=4, help='number of divisions [Default: %(default)s]')
parser.add_argument('-m', '--method', type=str, choices=['Bottom-up', 'Top-down'])
parser.add_argument('--flag', type=int, default=0, help='Step [Default: %(default)s]')
parser.add_argument('-d', '--device', default="0", help='GPU used (0 or 1) [Default: %(default)s]')
parser.add_argument('-j', '--jobs', type=int)
parser.add_argument('-b', '--features_batch', action='store_true')


args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

slides = args.slides
outpath = args.outpath
n_division = args.n_division
level = args.level
tissue_ratio = args.tissue_ratio
tile_size = args.tile_size
feature_method = args.feature_method
n_division = args.n_division
jobs = args.jobs
features_batch = args.features_batch
flag = args.flag
method = args.method
if method == 'Bottom-up':
    n_division = 1
# Flag is an argument that determines in which step start the execution. It is

if flag == 0:
    start = time.time()
    slide_list = patch_division(slides, outpath, level, tile_size=tile_size, tissue_ratio=tissue_ratio, jobs=jobs)
    end = time.time()
    print('***** Total time patch_division {:.4f} s *****'.format(end-start))
    print()

if flag <= 1:
    if flag == 1:
        slide_list = os.path.join(outpath, 'list_{}_{}.p'.format(level, tile_size))
        slide_list = pickle_load(slide_list)
    start = time.time()
    classifiers, list_positive = detect_dab(slide_list, outpath, jobs=jobs, threshold=85)
    end = time.time()
    print('***** Total time detect_dab {:.4f} s *****'.format(end-start))
    print()

if flag <= 2:
    if flag == 2:
        list_positive = os.path.join(outpath, 'list_positive_{}_{}.p'.format(level, tile_size))
        list_positive = pickle_load(list_positive)
        classifiers = os.path.join(outpath, 'class_{}_{}.p'.format(level, tile_size))
        classifiers = pickle_load(classifiers)
    start = time.time()
    if features_batch:
        features = feature_extraction_batch(list_positive, outpath, feature_method)
    features = feature_extraction(list_positive, outpath, feature_method)
    end = time.time()
    print('***** Total time feature_extraction {:.4f} s *****'.format(end-start))
    print()

if flag <= 3:
    if flag == 3:
        classifiers = os.path.join(outpath, 'class_{}_{}.p'.format(level, tile_size))
        classifiers = pickle_load(classifiers)
        features = os.path.join(outpath, 'features_{}_level{}.p'.format(feature_method, level))
        features = pickle_load(features)
    start = time.time()
    classifiers = cluster_division(features, classifiers, n_division, outpath, feature_method, method = method)
    end = time.time()
    print('***** Total time cluster_division {:.4f} s *****'.format(end-start))
    print()

if flag <= 4:
    if flag == 4:
        classifiers = os.path.join(outpath, 'class-{}-{}-{}.p'.format(feature_method, level, method))
        classifiers = pickle_load(classifiers)
    outpath = os.path.join(outpath, 'Results_{}_{}'.format(feature_method, method))
    try:
        os.mkdir(outpath)
        print("Directory", outpath, "created")
    except FileExistsError:
        print("Directory", outpath, "already exists")
    show_preview(classifiers, level, tile_size, slides, outpath, feature_method, n_division, method=method)
    save_cluster(classifiers, outpath, feature_method)
