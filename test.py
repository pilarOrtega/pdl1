from patch_division import *
from detect_dab import *
from feature_extraction import *
from cluster_division import *
from show_preview import *
from save_cluster import *
from auxiliary_functions.get_clusterlist import *
from auxiliary_functions.pickle_functions import *
import time


# Manage parameters
parser = argparse.ArgumentParser(
    description='Script that divides a WSI in individual patches and classifies the resulting tiles in similarity groups.')
parser.add_argument('-s', '--slides', type=str, help='Path to slides folder')
parser.add_argument('-o', '--outpath', type=str,
                    required='True', help='Path to outfolder')
parser.add_argument('-l', '--level', type=int, default=16,
                    help='Division level of slide [Default: %(default)s]')
parser.add_argument('-tr', '--tissue_ratio', type=float, default=0.5,
                    help='Tissue ratio per patch [Default: %(default)s]')
parser.add_argument('-ts', '--tile_size', type=int, default=256,
                    help='Tile heigth and width in pixels [Default: %(default)s]')
parser.add_argument('-f', '--feature_method', type=str, default='Dense',
                    help='Features extracted from individual patches [Default: %(default)s]')
parser.add_argument('-c', '--nclusters', type=int, default=16,
                    help='Number of clusters [Default: %(default)s]')
parser.add_argument('--flag', type=int, default=0,
                    help='Step [Default: %(default)s]')
parser.add_argument('-d', '--device', default="0",
                    help='GPU used (0 or 1) [Default: %(default)s]')
parser.add_argument('-j', '--jobs', type=int,
                    help='Number of simultaneous running processes [Default: %(default)s]')
parser.add_argument('--pca', default=0.9,
                    help='PCA level [Default: %(default)s]')
parser.add_argument('-n', '--number', type=int, default=0,
                    help='Number of experience - results will be saved as [n]_Results_[feature_method]_[nclusters]')
parser.add_argument('-b', '--features_batch', action='store_true',
                    help='Option for batch feature processing')
parser.add_argument('--da', action='store_true',
                    help='Option for domain adaptation in CNN feature extraction')
parser.add_argument('--NoDAB', type=int, action='store_true',
                    help='Prevents DAB filtering of patches')


args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

slides = args.slides
outpath = args.outpath
level = args.level
tissue_ratio = args.tissue_ratio
tile_size = args.tile_size
feature_method = args.feature_method
jobs = args.jobs
features_batch = args.features_batch
flag = args.flag
nclusters = args.nclusters
da = args.da
pca_level = args.pca
number = args.number
threshold = args.threshold
NoDAB = args.NoDAB

# Flag is an argument that determines in which step start the execution. It is

if flag == 0:
    start = time.time()
    classifier = patch_division(
        slides, outpath, level, tile_size=tile_size, tissue_ratio=tissue_ratio, jobs=jobs)
    end = time.time()
    print('***** Total time patch_division {:.4f} s *****'.format(end - start))
    print()

if flag <= 1:
    if flag == 1:
        classifier = os.path.join(
            outpath, 'class_{}_{}.p'.format(level, tile_size))
        classifier = pickle_load(classifier)
    if not NoDAB:
        start = time.time()
        classifiers = detect_dab(
            classifier, outpath, jobs=jobs, threshold=threshold)
        end = time.time()
        print('***** Total time detect_dab {:.4f} s *****'.format(end - start))
        print()

if flag <= 2:
    if flag == 2:
        classifiers = os.path.join(
            outpath, 'class_{}_{}.p'.format(level, tile_size))
        classifiers = pickle_load(classifiers)
    start = time.time()
    if features_batch:
        features = feature_extraction_batch(
            list_positive, outpath, feature_method)
    features = feature_extraction(
        classifier, outpath, feature_method, da=da, pca_level=pca_level)
    end = time.time()
    print(
        '***** Total time feature_extraction {:.4f} s *****'.format(end - start))
    print()

if flag <= 3:
    if flag == 3:
        classifiers = os.path.join(
            outpath, 'class_{}_{}.p'.format(level, tile_size))
        classifiers = pickle_load(classifiers)
        features = os.path.join(
            outpath, 'features_{}_level{}.p'.format(feature_method, level))
        features = pickle_load(features)
    outpath = os.path.join(outpath, '{}_Results_{}_{}'.format(
        number, feature_method, nclusters))
    try:
        os.mkdir(outpath)
        print("Directory", outpath, "created")
    except FileExistsError:
        print("Directory", outpath, "already exists")
    start = time.time()
    classifiers = cluster_division(
        features, classifiers, outpath, feature_method, slides, ncluster=nclusters)
    end = time.time()
    print(
        '***** Total time cluster_division {:.4f} s *****'.format(end - start))
    print()

if flag <= 4:
    if flag == 4:
        classifiers = os.path.join(
            outpath, 'class-{}-{}-Final.p'.format(feature_method, nclusters))
        classifiers = pickle_load(classifiers)
    show_preview(classifiers, level, tile_size,
                 slides, outpath, feature_method)
    cluster_list, n = extract_complete_clusterlist(classifiers, feature_method, level)
    save_cluster(cluster_list, outpath, feature_method)
