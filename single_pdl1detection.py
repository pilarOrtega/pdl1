from patch_division import *
from detect_dab import *
from feature_extraction import *
from cluster_division import *
from show_preview import *
from save_cluster import *
from auxiliary_functions.get_clusterlist import *
from auxiliary_functions.pickle_functions import *
from auxiliary_functions.feature_list_division import *
import pickle
import time


# Manage parameters
parser = argparse.ArgumentParser(description='Script that divides a WSI in individual patches and classifies the resulting tiles in similarity groups.')
parser.add_argument('-s', '--slides', type=str,  nargs='+', help='Path(s) to slides')
parser.add_argument('-o', '--outpath', type=str, required='True', help='path to outfolder')
parser.add_argument('-f', '--feature_method', type=str, default='Dense', help='features extracted from individual patches [Default: %(default)s]')
parser.add_argument('-n', '--nclusters', type=int, default=16, help='number of clusters [Default: %(default)s]')
parser.add_argument('--pca', type=str, help='Path to pca.p file')
parser.add_argument('--scaler', type=str, help='Path to scaler file')
parser.add_argument('--f_kmeans', type=str, help='Path to feature_extraction kmeans')
parser.add_argument('--c_kmeans', type=str, help='Path to cluster_division kmeans')
parser.add_argument('--flag', type=int, default=0, help='Step [Default: %(default)s]')


args = parser.parse_args()

slides = args.slides
outpath = args.outpath
feature_method = args.feature_method
flag = args.flag
nclusters = args.nclusters
f_kmeans = pickle_load(args.f_kmeans)
pca = pickle_load(args.pca)
scaler = pickle_load(args.scaler)
c_kmeans = pickle_load(args.c_kmeans)

# Creates outpath if it doesn't exist yet
try:
    os.mkdir(outpath)
    print("Directory", outpath, "created")
    print()
except FileExistsError:
    print("Directory", outpath, "already exists")
    print()

print('[STEP 1] Patch division')
start = time.time()
slide_list = []
for slidepath in slides:
    get_patches(slidepath, outpath, level=16, tissue_ratio=0.5, size=224)
    slidename = os.path.basename(slidepath)
    outpath_slide = os.path.join(outpath, slidename)
    slide_list.append((slidename, outpath_slide))
end = time.time()
print('***** Total time patch_division {:.4f} s *****'.format(end-start))
print()

print('[STEP 2] DAB Detection')
start = time.time()
classifiers, list_positive, list_negative = detect_dab(slide_list, outpath, jobs=-1, threshold=85, level=16, tile_size=224)
end = time.time()
print('***** Total time detect_dab {:.4f} s *****'.format(end-start))
print()

print('[STEP 3.a] Feature extraction')
start = time.time()
features = Parallel(n_jobs=-2)(delayed(hof_dense)(im, f_kmeans, 256, method='DenseDAB') for im in tqdm(list_positive))
end = time.time()
print('***** Total time feature extraction {:.4f} s *****'.format(end-start))
print()

print('[STEP 3.b] Feature reduction')
start = time.time()
features_list, image_list = feature_list_division(features)
features_pca = pca.transform(features_list)
initial_features = features_list.shape[1]
pca_features = features_pca.shape[1]
print('Number of features reduced from {} to {}'.format(initial_features, pca_features))
print()
features_scaled = scaler.transform(features_pca)
features = [(image_list[i], features_scaled[i]) for i in range(len(image_list))]
end = time.time()
print('***** Total time feature reduction {:.4f} s *****'.format(end-start))
print()

print('[STEP 4] Cluster division')
start = time.time()
features, image_list = feature_list_division(features)
labels = c_kmeans.predict(features)
score = davies_bouldin_score(features, labels)
print('Davies-Bouldin Score: {}'.format(score))
slide_list = []
classifiers_0 = []
for s in classifiers:
    slide_list.append(s[0])
    n_samples = s[2].shape[0]
    n_features = s[2].shape[1] + 1
    c = numpy.zeros((n_samples, n_features))
    c[:, : - 1] = s[2]
    classifiers_0.append((s[0], s[1], c))
for im in image_list:
    index = image_list.index(im)
    image_name = os.path.basename(im)
    image_name = image_name.split('#')[1]
    number = image_name.split('-')
    number = int(number[0])
    slide_path = os.path.dirname(im)
    index_slide = slide_list.index(os.path.basename(slide_path))
    classifiers_0[index_slide][2][number][4] = labels[index]
end = time.time()
print('***** Total time cluster division {:.4f} s *****'.format(end-start))
print()

pickle_save(classifiers_0, outpath, 'class-{}-{}-{}.p'.format(feature_method, 16, 'BottomUp'))
