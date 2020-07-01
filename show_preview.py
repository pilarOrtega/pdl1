import numpy
from matplotlib import pyplot as plt
import argparse
import pickle
from openslide import OpenSlide, deepzoom
import csv
from tqdm import tqdm
from joblib import Parallel, delayed
import time
import os


def get_preview(slidename, classifier, level, size, slide_folder, n_division, method='TopDown', neg=0):
    result = []
    slidepath = os.path.join(slide_folder, slidename)
    slide = OpenSlide(slidepath)
    slide_dz = deepzoom.DeepZoomGenerator(slide, tile_size=(size - 2), overlap=1)
    tiles = slide_dz.level_tiles[level]
    preview = numpy.zeros(tiles)
    for x in classifier:
        im_x = int(x[1])
        im_y = int(x[2])
        if x[3] == neg:
            preview[im_x][im_y] = 1
        else:
            if method == 'TopDown':
                cluster = 0
                for j in range(n_division):
                    exp = n_division - j - 1
                    cluster = cluster + x[j+4] * (2**exp)
            if method == 'BottomUp':
                cluster = x[4]
            preview[im_x][im_y] = cluster + 2
    result.extend((slidename, preview))
    return result


def show_preview(classifiers, level, size, slide_folder, outpath, feature_method, n_division=0, method='TopDown', neg=0):
    start = time.time()
    previews = Parallel(n_jobs=4)(delayed(get_preview)(s[0], s[2], level, size, slide_folder, n_division, method=method, neg=neg) for s in tqdm(classifiers))
    end = time.time()
    print('Total time get previews: {:.4f} s'.format(end-start))

    for im in previews:
        slidename = '{}-{}-level{}-ts{}-{}-{}.png'.format(im[0], feature_method, level, size, n_division, method)
        name = os.path.join(outpath, slidename)
        fig = plt.figure()
        image = plt.imshow(im[1], cmap=plt.cm.get_cmap('tab20b', 18))
        plt.colorbar(image, fraction=0.046, pad=0.04)
        fig.savefig(name, bbox_inches='tight', dpi=fig.dpi)
        #plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Obtains a preview of the slide which display the clusters in different colors')
    parser.add_argument('-c', '--classifiers', type=str, help='path to classifier file')
    parser.add_argument('-o', '--outpath', type=str, help='name of the out file (.png)')
    parser.add_argument('-l', '--level', type=int, default=13, help='division level [Default: %(default)s]')
    parser.add_argument('-ts', '--tile_size', type=int, default=256, help='tile heigth and width in pixels [Default: %(default)s]')
    parser.add_argument('-s', '--slide_folder', type=str, default=0.5, help='path to slide folder')
    parser.add_argument('-n', '--n_division', type=int, default=0, help='number of divisions')
    parser.add_argument('-m', '--method', type=str, choices=['BottomUp', 'TopDown'])
    parser.add_argument('-f', '--feature_method', type=str)

    args = parser.parse_args()

    with open(args.classifiers, "rb") as f:
        classifiers = pickle.load(f)

    show_preview(classifiers, args.level, args.tile_size, args.slide_folder, args.outpath, args.feature_method, n_division=args.n_division, method=args.method)
