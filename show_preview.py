import numpy
from matplotlib import pyplot as plt
import argparse
import pickle
import csv
from tqdm import tqdm
from joblib import Parallel, delayed
import time
import os
from auxiliary_functions.read_csv import *
from auxiliary_functions.get_preview import *


def show_preview(classifiers, level, size, slide_folder, outpath, feature_method, neg=0):
    start = time.time()
    previews = Parallel(n_jobs=4)(delayed(get_preview)(s[0], s[2], level, size, slide_folder, neg=neg) for s in tqdm(classifiers))
    end = time.time()
    print('Total time get previews: {:.4f} s'.format(end-start))
    colordict = './dict/color_dict.csv'
    colordict = read_csv(colordict)
    colordict = {(int(c[0])): (float(c[1]), float(c[2]), float(c[3])) for c in colordict}

    for im in previews:
        image = numpy.array([[colordict[x] for x in row] for row in im[1]])
        image2 = numpy.transpose(image, (1, 0, 2))
        plt.imsave(os.path.join(outpath,'cmap_{}_{}.png'.format(im[0], feature_method)), image2)
        slidename = '{}-{}-level{}-ts{}-BottomUp.png'.format(im[0], feature_method, level, size)
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
    parser.add_argument('-f', '--feature_method', type=str)

    args = parser.parse_args()

    with open(args.classifiers, "rb") as f:
        classifiers = pickle.load(f)

    show_preview(classifiers, args.level, args.tile_size, args.slide_folder, args.outpath, args.feature_method)
