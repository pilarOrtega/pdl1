import numpy
from matplotlib import pyplot as plt
import argparse
import pickle
from save_cluster import *
from openslide import OpenSlide, deepzoom
import csv


def get_preview(classifiers, level, size, slide_folder, n_division):
    previews = []
    for s in classifiers:
        slidename = s[0]
        slidepath = os.path.join(slide_folder, slidename)
        classifier = s[2]
        slide = OpenSlide(slidepath)
        slide_dz = deepzoom.DeepZoomGenerator(slide, tile_size=(size - 2), overlap=1)
        tiles = slide_dz.level_tiles[level]
        preview = numpy.zeros(tiles)
        for x in classifier:
            im_x = int(x[1])
            im_y = int(x[2])
            if x[3] == 0:
                preview[im_x][im_y] = 1
            else:
                cluster = 0
                for j in range(n_division):
                    exp = n_division - j - 1
                    cluster = cluster + x[j+4] * (2**exp)
                preview[im_x][im_y] = cluster + 2
        previews.append((slidename, preview))
    return previews


def show_preview(classifiers, level, size, slide_folder, outpath, feature_method, n_division=0):
    if n_division == 0:
        n_division = (s[2].shape[1]) - 4
    previews = get_preview(classifiers, level, size, slide_folder, n_division)
    for im in previews:
        slidename = '{}-{}-level{}-ts{}-{}.png'.format(im[0], feature_method, level, size, n_division)
        name = os.path.join(outpath, slidename)
        fig = plt.figure()
        plt.imshow(im[1], cmap='tab20b')
        plt.colorbar()
        fig.savefig(name, bbox_inches='tight', dpi=fig.dpi)
        plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Obtains a preview of the slide which display the clusters in different colors')
    parser.add_argument('-c', '--classifiers', type=str, help='path to classifier file')
    parser.add_argument('-o', '--outpath', type=str, help='name of the out file (.png)')
    parser.add_argument('-l', '--level', type=int, default=13, help='division level [Default: %(default)s]')
    parser.add_argument('-ts', '--tile_size', type=int, default=256, help='tile heigth and width in pixels [Default: %(default)s]')
    parser.add_argument('-s', '--slide_folder', type=str, default=0.5, help='path to slide folder')
    parser.add_argument('-n', '--n_division', type=int, default=0, help='number of divisions')
    parser.add_argument('-f', '--feature_method', type=str)

    args = parser.parse_args()

    with open(args.classifiers, "rb") as f:
        classifiers = pickle.load(f)

    show_preview(classifiers, args.level, args.tile_size, args.slide_folder, args.outpath, args.feature_method, args.n_division)
