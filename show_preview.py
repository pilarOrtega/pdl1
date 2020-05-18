import numpy
from matplotlib import pyplot as plt
import argparse
import pickle
from save_cluster import *
from openslide import OpenSlide, deepzoom
import csv


def get_preview(classifiers, level, size, slide_folder):
    previews = []
    for s in classifiers:
        slidename = s[0]
        slidepath = os.path.join(slide_folder, slidename)
        classifier = s[2]
        slide = OpenSlide(slidepath)
        slide_dz = deepzoom.DeepZoomGenerator(slide, tile_size=(size - 2), overlap=1)
        tiles = slide_dz.level_tiles[level]
        preview = numpy.zeros(tiles)
        n_division = (s[2].shape[1]) - 4
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


def show_preview(classifiers, level, size, slide_folder, outpath):
    previews = get_preview(classifiers, level, size, slide_folder)
    for im in previews:
        slidename = '{}_clusters-level{}-ts{}.png'.format(im[0], level, size)
        name = os.path.join(outpath, slidename)
        fig = plt.figure()
        plt.imshow(im[1], cmap='tab20b')
        plt.colorbar()
        fig.savefig(name, bbox_inches='tight', dpi=fig.dpi)
        plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Obtains a preview of the slide which display the clusters in different colors')
    parser.add_argument('-c', '--csv_files', type=str, nargs='+', help='path(s) to csv file(s)')
    parser.add_argument('-o', '--outpath', type=str, help='name of the out file (.png)')
    parser.add_argument('-l', '--level', type=int, default=13, help='division level [Default: %(default)s]')
    parser.add_argument('-ts', '--tile_size', type=int, default=256, help='tile heigth and width in pixels [Default: %(default)s]')
    parser.add_argument('-s', '--slide_folder', type=str, default=0.5, help='path to slide folder')

    args = parser.parse_args()

    classifiers = []
    csv_files = args.csv_files
    for file in csv_files:
        print('Reading file '+file)
        list_file = read_csv(file)
        path = os.path.dirname(file)
        slide = os.path.basename(file)
        slide = os.path.splitext(slide)[0]
        slide = slide.split('-')[0]
        path = os.path.join(path, slide)
        classifiers.append((slide, path, list_to_array(list_file)))

    show_preview(classifiers, args.level, args.tile_size, args.slide_folder, args.outpath)
