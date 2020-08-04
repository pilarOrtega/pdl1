from get_preview import *
from joblib import Parallel, delayed
import numpy
from skimage.util.shape import view_as_windows
from tqdm import tqdm
from openslide import OpenSlide, deepzoom
import os


def get_preview(slidename, classifier, level, size, slide_folder, neg=0):
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
            cluster = x[4]
            preview[im_x][im_y] = cluster + 2
    result.extend((slidename, preview))
    return result


def improve_cluster(classifiers, slide_folder):
    i = 0
    pos_dict = {}
    for c in classifiers:
        for x in range(len(c[2])):
            pos_dict[(i, (c[2][x][1], c[2][x][2]))] = x
        i += 1
    slide_list = []
    for x in classifiers:
        slide_list.append(x[0])
    previews = Parallel(n_jobs=-1)(delayed(get_preview)(s[0], s[2], 16, 224, slide_folder) for s in tqdm(classifiers))
    previews_dict = {}
    for p in previews:
        previews_dict[p[0]] = p[1]
    window_shape = (3, 3)
    for c in classifiers:
        n = 0
        index_slide = slide_list.index(c[0])
        preview = previews_dict[c[0]]
        print(preview.shape)
        windows = view_as_windows(preview, window_shape)
        for i in range(windows.shape[0]):
            for j in range(windows.shape[1]):
                window = windows[i, j]
                window = window.reshape(9)
                x = window[4]
                if x == 0:
                    continue
                if numpy.any(window[:4] == x) and numpy.any(window[5:] == x):
                    continue
                counts = numpy.bincount(window.astype(int))
                mf = numpy.argmax(counts)
                if counts[mf] < 7:
                    continue
                if mf == 0 and counts[mf] < 8:
                    continue
                # Falta añadir la condicion del clustering
                number = pos_dict[(index_slide, (i+1, j+1))]
                c[2][number][4] = mf-2
    return classifiers
