from joblib import Parallel, delayed
import numpy
from skimage.util.shape import view_as_windows
from tqdm import tqdm
from openslide import OpenSlide, deepzoom
import os


def get_preview(slidename, classifier, level, size, slide_folder, neg=0):
    result = []
    slidepath = os.path.join(slide_folder, os.path.basename(slidename))
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


def improve_clustering(classifiers, slide_folder):
    i = 0
    pos_dict = {}
    for c in classifiers:
        for x in range(len(c[1])):
            pos_dict[(i, (c[1][x][1], c[1][x][2]))] = x
        i += 1
    slide_list = []
    for x in classifiers:
        slide_list.append(x[0])
    previews = Parallel(n_jobs=-1)(delayed(get_preview)(s[0], s[1], 16, 224, slide_folder) for s in tqdm(classifiers))
    previews_dict = {}
    for p in previews:
        previews_dict[p[0]] = p[1]
    window_shape = (3, 3)
    n=0
    for c in classifiers:
        index_slide = slide_list.index(c[0])
        preview = previews_dict[c[0]]
        windows = view_as_windows(preview, window_shape)
        for i in range(windows.shape[0]):
            for j in range(windows.shape[1]):
                window = windows[i, j]
                window = window.reshape(9)
                x = window[4]
                if x == 0:
                    continue
                if numpy.any(window[:4] == x) or numpy.any(window[5:] == x):
                    continue
                counts = numpy.bincount(window.astype(int))
                mf = numpy.argmax(counts)
                if counts[mf] < 5:
                    continue
                number = pos_dict[(index_slide, (i+1, j+1))]
                if mf == 0 and counts[mf] >= 7:
                    c[1][number][4] = mf-2
                    n += 1
                    continue
                if mf == 1 and counts[mf] >= 7:
                    c[1][number][4] = mf-2
                    n += 1
                    continue
                if mf == (c[1][number][5]+2) or mf == (c[1][number][6]+2):
                    c[1][number][4] = mf-2
                    n += 1
    print('Total of {} patches changed'.format(n))
    return classifiers
