from openslide import OpenSlide, deepzoom
import os
import numpy


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
