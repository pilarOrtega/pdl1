import os
from openslide import OpenSlide, deepzoom
import numpy
from matplotlib import pyplot as plt
import pysliderois.tissue as tissue

def get_patches(slidepath, outpath, level = 10):
    """
    Function that divides the slide in slidepath into level_tiles

    Arguments:
        - slidepath: str, path to the image to patchify
        - outpath: str, path to the resulting patches
        - level: int, level in which image is patchified

    Returns:
        - n: int, number of patches
        - outpath: str,

    """
    slide = OpenSlide(slidepath)
    slide_dz = deepzoom.DeepZoomGenerator(slide)
    PATH = outpath + "/level_{}".format(level)

    try:
        os.mkdir(PATH)
        print("Directory", PATH, "created")
    except FileExistsError:
        print("Directory", PATH, "already exists")

    tiles = slide_dz.level_tiles[level]
    n=0
    print("Saving tiles image "+ slidepath + "...")
    for i in range(tiles[0]):
      for j in range(tiles[1]):
        tile = slide_dz.get_tile(level,(i,j))
        tile_path = PATH + '/{}_slide1_level{}_{}_{}.jpg'.format(n,level,i,j)
        tile_bw = tile.convert(mode='L')
        tile_bw = numpy.array(tile_bw)
        x = tile_bw.mean()
        if x < 250:
          tile.save(tile_path)
          n = n + 1
          if n%20 == 0:
              sys.stdout.write(n)
    print('Total of {} tiles in level {}'.format(n,level))
    return n, PATH

slidefile1 = "/Users/pilarortega/Desktop/STAGE_ONCOPOLE/slides/NVA_RC.PDL1.V1_18T040165.2B.4963.PDL1.mrxs"
outpath = "/Users/pilarortega/Desktop/STAGE_ONCOPOLE/patches"
n = get_patches(slidefile1, outpath, 15)
