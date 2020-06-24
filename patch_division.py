from openslide import OpenSlide, deepzoom
import os
from tqdm import tqdm
import glob
import Pysiderois_Arnaud.pysliderois.tissue as tissue
import numpy
import pickle
import argparse
from joblib import Parallel, delayed
from auxiliary_functions.pickle_functions import *
from auxiliary_functions.slide_preview import *
import time


def get_patches(slidepath, outpath, level=10, tissue_ratio=0.25, size=256):
    """
    Function that divides a slide into patches with different resolution. The
    patches are saved inside a folder with the slide name, and have the format
    {slide_name}#{patch_number}-level{}-{x}-{y}.jpg. It also saves a preview
    for each slide under the format slidename.png

    Arguments:
        - slidepath: str, path to the image to patchify
        - outpath: str, path to the folder in which a new folder will be
          created, where the patches will be saved. This folder has the same
          name as the image to patchify
        - level: int, level in which image is patchified. The bigger the level,
          the higher the number of patches and the resolution of the images.
          Default = 10
        - tissue_ratio: float, minimum surface of tissue tile to be considered.
          Default = 0.25
        - size: int, side number of pixels (n pixels size*size). Default = 256

    Returns:
        - n: int, number of patches
        - outpath: str, path to folder where the patches are saved

    """

    # Opens the slide with OpenSlide
    slide = OpenSlide(slidepath)

    # Gets deepzoom tile division
    slide_dz = deepzoom.DeepZoomGenerator(slide, tile_size=(size - 2), overlap=1)

    # Gets the name and number of the slide
    slidename = os.path.basename(slidepath)

    # Saves a preview of the slide under 'slidename.png'
    slide_preview(slide, slidename, outpath)

    # Asures that the chosen level is valid
    if level < slide_dz.level_count:
        tiles = slide_dz.level_tiles[level]
        print('Level {} contains {} tiles (empty tiles included)'.format(level, slide_dz.level_tiles[level][0]*slide_dz.level_tiles[level][1]))
    else:
        print('Invalid level')
        return

    # Creates new directory - where patches will be stored
    outpath = os.path.join(outpath, slidename)
    try:
        os.mkdir(outpath)
        print("Directory", outpath, "created")
    except FileExistsError:
        print("Directory", outpath, "already exists")

    # Saves tiles if detects tissue presence higher than tissue_ratio
    n = 0
    print("Saving tiles image " + slidepath + "...")
    for i in tqdm(range(tiles[0])):
        for j in range(tiles[1]):
            # Gets the tile in position (i, j)
            tile = slide_dz.get_tile(level, (i, j))
            image = numpy.array(tile)[..., :3]
            mask = tissue.get_tissue_from_rgb(image, blacktol=10, whitetol=240)
            # Saves tile in outpath only if tissue ratio is higher than threshold
            if mask.sum() > tissue_ratio * tile.size[0] * tile.size[1]:
                tile_path = os.path.join(outpath, '{}#{}-level{}-{}-{}.jpg'.format(slidename, n, level, i, j))
                tile.save(tile_path)
                n = n + 1
    print('Total of {} tiles with tissue ratio >{} in slide {}'.format(n, tissue_ratio, slidepath))
    print()

    return n


def patch_division(slides, outpath, level, tile_size=224, tissue_ratio=0.50, jobs=1):
    """
    Gets a set of slides (*.PDL1.mrxs) and divides each one of them into patches.
    The final patches are stored in a folder with the same name as the slide.

    Arguments:
        - Slides: str, path to folder with slides.
        - Outpath: str, directory in which the slide folders will be saved.
          Will be created if doesn't exist
        - level: int, resolution level
        - tissue_ratio: float, minimum surface of tissue tile to be considered.
          Default = 0.25
        - tile_size: int, side number of pixels (n pixels size*size). Default = 256
    """

    # Creates directory outpath if doesn't exist yet
    try:
        os.mkdir(outpath)
        print("Directory", outpath, "created")
        print()
    except FileExistsError:
        print("Directory", outpath, "already exists")
        print()

    # Collects all files in folder slide with the format *.PDL1.mrxs
    slides = os.path.join(slides, '*.PDL1.mrxs')
    slide_list = []
    start = time.time()
    n = Parallel(n_jobs=jobs)(delayed(get_patches)(s, outpath, level, tissue_ratio, tile_size) for s in glob.glob(slides))
    end = time.time()
    print('Total time patch extraction: {:.4f} s'.format(end-start))
    for s in glob.glob(slides):
        slidename = os.path.basename(s)
        outpath_slide = os.path.join(outpath, slidename)
        slide_list.append((slidename, outpath_slide))

    pickle_save(slide_list, outpath, 'list_{}_{}.p'.format(level, tile_size))

    n = sum(n)
    print('Total number of patches extracted {}'.format(n))
    print()
    return slide_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script that divides a WSI in individual patches.')
    parser.add_argument('-s', '--slides', type=str, help='path to slide folder')
    parser.add_argument('-o', '--outpath', type=str, help='path to outfolder')
    parser.add_argument('-l', '--level', type=int, default=13, help='division level [Default: %(default)s]')
    parser.add_argument('-ts', '--tile_size', type=int, default=256, help='tile heigth and width in pixels [Default: %(default)s]')
    parser.add_argument('-tr', '--tissue_ratio', type=float, default=0.5, help='tissue ratio per patch [Default: %(default)s]')
    parser.add_argument('-j', '--jobs', type=int)

    args = parser.parse_args()

    outpath = args.outpath
    tile_size = args.tile_size
    tissue_ratio = args.tissue_ratio
    slides = args.slides
    level = args.level
    jobs = args.jobs

    slide_list = patch_division(slides, outpath, level, tile_size=tile_size, tissue_ratio=tissue_ratio, jobs=jobs)
