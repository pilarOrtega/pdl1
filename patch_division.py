from openslide import OpenSlide, deepzoom
import os
from tqdm import tqdm
import glob
import Pysiderois_Arnaud.pysliderois.tissue as tissue
import numpy
import pickle
import argparse


def get_patches(slidepath, outpath, level=10, tissue_ratio=0.25, size=256):
    """
    Function that divides a slide into patches with different resolution

    Arguments:
        - slidepath: str, path to the image to patchify
        - outpath: str, path to the folder in which the patches will be saved
        - level: int, level in which image is patchified. The bigger the level,
          the higher the number of patches and the resolution of the images.
        - tissue_ratio: float, minimum surface of tissue tile to be considered
        - size: int, side number of pixels (n pixels size*size)

    Returns:
        - n: int, number of patches
        - outpath
    """

    # Opens the slide with OpenSlide
    slide = OpenSlide(slidepath)
    slide_dz = deepzoom.DeepZoomGenerator(slide, tile_size=(size - 2), overlap=1)
    slidename = os.path.basename(slidepath)
    slidenumber = slidename.split('.')
    slidenumber = slidenumber[2]

    # Asures that the chosen level is valid
    if level < slide_dz.level_count:
        tiles = slide_dz.level_tiles[level]
        print('Level {} contains {} tiles (empty tiles included)'.format(level, slide_dz.level_tiles[level][0]*slide_dz.level_tiles[level][1]))
    else:
        print('Invalid level')
        return

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
            tile = slide_dz.get_tile(level, (i, j))
            tile_path = os.path.join(outpath, '{}-{}-level{}-{}-{}.jpg'.format(slidenumber, n, level, i, j))
            image = numpy.array(tile)[..., :3]
            mask = tissue.get_tissue_from_rgb(image)
            if mask.sum() > tissue_ratio * tile.size[0] * tile.size[1]:
                tile.save(tile_path)
                n = n + 1
    print('Total of {} tiles with tissue ratio >{} in slide {}'.format(n, tissue_ratio, slidepath))
    print()

    return n, outpath


def pickle_save(file, path, name):
    file_path = os.path.join(path, name)
    with open(file_path, "wb") as f:
        pickle.dump(file, f)


def patch_division(slides, outpath, level, tile_size, tissue_ratio):

    slides = os.path.join(slides, '*.mrxs')

    try:
        os.mkdir(outpath)
        print("Directory", outpath, "created")
        print()
    except FileExistsError:
        print("Directory", outpath, "already exists")
        print()

    patch_list = []
    for s in glob.glob(slides):
        print('[INFO] Extracting patches from slide {}'.format(s))
        n_s, outpath_slide = get_patches(s, outpath, level, tissue_ratio, tile_size)
        patch_list.append((os.path.basename(s), outpath_slide))

    pickle_save(patch_list, outpath, 'list_{}_{}.p'.format(level, tile_size))

    return patch_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script that divides a WSI in individual patches.')
    parser.add_argument('-s', '--slides', type=str, help='path to slide folder')
    parser.add_argument('-o', '--outpath', type=str, help='path to outfolder')
    parser.add_argument('-l', '--level', type=int, default=13, help='division level [Default: %(default)s]')
    parser.add_argument('-ts', '--tile_size', type=int, default=256, help='tile heigth and width in pixels [Default: %(default)s]')
    parser.add_argument('-tr', '--tissue_ratio', type=float, default=0.5, help='tissue ratio per patch [Default: %(default)s]')

    args = parser.parse_args()

    outpath = args.outpath
    tile_size = args.tile_size
    tissue_ratio = args.tissue_ratio
    slides = args.slides
    level = args.level

    patch_list = patch_division(slides, outpath, level, tile_size, tissue_ratio)
