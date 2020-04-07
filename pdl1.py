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

image1 = slide1.read_region((0,0), 7, slide1.level_dimensions[7])
image1 = numpy.array(image1)[:, :, 0:3]

plt.figure(figsize=(7, 14))
plt.imshow(image1)
plt.show()

image1_tejido_mask = tissue.get_tissue(image1, method = "rgb")
#imagen1_tejido = image1*image1_tejido_mask

plt.imshow(image1_tejido_mask)

deepzoom1 = deepzoom.DeepZoomGenerator(slide1)

level = 16
PATH = "/Users/pilarortega/Desktop/STAGE_ONCOPOLE/level_{}".format(level)
try:
  os.mkdir(PATH)
  print("Directory", PATH, "created")
except FileExistsError:
  print("Directory", PATH, "already exists")

tiles = deepzoom1.level_tiles[level]
n=0
for i in range(tiles[0]):
  for j in range(tiles[1]):
    tile = deepzoom1.get_tile(level,(i,j))
    tile_path = PATH + '/{}_slide1_level{}_{}_{}.jpg'.format(n,level,i,j)
    tile_bw = tile.convert(mode='L')
    tile_bw = numpy.array(tile_bw)
    x = tile_bw.mean()
    if x < 250:
      tile.save(tile_path)
      n = n + 1
