from PIL import ImageOps
from openslide import OpenSlide, deepzoom
from matplotlib import pyplot as plt
import numpy
import os

def slide_preview(slide, slidename, outpath):
    """
    Gets the preview from a given Openslide slide
    """
    image = slide.read_region((0, 0), 7, slide.level_dimensions[7])
    image = ImageOps.mirror(image)
    image = numpy.array(image)[:, :, 0:3]
    name = '{}.png'.format(slidename)
    name = os.path.join(outpath, name)
    fig = plt.figure()
    plt.imshow(image)
    fig.savefig(name, bbox_inches='tight', dpi=fig.dpi)
