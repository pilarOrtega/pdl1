import numpy
from skimage.color import rgb2grey, rgb2hed

def imagetoDAB(image, h=False):
    """
    Transforms a RGB image into a 3 channel image in which all 3 channels are
    channel DAB from color space HED.
    """
    image_hed = rgb2hed(image)
    d = image_hed[:, :, 2]
    he = image_hed[:, :, 0]
    img_dab = numpy.zeros_like(image)
    if h:
        img_dab[:, :, 0] = he
        img_dab[:, :, 1] = he
        img_dab[:, :, 2] = he
        return img_dab
    img_dab[:, :, 0] = d
    img_dab[:, :, 1] = d
    img_dab[:, :, 2] = d
    return img_dab
