import os
from openslide import OpenSlide, deepzoom
import numpy
from matplotlib import pyplot as plt
import pysliderois.tissue as tissue

slidefile1 = "/Users/pilarortega/Desktop/STAGE_ONCOPOLE/slides/NVA_RC.PDL1.V1_18T040165.2B.4963.PDL1.mrxs"
slide1 = OpenSlide(slidefile1)

image1 = slide1.read_region((0,0), 7, slide1.level_dimensions[7])
image1 = numpy.array(image1)[:, :, 0:3]

plt.figure(figsize=(7, 14))
plt.imshow(image1)
plt.show()

image1_tejido_mask = tissue.get_tissue(image1, method = "rgb")
#imagen1_tejido = image1*image1_tejido_mask
