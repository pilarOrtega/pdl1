# coding: utf8
"""
A module to produce data generators for models.

***********************************************
"""
from skimage.io import imread
from keras.applications import xception as xce
import numpy


class SingleClassBatchGeneratorFromFolder(object):
    """
    Docstring for SingleClassGenerator.

    Yield batches of images from one given class.
    """

    def __init__(self, imlist, batch_size):
        """
        Instantiate generator.

        Need folder path.
        """
        self.index = 0
        self.imlist = imlist
        numpy.random.shuffle(self.imlist)
        self.batch_size = batch_size

    def __iter__(self):
        """
        Re-implement iter.

        ******************
        """
        return self

    def __next__(self):
        """
        Go to next batch.

        *****************
        """
        x = []

        for splidx in range(self.batch_size):
            if self.index < len(self.imlist) - 1:
                x.append(imread(self.imlist[self.index]))
                self.index += 1
            else:
                x.append(imread(self.imlist[self.index]))
                self.index = 0

        return xce.preprocess_input(numpy.array(x))
