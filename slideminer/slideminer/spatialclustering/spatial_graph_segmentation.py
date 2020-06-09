"""
Slide segmentation using graph methods.

For now it only uses Felzenszwalb technique with L2 norm in feature space.
Features are usually computed by neural networks.
"""
# coding: utf8
from ..util import graph
import os


def slide_folder_felzenszwalb(inputfolder,
                              outputfolder,
                              sizecoeff=10):
    """
    Segment a slide description with Felzenszwalb technique.

    ********************************************************
    """
    for name in os.listdir(inputfolder):
        if name[0] != '.' and os.path.splitext(name)[1] == ".csv":
            print("processing file: {}".format(name))
            g = graph.PatchGraph()
            g.from_csv(os.path.join(inputfolder, name))
            print("graph has: {} edges".format(len(g.edges)))
            g.cluster_to_csv(os.path.join(outputfolder, name), coeff=sizecoeff)
            print("graph has: {} connected components".format(len(g.CCX)))
