"""
Image exportation for concept visualisation.

It takes a concept tree and match images with corresponding nodes.
"""
# coding: utf8
import os
from ..util.csvmanagement import global_by_local_by_name as readStack
from ..util.csvmanagement import read_segcsv
from openslide import OpenSlide
import numpy
from skimage.io import imsave
import pickle


class Tree:
    """
    Docstring for Tree class.

    ************************
    """

    def __init__(self):
        """
        Create tree.

        ***********
        """
        self.parents = {}
        self.children = {}
        self.population = {}
        self.weights = {}

    def leaves_under(self, node):
        """
        Find leaves under a node.

        ************************
        """
        if node not in self.children:
            return [node]
        else:
            c1, c2 = self.children[node]
            return self.leaves_under(c1) + self.leaves_under(c2)


def leaves_under(tree, node):
    """
    Find leaves under a node.

    ************************
    """
    if node not in tree.children:
        return [node]
    else:
        c1, c2 = tree.children[node]
        return leaves_under(tree, c1) + leaves_under(tree, c2)


def truncated_leaves(tree, trunc):
    """
    Find the leaves in a tree.

    Leaves have a minimal population = trunc.
    """
    leaves = []
    for cpt in tree.parents:
        if tree.population[cpt] >= trunc:
            if cpt not in tree.children:
                # we only enter here if trunc = 1
                leaves.append(cpt)
            else:
                c1, c2 = tree.children[cpt]
                if tree.population[c1] < trunc and tree.population[c2] < trunc:
                    leaves.append(cpt)

    print("Found {} leaf concepts.".format(len(leaves)))
    return leaves


def leaves_under_truncated_leaves(tree, truncated_leaves):
    """
    Find leaves under truncated leaves.

    Each truncated leaf is a dict key and has a list of leaves.
    """
    under_leaves = dict()
    for leaf in truncated_leaves:
        under_leaves[leaf] = leaves_under(tree, leaf)
    return under_leaves


def get_slide_and_pos_by_global(segdir, stackcsv):
    """
    Compute the dictionary slide and pos by global.

    Loop over the slide and read the information in the csv seg file.
    """
    slide_n_pos_by_global = dict()
    glob_by_loc_by_name = readStack(stackcsv)
    for name, glob_by_loc in glob_by_loc_by_name.items():
        segfile = os.path.join(segdir, "{}.csv".format(name))
        for localid, x, y in read_segcsv(segfile):
            gl = glob_by_loc[localid]
            if gl not in slide_n_pos_by_global:
                # one global id gl can only have one slidename
                slide_n_pos_by_global[gl] = {"Slidename": name,
                                             "Positions": [(x, y)]}
            else:
                # but it can have many positions
                slide_n_pos_by_global[gl]["Positions"].append((x, y))
    return slide_n_pos_by_global


def export_images(slide_pos_by_global,
                  slidedir,
                  leaves_by_trunc,
                  outfolder,
                  slidetype=".mrxs",
                  level=1,
                  size=299,
                  maxperconcept=1000):
    """
    Export images for every truncated leaf.

    Loop over slides,
    For every patch in slide,
    extract patch and put it in the right folder.
    """
    # Now, for every significant leaf concept, we get the corresponding images and store them
    significant_concept_counter = 1
    for trunc, leaves in leaves_by_trunc.items():
        print("processing significant concept {} / {}".format(significant_concept_counter,
                                                              len(leaves_by_trunc)))
        print("#" * 20)
        # create the directory if does not exists
        folder = os.path.join(outfolder, str(trunc))
        if not os.path.exists(folder):
            print("creating significant concept folder")
            os.makedirs(folder)
        # get unique slidenames in leaf_concepts as well as their count
        names, c = numpy.unique([slide_pos_by_global[idx]["Slidename"] for idx in leaves], return_counts=True)
        print("significant concept id={} has {} leaf-concepts".format(trunc, len(leaves)))
        print("leaf-concepts lay in {} different slides".format(len(names)))
        n_per_slide = int(float(maxperconcept) / c.sum())
        subconcept_counter = 1
        for global_id in leaves:
            # It is obvious that one global_id can only have one slidename.
            slidename = slide_pos_by_global[global_id]["Slidename"]
            positions = slide_pos_by_global[global_id]["Positions"]
            print("- processing concept {} / {}  |  id={}  |  slidename={}  |  patch extraction={}".format(subconcept_counter,
                                                                                                           len(leaves),
                                                                                                           global_id,
                                                                                                           slidename,
                                                                                                           n_per_slide))

            for p in numpy.random.permutation(positions)[0:min(n_per_slide, len(positions))]:
                slidepath = os.path.join(slidedir, slidename + slidetype)
                slide = OpenSlide(slidepath)
                image = slide.read_region(p, level, (size, size))
                image = numpy.array(image)[:, :, 0:3]
                imsave(os.path.join(folder, "{}_{}_{}.png".format(slidename, p[0], p[1])), image)
            subconcept_counter += 1
        significant_concept_counter += 1


def tree_visualisation(slidedir, segdir, stackcsv, pickle_tree, outfolder, trunc=25,
                       slidetype=".mrxs",
                       level=1,
                       size=299,
                       maxperconcept=5000):
    """
    Compute same as above, with pathname parameters 'only'.

    ************************************
    """
    slide_pos_by_global = get_slide_and_pos_by_global(segdir, stackcsv)
    # struct_tree = Tree()
    # with open(tree, "rb") as f:
    #     tree_data = pickle.load(f)
    # struct_tree.parents = tree_data["parents"]
    # struct_tree.children = tree_data["children"]
    # struct_tree.population = tree_data["population"]
    with open(pickle_tree, "rb") as f:
        struct_tree = pickle.load(f)
    truncs = truncated_leaves(struct_tree, trunc)
    leaves_by_trunc = leaves_under_truncated_leaves(struct_tree, truncs)
    export_images(slide_pos_by_global, slidedir, leaves_by_trunc, outfolder,
                  slidetype=slidetype,
                  level=level,
                  size=size,
                  maxperconcept=maxperconcept)
