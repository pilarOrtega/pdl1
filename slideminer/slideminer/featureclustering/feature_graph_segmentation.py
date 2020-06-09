"""
Feature space segmentation using graph methods.

For now it only uses Felzenszwalb technique with L2 norm in feature space.
Features are usually computed by neural networks.
Felzenszwalb is initialized with a knn graph.
"""

# coding: utf8
import os
import pickle
import csv
import numpy
from ..util.csvmanagement import get_segmentation_info
from sklearn.neighbors import KDTree
from ..util.graph import WeightedKruskal
from ..util.graph import Tree


def get_leaves_from_segment_stack(csvname):
    """
    Create nodes of the initial graph.

    Loop over the csv lines, get node ID and node description.
    """
    nodes = dict()
    slidenames = {}
    n_features, archi = get_segmentation_info(csvname)
    line_count = 0
    with open(csvname, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            line_count += 1
            slidenames[int(row["GlobalId"])] = row["Slidename"]
            nodes[int(row["GlobalId"])] = numpy.array([float(row["{}_feature_{}".format(archi, k)])
                                                       for k in range(n_features)])

    print("{} lines in the csv segment pot.".format(line_count))
    print("{} nodes found.".format(len(nodes)))
    print("should be the same.")
    return nodes, slidenames


def knn_edges(nodes, neighbors=5):
    """
    Find n-connectivity edges in the initial graph.

    Use knn to find edges.
    """
    sorted_idx = sorted(list(nodes.keys()))
    X = numpy.array([nodes[id] for id in sorted_idx])
    tree = KDTree(X)
    unique_edges = set()
    weights, indices = tree.query(X, neighbors)
    # print(weights)
    for k in range(len(weights)):
        # do not take the first (it's the node itself...)
        k_weights = weights[k][1::]
        k_indices = indices[k][1::]
        for w, i in zip(k_weights, k_indices):
            unique_edges.add((frozenset((sorted_idx[k], i)), w))

    # then transform unique_edges into a list of tuples...
    edges = []
    for edge in unique_edges:
        edges.append(tuple(edge[0]) + tuple([edge[1]]))

    return edges


def get_root(tree, node):
    """
    Get the root of a node in a tree.

    *********************************
    """
    if node not in tree.parents:
        tree.parents[node] = node
        tree.population[node] = 1
        return node

    root = node
    while(root != tree.parents[root]):
        root = tree.parents[root]
    return root


def find_significant_children(tree, node):
    """
    Find the significant children of a node.

    Significant is the children pair with biggest smallest. LOL.
    """
    if node not in tree.children:
        return None
    smax = 1
    c1, c2 = tree.children[node]
    sch = c1, c2
    while tree.population[c1] > 1 or tree.population[c2] > 1:
        if tree.population[c1] >= tree.population[c2]:
            small, big = c2, c1
        else:
            small, big = c1, c2
        if tree.population[small] >= smax:
            smax = tree.population[small]
            sch = small, big
        c1, c2 = tree.children[big]
    return sch


def build_kruskal_concept_tree(csvsegstack, outputdir):
    """
    Create and store the concept tree.

    Compute the kruskal hierarchical segmentation and store it.
    """
    tree_struct = Tree()
    nodes, slidenames = get_leaves_from_segment_stack(csvsegstack)
    for nodeid, desc in nodes.items():
        tree_struct.descriptor[nodeid] = desc
        tree_struct.slides[nodeid] = set()
        tree_struct.slides[nodeid].add(slidenames[nodeid])
    edges = knn_edges(nodes)
    krusk = WeightedKruskal()
    last_cpt_id = max(nodes.keys()) + 1
    for kedge in krusk.spanning_tree(edges):
        last_cpt = get_root(tree_struct, last_cpt_id)
        n1, n2, w = kedge
        rn1 = get_root(tree_struct, n1)
        rn2 = get_root(tree_struct, n2)
        tree_struct.parents[rn1] = last_cpt
        tree_struct.parents[rn2] = last_cpt
        tree_struct.children[last_cpt] = (rn1, rn2)
        tree_struct.weights[last_cpt] = w
        tree_struct.population[last_cpt] = tree_struct.population[rn1] + tree_struct.population[rn2]
        tree_struct.descriptor[last_cpt] = (tree_struct.population[rn1] * tree_struct.descriptor[rn1])\
            + (tree_struct.descriptor[rn2] * tree_struct.population[rn2])
        tree_struct.descriptor[last_cpt] /= tree_struct.population[last_cpt]
        tree_struct.slides[last_cpt] = tree_struct.slides[rn1] | tree_struct.slides[rn2]
        last_cpt_id += 1

    with open(os.path.join(outputdir, "full_tree.p"), "wb") as f:
        pickle.dump(tree_struct, f)
    return tree_struct


def _iter_build_most_significant_tree(ktree, stree, node):
    """
    Create the most significant tree.

    Compute the most significant tree iteratively by pruning nodes
    in the kruskal tree.
    """
    sch = find_significant_children(ktree, node)
    if sch is not None:
        small, big = sch
        stree.parents[small] = node
        stree.parents[big] = node
        stree.children[node] = [small, big]
        stree.population[node] = ktree.population[node]
        stree.descriptor[node] = ktree.descriptor[node]
        stree.weights[node] = ktree.weights[node]
        stree.slides[node] = ktree.slides[node]
        _iter_build_most_significant_tree(ktree, stree, small)
        _iter_build_most_significant_tree(ktree, stree, big)


def build_most_significant_tree(ktree, outputdir):
    """
    Create and store the most significant tree.

    Most significant tree is obtained by pruning the kruskal tree.
    """
    stree = Tree()
    root = max(ktree.parents.keys())
    for cpt, popval in ktree.population.items():
        if popval == 1:
            stree.population[cpt] = 1
    _iter_build_most_significant_tree(ktree, stree, root)
    with open(os.path.join(outputdir, "most_significant_tree.p"), "wb") as f:
        pickle.dump(stree, f)
    return stree
