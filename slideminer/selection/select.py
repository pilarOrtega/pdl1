# coding: utf8
"""
A Module to select promising concepts in a clustering.

Hierarchical clustering.
"""
from ..util.graph import Tree


def truncated_leaves(tree: Tree, trunc=25):
    """
    Trunc a tree based on nodes population.

    ***************************************
    """
    for node, children in tree.children:
        c1, c2 = children
        if tree.population[node] >= trunc and tree.population[c1] < trunc and tree.population[c2] < trunc:
            yield {"node": node,
                   "population": tree.population[node],
                   "descriptor": tree.descriptor[node],
                   "weights": tree.weights[node],
                   "slides": tree.slides[node]}
