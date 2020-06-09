# coding: utf8
"""
Functions that would be useful for WSI segmentation based on graphs.

Graph nodes are not pixels, but patches.
Well... a pixel is basically a patch with size 1 so it is more general.
"""
import os
from bidict import bidict
from itertools import product
import numpy
import csv
import networkx as nx


class Tree(object):
    """
    Very simple object to handle tree structures.

    *********************************************
    """

    def __init__(self):
        """
        Comes with a few dictionaries.

        ******************************
        """
        self.parents = {}
        self.children = {}
        self.weights = {}
        self.population = {}
        self.descriptor = {}
        self.slides = {}


class Error(Exception):
    """
    Base of custom errors.

    **********************
    """

    pass


class NotBinaryTreeError(Error):
    """
    Raise when no class is found in a datafolder.

    *********************************************
    """

    pass


def matches_and_score(node_list1, node_list2):
    """
    Match nodes in list1 and list2 (best matchings).

    Given we are testing binary trees, len(node_list{i}) = 2.
    node_list{i} = {id1: desc1, id2: desc2}.
    """
    # get tuples id, desc
    if len(node_list1) != 2:
        raise NotBinaryTreeError("Tree 1 is not Binary!")
    if len(node_list2) != 2:
        raise NotBinaryTreeError("Tree 2 is not Binary!")
    node11, node12 = list(node_list1.items())
    node21, node22 = list(node_list2.items())
    # compute matching scores
    match1 = (node11[1] * node21[1]).sum() + (node12[1] * node22[1]).sum()
    match2 = (node12[1] * node21[1]).sum() + (node11[1] * node22[1]).sum()
    # tuples returned are the ones that maximize score
    if match1 > match2:
        tuples = [(node11[0], node21[0]), (node12[0], node22[0])]
    else:
        tuples = [(node12[0], node21[0]), (node11[0], node22[0])]
    return tuples


def _iter_level_score(tree1, tree2, node1, node2):
    """
    Recursive function to compute similarity score of a tree.

    *********************************************************
    """
    # if node1 has no children or node2 has no children
    # always the same, return 0, there is no overlap
    if (node1 not in tree1.children) or (node2 not in tree2.children):
        return (tree1.descriptor[node1] * tree2.descriptor[node2]).sum()
    # case when they both have children
    ch1 = tree1.children[node1]
    ch2 = tree2.children[node2]
    node_list1 = {c: tree1.descriptor[c] for c in ch1}
    node_list2 = {c: tree2.descriptor[c] for c in ch2}
    tuples = matches_and_score(node_list1, node_list2)
    score = 0.
    for t in tuples:
        score += 0.5 * _iter_level_score(tree1, tree2, t[0], t[1])
    return score


def similarity_score(tree1, tree2):
    """
    Compute entire similarity_score betweent two trees.

    ***************************************************
    """
    chroot1 = max(tree1.children.keys())
    chroot2 = max(tree2.children.keys())
    return _iter_level_score(tree1, tree2, chroot1, chroot2)


class UFDS:
    """
    Union-find data structure.

    Each unionFind instance X maintains a family of disjoint sets of
    hashable objects, supporting the following two methods:

    - X[item] returns a name for the set containing the given item.
      Each set is named by an arbitrarily-chosen one of its members; as
      long as the set remains unchanged it will keep the same name. If
      the item is not yet part of a set in X, a new singleton set is
      created for it.

    - X.union(item1, item2, ...) merges the sets containing each item
      into a single larger set.  If any item is not yet part of a set
      in X, it is added to X as one of the members of the merged set.

      Union-find data structure. Based on Josiah Carlson's code,
      http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/215912
      with significant additional changes by D. Eppstein.
      http://www.ics.uci.edu/~eppstein/PADS/UnionFind.py

    """

    def __init__(self):
        """
        Create a new empty union-find structure.

        ****************************************
        """
        self.weights = {}
        self.parents = {}

    def get_root(self, node):
        """
        Find and return the name of the set containing the node.

        ********************************************************
        """
        # check for previously unknown object
        if node not in self.parents:
            self.parents[node] = node
            self.weights[node] = 1
            return node
        # find path of objects leading to the root
        path = [node]
        root = self.parents[node]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]
        # compress the path and return
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def __iter__(self):
        """
        Iterate through all items ever found or unioned by this structure.

        ******************************************************************
        """
        return iter(self.parents)

    def union(self, node1, node2):
        """
        Find the sets containing the objects and merge them all.

        ********************************************************
        """
        roots = [self.get_root(node1), self.get_root(node2)]
        # Find the heaviest root according to its weight.
        heaviest = max(roots, key=lambda r: self.weights[r])
        for r in roots:
            if r != heaviest:
                self.weights[heaviest] += self.weights[r]
                self.parents[r] = heaviest


class FELZUFDS:
    """
    Union-find data structure for felzenszwalb algorithm.

    Each unionFind instance X maintains a family of disjoint sets of
    hashable objects, supporting the following two methods:

    - X[item] returns a name for the set containing the given item.
      Each set is named by an arbitrarily-chosen one of its members; as
      long as the set remains unchanged it will keep the same name. If
      the item is not yet part of a set in X, a new singleton set is
      created for it.

    - X.union(item1, item2, ...) merges the sets containing each item
      into a single larger set.  If any item is not yet part of a set
      in X, it is added to X as one of the members of the merged set.

      Union-find data structure. Based on Josiah Carlson's code,
      http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/215912
      with significant additional changes by D. Eppstein.
      http://www.ics.uci.edu/~eppstein/PADS/UnionFind.py

    """

    def __init__(self):
        """
        Create a new empty union-find structure.

        ****************************************
        """
        self.population = {}
        self.parents = {}
        self.internal_nrj = {}

    def get_root(self, node):
        """
        Find and return the name of the set containing the node.

        ********************************************************
        """
        # check for previously unknown object
        if node not in self.parents:
            self.parents[node] = node
            self.population[node] = 1
            self.internal_nrj[node] = 0
            return node
        # find path of objects leading to the root
        path = [node]
        root = self.parents[node]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]
        # compress the path and return
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def __iter__(self):
        """
        Iterate through all items ever found or unioned by this structure.

        ******************************************************************
        """
        return iter(self.parents)

    def union(self, node1, node2, weight):
        """
        Find the sets containing the objects and merge them all.

        ********************************************************
        """
        roots = [self.get_root(node1), self.get_root(node2)]
        # Find the heaviest root according to its weight.
        heaviest = max(roots, key=lambda r: self.population[r])
        for r in roots:
            if r != heaviest:
                self.population[heaviest] += self.population[r]
                # internal energy is computed as the max of the weights among a segment.
                # Since considered weight is bigger (or equal) than any other weight
                # visited before, we can safely set this weight as the internal energy
                # of the newly created segment.
                self.internal_nrj[heaviest] = weight
                self.parents[r] = heaviest


class Kruskal:
    """
    Convenient class to compute Kruskal edges in a graph.

    *****************************************************
    """

    def __init__(self):
        """
        Kruskal algorithm efficient computation.

        Relies on "Union-find" data structure.
        """
        self.subtrees = UFDS()

    def edge_key_function(self, edge):
        """
        Given an edge, returns the weight of that edge.

        ***********************************************
        """
        return edge[-1]

    def spanning_tree(self, edges):
        """
        Given a list of edges, yields the kruskal edges.

        ************************************************
        """
        # edges are sorted by non-decreasing order of weight
        edges = sorted(edges, key=self.edge_key_function)

        for edge in edges:
            # if graph node1 container is different from node2 container
            n1 = edge[0]
            n2 = edge[1]
            rn1 = self.subtrees.get_root(n1)
            rn2 = self.subtrees.get_root(n2)
            if rn1 != rn2:
                self.subtrees.union(n1, n2)
                yield n1, n2


class WeightedKruskal:
    """
    Same as above, just add descriptor to new fused nodes.

    ******************************************************
    """

    def __init__(self):
        """
        Kruskal algorithm efficient computation.

        Relies on "Union-find" data structure.
        """
        self.subtrees = UFDS()

    def edge_key_function(self, edge):
        """
        Given an edge, returns the weight of that edge.

        ***********************************************
        """
        return edge[-1]

    def spanning_tree(self, edges):
        """
        Given a list of edges, yields the kruskal edges.

        ************************************************
        """
        # edges are sorted by non-decreasing order of weight
        edges = sorted(edges, key=self.edge_key_function)

        for edge in edges:
            # if graph node1 container is different from node2 container
            n1 = edge[0]
            n2 = edge[1]
            rn1 = self.subtrees.get_root(n1)
            rn2 = self.subtrees.get_root(n2)
            if rn1 != rn2:
                self.subtrees.union(n1, n2)
                yield n1, n2, edge[-1]


class Felzenszwalb:
    """
    Kruskal with stopping criteria.

    *********************************
    """

    def __init__(self, coeff=10):
        """
        Kruskal algorithm efficient computation.

        Relies on "Union-find" data structure.
        """
        self.subtrees = FELZUFDS()
        self.k = float(coeff)

    def edge_key_function(self, edge):
        """
        Given an edge, returns the weight of that edge.

        ***********************************************
        """
        return edge[-1]

    def segmentation(self, edges):
        """
        Given a list of edges, yields the kruskal edges.

        ************************************************
        """
        # edges are sorted by non-decreasing order of weight
        edges = sorted(edges, key=self.edge_key_function)

        for edge in edges:
            # if graph node1 container is different from node2 container
            n1 = edge[0]
            n2 = edge[1]
            w = edge[-1]
            rn1 = self.subtrees.get_root(n1)
            rn2 = self.subtrees.get_root(n2)
            Mint = min(self.subtrees.internal_nrj[rn1] + self.k / self.subtrees.population[rn1],
                       self.subtrees.internal_nrj[rn2] + self.k / self.subtrees.population[rn2])
            if rn1 != rn2 and w <= Mint:
                self.subtrees.union(n1, n2, w)
                yield n1, n2


class PatchGraph:
    """
    Convenient class to handle patch graphs.

    ****************************************
    """

    def __init__(self):
        """
        Create a pseudo-graph structure from a csv file.

        ************************************************
        """
        self.delta = 0
        self.archi = ""
        self.n_features = 0
        self.position_by_id = bidict()
        self.feature_by_id = dict()
        self.edges = []

    def from_csv(self, filepath):
        """
        Load edge data from csv.

        ************************
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)
        # if file exists, open in read mode
        with open(filepath, "r") as f:
            # instanciate the reader
            reader = csv.DictReader(f)
            # set the number of features
            self.n_features = max([int(name.rsplit("_", 1)[1]) for name in reader.fieldnames if "feature" in name]) + 1
            # set the archi
            for name in reader.fieldnames:
                if "feature" in name:
                    self.archi = name.rsplit("_", 2)[0]
                    break
            # line count
            line_count = 0
            px1 = None
            px2 = None
            # loop over rows in csv
            # and manage to find delta
            for row in reader:
                # get position
                x = int(row["XPosition"])
                y = int(row["YPosition"])
                # update delta
                # case of the first attribution of px1
                if px1 is None:
                    px1 = x
                else:
                    # case of the first attribution of px2
                    # only  if px1 has already been attributed a value
                    # also the first attribution of delta
                    if px2 is None:
                        if x != px1:
                            px2 = x
                            self.delta = abs(px1 - px2)
                    else:
                        # case where px1 is not None and px2 is not None
                        if 0 < abs(px1 - x) < self.delta:
                            px2 = x
                            self.delta = abs(px1 - x)
                        elif 0 < abs(px2 - x) < self.delta:
                            px1 = x
                            self.delta = abs(px2 - x)

                self.position_by_id[line_count] = x, y
                self.feature_by_id[line_count] = numpy.array([float(row["{}_feature_{}".format(self.archi, k)]) for k in range(self.n_features)])
                line_count += 1
        # end of the file,
        # check the interval
        print("read {} lines in csv file, found {} pixels interval.".format(line_count, self.delta))
        # now create the edges
        edge_set = set()
        for index, position in self.position_by_id.items():
            x, y = position
            for dx, dy in product([-self.delta, 0, self.delta], repeat=2):
                if (x + dx, y + dy) in self.position_by_id.inverse:
                    if not(dx == 0 and dy == 0):
                        edge_set.add(frozenset((self.position_by_id.inverse[(x, y)],
                                                self.position_by_id.inverse[(x + dx, y + dy)])))

        self.edges = []
        for n1, n2 in edge_set:

            w = numpy.sqrt(((self.feature_by_id[n1] - self.feature_by_id[n2]) ** 2).sum())
            self.edges.append((n1, n2, w))

    def clusters(self, coeff=10):
        """
        Felzenszwalb segmentation of the graph.

        ***************************************
        """
        felz = Felzenszwalb(coeff=coeff)
        segedges = [e for e in felz.segmentation(self.edges)]
        totGraph = nx.Graph(segedges)
        # sort by decreasing order of component size
        self.CCX = sorted([totGraph.subgraph(c) for c in nx.connected_components(totGraph)], reverse=True, key=lambda x: len(list(x.nodes())))
        for component in self.CCX:
            nodes = list(component.nodes())
            positions = [self.position_by_id[node] for node in nodes]
            features = [self.feature_by_id[node] for node in nodes]
            yield positions, features

    def cluster_to_csv(self, filepath, coeff=10):
        """
        Write a felzenszwalb segmentation into a csv file.

        Usually, same file as the one loaded previously.
        It basically just add a "Label" column to the csv.
        """
        csv_columns = ["XPosition", "YPosition", "Label"]
        csv_columns += ["{}_feature_{}".format(self.archi, k) for k in range(self.n_features)]
        felz = Felzenszwalb(coeff=coeff)
        # felzenszwalb edges
        segedges = [e for e in felz.segmentation(self.edges)]
        # graphs from felzenszwalb forest
        totGraph = nx.Graph(segedges)
        self.CCX = [totGraph.subgraph(c) for c in nx.connected_components(totGraph)]
        # loop over connected components and write in file
        with open(filepath, "w") as f:
            writer = csv.DictWriter(f, csv_columns)
            writer.writeheader()
            compcount = 0
            for component in self.CCX:
                # get patch ids of the given segment
                nodes = list(component.nodes())
                # write every patch with its label as a csv row
                for node in nodes:
                    dico = dict()
                    x, y = self.position_by_id[node]
                    dico["XPosition"] = x
                    dico["YPosition"] = y
                    dico["Label"] = compcount
                    for k in range(self.n_features):
                        dico["{}_feature_{}".format(self.archi, k)] = self.feature_by_id[node][k]
                    writer.writerow(dico)
                compcount += 1
