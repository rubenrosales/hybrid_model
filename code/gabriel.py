import numpy as np
import math
from itertools import product, chain, combinations, permutations, starmap
from functools import partial

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def mid(p1, p2):
    return (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2

def get_gabriel_graph(points: np.array, points2: np.array = None):
    if points2 is None:
        points2 = points
    aux = []
    for p1, p2 in product(points, points2):
        if p1 == p2:
            continue
        m = mid(p1, p2)
        aux.append((m, dist(m, p1), p1, p2))

    edges = []
    for m, rad, p1, p2 in aux:
        if (True in map(lambda p: dist(m, p) < rad, points) and
            True in map(lambda p: dist(m, p) < rad, points2)):
            continue
        edges.append((p1, p2))

    return edges

def brute_match(points, points2):
    edges = []
    r = None
    for _points2 in permutations(points2):
        _r = max(starmap(dist, zip(points, _points2)))
        if r is None or _r < r:
            r = _r
            edges = list(zip(points, _points2))
    return edges

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import collections  as mc

if __name__ == '__main__':
    points = list(map(tuple, np.random.rand(5, 2)))
    points2 = list(map(tuple, np.random.rand(5, 2)))
    edges = set(get_gabriel_graph(points, points2))

    print('Checking Brutely')
    brute_edges = set(brute_match(points, points2))
    if brute_edges.issubset(edges):
        print('TRUTH')
    else:
        print('FALSE')

    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()
    ax.scatter(*zip(*points), color='b')
    ax.scatter(*zip(*points2), color='y')

    lc = mc.LineCollection(edges)
    ax.add_collection(lc)
    
    lc_brute = mc.LineCollection(brute_edges, color='g')
    ax.add_collection(lc_brute)
    
    lc_absent = mc.LineCollection(brute_edges - edges, color='r')
    ax.add_collection(lc_absent)

    ax.autoscale()
    ax.margins(0.1)
    plt.show()

