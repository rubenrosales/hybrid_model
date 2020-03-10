import numpy as np 
import math

import matplotlib.pyplot as plt

from functools import partial 
from itertools import permutations, combinations
from typing import Callable, Dict, Tuple, List

from plotters import *

def get_transform(v1: np.array, v2: np.array) -> float:
    norm_1 = np.linalg.norm(v1)
    norm_2 = np.linalg.norm(v2)
    angle = math.acos(np.dot(v1, v2) / (norm_1 * norm_2))
    scale = norm_2 / norm_1

    if np.linalg.det([v1.astype(np.float32), v2.astype(np.float32)]) > 0:
        angle = -angle

    return angle, scale

def replication(formation: np.array, 
                q_0: np.array, q_1: np.array, 
                i: int=0, j: int=1) -> np.array:
    """Returns the replicated shape.

    This returns the shape P similar to formation such that
    P's ith and jth points are fixed to q_0 and q_1, respectively.

    Args:
        formation (np.array): formation to replicate
        q_0: Point to fix ith point in target formation to
        q_1: Point to fix jth point in target formation to
        i: index of point in target formation to fix to q_0
        j: index of point in target formation to fix to q_1

    Returns:
        np.array: The replicated shape
    """
    formation = np.copy(formation)
    q_0 = np.copy(q_0)
    q_1 = np.copy(q_1)

    assert(len(formation.shape) == 2)   # two-dimensional
    assert(formation.shape[1] == 2)     # collection of two-dimensional points
    assert(q_0.shape == (2,))           # two-dimensional point
    assert(q_1.shape == (2,))           # two-dimensional point

    formation -= (formation[i] - q_0)   # shift to i to q_0

    offset = np.copy(formation[i])
    formation -= offset     # temp shift i to origin
    q_1 -= offset

    theta, scale = get_transform(formation[j], q_1)

    c = math.cos(-theta)
    s = math.sin(-theta)

    formation = np.array(
        [
            formation[:,0] * c - formation[:,1] * s,
            formation[:,0] * s + formation[:,1] * c
        ]
    ).T * scale + offset

    return formation

def replication_machine_circles(formation: np.array,  
                                q_0: np.array, q_1: np.array, r: float,
                                i: int=0, j: int=1):
    T = replication(formation, q_0, q_1, i, j)
    radiuses = r * np.linalg.norm(T - q_0, axis=1) / np.linalg.norm(q_1 - q_0)
    return T, radiuses

def replication_spanner_circles(formation: np.array,  
                                q_0: np.array, q_1: np.array, r: float,
                                i: int=0, j: int=1):
    T = replication(formation, q_0, q_1, i, j)
    numerator = np.linalg.norm(T - q_0, axis=1) + np.linalg.norm(T - q_1, axis=1)
    denominator = np.linalg.norm(q_1 - q_0)
    radiuses = r * numerator / denominator
    return T, radiuses

def replication_machine(formation: np.array,  
                        q_0: np.array, q_1: np.array, r: float,
                        num: int, i: int=0, j: int=1):
    angles = np.linspace(0, 2*math.pi, num=num)
    q_1s = np.array([np.cos(angles) * r + q_1[0], np.sin(angles) * r + q_1[1]]).T
    fun = partial(replication, formation, q_0, i=i, j=j)
    return np.apply_along_axis(fun, axis=1, arr=q_1s)

def replication_spanner(formation: np.array,  
                        q_0: np.array, q_1: np.array, r: float,
                        num_0: int, num_1: int,
                        i: int=0, j: int=1):
    shapes = []
    for angle_0 in np.linspace(0, 2*math.pi, num=num_0):
        _q_0 = q_0 + r * np.array([math.cos(angle_0), math.sin(angle_0)])
        for angle_1 in np.linspace(0, 2*math.pi, num=num_1):
            _q_1 = q_1 + r * np.array([math.cos(angle_1), math.sin(angle_1)])
            shapes.append(replication(formation, _q_0, _q_1, i, j))
    return shapes

normalize = lambda x: x / np.linalg.norm(x, axis=1)[:, np.newaxis]
dists = lambda x, y: np.linalg.norm(x - y, axis=1)
get_sides = lambda x: np.linalg.norm(x - np.roll(x, 1, axis=0), axis=1)

def triangle_solution_no_perm(positions: np.array, formation: np.array):
    target = np.zeros_like(positions)
    repls = []
    norm_form = formation / get_sides(formation).sum()
    side_12 = get_sides(norm_form)[2] # Get opposite side
    radius = None
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        repl = replication(formation, positions[j], positions[k], j, k)
        repls.append(repl)
        direction = repl[i] - positions[i]
        direction = direction / np.linalg.norm(direction)
        if i == 0: # compute radius only once
            radius = np.linalg.norm(positions[0] - repl[0]) * side_12
        target[i] = positions[i] + direction * radius

    return target, radius# , repls

def triangle_solution(positions: np.array, formation: np.array):
    assert(positions.shape[0] == 3)
    assert(formation.shape[0] == 3)

    radius, perm, repl = None, None, None
    for _perm in map(list, permutations(range(formation.shape[0]))):
        _formation = formation[_perm]
        norm_form = _formation / get_sides(_formation).sum()
        side_12 = get_sides(norm_form)[2] # Get opposite side

        _repl = replication(_formation, q_0=positions[1], q_1=positions[2], i=1, j=2)
        _radius = np.linalg.norm(positions[0] - _repl[0]) * side_12
        if radius is None or _radius < radius:
            radius, perm, repl = _radius, _perm, _repl
    
    target, radius = triangle_solution_no_perm(positions, formation[perm])
    return target, radius, perm

  
def approximation(positions: np.array, formation: np.array): #, return_replications: bool = False):
    maxd, target, radius = None, None, None
    for _formation in permutations(formation):
        tri_target, _radius, _ = triangle_solution_no_perm(positions[:3], _formation[:3])
        _target = replication(_formation, tri_target[0], tri_target[1])
        distances = np.linalg.norm(_target - positions, axis=1)
        _maxd = distances.max()
        if maxd is None or _maxd < maxd:
            target = _target
            radius = _radius
            maxd = _maxd

    return target, radius
  

import matplotlib.pyplot as plt
from shapes import random_shape
from plotters import get_colors

if __name__ == '__main__':
    from optimization import convex_solution_no_perm
    p = np.array([
        [0.02394594, 0.84032009],
        [0.53647123, 0.58477441],
        [0.78407296, 0.02645303],
        [0.96061581, 0.75271376],
    ])
    s = np.array([ # formation
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])

    p_perm = [0,1,2]
    s_perm = [0,3,1]

    target, radius, _ = triangle_solution_no_perm(p[p_perm], s[s_perm])
    print(radius)

    _s = s[s_perm] - s[s_perm][0]
    opt_target, opt_radius = convex_solution_no_perm(p[p_perm], _s)
    print(opt_radius)


    fig, ax = plt.subplots()
    plt.gca().set_aspect('equal', adjustable='box')
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.25, 1.25)

    ax.scatter(p[:,0], p[:,1], marker='o')
    ax.scatter(target[:,0], target[:,1], marker='X')

    ax.scatter(opt_target[:,0], opt_target[:,1], c='b', marker='x')

    plt.show()