from cvxopt import matrix, solvers
from math import sqrt 
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, combinations, chain
from plotters import get_colors

from typing import List

from formation import triangle_solution, triangle_solution_no_perm, replication
from shapes import random_shape, regular_shape, Triangle
solvers.options['show_progress'] = False

def print_arrays(**kwargs):
    name: str
    value: np.array
    for name, value in kwargs.items():
        print(f'{name} {value.shape}')
        print(value)
        print()

def format_eqs(lefts: List[List[float]], right: List[float]):
    *xs, cs = zip(*([right] + lefts))    
    mat = []
    for elems in xs:
        mat.append([float(-elem) for elem in elems])

    return matrix(mat), matrix(cs)

import math
def convex_solution_no_perm(positions: np.array, formation: np.array, verbose: bool = False) -> np.array:
    positions = np.copy(positions)
    formation = np.copy(formation) - formation[0] # adjust formation to origin
    angle = -math.atan2(formation[1][1], formation[1][0])
    xs, ys = formation[:,0].copy(), formation[:,1].copy()
    formation[:,0] = math.cos(angle) * xs - math.sin(angle) * ys
    formation[:,1] = math.sin(angle) * xs + math.cos(angle) * ys

    # assert((formation[0] == np.array([0.0, 0.0])).all())
    assert(positions.shape == formation.shape)

    s2norm = np.linalg.norm(formation[1])
    n = len(formation)

    # | q_i.x - p_i.x |    
    # |               |     <= r
    # | q_i.y - p_i.y |_2 
    # 
    # such that 
    # Aq = 0    <---    this says the shape is similar

    Gs = []
    hs = []

    eqs = np.append(np.eye(2*n), np.zeros((2*n, 2)), axis=1)
    eqs[:,-1][::2] = -positions[:,0]
    eqs[:,-1][1::2] = -positions[:,1]

    eq_r = np.zeros(2*n + 2)
    eq_r[-2] = 1.0
    for i in range(n):
        G, h = format_eqs(
            [eqs[2*i].tolist(), eqs[2*i+1].tolist()],
            eq_r
        )
        Gs.append(G)
        hs.append(h)

    # A has 2 (n-2) equations with 2n + 1 variables 
    A = np.zeros((2*(n-2), 2*n+1), dtype=float)
    b = np.zeros(2*(n-2), dtype=float)
    for i in range(2, n):
        x_idx = 2*i 
        A[x_idx - 4][0] = formation[i][0] - s2norm      # q_1.x
        A[x_idx - 4][1] = -formation[i][1]              # q_1.y
        A[x_idx - 4][2] = -formation[i][0]              # q_2.x
        A[x_idx - 4][3] = formation[i][1]               # q_2.y

        A[x_idx - 4][x_idx] = s2norm                    # q_i.x

        y_idx = x_idx + 1
        A[y_idx - 4][0] = formation[i][1]               # q_1.x
        A[y_idx - 4][1] = formation[i][0] - s2norm      # q_1.y
        A[y_idx - 4][2] = -formation[i][1]              # q_2.x
        A[y_idx - 4][3] = -formation[i][0]              # q_2.y

        A[y_idx - 4][y_idx] = s2norm                    # q_i.y

    # c is what we are trying to minimize
    #  we are trying to minimize t (the last element of (q, t))
    c = np.zeros(2*n+1)
    c[-1] = 1

    if verbose:
        print_arrays(A=A, b=b, **dict({f'G{i}': G for i, G in enumerate(Gq)}), h=h, c=c)

    # Shape/data-type conversions for optimization function
    c = matrix(c.tolist())
    A = matrix(A.T.tolist())
    b = matrix([b.tolist()])

    sol = solvers.socp(c, A=A, b=b, Gq=Gs, hq=hs) # second-order cone programming
    return np.array(sol['x'][:-1]).reshape((n, 2)), sol['x'][-1]

def convex_solution(positions: np.array, formation: np.array, verbose: bool = False) -> np.array:
    positions = np.copy(positions)
    formation = np.copy(formation) - formation[0]
    ran = np.arange(len(positions))
    perm, target, radius = ran.copy(), None, None
    for _perm in map(np.array, permutations(ran)):
        if verbose:
            print(f'PERM: [{", ".join(map(str, _perm))}]')
        _target, _radius = convex_solution_no_perm(positions[_perm], formation, verbose=verbose)
        if radius is None or _radius < radius:
            perm[_perm], target, radius = ran.copy(), _target, _radius

    return target[perm], radius, perm

def is_similar(shp, _shp):
    _, radius, _ = convex_solution(shp, _shp)
    return np.isclose(radius, 0.0)

def test_sub_solutions(n=3):
    # Set up Plots
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_aspect('equal', 'box')
    ax2.set_aspect('equal', 'box')
    colors = get_colors(n)

    i = 0
    while True:
        i += 1
        pos = random_shape(n, 0.0, 1.0)
        shp = random_shape(n, 0.0, 1.0)
        target, radius, perm = convex_solution(pos, shp)
        idx = np.isclose(np.linalg.norm(target - pos, axis=1), radius)
        print(f'{i}. Critical Robots: {pos[idx].shape[0]}')
        if n < 6 or pos[idx].shape[0] == 5:
            break

    print(f'CVX: {radius:.2f} - {np.linalg.norm(target - pos, axis=1)}')
    
    if pos[idx].shape[0] == 5:
        _shp = shp[perm][idx]
        for _comb in map(np.array, combinations(list(range(5)), r=4)):
            _target, _radius = convex_solution_no_perm(pos[idx][_comb], _shp[_comb])
            ax1.scatter(
                *zip(*_target), 
                color=colors[idx][_comb], 
                marker='s', 
                alpha=0.5
            )

    if n == 3:
        t_target, t_radius, perm = triangle_solution(pos, shp)
        print(f'TRI: {t_radius:.2f} - {np.linalg.norm(t_target - pos, axis=1)}')

    # Plot Data
    ax1.scatter(*zip(*pos), color=colors)
    ax1.scatter(*zip(*target), color=colors, marker='x')
    
    if n == 3:
        ax1.scatter(*zip(*t_target), color=colors, marker='.')

    ax2.scatter(*zip(*shp))
    plt.show()

if __name__ == '__main__':
    test_sub_solutions(6)
