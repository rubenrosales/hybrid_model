from formation import triangle_solution_no_perm, replication
from itertools import product, permutations, combinations

import numpy as np

def is_similar(shp, _shp):
    sides = np.linalg.norm(shp - np.roll(shp, 1, axis=0), axis=1)
    sides =  np.sort(sides) / np.sum(sides)
    _sides = np.linalg.norm(_shp - np.roll(_shp, 1, axis=0), axis=1)
    _sides = np.sort(_sides) / np.sum(_sides)
    return np.isclose(sides, _sides).all()

def solution(positions: np.array, formation: np.array) -> np.array:
    combos = lambda x: map(np.array, combinations(range(x), 3))
    n = len(positions)
    target, radius = None, None
    for i, j in product(combos(n), combos(n)):
        pos = positions[i]
        for perm in map(np.array, permutations(range(3))):
            tri_target, tri_radius = triangle_solution_no_perm(pos, formation[j[perm]])
                
            _target = replication(formation, tri_target[0], tri_target[1], j[perm][0], j[perm][1])

            tmp_radius = None
            for _tar in map(np.array, permutations(_target)):
                _radiuses = np.linalg.norm(positions - _tar, axis=1)
                _radius = np.max(_radiuses)
                if radius is None or _radius < radius:
                    target, radius = _tar, _radius
                if tmp_radius is None or _radius < tmp_radius:
                    tmp_radius = _radius
    return target

from shapes import random_shape
from optimization import convex_solution, convex_solution_no_perm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# if __name__ == '__main__':
#     n, m = 50, 20
#     data = np.zeros((n, m))
#     for i in range(3, n):
#         total = 0
#         for j in range(m):
#             positions, formation = random_shape(i), random_shape(i)
#             target, radius = convex_solution(positions, formation)
#             radiuses = np.linalg.norm(target - positions, axis=1)
#             num = len(radiuses[np.isclose(radiuses, radius)])
#             print(f'{i}\t{j}\t- {num}')
#             data[i,j] = num
#             np.savetxt('data.txt', data)

#         data[i] = total / m

#     fig, ax = plt.subplots()
#     ax.plot(data)
#     plt.show()

import matplotlib.pyplot as plt
from shapes import random_shape
from optimization import convex_solution, convex_solution_no_perm
from plotters import get_colors

from matplotlib.widgets import Slider, Button, RadioButtons

if __name__ == '__main__':  
    formation = np.array([ # formation
        [0.0, 0.0],
        [0.1, 0.8],
        [0.2, 0.9],
        # [1.0, 1.0],
        # [1.0, 2.0],
        # [0.0, 2.0],
    ])
    n = len(formation)
    colors = get_colors(n) # purple, blue, green, red
    positions  = np.array([ # formation
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 1.0],
        # [1.5, 1.0], # just change
        # [1.0, 2.5],
        # [0.0, 2.0],
    ])
    # positions = random_shape(n, 0.0, 1.0)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    # ax.set_xlim(-0.25, 1.25)
    # ax.set_ylim(-0.25, 1.25)
    plt.gca().set_aspect('equal', adjustable='box')
    axbutton = plt.axes([0.25, 0.15, 0.50, 0.03], facecolor='lightgoldenrodyellow')
    bnew = Button(axbutton, 'New')
    
    # Initial
    q = solution(positions, formation)
    scatter_p = ax.scatter(positions[:,0], positions[:,1], c=colors, marker='o')
    scatter_sol = ax.scatter(q[:,0], q[:,1], c=colors, marker='s')
    print(f'Ours: {is_similar(q, formation)}')

    q, radius = convex_solution_no_perm(positions, formation)
    scatter_opt = ax.scatter(q[:,0], q[:,1], c=colors, marker='x')

    print(f'Theirs: {is_similar(q, formation)}')

    def update(event=None):
        p = random_shape(n, 0.0, 1.0)
        # print('pos:', p)
        scatter_p.set_offsets(p)

        formation = random_shape(n, 0.0, 1.0)

        q_sol = solution(p, formation)
        # print('sol:', np.linalg.norm(p - q_sol, axis=1))
        scatter_sol.set_offsets(q_sol)

        q_opt, _, _ = convex_solution(p, formation)
        r_opt = np.linalg.norm(p - q_opt, axis=1)

        print(f'Ours: {is_similar(q_sol, formation)}')
        print(f'Theirs: {is_similar(q_opt, formation)}')
        # print('opt:', r_opt)

        scatter_opt.set_offsets(q_opt)
        fig.canvas.draw_idle()

    bnew.on_clicked(update)
    plt.show()