import matplotlib.pyplot as plt
import numpy as np
import math
from itertools import combinations, product
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.collections import PatchCollection

from multiprocessing import Pool
from functools import partial
import signal

from formation import (
    replication,
    replication_machine,
    replication_machine_circles,
    replication_spanner,
    replication_spanner_circles,
    triangle_solution,
    triangle_solution_no_perm,
    approximation,
    get_sides
)
from geometry import circle_intersection
from plotters import *
from shapes import Triangle, random_shape, regular_shape
from typing import Tuple, List, Optional

# https://stackoverflow.com/questions/3252194/numpy-and-line-intersections/3252222
def get_intersect(a1, a2, b1, b2):
    """
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1, a2, b1, b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1))))    # h for homogeneous
    l1 = np.cross(h[0], h[1])              # get first line
    l2 = np.cross(h[2], h[3])              # get second line
    x, y, z = np.cross(l1, l2)             # point of intersection
    if z == 0:                             # lines are parallel
        return np.array([float('inf'), float('inf')])
    return np.array([x / z, y / z])


def test_same_solution(positions: Optional[np.array] = None,
                       formation: Optional[np.array] = None,
                       speed: float = 0.002):
    if positions is None:
        positions = random_shape(3)
        print(positions)
    if formation is None:
        formation = regular_shape(3)

    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    ax.axis('equal')
    
    axslider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    axbutton = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    bnew = Button(axbutton, 'New')
    speed_slider = Slider(axslider, 'Speed', 0, 0.05, valinit=speed)

    # Intial computation
    target, radius = triangle_solution(positions, formation)
    directions = (target - positions) / np.linalg.norm(target -
                                                       positions, axis=1)[:, np.newaxis] * speed
    focuses = []
    focus = get_intersect(positions[0], target[0], positions[1], target[1])
    focuses.append(focus)

    # Create scatter plots
    pos_scatter = ax.scatter(positions[:, 0], positions[:, 1], c='red')
    tar_scatter = ax.scatter(target[:, 0], target[:, 1], c='blue')
    foc_scatter = ax.scatter([focus[0]], [focus[1]], c='green')

    def init():
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        return pos_scatter, tar_scatter, foc_scatter

    def new_positions(event):
        nonlocal positions, focuses, target
        positions = random_shape(3)
        target, radius = triangle_solution(positions, formation)
        focus = get_intersect(positions[0], target[0], positions[1], target[1])
        focuses = [focus]

    bnew.on_clicked(new_positions)

    def new_speed(event):
        nonlocal speed, directions
        speed = speed_slider.val
        directions = np.linalg.norm(directions, axis=0) * speed


    speed_slider.on_changed(new_speed)

    def update(frame):  # Gets called for every frame in animation
        nonlocal target, positions, directions, speed
        _target, _ = triangle_solution(positions, formation)

        # switch direction only when a new target is found
        if not np.allclose(_target, target):
            target = _target
            directions = (target - positions) / np.linalg.norm(target - positions, axis=1)[:, np.newaxis] * speed

            focus = get_intersect(positions[0], target[0], positions[1], target[1])
            # foc_scatter.set_offsets([focus])
            focuses.append(focus)
            foc_scatter.set_offsets(focuses)

        positions += directions
        pos_scatter.set_offsets(positions)
        tar_scatter.set_offsets(target)

        return pos_scatter, tar_scatter, foc_scatter

    ani = FuncAnimation(fig, update, frames=None,
                        init_func=init, blit=True, repeat=False, interval=1, cache_frame_data=False)
    plt.show()

def random_positions(plot_radius: bool = False):
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    ax.axis('equal')
    
    axslider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    axbutton = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    bnew = Button(axbutton, 'New')
    num_slider = Slider(axslider, 'Robots', 3, 7, valinit=3, valstep=1)

    # Initalize
    formation = regular_shape(int(num_slider.val))
    positions = random_shape(len(formation))
    target, radius = approximation(positions, formation)

    colors = get_colors(len(formation))
    arr = np.arange(positions.shape[0])
    srcs = ax.scatter(positions[:,0], positions[:,1], marker='o', color=colors)
    dsts = ax.scatter(target[:,0], target[:,1], marker='x', color=colors)

    def get_circles(positions: np.array, radius: float) -> List[plt.Circle]:
        return [plt.Circle(position, radius) for position in positions]

    circles = PatchCollection(get_circles(positions, radius), alpha=0.4)
    circles.set_color(colors)
    ax.add_collection(circles)

    def update(event):
        formation = regular_shape(int(num_slider.val))
        positions = random_shape(len(formation))
        target, radius = approximation(positions, formation)
        srcs.set_offsets(positions)
        dsts.set_offsets(target)
        circles.set_paths(get_circles(positions, radius))

        colors = get_colors(positions.shape[0])
        srcs.set_color(colors)
        dsts.set_color(colors)
        circles.set_color(colors)

        # Set Axis Scale
        points = np.concatenate([positions, target])
        llim, hlim = points.min() - radius, points.max() + radius
        ax.set_xlim(llim, hlim)
        ax.set_ylim(llim, hlim)
        ax.autoscale_view()
        fig.canvas.draw_idle()

    bnew.on_clicked(update)
    num_slider.on_changed(update)

    ax.autoscale_view()
    plt.show()


def test_formation(
        positions: np.array,
        formation: np.array,
        plot_replications: bool = False,
        plot_radius: bool = False) -> None:
    fig, ax = plt.subplots()
    ax.axis('equal')

    number = len(positions)
    assert(number == len(formation))
    colors = get_colors(number)

    repls = None
    if number == 3:
        target, radius = triangle_solution(positions, formation)
        print(f'radius: {radius}')
    else:
        positions, target, radius = approximation(positions, formation)
        plot_replications = False

    for position, target, color in zip(positions, target, colors):
        plot_shape(ax, np.array([position]), color=color)
        plot_shape(ax, np.array([target]), marker='x', color=color)

    if plot_replications:
        for repl, color in zip(repls, colors):
            plot_polygon(ax, repl, color=color)
    if plot_radius:
        for position, color in zip(positions, colors):
            plot_circle(ax, position, radius, color=color)

    ax.autoscale_view()
    plt.show()


def test_replications(
        formation: Optional[np.array] = None,
        radius_range: Tuple[float, float] = (0.0, 0.5)) -> None:
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.3)
    ax.axis('equal')

    number = 3
    positions = random_shape(number)
    if not formation:
        formation = regular_shape(number)


    init_radius = radius_range[1]
    ax_radius = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    ax_button = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    ax_number = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    num_slider = Slider(ax_number, 'Robots', 3, 7, valinit=3, valstep=1)
    s_radius = Slider(ax_radius, 'Radius', *radius_range, valinit=init_radius)
    b_random = Button(ax_button, 'New')
    scatter_positions = ax.scatter(x=positions[:, 0], y=positions[:, 1], color='black')


    def get_shapes(radius):
        c_positions = list(combinations(range(len(positions)), 2))
        c_formations = list(combinations(range(len(formation)), 2))
        combs = list(product(c_positions, c_formations))
        patches = []
        colors = get_colors(len(combs))
        for color, ((i_pos1, i_pos2), (i_for1, i_for2) )in zip(colors, combs):
            repl, radiuses = replication_spanner_circles(
                formation,
                positions[i_pos1], positions[i_pos2],
                radius,
                *sorted([i_for1, i_for2])
            )

            patches.append(plt.Polygon(repl, color=color, fill=False))
            for i, _radius in enumerate(radiuses):
                # if i in (i_pos1, i_pos2):
                #     continue
                patches.append(plt.Circle(repl[i], _radius, color=color, fill=False))

        for position in positions:
            patches.append(plt.Circle(position, radius, color='black', fill=False))

        return patches

    collection = PatchCollection(get_shapes(init_radius), match_original=True)
    ax.add_collection(collection)

    def update(val=None):
        collection.set_paths(get_shapes(s_radius.val))
        # collection.set_edgecolor(get_colors(len(positions) * (len(positions) - 1)))
        # collection.set_array(np.array(list(range(len(positions)))))
        
        fig.canvas.draw_idle()

    def new_positions(val=None):
        nonlocal positions, formation
        positions = random_shape(number)
        formation = regular_shape(number)
        scatter_positions.set_offsets(positions)

        # Set Axis Scale
        points = np.concatenate([positions, formation])
        llim, hlim = points.min() - 1, points.max() + 1
        ax.set_xlim(llim, hlim)
        ax.set_ylim(llim, hlim)
        ax.autoscale_view()

        update()

    def update_number(val=None):
        nonlocal number
        number = int(num_slider.val)
        new_positions()

    s_radius.on_changed(update)
    b_random.on_clicked(new_positions)
    num_slider.on_changed(update_number)

    ax.autoscale_view()
    plt.show()


def test_four():
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.axis('equal')

    ax_radius = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    s_radius = Slider(ax_radius, 'Radius', 0, 1, valinit=0.1)

    formation = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ]).astype(float)

    positions = np.array([
        [0, 0],
        [1, 0],
        [1.1, 1.2],
        [0.3, 1.1]
    ]).astype(float)

    points, radiuses = replication_spanner_circles(formation, positions[0], positions[1], r=s_radius.val)
    circles = [
        plt.Circle(point, r, fill=False)
        for (point, r) in zip(points, radiuses)
    ] + [
        plt.Circle(point, s_radius.val, fill=False)
        for point in positions
    ]

    collection = PatchCollection(circles, match_original=True)
    pos_scatter = ax.scatter(positions[:, 0], positions[:, 1])
    ax.add_collection(collection)

    ints_2 = circle_intersection((positions[2][0], positions[2][1], s_radius.val), (points[2][0], points[2][1], radiuses[3]))
    ints_3 = circle_intersection((positions[3][0], positions[3][1], s_radius.val), (points[3][0], points[3][1], radiuses[3]))

    if ints_2 is None:
        ints_2 = np.array([[0, 0], [0, 0]])
    if ints_3 is None:
        ints_3 = np.array([[0, 0], [0, 0]])

    ints_2_scatter = ax.scatter(ints_2[:,0], ints_2[:,1])
    ints_3_scatter = ax.scatter(ints_3[:,0], ints_3[:,1])

    def update(val=None):
        points, radiuses = replication_spanner_circles(formation, positions[0], positions[1], r=s_radius.val)
        circles = [
            plt.Circle(point, r, fill=False)
            for (point, r) in zip(points, radiuses)
        ] + [
            plt.Circle(point, s_radius.val, fill=False)
            for point in positions
        ]
        collection.set_paths(circles)
        
        ints_2 = circle_intersection((positions[2][0], positions[2][1], s_radius.val), (points[2][0], points[2][1], radiuses[3]))
        ints_3 = circle_intersection((positions[3][0], positions[3][1], s_radius.val), (points[3][0], points[3][1], radiuses[3]))

        if ints_2 is None:
            ints_2 = np.array([[0, 0], [0, 0]])
        if ints_3 is None:
            ints_3 = np.array([[0, 0], [0, 0]])
            
        ints_2_scatter.set_offsets(ints_2)
        ints_3_scatter.set_offsets(ints_3)

        # max_radius = radiuses.max()

        # points = np.concatenate([positions, points])
        # llim, hlim = points.min() - max_radius, points.max() + max_radius
        # ax.set_xlim(llim, hlim)
        # ax.set_ylim(llim, hlim)
        # ax.autoscale_view()
        fig.canvas.draw_idle()
        
    s_radius.on_changed(update)
    
    # ax.autoscale_view()
    plt.show()

def show_symmetric_problem():
    formation = np.array([    
        [0.0, 0.0],
        [1.0, 0.0], 
        [1.0, 1.0]
    ])
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = Pool()
    signal.signal(signal.SIGINT, original_sigint_handler)
    try:
        res = pool.map_async(
            partial(test_formation, formation=formation),
            map(partial(np.roll, Triangle.EQUILATERAL, axis=0), range(3))
        )
        res.get(99999) # Without the timeout this blocking call ignores all signals.
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
    else:
        print("Normal termination")
        pool.close()
    pool.join()


def test_four_circles():
    """Tests the conjecture that at most four must travel the solution"""
    pos = random_shape(4, 0.0, 1.0)
    shp = random_shape(4, 0.0, 1.0)
    shp = regular_shape(4)

    rep, radiuses = replication_spanner_circles(shp, pos[0], pos[1], 0.1)

    ax: plt.Axes
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    ax.scatter(*zip(*pos), color='black')
    ax.scatter(*zip(*rep), color='blue')

    patches = [plt.Circle(c, radius) for c, radius in zip(rep[2:], radiuses[2:])]
    ax.add_collection(PatchCollection(patches, alpha=0.4))

    plt.show()

if __name__ == '__main__':
    test_four_circles()


toTree = foldl (flip insert) Tip