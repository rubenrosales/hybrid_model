
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np 

from typing import Iterable

def get_colors(num: int) -> Iterable[str]:
    return cm.rainbow(np.linspace(0, 1, num))

def plot_shape(ax: plt.Axes, shape: np.array, color: str = 'black', marker: str = 'o') -> None:
    ax.scatter(x=shape[:,0], y=shape[:,1], color=color, marker=marker)

def plot_shapes(ax: plt.Axes, *shapes: np.array, marker: str = 'o') -> None:
    colors = get_colors(len(shapes))
    for shape, c in zip(shapes, colors):
        plot_shape(ax, shape, color=c, marker=marker)

def plot_polygon(ax: plt.Axes, polygon: np.array, color: str = 'black') -> None:
    ax.add_patch(plt.Polygon(polygon, color=color, fill=False))

def plot_polygons(ax: plt.Axes, *polygons: np.array) -> None:
    colors = get_colors(len(polygons))
    for polygon, color in zip(polygons, colors):
        plot_polygon(ax, polygon, color=color)

def plot_circle(ax: plt.Axes, point: np.array, radius: float, color: str = 'black') -> None:
    ax.add_patch(plt.Circle(point, radius, color=color, fill=False))

def plot_circles(ax: plt.Axes, points: np.array, radiuses: np.array) -> None:
    colors = get_colors(len(points))
    for point, radius, color in zip(points, radiuses, colors):
        plot_circle(ax, point, radius, color=color)