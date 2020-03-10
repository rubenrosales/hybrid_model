import numpy as np 
import math 

def random_shape(num: int, low: float = -1.0, high: float = 1.0) -> np.ndarray:
    return np.random.random_sample((num, 2)).astype(np.longdouble) * (high - low) + low

def regular_shape(num: int) -> np.ndarray:
    return np.array(
        [
            [math.cos(2 * math.pi * i / num), math.sin(2 * math.pi * i / num)]
            for i in range(num)
        ]
    ).astype(np.longdouble)

class Triangle:
    EQUILATERAL = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, math.sqrt(3) / 2]
    ]).astype(np.longdouble)
    
    @staticmethod
    def right(angle: float, deg: float = False) -> np.ndarray:
        if deg:
            angle = angle * math.pi / 180
        return np.array([
            [0.0, 0.0],
            [1.0, 0.0], 
            [1.0, math.tan(angle)]
        ]).astype(np.longdouble)

    @staticmethod
    def isosoles(height: float) -> np.ndarray:
        return np.array([
            [-0.5, 0.0],
            [0.0, height],
            [0.5, 0.0],
        ]).astype(np.longdouble)

