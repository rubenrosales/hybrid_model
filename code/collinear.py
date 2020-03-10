import numpy as np
from shapes import random_shape

def random_example():
    points = np.random.random_sample((3, 2))
    speeds = np.random.random_sample((3, 2))

    speeds = np.linalg.norm(speeds, axis=0) / 1e-3
    
    positions = np.arange(1, 1001)[:,np.newaxis] * speeds 

    print(positions)

if __name__ == '__main__':
    random_example()