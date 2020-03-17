import random
import math
import numpy as np

def polar(z):
    a= z.real
    b= z.imag
    r = math.hypot(a,b)
    x = a / (math.sqrt(a**2 + b **2))
    theta = math.acos(x)
    return r,theta 

def normalize(a):
    return (a - np.min(a)) / (np.max(a) - np.min(a))

def generateTimeSeries(N, i):
    complex_numbers_time_series = []
    time = 1
    lower_bound = int(1 * i)
    upper_bound = int(10* i)
    for _ in range(N):
        rand_complex_number = complex(random.randint(lower_bound, upper_bound), random.randint(lower_bound, upper_bound))
        time = random.randint(int(time) + 1, int(time) + 10) + (10 - random.uniform(0., 10.))
        time = time + (10 - random.uniform(0., 10.))        
        
        complex_numbers_time_series.append((time, rand_complex_number))

    return complex_numbers_time_series


def processRawMatrix(theta, time, r):
    theta = np.asarray(normalize(theta)).reshape(1, -1)
    r = np.asarray(normalize(r)).reshape(1, -1)
    time = np.asarray(time).reshape(1, -1)

    return theta, time, r

def printHelper(theta, r, time, quantiles):
    for i in range(len(theta)):
        for j in range(len(theta[i])):
            print("Data Point:", theta[i][j], r[i][j], time[i][j])
            print("Quantile p,q,r:", quantile_bins[j])

def printPolar(theta, r):
    theta *= 2 * np.pi
    ax = plt.subplot(projection='polar')
    ax.set_rticks(r_bins.flatten())  # set radial ticks based on quantiles

    ax.plot(theta, r, 'k.')

    # ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
    ax.set_rlabel_position(22.5)  # get radial labels away from plotted line
    ax.grid(True)

    # ax.set_title("Not scaled yet", va='bottom')
    plt.show()