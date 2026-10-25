import numpy as np

def linear_trajectory(start, velocity, steps):
    start=np.asarray(start,float); velocity=np.asarray(velocity,float)
    return np.array([start+i*velocity for i in range(steps)])
