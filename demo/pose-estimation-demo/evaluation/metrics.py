import numpy as np

def mean_position_error(expected, actual):
    return float(np.mean(np.linalg.norm(np.asarray(expected)-np.asarray(actual),axis=-1)))

def rmse(values):
    v=np.asarray(values,float); return float(np.sqrt(np.mean(v*v)))
