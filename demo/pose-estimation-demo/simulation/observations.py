import numpy as np

def add_pixel_noise(observations, std=0.0, seed=0):
    rng=np.random.default_rng(seed); return np.asarray(observations)+rng.normal(0,std,np.asarray(observations).shape)
