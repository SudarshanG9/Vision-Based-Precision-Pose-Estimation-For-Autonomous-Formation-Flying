import numpy as np

def normalize_keypoints(points, width, height):
    p=np.asarray(points,float).copy(); p[...,0]/=width; p[...,1]/=height; return p
