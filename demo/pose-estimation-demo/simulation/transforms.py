import numpy as np

def compose_transform(rotation=None, translation=None):
    T=np.eye(4); T[:3,:3]=np.eye(3) if rotation is None else np.asarray(rotation); T[:3,3]=0 if translation is None else translation; return T

def transform_points(points, transform):
    p=np.asarray(points,float); return (np.c_[p,np.ones(len(p))] @ np.asarray(transform).T)[:,:3]
