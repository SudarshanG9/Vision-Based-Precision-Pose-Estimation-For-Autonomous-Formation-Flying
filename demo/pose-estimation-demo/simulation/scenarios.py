from .trajectory import linear_trajectory

def scenario(name='nominal', steps=10):
    return linear_trajectory([0,0,8], [0.02,0,0], steps)
