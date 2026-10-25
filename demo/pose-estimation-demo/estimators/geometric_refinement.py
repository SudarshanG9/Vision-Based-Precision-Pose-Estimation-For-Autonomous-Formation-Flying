import numpy as np

def refine_translation(initial, target, steps=3):
    x=np.asarray(initial,float); y=np.asarray(target,float)
    for _ in range(steps): x += (y-x)*0.5
    return x
