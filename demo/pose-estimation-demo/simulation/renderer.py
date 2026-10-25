import numpy as np

def render(points, camera):
    return camera.project(points)
