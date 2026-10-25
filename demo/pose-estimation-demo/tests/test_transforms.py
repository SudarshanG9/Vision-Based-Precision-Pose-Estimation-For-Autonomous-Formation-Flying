from simulation.transforms import transform_points, compose_transform
import numpy as np

def test_transform():
    assert np.allclose(transform_points([[1,2,3]], compose_transform(translation=[1,0,0])), [[2,2,3]])
