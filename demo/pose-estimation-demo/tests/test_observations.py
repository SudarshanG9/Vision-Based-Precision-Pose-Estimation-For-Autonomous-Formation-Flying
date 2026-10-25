from simulation.observations import add_pixel_noise

def test_observations(): assert add_pixel_noise([[1,2]],0).tolist() == [[1,2]]
