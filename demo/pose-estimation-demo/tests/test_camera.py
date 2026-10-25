from simulation.camera import Camera

def test_project():
    assert Camera().project([[0,0,8]]).tolist() == [[640.0,360.0]]
