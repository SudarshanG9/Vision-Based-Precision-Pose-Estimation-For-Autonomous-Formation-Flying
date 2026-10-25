from simulation.trajectory import linear_trajectory

def test_trajectory(): assert len(linear_trajectory([0,0,0],[1,0,0],4)) == 4
