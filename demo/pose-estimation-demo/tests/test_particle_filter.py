from estimators.particle_filter import ParticleFilter

def test_filter(): assert ParticleFilter().estimate([1,2]).tolist() == [1,2]
