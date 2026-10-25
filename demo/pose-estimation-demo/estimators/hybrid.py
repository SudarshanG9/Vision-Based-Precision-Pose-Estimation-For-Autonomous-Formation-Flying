from .particle_filter_improved import ImprovedParticleFilter
from .geometric_refinement import refine_translation

class HybridEstimator:
    def estimate(self, observation):
        return refine_translation(ImprovedParticleFilter().estimate(observation), observation)
