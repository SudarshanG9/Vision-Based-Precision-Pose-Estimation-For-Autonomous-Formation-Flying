from estimators.geometric_refinement import refine_translation

def test_refinement(): assert refine_translation([0],[1],20)[0] > .99
