from services.experiment_runner import ExperimentRunner

def test_runner(): assert len(ExperimentRunner().run(3)) == 3
