from dataclasses import dataclass
from simulation.scenarios import scenario
from estimators.hybrid import HybridEstimator

@dataclass
class ExperimentRunner:
    def run(self, steps=10):
        estimator=HybridEstimator(); states=scenario(steps=steps)
        return [{'step':i,'estimate':estimator.estimate(state).tolist()} for i,state in enumerate(states)]
