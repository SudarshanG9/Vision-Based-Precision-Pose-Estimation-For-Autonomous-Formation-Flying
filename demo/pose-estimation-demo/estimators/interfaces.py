from typing import Protocol

class Estimator(Protocol):
    def estimate(self, observation): ...
