from dataclasses import dataclass
import numpy as np

@dataclass
class ParticleFilter:
    particles: int = 256
    state: np.ndarray | None = None
    def estimate(self, observation):
        self.state=np.asarray(observation,float); return self.state.copy()
