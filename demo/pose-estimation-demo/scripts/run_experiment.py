import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.experiment_runner import ExperimentRunner

if __name__ == '__main__':
    print(ExperimentRunner().run())
