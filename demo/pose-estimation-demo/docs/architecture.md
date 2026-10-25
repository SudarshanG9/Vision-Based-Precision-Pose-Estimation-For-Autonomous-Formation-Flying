
# System Architecture

## Project

Vision-Based Precision Pose Estimation for Autonomous
Formation Flying

## 1. Objective

Develop a reproducible research platform that demonstrates
vision-based relative aircraft pose estimation using:

1. Synthetic aircraft observations
2. Artificial neural network coarse pose classification
3. Bayesian particle filtering
4. Geometric pose refinement
5. Quantitative benchmarking
6. An interactive telemetry dashboard

The system implements and evaluates the original research
methodology and clearly distinguishes proposed improvements.

## 2. Operating Model

The project is an open-loop estimation visualizer.

The simulator generates predefined leader/follower flight
trajectories and synthetic camera observations.

Estimation algorithms process these observations without
receiving the simulator's ground-truth pose.

The evaluation engine compares estimates with ground truth.

The dashboard visualizes stored results using synchronized
playback rather than directly controlling aircraft motion.

This is not an autonomous flight controller.

## 3. System Layers

### Layer A: Simulation

Directory: simulation/

Responsibilities:
- Aircraft structural geometry
- Coordinate transformations
- Leader/follower trajectories
- Camera projection
- Synthetic image rendering
- Landmark visibility
- Observation noise
- Scenario configuration

Outputs:
- Ground-truth pose trajectory
- Synthetic camera observations
- Optional rendered images

### Layer B: Learned Models

Directory: models/

Responsibilities:
- Landmark detection
- Landmark preprocessing
- Coarse pose classification
- Pose label encoding and decoding

Paper-faithful classifier:
- Input dimension: 42
- Hidden layers: 100, 100
- Output classes: 4800
- Hidden activation: ReLU
- Final probabilities: Softmax

The original classifier covers:
- Relative forward position
- Relative lateral position
- Relative vertical position
- Roll

It does not directly predict all six continuous pose coordinates.

### Layer C: Estimation

Directory: estimators/

Responsibilities:
- Original paper's particle filter
- Improved particle filter
- PnP estimation
- Geometric refinement
- Optional hybrid pose estimation

Every continuous estimator must implement the PoseEstimator
contract defined in estimators/interfaces.py.

Ground truth must not be accessible from estimators/.

### Layer D: Evaluation

Directory: evaluation/

Responsibilities:
- Position error
- Rotation error
- Reprojection error
- Classification accuracy
- Runtime statistics
- Particle degeneracy diagnostics
- Experiment comparison
- Ablation studies

Evaluation must operate on fixed experiment records.

### Layer E: Experiment Services

Directory: services/

Responsibilities:
- Load scenario configuration
- Load trained model checkpoints
- Construct independent estimators
- Execute experiment runs
- Build telemetry outputs
- Store experiment metadata and results

The experiment runner must support execution without
FastAPI, Streamlit, or Next.js.

### Layer F: API

Directory: api/

Framework: FastAPI

Responsibilities:
- Accept validated experiment requests
- Run or queue experiment execution
- Expose experiment status
- Return telemetry
- Return benchmark summaries

API routes must not contain pose estimation mathematics.

### Layer G: Frontend

Directory: frontend/

Framework: Next.js with TypeScript

Responsibilities:
- Experiment configuration
- Trajectory selection
- Playback controls
- Synthetic camera tracker
- 3D trajectory visualization
- Error telemetry
- Estimator comparison
- Experimental diagnostics

The frontend is read-only with respect to numerical
estimation results.

It may request new computations but must not alter
previously saved results.

## 4. Data Flow

Scenario configuration
    |
    v
Flight simulation
    |
    +-----> Ground truth storage
    |
    v
Camera observations
    |
    +-----> Landmark detector (optional)
    |
    v
Coarse ANN classifier
    |
    +-----> ANN classification results
    |
    v
Estimator execution
    |
    +-----> Original Particle Filter
    +-----> Improved Particle Filter
    +-----> PnP / geometric refinement
    +-----> Hybrid estimator (optional)
    |
    v
Evaluation against ground truth
    |
    v
Experiment artifact storage
    |
    v
FastAPI telemetry service
    |
    v
Next.js dashboard

All estimation branches must be evaluated using matching
input observations and timestamps.

## 5. Repository Ownership

simulation/
    Physical and camera simulation only.

models/
    Neural network definitions and preprocessing.

estimators/
    Continuous pose estimation algorithms.

training/
    Dataset preparation and model training.

evaluation/
    Quantitative research evaluation.

services/
    Experiment orchestration.

api/
    Network-facing request and response layer.

frontend/
    Browser-facing visual interface.

tests/
    Automated verification.

configs/
    Declarative experiment configuration.

assets/
    Aircraft geometry and fixed visualization metadata.

checkpoints/
    Trained model artifacts.

outputs/
    Experimental results and benchmark reports.

docs/
    Architecture, contracts and reproducibility.

## 6. Execution Strategy

Experiments must be executable from Python without
starting the dashboard.

FastAPI executes or schedules an experiment and returns
an experiment identifier.

The frontend polls for experiment completion and then
loads its telemetry.

Playback operates on immutable experiment results.

Parameter changes trigger a new experiment.

## 7. Performance Strategy

Simulation and estimation are independent of browser
rendering.

Particle filtering uses vectorized numerical operations
where practical.

Particle clouds may be downsampled for visualization.

The numerical estimator must process the full particle
population even when fewer particles are rendered.

Model checkpoints should be loaded once per process
where practical.

Computation runtime and visualization FPS are recorded
separately.

## 8. Research Integrity

The original research baseline and modified estimators
must remain separate implementations or configurations.

Unreported paper details must be documented as
reproduction assumptions.

Reported paper results must not be treated as automatically
reproduced results.

All evaluation plots must use actual experiment outputs.

No artificial improvements may be inserted into telemetry.

## 9. Integration Requirements

1. Shared interfaces are defined in estimators/interfaces.py.
2. Coordinate conventions follow docs/coordinate_conventions.md.
3. All estimators use common camera observations.
4. Random seeds are recorded.
5. Ground truth is isolated from inference.
6. Model checkpoints are versioned.
7. Saved experiments are immutable.
8. API outputs follow docs/api_contract.md.
9. Every development phase has a consolidated test gate.
