
# Experimental Protocol

## 1. Purpose

Define the experimental methodology for evaluating
vision-based aircraft relative pose estimation.

All experiments must be reproducible and must distinguish
paper reproduction from proposed improvements.

## 2. Experiment Categories

### A. Paper Reproduction

Objective:
Implement the original ANN classification and particle
filtering methodology as faithfully as available details permit.

### B. Improved Particle Filtering

Objective:
Evaluate changes to likelihood computation, resampling,
state propagation, and initialization.

### C. Geometric Refinement

Objective:
Compare geometric pose recovery against the original
neural-network-assisted particle filter.

### D. Robustness Experiments

Objective:
Analyze the effect of noise, missing observations,
occlusion, and maneuvering.

## 3. Scenario Dataset

Initial scenario categories:

- Approach_L1
- Approach_L2
- High_Turbulence
- Aggressive_Banking
- High_Occlusion
- Low_Visibility
- Noise_Stress

The simulator generates ground-truth poses and their
corresponding camera measurements.

Scenario identifiers are stable across experiments.

## 4. Data Generation

Each generated trajectory must record:

- Scenario identifier
- Random seed
- Simulation duration
- Simulation frequency
- Camera calibration
- Camera extrinsics
- Aircraft geometry identifier
- True 6-DoF pose per frame
- Noisy 2D landmark locations
- Landmark visibility mask
- Noise parameters

All quantities must follow the shared coordinate contracts.

## 5. Dataset Partitioning

Training, validation, and testing are separated by
trajectory identity.

Adjacent frames from the same generated trajectory
must not appear in multiple partitions.

Suggested initial allocation:

- Training: 70%
- Validation: 15%
- Testing: 15%

These are implementation choices, not ratios specified
by the original paper.

Final results are reported on held-out trajectories.

The test partition must not be used for training,
hyperparameter tuning, or error-model calibration.

## 6. Randomness and Repeatability

Every experiment records separate random seeds for:

- Trajectory generation
- Observation noise
- Neural network initialization
- Training data shuffling
- Particle filter sampling
- Visualization particle subsampling

Initial standard seed: 42.

Published benchmark comparisons should include
multiple independent seeds.

The experiment manifest must record the exact seeds used.

## 7. ANN Training Protocol

The original classifier uses 42 inputs and 4800 classes.

Train using cross-entropy loss.

Evaluate:
- Exact class accuracy
- Top-k class accuracy
- Position-bin error
- Roll-bin error
- Physical decoded-pose error
- Classification confusion
- Probability calibration

The validation dataset is used for checkpoint selection.

Classifier-error distributions used for particle
proposals must be calibrated on training/validation data.

They must not be estimated using test-set ground truth.

## 8. Estimator Comparison

All estimators must receive identical observations.

Experimental factors must be held constant unless
they are explicitly being varied.

Comparison methods:

1. ANN coarse classifier
2. Original paper Particle Filter
3. Improved Particle Filter
4. PnP pose estimation
5. Geometric pose refinement
6. Optional hybrid estimator

Full 6-DoF rotation error is reported only for methods
that produce a full rotation estimate.

The ANN's roll error must be labeled separately from
full rotation error.

## 9. Evaluation Metrics

### Position Error

Euclidean translation error in meters.

### Full Attitude Error

Geodesic rotation distance in degrees.

### Roll Error

Wrapped roll angle difference in degrees.

### Reprojection Error

Mean Euclidean pixel error across valid landmarks.

### Classification Accuracy

Correct discrete pose label rate.

### Particle Diagnostics

- Effective sample size
- Number of resampling events
- Particle spread
- Weight concentration
- Optional posterior covariance

### Performance

- Mean inference time per frame
- Median inference time
- 95th percentile inference time
- Estimator throughput
- Peak memory where measured

Dashboard rendering FPS is not estimator throughput.

## 10. Robustness Studies

Vary one factor at a time:

- Pixel noise
- Feature occlusion
- Missing detections
- Particle count
- Resampling alpha
- Process noise
- Motion complexity

For each condition, record:
- Accuracy
- Failure frequency
- Recovery time
- Runtime

## 11. Experiment Manifest

Each experiment must store:

experiment_id
scenario_id
experiment_type
code_version
configuration
model_checkpoint_id
dataset_version
random_seeds
created_at
simulation_fps
frame_count
estimator_names
evaluation_metrics

The code version should be the Git commit SHA when
available, or an explicitly labeled local revision.

## 12. Statistical Reporting

Results should be aggregated across independent seeds
and held-out trajectories.

Report:
- Mean
- Median
- Standard deviation
- 95th percentile
- Number of evaluated trajectories

For relevant comparisons, report paired differences
using identical scenario and noise seeds.

Do not claim statistically significant improvements
without appropriate statistical analysis.

## 13. Performance Claims

A method is considered improved only when experimental
results support that conclusion.

No result may be hardcoded into the dashboard.

Hypothetical demonstrations must be clearly labeled.

Training metrics and test metrics must never be
presented interchangeably.

## 14. Phase Acceptance

A development phase is complete only when:

1. Required implementations are present.
2. The consolidated phase test passes.
3. Inputs and outputs follow the shared contracts.
4. No ground-truth information leaks into inference.
5. Results are reproducible when randomness is fixed.
6. Relevant findings are documented.

## 15. Known Scope Limitations

The first implementation uses synthetic camera observations.

It does not demonstrate flight-ready autonomous control.

Camera calibration, aircraft geometry, visibility,
and the assumed dynamics are simplified.

Real-world performance requires separate validation.



## 16. Final Integration Validation

The project is accepted only after the following complete
workflow has been demonstrated:

1. Python dependencies import successfully.
2. Aircraft geometry contains exactly 14 landmarks.
3. Camera projection follows the established conventions.
4. The training dataset contains 42 input features.
5. Training, validation, and test trajectories are disjoint.
6. The ANN checkpoint loads with 4800 output classes.
7. Calibration residuals are loaded from validation data.
8. The paper Particle Filter initializes and runs.
9. Improved estimators execute on the same observations.
10. Evaluation records position and attitude errors.
11. FastAPI successfully completes an experiment.
12. The frontend retrieves the experiment telemetry.
13. All visualization panels share a common simulation frame.
14. No benchmark displays fabricated performance values.

## 17. Scientific Comparisons

The original paper Particle Filter must remain available
as an independently identifiable baseline.

All comparisons must use matching observation streams.

Estimator initialization time must be separated from
per-frame processing time unless explicitly labeled
as end-to-end runtime.

Estimation failure and visual measurement failure
must be reported as separate properties.

A method cannot be declared superior solely because
its successful-frame error is lower when it fails on
a larger proportion of observations.

## 18. Reproduction Assumptions

The present implementation uses:

- A simplified aircraft geometry
- A synthetic monocular pinhole camera
- Kinematic leader trajectories
- A restricted training pose distribution
- Empirical classifier residual proposals
- Assumed stochastic particle motion disturbances
- Approximate geometric self-occlusion

These assumptions must be stated during the demonstration.

The work reproduces the research methodology while
investigating experimental improvements.

It does not constitute an exact numerical reproduction
of all figures or source-code behavior in the paper.

## 19. Presentation Requirements

A valid demonstration should explain:

1. What the leader and follower aircraft represent.
2. Why the camera observes 14 structural landmarks.
3. How 42 input features are formed.
4. Why the ANN predicts 4800 pose classes.
5. How coarse classifier predictions initialize particles.
6. How particle propagation and measurement weighting work.
7. Why the original method may fail to converge.
8. How improved likelihoods and resampling differ.
9. How PnP uses 2D-3D correspondences.
10. How the hybrid method selects an estimate.
11. How the dashboard visualizes measured errors.
12. What limitations remain.

## 20. Final Completion Criteria

The final system must:

- Complete a trained-model experiment without an exception.
- Persist experiment telemetry.
- Load the experiment through the REST API.
- Display valid camera landmarks.
- Display matching trajectory timestamps.
- Display position and attitude errors.
- Preserve missing data as unavailable.
- Allow timeline scrubbing without rerunning estimation.
- Produce repeatable numerical outputs with fixed seeds.
- Identify baseline and extended algorithms separately.
