
# Paper Reproduction Specification

## 1. Reference Paper

Title:
Vision-Based Precision Pose Estimation For
Autonomous Formation Flying

Author:
Rohan Punnoose

Objective:
Estimate the relative pose of a leader aircraft
observed by a monocular camera on a follower aircraft.

Primary methodology:
Artificial neural network coarse pose classification
combined with Bayesian particle filtering.

## 2. Research Problem

The leader has:
- Inertial position
- Inertial orientation

The follower has:
- Inertial position
- Inertial orientation
- A calibrated monocular camera

The follower observes the leader and estimates its
relative position and orientation.

The research targets precision relative navigation
for formation flying.

## 3. Visual Feature Model

The paper assumes a feature detection subsystem
capable of detecting known aircraft structural points.

Number of structural features:

    14

Each feature provides:
- Image x coordinate
- Image y coordinate
- Occlusion / visibility information

The paper does not implement or evaluate a
complete CNN landmark detector as its central
research contribution.

Our baseline therefore supports synthetic detected
features without requiring a learned detector.

A CNN detector is an optional system extension.

## 4. ANN Architecture

Input:

    42 values

This consists of 14 features with:
- x image coordinate
- y image coordinate
- occlusion flag

Hidden layers:

    100 neurons
    100 neurons

Activation:

    ReLU

Output:

    4800-way Softmax

Training optimizer:

    Adam

The paper reports approximately 54% validation
classification accuracy.

The reproduction must not assume that the same
validation accuracy will automatically be achieved.

## 5. Pose Discretization

The paper discretizes four dimensions.

Forward position:

    Range: -100 to 50 meters
    Bins: 10

Lateral position:

    Range: -50 to 50 meters
    Bins: 10

Vertical position:

    Range: -20 to 20 meters
    Bins: 8

Roll:

    Range: -45 to 45 degrees
    Bins: 6

Total:

    10 * 10 * 8 * 6 = 4800

The classifier does not discretize all six
continuous pose coordinates.

The particle filter operates with fuller
relative-pose dynamics.

Exact bin-edge conventions must be documented
as an implementation assumption unless otherwise
established from the source.

## 6. Particle Filter Initialization

The original method initializes particles using:

- ANN coarse pose predictions
- Classifier error statistics

The error statistics are obtained from a
validation error distribution.

Our implementation must estimate these statistics
without using evaluation test ground truth.

## 7. Particle Propagation

Particles propagate through a simplified nonlinear
relative-motion model.

The leader's unknown control inputs are modeled
as disturbances.

The original paper describes uniformly sampled
unknown quantities during propagation.

Exact disturbance bounds and other unspecified
numerical details must be identified as reproduction
assumptions.

## 8. Measurement Likelihood

For each particle:

1. Generate predicted structural feature projections.
2. Compare them with observed camera features.
3. Ignore undetected or occluded features.
4. Calculate reprojection error.
5. Derive particle likelihood weights.

The paper uses inverse mean squared reprojection error.

A faithful reproduction must retain this approach
before evaluating alternative likelihood functions.

Numerical safeguards such as epsilon values must be
documented.

The baseline must not silently substitute a Gaussian
likelihood and describe it as the original method.

## 9. Particle Resampling

Let:
    N = total number of particles
    alpha = resampling retention fraction

The paper resamples:

    floor(alpha * N)

particles from the weighted current particle set.

The remaining:

    N - floor(alpha * N)

particles are sampled from an ANN-informed
pose proposal distribution.

This means alpha is not a direct ANN confidence weight.

The reseeding schedule must be configurable.

## 10. Reported Experimental Settings

The paper describes a formation flight approach
trajectory lasting approximately 10 seconds.

Two displayed Particle Filter configurations are:

Configuration A:
    N = 1000
    alpha = 0.9
    ANN-informed resampling every 10 cycles

Configuration B:
    N = 5000
    alpha = 0.9
    ANN-informed resampling every cycle

These must be supported by the reproduction system.

## 11. Reported Findings

The ANN classifier can identify useful coarse poses.

Classification error is influenced by discretization.

Misclassified labels are often close to the
correct pose region.

The reported Particle Filter experiments do not
demonstrate the intended strong convergence
toward the true pose.

The paper states that estimation error remains
substantially influenced by classification error.

Increased particle counts may become
computationally expensive.

## 12. Proposed Research Extensions

The paper suggests investigating alternative
refinement methods such as bundle adjustment.

Our implementation investigates:

- Vectorized particle filtering
- Stable measurement likelihoods
- Adaptive resampling
- Geometric pose estimation
- Nonlinear reprojection optimization
- Optional hybrid estimation

These are extensions, not original reported results.

## 13. Reproduction Limitations

The available paper does not completely specify:

- Original source code
- Exact full training dataset
- All data generation distributions
- Exact dataset split
- All random seeds
- Full network training schedule
- Exact optimizer hyperparameters
- All numerical particle filter safeguards
- Every dynamics disturbance parameter

Consequently, strict numerical replication of
all published plots cannot be guaranteed.

The target is a transparent methodological
reproduction followed by controlled experiments.

## 14. Required Baseline Validation

The original ANN and Particle Filter must be tested
before implementing performance improvements.

Compare:
- Classification accuracy
- Coarse pose error
- PF position error
- PF attitude error
- Particle count
- Runtime

Original and improved methods must use the same
observations during head-to-head evaluation.

## 15. Scientific Reporting

Every result must indicate its provenance:

1. Reported in original paper
2. Reproduced by this implementation
3. Produced by an extended algorithm
4. Hypothetical or illustrative

The dashboard must use actual experimental values.

Unverified performance improvements must not
be presented as established findings.

## 16. Project Deliverable

A reproducible research platform demonstrating:

- Paper-faithful ANN classification
- Original Particle Filter behavior
- Improved estimator comparisons
- Pose error telemetry
- Robustness studies
- Computational benchmarks
- Interactive visualization

The end system is an open-loop research demonstrator,
not a certified autonomous formation-flight controller.
