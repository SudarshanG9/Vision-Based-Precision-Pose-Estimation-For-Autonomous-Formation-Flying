
# Coordinate Conventions

## 1. Purpose

This document defines the coordinate systems, pose representations,
and transformation conventions for the Vision-Based Precision Pose
Estimation for Autonomous Formation Flying project.

All simulation, estimation, evaluation, and visualization modules
must follow these conventions.

## 2. Coordinate Frames

### 2.1 World Frame (W)

The world frame is a fixed, right-handed North-East-Down
(NED) reference frame.

Axes:
- +X: North
- +Y: East
- +Z: Down

Units: meters.

The world frame is used for generating leader and follower
flight trajectories.

### 2.2 Leader Aircraft Frame (L)

The leader uses the right-handed Forward-Right-Down
(FRD) convention.

Axes:
- +X: Forward (toward the aircraft nose)
- +Y: Right wing
- +Z: Down

The 14 structural aircraft landmarks are defined in this frame.

The origin is a fixed reference point on the aircraft.
For this implementation, it is the aircraft body reference
point used by the landmark geometry.

Landmark coordinates remain constant in the leader frame.

### 2.3 Follower Aircraft Frame (F)

The follower also uses the Forward-Right-Down convention.

Axes:
- +X: Forward
- +Y: Right
- +Z: Down

The follower's camera is mounted relative to this frame.

### 2.4 Camera Frame (C)

We use the OpenCV-compatible optical camera convention.

Axes:
- +X: Image right
- +Y: Image down
- +Z: Optical axis forward

Pixel coordinates:
- u increases toward the right
- v increases toward the bottom

The image origin is the top-left corner.

The principal point is approximately at the center of the image.

## 3. Rotation Representations

The system uses:

1. Rotation matrices for coordinate transformations.
2. Unit quaternions for storing and propagating orientations.
3. Euler angles for configuration and human-readable telemetry.

Quaternion convention:

    q = [qx, qy, qz, qw]

The quaternion is normalized before conversion to a
rotation matrix.

Rotation matrices are elements of SO(3):

    R.T @ R = I
    det(R) = +1

Euler angles follow an intrinsic ZYX yaw-pitch-roll sequence.

For column vectors:

    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

Angles are stored in radians internally.

Dashboard values may be converted into degrees.

## 4. Transformation Convention

T_A_B transforms coordinates expressed in frame B
into coordinates expressed in frame A.

For a point p_B:

    p_A = R_A_B @ p_B + t_A_B

Homogeneous form:

    [p_A]   [R_A_B  t_A_B] [p_B]
    [ 1 ] = [0 0 0    1  ] [ 1 ]

Composition:

    T_A_C = T_A_B @ T_B_C

Inverse:

    R_B_A = R_A_B.T

    t_B_A = -R_A_B.T @ t_A_B

Therefore:

    T_B_A = inverse(T_A_B)

All translation vectors are expressed in the destination
frame of their corresponding transformation.

## 5. Relative Aircraft Pose

Leader and follower world poses are represented by:

    T_W_L
    T_W_F

The leader pose relative to the follower is:

    T_F_L = inverse(T_W_F) @ T_W_L

Its translation component is the position of the
leader-frame origin expressed in follower coordinates.

Its rotation component is the orientation of the leader
frame relative to the follower frame.

The system must distinguish this frame-resolved translation
from the raw world-frame position difference:

    r_W_L - r_W_F

These are generally not numerically equal unless the
follower orientation matches the world-frame orientation.

## 6. Camera Extrinsics

T_F_C describes the camera pose relative to the
follower aircraft.

The transformation of points from the leader frame
into camera coordinates is:

    T_C_L = inverse(T_F_C) @ T_F_L

For a structural landmark P_L:

    P_C = R_C_L @ P_L + t_C_L

Camera orientation may change over time when
gimbal tracking is enabled.

The simulator must record the camera extrinsics
associated with each observation frame.

Estimators may only use camera extrinsics that are
explicitly declared available in the observation contract.

## 7. Camera Projection

For a camera-space point:

    P_C = [Xc, Yc, Zc]

Pinhole projection:

    u = fx * Xc / Zc + cx
    v = fy * Yc / Zc + cy

where:
- fx, fy: Focal lengths in pixels
- cx, cy: Principal point
- Zc: Depth relative to the optical axis

A feature is geometrically projectable only when:

    Zc > z_near

The feature is within the camera image when:

    0 <= u < image_width
    0 <= v < image_height

The default implementation assumes zero lens distortion.

Camera distortion may be added later through explicit
camera calibration parameters.

## 8. Landmark Observations

Each observation contains 14 landmark slots.

For each landmark:

    [u, v, visible]

The complete classifier input contains:

    14 * 3 = 42 values

Visibility is represented by a boolean mask internally.

Invisible landmarks must not contribute to the
measurement residual.

Before ANN inference, missing coordinates must be
replaced with a consistent sentinel value and normalized
using the same preprocessing as the training dataset.

Visibility values remain explicitly available to the ANN.

Feature identifiers must remain consistent between:
- aircraft geometry
- synthetic observations
- training labels
- particle filter measurement model
- geometric pose estimation
- dashboard visualization

## 9. Ground Truth Separation

Ground-truth transformations are generated by the
simulation system.

They are used for:
- dataset target generation
- evaluation metrics
- dashboard ground-truth visualization

Ground-truth poses must never be passed to estimation
algorithms during evaluation.

Estimator initialization must use permitted observations,
model outputs, and declared prior information only.

## 10. Pose Error Metrics

### 10.1 Position Error

For estimated and true translations expressed in the
same reference frame:

    position_error = norm(t_est - t_true)

Units: meters.

### 10.2 Orientation Error

For true and estimated rotation matrices:

    R_error = R_true.T @ R_est

    theta = acos(
        clip((trace(R_error) - 1) / 2, -1, 1)
    )

Units: radians internally.

Dashboard conversion:

    degrees = radians * 180 / pi

This metric measures the geodesic angular distance
between the two rotations.

### 10.3 Reprojection Error

For visible landmarks:

    e_i = norm(z_observed_i - z_projected_i)

Mean reprojection error is computed over valid,
visible correspondences only.

Units: pixels.

If no valid correspondences exist, the reprojection
error must be marked unavailable rather than zero.

## 11. Time Conventions

Simulation timestamps use seconds.

All timestamps refer to simulation time, not wall-clock time.

The numerical integration timestep is:

    dt = 1 / simulation_frequency

The dashboard playback rate may differ from the
estimator processing rate.

The implementation must distinguish:

- simulation frequency
- estimator processing throughput
- dashboard rendering FPS

These quantities must not be used interchangeably.

## 12. Numerical Conventions

Default numerical type:

    float64 for simulation and geometric estimation.

PyTorch training may use float32.

Randomness must be controlled through explicit seeds.

Rotation matrices should be validated for:
- orthogonality
- positive determinant
- finite values

Quaternions must have unit norm within numerical tolerance.

All position, velocity, angle, and pixel units must be
documented at module boundaries.

## 13. Non-Negotiable Invariants

1. All coordinate systems are right-handed.
2. FRD is used for aircraft body frames.
3. The camera uses OpenCV optical coordinates.
4. Transform names follow T_destination_source.
5. All internal angular computations use radians.
6. Quaternions use [qx, qy, qz, qw].
7. The same 14 landmark IDs are used across all modules.
8. Occluded features are excluded from reprojection losses.
9. Ground truth is inaccessible to estimation algorithms.
10. Dashboard telemetry is generated from actual estimator outputs.
