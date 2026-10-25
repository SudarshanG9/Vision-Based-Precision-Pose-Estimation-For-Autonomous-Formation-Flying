
"""Paper-inspired ANN-guided particle filter baseline.

State:
    particles: (N, 7)
    columns: [x, y, z, qx, qy, qz, qw]

Particle poses represent T_F_L:
    Leader frame L -> follower frame F.

Measurement:
    14 structural landmarks projected into the camera.

Likelihood:
    Inverse mean squared reprojection error.

Resampling:
    floor(alpha*N) particles sampled from weighted posterior.
    Remaining particles sampled from ANN-informed proposals.

This is a methodological reproduction, not an exact recovery
of unpublished implementation details.
"""

from __future__ import annotations

import time

import numpy as np

from estimators.interfaces import (
    CameraObservation,
    EstimatorConfig,
    EstimatorResult,
    Pose3D,
)

from simulation.aircraft_model import create_aircraft_model
from simulation.camera import default_camera
from simulation.transforms import euler_to_quaternion


def normalize_quaternions(q):
    q = np.asarray(q, dtype=np.float64)
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("Invalid particle quaternion")
    return q / norms


def quaternion_products(q1, q2):
    """Vectorized Hamilton multiplication, scalar-last."""
    a, b, c, d = np.moveaxis(q1, -1, 0)
    e, f, g, h = np.moveaxis(q2, -1, 0)

    return normalize_quaternions(np.stack([
        d*e + a*h + b*g - c*f,
        d*f - a*g + b*h + c*e,
        d*g + a*f - b*e + c*h,
        d*h - a*e - b*f - c*g,
    ], axis=-1))


def quaternion_matrices(q):
    """Convert (N,4) quaternions into (N,3,3) matrices."""
    q = normalize_quaternions(q)
    x, y, z, w = np.moveaxis(q, -1, 0)

    R = np.empty((len(q), 3, 3), dtype=np.float64)

    R[:, 0, 0] = 1 - 2*(y*y + z*z)
    R[:, 0, 1] = 2*(x*y - z*w)
    R[:, 0, 2] = 2*(x*z + y*w)

    R[:, 1, 0] = 2*(x*y + z*w)
    R[:, 1, 1] = 1 - 2*(x*x + z*z)
    R[:, 1, 2] = 2*(y*z - x*w)

    R[:, 2, 0] = 2*(x*z - y*w)
    R[:, 2, 1] = 2*(y*z + x*w)
    R[:, 2, 2] = 1 - 2*(x*x + y*y)

    return R


def small_angle_quaternions(rotation_vectors):
    """Convert axis-angle rotation vectors to quaternions."""
    vectors = np.asarray(rotation_vectors, dtype=np.float64)

    angles = np.linalg.norm(vectors, axis=1)
    half = 0.5 * angles

    scale = np.empty_like(angles)

    small = angles < 1e-8
    scale[small] = 0.5 - angles[small]**2 / 48.0
    scale[~small] = np.sin(half[~small]) / angles[~small]

    q = np.column_stack([
        vectors * scale[:, None],
        np.cos(half),
    ])

    return normalize_quaternions(q)


class PaperParticleFilter:
    """Original-method baseline with documented numerical assumptions."""

    def __init__(
        self,
        ann_predictor,
        calibration_errors: np.ndarray,
        *,
        reseed_interval: int = 1,
        likelihood_epsilon: float = 1e-6,
        particle_sample_limit: int = 250,
    ):
        """
        ann_predictor:
            Callable(CameraObservation) -> integer class ID.

        calibration_errors:
            Empirical residuals [dx,dy,dz,droll] from Phase 3.
        """
        self.ann_predictor = ann_predictor

        self.calibration_errors = np.asarray(
            calibration_errors, dtype=np.float64
        )

        if (
            self.calibration_errors.ndim != 2
            or self.calibration_errors.shape[1] != 4
            or len(self.calibration_errors) == 0
            or not np.all(np.isfinite(self.calibration_errors))
        ):
            raise ValueError(
                "Calibration errors must have shape (M,4)"
            )

        if reseed_interval < 1:
            raise ValueError("reseed_interval must be positive")

        if likelihood_epsilon <= 0:
            raise ValueError("likelihood_epsilon must be positive")

        self.reseed_interval = reseed_interval
        self.likelihood_epsilon = likelihood_epsilon
        self.particle_sample_limit = particle_sample_limit

        self.aircraft = create_aircraft_model()
        self.camera = default_camera()

        self.config = None
        self.rng = None
        self.particles = None
        self.weights = None
        self.step_index = 0

    def _ann_pose(self, observation):
        """Decode the ANN class label into coarse [x,y,z,roll]."""
        from models.labels import decode_pose

        label = int(self.ann_predictor(observation))
        return np.asarray(decode_pose(label), dtype=np.float64)

    def _sample_ann_particles(self, coarse_pose, count):
        """Sample 6-DoF states from ANN pose and error distribution."""
        indices = self.rng.integers(
            0, len(self.calibration_errors), size=count
        )

        residuals = self.calibration_errors[indices]

        translation = (
            coarse_pose[None, :3] + residuals[:, :3]
        )

        roll = coarse_pose[3] + residuals[:, 3]

        # Pitch and yaw are not predicted by the four-dimensional
        # ANN. Their prior spread is a documented approximation.
        pitch = self.rng.uniform(
            -np.deg2rad(8), np.deg2rad(8), size=count
        )
        yaw = self.rng.uniform(
            -np.deg2rad(8), np.deg2rad(8), size=count
        )

        cr, sr = np.cos(roll/2), np.sin(roll/2)
        cp, sp = np.cos(pitch/2), np.sin(pitch/2)
        cy, sy = np.cos(yaw/2), np.sin(yaw/2)

        quaternion = np.stack([
            sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy,
            cr*cp*cy + sr*sp*sy,
        ], axis=-1)

        return np.column_stack([
            translation,
            normalize_quaternions(quaternion),
        ])

    def initialize(self, config, initial_observation):
        if not isinstance(config, EstimatorConfig):
            raise TypeError("Expected EstimatorConfig")

        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.step_index = 0

        coarse_pose = self._ann_pose(initial_observation)

        self.particles = self._sample_ann_particles(
            coarse_pose,
            config.particle_count,
        )

        self.weights = np.full(
            config.particle_count,
            1.0 / config.particle_count,
        )

    def _propagate(self, dt):
        """Uniform stochastic motion disturbance model."""
        n = len(self.particles)

        # Position disturbance is scaled with elapsed time.
        translation_noise = self.rng.uniform(
            -self.config.position_noise_std,
            self.config.position_noise_std,
            size=(n, 3),
        ) * np.sqrt(dt)

        self.particles[:, :3] += translation_noise

        # Small local-frame angular perturbations.
        rotation_vectors = self.rng.uniform(
            -self.config.rotation_noise_std,
            self.config.rotation_noise_std,
            size=(n, 3),
        ) * np.sqrt(dt)

        delta_q = small_angle_quaternions(rotation_vectors)

        self.particles[:, 3:] = quaternion_products(
            self.particles[:, 3:],
            delta_q,
        )

    def _project_particles(self, observation):
        """Project 14 features for every particle without particle loops."""
        points = self.aircraft.landmarks
        R_F_L = quaternion_matrices(self.particles[:, 3:])

        # (N,14,3), landmark coordinates in follower frame.
        points_follower = np.einsum(
            "nij,kj->nki",
            R_F_L,
            points,
        ) + self.particles[:, None, :3]

        # Camera extrinsics are provided by the observation.
        T_C_F = observation.camera_pose.inverse()
        R_C_F = T_C_F.rotation_matrix()

        points_camera = (
            np.einsum(
                "ij,nkj->nki",
                R_C_F,
                points_follower,
            ) + T_C_F.position
        )

        depth = points_camera[..., 2]

        # Safe denominator; invalid depth is masked afterward.
        safe_depth = np.where(
            depth > observation.camera.near_plane,
            depth,
            1.0,
        )

        intr = observation.camera

        u = intr.fx * points_camera[..., 0] / safe_depth + intr.cx
        v = intr.fy * points_camera[..., 1] / safe_depth + intr.cy

        pixels = np.stack([u, v], axis=-1)

        valid = (
            (depth > intr.near_plane)
            & (u >= 0) & (u < intr.width)
            & (v >= 0) & (v < intr.height)
            & np.isfinite(pixels).all(axis=-1)
        )

        return pixels, valid

    def _measurement_update(self, observation):
        """Update weights with the paper's inverse-MSE structure."""
        projected, valid = self._project_particles(observation)

        observed_mask = observation.visible
        visible_count = int(np.count_nonzero(observed_mask))

        if visible_count == 0:
            return None, 0

        observed = observation.keypoints[observed_mask]
        predicted = projected[:, observed_mask, :]

        valid_features = valid[:, observed_mask]

        squared_error = np.sum(
            (predicted - observed[None, :, :])**2,
            axis=-1,
        )

        squared_error = np.where(
            valid_features,
            squared_error,
            0.0,
        )

        counts = valid_features.sum(axis=1)

        mse = np.full(
            len(self.particles),
            np.inf,
            dtype=np.float64,
        )

        np.divide(
            squared_error.sum(axis=1),
            counts,
            out=mse,
            where=counts > 0,
        )

        # The paper ignores observed occlusions. A prediction
        # outside the image is additionally penalized here.
        missing_predictions = visible_count - counts
        mse += missing_predictions * (
            max(observation.camera.width,
                observation.camera.height)**2
        )

        likelihood = 1.0 / (
            mse + self.likelihood_epsilon
        )

        likelihood = np.nan_to_num(
            likelihood,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # Multiplicative importance update.
        self.weights *= likelihood

        total = self.weights.sum()

        if total <= 0 or not np.isfinite(total):
            self.weights.fill(1.0 / len(self.weights))
        else:
            self.weights /= total

        finite_mse = mse[np.isfinite(mse)]

        return (
            float(np.mean(finite_mse))
            if len(finite_mse) else float("nan")
        ), visible_count

    def _estimate_pose(self):
        """Weighted translation mean and hemisphere-aligned quaternion mean."""
        weights = self.weights

        translation = np.average(
            self.particles[:, :3],
            axis=0,
            weights=weights,
        )

        quaternions = self.particles[:, 3:].copy()

        reference = quaternions[np.argmax(weights)]

        flip = np.sum(
            quaternions * reference[None, :],
            axis=1,
        ) < 0

        quaternions[flip] *= -1.0

        mean_q = np.average(
            quaternions,
            axis=0,
            weights=weights,
        )

        if np.linalg.norm(mean_q) < 1e-10:
            mean_q = reference

        return Pose3D(
            position=translation,
            quaternion=mean_q,
            frame_from="L",
            frame_to="F",
        )

    def _resample(self, observation, ann_reseed):
        n = len(self.particles)
        alpha = self.config.resampling_alpha

        if ann_reseed:
            retain_count = int(np.floor(alpha * n))
        else:
            retain_count = n

        indices = self.rng.choice(
            n,
            size=retain_count,
            replace=True,
            p=self.weights,
        )

        retained = self.particles[indices].copy()

        new_count = n - retain_count

        if new_count > 0:
            coarse = self._ann_pose(observation)
            proposals = self._sample_ann_particles(
                coarse,
                new_count,
            )
            self.particles = np.concatenate([
                retained, proposals
            ], axis=0)
        else:
            self.particles = retained

        self.weights = np.full(n, 1.0 / n)

    def step(self, observation, dt):
        if self.particles is None:
            raise RuntimeError("Particle Filter not initialized")

        if dt <= 0:
            raise ValueError("dt must be positive")

        start = time.perf_counter()

        if self.step_index > 0:
            self._propagate(dt)

        mse, visible_count = self._measurement_update(
            observation
        )

        ess = float(
            1.0 / np.sum(self.weights**2)
        )

        estimate = self._estimate_pose()

        reseed = (
            self.step_index % self.reseed_interval == 0
        )

        # Save a representative particle subset for visualization.
        sample_count = min(
            self.particle_sample_limit,
            len(self.particles),
        )

        sample_indices = np.linspace(
            0,
            len(self.particles) - 1,
            sample_count,
            dtype=np.int64,
        )

        particle_sample = self.particles[
            sample_indices, :3
        ].tolist()

        self._resample(observation, ann_reseed=reseed)
        self.step_index += 1

        runtime_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return EstimatorResult(
            timestamp=observation.timestamp,
            estimator_name="paper_pf",
            pose=estimate,
            runtime_ms=runtime_ms,
            diagnostics={
                "effective_sample_size": ess,
                "particle_count": len(self.particles),
                "mean_reprojection_mse": mse,
                "visible_features": visible_count,
                "ann_reseed_performed": reseed,
                "particle_sample": particle_sample,
            },
        )

    def reset(self):
        self.particles = None
        self.weights = None
        self.config = None
        self.rng = None
        self.step_index = 0
