
"""Continuous 6-DoF geometric pose estimation.

Methods:
    - OpenCV solvePnPRansac
    - OpenCV solvePnPRefineLM

Output:
    T_F_L, leader pose relative to follower.

This is a geometric refinement baseline, not a
component originally implemented in the source paper.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from estimators.interfaces import (
    CameraObservation,
    EstimatorConfig,
    EstimatorResult,
    Pose3D,
)

from simulation.aircraft_model import create_aircraft_model
from simulation.transforms import (
    compose_poses,
    matrix_to_quaternion,
)


def pose_reprojection_error(
    observation: CameraObservation,
    pose: Pose3D,
    aircraft_points=None,
):
    """
    Compute mean observed-landmark reprojection error.

    Returns None if no valid visible correspondences exist.
    Invalid projected depths receive a large penalty.
    """
    if aircraft_points is None:
        aircraft_points = create_aircraft_model().landmarks

    if pose.frame_from != "L" or pose.frame_to != "F":
        raise ValueError("Expected leader pose T_F_L")

    visible = observation.visible

    if not np.any(visible):
        return None

    T_C_L = compose_poses(
        observation.camera_pose.inverse(),
        pose,
    )

    points_camera = T_C_L.transform_points(
        aircraft_points[visible]
    )

    intr = observation.camera

    depth = points_camera[:, 2]

    valid = (
        depth > intr.near_plane
    ) & np.isfinite(points_camera).all(axis=1)

    errors = np.full(
        len(points_camera),
        float(max(intr.width, intr.height)),
        dtype=np.float64,
    )

    if np.any(valid):
        points = points_camera[valid]

        projected = np.column_stack([
            intr.fx * points[:, 0] / points[:, 2] + intr.cx,
            intr.fy * points[:, 1] / points[:, 2] + intr.cy,
        ])

        observed = observation.keypoints[visible][valid]

        errors[valid] = np.linalg.norm(
            projected - observed,
            axis=1,
        )

    return float(np.mean(errors))


class GeometricPoseEstimator:
    """PnP-RANSAC + continuous nonlinear pose refinement."""

    def __init__(
        self,
        *,
        reprojection_threshold_px: float = 5.0,
        ransac_iterations: int = 200,
        confidence: float = 0.99,
        refine: bool = True,
        minimum_points: int = 6,
    ):
        if reprojection_threshold_px <= 0:
            raise ValueError(
                "Reprojection threshold must be positive"
            )

        if ransac_iterations < 1:
            raise ValueError(
                "RANSAC iterations must be positive"
            )

        if not 0 < confidence < 1:
            raise ValueError(
                "Confidence must be in (0,1)"
            )

        if minimum_points < 4:
            raise ValueError(
                "At least four correspondences required"
            )

        self.reprojection_threshold_px = (
            reprojection_threshold_px
        )
        self.ransac_iterations = ransac_iterations
        self.confidence = confidence
        self.refine = refine
        self.minimum_points = minimum_points

        self.aircraft = create_aircraft_model()

        self.config = None
        self.last_valid_pose = None

    def initialize(
        self,
        config: EstimatorConfig,
        initial_observation: CameraObservation,
    ) -> None:
        self.config = config
        self.last_valid_pose = None

    def _solve(self, observation):
        visible_indices = np.flatnonzero(
            observation.visible
        )

        if len(visible_indices) < self.minimum_points:
            return None, 0

        object_points = np.ascontiguousarray(
            self.aircraft.landmarks[visible_indices],
            dtype=np.float64,
        )

        image_points = np.ascontiguousarray(
            observation.keypoints[visible_indices],
            dtype=np.float64,
        )

        intrinsic = observation.camera.matrix()

        distortion = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=object_points,
            imagePoints=image_points,
            cameraMatrix=intrinsic,
            distCoeffs=distortion,
            iterationsCount=self.ransac_iterations,
            reprojectionError=self.reprojection_threshold_px,
            confidence=self.confidence,
            flags=cv2.SOLVEPNP_EPNP,
        )

        if not success or inliers is None:
            return None, 0

        inlier_indices = inliers.reshape(-1)

        if len(inlier_indices) < 4:
            return None, len(inlier_indices)

        if self.refine:
            rvec, tvec = cv2.solvePnPRefineLM(
                objectPoints=object_points[inlier_indices],
                imagePoints=image_points[inlier_indices],
                cameraMatrix=intrinsic,
                distCoeffs=distortion,
                rvec=rvec,
                tvec=tvec,
            )

        R_C_L, _ = cv2.Rodrigues(rvec)

        camera_relative_pose = Pose3D(
            position=tvec.reshape(3),
            quaternion=matrix_to_quaternion(R_C_L),
            frame_from="L",
            frame_to="C",
        )

        follower_relative_pose = compose_poses(
            observation.camera_pose,
            camera_relative_pose,
        )

        return follower_relative_pose, len(inlier_indices)

    def step(
        self,
        observation: CameraObservation,
        dt: float,
    ) -> EstimatorResult:
        if self.config is None:
            raise RuntimeError(
                "Geometric estimator not initialized"
            )

        if not np.isfinite(dt) or dt <= 0:
            raise ValueError(
                "dt must be finite and positive"
            )

        start = time.perf_counter()

        pose, inlier_count = self._solve(observation)

        success = pose is not None

        if success:
            self.last_valid_pose = pose

        if pose is None:
            if self.last_valid_pose is None:
                # A pose failure is not an identity-pose measurement.
                # Return a flagged fallback while preserving API shape.
                raise RuntimeError(
                    "PnP failed without an available prior estimate"
                )

            pose = self.last_valid_pose

        reprojection_error = pose_reprojection_error(
            observation,
            pose,
            aircraft_points=self.aircraft.landmarks,
        )

        runtime_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return EstimatorResult(
            timestamp=observation.timestamp,
            estimator_name="geometric",
            pose=pose,
            runtime_ms=runtime_ms,
            diagnostics={
                "measurement_success": success,
                "used_previous_pose": not success,
                "inlier_count": int(inlier_count),
                "visible_features": observation.visible_count,
                "reprojection_error_px": reprojection_error,
            },
        )

    def reset(self):
        self.config = None
        self.last_valid_pose = None
