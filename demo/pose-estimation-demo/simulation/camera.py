
"""Calibrated monocular camera projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from estimators.interfaces import CameraIntrinsics, Pose3D
from simulation.transforms import (
    compose_poses,
    matrix_to_quaternion,
)


def forward_camera_mount() -> Pose3D:
    """
    Camera-to-follower transformation T_F_C.

    Camera +X maps to follower +Y.
    Camera +Y maps to follower +Z.
    Camera +Z maps to follower +X.
    """
    R_F_C = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    return Pose3D(
        position=np.zeros(3),
        quaternion=matrix_to_quaternion(R_F_C),
        frame_from="C",
        frame_to="F",
    )


@dataclass
class PinholeCamera:

    intrinsics: CameraIntrinsics
    mount: Pose3D

    def __post_init__(self):
        if (
            self.mount.frame_from != "C"
            or self.mount.frame_to != "F"
        ):
            raise ValueError("Camera mount must be T_F_C")

    def leader_to_camera(self, leader_in_follower: Pose3D):
        """Convert T_F_L into T_C_L."""
        if (
            leader_in_follower.frame_from != "L"
            or leader_in_follower.frame_to != "F"
        ):
            raise ValueError("Expected leader pose T_F_L")

        return compose_poses(
            self.mount.inverse(),
            leader_in_follower,
        )

    def project_camera_points(self, points_camera):
        """
        Project camera-frame 3D points into pixels.

        Returns:
            pixels: (N,2), NaN for unprojectable points
            valid:  (N,), geometric and image-bound validity
        """
        points = np.asarray(points_camera, dtype=np.float64)

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Expected points with shape (N,3)")

        pixels = np.full((len(points), 2), np.nan)
        depth = points[:, 2]

        valid = (
            np.isfinite(points).all(axis=1)
            & (depth > self.intrinsics.near_plane)
        )

        if np.any(valid):
            p = points[valid]

            u = self.intrinsics.fx * p[:, 0] / p[:, 2]
            v = self.intrinsics.fy * p[:, 1] / p[:, 2]

            pixels[valid, 0] = u + self.intrinsics.cx
            pixels[valid, 1] = v + self.intrinsics.cy

        in_image = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < self.intrinsics.width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < self.intrinsics.height)
        )

        valid &= in_image
        pixels[~valid] = np.nan

        return pixels, valid

    def project_landmarks(self, landmarks, leader_in_follower):
        """Return pixels, validity and camera-frame points."""
        T_C_L = self.leader_to_camera(leader_in_follower)

        camera_points = T_C_L.transform_points(landmarks)

        pixels, valid = self.project_camera_points(camera_points)

        return pixels, valid, camera_points


def default_camera() -> PinholeCamera:
    intrinsics = CameraIntrinsics(
        fx=800.0,
        fy=800.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
        near_plane=0.05,
    )

    return PinholeCamera(
        intrinsics=intrinsics,
        mount=forward_camera_mount(),
    )
