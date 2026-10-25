
"""Rigid-body transforms using the Phase 0 pose convention."""

from __future__ import annotations

import numpy as np

from estimators.interfaces import Pose3D


def normalize_quaternion(q):
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("Quaternion must be a finite (4,) array")

    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("Quaternion norm must be nonzero")

    return q / norm


def quaternion_multiply(q1, q2):
    """Hamilton product, scalar-last convention [x,y,z,w]."""
    x1, y1, z1, w1 = normalize_quaternion(q1)
    x2, y2, z2, w2 = normalize_quaternion(q2)

    return normalize_quaternion([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def euler_to_quaternion(roll=0.0, pitch=0.0, yaw=0.0):
    """Intrinsic ZYX yaw-pitch-roll, angles in radians."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)

    return normalize_quaternion([
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
        cr*cp*cy + sr*sp*sy,
    ])


def matrix_to_quaternion(matrix):
    """Convert a valid rotation matrix to a scalar-last quaternion."""
    R = np.asarray(matrix, dtype=np.float64)

    if R.shape != (3, 3) or not np.all(np.isfinite(R)):
        raise ValueError("Expected a finite 3x3 rotation matrix")

    if not np.allclose(R.T @ R, np.eye(3), atol=1e-7):
        raise ValueError("Rotation matrix is not orthogonal")

    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-7):
        raise ValueError("Rotation determinant must be +1")

    trace = np.trace(R)

    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = [
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s,
            0.25 * s,
        ]
    else:
        i = int(np.argmax(np.diag(R)))
        j = (i + 1) % 3
        k = (i + 2) % 3

        s = np.sqrt(max(0.0, 1 + R[i, i] - R[j, j] - R[k, k])) * 2
        if s < 1e-12:
            raise ValueError("Degenerate rotation matrix")

        q = np.zeros(4)
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        q[3] = (R[k, j] - R[j, k]) / s

    return normalize_quaternion(q)


def quaternion_to_euler(quaternion):
    """Return (roll, pitch, yaw), in radians."""
    x, y, z, w = normalize_quaternion(quaternion)

    roll = np.arctan2(
        2 * (w*x + y*z),
        1 - 2 * (x*x + y*y),
    )

    sin_pitch = np.clip(2 * (w*y - z*x), -1, 1)
    pitch = np.arcsin(sin_pitch)

    yaw = np.arctan2(
        2 * (w*z + x*y),
        1 - 2 * (y*y + z*z),
    )

    return np.array([roll, pitch, yaw], dtype=np.float64)


def compose_poses(outer: Pose3D, inner: Pose3D) -> Pose3D:
    """Compose T_A_B @ T_B_C, returning T_A_C."""
    if outer.frame_from != inner.frame_to:
        raise ValueError(
            "Frame mismatch: outer source must equal inner destination"
        )

    R = outer.rotation_matrix()

    return Pose3D(
        position=R @ inner.position + outer.position,
        quaternion=quaternion_multiply(
            outer.quaternion, inner.quaternion
        ),
        frame_from=inner.frame_from,
        frame_to=outer.frame_to,
    )


def relative_pose(
    destination_world_pose: Pose3D,
    source_world_pose: Pose3D,
) -> Pose3D:
    """Return T_destination_source from two world poses."""
    if destination_world_pose.frame_to != "W":
        raise ValueError("First pose must have destination W")
    if source_world_pose.frame_to != "W":
        raise ValueError("Second pose must have destination W")

    return compose_poses(
        destination_world_pose.inverse(),
        source_world_pose,
    )


def transform_points(pose: Pose3D, points):
    return pose.transform_points(points)


def rotation_error_degrees(true_pose: Pose3D, estimate: Pose3D):
    """Full SO(3) geodesic angular error."""
    R = true_pose.rotation_matrix().T @ estimate.rotation_matrix()
    cos_angle = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return float(np.rad2deg(np.arccos(cos_angle)))
