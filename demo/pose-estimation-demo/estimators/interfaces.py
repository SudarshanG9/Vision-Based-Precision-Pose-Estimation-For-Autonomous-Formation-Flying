
"""
Shared data contracts for the pose estimation system.

Coordinate conventions:
    Aircraft frames: Forward-Right-Down (FRD)
    Camera frame:    Right-Down-Forward (OpenCV)
    Quaternion:      [qx, qy, qz, qw]
    Angles:          radians
    Translation:     meters

Transform notation:
    T_A_B maps coordinates from frame B into frame A.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

NUM_LANDMARKS = 14
ANN_INPUT_DIM = NUM_LANDMARKS * 3
ANN_NUM_CLASSES = 4800


def _float_array(
    value: Any,
    shape: tuple[int, ...],
    name: str,
) -> FloatArray:
    """Create an independent finite float64 array."""

    array = np.asarray(value, dtype=np.float64)

    if array.shape != shape:
        raise ValueError(
            f"{name} must have shape {shape}; "
            f"received {array.shape}"
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} contains NaN or infinity"
        )

    return array.copy()


@dataclass
class Pose3D:
    """
    Rigid pose of a source frame relative to a destination frame.

    position:
        Source-frame origin expressed in destination coordinates.

    quaternion:
        Rotation from source into destination coordinates.
        Ordering: [qx, qy, qz, qw].

    frame_from:
        Source coordinate frame.

    frame_to:
        Destination coordinate frame.
    """

    position: FloatArray
    quaternion: FloatArray

    frame_from: str = "L"
    frame_to: str = "F"

    def __post_init__(self) -> None:

        self.position = _float_array(
            self.position, (3,), "position"
        )

        self.quaternion = _float_array(
            self.quaternion, (4,), "quaternion"
        )

        magnitude = np.linalg.norm(self.quaternion)

        if magnitude < 1e-12:
            raise ValueError(
                "Quaternion must have nonzero magnitude"
            )

        self.quaternion /= magnitude

        if not self.frame_from or not self.frame_to:
            raise ValueError(
                "Coordinate frame names cannot be empty"
            )

    @classmethod
    def identity(
        cls,
        frame: str = "F",
    ) -> Pose3D:
        return cls(
            position=np.zeros(3),
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
            frame_from=frame,
            frame_to=frame,
        )

    def rotation_matrix(self) -> FloatArray:
        """Convert unit quaternion to a 3x3 rotation matrix."""

        x, y, z, w = self.quaternion

        return np.array([
            [
                1.0 - 2.0 * (y*y + z*z),
                2.0 * (x*y - z*w),
                2.0 * (x*z + y*w),
            ],
            [
                2.0 * (x*y + z*w),
                1.0 - 2.0 * (x*x + z*z),
                2.0 * (y*z - x*w),
            ],
            [
                2.0 * (x*z - y*w),
                2.0 * (y*z + x*w),
                1.0 - 2.0 * (x*x + y*y),
            ],
        ], dtype=np.float64)

    def as_matrix(self) -> FloatArray:
        """Return the 4x4 homogeneous transformation."""

        transform = np.eye(4, dtype=np.float64)

        transform[:3, :3] = self.rotation_matrix()
        transform[:3, 3] = self.position

        return transform

    def transform_points(
        self,
        points: FloatArray,
    ) -> FloatArray:
        """
        Transform a collection of 3D points.

        Input:  (N, 3), in frame_from
        Output: (N, 3), in frame_to
        """

        points = np.asarray(points, dtype=np.float64)

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                "points must have shape (N, 3)"
            )

        if not np.all(np.isfinite(points)):
            raise ValueError(
                "points contain non-finite values"
            )

        rotation = self.rotation_matrix()

        return points @ rotation.T + self.position

    def inverse(self) -> Pose3D:
        """Return the inverse rigid transformation."""

        rotation = self.rotation_matrix()

        inverse_position = -rotation.T @ self.position

        x, y, z, w = self.quaternion

        inverse_quaternion = np.array(
            [-x, -y, -z, w],
            dtype=np.float64,
        )

        return Pose3D(
            position=inverse_position,
            quaternion=inverse_quaternion,
            frame_from=self.frame_to,
            frame_to=self.frame_from,
        )


@dataclass
class CameraIntrinsics:
    """Pinhole camera calibration parameters."""

    fx: float
    fy: float
    cx: float
    cy: float

    width: int
    height: int

    near_plane: float = 0.01

    def __post_init__(self) -> None:

        values = [
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.near_plane,
        ]

        if not all(np.isfinite(values)):
            raise ValueError(
                "Camera calibration contains non-finite values"
            )

        if self.fx <= 0 or self.fy <= 0:
            raise ValueError(
                "Focal lengths must be positive"
            )

        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                "Image dimensions must be positive"
            )

        if self.near_plane <= 0:
            raise ValueError(
                "Near-plane distance must be positive"
            )

    def matrix(self) -> FloatArray:
        """Return the 3x3 intrinsic camera matrix."""

        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)


@dataclass
class CameraObservation:
    """
    One synthetic or CNN-generated camera observation.

    Invalid or invisible landmark coordinates are stored as NaN.
    Visibility masks determine which observations are valid.

    The estimator must never use ground truth.
    """

    timestamp: float
    frame_index: int

    keypoints: FloatArray
    visible: BoolArray

    camera: CameraIntrinsics

    camera_pose: Pose3D = field(
        default_factory=lambda: Pose3D.identity("C")
    )

    def __post_init__(self) -> None:

        self.keypoints = np.asarray(
            self.keypoints, dtype=np.float64
        ).copy()

        self.visible = np.asarray(
            self.visible, dtype=np.bool_
        ).copy()

        if self.keypoints.shape != (NUM_LANDMARKS, 2):
            raise ValueError(
                "keypoints must have shape (14, 2)"
            )

        if self.visible.shape != (NUM_LANDMARKS,):
            raise ValueError(
                "visible must have shape (14,)"
            )

        if not np.isfinite(self.timestamp):
            raise ValueError(
                "timestamp must be finite"
            )

        if self.timestamp < 0:
            raise ValueError(
                "timestamp cannot be negative"
            )

        if self.frame_index < 0:
            raise ValueError(
                "frame_index cannot be negative"
            )

        valid_points = self.keypoints[self.visible]

        if not np.all(np.isfinite(valid_points)):
            raise ValueError(
                "Visible keypoints must have finite coordinates"
            )

        self.keypoints[~self.visible] = np.nan

        if (
            self.camera_pose.frame_from != "C"
            or self.camera_pose.frame_to != "F"
        ) and not (
            self.camera_pose.frame_from == "C"
            and self.camera_pose.frame_to == "C"
        ):
            raise ValueError(
                "camera_pose must represent T_F_C "
                "or the camera identity transform"
            )

    @property
    def visible_count(self) -> int:
        return int(np.count_nonzero(self.visible))

    def ann_input(self) -> FloatArray:
        """
        Flatten features into 42 ANN inputs.

        Each feature is [normalized_u, normalized_v, visibility].

        Missing coordinates use zero after visibility masking.

        This is the canonical Phase 0 preprocessing contract.
        Any paper-specific alternative must be documented.
        """

        normalized = np.zeros(
            (NUM_LANDMARKS, 3),
            dtype=np.float64,
        )

        normalized[self.visible, 0] = (
            self.keypoints[self.visible, 0] / self.camera.width
        )

        normalized[self.visible, 1] = (
            self.keypoints[self.visible, 1] / self.camera.height
        )

        normalized[:, 2] = self.visible.astype(np.float64)

        return normalized.reshape(ANN_INPUT_DIM)


@dataclass
class EstimatorConfig:
    """Common estimator configuration."""

    name: str

    particle_count: int = 5000
    resampling_alpha: float = 0.9

    position_noise_std: float = 0.5
    rotation_noise_std: float = 0.01

    seed: int = 42

    def __post_init__(self) -> None:

        if not self.name:
            raise ValueError(
                "Estimator name cannot be empty"
            )

        if self.particle_count < 1:
            raise ValueError(
                "particle_count must be positive"
            )

        if not 0.0 <= self.resampling_alpha <= 1.0:
            raise ValueError(
                "resampling_alpha must lie in [0, 1]"
            )

        if (
            not np.isfinite(self.position_noise_std)
            or self.position_noise_std < 0
        ):
            raise ValueError(
                "position_noise_std must be nonnegative and finite"
            )

        if (
            not np.isfinite(self.rotation_noise_std)
            or self.rotation_noise_std < 0
        ):
            raise ValueError(
                "rotation_noise_std must be nonnegative and finite"
            )


@dataclass
class EstimatorResult:
    """Standard output from a continuous pose estimator."""

    timestamp: float
    estimator_name: str

    pose: Pose3D

    runtime_ms: float

    covariance: FloatArray | None = None

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:

        if not np.isfinite(self.timestamp):
            raise ValueError(
                "Estimator timestamp must be finite"
            )

        if self.timestamp < 0:
            raise ValueError(
                "Estimator timestamp cannot be negative"
            )

        if not self.estimator_name:
            raise ValueError(
                "Estimator name cannot be empty"
            )

        if not np.isfinite(self.runtime_ms):
            raise ValueError(
                "Estimator runtime must be finite"
            )

        if self.runtime_ms < 0:
            raise ValueError(
                "Estimator runtime cannot be negative"
            )

        if self.covariance is not None:

            self.covariance = _float_array(
                self.covariance,
                (6, 6),
                "covariance",
            )

            if not np.allclose(
                self.covariance,
                self.covariance.T,
                atol=1e-9,
            ):
                raise ValueError(
                    "Covariance must be symmetric"
                )

            eigenvalues = np.linalg.eigvalsh(
                self.covariance
            )

            if np.min(eigenvalues) < -1e-9:
                raise ValueError(
                    "Covariance must be positive semidefinite"
                )


@runtime_checkable
class PoseEstimator(Protocol):
    """
    Contract implemented by continuous pose estimators.

    The ANN classifier is a separate coarse-label model and
    does not have to implement this interface directly.
    """

    def initialize(
        self,
        config: EstimatorConfig,
        initial_observation: CameraObservation,
    ) -> None:
        ...

    def step(
        self,
        observation: CameraObservation,
        dt: float,
    ) -> EstimatorResult:
        ...

    def reset(self) -> None:
        ...
