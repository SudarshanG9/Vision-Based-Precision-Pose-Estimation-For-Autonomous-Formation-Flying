
"""Paper-defined 4800-class pose discretization.

Pose coordinates are expressed as:
    [forward_x_m, lateral_y_m, vertical_z_m, roll_rad]

The paper defines 10 x 10 x 8 x 6 discrete classes.

Out-of-range inputs raise ValueError by default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PoseDiscretization:
    x_range: tuple[float, float] = (-100.0, 50.0)
    y_range: tuple[float, float] = (-50.0, 50.0)
    z_range: tuple[float, float] = (-20.0, 20.0)
    roll_range: tuple[float, float] = (
        -np.pi / 4,
        np.pi / 4,
    )

    x_bins: int = 10
    y_bins: int = 10
    z_bins: int = 8
    roll_bins: int = 6

    @property
    def shape(self):
        return (
            self.x_bins,
            self.y_bins,
            self.z_bins,
            self.roll_bins,
        )

    @property
    def num_classes(self):
        return int(np.prod(self.shape))

    def limits(self):
        return np.array([
            self.x_range,
            self.y_range,
            self.z_range,
            self.roll_range,
        ], dtype=np.float64)

    def bin_widths(self):
        limits = self.limits()
        return (
            (limits[:, 1] - limits[:, 0])
            / np.asarray(self.shape)
        )

    def validate(self, poses):
        poses = np.asarray(poses, dtype=np.float64)

        if poses.ndim == 1:
            poses = poses[None, :]

        if poses.ndim != 2 or poses.shape[1] != 4:
            raise ValueError(
                "Pose labels require shape (N, 4)"
            )

        if not np.isfinite(poses).all():
            raise ValueError(
                "Pose labels contain non-finite values"
            )

        limits = self.limits()

        valid = np.all(
            (poses >= limits[:, 0])
            & (poses <= limits[:, 1]),
            axis=1,
        )

        return valid

    def encode(self, poses):
        """Map continuous 4D poses to integer class IDs."""
        poses = np.asarray(poses, dtype=np.float64)
        single = poses.ndim == 1

        if single:
            poses = poses[None, :]

        valid = self.validate(poses)

        if not valid.all():
            first = int(np.where(~valid)[0][0])
            raise ValueError(
                f"Pose outside paper label ranges "
                f"at sample {first}: {poses[first]}"
            )

        limits = self.limits()
        widths = self.bin_widths()

        indices = np.floor(
            (poses - limits[:, 0]) / widths
        ).astype(np.int64)

        # Include the exact upper endpoints in the final bin.
        indices = np.clip(
            indices,
            0,
            np.asarray(self.shape) - 1,
        )

        labels = np.ravel_multi_index(
            indices.T,
            dims=self.shape,
        )

        return int(labels[0]) if single else labels

    def decode_indices(self, labels):
        """Return integer bin indices in x, y, z, roll order."""
        labels = np.asarray(labels, dtype=np.int64)

        single = labels.ndim == 0
        labels = np.atleast_1d(labels)

        if np.any(labels < 0) or np.any(
            labels >= self.num_classes
        ):
            raise ValueError("Class ID outside [0, 4799]")

        indices = np.column_stack(
            np.unravel_index(labels, self.shape)
        )

        return indices[0] if single else indices

    def decode(self, labels):
        """Decode class IDs to 4D pose-bin centers."""
        indices = np.asarray(
            self.decode_indices(labels)
        )

        single = indices.ndim == 1

        if single:
            indices = indices[None, :]

        limits = self.limits()
        widths = self.bin_widths()

        centers = (
            limits[:, 0]
            + (indices + 0.5) * widths
        )

        return centers[0] if single else centers

    def quantization_error(self, poses):
        poses = np.asarray(poses, dtype=np.float64)
        return poses - self.decode(self.encode(poses))

    def nearest_class_distance(self, predicted, target):
        """Manhattan distance between pose-bin indices."""
        a = np.asarray(self.decode_indices(predicted))
        b = np.asarray(self.decode_indices(target))

        return np.sum(np.abs(a - b), axis=-1)


DEFAULT_DISCRETIZATION = PoseDiscretization()

NUM_CLASSES = DEFAULT_DISCRETIZATION.num_classes


def encode_pose(pose):
    return DEFAULT_DISCRETIZATION.encode(pose)


def decode_pose(label):
    return DEFAULT_DISCRETIZATION.decode(label)
