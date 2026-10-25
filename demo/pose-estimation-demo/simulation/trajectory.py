
"""Deterministic 6-DoF flight trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from estimators.interfaces import Pose3D
from simulation.transforms import (
    euler_to_quaternion,
    relative_pose,
)


@dataclass
class Trajectory:
    timestamps: np.ndarray
    leader_world: list[Pose3D]
    follower_world: list[Pose3D]
    relative_poses: list[Pose3D]

    def __post_init__(self):
        count = len(self.timestamps)

        if (
            len(self.leader_world) != count
            or len(self.follower_world) != count
            or len(self.relative_poses) != count
        ):
            raise ValueError("Trajectory arrays must have equal lengths")

        if count < 2:
            raise ValueError("Trajectory requires at least two frames")

        if not np.all(np.diff(self.timestamps) > 0):
            raise ValueError("Trajectory timestamps must increase")


@dataclass
class TrajectoryConfig:

    duration: float = 10.0
    fps: int = 100

    initial_distance: float = 80.0
    final_distance: float = 35.0

    lateral_offset: float = 2.0
    vertical_offset: float = -1.0

    turbulence: float = 0.0
    aggressive_bank: bool = False

    def __post_init__(self):
        if not np.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("duration must be positive")
        if self.fps < 2:
            raise ValueError("fps must be at least 2")
        if self.final_distance <= 15.0:
            raise ValueError("Final distance is too close for this model")
        if self.initial_distance <= self.final_distance:
            raise ValueError("Initial distance must exceed final distance")
        if not np.isfinite(self.turbulence) or self.turbulence < 0:
            raise ValueError("turbulence must be nonnegative")


def generate_trajectory(
    config: TrajectoryConfig,
    seed: int = 42,
) -> Trajectory:
    """
    Build a 10-second (by default) closing formation trajectory.

    The follower remains at a fixed world-frame reference pose.
    Leader-relative position and attitude vary continuously.
    """
    rng = np.random.default_rng(seed)

    count = int(round(config.duration * config.fps))
    if count < 2:
        raise ValueError("Trajectory must contain at least two frames")

    timestamps = np.arange(count) / config.fps
    tau = timestamps / config.duration

    # Smooth closing flight path.
    x = (
        config.initial_distance
        + (config.final_distance - config.initial_distance) * tau
    )

    y = (
        config.lateral_offset
        + 2.0 * np.sin(2 * np.pi * tau)
    )

    z = (
        config.vertical_offset
        + 1.2 * np.sin(2 * np.pi * tau + 0.4)
    )

    roll = np.deg2rad(
        4.0 * np.sin(2 * np.pi * tau)
    )

    pitch = np.deg2rad(
        2.5 * np.sin(2 * np.pi * tau + 0.5)
    )

    yaw = np.deg2rad(
        2.0 * np.sin(2 * np.pi * tau + 0.2)
    )

    if config.aggressive_bank:
        roll += np.deg2rad(
            24.0 * np.sin(4 * np.pi * tau)
        )
        yaw += np.deg2rad(
            7.0 * np.sin(3 * np.pi * tau)
        )

    if config.turbulence > 0:
        # Smooth disturbances avoid independent jumps at each frame.
        knots = np.linspace(0, config.duration, 21)

        def smooth_noise(scale):
            samples = rng.normal(
                0.0, scale, size=len(knots)
            )
            return np.interp(timestamps, knots, samples)

        x += smooth_noise(0.25 * config.turbulence)
        y += smooth_noise(0.8 * config.turbulence)
        z += smooth_noise(0.55 * config.turbulence)

        roll += np.deg2rad(
            smooth_noise(3.0 * config.turbulence)
        )

        pitch += np.deg2rad(
            smooth_noise(2.0 * config.turbulence)
        )

        yaw += np.deg2rad(
            smooth_noise(2.0 * config.turbulence)
        )

    leader_world = []
    follower_world = []
    relative_poses = []

    for i in range(count):
        leader = Pose3D(
            position=[x[i], y[i], z[i]],
            quaternion=euler_to_quaternion(
                roll[i], pitch[i], yaw[i]
            ),
            frame_from="L",
            frame_to="W",
        )

        follower = Pose3D(
            position=[0.0, 0.0, 0.0],
            quaternion=[0.0, 0.0, 0.0, 1.0],
            frame_from="F",
            frame_to="W",
        )

        relative = relative_pose(follower, leader)

        leader_world.append(leader)
        follower_world.append(follower)
        relative_poses.append(relative)

    return Trajectory(
        timestamps=timestamps,
        leader_world=leader_world,
        follower_world=follower_world,
        relative_poses=relative_poses,
    )
