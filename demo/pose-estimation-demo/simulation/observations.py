
"""Synthetic camera observation generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from estimators.interfaces import CameraObservation
from simulation.renderer import RenderResult
from simulation.camera import PinholeCamera


@dataclass
class ObservationConfig:

    pixel_noise_std: float = 1.5
    dropout_probability: float = 0.0

    def __post_init__(self):
        if (
            not np.isfinite(self.pixel_noise_std)
            or self.pixel_noise_std < 0
        ):
            raise ValueError(
                "pixel_noise_std must be finite and nonnegative"
            )

        if not (
            0.0 <= self.dropout_probability <= 1.0
        ):
            raise ValueError(
                "dropout_probability must lie in [0,1]"
            )


def generate_observation(
    render: RenderResult,
    camera: PinholeCamera,
    timestamp: float,
    frame_index: int,
    rng: np.random.Generator,
    config: ObservationConfig,
) -> CameraObservation:
    """Convert a rendered frame into noisy landmark observations."""

    keypoints = render.pixels.copy()
    visible = render.visible.copy()

    # Gaussian noise only for geometrically visible features.
    count = np.count_nonzero(visible)

    if count > 0 and config.pixel_noise_std > 0:
        keypoints[visible] += rng.normal(
            loc=0.0,
            scale=config.pixel_noise_std,
            size=(count, 2),
        )

    # Simulated feature detection failure.
    dropout = (
        rng.random(len(visible))
        < config.dropout_probability
    )

    visible &= ~dropout

    # Measurement noise can move features outside the image.
    width = camera.intrinsics.width
    height = camera.intrinsics.height

    inside = (
        (keypoints[:, 0] >= 0)
        & (keypoints[:, 0] < width)
        & (keypoints[:, 1] >= 0)
        & (keypoints[:, 1] < height)
    )

    visible &= inside
    keypoints[~visible] = np.nan

    return CameraObservation(
        timestamp=float(timestamp),
        frame_index=int(frame_index),
        keypoints=keypoints,
        visible=visible,
        camera=camera.intrinsics,
        camera_pose=camera.mount,
    )


def observation_to_json(observation: CameraObservation):
    """
    Convert observations into JSON-safe dashboard telemetry.

    Invisible landmark coordinates are represented as null.
    """
    keypoints = []

    for pixel, valid in zip(
        observation.keypoints,
        observation.visible,
    ):
        if valid:
            keypoints.append([
                float(pixel[0]),
                float(pixel[1]),
            ])
        else:
            keypoints.append(None)

    return {
        "frame_index": observation.frame_index,
        "timestamp": observation.timestamp,
        "keypoints": keypoints,
        "visible": observation.visible.tolist(),
    }
